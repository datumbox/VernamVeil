#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "blake3.h"

#ifdef _OPENMP
// Enable parallelisation with OpenMP for multi-core performance
#include <omp.h>
#define CHUNK_SIZE 1024  // Size of each chunk in bytes for parallel processing
#define MIN_PARALLEL_SIZE (2 * CHUNK_SIZE)  // Minimum size to enable parallel tree hashing
#define BLAKE3_OUT_DOUBLE_LEN (2 * BLAKE3_OUT_LEN)  // Double the output length for combining CVs
#define MIN(a, b) ((a) < (b) ? (a) : (b))  // Macro to find the minimum of two values
#endif

#define BLOCK_SIZE 8  // Each input element is a uint64 block

// Inline helper to prepare a BLAKE3 key from a seed (up to 32 bytes, zero-padded if shorter)
static inline void prepare_blake3_key(const bool seeded, const char* restrict seed, const size_t seedlen, uint8_t key[BLAKE3_KEY_LEN]) {
    if (seeded) {
        const size_t copylen = seedlen < BLAKE3_KEY_LEN ? seedlen : BLAKE3_KEY_LEN;
        memcpy(key, seed, copylen);
    }
}

// Inline helper to hash data with BLAKE3, with or without a key, and output variable-length hash
static inline void blake3_hash_bytes(const uint8_t* restrict data, const size_t datalen, const uint8_t* restrict key, const bool seeded, uint8_t* restrict out, const size_t hash_size) {
    // Initialise the BLAKE3 hasher
    blake3_hasher hasher;
    if (seeded) {
        // If a seed is provided, use it as the BLAKE3 key (up to 32 bytes, zero-padded if shorter)
        blake3_hasher_init_keyed(&hasher, key);
    } else {
        // If no seed is provided, use the default BLAKE3 hasher
        blake3_hasher_init(&hasher);
    }
    // Hash the data
    blake3_hasher_update(&hasher, data, datalen);
    // Finalise the hash and write it to the output buffer (arbitrary length)
    blake3_hasher_finalize(&hasher, out, hash_size);
}

// Hashes an array of uint64 elements with a seed using BLAKE3, outputs variable-length hashes
// - If a seed is provided, the keyed mode is used by setting the key (up to 32 bytes, zero-padded if shorter)
// - Output is a 2D uint8 array of shape n x hash_size (n rows, hash_size columns)
// - hash_size can be any positive value (BLAKE3 is an XOF)
void numpy_blake3(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out, const size_t hash_size) {
    // Input: native-endian uint64_t array; each element is hashed as an 8-byte block
    const unsigned char (*arr8)[BLOCK_SIZE] = (const unsigned char (*)[BLOCK_SIZE])arr;
    const bool seeded = seed != NULL && seedlen > 0;
    const int n_int = (int)n;
    int i;

    // Prepare the key outside the loop if seeded
    uint8_t key[BLAKE3_KEY_LEN] = {0};
    prepare_blake3_key(seeded, seed, seedlen, key);

    #ifdef _OPENMP
    // Parallelise the loop with OpenMP to use multiple CPU cores
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < n_int; ++i) {
        // Hash the 8-byte block
        blake3_hash_bytes(arr8[i], BLOCK_SIZE, key, seeded, &out[i * hash_size], hash_size);
    }
}

// Hashes a single byte array with BLAKE3, outputs variable-length hash
// - If a seed is provided, the keyed mode is used by setting the key (up to 32 bytes, zero-padded if shorter)
// - Output is a uint8 array of length hash_size
void bytes_blake3(const uint8_t* restrict data, const size_t datalen, const char* restrict seed, const size_t seedlen, uint8_t* restrict out, const size_t hash_size) {
    // Input: byte array; each byte is hashed as a single byte block
    const bool seeded = seed != NULL && seedlen > 0;
    const uint8_t *final_data = data;
    size_t final_len = datalen;
    int i;

    // Prepare the key if seeded
    uint8_t key[BLAKE3_KEY_LEN] = {0};
    prepare_blake3_key(seeded, seed, seedlen, key);

    #ifdef _OPENMP
    // Attempt to chunk and parallelise with OpenMP to use multiple CPU cores

    // Calculate the number of chunks based on the input data length
    const int datalen_int = (int)datalen;
    const int num_chunks = (datalen_int + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Define a structure to hold the chaining values (CVs) for each chunk
    typedef struct {
        uint8_t cv[BLAKE3_OUT_LEN];
        uint64_t chunk_index;
        int num_blocks;
    } cv_node_t;
    cv_node_t* cvs = NULL;
    if (num_chunks > 1) {
        cvs = (cv_node_t*)malloc(num_chunks * sizeof(cv_node_t));
    }

    // Use parallel tree hashing for large inputs, fallback to serial for small or allocation failure
    if (datalen >= MIN_PARALLEL_SIZE && num_chunks > 1 && cvs != NULL) {
        // Parallel tree hashing path

        // Estimate the CVs for each chunk
        #pragma omp parallel for schedule(static)
        for (i = 0; i < num_chunks; ++i) {
            // Calculate the offset and length for this chunk
            const int offset = i * CHUNK_SIZE;
            const int end = MIN(offset + CHUNK_SIZE, datalen_int);
            const int len = end - offset;

            // Hash the chunk
            blake3_hash_bytes(data + offset, len, key, seeded, cvs[i].cv, BLAKE3_OUT_LEN);

            // Set the chunk index and number of blocks
            cvs[i].chunk_index = i;
            cvs[i].num_blocks = 1;
        }

        // Two buffers (ping-pong buffers) are required here to ensure that each reduction round
        // reads from one buffer and writes to the other.
        cv_node_t* next_cvs = (cv_node_t*)malloc(num_chunks * sizeof(cv_node_t));
        if (next_cvs) {
            // Perform tree reduction to combine CVs
            cv_node_t* read_buf = cvs;
            cv_node_t* write_buf = next_cvs;
            int num_nodes = num_chunks;
            while (num_nodes > 1) {
                const int new_nodes = (num_nodes + 1) / 2;
                #pragma omp parallel for schedule(static)
                for (i = 0; i < new_nodes; ++i) {
                    // For each node, combine the left and right children
                    const int left_index = 2 * i;
                    const int right_index = left_index + 1;
                    const cv_node_t *const left_node = &read_buf[left_index];

                    if (right_index < num_nodes) {
                        // Merge left/right children
                        const cv_node_t *const right_node = &read_buf[right_index];

                        // Create a block of BLAKE3_OUT_DOUBLE_LEN bytes from two CVs
                        uint8_t block[BLAKE3_OUT_DOUBLE_LEN];
                        memcpy(block, left_node->cv, BLAKE3_OUT_LEN);
                        memcpy(block + BLAKE3_OUT_LEN, right_node->cv, BLAKE3_OUT_LEN);

                        // Hash the combined block
                        blake3_hash_bytes(block, BLAKE3_OUT_DOUBLE_LEN, key, seeded, write_buf[i].cv, BLAKE3_OUT_LEN);

                        // Set the chunk index and number of blocks
                        write_buf[i].chunk_index = left_node->chunk_index;
                        write_buf[i].num_blocks = left_node->num_blocks + right_node->num_blocks;
                    } else {
                        // Odd node: promote as-is
                        memcpy(write_buf[i].cv, left_node->cv, BLAKE3_OUT_LEN);
                        write_buf[i].chunk_index = left_node->chunk_index;
                        write_buf[i].num_blocks = left_node->num_blocks;
                    }
                }
                // Swap buffers for the next iteration
                cv_node_t* tmp = read_buf;
                read_buf = write_buf;
                write_buf = tmp;
                num_nodes = new_nodes;
            }

            // Prepare to finalise with the root CV
            final_data = read_buf[0].cv;
            final_len = BLAKE3_OUT_LEN;
        }
        free(next_cvs);
    }
    free(cvs);
    #endif

    // Finalise hash
    blake3_hash_bytes(final_data, final_len, key, seeded, out, hash_size);
}



