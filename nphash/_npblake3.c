#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "blake3.h"

#ifdef _OPENMP
// Enable parallelisation with OpenMP for multi-core performance
#include <omp.h>
#define CHUNK_SIZE 1024  // Size of each chunk in bytes for parallel processing
#define MAX_CHUNKS 256  // Maximum number of chunks to process in parallel
#endif

#define BLOCK_SIZE 8  // Each input element is a uint64 block

// Inline helper to prepare a BLAKE3 key from a seed (up to 32 bytes, zero-padded if shorter)
static inline void prepare_blake3_key(bool seeded, const char* restrict seed, const size_t seedlen, uint8_t key[BLAKE3_KEY_LEN]) {
    if (seeded) {
        const size_t copylen = seedlen < BLAKE3_KEY_LEN ? seedlen : BLAKE3_KEY_LEN;
        memcpy(key, seed, copylen);
    }
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
        // Initialise the BLAKE3 hasher
        blake3_hasher hasher;

        if (seeded) {
            // If a seed is provided, use it as the BLAKE3 key (up to 32 bytes, zero-padded if shorter)
            blake3_hasher_init_keyed(&hasher, key);
        } else {
            blake3_hasher_init(&hasher);
        }

        // Hash the 8-byte block
        blake3_hasher_update(&hasher, arr8[i], BLOCK_SIZE);

        // Finalize the hash and write it to the output buffer (arbitrary length)
        blake3_hasher_finalize(&hasher, &out[i * hash_size], hash_size);
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
    int num_chunks = (datalen + CHUNK_SIZE - 1) / CHUNK_SIZE;
    if (num_chunks > MAX_CHUNKS) num_chunks = MAX_CHUNKS;

    // Define a structure to hold the chaining values (CVs) for each chunk
    typedef struct {
        uint8_t cv[BLAKE3_OUT_LEN];
        uint64_t chunk_index;
        size_t  num_blocks;
    } cv_node_t;
    cv_node_t cvs[MAX_CHUNKS];

    // Use parallel tree hashing for large inputs, fallback to serial for small
    if (datalen >= CHUNK_SIZE * 2 && num_chunks > 1) {
        // Parallel tree hashing path
        #pragma omp parallel for schedule(static)
        for (i = 0; i < num_chunks; ++i) {
            size_t offset = i * CHUNK_SIZE;
            size_t len = (offset + CHUNK_SIZE <= datalen) ? CHUNK_SIZE : (datalen - offset);
            // Initialise the BLAKE3 hasher
            blake3_hasher hasher;

            if (seeded) {
                // If a seed is provided, use it as the BLAKE3 key (up to 32 bytes, zero-padded if shorter)
                blake3_hasher_init_keyed(&hasher, key);
            } else {
                blake3_hasher_init(&hasher);
            }
            blake3_hasher_update(&hasher, data + offset, len);
            blake3_hasher_finalize(&hasher, cvs[i].cv, BLAKE3_OUT_LEN);

            cvs[i].chunk_index = i;
            cvs[i].num_blocks = 1;
        }
        // Tree reduction (combine CVs)
        int num_nodes = num_chunks;
        while (num_nodes > 1) {
            size_t new_nodes = (num_nodes + 1) / 2;
            #pragma omp parallel for schedule(static)
            for (i = 0; i < new_nodes; ++i) {
                const int left_index = 2 * i;
                const int right_index = left_index + 1;
                if (right_index < num_nodes) {
                    // Merge left/right children
                    uint8_t block[64];
                    memcpy(block, cvs[left_index].cv, 32);
                    memcpy(block + 32, cvs[right_index].cv, 32);
                    blake3_hasher hasher;
                    if (seeded) {
                        blake3_hasher_init_keyed(&hasher, key);
                    } else {
                        blake3_hasher_init(&hasher);
                    }
                    // Parent node: always 64 bytes, PARENT flag
                    blake3_hasher_update(&hasher, block, 64);
                    blake3_hasher_finalize(&hasher, cvs[i].cv, BLAKE3_OUT_LEN);
                    cvs[i].chunk_index = cvs[left_index].chunk_index;
                    cvs[i].num_blocks = cvs[left_index].num_blocks + cvs[right_index].num_blocks;
                } else {
                    // Odd node: promote as-is
                    memcpy(cvs[i].cv, cvs[left_index].cv, BLAKE3_OUT_LEN);
                    cvs[i].chunk_index = cvs[left_index].chunk_index;
                    cvs[i].num_blocks = cvs[left_index].num_blocks;
                }
            }
            num_nodes = new_nodes;
        }
        // Prepare to finalize with the root CV
        final_data = cvs[0].cv;
        final_len = BLAKE3_OUT_LEN;
    }
    #endif

    // Finalize hash
    blake3_hasher hasher;
    if (seeded) {
        blake3_hasher_init_keyed(&hasher, key);
    } else {
        blake3_hasher_init(&hasher);
    }
    blake3_hasher_update(&hasher, final_data, final_len);
    blake3_hasher_finalize(&hasher, out, hash_size);
}
