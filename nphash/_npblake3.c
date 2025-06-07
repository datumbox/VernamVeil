#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "_npblake3.h"

#include "blake3.h"

#ifdef _OPENMP
// Enable parallelisation with OpenMP for multi-core performance
#include <omp.h>
#endif

#define BLOCK_SIZE 8  // Each input element is a uint64 block
#define MIN_PARALLEL_LEN (2 * BLAKE3_CHUNK_LEN) // Minimum data length to enable parallel tree hashing

// Inline helper to prepare a BLAKE3 key from a seed (up to 32 bytes, zero-padded if shorter)
static inline void prepare_blake3_key(bool seeded, const char* seed, size_t seedlen, uint8_t key[BLAKE3_KEY_LEN]) {
    if (seeded) {
        const size_t copylen = seedlen < BLAKE3_KEY_LEN ? seedlen : BLAKE3_KEY_LEN;
        memcpy(key, seed, copylen);
    }
}

// Hashes an array of uint64 elements with a seed using BLAKE3, outputs variable-length hashes
// - If a seed is provided, the keyed mode is used by setting the key (up to 32 bytes, zero-padded if shorter)
// - Output is a 2D uint8 array of shape n x hash_size (n rows, hash_size columns)
// - hash_size can be any positive value (BLAKE3 is an XOF)
void numpy_blake3(const uint64_t* arr, size_t n, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size) {
    // Input: native-endian uint64_t array; each element is hashed as an 8-byte block
    const unsigned char (*arr8)[BLOCK_SIZE] = (const unsigned char (*)[BLOCK_SIZE])arr;
    bool seeded = seed != NULL && seedlen > 0;
    int n_int = (int)n;
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
            // If no seed is provided, use the default BLAKE3 hasher
            blake3_hasher_init(&hasher);
        }

        // Hash the data
        blake3_hasher_update(&hasher, arr8[i], BLOCK_SIZE);

        // Finalise the hash and write it to the output buffer (arbitrary length)
        blake3_hasher_finalize(&hasher, &out[i * hash_size], hash_size);
    }
}

// Hashes multiple data chunks with BLAKE3, outputs variable-length hash
// - If a seed is provided, the keyed mode is used by setting the key (up to 32 bytes, zero-padded if shorter)
// - Output is a uint8 array of length hash_size
void bytes_blake3_multi_chunk(const uint8_t* const* data_chunks, const size_t* data_lengths, size_t num_chunks, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size) {
    // Input: data_chunks is an array of pointers to data buffers that are to be hashed.
    blake3_hasher hasher;
    bool seeded = seed != NULL && seedlen > 0;

    // Prepare the key if seeded
    uint8_t key[BLAKE3_KEY_LEN] = {0};
    prepare_blake3_key(seeded, seed, seedlen, key);

    if (seeded) {
        // If a seed is provided, use it as the BLAKE3 key (up to 32 bytes, zero-padded if shorter)
        blake3_hasher_init_keyed(&hasher, key);
    } else {
        // If no seed is provided, use the default BLAKE3 hasher
        blake3_hasher_init(&hasher);
    }

    for (size_t i = 0; i < num_chunks; ++i) {
         // Hash the data
#ifdef BLAKE3_USE_TBB
        if (data_lengths[i] >= MIN_PARALLEL_LEN) {
            blake3_hasher_update_tbb(&hasher, data_chunks[i], data_lengths[i]);
        } else {
#else
        {
            // If not using TBB, use the standard update function
#endif
            blake3_hasher_update(&hasher, data_chunks[i], data_lengths[i]);
        }
    }

    // Finalise the hash and write it to the output buffer (arbitrary length)
    blake3_hasher_finalize(&hasher, out, hash_size);
}
