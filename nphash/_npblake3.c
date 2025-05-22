#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "blake3.h"

#ifdef _OPENMP
// Enable parallelisation with OpenMP for multi-core performance
#include <omp.h>
#endif

#define BLOCK_SIZE 8  // Each input element is a uint64 block

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

    // Prepare the key and copylen outside the loop if seeded
    uint8_t key[BLAKE3_KEY_LEN] = {0};
    const size_t copylen = seedlen < BLAKE3_KEY_LEN ? seedlen : BLAKE3_KEY_LEN;
    if (seeded) {
        memcpy(key, seed, copylen);
    }

    #ifdef _OPENMP
    // Parallelise the loop with OpenMP to use multiple CPU cores
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < n_int; ++i) {
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
