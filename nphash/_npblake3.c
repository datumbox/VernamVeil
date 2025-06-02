#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "_npblake3.h"

#include "blake3.h"

#ifdef _OPENMP
// Enable parallelisation with OpenMP for multi-core performance
#include <omp.h>
#endif

#define BLOCK_SIZE 8  // Each input element is a uint64 block
#define MIN_PARALLEL_LEN (2 * BLAKE3_CHUNK_LEN) // Minimum data length to enable parallel tree hashing

// Platform-specific alignment for SIMD acceleration
#if defined(__AVX512F__) || defined(__AVX512VL__)
    #define BLAKE3_ALIGNMENT 64
#elif defined(__AVX2__)
    #define BLAKE3_ALIGNMENT 32
#elif defined(__SSE2__)
    #define BLAKE3_ALIGNMENT 16
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define BLAKE3_ALIGNMENT 16
#else
    #define BLAKE3_ALIGNMENT 8
#endif

// Inline helper to prepare a BLAKE3 key from a seed (up to 32 bytes, zero-padded if shorter)
static inline void prepare_blake3_key(bool seeded, const char* seed, size_t seedlen, uint8_t key[BLAKE3_KEY_LEN]) {
    if (seeded) {
        const size_t copylen = seedlen < BLAKE3_KEY_LEN ? seedlen : BLAKE3_KEY_LEN;
        memcpy(key, seed, copylen);
    }
}

// Inline helper to hash data with BLAKE3, with or without a key, and output variable-length hash
// If parallel, uses blake3_hasher_update_tbb for multithreading
static inline void blake3_hash_bytes(const uint8_t* data, size_t datalen, const uint8_t* key, bool seeded, uint8_t* out, size_t hash_size, bool parallel) {
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
#ifdef BLAKE3_USE_TBB
    if (parallel) {
        blake3_hasher_update_tbb(&hasher, data, datalen);
    } else {
#else
    {
        // If not using TBB, use the standard update function
#endif
        blake3_hasher_update(&hasher, data, datalen);
    }
    // Finalise the hash and write it to the output buffer (arbitrary length)
    blake3_hasher_finalize(&hasher, out, hash_size);
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
        // Hash the 8-byte block
        blake3_hash_bytes(arr8[i], BLOCK_SIZE, key, seeded, &out[i * hash_size], hash_size, false);
    }
}

// Hashes a single byte array with BLAKE3, outputs variable-length hash
// - If a seed is provided, the keyed mode is used by setting the key (up to 32 bytes, zero-padded if shorter)
// - Output is a uint8 array of length hash_size
void bytes_blake3(const uint8_t* data, size_t datalen, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size) {
    // Input: byte array; each byte is hashed as a single byte block
    bool seeded = seed != NULL && seedlen > 0;
    bool use_parallel = datalen >= MIN_PARALLEL_LEN;

    // Prepare the key if seeded
    uint8_t key[BLAKE3_KEY_LEN] = {0};
    prepare_blake3_key(seeded, seed, seedlen, key);

    // Alignment check and copy if needed
    const uint8_t* aligned_data = data;
    void* aligned_buf = NULL;

    if (use_parallel && ((uintptr_t)data) % BLAKE3_ALIGNMENT != 0) {
        // Data is not aligned and large enough to benefit from alignment; allocate aligned buffer and copy
        if (posix_memalign(&aligned_buf, BLAKE3_ALIGNMENT, datalen) == 0) {
            memcpy(aligned_buf, data, datalen);
            aligned_data = (const uint8_t*)aligned_buf;
        }
    }

    // Hash the byte array
    blake3_hash_bytes(aligned_data, datalen, key, seeded, out, hash_size, use_parallel);

    // Free aligned buffer
    free(aligned_buf);
}
