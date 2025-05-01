#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include "fold_bytes.h"

#ifdef _OPENMP
// Enable parallelisation with OpenMP for multi-core performance
#include <omp.h>
#endif

#define BLOCK_SIZE 4  // Each input element is a 4-byte block

// HMACs or hashes an array of 4-byte elements with a seed using SHA256, outputs 64-bit values
// - If a seed is provided, HMAC is used for cryptographic safety.
// - If no seed is provided, a plain hash is used (not recommended for security).
void numpy_sha256(const char* const arr, const size_t n, const char* const seed, const size_t seedlen, uint64_t* restrict out) {
    // Treat input as an array of 4-byte blocks for hashing/HMAC
    const unsigned char (*arr4)[BLOCK_SIZE] = (const unsigned char (*)[BLOCK_SIZE])arr;
    const int n_int = (int)n;

    if (seed != NULL && seedlen > 0) {
        int i;
        const int seedlen_int = (int)seedlen;

        // HMAC mode: Use HMAC with SHA256 (cryptographically safer)
        #ifdef _OPENMP
        // Parallelise the loop with OpenMP to use multiple CPU cores
        #pragma omp parallel for schedule(static)
        #endif
        for (i = 0; i < n_int; ++i) {
            unsigned char hash[32]; // Buffer for SHA256 output (32 bytes)
            unsigned int hash_len = 32;
            // Compute HMAC-SHA256 using OpenSSL: key=seed, msg=arr4[i]
            HMAC(EVP_sha256(), seed, seedlen_int, arr4[i], BLOCK_SIZE, hash, &hash_len);
            // Fold the hash into a uint64 for output
            out[i] = fold_bytes_to_uint64(hash, 32);
        }
    } else {
        // No-seed mode: Just hash the block (not recommended)
        int i;

        #ifdef _OPENMP
        // Parallelise the loop with OpenMP to use multiple CPU cores
        #pragma omp parallel for schedule(static)
        #endif
        for (i = 0; i < n_int; ++i) {
            unsigned char hash[32]; // Buffer for SHA256 output (32 bytes)
            // Compute SHA256 hash of the 4-byte block
            SHA256(arr4[i], BLOCK_SIZE, hash);
            // Fold the hash into a uint64 for output
            out[i] = fold_bytes_to_uint64(hash, 32);
        }
    }
}
