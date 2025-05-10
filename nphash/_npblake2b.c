#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include "fold_bytes.h"

#ifdef _OPENMP
// Enable parallelisation with OpenMP for multi-core performance
#include <omp.h>
#endif

#define BLOCK_SIZE 8  // Each input element is an 8-byte (uint64) block

// HMACs or hashes an array of 8-byte (uint64) elements with a seed using BLAKE2b, outputs 64-bit values
// - If a seed is provided, HMAC is used for cryptographic safety.
// - If no seed is provided, a plain hash is used (not recommended for security).
void numpy_blake2b(const char* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint64_t* restrict out) {
    // Treat input as an array of 8-byte (uint64) blocks for hashing/HMAC
    const unsigned char (*arr8)[BLOCK_SIZE] = (const unsigned char (*)[BLOCK_SIZE])arr;
    const int n_int = (int)n;
    int i;

    if (seed != NULL && seedlen > 0) {
        const int seedlen_int = (int)seedlen;

        // HMAC mode: Use HMAC with BLAKE2b (cryptographically safer)
        #ifdef _OPENMP
        // Parallelise the loop with OpenMP to use multiple CPU cores
        #pragma omp parallel for schedule(static)
        #endif
        for (i = 0; i < n_int; ++i) {
            unsigned char hash[64]; // Buffer for BLAKE2b output (64 bytes)
            unsigned int hash_len = 64;
            // Compute HMAC-BLAKE2b using OpenSSL: key=seed, msg=arr8[i]
            HMAC(EVP_blake2b512(), seed, seedlen_int, arr8[i], BLOCK_SIZE, hash, &hash_len);
            // Fold the hash into a uint64 for output
            out[i] = fold_bytes_to_uint64(hash, 64);
        }
    } else {
        // No-seed mode: Just hash the block
        #ifdef _OPENMP
        // Parallelise the loop with OpenMP to use multiple CPU cores
        #pragma omp parallel for schedule(static)
        #endif
        for (i = 0; i < n_int; ++i) {
            unsigned char hash[64]; // Buffer for BLAKE2b output (64 bytes)
            // Create a new digest context for each hash computation
            EVP_MD_CTX* ctx = EVP_MD_CTX_new();
            // Compute BLAKE2b hash of the 8-byte (uint64) block
            // Chain all hash steps and check for failure
            if (ctx != NULL &&
                EVP_DigestInit_ex(ctx, EVP_blake2b512(), NULL) == 1 &&
                EVP_DigestUpdate(ctx, arr8[i], BLOCK_SIZE) == 1 &&
                EVP_DigestFinal_ex(ctx, hash, NULL) == 1) {
                // Fold the hash into a uint64 for output
                out[i] = fold_bytes_to_uint64(hash, 64);
            }
            EVP_MD_CTX_free(ctx);
        }
    }
}
