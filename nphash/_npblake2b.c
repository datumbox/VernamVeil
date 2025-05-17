#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>

#ifdef _OPENMP
// Enable parallelisation with OpenMP for multi-core performance
#include <omp.h>
#endif

#define BLOCK_SIZE 8  // Each input element is an uint64 block
#define HASH_SIZE 64  // BLAKE2b output size in bytes

// HMACs or hashes an array of uint64 elements with a seed using BLAKE2b, outputs 64-byte hashes
// - If a seed is provided, HMAC is used for cryptographic safety.
// - If no seed is provided, a plain hash is used (not recommended for security).
// - Output is a 2D uint8 array of shape n x 64 (n rows, 64 columns)
void numpy_blake2b(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out) {
    // Input: native-endian uint64_t array; each element is hashed as an 8-byte block
    // When passing to hash/HMAC, use (const unsigned char *)&arr[i] and BLOCK_SIZE
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
            // Write HMAC output directly to the output buffer
            HMAC(EVP_blake2b512(), seed, seedlen_int, arr8[i], BLOCK_SIZE, &out[i * HASH_SIZE], NULL);
        }
    } else {
        // No-seed mode: Just hash the block
        #ifdef _OPENMP
        // Parallelise the loop with OpenMP to use multiple CPU cores
        #pragma omp parallel for schedule(static)
        #endif
        for (i = 0; i < n_int; ++i) {
            // Create a new digest context for each hash computation
            EVP_MD_CTX* ctx = EVP_MD_CTX_new();
            // Compute BLAKE2b hash of the uint64 block by chaining all hash steps
            int ok = (ctx != NULL);
            ok &= EVP_DigestInit_ex(ctx, EVP_blake2b512(), NULL) == 1;
            ok &= EVP_DigestUpdate(ctx, arr8[i], BLOCK_SIZE) == 1;
            ok &= EVP_DigestFinal_ex(ctx, &out[i * HASH_SIZE], NULL) == 1;
            EVP_MD_CTX_free(ctx);
        }
    }
}
