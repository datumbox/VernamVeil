#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <openssl/evp.h>

#ifdef _OPENMP
// Enable parallelisation with OpenMP for multi-core performance
#include <omp.h>
#endif

#define BLOCK_SIZE 8  // Each input element is an uint64 block
#define HASH_SIZE 32  // SHA256 output size in bytes

// Hashes an array of uint64 elements with a seed using SHA256, outputs 32-byte hashes
// - If a seed is provided, the keyed mode is used by prepending it to the input
// - Output is a 2D uint8 array of shape n x 32 (n rows, 32 columns)
void numpy_sha256(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out) {
    // Input: native-endian uint64_t array; each element is hashed as an 8-byte block
    const unsigned char (*arr8)[BLOCK_SIZE] = (const unsigned char (*)[BLOCK_SIZE])arr;
    const bool seeded = seed != NULL && seedlen > 0;
    const int seedlen_int = (int)seedlen;
    const int n_int = (int)n;
    int i;

    #ifdef _OPENMP
    // Parallelise the loop with OpenMP to use multiple CPU cores
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < n_int; ++i) {
        // Create a new digest context for each hash computation; ensure thread safety
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();

        // Compute SHA256 hash of the uint64 block by chaining all hash steps
        bool ok = (ctx != NULL) && (EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) == 1);
        if (seeded) {
            // If a seed is provided, add it first
            ok &= EVP_DigestUpdate(ctx, seed, seedlen_int) == 1;
        }
        ok &= EVP_DigestUpdate(ctx, arr8[i], BLOCK_SIZE) == 1;
        if (ok) {
            // Finalize the hash and write it to the output buffer
            EVP_DigestFinal_ex(ctx, &out[i * HASH_SIZE], NULL);
        }

        // Free the context to avoid memory leaks
        EVP_MD_CTX_free(ctx);
    }
}
