#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <openssl/evp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Use a stack buffer for most cases (seedlen up to 516)
#define STACK_BUF_SIZE 516

// Hashes an array of 4-byte elements with a seed using BLAKE2b, outputs 64-bit values
void numpy_blake2b(const char* arr, size_t n, const char* seed, size_t seedlen, uint64_t* out) {
    // Treat input as an array of 4-byte blocks
    const char (*arr4)[4] = (const char (*)[4])arr;

    size_t buflen = seedlen + 4;

    int i;
    int n_int = (int)n;
    #ifdef _OPENMP
    // Parallelise the loop with OpenMP to use multiple CPU cores
    #pragma omp parallel for
    #endif
    for (i = 0; i < n_int; ++i) {
        unsigned char hash[64]; // Buffer for BLAKE2b output (64 bytes)
        unsigned char stack_buf[STACK_BUF_SIZE];
        unsigned char* buf;

        // Dynamically allocate buffer for large seeds
        if (buflen > STACK_BUF_SIZE) {
            buf = (unsigned char*)malloc(buflen);
            if (!buf) {
                // Allocation failed, skip this output
                out[i] = 0;
                continue;
            }
        } else {
            buf = stack_buf;
        }

        // Copy the seed into the buffer
        if (seedlen > 0 && seed != NULL) {
            memcpy(buf, seed, seedlen);
        }
        // Append the current 4-byte block to the buffer
        memcpy(buf + seedlen, arr4[i], 4);

        // Compute BLAKE2b hash of (seed || 4-byte block)
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (ctx == NULL) {
            out[i] = 0;
            if (buflen > STACK_BUF_SIZE) {
                free(buf);
            }
            continue;
        }

        if (EVP_DigestInit_ex(ctx, EVP_blake2b512(), NULL) != 1 ||
            EVP_DigestUpdate(ctx, buf, buflen) != 1 ||
            EVP_DigestFinal_ex(ctx, hash, NULL) != 1) {
            out[i] = 0;
            EVP_MD_CTX_free(ctx);
            if (buflen > STACK_BUF_SIZE) {
                free(buf);
            }
            continue;
        }

        EVP_MD_CTX_free(ctx);

        // Convert the 64-byte hash to a 64-bit integer (big-endian)
        uint64_t val = 0;
        for (size_t j = 0; j < 64; ++j) {
            val = (val << 8) | hash[j];
        }
        // Store the result in the output array
        out[i] = val;

        // Free the dynamically allocated buffer if it was used
        // (i.e., when the required buffer size exceeded the stack buffer)
        if (buflen > STACK_BUF_SIZE) {
            free(buf);
        }
    }
}