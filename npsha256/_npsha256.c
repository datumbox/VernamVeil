#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <openssl/sha.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Hashes an array of 4-byte elements with a seed using SHA256, outputs 64-bit values
void numpy_sha256(const char* arr, size_t n, const char* seed, size_t seedlen, uint64_t* out) {
    // Treat input as an array of 4-byte blocks
    const char (*arr4)[4] = (const char (*)[4])arr;

    int i;
    #ifdef _OPENMP
    // Parallelize the loop with OpenMP to use multiple CPU cores
    #pragma omp parallel for
    #endif
    for (i = 0; i < n; ++i) {
        unsigned char hash[32]; // Buffer for SHA256 output (32 bytes)
        // Dynamically allocate buffer for seed + 4 bytes
        unsigned char* buf = (unsigned char*)malloc(seedlen + 4);
        if (!buf) {
            // Allocation failed, skip this output
            out[i] = 0;
            continue;
        }

        // Copy the seed into the buffer
        if (seedlen > 0 && seed != NULL) {
            memcpy(buf, seed, seedlen);
        }
        // Append the current 4-byte block to the buffer
        memcpy(buf + seedlen, arr4[i], 4);

        // Compute SHA256 hash of (seed || 4-byte block)
        SHA256(buf, seedlen + 4, hash);

        // Convert the 32-byte hash to a 64-bit integer (big-endian)
        uint64_t val = 0;
        for (size_t j = 0; j < 32; ++j) {
            val = (val << 8) | hash[j];
        }
        // Store the result in the output array
        out[i] = val;

        free(buf);
    }
}
