"""
npsha256/build.py

Build script for the npsha256 CFFI extension.

This script uses cffi to compile a C extension (_npsha256ffi) that provides a fast, parallelized
SHA256-based hashing function for NumPy arrays. The C implementation leverages OpenMP for
multithreading and OpenSSL for cryptographic hashing.

Usage:
    python npsha256/build.py

This will generate the _npsha256ffi extension module, which can be imported from Python code.
"""

from cffi import FFI

ffibuilder = FFI()
ffibuilder.cdef(
    """
    void numpy_sha256(const char* arr, size_t n, const char* seed, size_t seedlen, uint64_t* out);
"""
)

ffibuilder.set_source(
    "_npsha256ffi",
    """
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>
#include <omp.h>

// Hashes an array of 4-byte elements with a seed using SHA256, outputs 64-bit values
void numpy_sha256(const char* arr, size_t n, const char* seed, size_t seedlen, uint64_t* out) {
    // Treat input as an array of 4-byte blocks
    const char (*arr4)[4] = (const char (*)[4])arr;

    // Parallelize the loop with OpenMP to use multiple CPU cores
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        unsigned char hash[32]; // Buffer for SHA256 output (32 bytes)
        unsigned char buf[seedlen + 4]; // Buffer for seed + 4 bytes of input

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
    }
}
    """,
    libraries=["ssl", "crypto", "gomp"],
    extra_compile_args=["-std=c99", "-fopenmp"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
