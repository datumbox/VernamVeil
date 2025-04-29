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

import sys
import platform
import os

from cffi import FFI

ffibuilder = FFI()
ffibuilder.cdef(
    """
    void numpy_sha256(const char* arr, size_t n, const char* seed, size_t seedlen, uint64_t* out);
"""
)

# Platform-specific build options
libraries = []
extra_compile_args = []
extra_link_args = []
include_dirs = []
library_dirs = []

if sys.platform.startswith("linux"):
    libraries = ["ssl", "crypto", "gomp"]
    extra_compile_args = ["-std=c99", "-fopenmp"]
elif sys.platform == "darwin":
    # Homebrew OpenMP: brew install libomp openssl
    libraries = ["ssl", "crypto"]
    extra_compile_args = ["-std=c99", "-Xpreprocessor", "-fopenmp"]
    extra_link_args = ["-lomp"]
    # Add include/library dirs for both OpenSSL and libomp
    for prefix in [
        "/opt/homebrew/opt/openssl",
        "/usr/local/opt/openssl",
        "/opt/homebrew/opt/libomp",
        "/usr/local/opt/libomp",
    ]:
        if os.path.exists(prefix):
            include_dirs.append(os.path.join(prefix, "include"))
            library_dirs.append(os.path.join(prefix, "lib"))
elif sys.platform == "win32":
    # For MSVC: /openmp, for MinGW: -fopenmp
    if "GCC" in platform.python_compiler():
        libraries = ["libssl", "libcrypto", "gomp"]
        extra_compile_args = ["-std=c99", "-fopenmp"]
    else:
        # MSVC
        libraries = ["libssl", "libcrypto"]
        extra_compile_args = ["/openmp"]
    # Check all possible OpenSSL install locations
    for openssl_dir in [
        r"C:\Program Files\OpenSSL",
        r"C:\Program Files\OpenSSL-Win64",
        r"C:\Program Files\OpenSSL-Win32",
    ]:
        if os.path.exists(os.path.join(openssl_dir, "include")):
            include_dirs.append(os.path.join(openssl_dir, "include"))
            library_dirs.append(os.path.join(openssl_dir, "lib"))
            break
else:
    raise RuntimeError("Unsupported platform")

ffibuilder.set_source(
    "_npsha256ffi",
    """
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
    """,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
