"""
npsha256/build.py

Build script for the npsha256 CFFI extension.

This script uses cffi to compile a C extension (_npsha256ffi) that provides a fast, parallelised
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
    for prefix in [
        r"C:\Program Files\OpenSSL",
        r"C:\Program Files\OpenSSL-Win64",
        r"C:\Program Files\OpenSSL-Win32",
    ]:
        if os.path.exists(prefix):
            include_dirs.append(os.path.join(prefix, "include"))
            library_dirs.append(os.path.join(prefix, "lib"))
            break
else:
    raise RuntimeError("Unsupported platform")


c_path = os.path.join(os.path.dirname(__file__), "_npsha256.c")
with open(c_path) as f:
    c_source = f.read()


ffibuilder.set_source(
    "_npsha256ffi",
    c_source,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
