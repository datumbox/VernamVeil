"""
nphash/build.py

Build script for the nphash CFFI extension.

This script uses cffi to compile the _npsha256ffi and _npblake2bffi C extensions that provide fast, parallelised
SHA256-based and BLAKE2b-based hashing functions for NumPy arrays. The C implementations leverage OpenMP for
multithreading and OpenSSL for cryptographic hashing.

Usage:
    python nphash/build.py

This will generate the _npsha256ffi and _npblake2bffi extension modules, which can be imported from Python code.
"""

import sys
import platform
from pathlib import Path

from cffi import FFI

# FFI builders
ffibuilder_blake2b = FFI()
ffibuilder_blake2b.cdef(
    """
    void numpy_blake2b(const char* arr, size_t n, const char* seed, size_t seedlen, uint64_t* out);
"""
)

ffibuilder_sha256 = FFI()
ffibuilder_sha256.cdef(
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
        Path("/opt/homebrew/opt/openssl"),
        Path("/usr/local/opt/openssl"),
        Path("/opt/homebrew/opt/libomp"),
        Path("/usr/local/opt/libomp"),
    ]:
        if prefix.exists():
            include_dirs.append(prefix / "include")
            library_dirs.append(prefix / "lib")
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
        Path(r"C:\Program Files\OpenSSL"),
        Path(r"C:\Program Files\OpenSSL-Win64"),
        Path(r"C:\Program Files\OpenSSL-Win32"),
    ]:
        if prefix.exists():
            include_dirs.append(prefix / "include")
            library_dirs.append(prefix / "lib")
            break
else:
    raise RuntimeError("Unsupported platform")


# Add C source
c_path_blake2b = Path(__file__).parent / "_npblake2b.c"
with c_path_blake2b.open() as f:
    c_source_blake2b = f.read()

c_path = Path(__file__).parent / "_npsha256.c"
with c_path.open() as f:
    c_source = f.read()

# Add extension build
ffibuilder_blake2b.set_source(
    "_npblake2bffi",
    c_source_blake2b,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=[str(p) for p in include_dirs],
    library_dirs=[str(p) for p in library_dirs],
)

ffibuilder_sha256.set_source(
    "_npsha256ffi",
    c_source,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=[str(p) for p in include_dirs],
    library_dirs=[str(p) for p in library_dirs],
)

if __name__ == "__main__":
    ffibuilder_blake2b.compile(verbose=True)
    ffibuilder_sha256.compile(verbose=True)
