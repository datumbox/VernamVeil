"""
nphash/build.py

Build script for the nphash CFFI extension.

This script uses cffi to compile the _npsha256ffi C extension that provides a fast, parallelised
SHA256-based hashing function for NumPy arrays. The C implementation leverages OpenMP for
multithreading and OpenSSL for cryptographic hashing.

Usage:
    python nphash/build.py

This will generate the _npsha256ffi extension module, which can be imported from Python code.
"""

import sys
import platform
from pathlib import Path

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


c_path = Path(__file__).parent / "_npsha256.c"
with c_path.open() as f:
    c_source = f.read()


ffibuilder.set_source(
    "_npsha256ffi",
    c_source,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=[str(p) for p in include_dirs],
    library_dirs=[str(p) for p in library_dirs],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
