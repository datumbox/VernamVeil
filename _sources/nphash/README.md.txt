# Building the `nphash` C Library with `build.py`

This project provides an optional C extension called `nphash` to efficiently compute BLAKE2b, BLAKE3 and SHA-256 based hashes from Python. The Python method `hash_numpy` can be used in `fx` methods to quickly produce required keyed hashing in vectorised implementations. We also provide a `blake3` class, which offers a hashlib-style BLAKE3 hash object using the C backend. **The `blake3` implementation is only available when the C extension is built.**

In addition to hashing, the extension provides a fast byte search capability for efficiently finding all occurrences of a byte pattern within a buffer. This is exposed via the `find` and `find_all` methods in the C extension and is used internally by VernamVeil for pattern and delimiter detection in binary data. When imported, the `find` and `find_all` methods will automatically use the C extension for maximum performance if available, otherwise falling back to a pure Python implementation.

The C and C++ code is compiled and wrapped for Python using the [cffi](https://cffi.readthedocs.io/en/latest/) library.

> **Note:** The C extension is optional. If it fails to build or is unavailable, VernamVeil will transparently fall back to a pure Python/NumPy implementation (with reduced performance).

> **Note:** The `build.py` script will automatically download the required BLAKE3 C and C++ source files from the [official BLAKE3 repository](https://github.com/BLAKE3-team/BLAKE3) if they are not already present locally.

## Hardware Acceleration and SIMD Support

The BLAKE3 implementation in this extension uses the official [BLAKE3 C/C++ codebase](https://github.com/BLAKE3-team/BLAKE3/tree/master/c) and can automatically take advantage of hardware acceleration features for optimal performance. Specifically, BLAKE3 supports the following SIMD instruction sets, if available on your platform and enabled during build:

- SSE2, SSE4.1, AVX2, AVX512F, AVX512VL (on x86_64)
- NEON (on ARM)
- Hand-written assembly routines (where supported)

These features are detected and enabled automatically by the build system. No manual configuration is required unless you wish to disable them (see below).

## Prerequisites

Before building, ensure you have the following dependencies installed:

- **Python 3.10 or later**
- **pip** (Python package manager)
- **gcc/g++** (GNU Compiler Collection, including C++ support)  
  _or_ **Microsoft Visual Studio Build Tools** (MSVC, for Windows)
- **OpenMP** (usually included with gcc/g++)
- **OpenSSL development libraries**
- **TBB (Threading Building Blocks) development libraries**
- **cffi** and **numpy** Python packages

Supported platforms: Linux, macOS, and Windows (with suitable build tools).

## Installation Steps

1. **Install system dependencies**

   On Ubuntu/Debian:  
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential g++ libssl-dev libtbb-dev python3-dev
   ```

   On Fedora:  
   ```bash
   sudo dnf install gcc gcc-c++ openssl-devel python3-devel tbb-devel
   ```

   On Mac (with Homebrew):
   ```bash
   brew install libomp openssl tbb
   ```
   
   On Windows (with vcpkg and Chocolatey):

   1. Install `tbb` with [vcpkg](https://github.com/microsoft/vcpkg):
      ```bash
      git clone https://github.com/microsoft/vcpkg.git
      call .\vcpkg\bootstrap-vcpkg.bat -disableMetrics
      call .\vcpkg\vcpkg.exe install tbb:x64-windows
      ```
   2. Copy the TBB DLLs to your build directory (if needed):
      ```bash
      copy .\vcpkg\installed\x64-windows\bin\tbb*.dll nphash\
      ```
   3. Set the environment variables for include and lib paths (if needed for the current session):
      ```bash
      set INCLUDE=%CD%\vcpkg\installed\x64-windows\include;%INCLUDE%
      set LIB=%CD%\vcpkg\installed\x64-windows\lib;%LIB%
      ```
   4. Install OpenSSL and Visual Studio 2022 Build Tools with Chocolatey:
      ```bash
      choco install openssl visualstudio2022buildtools --no-progress -y
      ```
      This installs the required compiler and assembler (ml64) for building assembly files on Windows.
   5. Open a Developer Command Prompt for VS 2022, or run the following command to set up the build environment in your terminal (replace the path if you installed Visual Studio elsewhere):
      ```bash
      call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
      ```
      This step is required to enable the Microsoft assembler (ml64) and compiler for building the C extension with assembly files.

2. **Install Python dependencies**

   ```bash
   pip install cffi numpy
   ```

3. **Build the C extension**

   Run the following command in the project directory (where `build.py` is located):

   ```bash
   python build.py
   ```

   This will compile the C code and generate libraries named `_bytesearchffi.*.so`, `_npblake2bffi.*.so`, `_npblake3ffi.*.so`  and `_npsha256ffi.*.so` (the exact filenames depend on your platform and Python version).

4. **Reinstall the library**

   Following successful compilation, the C extension reinstall the `vernamveil` package to ensure the new C extension is used. Execute the following from the root of the project:

   ```bash
   pip install .
   ```

## Advanced Configuration

By default, the build enables all available hardware and threading acceleration for optimal performance. This includes features for BLAKE3 hashing and the choice of algorithm for byte searching.

The BLAKE3 implementation benefits from:
- [oneAPI Threading Building Blocks (oneTBB)](https://uxlfoundation.github.io/oneTBB/) for multi-threaded hashing.
- SIMD (Single Instruction, Multiple Data) acceleration via C intrinsics (SSE/AVX/NEON).
- Hand-written assembly acceleration (platform-specific .S/.asm files).

For byte searching, by default we use our custom Boyer-Moore-Horspool (BMH) implementation, which is fast, consistent and portable. However, you can opt to use the `memmem` if preferred.

If you encounter issues installing dependencies for these features, or wish to use alternative algorithms for specific functionalities, you can disable or change them individually:

- **Disable TBB (multithreading) for BLAKE3:**
  - Using an environment variable:
    ```bash
    export NPBLAKE3_NO_TBB=1
    python build.py
    ```
  - Or using a command-line flag:
    ```bash
    python build.py --no-tbb
    ```

- **Disable SIMD C acceleration (SSE/AVX/NEON) for BLAKE3:**
  - Using an environment variable:
    ```bash
    export NPBLAKE3_NO_SIMD=1
    python build.py
    ```
  - Or using a command-line flag:
    ```bash
    python build.py --no-simd
    ```

- **Disable assembly acceleration for BLAKE3:**
  - Using an environment variable:
    ```bash
    export NPBLAKE3_NO_ASM=1
    python build.py
    ```
  - Or using a command-line flag:
    ```bash
    python build.py --no-asm
    ```

- **Use `memmem` instead of BMH for byte searching:**
  - Using an environment variable:
    ```bash
    export BYTESEARCH_NO_BMH=1
    python build.py
    ```
  - Or using a command-line flag:
    ```bash
    python build.py --no-bmh
    ```

If any of these are set, the build will adjust the corresponding features. **Note:** Disabling acceleration features is not recommended unless necessary, as it will reduce performance for the affected operations.

## Usage

To confirm that the C extension is compiled and being used by VernamVeil, you can check the internal boolean `_HAS_C_MODULE`:

```python
from vernamveil._types import _HAS_C_MODULE
# True if the C extension is available and in use, False otherwise.
```

After building, you can use the extension from Python code:

```python
from vernamveil import blake3, find, find_all, hash_numpy
# The `blake3` class provides a hashlib-style BLAKE3 hash object using the C backend.
# The `find` and `find_all` methods will use the C extension for fast byte search if available, otherwise a pure Python fallback.
# The `hash_numpy` will use the C extension if available, otherwise a pure NumPy fallback. All BLAKE2b, BLAKE3 and SHA-256 are supported via the C extension.
```

If the C extension is not built or importable, `find`, `find_all` and `hash_numpy` will transparently fall back to slower pure Python/NumPy implementations. No code changes are needed.

## Notes

- If you change the C code, rerun `python build.py` to rebuild the extension. Reinstall the package afterwards.
- If you encounter errors about missing OpenMP or OpenSSL, ensure the development libraries are installed as shown above.
- **Apple Silicon (M1/M2/M3) macOS users:** If you see linker warnings about missing `x86_64` architecture or universal/fat binaries, set the following environment variable before building to ensure the build targets only your native architecture:
   ```bash
   export ARCHFLAGS=-arch arm64
   python build.py
   ```
- **macOS users:** If you encounter an error like `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate` when building or downloading sources, run the `Install Certificates.command` script that comes with your Python installation. For example, in your terminal:
   ```bash
   /Applications/Python\ <your-version>/Install\ Certificates.command
   ```
  Replace `<your-version>` with your installed Python version (e.g., `3.10`). This will install the required root certificates for SSL verification.
