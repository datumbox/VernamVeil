# Building the `nphash` C Library with `build.py`

This project provides an optional C extension called `nphash` to efficiently compute BLAKE2b and SHA-256 based hashes from Python. The Python method `hash_numpy` can be used in `fx` methods to quickly produce required keyed hashing in vectorised implementations.

The C code is compiled and wrapped for Python using the [cffi](https://cffi.readthedocs.io/en/latest/) library.

> **Note:** The C extension is optional. If it fails to build or is unavailable, VernamVeil will transparently fall back to a pure Python/NumPy implementation (with reduced performance).

## Prerequisites

Before building, ensure you have the following dependencies installed:

- **Python 3.10 or later**
- **pip** (Python package manager)
- **gcc** (GNU Compiler Collection)
- **OpenMP** (usually included with gcc)
- **OpenSSL development libraries**
- **cffi** and **numpy** Python packages

Supported platforms: Linux, macOS, and Windows (with suitable build tools).

## Installation Steps

1. **Install system dependencies**

   On Ubuntu/Debian:  
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential libssl-dev python3-dev
   ```

   On Fedora:  
   ```bash
   sudo dnf install gcc openssl-devel python3-devel
   ```

   On Mac (with Homebrew):
   ```bash
   brew install libomp openssl
   ```
   
   On Windows (with Chocolatey):
   ```bash
   choco install openssl
   ```

2. **Install Python dependencies**

   ```bash
   pip install cffi numpy
   ```

3. **Build the C extension**

   Run the following command in the project directory (where `build.py` is located):

   ```bash
   python build.py
   ```

   This will compile the C code and generate libraries named `_npblake2bffi.*.so` and `_npsha256ffi.*.so` (the exact filenames depend on your platform and Python version).

4. **Reinstall the library**

   Following successful compilation, the C extension reinstall the `vernamveil` package to ensure the new C extension is used. Execute the following from the root of the project:

   ```bash
   pip install .
   ```

## Usage

To confirm that the C extension is compiled and being used by VernamVeil, you can check the internal boolean `_HAS_C_MODULE`:

```python
from vernamveil._hash_utils import _HAS_C_MODULE
# True if the C extension is available and in use, False otherwise.
```

After building, you can use the extension from Python code:

```python
from vernamveil import hash_numpy
# hash_numpy will use the C extension if available, otherwise a pure NumPy fallback.
# Both BLAKE2b and SHA-256 are supported via the C extension.
```

If the C extension is not built or importable, `hash_numpy` will transparently fall back to a slower pure NumPy implementation. No code changes are needed.

## Notes

- If you change the C code, rerun `python build.py` to rebuild the extension. Reinstall the package afterwards.
- If you encounter errors about missing OpenMP or OpenSSL, ensure the development libraries are installed as shown above.
- **Apple Silicon (M1/M2/M3) macOS users:** If you see linker warnings about missing `x86_64` architecture or universal/fat binaries, set the following environment variable before building to ensure the build targets only your native architecture:
   ```bash
   export ARCHFLAGS=-arch arm64
   python build.py
   ```
