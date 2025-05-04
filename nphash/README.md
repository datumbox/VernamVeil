# Building the `nphash` C Library with `build.py`

This project optionally uses a C extension called `nphash` to efficiently compute BLAKE2b and SHA-256 based hashes from Python. The Python method `hash_numpy` can be used in `fx` methods to quickly produce required HMACs in vectorised implementations.

The C code is compiled and wrapped for Python using the [cffi](https://cffi.readthedocs.io/en/latest/) library.

> **Note:** The C extension is optional. If it fails to build or is unavailable, VernamVeil will transparently fall back to a pure Python/NumPy implementation (with reduced performance).

## Prerequisites

Before building, ensure you have the following dependencies installed:

- **Python 3.x**
- **pip** (Python package manager)
- **gcc** (GNU Compiler Collection)
- **OpenMP** (usually included with gcc)
- **OpenSSL development libraries**
- **cffi** and **numpy** Python packages

Supported platforms: Linux, macOS, and Windows (with suitable build tools).

## Installation Steps

1. **Install system dependencies**

   On Ubuntu/Debian:  
   ```
   sudo apt-get update
   sudo apt-get install build-essential libssl-dev python3-dev
   ```

   On Fedora:  
   ```
   sudo dnf install gcc openssl-devel python3-devel
   ```

   On Mac (with Homebrew):
   ```
   brew install libomp openssl
   ```
   
   On Windows (with Chocolatey):
   ```
   choco install openssl
   ```

2. **Install Python dependencies**

   ```
   pip install cffi numpy
   ```

3. **Build the C extension**

   Run the following command in the project directory (where `build.py` is located):

   ```
   python build.py
   ```

   This will compile the C code and generate libraries named `_npblake2bffi.*.so` and `_npsha256ffi.*.so` (the exact filenames depend on your platform and Python version).

   Following successful compilation, the C extension reinstall the `vernamveil` package to ensure the new C extension is used.

## Usage

After building, you can import and use the extension from Python code:

```python
from vernamveil import hash_numpy
# hash_numpy will use the C extension if available, otherwise a pure NumPy fallback.
# Both BLAKE2b and SHA-256 are supported via the C extension.
```

If the C extension is not built or importable, `hash_numpy` will transparently fall back to a slower pure NumPy implementation. No code changes are needed.

## Notes

- If you change the C code in `build.py`, rerun `python build.py` to rebuild the extension.
- If you encounter errors about missing OpenMP or OpenSSL, ensure the development libraries are installed as shown above.
