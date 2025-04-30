# Building the `npsha256` C Library with `build.py`

This project can optionally use a C extension called `npsha256` to efficiently compute SHA256-based hashes from Python. Then, the Python method `numpy_sha256` can be used in `fx` methods to quickly produce required hashes in vectorised implementations. 

The C code is compiled and wrapped for Python using the [cffi](https://cffi.readthedocs.io/en/latest/) library.

## Prerequisites

Before building, ensure you have the following dependencies installed:

- **Python 3.x**
- **pip** (Python package manager)
- **gcc** (GNU Compiler Collection)
- **OpenMP** (usually included with gcc)
- **OpenSSL development libraries**
  - On Ubuntu/Debian: `libssl-dev`
  - On Fedora: `openssl-devel`
- **cffi** and **numpy** Python packages

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

   This will compile the C code and generate a Python extension module named `_npsha256ffi.*.so` (the exact filename depends on your platform and Python version).

## Usage

After building, you can import and use the extension from Python code:

```python
from vernamveil import numpy_sha256
# numpy_sha256 will use the C extension if available, otherwise a pure NumPy fallback.
```

If the C extension is not built or importable, `numpy_sha256` will transparently fall back to a slower pure NumPy implementation. No code changes are needed.

## Notes

- If you change the C code in `build.py`, rerun `python build.py` to rebuild the extension.
- If you encounter errors about missing OpenMP or OpenSSL, ensure the development libraries are installed as shown above.
