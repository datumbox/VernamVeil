"""Build script for the nphash CFFI extension.

This script uses cffi to compile the _bytesearchffi, _npblake2bffi, _npblake3ffi and _npsha256ffi C extensions that provide fast byte search
and fast, parallelised BLAKE2b, BLAKE3 and SHA-256 based hashing functions for NumPy arrays. The C implementations leverage OpenMP for
multithreading and OpenSSL for cryptographic hashing.

Usage:
    python build.py

This will generate the _bytesearchffi, _npblake2bffi, _npblake3ffi and _npsha256ffi extension modules, which can be imported from Python code.
"""

import distutils.command.build_ext
import os
import tempfile
from distutils.dist import Distribution
from pathlib import Path

from nphash._build_utils._config_builder import _get_build_config
from nphash._build_utils._ffi_builders import (
    _build_ext_with_cpp11,
    _get_bytesearch_ffi,
    _get_npblake2b_ffi,
    _get_npblake3_ffi,
    _get_npsha256_ffi,
)

__all__ = ["main"]


def main() -> None:
    """Main entry point for building the nphash CFFI extensions.

    Sets up platform-specific build options, reads C source files, and compiles the extensions.
    """
    # Ensure the script runs from the directory where it is located
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Get the configuration for the build process
    config = _get_build_config()
    nphash_dir = config.nphash_dir

    # Define a temporary directory for all build artifacts
    with tempfile.TemporaryDirectory(prefix="nphash_build_") as temp_dir_str:
        build_dir = Path(temp_dir_str)
        build_dir.mkdir(parents=True, exist_ok=True)

        # Get FFI Builder Instances
        ffibuilder_bytesearch = _get_bytesearch_ffi(config, build_dir)
        ffibuilder_blake2b = _get_npblake2b_ffi(config, build_dir)
        ffibuilder_npblake3 = _get_npblake3_ffi(config, build_dir)
        ffibuilder_sha256 = _get_npsha256_ffi(config, build_dir)

        if config.tbb_enabled:
            # Patch distutils' build_ext to ensure -std=c++11 is added only for .cpp files during CFFI builds.
            # This is required for BLAKE3/TBB on macOS, and avoids breaking C builds.
            # Using setattr avoids mypy errors and keeps the patch local to this build process.
            setattr(distutils.command.build_ext, "build_ext", _build_ext_with_cpp11)

        # Compile FFI Modules with target and temp dir
        print("Compiling CFFI extensions...")

        dist = Distribution()
        build_ext_cmd = distutils.command.build_ext.build_ext(dist)
        build_ext_cmd.initialize_options()

        # Compile the CFFI extensions
        ffibuilder_bytesearch.compile(
            tmpdir=str(build_dir),
            target=str(nphash_dir / build_ext_cmd.get_ext_filename("_bytesearchffi")),
            verbose=True,
        )
        ffibuilder_blake2b.compile(
            tmpdir=str(build_dir),
            target=str(nphash_dir / build_ext_cmd.get_ext_filename("_npblake2bffi")),
            verbose=True,
        )
        ffibuilder_npblake3.compile(
            tmpdir=str(build_dir),
            target=str(nphash_dir / build_ext_cmd.get_ext_filename("_npblake3ffi")),
            verbose=True,
        )
        ffibuilder_sha256.compile(
            tmpdir=str(build_dir),
            target=str(nphash_dir / build_ext_cmd.get_ext_filename("_npsha256ffi")),
            verbose=True,
        )

        print("All CFFI extensions compiled successfully.")


if __name__ == "__main__":
    main()
