"""Build script for the nphash CFFI extension.

This script uses cffi to compile the _bytesearchffi, _npblake2bffi, _npblake3ffi and _npsha256ffi C extensions that provide fast byte search
and fast, parallelised BLAKE2b, BLAKE3 and SHA-256 based hashing functions for NumPy arrays. The C implementations leverage OpenMP for
multithreading and OpenSSL for cryptographic hashing.

Usage:
    python build.py

This will generate the _bytesearchffi, _npblake2bffi, _npblake3ffi and _npsha256ffi extension modules, which can be imported from Python code.
"""

import distutils.command.build_ext  # For setattr patch
import platform
import sys
import tempfile
from distutils.dist import Distribution
from pathlib import Path

from nphash._build_utils._ffi_builders import (
    _build_ext_with_cpp11,
    _get_bytesearch_ffi,
    _get_npblake2b_ffi,
    _get_npblake3_ffi,
    _get_npsha256_ffi,
)

__all__ = ["main"]


def _print_build_summary(
    libraries_c: list[str],
    libraries_cpp: list[str],
    extra_compile_args: list[str],
    extra_link_args: list[str],
    include_dirs: list[str],
    library_dirs: list[str],
    extra_objects: list[str],
) -> None:
    """Print a summary of the build configuration.

    Args:
        libraries_c (list): C Libraries to link against.
        libraries_cpp (list): C++ libraries to link against.
        extra_compile_args (list): Extra compiler arguments.
        extra_link_args (list): Extra linker arguments.
        include_dirs (list): Include directories.
        library_dirs (list): Library directories.
        extra_objects (list): Extra object files to link.
    """
    print("Build configuration summary:")
    print("Platform:")
    print(f"  OS: {sys.platform}")
    print(f"  Machine: {platform.machine()}")
    print(f"  Architecture: {platform.architecture()[0]}")
    print(
        f"  Python: {platform.python_implementation()} {platform.python_version()} ({platform.python_compiler()})"
    )
    print(f"  Uname: {platform.uname()}")
    print("Build options:")
    print(f"  C Libraries: {libraries_c}")
    print(f"  C++ Libraries: {libraries_cpp}")
    print(f"  Extra compile args: {extra_compile_args}")
    print(f"  Extra link args: {extra_link_args}")
    print(f"  Include dirs: {include_dirs}")
    print(f"  Library dirs: {library_dirs}")
    print(f"  Extra objects: {extra_objects}")


def main() -> None:
    """Main entry point for building the nphash CFFI extensions.

    Sets up platform-specific build options, reads C source files, and compiles the extensions.
    """

    nphash_dir = Path(__file__).parent.resolve()
    build_dir = nphash_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)



    # Get FFI Builder Instances
    ffibuilder_bytesearch = _get_bytesearch_ffi()
    ffibuilder_blake2b = _get_npblake2b_ffi()
    ffibuilder_npblake3 = _get_npblake3_ffi()
    ffibuilder_sha256 = _get_npsha256_ffi()

    # _print_build_summary(
    #     config.libraries_c,
    #     config.libraries_cpp,
    #     config.extra_compile_args,
    #     config.extra_link_args,
    #     config.include_dirs,
    #     config.library_dirs,
    #     blake3_extra_object_files,
    # )

    if True: # config.tbb_enabled:
        # Patch distutils' build_ext to ensure -std=c++11 is added only for .cpp files during CFFI builds.
        # This is required for BLAKE3/TBB on macOS, and avoids breaking C builds.
        # Using setattr avoids mypy errors and keeps the patch local to this build process.
        setattr(distutils.command.build_ext, "build_ext", _build_ext_with_cpp11)

    # Compile FFI Modules with target and temp dir
    print("Compiling CFFI extensions...")

    dist = Distribution()
    build_ext_cmd = distutils.command.build_ext.build_ext(dist)
    build_ext_cmd.initialize_options()

    # Module names for get_ext_filename must match module_name in set_source
    final_target_path_bytesearch = nphash_dir / build_ext_cmd.get_ext_filename("_bytesearchffi")
    ffibuilder_bytesearch.compile(
        tmpdir=str(build_dir), target=str(final_target_path_bytesearch), verbose=True
    )
    print(f"_bytesearchffi compiled as {final_target_path_bytesearch}")

    final_target_path_blake2b = nphash_dir / build_ext_cmd.get_ext_filename("_npblake2bffi")
    ffibuilder_blake2b.compile(
        tmpdir=str(build_dir), target=str(final_target_path_blake2b), verbose=True
    )
    print(f"_npblake2bffi compiled as {final_target_path_blake2b}")

    final_target_path_npblake3 = nphash_dir / build_ext_cmd.get_ext_filename("_npblake3ffi")
    ffibuilder_npblake3.compile(
        tmpdir=str(build_dir), target=str(final_target_path_npblake3), verbose=True
    )
    print(f"_npblake3ffi compiled as {final_target_path_npblake3}")

    final_target_path_sha256 = nphash_dir / build_ext_cmd.get_ext_filename("_npsha256ffi")
    ffibuilder_sha256.compile(
        tmpdir=str(build_dir), target=str(final_target_path_sha256), verbose=True
    )
    print(f"_npsha256ffi compiled as {final_target_path_sha256}")

    print("All CFFI extensions compiled successfully.")


if __name__ == "__main__":
    main()
