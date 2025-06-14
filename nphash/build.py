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

from nphash._build_utils._blake3_builder import (
    _compile_blake3_simd_objects,
    _detect_and_compile_blake3_asm,
    _detect_blake3_simd_support,
    _ensure_blake3_sources,
)
from nphash._build_utils._config_builder import _get_build_config
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
    # Get the comprehensive build configuration
    config = _get_build_config(sys.argv[1:])

    nphash_dir = Path(__file__).parent.resolve()
    blake3_source_dir = nphash_dir.parent / "third_party" / "blake3"

    _ensure_blake3_sources(blake3_source_dir, version="1.8.2")

    # BLAKE3 C/C++ source files for FFI
    blake3_c_source_files: list[Path] = [nphash_dir / "c" / "npblake3.c"]
    core_blake3_c_files_names = ["blake3.c", "blake3_dispatch.c", "blake3_portable.c"]
    if config.tbb_enabled:
        core_blake3_c_files_names.append("blake3_tbb.cpp")

    for fname in core_blake3_c_files_names:
        blake3_c_source_files.append(blake3_source_dir / fname)

    blake3_extra_object_files: list[str] = []
    blake3_specific_compile_args: list[str] = []
    asm_implemented_flags: set[str] = set()

    # Define a temporary directory for all build artifacts
    with tempfile.TemporaryDirectory(prefix="nphash_build_") as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)
        temp_dir_path.mkdir(parents=True, exist_ok=True)

        # BLAKE3 hardware acceleration detection and compilation
        if config.asm_enabled:
            asm_object_files, asm_implemented_flags = _detect_and_compile_blake3_asm(
                config, blake3_source_dir, temp_dir_path
            )
            blake3_extra_object_files.extend(str(p) for p in asm_object_files)

        if config.simd_enabled:
            supported_simd_features = _detect_blake3_simd_support(
                config, blake3_specific_compile_args
            )
            simd_features_to_compile = [
                feat
                for feat in supported_simd_features
                if not (set(feat.flags) & asm_implemented_flags)
            ]
            simd_object_files = _compile_blake3_simd_objects(
                config,
                simd_features_to_compile,
                blake3_source_dir,
                temp_dir_path,
            )
            blake3_extra_object_files.extend(str(p) for p in simd_object_files)
        elif config.platform_system == "Darwin":
            blake3_specific_compile_args.append("-DBLAKE3_USE_NEON=0")

        # Get FFI Builder Instances
        ffibuilder_bytesearch = _get_bytesearch_ffi(config, nphash_dir)
        ffibuilder_blake2b = _get_npblake2b_ffi(config, nphash_dir)
        ffibuilder_npblake3 = _get_npblake3_ffi(
            config,
            blake3_source_dir,
            blake3_c_source_files,
            blake3_extra_object_files,
            blake3_specific_compile_args,
        )
        ffibuilder_sha256 = _get_npsha256_ffi(config, nphash_dir)

        _print_build_summary(
            config.libraries_c,
            config.libraries_cpp,
            config.extra_compile_args,
            config.extra_link_args,
            config.include_dirs,
            config.library_dirs,
            blake3_extra_object_files,
        )

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

        # Module names for get_ext_filename must match module_name in set_source
        final_target_path_bytesearch = nphash_dir / build_ext_cmd.get_ext_filename("_bytesearchffi")
        ffibuilder_bytesearch.compile(
            tmpdir=str(temp_dir_path), target=str(final_target_path_bytesearch), verbose=True
        )
        print(f"_bytesearchffi compiled as {final_target_path_bytesearch}")

        final_target_path_blake2b = nphash_dir / build_ext_cmd.get_ext_filename("_npblake2bffi")
        ffibuilder_blake2b.compile(
            tmpdir=str(temp_dir_path), target=str(final_target_path_blake2b), verbose=True
        )
        print(f"_npblake2bffi compiled as {final_target_path_blake2b}")

        final_target_path_npblake3 = nphash_dir / build_ext_cmd.get_ext_filename("_npblake3ffi")
        ffibuilder_npblake3.compile(
            tmpdir=str(temp_dir_path), target=str(final_target_path_npblake3), verbose=True
        )
        print(f"_npblake3ffi compiled as {final_target_path_npblake3}")

        final_target_path_sha256 = nphash_dir / build_ext_cmd.get_ext_filename("_npsha256ffi")
        ffibuilder_sha256.compile(
            tmpdir=str(temp_dir_path), target=str(final_target_path_sha256), verbose=True
        )
        print(f"_npsha256ffi compiled as {final_target_path_sha256}")

        print("All CFFI extensions compiled successfully.")


if __name__ == "__main__":
    main()
