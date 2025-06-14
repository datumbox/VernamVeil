"""Compiler utilities for building CFFI extensions for the nphash library.

This module provides specialised CFFI (C Foreign Function Interface) setup
functions and a custom distutils build_ext command. These components are
designed to correctly compile C and C++ sources, including those requiring
specific compiler flags for features like OpenMP and C++11, particularly
for the BLAKE3 TBB (Threading Building Blocks) integration.
"""

from distutils.command.build_ext import build_ext as _build_ext
from pathlib import Path
from typing import Any, List

from cffi import FFI

from nphash._utils._build_config import _BuildConfig, _supports_flag

__all__: list[str] = []


def _get_c_source(path: Path) -> str:
    """Read and return the contents of a C source file.

    Args:
        path (Path): Path to the C source file.

    Returns:
        str: Contents of the C source file.
    """
    with path.open() as f:
        return f.read()


class _build_ext_with_cpp11(_build_ext):
    """Custom build_ext command that ensures C++11 support for .cpp files.

    This class overrides the default build_ext command to add the -std=c++11 flag
    when compiling C++ source files. This is necessary for compatibility with
    modern C++ features used in the BLAKE3 implementation.
    """

    def build_extensions(self) -> None:
        """Build the C/C++ extensions, ensuring C++11 support for .cpp files.

        This method overrides the default build_extensions method to modify the
        compilation process for C++ files. It adds the -std=c++11 flag to the
        compilation arguments if it is not already present. This is necessary to
        ensure compatibility with C++11 features used in the BLAKE3 implementation.
        """
        # Save original compile method
        orig_compile = self.compiler._compile

        def custom_compile(
            obj: str,
            src: str,
            ext: str,
            cc_args: list[str],
            extra_postargs: list[str],
            pp_opts: list[str],
        ) -> Any:
            """Custom compile function to add -std=c++11 for .cpp files only.

            Args:
                obj (str): Object file to generate.
                src (str): Source file to compile.
                ext (str): Extension of the source file.
                cc_args (list[str]): Compiler arguments.
                extra_postargs (list[str]): Extra compiler arguments.
                pp_opts (list[str]): Preprocessor options.

            Returns:
                The result of the original compile call (usually None, but may be implementation-dependent).
            """
            # If compiling a .cpp file, add -std=c++11 if not present
            extra_postargs_copy = list(extra_postargs)  # Ensure we don't modify the original list
            if src.endswith(".cpp"):
                # Check if a C++ standard is already specified in existing flags
                if all("std=c++" not in arg for arg in extra_postargs_copy):
                    if self.compiler.compiler_type == "msvc":
                        # Use /std:c++14 for MSVC for C++11 features
                        extra_postargs_copy.append("/std:c++14")
                    else:
                        extra_postargs_copy.append("-std=c++11")
            return orig_compile(obj, src, ext, cc_args, extra_postargs_copy, pp_opts)

        self.compiler._compile = custom_compile
        try:
            super().build_extensions()
        finally:
            self.compiler._compile = orig_compile


def _get_bytesearch_ffi(config: _BuildConfig, nphash_dir: Path) -> FFI:
    """Creates and configures the FFI builder for the _bytesearchffi module.

    Args:
        config (_BuildConfig): The build configuration object.
        nphash_dir (Path): Path to the nphash directory containing C sources/headers.

    Returns:
        FFI: The configured CFFI FFI instance for bytesearch.
    """
    ffibuilder = FFI()
    ffibuilder.cdef(
        """
        size_t* find_all(const unsigned char * restrict text, size_t n, const unsigned char * restrict pattern, size_t m, size_t * restrict count_ptr, int allow_overlap);
        void free_indices(size_t *indices_ptr);
        ptrdiff_t find(const unsigned char *text, size_t n, const unsigned char *pattern, size_t m);
        """
    )

    ffibuilder.set_source(
        module_name="_bytesearchffi",
        source="""
            #include "bytesearch.h"
            #include "bmh.h"
        """,
        sources=[str(nphash_dir / "c" / "bytesearch.c")],
        include_dirs=config.include_dirs,
        libraries=config.libraries_c,
        extra_compile_args=config.extra_compile_args
        + [f"-DUSE_MEMMEM={int(not config.bmh_enabled)}"],
        extra_link_args=config.extra_link_args,
        library_dirs=config.library_dirs,
    )
    return ffibuilder


def _get_npblake2b_ffi(config: _BuildConfig, nphash_dir: Path) -> FFI:
    """Creates and configures the FFI builder for the _npblake2bffi module.

    Args:
        config (_BuildConfig): The build configuration object.
        nphash_dir (Path): Path to the nphash directory containing C sources/headers.

    Returns:
        FFI: The configured CFFI FFI instance for npblake2b.
    """
    ffibuilder = FFI()
    ffibuilder.cdef(
        """
        void numpy_blake2b(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out);
        """
    )

    ffibuilder.set_source(
        module_name="_npblake2bffi",
        source=_get_c_source(nphash_dir / "c" / "npblake2b.c"),
        libraries=config.libraries_c,
        extra_compile_args=config.extra_compile_args,
        extra_link_args=config.extra_link_args,
        include_dirs=config.include_dirs,
        library_dirs=config.library_dirs,
    )
    return ffibuilder


def _get_npblake3_ffi(
    config: _BuildConfig,
    blake3_header_dir: Path,
    blake3_c_source_files: List[Path],
    blake3_extra_objects: List[str],
    blake3_specific_defines: List[str],
) -> FFI:
    """Creates and configures the FFI builder for the _npblake3ffi module.

    Args:
        config (_BuildConfig): The build configuration object.
        blake3_header_dir (Path): Path to the BLAKE3 sources directory (for blake3.h).
        blake3_c_source_files (List[Path]): List of absolute paths to BLAKE3 .c/.cpp source files.
        blake3_extra_objects (List[str]): List of absolute paths to precompiled ASM/SIMD objects.
        blake3_specific_defines (List[str]): List of BLAKE3 specific defines (e.g., -DBLAKE3_NO_SSE2).

    Returns:
        FFI: The configured CFFI FFI instance for npblake3.
    """
    ffibuilder = FFI()
    ffibuilder.cdef(
        # No const/restrict qualifiers here, as these files are mixed with C++.
        """
        void numpy_blake3(const uint64_t* arr, size_t n, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size);
        void bytes_multi_chunk_blake3(const uint8_t* const* data_chunks, const size_t* data_lengths, size_t num_chunks, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size);
        """
    )

    # Do NOT specify -std=c99 or -std=c++11. This avoids errors related to C++11 features in C code.
    blake3_compile_args = [
        arg for arg in config.extra_compile_args if not arg.startswith(("-std", "/std"))
    ]

    # Add TBB-specific defines if TBB is enabled
    if config.tbb_enabled:
        blake3_compile_args.extend(
            [
                "-DBLAKE3_USE_TBB",
                "-DTBB_USE_EXCEPTIONS=0",  # As per original build.py
            ]
        )
        # Add -fno-rtti if supported and not using MSVC
        if not config.is_msvc and _supports_flag(config.compiler, "-fno-rtti"):
            blake3_compile_args.append("-fno-rtti")

    # Add SIMD/ASM related defines collected previously (e.g. -DBLAKE3_NO_SSE2)
    for define in blake3_specific_defines:
        if define not in blake3_compile_args:
            blake3_compile_args.append(define)

    # Convert absolute Path objects to strings for CFFI sources list
    c_paths_blake3 = [str(p) for p in blake3_c_source_files]

    ffibuilder.set_source(
        module_name="_npblake3ffi",
        source='#include "npblake3.h"',
        sources=c_paths_blake3,
        libraries=config.libraries_cpp if config.tbb_enabled else config.libraries_c,
        extra_compile_args=blake3_compile_args,
        extra_link_args=config.extra_link_args,
        include_dirs=config.include_dirs + [str(blake3_header_dir)],
        library_dirs=config.library_dirs,
        extra_objects=blake3_extra_objects,
    )
    return ffibuilder


def _get_npsha256_ffi(config: _BuildConfig, nphash_dir: Path) -> FFI:
    """Creates and configures the FFI builder for the _npsha256ffi module.

    Args:
        config (_BuildConfig): The build configuration object.
        nphash_dir (Path): Path to the nphash directory containing C sources/headers.

    Returns:
        FFI: The configured CFFI FFI instance for npsha256.
    """
    ffibuilder = FFI()
    ffibuilder.cdef(
        """
        void numpy_sha256(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out);
        """
    )

    ffibuilder.set_source(
        source=_get_c_source(nphash_dir / "c" / "npsha256.c"),
        module_name="_npsha256ffi",
        libraries=config.libraries_c,
        extra_compile_args=config.extra_compile_args,
        extra_link_args=config.extra_link_args,
        include_dirs=config.include_dirs,
        library_dirs=config.library_dirs,
    )
    return ffibuilder
