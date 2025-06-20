"""Compiler utilities for building CFFI extensions for the nphash library.

This module provides specialised CFFI (C Foreign Function Interface) setup
functions and a custom setuptools build_ext command. These components are
designed to correctly compile C and C++ sources, including those requiring
specific compiler flags for features like OpenMP and C++11, particularly
for the BLAKE3 TBB (Threading Building Blocks) integration.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from cffi import FFI
from setuptools.command.build_ext import build_ext

from nphash._build_utils._blake3_builder import (
    _compile_blake3_simd_objects,
    _detect_and_compile_blake3_asm,
    _detect_blake3_simd_support,
    _ensure_blake3_sources,
)
from nphash._build_utils._config_builder import _BuildConfig, _supports_flag

__all__: list[str] = []


if TYPE_CHECKING:
    # Fixing mypy error by defining a dummy _BuildExtBase class

    class _BuildExtBase:
        compiler: Any

        def build_extensions(self) -> None:
            pass

else:
    _BuildExtBase = build_ext


def _set_source_and_print_details(
    ffibuilder: FFI, module_name: str, source: str, **kwargs: Any
) -> None:
    """Print a summary of the build configuration and set the source for the CFFI module.

    Args:
        ffibuilder (FFI): The FFI instance to configure.
        module_name (str): Name of the CFFI module.
        source (str): Inline C source code.
        **kwargs: Additional keyword arguments for FFI configuration.
    """
    print(f"\n--- Configuring CFFI module: {module_name} via set_source ---")
    for key, value in kwargs.items():
        print(f"  {key}: {value!r}")
    print("-----------------------------------------------------------")

    ffibuilder.set_source(module_name=module_name, source=source, **kwargs)


def _get_c_source(path: Path) -> str:
    """Read and return the contents of a C source file.

    Args:
        path (Path): Path to the C source file.

    Returns:
        str: Contents of the C source file.
    """
    with path.open() as f:
        return f.read()


class _build_ext_with_cpp11(_BuildExtBase):
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


def _get_bytesearch_ffi(config: _BuildConfig, build_dir: Path) -> FFI:
    """Creates and configures the FFI builder for the _bytesearchffi module.

    Args:
        config (_BuildConfig): The build configuration.
        build_dir (Path): The directory to place the compiled files

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

    _set_source_and_print_details(
        ffibuilder,
        module_name="_bytesearchffi",
        source=_get_c_source(config.nphash_dir / "c" / "bytesearch.c"),
        include_dirs=config.include_dirs,
        libraries=config.libraries_c,
        extra_compile_args=config.extra_compile_args
        + [f"-DUSE_MEMMEM={int(not config.bmh_enabled)}"],
        extra_link_args=config.extra_link_args,
        library_dirs=config.library_dirs,
    )

    return ffibuilder


def _get_npblake2b_ffi(config: _BuildConfig, build_dir: Path) -> FFI:
    """Creates and configures the FFI builder for the _npblake2bffi module.

    Args:
        config (_BuildConfig): The build configuration.
        build_dir (Path): The directory to place the compiled files

    Returns:
        FFI: The configured CFFI FFI instance for npblake2b.
    """
    ffibuilder = FFI()
    ffibuilder.cdef(
        """
        void numpy_blake2b(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out);
        """
    )

    _set_source_and_print_details(
        ffibuilder,
        module_name="_npblake2bffi",
        source=_get_c_source(config.nphash_dir / "c" / "npblake2b.c"),
        libraries=config.libraries_c,
        extra_compile_args=config.extra_compile_args,
        extra_link_args=config.extra_link_args,
        include_dirs=config.include_dirs,
        library_dirs=config.library_dirs,
    )

    return ffibuilder


def _get_npblake3_ffi(config: _BuildConfig, build_dir: Path) -> FFI:
    """Creates and configures the FFI builder for the _npblake3ffi module.

    Args:
        config (_BuildConfig): The build configuration.
        build_dir (Path): The directory to place the compiled files

    Returns:
        FFI: The configured CFFI FFI instance for npblake3.
    """
    nphash_dir = config.nphash_dir
    blake3_source_dir = nphash_dir.parent / "third_party" / "blake3"
    _ensure_blake3_sources(blake3_source_dir, version="1.8.2")

    # BLAKE3 C/C++ source files for FFI
    blake3_c_source_files: list[Path] = []
    core_blake3_c_files_names = ["blake3.c", "blake3_dispatch.c", "blake3_portable.c"]
    if config.tbb_enabled:
        core_blake3_c_files_names.append("blake3_tbb.cpp")

    for fname in core_blake3_c_files_names:
        blake3_c_source_files.append(blake3_source_dir / fname)

    blake3_extra_objects: list[str] = []
    blake3_specific_defines: list[str] = []
    asm_implemented_flags: set[str] = set()

    # BLAKE3 hardware acceleration detection and compilation
    if config.asm_enabled:
        asm_object_files, asm_implemented_flags = _detect_and_compile_blake3_asm(
            config, blake3_source_dir, build_dir
        )
        blake3_extra_objects.extend(str(p) for p in asm_object_files)

    if config.simd_enabled:
        supported_simd_features = _detect_blake3_simd_support(config, blake3_specific_defines)
        simd_features_to_compile = [
            feat
            for feat in supported_simd_features
            if not (set(feat.flags) & asm_implemented_flags)
        ]
        simd_object_files = _compile_blake3_simd_objects(
            config,
            simd_features_to_compile,
            blake3_source_dir,
            build_dir,
        )
        blake3_extra_objects.extend(str(p) for p in simd_object_files)
    elif config.platform_system == "Darwin":
        blake3_specific_defines.append("-DBLAKE3_USE_NEON=0")

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

    _set_source_and_print_details(
        ffibuilder,
        module_name="_npblake3ffi",
        source=_get_c_source(nphash_dir / "c" / "npblake3.c"),
        sources=blake3_c_source_files,
        libraries=config.libraries_cpp if config.tbb_enabled else config.libraries_c,
        extra_compile_args=blake3_compile_args,
        extra_link_args=config.extra_link_args,
        include_dirs=config.include_dirs + [str(blake3_source_dir)],
        library_dirs=config.library_dirs,
        extra_objects=blake3_extra_objects,
    )

    return ffibuilder


def _get_npsha256_ffi(config: _BuildConfig, build_dir: Path) -> FFI:
    """Creates and configures the FFI builder for the _npsha256ffi module.

    Args:
        config (_BuildConfig): The build configuration.
        build_dir (Path): The directory to place the compiled files

    Returns:
        FFI: The configured CFFI FFI instance for npsha256.
    """
    ffibuilder = FFI()
    ffibuilder.cdef(
        """
        void numpy_sha256(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out);
        """
    )

    _set_source_and_print_details(
        ffibuilder,
        module_name="_npsha256ffi",
        source=_get_c_source(config.nphash_dir / "c" / "npsha256.c"),
        libraries=config.libraries_c,
        extra_compile_args=config.extra_compile_args,
        extra_link_args=config.extra_link_args,
        include_dirs=config.include_dirs,
        library_dirs=config.library_dirs,
    )

    return ffibuilder
