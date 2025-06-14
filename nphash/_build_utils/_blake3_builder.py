"""Build utilities specific to the BLAKE3 cryptographic hash function.

This module provides functions for managing BLAKE3 C source files and for
detecting and compiling its various SIMD (Single Instruction, Multiple Data)
and assembly language implementations. It handles cloning the official BLAKE3
repository if sources are not found, detecting compiler support for different
SIMD instruction sets (SSE2, SSE4.1, AVX2, AVX512, NEON), and compiling
the relevant C or assembly files into object files. These object files can
then be linked into the main CFFI extension for BLAKE3.
"""

import shutil
import subprocess
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Tuple

from nphash._build_utils._build_config import _BuildConfig, _supports_flag

__all__: list[str] = []

# SimdFeature namedtuple to hold SIMD feature flags and source file names
SimdFeature = namedtuple("SimdFeature", ["flags", "filename"])


def _ensure_blake3_sources(blake3_dir: Path, version: str) -> None:
    """Ensure BLAKE3 sources are present in blake3_dir by cloning the repo if missing.

    Args:
        blake3_dir (Path): Directory where BLAKE3 sources should be located.
        version (str): Version to clone from the BLAKE3 repository.
    """
    if blake3_dir.exists() and any(blake3_dir.iterdir()):
        print(f"BLAKE3 sources already present at {blake3_dir.resolve()}. Skipping clone.")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_url = "https://github.com/BLAKE3-team/BLAKE3.git"
        print(f"Cloning BLAKE3 {version} from {repo_url} to {tmpdir} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", version, repo_url, tmpdir], check=True
        )
        src_dir = Path(tmpdir) / "c"
        blake3_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, blake3_dir / f.name)
        print(f"Copied BLAKE3 C sources to {blake3_dir.resolve()}")


def _detect_blake3_simd_support(
    config: _BuildConfig,
    blake3_specific_compile_args: list[str],
) -> list[SimdFeature]:
    """Detects supported SIMD features for BLAKE3 based on the build configuration.

    This function checks for support of various SIMD instruction sets (SSE2, SSE4.1,
    AVX2, AVX512, NEON) and updates `blake3_specific_compile_args` in-place
    with macros to disable unsupported features (e.g., -DBLAKE3_NO_SSE2).

    Args:
        config (_BuildConfig): The overall build configuration.
        blake3_specific_compile_args (list[str]): List of compile arguments to update with disables/enables.

    Returns:
        list[SimdFeature]: Each SimdFeature contains 'flags' (list of str) and
                           'filename' (str) for a C SIMD implementation source file.
    """
    supported_simd: list[SimdFeature] = []

    def _add_simd_flag(flags: list[str], disable_option: str, src_filename: str) -> bool:
        # For MSVC, treat /arch:SSE2 and /arch:SSE41 as placeholders, not real compiler flags
        if config.is_msvc and (flags == ["/arch:SSE2"] or flags == ["/arch:SSE41"]):
            supported_simd.append(SimdFeature(flags, src_filename))
            return True
        elif all(_supports_flag(config.compiler, flag) for flag in flags if flag):
            # All flags in the list must be supported
            supported_simd.append(SimdFeature(flags, src_filename))
            return True
        else:
            blake3_specific_compile_args.append(disable_option)
            return False

    if config.is_arm and config.platform_system == "Darwin":
        # On Apple Silicon, don't require -mfpu=neon flag support as it is always available
        _add_simd_flag([], "-DBLAKE3_USE_NEON=0", "blake3_neon.c")
        blake3_specific_compile_args.append("-DBLAKE3_USE_NEON=1")
    elif config.is_x86_64:
        if config.is_msvc:
            # SSE2/SSE4.1 enabled by default, use explicit placeholders for filtering
            _add_simd_flag(["/arch:SSE2"], "-DBLAKE3_NO_SSE2", "blake3_sse2.c")
            _add_simd_flag(["/arch:SSE41"], "-DBLAKE3_NO_SSE41", "blake3_sse41.c")
            _add_simd_flag(["/arch:AVX2"], "-DBLAKE3_NO_AVX2", "blake3_avx2.c")
            _add_simd_flag(["/arch:AVX512"], "-DBLAKE3_NO_AVX512", "blake3_avx512.c")
        else:  # Not MSVC (e.g., GCC, Clang)
            _add_simd_flag(["-msse2"], "-DBLAKE3_NO_SSE2", "blake3_sse2.c")
            _add_simd_flag(["-msse4.1"], "-DBLAKE3_NO_SSE41", "blake3_sse41.c")
            _add_simd_flag(["-mavx2"], "-DBLAKE3_NO_AVX2", "blake3_avx2.c")
            if _supports_flag(config.compiler, "-mavx512f"):
                avx512_flags = ["-mavx512f"]
                if _supports_flag(config.compiler, "-mavx512vl"):
                    avx512_flags.append("-mavx512vl")
                supported_simd.append(SimdFeature(avx512_flags, "blake3_avx512.c"))
            else:
                blake3_specific_compile_args.append("-DBLAKE3_NO_AVX512")

    return supported_simd


def _compile_blake3_simd_objects(
    config: _BuildConfig,
    supported_simd: list[SimdFeature],
    blake3_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Compile BLAKE3 SIMD source files to object files and return their paths.

    For MSVC, placeholder flags like /arch:SSE2 and /arch:SSE41 are used for feature tracking and filtering only,
    and must NOT be passed to the compiler as they are not valid MSVC flags. Only AVX2/AVX512 are real flags.

    Args:
        config (_BuildConfig): The overall build configuration.
        supported_simd (list[SimdFeature]): List of SIMD features to compile.
        blake3_dir (Path): Path to the BLAKE3 source directory.
        output_dir (Path): Directory where the compiled object files should be placed.

    Returns:
        list[Path]: List of paths to the compiled object files.
    """
    simd_objects: list[Path] = []

    for simd in supported_simd:
        simd_path = blake3_dir / simd.filename
        if simd_path.exists():
            # Filter out MSVC placeholder flags before passing to the compiler
            filtered_flags = [f for f in simd.flags if f not in {"/arch:SSE2", "/arch:SSE41"}]
            compile_args = config.extra_compile_args + filtered_flags

            if config.is_msvc:
                obj = output_dir / (simd_path.stem + ".obj")
                print(f"Compiling {simd_path} with {compile_args} -> {obj}")
                subprocess.run(
                    [config.compiler, "/c", str(simd_path), *compile_args, f"/Fo{obj.name}"],
                    check=True,
                    cwd=str(output_dir),  # Set current working directory for cl.exe
                )
            else:
                obj = output_dir / (simd_path.stem + ".o")
                print(f"Compiling {simd_path} with {compile_args} -> {obj}")
                subprocess.run(
                    [config.compiler, "-c", str(simd_path), *compile_args, "-o", str(obj)],
                    check=True,
                )
            simd_objects.append(obj)
    return simd_objects


def _detect_and_compile_blake3_asm(
    config: _BuildConfig,
    blake3_dir: Path,
    output_dir: Path,
) -> Tuple[list[Path], set[str]]:
    """Detects available BLAKE3 assembly language source files for the current platform.

    It checks for platform-specific assembly files (e.g., for Windows MSVC/GNU,
    or Unix-like systems for x86-64 SSE/AVX instruction sets), compiles them
    into object files, and returns their paths along with the SIMD flags they implement.

    Args:
        config (_BuildConfig): The overall build configuration.
        blake3_dir (Path): Path to the BLAKE3 source directory.
        output_dir (Path): Directory where the compiled object files should be placed.

    Returns:
        Tuple[list[Path], set[str]]: (List of object file paths,
                                      Set of SIMD flags implemented by these assembly files)
    """
    asm_objects: list[Path] = []
    asm_implemented_flags: set[str] = set()

    def _add_asm_file(asm_path: Path, flag: str) -> None:
        if asm_path.exists():
            obj_path: Path
            if asm_path.suffix.lower() == ".asm" and config.is_msvc:
                obj_path = output_dir / (asm_path.stem + ".obj")
                print(f"Compiling assembly (ml64): {asm_path} -> {obj_path}")
                subprocess.run(
                    ["ml64", "/c", str(asm_path), f"/Fo{obj_path.name}"],
                    check=True,
                    cwd=str(output_dir),
                )
            else:
                obj_path = output_dir / (asm_path.stem + ".o")
                print(f"Compiling assembly: {asm_path} -> {obj_path}")
                subprocess.run(
                    [config.compiler, "-c", str(asm_path), "-o", str(obj_path)],
                    check=True,
                )
            asm_objects.append(obj_path)
            asm_implemented_flags.add(flag)

    # Determine which assembly files to compile based on platform and compiler from 'config'
    if config.platform_system == "Windows":
        if not config.is_msvc:
            _add_asm_file(blake3_dir / "blake3_sse2_x86-64_windows_gnu.S", "-msse2")
            _add_asm_file(blake3_dir / "blake3_sse41_x86-64_windows_gnu.S", "-msse4.1")
            _add_asm_file(blake3_dir / "blake3_avx2_x86-64_windows_gnu.S", "-mavx2")
            _add_asm_file(blake3_dir / "blake3_avx512_x86-64_windows_gnu.S", "-mavx512f")
        else:
            _add_asm_file(blake3_dir / "blake3_sse2_x86-64_windows_msvc.asm", "/arch:SSE2")
            _add_asm_file(blake3_dir / "blake3_sse41_x86-64_windows_msvc.asm", "/arch:SSE41")
            _add_asm_file(blake3_dir / "blake3_avx2_x86-64_windows_msvc.asm", "/arch:AVX2")
            _add_asm_file(blake3_dir / "blake3_avx512_x86-64_windows_msvc.asm", "/arch:AVX512")
    elif config.is_x86_64:
        _add_asm_file(blake3_dir / "blake3_sse2_x86-64_unix.S", "-msse2")
        _add_asm_file(blake3_dir / "blake3_sse41_x86-64_unix.S", "-msse4.1")
        _add_asm_file(blake3_dir / "blake3_avx2_x86-64_unix.S", "-mavx2")
        _add_asm_file(blake3_dir / "blake3_avx512_x86-64_unix.S", "-mavx512f")

    return asm_objects, asm_implemented_flags
