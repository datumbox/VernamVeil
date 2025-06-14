"""Configuration and build utilities for nphash extensions."""

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import NamedTuple

__all__: list[str] = []


# _BuildConfig NamedTuple to hold all build configuration details
class _BuildConfig(NamedTuple):
    """Holds all build configuration details.

    Attributes:
        tbb_enabled (bool): Whether TBB is enabled for BLAKE3.
        simd_enabled (bool): Whether SIMD acceleration is enabled for BLAKE3.
        asm_enabled (bool): Whether assembly acceleration is enabled for BLAKE3.
        bmh_enabled (bool): Whether BMH algorithm is enabled for byte search.
        platform_system (str): The operating system platform (e.g., "Linux", "Darwin", "Windows").
        is_x86_64 (bool): Whether the architecture is x86_64.
        is_arm (bool): Whether the architecture is ARM.
        compiler (str): The path to the compiler executable.
        is_msvc (bool): Whether the compiler is Microsoft Visual C++.
        extra_compile_args (list[str]): Extra compiler arguments.
        extra_link_args (list[str]): Extra linker arguments.
        include_dirs (list[str]): List of include directories for CFFI.
        library_dirs (list[str]): List of library directories for CFFI.
        libraries_c (list[str]): C libraries to link against.
        libraries_cpp (list[str]): C++ libraries to link against.
    """

    # Feature toggles
    tbb_enabled: bool
    simd_enabled: bool
    asm_enabled: bool
    bmh_enabled: bool

    # Platform details
    platform_system: str
    is_x86_64: bool
    is_arm: bool
    compiler: str
    is_msvc: bool

    # Build arguments and paths
    extra_compile_args: list[str]
    extra_link_args: list[str]
    include_dirs: list[str]
    library_dirs: list[str]
    libraries_c: list[str]
    libraries_cpp: list[str]


def _detect_compiler() -> str:
    """Detect the compiler being used.

    Returns:
        str: Compiler name.
    """
    if sys.platform == "win32":
        if "gcc" in platform.python_compiler().lower():
            return "gcc"
        else:
            return "cl"  # MSVC
    cc_var = sysconfig.get_config_vars().get("CC")
    if cc_var:
        # Use shlex.split to parse and extract the executable
        cc_cmd = shlex.split(cc_var)
        if cc_cmd:
            return cc_cmd[0]
    return "cc"


def _supports_flag(compiler: str, flag: str) -> bool:
    """Check if the compiler supports a given flag.

    Args:
        compiler (str): Compiler to check.
        flag (str): Compiler flag to test.

    Returns:
        bool: True if the compiler supports the flag, False otherwise.
    """
    compiler_path = shutil.which(compiler)
    if not compiler_path:
        return False  # Compiler not found
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src = tmp_path / "test.c"
        exe = tmp_path / "test.out"
        src.write_text("int main(void) { return 0; }")
        result = subprocess.run(
            [compiler, str(src), flag, "-o", str(exe)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0


def _get_build_config(argv: list[str] | None = None) -> _BuildConfig:
    """Detects OS, architecture, parses environment variables and command-line arguments.

    This function produces a comprehensive build configuration.

    Args:
        argv (list[str], optional): Command line arguments. If None, sys.argv[1:] is used.

    Returns:
        _BuildConfig: A NamedTuple containing all build configuration details.

    Raises:
        RuntimeError: If the platform is unsupported.
    """
    parser = argparse.ArgumentParser(description="Build configuration for nphash extensions.")
    parser.add_argument(
        "--no-tbb", action="store_true", help="Disable TBB (Threading Building Blocks) for BLAKE3"
    )
    parser.add_argument(
        "--no-simd",
        action="store_true",
        help="Disable SIMD C acceleration for BLAKE3 (SSE/AVX/NEON)",
    )
    parser.add_argument(
        "--no-asm",
        action="store_true",
        help="Disable assembly acceleration for BLAKE3 (platform-specific .S/.asm files)",
    )
    parser.add_argument(
        "--no-bmh",
        action="store_true",
        help="Disable BMH algorithm for byte search and use memmem instead.",
    )
    # Use provided argv or sys.argv if no arguments are given
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Environment variables override default (False) or command-line arguments
    on_values = {"1", "true", "yes", "on", "enabled"}
    # Feature toggles from args and environment variables
    tbb_enabled = not (
        args.no_tbb or os.environ.get("NPBLAKE3_NO_TBB", "0").strip().lower() in on_values
    )
    simd_enabled = not (
        args.no_simd or os.environ.get("NPBLAKE3_NO_SIMD", "0").strip().lower() in on_values
    )
    asm_enabled = not (
        args.no_asm or os.environ.get("NPBLAKE3_NO_ASM", "0").strip().lower() in on_values
    )
    bmh_enabled = not (
        args.no_bmh or os.environ.get("BYTESEARCH_NO_BMH", "0").strip().lower() in on_values
    )

    # Platform detection
    platform_system = platform.system()
    machine = platform.machine().lower()
    is_x86_64 = any(plat in machine for plat in {"x86_64", "amd64"})
    is_arm = machine in ("arm64", "aarch64")
    compiler = _detect_compiler()
    is_msvc = platform_system == "Windows" and "gcc" not in compiler.lower()

    # Initialise build parameters
    include_dirs: list[Path] = [(Path(__file__).parent.parent / "c").resolve()]
    library_dirs: list[Path] = []
    libraries_c: list[str] = []
    libraries_cpp: list[str] = []
    compile_args: list[str] = []
    extra_link_args: list[str] = []

    if platform_system == "Linux":
        libraries_c += ["ssl", "crypto", "gomp"]
        libraries_cpp += ["tbb", "stdc++", "gomp"] if tbb_enabled else []
        compile_args += [
            "-std=c99",
            "-fopenmp",
            "-O3",
            "-mtune=native",
            "-march=native",
            "-funroll-loops",
            "-DNDEBUG",
        ]
        extra_link_args += ["-fopenmp"]
    elif platform_system == "Darwin":
        libraries_c += ["ssl", "crypto"]
        libraries_cpp += ["tbb", "c++"] if tbb_enabled else []
        min_version_flag = (
            f"-mmacosx-version-min={os.environ.get('MACOSX_DEPLOYMENT_TARGET', '11.0')}"
        )
        compile_args += [
            "-std=c99",
            "-Xpreprocessor",  # For OpenMP on macOS with Clang
            "-fopenmp",
            "-O3",
            min_version_flag,
            "-DNDEBUG",
        ]
        extra_link_args += ["-lomp", min_version_flag]

        if is_arm:
            compile_args.extend(["-arch", "arm64"])
            extra_link_args.extend(["-arch", "arm64"])
            brew_prefixes = [
                Path(p) for p in ["/opt/homebrew/opt/openssl", "/opt/homebrew/opt/libomp"]
            ]
            if tbb_enabled:
                brew_prefixes.append(Path("/opt/homebrew/opt/tbb"))
        else:  # Intel x86_64
            compile_args.extend(["-arch", "x86_64"])
            extra_link_args.extend(["-arch", "x86_64"])
            brew_prefixes = [Path(p) for p in ["/usr/local/opt/openssl", "/usr/local/opt/libomp"]]
            if tbb_enabled:
                brew_prefixes.append(Path("/usr/local/opt/tbb"))

        for prefix in brew_prefixes:
            if prefix.exists():
                include_dirs.append(prefix / "include")
                library_dirs.append(prefix / "lib")
    elif platform_system == "Windows":
        if "gcc" in compiler.lower() or "g++" in compiler.lower():  # MinGW
            libraries_c += ["libssl", "libcrypto", "gomp"]
            libraries_cpp += ["tbb12", "stdc++", "gomp"] if tbb_enabled else []
            compile_args += [
                "-std=c99",
                "-fopenmp",
                "-O3",
                "-mtune=native",
                "-march=native",
                "-funroll-loops",
                "-DNDEBUG",
            ]
            extra_link_args += ["-fopenmp"]
            # Common MSYS2 MinGW-w64 paths
            for prefix in [
                Path(p)
                for p in [os.getenv("MINGW_PREFIX"), r"C:/msys64/mingw64", r"C:/msys2/mingw64"]
                if p
            ]:
                if prefix.exists():
                    include_dirs.append(prefix / "include")
                    library_dirs.append(prefix / "lib")
                    break
        else:  # MSVC
            libraries_c += ["libssl", "libcrypto"]
            libraries_cpp += ["tbb12"] if tbb_enabled else []
            compile_args += [
                "/openmp",
                "/O2",
                "/DNDEBUG",
                "/EHsc",
            ]
            extra_link_args += []
            # Check some possible dependency install locations
            openssl_base_dirs = [
                Path(p)
                for p in [
                    r"C:/vcpkg/installed/x64-windows",
                    r"C:/Program Files/OpenSSL",
                    r"C:/Program Files/OpenSSL-Win64",
                    r"C:/OpenSSL-Win64",
                ]
                if p
            ]
            for ssl_dir in openssl_base_dirs:
                if ssl_dir.exists():
                    include_dirs.append(ssl_dir / "include")
                    library_dirs.append(ssl_dir / "lib")
                    break

            if tbb_enabled:
                tbb_base_dirs = [
                    Path(p)
                    for p in [
                        r"C:/vcpkg/installed/x64-windows",
                        r"C:/Program Files (x86)/Intel/oneAPI/tbb/latest",
                        r"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/tbb",
                    ]
                    if p
                ]
                for tbb_dir in tbb_base_dirs:
                    if tbb_dir.exists():
                        include_dirs.append(tbb_dir / "include")
                        library_dirs.append(tbb_dir / "lib")
                        break
    else:
        raise RuntimeError(f"Unsupported platform: {platform_system}")

    # Add optional flags if supported (non-MSVC)
    if not is_msvc:
        for flag in [
            "-flto",
            "-fomit-frame-pointer",
            "-ftree-vectorize",
            "-fvisibility=hidden",
            "-Wl,-O1",
            "-Wl,--as-needed",
            # "-D_FORTIFY_SOURCE=2",
            # "-fstack-protector-strong",
        ]:
            if _supports_flag(compiler, flag):
                if flag.startswith("-Wl,"):
                    extra_link_args.append(flag)
                else:
                    compile_args.append(flag)

    # Convert Path objects to absolute string paths for CFFI and consistency
    final_include_dirs = [str(p.resolve()) for p in include_dirs if p.is_dir()]
    final_library_dirs = [str(p.resolve()) for p in library_dirs if p.is_dir()]

    return _BuildConfig(
        tbb_enabled=tbb_enabled,
        simd_enabled=simd_enabled,
        asm_enabled=asm_enabled,
        bmh_enabled=bmh_enabled,
        platform_system=platform_system,
        is_x86_64=is_x86_64,
        is_arm=is_arm,
        compiler=compiler,
        is_msvc=is_msvc,
        extra_compile_args=compile_args,
        extra_link_args=extra_link_args,
        include_dirs=final_include_dirs,
        library_dirs=final_library_dirs,
        libraries_c=libraries_c,
        libraries_cpp=libraries_cpp,
    )
