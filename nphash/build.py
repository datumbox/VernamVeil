"""Build script for the nphash CFFI extension.

This script uses cffi to compile the _npblake2bffi and _npsha256ffi C extensions that provide fast, parallelised
BLAKE2b and SHA-256 based hashing functions for NumPy arrays. The C implementations leverage OpenMP for
multithreading and OpenSSL for cryptographic hashing.

Usage:
    python build.py

This will generate the _npblake2bffi and _npsha256ffi extension modules, which can be imported from Python code.
"""

import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from cffi import FFI

__all__ = [
    "main",
]


def _get_c_source(path: Path) -> str:
    """Read and return the contents of a C source file.

    Args:
        path (Path): Path to the C source file.

    Returns:
        str: Contents of the C source file.
    """
    if not path.exists():
        print(f"Error: C source file '{path}' was not found.")
        sys.exit(1)
    with path.open() as f:
        return f.read()


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
            shlex.split(compiler) + [str(src), flag, "-o", str(exe)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0


def _print_build_summary(
    libraries: list[str],
    extra_compile_args: list[str],
    extra_link_args: list[str],
    include_dirs: list[str],
    library_dirs: list[str],
) -> None:
    """Print a summary of the build configuration.

    Args:
        libraries (list): Libraries to link against.
        extra_compile_args (list): Extra compiler arguments.
        extra_link_args (list): Extra linker arguments.
        include_dirs (list): Include directories.
        library_dirs (list): Library directories.
    """
    print("Build configuration summary:")
    print(f"  Platform: {sys.platform}")
    print(f"  Libraries: {libraries}")
    print(f"  Extra compile args: {extra_compile_args}")
    print(f"  Extra link args: {extra_link_args}")
    print(f"  Include dirs: {include_dirs}")
    print(f"  Library dirs: {library_dirs}")


def main() -> None:
    """Main entry point for building the nphash CFFI extensions.

    Sets up platform-specific build options, reads C source files, and compiles the extensions.

    Raises:
        RuntimeError: If the platform is unsupported.
    """
    # FFI builders
    ffibuilder_blake2b = FFI()
    ffibuilder_blake2b.cdef(
        """
        void numpy_blake2b(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out);
        """
    )

    ffibuilder_sha256 = FFI()
    ffibuilder_sha256.cdef(
        """
        void numpy_sha256(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out);
        """
    )

    # Platform-specific build options
    libraries = []
    extra_compile_args = []
    extra_link_args: list[str] = []
    include_dirs: list[Path] = []
    library_dirs: list[Path] = []

    compiler = _detect_compiler()
    if sys.platform.startswith("linux"):
        libraries = ["ssl", "crypto", "gomp"]
        extra_compile_args = [
            "-std=c99",
            "-fopenmp",
            "-O3",
            "-mtune=native",
            "-march=native",
            "-funroll-loops",
        ]
        extra_link_args = ["-fopenmp"]
    elif sys.platform == "darwin":
        libraries = ["ssl", "crypto"]
        extra_compile_args = ["-std=c99", "-Xpreprocessor", "-fopenmp", "-O3"]
        extra_link_args = ["-lomp"]
        # Add include/library dirs for both OpenSSL and libomp
        for prefix in [
            Path("/opt/homebrew/opt/openssl"),
            Path("/usr/local/opt/openssl"),
            Path("/opt/homebrew/opt/libomp"),
            Path("/usr/local/opt/libomp"),
        ]:
            if prefix.exists():
                include_dirs.append(prefix / "include")
                library_dirs.append(prefix / "lib")
    elif sys.platform == "win32":
        # For MSVC: /openmp, for MinGW: -fopenmp
        if "gcc" in compiler.lower():
            libraries = ["libssl", "libcrypto", "gomp"]
            extra_compile_args = [
                "-std=c99",
                "-fopenmp",
                "-O3",
                "-mtune=native",
                "-march=native",
                "-funroll-loops",
            ]
            extra_link_args = ["-fopenmp"]
        else:
            # MSVC
            libraries = ["libssl", "libcrypto"]
            extra_compile_args = ["/openmp", "-O2"]
        # Check all possible OpenSSL install locations
        for prefix in [
            Path(r"C:\Program Files\OpenSSL"),
            Path(r"C:\Program Files\OpenSSL-Win64"),
            Path(r"C:\Program Files\OpenSSL-Win32"),
        ]:
            if prefix.exists():
                include_dirs.append(prefix / "include")
                library_dirs.append(prefix / "lib")
                break
    else:
        raise RuntimeError("Unsupported platform")

    # Try to add optional flags if supported
    for flag in ["-flto", "-fomit-frame-pointer", "-ftree-vectorize", "-Wl,-O1", "-Wl,--as-needed"]:
        if _supports_flag(compiler, flag):
            if flag.startswith("-Wl,"):
                extra_link_args.append(flag)
            else:
                extra_compile_args.append(flag)

    # Add C source
    parent_dir = Path(__file__).parent
    c_source_blake2b = _get_c_source(parent_dir / "_npblake2b.c")
    c_source_sha256 = _get_c_source(parent_dir / "_npsha256.c")

    # Dependencies
    include_paths = [str(p) for p in include_dirs]
    library_paths = [str(p) for p in library_dirs]

    # Add extension build
    ffibuilder_blake2b.set_source(
        "_npblake2bffi",
        c_source_blake2b,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_paths,
        library_dirs=library_paths,
    )

    ffibuilder_sha256.set_source(
        "_npsha256ffi",
        c_source_sha256,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_paths,
        library_dirs=library_paths,
    )

    _print_build_summary(
        libraries, extra_compile_args, extra_link_args, include_paths, library_paths
    )
    ffibuilder_blake2b.compile(verbose=True)
    ffibuilder_sha256.compile(verbose=True)


if __name__ == "__main__":
    main()
