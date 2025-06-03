"""Build script for the nphash CFFI extension.

This script uses cffi to compile the _npblake2bffi, _npblake3ffi and _npsha256ffi C extensions that provide fast, parallelised
BLAKE2b, BLAKE3 and SHA-256 based hashing functions for NumPy arrays. The C implementations leverage OpenMP for
multithreading and OpenSSL for cryptographic hashing.

Usage:
    python build.py

This will generate the _npblake2bffi, _npblake3ffi and _npsha256ffi extension modules, which can be imported from Python code.
"""

import argparse
import distutils
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections import namedtuple
from distutils.command.build_ext import build_ext as _build_ext
from pathlib import Path
from typing import List

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
    extra_objects: list[str],
) -> None:
    """Print a summary of the build configuration.

    Args:
        libraries (list): Libraries to link against.
        extra_compile_args (list): Extra compiler arguments.
        extra_link_args (list): Extra linker arguments.
        include_dirs (list): Include directories.
        library_dirs (list): Library directories.
        extra_objects (list): Extra object files to link.
    """
    print("Build configuration summary:")
    print(f"  Platform: {sys.platform}")
    print(f"  Libraries: {libraries}")
    print(f"  Extra compile args: {extra_compile_args}")
    print(f"  Extra link args: {extra_link_args}")
    print(f"  Include dirs: {include_dirs}")
    print(f"  Library dirs: {library_dirs}")
    print(f"  Extra objects: {extra_objects}")


def _ensure_blake3_sources(blake3_dir: Path, version: str) -> None:
    """Ensure BLAKE3 sources are present in blake3_dir by cloning the repo if missing.

    Args:
        blake3_dir (Path): Directory where BLAKE3 sources should be located.
        version (str): Version tag to clone from the BLAKE3 repository.
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
        ) -> object:
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
            if src.endswith(".cpp") and "-std=c++11" not in extra_postargs:
                extra_postargs = list(extra_postargs) + ["-std=c++11"]
            return orig_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = custom_compile
        try:
            super().build_extensions()
        finally:
            self.compiler._compile = orig_compile


# SimdFeature namedtuple to hold SIMD feature flags and source file names
SimdFeature = namedtuple("SimdFeature", ["flags", "filename"])


def _detect_blake3_simd_support(
    compiler: str,
    blake3_compile_args: list[str],
) -> List[SimdFeature]:
    """Detect supported SIMD features for BLAKE3 and update compile args as needed.

    Args:
        compiler (str): Compiler executable name.
        blake3_compile_args (list[str]): List of compile arguments to update with disables/enables.

    Returns:
        List[SimdFeature]: Each SimdFeature contains 'flags' (list of str) and 'filename' (str) for a SIMD file.
    """
    supported_simd: List[SimdFeature] = []
    machine = platform.machine().lower()
    is_x86 = any(plat in machine for plat in {"x86", "amd64", "i386", "i686"})
    is_arm = "arm" in machine or "aarch64" in machine

    def _add_simd_flag(flags: List[str], disable_option: str, src_file: str) -> bool:
        # All flags in the list must be supported
        if all(_supports_flag(compiler, flag) for flag in flags if flag):
            supported_simd.append(SimdFeature(flags, src_file))
            return True
        else:
            blake3_compile_args.append(disable_option)
            return False

    if is_arm and sys.platform == "darwin":
        # On Apple Silicon, don't require -mfpu=neon flag support as it is always available
        _add_simd_flag([], "-DBLAKE3_USE_NEON=0", "blake3_neon.c")
        blake3_compile_args.append("-DBLAKE3_USE_NEON=1")
    elif is_x86:
        _add_simd_flag(["-msse2"], "-DBLAKE3_NO_SSE2", "blake3_sse2.c")
        _add_simd_flag(["-msse4.1"], "-DBLAKE3_NO_SSE41", "blake3_sse41.c")
        _add_simd_flag(["-mavx2"], "-DBLAKE3_NO_AVX2", "blake3_avx2.c")
        if _supports_flag(compiler, "-mavx512f"):
            avx512_flags = ["-mavx512f"]
            if _supports_flag(compiler, "-mavx512vl"):
                avx512_flags.append("-mavx512vl")
            supported_simd.append(SimdFeature(avx512_flags, "blake3_avx512.c"))
        else:
            blake3_compile_args.append("-DBLAKE3_NO_AVX512")
    return supported_simd


def _compile_blake3_simd_objects(
    supported_simd: List[SimdFeature],
    blake3_dir: Path,
    extra_compile_args: list[str],
    compiler: str,
) -> list[str]:
    """Compile BLAKE3 SIMD source files to object files and return their paths.

    Args:
        supported_simd (List[SimdFeature]): List of SimdFeature with 'flags' (list of str) and 'filename' (str).
        blake3_dir (Path): Path to the BLAKE3 source directory.
        extra_compile_args (list[str]): Base compile arguments.
        compiler (str): Compiler executable.

    Returns:
        list[str]: List of paths to compiled object files.
    """
    extra_objects = []
    for simd in supported_simd:
        src = blake3_dir / simd.filename
        if src.exists():
            obj = blake3_dir / (simd.filename + ".o")
            compile_args = list(extra_compile_args)
            compile_args += simd.flags
            print(f"Compiling {src} with {compile_args} -> {obj}")
            subprocess.run([compiler, "-c", str(src)] + compile_args + ["-o", str(obj)], check=True)
            extra_objects.append(str(obj))
    return extra_objects


def _detect_and_compile_blake3_asm(blake3_dir: Path, compiler: str) -> tuple[list[str], set[str]]:
    """Detect and compile BLAKE3 assembly files for the current platform and compiler.

    Args:
        blake3_dir (Path): Path to the BLAKE3 source directory.
        compiler (str): Compiler executable name.

    Returns:
        tuple[list[str], set[str]]: (asm_objects, asm_flags)
            asm_objects: List of compiled object file paths.
            asm_flags: Set of SIMD flags implemented in assembly (e.g. '-msse2', '-msse4.1', ...)
    """
    asm_objects: list[str] = []
    asm_flags: set[str] = set()
    machine = platform.machine().lower()
    is_x86 = any(plat in machine for plat in {"x86", "amd64", "i386", "i686"})
    is_arm = "arm" in machine or "aarch64" in machine

    def _add_asm_file(asm_path: Path, flag: str) -> None:
        if asm_path.exists():
            if (
                sys.platform == "win32"
                and asm_path.suffix.lower() == ".asm"
                and "gcc" not in compiler.lower()
            ):
                obj_path = asm_path.parent / (asm_path.stem + ".obj")
            else:
                obj_path = asm_path.with_suffix(asm_path.suffix + ".o")
            print(f"Compiling assembly: {asm_path} -> {obj_path}")
            suffix = asm_path.suffix.lower()
            if suffix == ".s":
                subprocess.run([compiler, "-c", str(asm_path), "-o", str(obj_path)], check=True)
            elif suffix == ".asm":
                if sys.platform == "win32" and "gcc" not in compiler.lower():
                    subprocess.run(
                        ["ml64", "/c", str(asm_path), f"/Fo{obj_path.name}"],
                        check=True,
                        cwd=str(asm_path.parent),
                    )
                else:
                    subprocess.run([compiler, "-c", str(asm_path), "-o", str(obj_path)], check=True)
            else:
                raise RuntimeError(f"Unknown assembly file type: {asm_path}")
            asm_objects.append(str(obj_path))
            asm_flags.add(flag)

    if sys.platform == "win32":
        if "gcc" in compiler.lower():
            _add_asm_file(blake3_dir / "blake3_sse2_x86-64_windows_gnu.S", "-msse2")
            _add_asm_file(blake3_dir / "blake3_sse41_x86-64_windows_gnu.S", "-msse4.1")
            _add_asm_file(blake3_dir / "blake3_avx2_x86-64_windows_gnu.S", "-mavx2")
            _add_asm_file(blake3_dir / "blake3_avx512_x86-64_windows_gnu.S", "-mavx512f")
        else:
            _add_asm_file(blake3_dir / "blake3_sse2_x86-64_windows_msvc.asm", "-msse2")
            _add_asm_file(blake3_dir / "blake3_sse41_x86-64_windows_msvc.asm", "-msse4.1")
            _add_asm_file(blake3_dir / "blake3_avx2_x86-64_windows_msvc.asm", "-mavx2")
            _add_asm_file(blake3_dir / "blake3_avx512_x86-64_windows_msvc.asm", "-mavx512f")
    elif sys.platform.startswith("linux") or sys.platform == "darwin":
        if is_x86:
            _add_asm_file(blake3_dir / "blake3_sse2_x86-64_unix.S", "-msse2")
            _add_asm_file(blake3_dir / "blake3_sse41_x86-64_unix.S", "-msse4.1")
            _add_asm_file(blake3_dir / "blake3_avx2_x86-64_unix.S", "-mavx2")
            _add_asm_file(blake3_dir / "blake3_avx512_x86-64_unix.S", "-mavx512f")
        elif is_arm:
            pass
    return asm_objects, asm_flags


def main() -> None:
    """Main entry point for building the nphash CFFI extensions.

    Sets up platform-specific build options, reads C source files, and compiles the extensions.

    Raises:
        RuntimeError: If the platform is unsupported.
    """
    # Parse --no-tbb flag and env var
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-tbb", action="store_true", help="Disable TBB (Threading Building Blocks) for BLAKE3"
    )
    args = parser.parse_args()
    no_tbb_env = os.environ.get("NPBLAKE3_NO_TBB", "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
    }
    tbb_enabled = not args.no_tbb and not no_tbb_env

    # FFI builders
    ffibuilder_blake2b = FFI()
    ffibuilder_blake2b.cdef(
        """
        void numpy_blake2b(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out);
        """
    )

    ffibuilder_blake3 = FFI()
    ffibuilder_blake3.cdef(
        """
        void numpy_blake3(const uint64_t* arr, size_t n, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size);
        void bytes_blake3(const uint8_t* data, size_t datalen, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size);
        """
    )

    ffibuilder_sha256 = FFI()
    ffibuilder_sha256.cdef(
        """
        void numpy_sha256(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out);
        """
    )

    # Platform-specific build options
    libraries_c = []
    libraries_cpp = []
    extra_compile_args = []
    extra_link_args: list[str] = []
    include_dirs: list[Path] = []
    library_dirs: list[Path] = []

    compiler = _detect_compiler()
    if sys.platform.startswith("linux"):
        libraries_c = ["ssl", "crypto", "gomp"]
        libraries_cpp = ["tbb", "stdc++", "gomp"]
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
        libraries_c = ["ssl", "crypto"]
        libraries_cpp = ["tbb", "c++"]
        extra_compile_args = ["-std=c99", "-Xpreprocessor", "-fopenmp", "-O3"]
        extra_link_args = ["-lomp"]
        # Add include/library dirs for both OpenSSL and libomp
        for prefix in [
            Path("/opt/homebrew/opt/openssl"),
            Path("/usr/local/opt/openssl"),
            Path("/opt/homebrew/opt/libomp"),
            Path("/usr/local/opt/libomp"),
            Path("/opt/homebrew/opt/tbb"),
            Path("/usr/local/opt/tbb"),
        ]:
            if prefix.exists():
                include_dirs.append(prefix / "include")
                library_dirs.append(prefix / "lib")
    elif sys.platform == "win32":
        # For MSVC: /openmp, for MinGW: -fopenmp
        if "gcc" in compiler.lower():
            libraries_c = ["libssl", "libcrypto", "gomp"]
            libraries_cpp = ["tbb12", "stdc++", "gomp"]
            extra_compile_args = [
                "-std=c99",
                "-fopenmp",
                "-O3",
                "-mtune=native",
                "-march=native",
                "-funroll-loops",
            ]
            extra_link_args = ["-fopenmp"]
            # Add common MSYS2 MinGW-w64 include and lib paths for TBB and OpenSSL
            for prefix in [
                Path(r"C:/msys64/mingw64"),
                Path(r"C:/msys2/mingw64"),
            ]:
                include_dir = prefix / "include"
                lib_dir = prefix / "lib"
                if include_dir.exists() and lib_dir.exists():
                    include_dirs.append(include_dir)
                    library_dirs.append(lib_dir)
                    break
        else:
            # MSVC
            libraries_c = ["libssl", "libcrypto"]
            libraries_cpp = ["tbb12"]
            extra_compile_args = ["/openmp", "-O2"]
        # Check all possible Dependecy install locations
        for prefix in [
            Path(r"C:\Program Files\OpenSSL"),
            Path(r"C:\Program Files\OpenSSL-Win64"),
            Path(r"C:\Program Files\OpenSSL-Win32"),
            Path(r"C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\tbb"),
        ]:
            if prefix.exists():
                include_dirs.append(prefix / "include")
                library_dirs.append(prefix / "lib")
                break
    else:
        raise RuntimeError("Unsupported platform")

    # Add nphash directory to include_dirs so _npblake3.h is found
    nphash_dir = Path(__file__).parent
    if nphash_dir not in include_dirs:
        include_dirs.append(nphash_dir)

    # Add blake3_dir to include_dirs
    blake3_dir = nphash_dir.parent / "third_party" / "blake3"
    _ensure_blake3_sources(blake3_dir, version="1.8.2")
    if blake3_dir.exists():
        include_dirs.append(blake3_dir)

    # Try to add optional flags if supported
    for flag in ["-flto", "-fomit-frame-pointer", "-ftree-vectorize", "-Wl,-O1", "-Wl,--as-needed"]:
        if _supports_flag(compiler, flag):
            if flag.startswith("-Wl,"):
                extra_link_args.append(flag)
            else:
                extra_compile_args.append(flag)

    # Dependencies
    include_paths = [str(p) for p in include_dirs]
    library_paths = [str(p) for p in library_dirs]

    # Add C source
    c_source_blake2b = _get_c_source(nphash_dir / "_npblake2b.c")
    c_source_sha256 = _get_c_source(nphash_dir / "_npsha256.c")

    c_sources_blake3 = [os.path.relpath(nphash_dir / "_npblake3.c", nphash_dir)]
    core_c_files = ["blake3.c", "blake3_dispatch.c", "blake3_portable.c"]
    if tbb_enabled:
        core_c_files.append("blake3_tbb.cpp")
    c_sources_blake3 += [
        os.path.relpath(blake3_dir / fname, nphash_dir)
        for fname in core_c_files
        if (blake3_dir / fname).exists()
    ]

    # Do NOT specify -std=c99 or -std=c++11. This avoids errors related to C++11 features in C code.
    blake3_compile_args = [arg for arg in extra_compile_args if not arg.startswith("-std=")]
    if tbb_enabled:
        blake3_compile_args.extend(
            [
                "-DBLAKE3_USE_TBB",
                "-DTBB_USE_EXCEPTIONS=0",
            ]
        )

    # BLAKE3 hardware acceleration detection and compilation
    extra_objects, asm_flags = _detect_and_compile_blake3_asm(blake3_dir, compiler)
    supported_simd = _detect_blake3_simd_support(compiler, blake3_compile_args)
    filtered_simd = [simd for simd in supported_simd if not (set(simd.flags) & asm_flags)]
    extra_objects += _compile_blake3_simd_objects(
        supported_simd=filtered_simd,
        blake3_dir=blake3_dir,
        extra_compile_args=extra_compile_args,
        compiler=compiler,
    )

    # Add extension build
    ffibuilder_blake2b.set_source(
        "_npblake2bffi",
        c_source_blake2b,
        libraries=libraries_c,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_paths,
        library_dirs=library_paths,
    )

    ffibuilder_blake3.set_source(
        "_npblake3ffi",
        '#include "_npblake3.h"\n',
        sources=c_sources_blake3,  # use relative paths to avoid issues with CFFI output structure
        libraries=libraries_cpp if tbb_enabled else libraries_c,
        extra_compile_args=blake3_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_paths,
        library_dirs=library_paths,
        extra_objects=extra_objects,
    )

    ffibuilder_sha256.set_source(
        "_npsha256ffi",
        c_source_sha256,
        libraries=libraries_c,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_paths,
        library_dirs=library_paths,
    )

    _print_build_summary(
        libraries_c + (libraries_cpp if tbb_enabled else []),
        extra_compile_args,
        extra_link_args,
        include_paths,
        library_paths,
        extra_objects,
    )

    if tbb_enabled:
        # Patch distutils' build_ext to ensure -std=c++11 is added only for .cpp files during CFFI builds.
        # This is required for BLAKE3/TBB on macOS, and avoids breaking C builds.
        # Using setattr avoids mypy errors and keeps the patch local to this build process.
        setattr(distutils.command.build_ext, "build_ext", _build_ext_with_cpp11)

    ffibuilder_blake2b.compile(verbose=True)
    ffibuilder_blake3.compile(verbose=True)
    ffibuilder_sha256.compile(verbose=True)


if __name__ == "__main__":
    main()
