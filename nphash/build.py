"""Build script for the nphash CFFI extension.

This script uses cffi to compile the _bytesearchffi, _npblake2bffi, _npblake3ffi and _npsha256ffi C extensions that provide fast byte search
and fast, parallelised BLAKE2b, BLAKE3 and SHA-256 based hashing functions for NumPy arrays. The C implementations leverage OpenMP for
multithreading and OpenSSL for cryptographic hashing.

Usage:
    python build.py

This will generate the _bytesearchffi, _npblake2bffi, _npblake3ffi and _npsha256ffi extension modules, which can be imported from Python code.
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
            [compiler, str(src), flag, "-o", str(exe)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0


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
                extra_postargs = extra_postargs + ["-std=c++11"]
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
    is_x86: bool,
    is_arm: bool,
    is_msvc: bool,
) -> List[SimdFeature]:
    """Detect supported SIMD features for BLAKE3 and update compile args as needed.

    Args:
        compiler (str): Compiler executable name.
        blake3_compile_args (list[str]): List of compile arguments to update with disables/enables.
        is_x86 (bool): True if the current architecture is x86/x86_64.
        is_arm (bool): True if the current architecture is ARM64 (arm64/aarch64).
        is_msvc (bool): True if using MSVC compiler.

    Returns:
        List[SimdFeature]: Each SimdFeature contains 'flags' (list of str) and 'filename' (str) for a SIMD file.
    """
    supported_simd: List[SimdFeature] = []

    def _add_simd_flag(flags: List[str], disable_option: str, src_filename: str) -> bool:
        # For MSVC, treat /arch:SSE2 and /arch:SSE41 as placeholders, not real compiler flags
        if is_msvc and (flags == ["/arch:SSE2"] or flags == ["/arch:SSE41"]):
            supported_simd.append(SimdFeature(flags, src_filename))
            return True
        elif all(_supports_flag(compiler, flag) for flag in flags if flag):
            # All flags in the list must be supported
            supported_simd.append(SimdFeature(flags, src_filename))
            return True
        else:
            blake3_compile_args.append(disable_option)
            return False

    if is_arm and sys.platform == "darwin":
        # On Apple Silicon, don't require -mfpu=neon flag support as it is always available
        _add_simd_flag([], "-DBLAKE3_USE_NEON=0", "blake3_neon.c")
        blake3_compile_args.append("-DBLAKE3_USE_NEON=1")
    elif is_x86:
        if is_msvc:
            # SSE2/SSE4.1 enabled by default, use explicit placeholders for filtering
            _add_simd_flag(["/arch:SSE2"], "-DBLAKE3_NO_SSE2", "blake3_sse2.c")
            _add_simd_flag(["/arch:SSE41"], "-DBLAKE3_NO_SSE41", "blake3_sse41.c")
            _add_simd_flag(["/arch:AVX2"], "-DBLAKE3_NO_AVX2", "blake3_avx2.c")
            _add_simd_flag(["/arch:AVX512"], "-DBLAKE3_NO_AVX512", "blake3_avx512.c")
        else:
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
    is_msvc: bool,
) -> list[Path]:
    """Compile BLAKE3 SIMD source files to object files and return their paths.

    For MSVC, placeholder flags like /arch:SSE2 and /arch:SSE41 are used for feature tracking and filtering only,
    and must NOT be passed to the compiler as they are not valid MSVC flags. Only AVX2/AVX512 are real flags.

    Args:
        supported_simd (List[SimdFeature]): List of SimdFeature with 'flags' (list of str) and 'filename' (str).
        blake3_dir (Path): Path to the BLAKE3 source directory.
        extra_compile_args (list[str]): Base compile arguments.
        compiler (str): Compiler executable.
        is_msvc (bool): True if using MSVC compiler.

    Returns:
        list[Path]: List of paths to compiled object files.
    """
    simd_objects = []

    for simd in supported_simd:
        simd_path = blake3_dir / simd.filename
        if simd_path.exists():
            # Filter out MSVC placeholder flags before passing to the compiler
            filtered_flags = [f for f in simd.flags if f not in {"/arch:SSE2", "/arch:SSE41"}]
            compile_args = extra_compile_args + filtered_flags

            if is_msvc:
                obj = simd_path.parent / (simd_path.stem + ".obj")
                print(f"Compiling {simd_path} with {compile_args} -> {obj}")
                subprocess.run(
                    [compiler, "/c", str(simd_path), *compile_args, f"/Fo{obj.name}"],
                    check=True,
                    cwd=str(simd_path.parent),
                )
            else:
                obj = simd_path.parent / (simd_path.stem + ".o")
                print(f"Compiling {simd_path} with {compile_args} -> {obj}")
                subprocess.run(
                    [compiler, "-c", str(simd_path), *compile_args, "-o", str(obj)],
                    check=True,
                )
            simd_objects.append(obj)
    return simd_objects


def _detect_and_compile_blake3_asm(
    blake3_dir: Path,
    compiler: str,
    is_x86: bool,
    is_msvc: bool,
) -> tuple[list[Path], set[str]]:
    """Detect and compile BLAKE3 assembly files for the current platform and compiler.

    Args:
        blake3_dir (Path): Path to the BLAKE3 source directory.
        compiler (str): Compiler executable name.
        is_x86 (bool): True if the current architecture is x86/x86_64.
        is_msvc (bool): True if using MSVC compiler.

    Returns:
        tuple[list[Path], set[str]]: (asm_objects, asm_flags)
            asm_objects: List of compiled object file paths.
            asm_flags: Set of SIMD flags implemented in assembly (e.g. '-msse2', '-msse4.1', ...)
    """
    asm_objects = []
    asm_flags = set()

    def _add_asm_file(asm_path: Path, flag: str) -> None:
        if asm_path.exists():
            suffix = asm_path.suffix.lower()
            if suffix == ".asm" and is_msvc:
                obj_path = asm_path.parent / (asm_path.stem + ".obj")
                print(f"Compiling assembly: {asm_path} -> {obj_path}")
                subprocess.run(
                    ["ml64", "/c", str(asm_path), f"/Fo{obj_path.name}"],
                    check=True,
                    cwd=str(asm_path.parent),
                )
            else:
                obj_path = asm_path.with_suffix(asm_path.suffix + ".o")
                print(f"Compiling assembly: {asm_path} -> {obj_path}")
                subprocess.run([compiler, "-c", str(asm_path), "-o", str(obj_path)], check=True)
            asm_objects.append(obj_path)
            asm_flags.add(flag)

    if sys.platform == "win32":
        if "gcc" in compiler.lower():
            _add_asm_file(blake3_dir / "blake3_sse2_x86-64_windows_gnu.S", "-msse2")
            _add_asm_file(blake3_dir / "blake3_sse41_x86-64_windows_gnu.S", "-msse4.1")
            _add_asm_file(blake3_dir / "blake3_avx2_x86-64_windows_gnu.S", "-mavx2")
            _add_asm_file(blake3_dir / "blake3_avx512_x86-64_windows_gnu.S", "-mavx512f")
        else:
            _add_asm_file(blake3_dir / "blake3_sse2_x86-64_windows_msvc.asm", "/arch:SSE2")
            _add_asm_file(blake3_dir / "blake3_sse41_x86-64_windows_msvc.asm", "/arch:SSE41")
            _add_asm_file(blake3_dir / "blake3_avx2_x86-64_windows_msvc.asm", "/arch:AVX2")
            _add_asm_file(blake3_dir / "blake3_avx512_x86-64_windows_msvc.asm", "/arch:AVX512")
    elif is_x86:
        _add_asm_file(blake3_dir / "blake3_sse2_x86-64_unix.S", "-msse2")
        _add_asm_file(blake3_dir / "blake3_sse41_x86-64_unix.S", "-msse4.1")
        _add_asm_file(blake3_dir / "blake3_avx2_x86-64_unix.S", "-mavx2")
        _add_asm_file(blake3_dir / "blake3_avx512_x86-64_unix.S", "-mavx512f")

    return asm_objects, asm_flags


def main() -> None:
    """Main entry point for building the nphash CFFI extensions.

    Sets up platform-specific build options, reads C source files, and compiles the extensions.

    Raises:
        RuntimeError: If the platform is unsupported.
    """
    # Parse flags and env vars
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    on_values = {"1", "true", "yes", "on", "enabled"}
    no_tbb_env = os.environ.get("NPBLAKE3_NO_TBB", "0").strip().lower() in on_values
    no_simd_env = os.environ.get("NPBLAKE3_NO_SIMD", "0").strip().lower() in on_values
    no_asm_env = os.environ.get("NPBLAKE3_NO_ASM", "0").strip().lower() in on_values
    tbb_enabled = not args.no_tbb and not no_tbb_env
    simd_enabled = not args.no_simd and not no_simd_env
    asm_enabled = not args.no_asm and not no_asm_env

    # FFI builders
    ffibuilder_bytesearch = FFI()
    ffibuilder_bytesearch.cdef(
        """
        size_t* find_all(const unsigned char * restrict text, size_t n, const unsigned char * restrict pattern, size_t m, size_t * restrict count_ptr, int allow_overlap);
        void free_indices(size_t *indices_ptr);
        """
    )

    ffibuilder_blake2b = FFI()
    ffibuilder_blake2b.cdef(
        """
        void numpy_blake2b(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out);
        """
    )

    ffibuilder_blake3 = FFI()
    ffibuilder_blake3.cdef(
        # No const/restrict qualifiers here, as these files are mixed with C++.
        """
        void numpy_blake3(const uint64_t* arr, size_t n, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size);
        void bytes_multi_chunk_blake3(const uint8_t* const* data_chunks, const size_t* data_lengths, size_t num_chunks, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size);
        """
    )

    ffibuilder_sha256 = FFI()
    ffibuilder_sha256.cdef(
        """
        void numpy_sha256(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out);
        """
    )

    # Platform-specific build options
    machine = platform.machine().lower()
    is_x86 = any(plat in machine for plat in {"x86_64", "amd64"})
    is_arm = machine in ("arm64", "aarch64")
    compiler = _detect_compiler()
    is_msvc = sys.platform == "win32" and "gcc" not in compiler.lower()

    include_dirs: list[Path] = []
    library_dirs: list[Path] = []
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
            "-DNDEBUG",
        ]
        extra_link_args = ["-fopenmp"]
    elif sys.platform == "darwin":
        libraries_c = ["ssl", "crypto"]
        libraries_cpp = ["tbb", "c++"]
        min_version_flag = (
            f"-mmacosx-version-min={os.environ.get('MACOSX_DEPLOYMENT_TARGET', '11.0')}"
        )
        extra_compile_args = [
            "-std=c99",
            "-Xpreprocessor",
            "-fopenmp",
            "-O3",
            min_version_flag,
            "-DNDEBUG",
        ]
        extra_link_args = ["-lomp", min_version_flag]
        # Detect architecture and set Homebrew prefix accordingly
        if is_arm:
            # Add -arch arm64 for both compile and link args to ensure correct architecture
            extra_compile_args += ["-arch", "arm64"]
            extra_link_args += ["-arch", "arm64"]
            brew_prefixes = [
                Path("/opt/homebrew/opt/openssl"),
                Path("/opt/homebrew/opt/libomp"),
                Path("/opt/homebrew/opt/tbb"),
            ]
        else:
            brew_prefixes = [
                Path("/usr/local/opt/openssl"),
                Path("/usr/local/opt/libomp"),
                Path("/usr/local/opt/tbb"),
            ]
        for prefix in brew_prefixes:
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
                "-DNDEBUG",
            ]
            extra_link_args = ["-fopenmp"]
            # Add common MSYS2 MinGW-w64 include and lib paths for TBB and OpenSSL
            for prefix in [
                Path(r"C:/msys64/mingw64"),
                Path(r"C:/msys2/mingw64"),
            ]:
                if prefix.exists():
                    include_dirs.append(prefix / "include")
                    library_dirs.append(prefix / "lib")
                    break
        else:
            # MSVC
            libraries_c = ["libssl", "libcrypto"]
            libraries_cpp = ["tbb12"]
            extra_compile_args = ["/openmp", "/O2", "/DNDEBUG"]
            extra_link_args = []
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

    # Try to add optional flags if supported
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
                    extra_compile_args.append(flag)

    # Add BLAKE3 sources
    nphash_dir = Path(__file__).parent
    include_dirs.append(nphash_dir)  # Ensure _npblake3.h is found

    blake3_dir = nphash_dir.parent / "third_party" / "blake3"
    _ensure_blake3_sources(blake3_dir, version="1.8.2")
    include_dirs.append(blake3_dir)

    twoway_dir = nphash_dir.parent / "third_party" / "twoway"
    include_dirs.append(twoway_dir)

    c_paths_blake3 = [os.path.relpath(nphash_dir / "_npblake3.c", nphash_dir)]
    core_c_files = ["blake3.c", "blake3_dispatch.c", "blake3_portable.c"]
    if tbb_enabled:
        core_c_files.append("blake3_tbb.cpp")
    c_paths_blake3 += [
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
        # Add -fno-rtti if supported and not using MSVC
        if not is_msvc and _supports_flag(compiler, "-fno-rtti"):
            blake3_compile_args.append("-fno-rtti")

    # BLAKE3 hardware acceleration detection and compilation
    if asm_enabled:
        extra_objects, asm_flags = _detect_and_compile_blake3_asm(
            blake3_dir, compiler, is_x86, is_msvc
        )
    else:
        extra_objects, asm_flags = [], set()
    if simd_enabled:
        supported_simd = _detect_blake3_simd_support(
            compiler, blake3_compile_args, is_x86, is_arm, is_msvc
        )
        filtered_simd = [simd for simd in supported_simd if not (set(simd.flags) & asm_flags)]
        extra_objects += _compile_blake3_simd_objects(
            filtered_simd, blake3_dir, extra_compile_args, compiler, is_msvc
        )
    elif sys.platform == "darwin":
        blake3_compile_args.append("-DBLAKE3_USE_NEON=0")

    # Add extension build
    include_paths = [str(p) for p in include_dirs]
    library_paths = [str(p) for p in library_dirs]
    object_paths = [str(p) for p in extra_objects]

    ffibuilder_bytesearch.set_source(
        "_bytesearchffi",
        '#include "_bytesearch.h"\n',
        sources=[
            os.path.relpath(nphash_dir / "_bytesearch.c", nphash_dir),
            os.path.relpath(twoway_dir / "_twoway.c", nphash_dir),
        ],
        include_dirs=include_paths,
        libraries=libraries_c,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        library_dirs=library_paths,
    )

    ffibuilder_blake2b.set_source(
        "_npblake2bffi",
        _get_c_source(nphash_dir / "_npblake2b.c"),
        libraries=libraries_c,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_paths,
        library_dirs=library_paths,
    )

    ffibuilder_blake3.set_source(
        "_npblake3ffi",
        '#include "_npblake3.h"\n',
        sources=c_paths_blake3,  # use relative paths to avoid issues with CFFI output structure
        libraries=libraries_cpp if tbb_enabled else libraries_c,
        extra_compile_args=blake3_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_paths,
        library_dirs=library_paths,
        extra_objects=object_paths,
    )

    ffibuilder_sha256.set_source(
        "_npsha256ffi",
        _get_c_source(nphash_dir / "_npsha256.c"),
        libraries=libraries_c,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_paths,
        library_dirs=library_paths,
    )

    _print_build_summary(
        libraries_c,
        libraries_cpp,
        extra_compile_args,
        extra_link_args,
        include_paths,
        library_paths,
        object_paths,
    )

    if tbb_enabled:
        # Patch distutils' build_ext to ensure -std=c++11 is added only for .cpp files during CFFI builds.
        # This is required for BLAKE3/TBB on macOS, and avoids breaking C builds.
        # Using setattr avoids mypy errors and keeps the patch local to this build process.
        setattr(distutils.command.build_ext, "build_ext", _build_ext_with_cpp11)

    ffibuilder_bytesearch.compile(verbose=True)
    ffibuilder_blake2b.compile(verbose=True)
    ffibuilder_blake3.compile(verbose=True)
    ffibuilder_sha256.compile(verbose=True)


if __name__ == "__main__":
    main()
