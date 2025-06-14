"""VernamVeil CLI utility.

This module provides a command-line interface for encrypting and decrypting files using the VernamVeil cypher.
It supports custom key stream functions, seed management, and various encryption parameters.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import IO, Callable, cast

from vernamveil import __version__
from vernamveil._fx_utils import OTPFX, check_fx_sanity, generate_default_fx, load_fx_from_file
from vernamveil._imports import _HAS_C_MODULE, _HAS_NUMPY
from vernamveil._vernamveil import VernamVeil

__all__ = ["main"]


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Add common CLI arguments for both encode and decode subcommands.

    Args:
        p (argparse.ArgumentParser): The argument parser to which the arguments will be added.
    """
    p.add_argument(
        "--infile",
        type=str,
        default="-",
        help="Input file path (default: -, meaning stdin). Use - or omit for stdin. Always binary mode.",
    )
    p.add_argument(
        "--outfile",
        type=str,
        default="-",
        help="Output file path (default: -, meaning stdout). Use - or omit for stdout. Always binary mode.",
    )
    p.add_argument("--fx-file", type=Path, help="Path to Python file containing the fx function.")
    p.add_argument("--seed-file", type=Path, help="Path to file containing the seed (bytes).")
    p.add_argument(
        "--buffer-size",
        type=int,
        default=1024 * 1024,
        help="Buffer size in bytes for reading blocks (default: 1048576, i.e., 1MB).",
    )
    p.add_argument(
        "--chunk-size", type=int, default=8192, help="Chunk size for VernamVeil (default: 8192)."
    )
    p.add_argument(
        "--delimiter-size",
        type=int,
        default=16,
        help="Delimiter size for VernamVeil (default: 16).",
    )
    p.add_argument(
        "--padding-range",
        type=int,
        nargs=2,
        default=(10, 20),
        metavar=("MIN", "MAX"),
        help="Padding range as two integers (default: 10 20).",
    )
    p.add_argument(
        "--decoy-ratio",
        type=float,
        default=0.05,
        help="Decoy ratio for VernamVeil (default: 0.05).",
    )
    p.add_argument(
        "--siv-seed-initialisation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Synthetic IV seed initialisation (default: True).",
    )
    p.add_argument(
        "--auth-encrypt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable authenticated encryption (default: True).",
    )
    p.add_argument(
        "--hash-name",
        type=str,
        default="blake2b",
        choices=["blake2b", "blake3", "sha256"] if _HAS_C_MODULE else ["blake2b", "sha256"],
        help="Hashing algorithm for the cypher and fx generation (default: blake2b).",
    )
    p.add_argument(
        "--verbosity",
        choices=["info", "warning", "error", "none"],
        default="warning",
        help="Verbosity level: info, warning (default), error, none. "
        "Info shows all messages, including progress information.",
    )


def _vprint(msg: str, level: str, verbosity: str, end: str = "\n", flush: bool = False) -> None:
    """Conditionally print a message to stderr based on the verbosity level.

    Args:
        msg (str): The message to print.
        level (str): The message level: 'info', 'warning', or 'error'.
        verbosity (str): The verbosity setting: 'info', 'warning', 'error', or 'none'.
        end (str): The string appended after the message (default: newline).
        flush (bool): Whether to forcibly flush the output buffer (default: False).
    """
    levels: dict[str, int] = {"info": 0, "warning": 1, "error": 2, "none": 3}
    msg_level = levels[level]
    user_level = levels[verbosity]
    if msg_level >= user_level:
        print(msg, file=sys.stderr, end=end, flush=flush)


def _open_file(file: str | None, mode: str, std_stream: IO[bytes] | object) -> IO[bytes]:
    """Opens a file in the specified binary mode or returns the provided standard stream if file is '-' or None.

    Args:
        file (str, optional): Path to the file or '-' for the standard stream.
        mode (str): File open mode, e.g., 'rb' or 'wb'.
        std_stream (object): Standard stream to use if file is '-' or None.

    Returns:
        IO[bytes]: A file-like object opened for binary reading or writing.
    """
    if file == "-" or file is None:
        # Return the binary buffer of a stream if available, else the stream itself.
        return getattr(std_stream, "buffer", cast(IO[bytes], std_stream))
    else:
        return open(file, mode)


def main(args: list[str] | None = None) -> None:
    """Entry point for the VernamVeil CLI utility.

    Parses arguments, handles key management, and dispatches encode/decode.

    Args:
        args (list or None): Optional list of arguments to parse instead of sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="VernamVeil CLI utility for encryption and decryption."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Operation mode: encode or decode."
    )

    # Encode subcommand
    enc = subparsers.add_parser("encode", help="Encrypt a file.")
    _add_common_args(enc)
    enc.add_argument(
        "--vectorise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Vectorise fx when generating a new one (ignored if --fx-file is used).",
    )
    enc.add_argument(
        "--check-sanity",
        action="store_true",
        help="Check the loaded/generated fx and seed for appropriateness.",
    )

    # Decode subcommand
    dec = subparsers.add_parser("decode", help="Decrypt a file.")
    _add_common_args(dec)

    parsed_args = parser.parse_args(args)
    verbosity = parsed_args.verbosity

    infile = parsed_args.infile
    outfile = parsed_args.outfile
    fx_file = parsed_args.fx_file
    seed_file = parsed_args.seed_file

    # Check if output file exists
    if outfile not in (None, "-"):
        out_path = Path(outfile)
        if out_path.exists():
            _vprint(
                f"Error: {out_path.resolve()} already exists. Refusing to overwrite. "
                "Use a different output file or remove the existing file.",
                "error",
                verbosity,
            )
            sys.exit(1)
    elif sys.stdout.isatty():
        _vprint(
            "Warning: Writing binary data to a terminal may corrupt your session. "
            "Redirect output to a file or use --outfile.",
            "warning",
            verbosity,
        )

    # Handle fx function
    if fx_file:
        if not fx_file.exists():
            _vprint(
                f"Error: fx file '{fx_file}' does not exist.",
                "error",
                verbosity,
            )
            sys.exit(1)
        fx = load_fx_from_file(fx_file)
    elif parsed_args.command == "encode":
        fx_py = Path("fx.py")
        if fx_py.exists():
            _vprint(
                f"Error: {fx_py.resolve()} already exists. Refusing to overwrite. "
                "Move or rename the existing fx.py file to proceed.",
                "error",
                verbosity,
            )
            sys.exit(1)
        fx_obj = generate_default_fx(
            hash_name=parsed_args.hash_name,
            vectorise=parsed_args.vectorise,
            block_size=512 if parsed_args.hash_name == "blake3" else None,
        )
        fx_py.write_text(fx_obj.source_code)
        _vprint(
            f"Warning: Generated a fx-file in {fx_py.resolve()}. "
            "Store securely, this file contains your key stream function.",
            "warning",
            verbosity,
        )
        fx = fx_obj
    else:
        _vprint(
            "Error: --fx-file must be specified when decoding. "
            "Provide the fx-file used for encoding with --fx-file.",
            "error",
            verbosity,
        )
        sys.exit(1)

    # Handle seed
    if seed_file:
        if not seed_file.exists():
            _vprint(
                f"Error: seed file '{seed_file}' does not exist.",
                "error",
                verbosity,
            )
            sys.exit(1)
        seed = bytes.fromhex(seed_file.read_text().strip())
    elif parsed_args.command == "encode":
        seed_hex = Path("seed.hex")
        if seed_hex.exists():
            _vprint(
                f"Error: {seed_hex.resolve()} already exists. Refusing to overwrite. "
                "Move or rename the existing seed.hex file to proceed.",
                "error",
                verbosity,
            )
            sys.exit(1)
        seed = VernamVeil.get_initial_seed()
        seed_hex.write_text(seed.hex())
        _vprint(
            f"Warning: Generated a seed-file in {seed_hex.resolve()}. "
            "Store securely, this file contains your encryption seed.",
            "warning",
            verbosity,
        )
    else:
        _vprint(
            "Error: --seed-file must be specified when decoding. "
            "Provide the seed file used for encoding with --seed-file.",
            "error",
            verbosity,
        )
        sys.exit(1)

    # Check input file existence unless stdin or stream
    if infile not in (None, "-"):
        if not Path(infile).exists():
            _vprint(
                f"Error: input file '{infile}' does not exist.",
                "error",
                verbosity,
            )
            sys.exit(1)

    # Optionally check fx and seed sanity
    if parsed_args.command == "encode" and parsed_args.check_sanity:
        if isinstance(fx, OTPFX):
            _vprint(
                "Warning: fx is an OTPFX. Skipping sanity check to avoid consuming the keystream.",
                "warning",
                verbosity,
            )
        elif not check_fx_sanity(fx, seed, num_samples=10000):
            _vprint(
                "Error: fx sanity check failed. Check your fx function for correctness.",
                "error",
                verbosity,
            )
            sys.exit(1)
        if len(seed) < 16:
            _vprint(
                "Error: Seed is too short. It must be at least 16 bytes for security.",
                "error",
                verbosity,
            )
            sys.exit(1)

    # Print version and platform information
    _vprint(
        f"VernamVeil CLI v{__version__} (numpy: {_HAS_NUMPY}, nphash: {_HAS_C_MODULE}) | "
        f"Python v{sys.version_info.major}.{sys.version_info.minor} | Platform: {sys.platform}",
        "info",
        verbosity,
    )

    # Initialise the VernamVeil object
    cypher = VernamVeil(
        fx,
        chunk_size=parsed_args.chunk_size,
        delimiter_size=parsed_args.delimiter_size,
        padding_range=tuple(parsed_args.padding_range),
        decoy_ratio=parsed_args.decoy_ratio,
        siv_seed_initialisation=parsed_args.siv_seed_initialisation,
        auth_encrypt=parsed_args.auth_encrypt,
        hash_name=parsed_args.hash_name,
    )

    # Define progress callback if verbosity is "info"
    progress_callback: Callable[[int, int], None] | None
    if verbosity == "info":

        def progress_callback(processed: int, total: int) -> None:
            """Callback to report progress during file processing.

            Args:
                processed (int): Number of bytes processed so far.
                total (int): Total number of bytes to process.
            """
            percent = 100.0 * processed / total if total > 0 else 0.0
            if processed >= total:
                _vprint("\rProgress: 100.00%", "info", verbosity, end="", flush=True)
                _vprint("", "info", verbosity, flush=True)
            else:
                _vprint(f"\rProgress: {percent:.2f}%", "info", verbosity, end="", flush=True)

    else:
        progress_callback = None

    # Open input/output (binary mode, handle stdin/stdout)
    try:
        with (
            _open_file(infile, "rb", sys.stdin) as fin,
            _open_file(outfile, "wb", sys.stdout) as fout,
        ):
            start_time = time.perf_counter()
            cypher.process_file(
                parsed_args.command,
                fin,
                fout,
                seed,
                buffer_size=parsed_args.buffer_size,
                progress_callback=progress_callback,
            )
            # Print elapsed time
            _vprint(
                f"The '{parsed_args.command}' step took {time.perf_counter() - start_time:.3f} seconds.",
                "info",
                verbosity,
            )
    except (BrokenPipeError, OSError) as e:
        _vprint(f"Error: I/O error during read/write: {e}", "error", verbosity)
        sys.exit(1)
    except KeyboardInterrupt:
        _vprint("\nOperation cancelled by user.", "error", verbosity)
        sys.exit(130)


if __name__ == "__main__":
    main()
