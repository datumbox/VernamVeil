import argparse
import sys
from pathlib import Path
from .cypher import VernamVeil
from .fx_utils import generate_default_fx, load_fx_from_file, check_fx_sanity


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """
    Adds common CLI arguments for both encode and decode subcommands.

    Args:
        p (argparse.ArgumentParser): The argument parser to which the arguments will be added.
    """
    p.add_argument("--infile", type=Path, required=True, help="Input file path (required).")
    p.add_argument("--outfile", type=Path, required=True, help="Output file path (required).")
    p.add_argument("--fx-file", type=Path, help="Path to Python file containing the fx function.")
    p.add_argument("--seed-file", type=Path, help="Path to file containing the seed (bytes).")
    p.add_argument(
        "--fx-complexity",
        type=int,
        default=20,
        help="Complexity for random fx (if fx-file omitted). Default: 20.",
    )
    p.add_argument(
        "--vectorise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vectorised fx (default: True).",
    )
    p.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size for VernamVeil (default: 512)."
    )
    p.add_argument(
        "--delimiter-size",
        type=int,
        default=10,
        help="Delimiter size for VernamVeil (default: 10).",
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
        "--auth-encrypt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable authenticated encryption (default: True).",
    )
    p.add_argument(
        "--check-fx-sanity",
        action="store_true",
        help="Check the loaded/generated fx with check_fx_sanity.",
    )


def main(args=None) -> None:
    """
    Entry point for the VernamVeil CLI utility.
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

    # Decode subcommand
    dec = subparsers.add_parser("decode", help="Decrypt a file.")
    _add_common_args(dec)

    parsed_args = parser.parse_args(args)

    infile = parsed_args.infile
    outfile = parsed_args.outfile
    fx_file = parsed_args.fx_file
    seed_file = parsed_args.seed_file

    # Check if output file exists
    if outfile.exists():
        print(f"Error: {outfile} already exists. Refusing to overwrite.", file=sys.stderr)
        sys.exit(1)

    # Handle fx function
    if fx_file:
        fx = load_fx_from_file(fx_file)
    elif parsed_args.command == "encode":
        fx_py = Path("fx.py")
        if fx_py.exists():
            print("Error: fx.py already exists. Refusing to overwrite.", file=sys.stderr)
            sys.exit(1)
        fx_obj = generate_default_fx(parsed_args.fx_complexity, vectorise=parsed_args.vectorise)
        fx_code = fx_obj._source_code
        if parsed_args.vectorise:
            fx_code = ("from vernamveil import hash_numpy\n" "import numpy as np\n") + fx_code
        else:
            fx_code = "import hashlib\n" + fx_code
        fx_py.write_text(fx_code)
        print("Generated fx.py in current directory. Store securely.", file=sys.stderr)
        fx = fx_obj
    else:
        print("fx-file is required for decode.", file=sys.stderr)
        sys.exit(1)

    # Handle seed
    if seed_file:
        seed = seed_file.read_bytes()
    elif parsed_args.command == "encode":
        seed_bin = Path("seed.bin")
        if seed_bin.exists():
            print("Error: seed.bin already exists. Refusing to overwrite.", file=sys.stderr)
            sys.exit(1)
        seed = VernamVeil.get_initial_seed()
        seed_bin.write_bytes(seed)
        print("Generated seed.bin in current directory. Store securely.", file=sys.stderr)
    else:
        print("seed-file is required for decode.", file=sys.stderr)
        sys.exit(1)

    # Optionally check fx sanity
    if parsed_args.check_fx_sanity:
        if not check_fx_sanity(fx, seed):
            print("fx sanity check failed.", file=sys.stderr)
            sys.exit(1)

    # Prepare VernamVeil keyword arguments
    vernamveil_kwargs = {
        "chunk_size": parsed_args.chunk_size,
        "delimiter_size": parsed_args.delimiter_size,
        "padding_range": tuple(parsed_args.padding_range),
        "decoy_ratio": parsed_args.decoy_ratio,
        "auth_encrypt": parsed_args.auth_encrypt,
        "vectorise": parsed_args.vectorise,
    }

    # Only file-to-file mode is supported
    VernamVeil.process_file(
        infile,
        outfile,
        fx,
        seed,
        mode=parsed_args.command,
        **vernamveil_kwargs,
    )


if __name__ == "__main__":
    main()
