"""VernamVeil: A Function-Based Stream Cypher."""

from importlib.metadata import PackageNotFoundError, version

from vernamveil._deniability_utils import forge_plausible_fx
from vernamveil._find import find_all
from vernamveil._fx_utils import (
    FX,
    OTPFX,
    check_fx_sanity,
    generate_default_fx,
    generate_keyed_hash_fx,
    generate_polynomial_fx,
    load_fx_from_file,
)
from vernamveil._hash_utils import blake3, fold_bytes_to_uint64, hash_numpy
from vernamveil._vernamveil import VernamVeil

__version__: str
"""The version of the library."""
try:
    __version__ = version("vernamveil")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "FX",
    "OTPFX",
    "VernamVeil",
    "blake3",
    "check_fx_sanity",
    "find_all",
    "fold_bytes_to_uint64",
    "forge_plausible_fx",
    "generate_default_fx",
    "generate_keyed_hash_fx",
    "generate_polynomial_fx",
    "hash_numpy",
    "load_fx_from_file",
]
