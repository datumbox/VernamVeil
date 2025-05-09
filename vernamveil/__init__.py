"""VernamVeil: A Function-Based Stream Cypher."""

from importlib.metadata import PackageNotFoundError, version

from vernamveil._deniability_utils import forge_plausible_fx
from vernamveil._fx_utils import (
    check_fx_sanity,
    generate_default_fx,
    generate_hmac_fx,
    generate_polynomial_fx,
    load_fx_from_file,
)
from vernamveil._hash_utils import hash_numpy
from vernamveil._vernamveil import VernamVeil

__version__: str
"""The version of the library."""
try:
    __version__ = version("vernamveil")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "VernamVeil",
    "check_fx_sanity",
    "forge_plausible_fx",
    "generate_default_fx",
    "generate_hmac_fx",
    "generate_polynomial_fx",
    "hash_numpy",
    "load_fx_from_file",
]
