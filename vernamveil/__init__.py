try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # type: ignore[no-redef]

from .cypher import VernamVeil
from .fx_utils import (
    check_fx_sanity,
    generate_default_fx,
    generate_hmac_fx,
    generate_polynomial_fx,
    load_fx_from_file,
)
from .hash_utils import hash_numpy

try:
    __version__ = version("vernamveil")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "check_fx_sanity",
    "generate_default_fx",
    "generate_hmac_fx",
    "generate_polynomial_fx",
    "hash_numpy",
    "load_fx_from_file",
    "VernamVeil",
]
