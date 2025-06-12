"""Type definitions and central imports for the VernamVeil project."""

from typing import Any, Literal

__all__: list[str] = []

np: Any
try:
    import numpy

    np = numpy
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

# C module types and imports
_HashType: Any
_bytesearchffi: Any
_npblake2bffi: Any
_npblake3ffi: Any
_npsha256ffi: Any
try:
    from nphash import _bytesearchffi, _npblake2bffi, _npblake3ffi, _npsha256ffi

    _HAS_C_MODULE = True
    _HashType = Literal["blake2b", "blake3", "sha256"]
except ImportError:
    _bytesearchffi = None
    _npblake2bffi = None
    _npblake3ffi = None
    _npsha256ffi = None

    _HAS_C_MODULE = False
    _HashType = Literal["blake2b", "sha256"]
