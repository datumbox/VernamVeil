"""Type definitions and central imports for the VernamVeil project."""

from typing import Any, Literal

# C module types and imports
HashType: Any
_bytesearchffi: Any
_npblake2bffi: Any
_npblake3ffi: Any
_npsha256ffi: Any
try:
    from nphash import _bytesearchffi, _npblake2bffi, _npblake3ffi, _npsha256ffi

    _HAS_C_MODULE = True
    HashType = Literal["blake2b", "blake3", "sha256"]
except ImportError:
    _bytesearchffi = None
    _npblake2bffi = None
    _npblake3ffi = None
    _npsha256ffi = None

    _HAS_C_MODULE = False
    HashType = Literal["blake2b", "sha256"]
