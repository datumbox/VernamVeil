"""Central imports for the VernamVeil library."""

from typing import Any

__all__: list[str] = []

np: Any
try:
    import numpy

    np = numpy
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

_bytesearchffi: Any
_npblake2bffi: Any
_npblake3ffi: Any
_npsha256ffi: Any
try:
    from nphash import _bytesearchffi, _npblake2bffi, _npblake3ffi, _npsha256ffi

    _HAS_C_MODULE = True
except ImportError:
    _bytesearchffi = None
    _npblake2bffi = None
    _npblake3ffi = None
    _npsha256ffi = None

    _HAS_C_MODULE = False
