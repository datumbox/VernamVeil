"""This module contains the optional C extension of VernamVeil for fast hashing."""

from types import ModuleType

_kmpffi: ModuleType
_npblake2bffi: ModuleType
_npblake3ffi: ModuleType
_npsha256ffi: ModuleType

__all__: list[str] = []
