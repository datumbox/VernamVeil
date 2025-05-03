"""
Hashing utilities for VernamVeil.

This module provides fast, optionally C-accelerated hashing functions for use in key stream generation.
"""

import hashlib
import hmac
from typing import Literal

try:
    import numpy as np
    from numpy.typing import NDArray

    from nphash import _npblake2bffi  # type: ignore[attr-defined]
    from nphash import _npsha256ffi  # type: ignore[attr-defined]

    _HAS_C_MODULE = True
except ImportError:
    _HAS_C_MODULE = False


_UINT64_BOUND = 2**64

__all__ = ["hash_numpy"]


def hash_numpy(
    i: "NDArray[np.uint64]",
    seed: bytes | None = None,
    hash_name: Literal["blake2b", "sha256"] = "blake2b",
) -> "NDArray[np.uint64]":
    """
    Computes a 64-bit integer NumPy array by HMAC-ing each index with a seed using a hashing algorithm.
    If no seed is provided, the index is hashed directly.

    This function optionally uses cffi to call a custom C library, which wraps an optimised C implementation
    (with OpenMP and OpenSSL) for efficient, parallelised HMAC hashing from Python. If the C module isn't available
    a NumPy fallback is used.

    Args:
        i (NDArray[np.uint64]): NumPy array of indices (dtype should be unsigned 64-bit integer).
        seed (bytes, optional): The seed bytes used as the HMAC key. If None, hashes only the index.
        hash_name (Literal["blake2b", "sha256"], optional): Hash algorithm to use. Defaults to "blake2b".

    Returns:
        NDArray[np.uint64]: An array of 64-bit integers derived from the HMAC of each index.

    Raises:
        ValueError: If a hash algorithm is not supported.
    """
    i_bytes = i.astype(">u8").tobytes()
    if _HAS_C_MODULE:
        if hash_name == "blake2b":
            ffi = _npblake2bffi.ffi
            method = _npblake2bffi.lib.numpy_blake2b
        elif hash_name == "sha256":
            ffi = _npsha256ffi.ffi
            method = _npsha256ffi.lib.numpy_sha256
        else:
            raise ValueError(f"Unsupported hash_name '{hash_name}'.")

        n = len(i_bytes) // 8
        out = np.empty(n, dtype=np.uint64)
        method(
            ffi.from_buffer(i_bytes),
            n,
            ffi.from_buffer(seed) if seed is not None else ffi.NULL,
            len(seed) if seed is not None else 0,
            ffi.cast("uint64_t*", out.ctypes.data),
        )
        return out
    else:
        if hash_name == "blake2b":
            method = hashlib.blake2b
        elif hash_name == "sha256":
            method = hashlib.sha256
        else:
            raise ValueError(f"Unsupported hash_name '{hash_name}'.")

        return np.fromiter(
            (
                int.from_bytes(
                    (
                        hmac.new(seed, i_bytes[j : j + 8], digestmod=method).digest()
                        if seed is not None
                        else method(i_bytes[j : j + 8]).digest()
                    ),
                    "big",
                )
                % _UINT64_BOUND
                for j in range(0, len(i_bytes), 8)
            ),
            dtype=np.uint64,
        )
