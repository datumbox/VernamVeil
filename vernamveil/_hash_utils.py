"""Hashing utilities for the library.

This module provides fast, optionally C-accelerated hashing functions for use in key stream generation.
"""

import hashlib
import hmac
from typing import Literal

try:
    import numpy as np

    from nphash import _npblake2bffi, _npsha256ffi

    _HAS_C_MODULE = True
except ImportError:
    _HAS_C_MODULE = False

__all__ = ["fold_bytes_to_uint64", "hash_numpy"]


def fold_bytes_to_uint64(
    hashes: "np.ndarray[tuple[int, int], np.dtype[np.uint8]]",
    fold_type: Literal["full", "view"] = "view",
) -> "np.ndarray[tuple[int], np.dtype[np.uint64]]":
    """Fold each row of a 2D uint8 hash output into a uint64 integer (big-endian).

    Args:
        hashes (np.ndarray[tuple[int, int], np.dtype[np.uint8]]): 2D array of shape (n, H) where H >= 8.
        fold_type (Literal["full", "view"] = "view"): Folding strategy.
            "view": Fastest; reinterprets the first 8 bytes as uint64.
            "full": Slower; folds all bytes in the row using bitwise operations.
            Default is "view".

    Returns:
        np.ndarray[tuple[int], np.dtype[np.uint64]]: 1D array of length n, each element is the folded uint64 value of the corresponding row.

    Raises:
        ValueError: If the input array is not 2D or has less than 8 columns.
        ValueError: If fold_type is not 'full' or 'view'.
    """
    if hashes.ndim != 2 or hashes.shape[1] < 8:
        raise ValueError("Input must be a 2D array with at least 8 columns per row.")
    if fold_type == "view":
        # Create a view of the first 8 bytes as uint64 (big-endian)
        return hashes[:, :8].view(np.uint64).reshape(-1).byteswap()
    elif fold_type == "full":
        # Compute the shifts for each byte position (big-endian)
        shifts = np.arange((hashes.shape[1] - 1) * 8, -1, -8, dtype=np.uint64)

        # Cast to uint64
        hashes_u64 = hashes.astype(np.uint64, copy=False)

        # Compute folded values in a fully vectorized way
        result: np.ndarray[tuple[int], np.dtype[np.uint64]] = np.bitwise_or.reduce(
            hashes_u64 << shifts, axis=1
        )
        return result
    else:
        raise ValueError(f"Unsupported fold_type '{fold_type}'. Use 'full' or 'view'.")


def hash_numpy(
    i: "np.ndarray[tuple[int], np.dtype[np.uint64]]",
    seed: bytes | None = None,
    hash_name: Literal["blake2b", "sha256"] = "blake2b",
) -> "np.ndarray[tuple[int, int], np.dtype[np.uint8]]":
    """Compute a 2D NumPy array of uint8 by applying a hash function to each index, optionally using a seed as a key.

    If no seed is provided, the index is hashed directly.

    This function optionally uses cffi to call a custom C library, which wraps an optimised C implementation
    (with OpenMP and OpenSSL) for efficient, parallelised hashing from Python. If the C module isn't available
    a NumPy fallback is used.

    Args:
        i (np.ndarray[tuple[int], np.dtype[np.uint64]]): NumPy array of indices (dtype should be unsigned 64-bit integer).
        seed (bytes, optional): The seed bytes are prepended to the index. If None, hashes only the index.
        hash_name (Literal["blake2b", "sha256"]): Hash algorithm to use. Defaults to "blake2b".

    Returns:
        np.ndarray[tuple[int, int], np.dtype[np.uint8]]: A 2D array of shape (n, H) where H is the hash output size in
        bytes (32 for sha256, 64 for blake2b). Each row contains the full hash output for the corresponding input.

    Raises:
        ValueError: If a hash algorithm is not supported.
    """
    if hash_name == "blake2b":
        hash_size = 64
        if _HAS_C_MODULE:
            ffi = _npblake2bffi.ffi
            method = _npblake2bffi.lib.numpy_blake2b
        else:
            ffi = None
            method = hashlib.blake2b
    elif hash_name == "sha256":
        hash_size = 32
        if _HAS_C_MODULE:
            ffi = _npsha256ffi.ffi
            method = _npsha256ffi.lib.numpy_sha256
        else:
            ffi = None
            method = hashlib.sha256
    else:
        raise ValueError(f"Unsupported hash_name '{hash_name}'.")
    n = len(i)
    out = np.empty((n, hash_size), dtype=np.uint8)

    if ffi is not None:
        method(
            ffi.cast("const uint64_t *", ffi.from_buffer(i)),
            n,
            ffi.from_buffer(seed) if seed is not None else ffi.NULL,
            len(seed) if seed is not None else 0,
            ffi.from_buffer(out.data),
        )
    else:
        i_bytes = i.view(np.uint8).data
        for idx, j in enumerate(range(0, len(i_bytes), 8)):
            if hash_name == "blake2b":
                hasher = method()
                if seed is not None:
                    hasher.update(seed)
                hasher.update(i_bytes[j : j + 8])
                digest = hasher.digest()
            elif hash_name == "sha256":
                if seed is not None:  # Not safe for seeded hashes, use HMAC
                    digest = hmac.new(seed, i_bytes[j : j + 8], digestmod=method).digest()
                else:
                    digest = method(i_bytes[j : j + 8]).digest()
            out[idx, :] = np.frombuffer(digest, dtype=np.uint8)

    return out
