"""Hashing utilities for the library.

This module provides fast, optionally C-accelerated hashing functions for use in key stream generation.
"""

import hashlib
from typing import Literal, cast

import numpy as np

from vernamveil._types import (
    _HAS_C_MODULE,
    HashType,
    _npblake2bffi,
    _npblake3ffi,
    _npsha256ffi,
)

__all__ = ["blake3", "fold_bytes_to_uint64", "hash_numpy"]


class blake3:
    """A hashlib-style BLAKE3 hash object using the C backend.

    This class provides a BLAKE3 hash object with a hashlib-like interface, using the C backend for fast hashing.
    It accumulates data via update() calls and processes all chunks in C upon digest().
    """

    def __init__(
        self,
        data: bytes | bytearray | memoryview = b"",
        key: bytes | bytearray | memoryview | None = None,
        length: int = 32,
    ) -> None:
        """Initialise a BLAKE3 hash object.

        Args:
            data (bytes or bytearray or memoryview): Initial data to hash. Defaults to an empty byte string.
            key (bytes or bytearray or memoryview, optional): Optional key for keyed hashing. If None, no key is used.
            length (int): Desired output length in bytes. Default is 32 bytes.
        """
        self._key = key
        self._length = length
        self._data_chunks: list[bytes | bytearray | memoryview]
        if data:
            self._data_chunks = [data]
        else:
            self._data_chunks = []

    @property
    def digest_size(self) -> int:
        """The size of the hash output in bytes.

        Returns:
            int: 32 bytes by default, can be set during initialisation.
        """
        return self._length

    @property
    def block_size(self) -> int:
        """The size of the internal block used for hashing.

        Returns:
            int: 64 bytes, which is the standard block size for BLAKE3.
        """
        return 64

    @property
    def name(self) -> str:
        """The name of the hash algorithm.

        Returns:
            str: The name of the hash algorithm, which is "blake3".
        """
        return blake3.__name__

    def copy(self) -> "blake3":
        """Return a copy of the current blake3 hash object.

        Returns:
            blake3: A new blake3 object with the same state as the current one.
        """
        new_obj = blake3(key=self._key, length=self._length)
        new_obj._data_chunks = list(self._data_chunks)
        return new_obj

    def update(self, data: bytes | bytearray | memoryview) -> None:
        """Update the hash object with additional data.

        Args:
            data (bytes or bytearray or memoryview): Data to add to the hash.
        """
        self._data_chunks.append(data)

    def digest(self, length: int | None = None) -> bytearray:
        """Compute the BLAKE3 hash of the accumulated data with optional keying and length.

        Args:
            length (int, optional): Desired output length in bytes. If None, uses the default length set during initialisation.

        Returns:
            bytearray: The BLAKE3 hash digest of the accumulated data, optionally keyed and of specified length.

        Raises:
            RuntimeError: If the C-backed BLAKE3 module is not available.
        """
        if not _HAS_C_MODULE:
            raise RuntimeError("C-backed BLAKE3 is not available.")

        if length is None:
            length = self._length

        ffi = _npblake3ffi.ffi

        # This list holds references to cdata objects created by ffi.from_buffer().
        # It's crucial for preventing these objects from being garbage-collected
        # while their underlying memory is still in use by C functions.
        # This ensures the C pointers remain valid throughout the C function call.
        keep_alive_buffers = []

        num_chunks = len(self._data_chunks)
        if num_chunks > 0:
            c_data_chunks_ptr = ffi.new(f"uint8_t*[{num_chunks}]")
            c_data_lengths_ptr = ffi.new(f"size_t[{num_chunks}]")

            for i, chunk_mv in enumerate(self._data_chunks):
                c_buf = ffi.from_buffer(chunk_mv)
                c_data_chunks_ptr[i] = c_buf
                c_data_lengths_ptr[i] = len(chunk_mv)

                keep_alive_buffers.append(c_buf)
        else:
            # Pass NULL pointers for empty chunk arrays if num_chunks is 0
            c_data_chunks_ptr = ffi.NULL
            c_data_lengths_ptr = ffi.NULL

        out = bytearray(length)
        _npblake3ffi.lib.bytes_multi_chunk_blake3(
            c_data_chunks_ptr,
            c_data_lengths_ptr,
            num_chunks,
            ffi.from_buffer(self._key) if self._key is not None else ffi.NULL,
            len(self._key) if self._key is not None else 0,
            ffi.from_buffer(out),
            length,
        )
        return out

    def hexdigest(self, length: int | None = None) -> str:
        """Compute the BLAKE3 hash of the accumulated data and return it as a hexadecimal string.

        Args:
            length (int, optional): Desired output length in bytes. If None, uses the default length set during initialisation.

        Returns:
            str: The BLAKE3 hash digest of the accumulated data as a hexadecimal string.
        """
        return self.digest(length=length).hex()


def fold_bytes_to_uint64(
    hashes: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    fold_type: Literal["full", "view"] = "view",
) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
    """Fold each row of a 2D uint8 hash output into a uint64 integer (big-endian).

    Args:
        hashes (np.ndarray[tuple[int, int], np.dtype[np.uint8]]): 2D array of shape (n, H) where H >= 8.
        fold_type (Literal["full", "view"]): Folding strategy.
            "view": Fastest; reinterprets the first 8 bytes as uint64.
            "full": Slower; folds all bytes in the row using bitwise operations.
            Default is "view".

    Returns:
        np.ndarray[tuple[int], np.dtype[np.uint64]]: 1D array of length n, each element is the folded uint64 value of the corresponding row.

    Raises:
        ValueError: If the input array is not 2D or has less than 8 columns.
        ValueError: If fold_type is not 'full' or 'view'.
    """
    if hashes.ndim != 2:
        raise ValueError("The input must be a 2D array.")
    cols = hashes.shape[1]
    if cols < 8:
        raise ValueError("The input must have at least 8 columns per row.")
    if fold_type == "view":
        # Create a view of the first 8 bytes as uint64 (big-endian)
        if cols > 8:
            # If there are more than 8 columns, we only take the first 8
            hashes = cast(np.ndarray[tuple[int, int], np.dtype[np.uint8]], hashes[:, :8])
        return hashes.view(np.uint64).reshape(-1).byteswap()
    elif fold_type == "full":
        # Compute the shifts for each byte position (big-endian)
        shifts = np.arange(8 * cols - 8, -1, -8, dtype=np.uint64)

        # Cast to uint64
        hashes_u64 = hashes.astype(np.uint64, copy=False)

        # Compute folded values in a fully vectorised way
        result: np.ndarray[tuple[int], np.dtype[np.uint64]] = np.bitwise_or.reduce(
            hashes_u64 << shifts, axis=1
        )
        return result
    else:
        raise ValueError(f"Unsupported fold_type '{fold_type}'. Use 'full' or 'view'.")


def hash_numpy(
    i: np.ndarray[tuple[int], np.dtype[np.uint64]],
    seed: bytes | bytearray | None = None,
    hash_name: HashType = "blake2b",
    hash_size: int | None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    """Compute a 2D NumPy array of uint8 by applying a hash function to each index, optionally using a seed as a key.

    If no seed is provided, the index is hashed directly.

    This function optionally uses cffi to call a custom C library, which wraps an optimised C implementation
    (with OpenMP and OpenSSL) for efficient, parallelised hashing from Python. If the C module isn't available
    a NumPy fallback is used.

    Args:
        i (np.ndarray[tuple[int], np.dtype[np.uint64]]): NumPy array of indices (dtype should be unsigned 64-bit integer).
        seed (bytes or bytearray, optional): The seed bytes are prepended to the index. If None, hashes only the index.
        hash_name (HashType): Hash function to use ("blake2b", "blake3" or "sha256"). The blake3 is only available
            if the C extension is installed. Defaults to "blake2b".
        hash_size (int, optional): Size of the hash output in bytes. Should be 64 for blake2b, larger than 0 for blake3
            and 32 for sha256. If None, the default size for the selected hash algorithm is used. Defaults to None.

    Returns:
        np.ndarray[tuple[int, int], np.dtype[np.uint8]]: A 2D array of shape (n, H) where H is the hash output size in
        bytes. Each row contains the full hash output for the corresponding input.

    Raises:
        ValueError: If the hash_size is not 64 for blake2b, larger than 0 for blake3 or 32 for sha256.
        ValueError: If a hash algorithm is not supported.
        ValueError: If `hash_name` is "blake3" but the C extension is not available.
    """
    if hash_name == "blake2b":
        if hash_size is None:
            hash_size = 64
        elif hash_size != 64:
            raise ValueError("blake2b hash_size must be 64 bytes.")
        if _HAS_C_MODULE:
            ffi = _npblake2bffi.ffi
            method = _npblake2bffi.lib.numpy_blake2b
        else:
            ffi = None
            method = hashlib.blake2b
    elif hash_name == "blake3":
        if hash_size is None:
            hash_size = 32
        elif hash_size <= 0:
            raise ValueError("blake3 hash_size must be larger than 0 bytes.")
        if _HAS_C_MODULE:
            ffi = _npblake3ffi.ffi
            method = _npblake3ffi.lib.numpy_blake3
        else:
            raise ValueError("blake3 requires the C extension.")
    elif hash_name == "sha256":
        if hash_size is None:
            hash_size = 32
        elif hash_size != 32:
            raise ValueError("sha256 hash_size must be 32 bytes.")
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
        args = [
            ffi.cast("const uint64_t *", ffi.from_buffer(i)),
            n,
            ffi.from_buffer(seed) if seed is not None else ffi.NULL,
            len(seed) if seed is not None else 0,
            ffi.from_buffer(out),
        ]
        if hash_name == "blake3":
            args.append(hash_size)
        method(*args)
    else:
        i_bytes = i.view(np.uint8)
        for idx, j in enumerate(range(0, len(i_bytes), 8)):
            hasher = method()
            if seed is not None:
                hasher.update(seed)
            hasher.update(i_bytes.data[j : j + 8])
            out[idx, :] = np.frombuffer(hasher.digest(), dtype=np.uint8)

    return out
