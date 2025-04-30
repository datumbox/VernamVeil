import hashlib
from typing import Literal

try:
    import numpy as np
    from nphash import _npblake2bffi
    from nphash import _npsha256ffi

    _HAS_C_MODULE = True
except ImportError:
    _HAS_C_MODULE = False


_UINT64_BOUND = 2**64


def hash_numpy(
    i: "np.ndarray", seed: bytes | None = None, hash_name: Literal["blake2b", "sha256"] = "blake2b"
) -> "np.ndarray":
    """
    Computes a 64-bit integer NumPy array by hashing each index (as a 4-byte big-endian block) with a seed using
    a hashing algorithm.

    This function optionally uses cffi to call a custom C library, which wraps an optimised C implementation
    (with OpenMP and OpenSSL) for efficient, parallelised hashing from Python. If the C module isn't available
    a NumPy fallback is used.

    Args:
        i (np.ndarray): NumPy array of indices (dtype should be unsigned 64-bit integer).
        seed (bytes, optional): The seed bytes used to influence the hash result. If None, hashes only the index.
        hash_name (Literal["blake2b", "sha256"], optional): Hash algorithm to use. Defaults to "blake2b".

    Returns:
        np.ndarray: An array of 64-bit integers derived from the hash of each (seed || 4-byte block) or
        just the block if seed is None.

    Raises:
        ValueError: If a hash algorithm is not supported.
    """
    i_bytes = np.frombuffer(i.astype(">u4"), dtype="S4").tobytes()
    if _HAS_C_MODULE:
        if hash_name == "blake2b":
            ffi = _npblake2bffi.ffi
            method = _npblake2bffi.lib.numpy_blake2b
        elif hash_name == "sha256":
            ffi = _npsha256ffi.ffi
            method = _npsha256ffi.lib.numpy_sha256
        else:
            raise ValueError(f"Unsupported hash_name '{hash_name}'.")

        n = len(i_bytes) // 4
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
                    method((seed if seed is not None else b"") + i_bytes[j : j + 4]).digest(),
                    "big",
                )
                % _UINT64_BOUND
                for j in range(0, len(i_bytes), 4)
            ),
            dtype=np.uint64,
        )
