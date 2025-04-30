import hashlib
from typing import Literal

try:
    import numpy as np
    from nphash import _npsha256ffi

    _HAS_C_MODULE = True
except ImportError:
    _HAS_C_MODULE = False


_UINT64_BOUND = 2**64


def hash_numpy(
    i: "np.ndarray", seed: bytes | None = None, hash_name: Literal["sha256"] = "sha256"
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
        hash_name (Literal["sha256"], optional): Hash algorithm to use. Only "sha256" is currently supported.

    Returns:
        np.ndarray: An array of 64-bit integers derived from the hash of each (seed || 4-byte block) or
        just the block if seed is None.

    Raises:
        ValueError: If a hash algorithm is not supported.
    """
    if hash_name not in {"sha256"}:
        raise ValueError(f"Unsupported hash_name '{hash_name}'.")

    i_bytes = np.frombuffer(i.astype(">u4"), dtype="S4").tobytes()
    if _HAS_C_MODULE:
        n = len(i_bytes) // 4
        out = np.empty(n, dtype=np.uint64)
        _npsha256ffi.lib.numpy_sha256(
            _npsha256ffi.ffi.from_buffer(i_bytes),
            n,
            _npsha256ffi.ffi.from_buffer(seed) if seed is not None else _npsha256ffi.ffi.NULL,
            len(seed) if seed is not None else 0,
            _npsha256ffi.ffi.cast("uint64_t*", out.ctypes.data),
        )
        return out
    else:
        return np.fromiter(
            (
                int.from_bytes(
                    hashlib.sha256(
                        (seed if seed is not None else b"") + i_bytes[j : j + 4]
                    ).digest(),
                    "big",
                )
                % _UINT64_BOUND
                for j in range(0, len(i_bytes), 4)
            ),
            dtype=np.uint64,
        )
