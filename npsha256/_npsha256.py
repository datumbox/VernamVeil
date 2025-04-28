import hashlib

try:
    import numpy as np
    from ._npsha256ffi import ffi, lib

    _HAS_C_MODULE = True
except ImportError:
    _HAS_C_MODULE = False


def numpy_sha256(i: "np.ndarray", seed: bytes | None = None) -> "np.ndarray":
    """
    Computes a 64-bit integer NumPy array by hashing each index (as a 4-byte big-endian block) with a seed using SHA256.

    This function optionally uses cffi to call a custom C library, which wraps an optimized C implementation
    (with OpenMP and OpenSSL) for efficient, parallelized hashing from Python. If the C module isn't available
    a NumPy fallback is used.

    Args:
        i (np.ndarray): NumPy array of indices (dtype should be unsigned 32-bit or 64-bit integer).
        seed (bytes, optional): The seed bytes used to influence the hash result. If None, hashes only the index.

    Returns:
        np.ndarray: An array of 64-bit integers derived from the SHA256 hash of each (seed || 4-byte block) or
        just the block if seed is None.
    """
    i_bytes = np.frombuffer(i.astype(">u4").tobytes(), dtype="S4").tobytes()
    if _HAS_C_MODULE:
        n = len(i_bytes) // 4
        out = np.empty(n, dtype=np.uint64)
        lib.numpy_sha256(
            ffi.from_buffer(i_bytes),
            n,
            ffi.from_buffer(seed) if seed is not None else ffi.NULL,
            len(seed) if seed is not None else 0,
            ffi.cast("uint64_t*", out.ctypes.data),
        )
        return out
    else:
        uint64_bound = 2**64
        return np.fromiter(
            (
                int.from_bytes(
                    hashlib.sha256(
                        (seed if seed is not None else b"") + i_bytes[j : j + 4]
                    ).digest(),
                    "big",
                )
                % uint64_bound
                for j in range(0, len(i_bytes), 4)
            ),
            dtype=np.uint64,
        )
