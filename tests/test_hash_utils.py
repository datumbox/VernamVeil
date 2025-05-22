import hashlib
import hmac
import math
import secrets
import unittest
from unittest.mock import patch

from vernamveil._cypher import _HAS_NUMPY
from vernamveil._hash_utils import _HAS_C_MODULE, fold_bytes_to_uint64, hash_numpy, hkdf_numpy

try:
    import numpy as np
except ImportError:
    pass


class TestHashUtils(unittest.TestCase):
    """Unit tests for the has_utils.py utilities."""

    def _get_hash_method_for_test(self, hash_name):
        """Returns the appropriate hashlib function for the given hash_name."""
        if hash_name == "sha256":
            return hashlib.sha256
        elif hash_name == "blake2b":
            return hashlib.blake2b
        else:
            raise ValueError(f"Unsupported hash_name '{hash_name}'.")

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_hash_numpy_correctness(self):
        """Check that hash_numpy output matches expected hash values for a range of inputs, for both fold types."""
        checks = [False]
        if _HAS_C_MODULE:
            checks.append(True)
        for has_c in checks:
            for hash_name in ("sha256", "blake2b"):
                for fold_type in ("full", "view"):
                    with self.subTest(
                        _HAS_C_MODULE=has_c, hash_name=hash_name, fold_type=fold_type
                    ):
                        print(
                            f"_HAS_C_MODULE={has_c}, hash_name={hash_name}, fold_type={fold_type}"
                        )
                        with patch("vernamveil._hash_utils._HAS_C_MODULE", has_c):
                            seed = secrets.token_bytes(64)
                            i = np.arange(1, 1000, dtype=np.uint64)

                            output = fold_bytes_to_uint64(
                                hash_numpy(i, seed, hash_name), fold_type=fold_type
                            )

                            i_bytes = i.tobytes()
                            method = self._get_hash_method_for_test(hash_name)

                            def get_digest(j):
                                hasher = method(seed)
                                hasher.update(i_bytes[j : j + 8])
                                digest = hasher.digest()
                                if fold_type == "full":
                                    return int.from_bytes(digest, "big") % 2**64
                                else:  # "view"
                                    return int.from_bytes(digest[:8], "big")

                            expected = np.fromiter(
                                (get_digest(j) for j in range(0, len(i_bytes), 8)),
                                dtype=np.uint64,
                            )

                            np.testing.assert_array_equal(expected, output)

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_hash_numpy_no_seed(self):
        """Check that hash_numpy works with no seed (seed=None), for both fold types."""
        checks = [False]
        if _HAS_C_MODULE:
            checks.append(True)
        for has_c in checks:
            for hash_name in ("sha256", "blake2b"):
                for fold_type in ("full", "view"):
                    with self.subTest(
                        _HAS_C_MODULE=has_c, hash_name=hash_name, fold_type=fold_type
                    ):
                        print(
                            f"_HAS_C_MODULE={has_c}, hash_name={hash_name}, fold_type={fold_type}"
                        )
                        with patch("vernamveil._hash_utils._HAS_C_MODULE", has_c):
                            i = np.arange(1, 1000, dtype=np.uint64)

                            output = fold_bytes_to_uint64(
                                hash_numpy(i, None, hash_name), fold_type=fold_type
                            )

                            i_bytes = i.tobytes()
                            method = self._get_hash_method_for_test(hash_name)

                            def get_digest(j):
                                digest = method(i_bytes[j : j + 8]).digest()
                                if fold_type == "full":
                                    return int.from_bytes(digest, "big") % 2**64
                                else:  # "view"
                                    return int.from_bytes(digest[:8], "big")

                            expected = np.fromiter(
                                (get_digest(j) for j in range(0, len(i_bytes), 8)),
                                dtype=np.uint64,
                            )

                            np.testing.assert_array_equal(expected, output)

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_numpy_hkdf_correctness(self):
        """Test numpy_hkdf output matches a pure Python HKDF-Expand for both algorithms and C/Python paths."""
        checks = [False]
        if _HAS_C_MODULE:
            checks.append(True)
        for has_c in checks:
            for digest_name, hash_size in (("sha256", 32), ("blake2b", 64)):
                with self.subTest(_HAS_C_MODULE=has_c, digest_name=digest_name):
                    with patch("vernamveil._hash_utils._HAS_C_MODULE", has_c):
                        key = secrets.token_bytes(hash_size)
                        info = secrets.token_bytes(16)
                        outlen = 100  # Not a multiple of hash_size to test partial block

                        # Reference HKDF-Expand (RFC 5869)
                        def hkdf_expand(key, info, outlen, digest_name):
                            if digest_name == "sha256":
                                digestmod = "sha256"
                                hash_size = 32
                            elif digest_name == "blake2b":
                                digestmod = "blake2b"
                                hash_size = 64
                            else:
                                raise ValueError
                            n_blocks = math.ceil(outlen / hash_size)
                            prev = b""
                            output = bytearray()
                            for i in range(1, n_blocks + 1):
                                prev = hmac.new(key, prev + info + bytes([i]), digestmod).digest()
                                output += prev
                            return bytes(output[:outlen])

                        expected = np.frombuffer(
                            hkdf_expand(key, info, outlen, digest_name), dtype=np.uint8
                        )
                        actual = hkdf_numpy(key, info, outlen, digest_name)
                        np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
