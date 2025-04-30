import unittest
import hashlib
import secrets
from unittest.mock import patch

from vernamveil.cypher import _HAS_NUMPY
from vernamveil.hash_utils import hash_numpy, _HAS_C_MODULE, _UINT64_BOUND

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
        """Check that hash_numpy output matches expected hash values for a range of inputs."""
        checks = [False]
        if _HAS_C_MODULE:
            checks.append(True)
        for has_c in checks:
            for hash_name in ("sha256", "blake2b"):
                with self.subTest(_HAS_C_MODULE=has_c, hash_name=hash_name):
                    print(f"_HAS_C_MODULE={has_c}, hash_name={hash_name}")
                    with patch("vernamveil.hash_utils._HAS_C_MODULE", has_c):
                        seed = secrets.token_bytes(32)
                        i = np.arange(1, 1000, dtype=np.uint64)

                        output = hash_numpy(i, seed, hash_name)

                        i_bytes = np.frombuffer(i.astype(">u4"), dtype="S4").tobytes()
                        method = self._get_hash_method_for_test(hash_name)

                        expected = np.fromiter(
                            (
                                int.from_bytes(method(seed + i_bytes[j : j + 4]).digest(), "big")
                                % _UINT64_BOUND
                                for j in range(0, len(i_bytes), 4)
                            ),
                            dtype=np.uint64,
                        )

                        np.testing.assert_array_equal(expected, output)

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_hash_numpy_no_seed(self):
        """Check that hash_numpy works with no seed (seed=None)."""
        checks = [False]
        if _HAS_C_MODULE:
            checks.append(True)
        for has_c in checks:
            for hash_name in ("sha256", "blake2b"):
                with self.subTest(_HAS_C_MODULE=has_c, hash_name=hash_name):
                    print(f"_HAS_C_MODULE={has_c}, hash_name={hash_name}")
                    with patch("vernamveil.hash_utils._HAS_C_MODULE", has_c):
                        i = np.arange(1, 100, dtype=np.uint64)

                        output = hash_numpy(i, None, hash_name)

                        i_bytes = np.frombuffer(i.astype(">u4"), dtype="S4").tobytes()
                        method = self._get_hash_method_for_test(hash_name)

                        expected = np.fromiter(
                            (
                                int.from_bytes(method(i_bytes[j : j + 4]).digest(), "big")
                                % _UINT64_BOUND
                                for j in range(0, len(i_bytes), 4)
                            ),
                            dtype=np.uint64,
                        )

                        np.testing.assert_array_equal(expected, output)


if __name__ == "__main__":
    unittest.main()
