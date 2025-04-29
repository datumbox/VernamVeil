import unittest
import hashlib
import secrets
from unittest.mock import patch

from npsha256 import numpy_sha256
from npsha256._npsha256 import _HAS_C_MODULE

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TestNumPySha256(unittest.TestCase):
    """Unit tests for the numpy_sha256 function."""

    @unittest.skipUnless(HAS_NUMPY, "NumPy not available")
    def test_numpy_sha256_correctness(self):
        """Check that numpy_sha256 output matches expected SHA-256 values for a range of inputs."""
        checks = [False]
        if _HAS_C_MODULE:
            checks.append(True)
        for has_c in checks:
            with self.subTest(_HAS_C_MODULE=has_c):
                print(f"_HAS_C_MODULE={has_c}")
                with patch("npsha256._npsha256._HAS_C_MODULE", has_c):
                    seed = secrets.token_bytes(32)
                    i = np.arange(1, 1000, dtype=np.uint64)

                    output = numpy_sha256(i, seed)

                    i_bytes = np.frombuffer(i.astype(">u4").tobytes(), dtype="S4").tobytes()
                    uint64_bound = 2**64
                    expected = np.fromiter(
                        (
                            int.from_bytes(
                                hashlib.sha256(seed + i_bytes[j : j + 4]).digest(), "big"
                            )
                            % uint64_bound
                            for j in range(0, len(i_bytes), 4)
                        ),
                        dtype=np.uint64,
                    )

                    np.testing.assert_array_equal(expected, output)

    @unittest.skipUnless(HAS_NUMPY, "NumPy not available")
    def test_numpy_sha256_no_seed(self):
        """Check that numpy_sha256 works with no seed (seed=None)."""
        checks = [False]
        if _HAS_C_MODULE:
            checks.append(True)
        for has_c in checks:
            with self.subTest(_HAS_C_MODULE=has_c):
                print(f"_HAS_C_MODULE={has_c}")
                with patch("npsha256._npsha256._HAS_C_MODULE", has_c):
                    i = np.arange(1, 100, dtype=np.uint64)

                    output = numpy_sha256(i, None)

                    i_bytes = np.frombuffer(i.astype(">u4").tobytes(), dtype="S4").tobytes()
                    uint64_bound = 2**64
                    expected = np.fromiter(
                        (
                            int.from_bytes(hashlib.sha256(i_bytes[j : j + 4]).digest(), "big")
                            % uint64_bound
                            for j in range(0, len(i_bytes), 4)
                        ),
                        dtype=np.uint64,
                    )

                    np.testing.assert_array_equal(expected, output)


if __name__ == "__main__":
    unittest.main()
