import hashlib
import secrets
import unittest
from unittest.mock import patch

from vernamveil._hash_utils import blake3, fold_bytes_to_uint64, hash_numpy
from vernamveil._types import _HAS_C_MODULE, _HAS_NUMPY, np


class TestHashUtils(unittest.TestCase):
    """Unit tests for the has_utils.py utilities."""

    def _get_hash_method_for_test(self, hash_name):
        """Returns the appropriate hashlib function for the given hash_name."""
        if hash_name == "sha256":
            return hashlib.sha256
        elif hash_name == "blake2b":
            return hashlib.blake2b
        elif hash_name == "blake3":
            return blake3
        else:
            raise ValueError(f"Unsupported hash_name '{hash_name}'.")

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_hash_numpy_correctness(self):
        """Check that hash_numpy output matches expected hash values for a range of inputs, for both fold types."""
        checks = [False]
        hashes = ["blake2b", "sha256"]
        if _HAS_C_MODULE:
            checks.append(True)
            hashes.append("blake3")
        for has_c in checks:
            for hash_name in hashes:
                if hash_name == "blake3" and not has_c:
                    # Skip blake3 if C module is not available, as it requires the CFFI implementation
                    continue
                for fold_type in ("full", "view"):
                    with self.subTest(
                        _HAS_C_MODULE=has_c, hash_name=hash_name, fold_type=fold_type
                    ):
                        print(
                            f"_HAS_C_MODULE={has_c}, hash_name={hash_name}, fold_type={fold_type}"
                        )
                        with patch("vernamveil._hash_utils._HAS_C_MODULE", has_c):
                            seed = secrets.token_bytes(64)
                            i = np.arange(1000, dtype=np.uint64)

                            output = fold_bytes_to_uint64(
                                hash_numpy(i, seed, hash_name), fold_type=fold_type
                            )

                            i_bytes = i.tobytes()
                            method = self._get_hash_method_for_test(hash_name)

                            def get_digest(j):
                                if hash_name == "blake3":
                                    hasher = method(key=seed)
                                else:
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
        hashes = ["blake2b", "sha256"]
        if _HAS_C_MODULE:
            checks.append(True)
            hashes.append(f"blake3")
        for has_c in checks:
            for hash_name in hashes:
                if hash_name == "blake3" and not has_c:
                    # Skip blake3 if C module is not available, as it requires the CFFI implementation
                    continue
                for fold_type in ("full", "view"):
                    with self.subTest(
                        _HAS_C_MODULE=has_c, hash_name=hash_name, fold_type=fold_type
                    ):
                        print(
                            f"_HAS_C_MODULE={has_c}, hash_name={hash_name}, fold_type={fold_type}"
                        )
                        with patch("vernamveil._hash_utils._HAS_C_MODULE", has_c):
                            i = np.arange(1000, dtype=np.uint64)

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


if __name__ == "__main__":
    unittest.main()
