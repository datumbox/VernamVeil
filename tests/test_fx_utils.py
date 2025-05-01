import tempfile
import unittest
import warnings
from pathlib import Path

from vernamveil.cypher import _HAS_NUMPY
from vernamveil.fx_utils import generate_default_fx, load_fx_from_file, check_fx_sanity

try:
    import numpy as np
except ImportError:
    pass


class TestFxUtils(unittest.TestCase):
    """Unit tests for the fx_utils.py utilities."""

    def setUp(self):
        """Set up common test parameters."""
        self.seed = b"testseed"
        self.bound = 256
        self.num_samples = 100

    def test_generate_default_fx_scalar(self):
        """Test that generate_default_fx returns a valid scalar function."""
        fx = generate_default_fx(5, max_weight=1000, vectorise=False)
        for i in range(10):
            out = fx(i, self.seed, self.bound)
            self.assertIsInstance(out, int)
            self.assertGreaterEqual(out, 0)
            self.assertLess(out, self.bound)

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_generate_default_fx_vectorised(self):
        """Test that generate_default_fx returns a valid vectorised function."""
        fx = generate_default_fx(5, max_weight=1000, vectorise=True)
        arr = np.arange(10, dtype=np.uint64)
        out = fx(arr, self.seed, self.bound)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, arr.shape)
        self.assertTrue((out >= 0).all() and (out < self.bound).all())

    def test_check_fx_sanity_constant_function(self):
        """Test check_fx_sanity fails and warns for a constant fx."""

        def fx(i, seed, bound):
            return 42

        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.bound, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(
            any("constant" in str(warn.message) or "low-entropy" in str(warn.message) for warn in w)
        )

    def test_check_fx_sanity_seed_insensitive(self):
        """Test check_fx_sanity fails and warns for a seed-insensitive fx."""

        def fx(i, seed, bound):
            return i % bound

        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.bound, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("seed" in str(warn.message) for warn in w))

    def test_check_fx_sanity_bound_violation(self):
        """Test check_fx_sanity fails and warns for fx that violates the bound."""

        def fx(i, seed, bound):
            return bound + 1  # Always out of bounds

        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.bound, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("bound" in str(warn.message) for warn in w))

    def test_check_fx_sanity_uniformity_violation(self):
        """Test check_fx_sanity fails and warns for fx that is heavily biased."""

        def fx(i, seed, bound):
            return 0 if i < self.num_samples - 1 else 1  # Almost always 0

        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.bound, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("biased" in str(warn.message) for warn in w))

    def test_check_default_fx_sanity_scalar(self):
        """Test check_default_fx_sanity passes for a good scalar default fx."""
        fx = generate_default_fx(4, max_weight=1000, vectorise=False)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.bound, self.num_samples)
        self.assertTrue(passed)
        self.assertEqual(len(w), 0)

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_check_default_fx_sanity_vectorised(self):
        """Test check_default_fx_sanity passes for a good vectorised default fx."""
        fx = generate_default_fx(4, vectorise=True)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.bound, self.num_samples)
        self.assertTrue(passed)
        self.assertEqual(len(w), 0)

    def test_load_fx_from_file_scalar(self):
        """Test that a scalar fx function can be saved to a file and loaded back using load_fx_from_file."""
        fx_obj = generate_default_fx(3, vectorise=False)
        fx_code = "import hashlib\n" + fx_obj._source_code

        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write(fx_code)
            tmp_path = Path(tmp.name)

        try:
            fx_loaded = load_fx_from_file(tmp_path)
            self.assertTrue(callable(fx_loaded))
            self.assertTrue(isinstance(fx_loaded(1, bytes(), self.bound), int))
        finally:
            tmp_path.unlink()

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_load_fx_from_file_vectorised(self):
        """Test that a vectorised fx function can be saved to a file and loaded back using load_fx_from_file."""
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy not available")

        fx_obj = generate_default_fx(3, vectorise=True)
        fx_code = "from vernamveil import hash_numpy\nimport numpy as np\n" + fx_obj._source_code

        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write(fx_code)
            tmp_path = Path(tmp.name)

        try:
            fx_loaded = load_fx_from_file(tmp_path)
            self.assertTrue(callable(fx_loaded))
            self.assertTrue(
                isinstance(
                    fx_loaded(np.arange(1, 10, dtype=np.uint64), bytes(), self.bound), np.ndarray
                )
            )
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
