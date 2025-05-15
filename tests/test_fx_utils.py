import tempfile
import unittest
import warnings
from pathlib import Path

from vernamveil._cypher import _HAS_NUMPY
from vernamveil._fx_utils import FX, check_fx_sanity, generate_default_fx, load_fx_from_file

try:
    import numpy as np
except ImportError:
    pass


class TestFxUtils(unittest.TestCase):
    """Unit tests for the fx_utils.py utilities."""

    def setUp(self):
        """Set up common test parameters."""
        self.seed = b"testseed"
        self.num_samples = 100

    def test_generate_default_fx_scalar(self):
        """Test that generate_default_fx returns a valid scalar function."""
        fx = generate_default_fx(vectorise=False)
        for i in range(10):
            out = fx(i, self.seed)
            self.assertIsInstance(out, bytes)

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_generate_default_fx_vectorised(self):
        """Test that generate_default_fx returns a valid vectorised function."""
        fx = generate_default_fx(vectorise=True)
        arr = np.arange(10, dtype=np.uint64)
        out = fx(arr, self.seed)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (arr.shape[0], fx.block_size))

    def test_check_fx_sanity_constant_function(self):
        """Test check_fx_sanity fails and warns for a constant fx."""

        def keystream_fn(i, seed):
            v = 42
            v &= 0xFFFFFFFFFFFFFFFF
            return v.to_bytes(8, "big")

        fx = FX(keystream_fn, 8, vectorise=False)

        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("constant" in str(warn.message) for warn in w))

    def test_check_fx_sanity_seed_insensitive(self):
        """Test check_fx_sanity fails and warns for a seed-insensitive fx."""

        def keystream_fn(i, seed):
            v = 42
            v &= 0xFFFFFFFFFFFFFFFF
            return v.to_bytes(8, "big")

        fx = FX(keystream_fn, 8, vectorise=False)

        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("does not depend on seed" in str(warn.message) for warn in w))

    def test_check_fx_sanity_uniformity_violation(self):
        """Test check_fx_sanity fails and warns for fx that is heavily biased."""

        def keystream_fn(i, seed):
            return b"x00" if i < self.num_samples - 1 else b"x01"  # Almost always 0

        fx = FX(keystream_fn, 2, vectorise=False)

        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("biased" in str(warn.message) for warn in w))

    def test_check_default_fx_sanity_scalar(self):
        """Test check_default_fx_sanity passes for a good scalar default fx."""
        fx = generate_default_fx(vectorise=False)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertTrue(passed)
        self.assertEqual(len(w), 0)

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_check_default_fx_sanity_vectorised(self):
        """Test check_default_fx_sanity passes for a good vectorised default fx."""
        fx = generate_default_fx(vectorise=True)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertTrue(passed)
        self.assertEqual(len(w), 0)

    def test_load_fx_from_file_scalar(self):
        """Test that a scalar fx function can be saved to a file and loaded back using load_fx_from_file."""
        fx_obj = generate_default_fx(vectorise=False)

        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write(fx_obj.source_code)
            tmp_path = Path(tmp.name)

        try:
            fx_loaded = load_fx_from_file(tmp_path)
            self.assertTrue(callable(fx_loaded))
            self.assertTrue(isinstance(fx_loaded(1, bytes()), bytes))
        finally:
            tmp_path.unlink()

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_load_fx_from_file_vectorised(self):
        """Test that a vectorised fx function can be saved to a file and loaded back using load_fx_from_file."""
        fx_obj = generate_default_fx(vectorise=True)

        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write(fx_obj.source_code)
            tmp_path = Path(tmp.name)

        try:
            fx_loaded = load_fx_from_file(tmp_path)
            self.assertTrue(callable(fx_loaded))
            self.assertTrue(
                isinstance(fx_loaded(np.arange(1, 10, dtype=np.uint64), bytes()), np.ndarray)
            )
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
