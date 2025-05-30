import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from vernamveil._fx_utils import FX, OTPFX, check_fx_sanity, generate_default_fx, load_fx_from_file
from vernamveil._types import _HAS_NUMPY
from vernamveil._vernamveil import VernamVeil

try:
    import numpy as np
except ImportError:
    pass


class TestFxUtils(unittest.TestCase):
    """Unit tests for the fx_utils.py utilities."""

    def setUp(self):
        """Set up common test parameters."""
        self.seed = b"testseed"
        self.num_samples = 1000

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

    def test_type_check_failure(self):
        """Test check_fx_sanity warns on type check failure (wrong type)."""

        def keystream_fn(i, seed):
            return 42

        fx = FX(keystream_fn, block_size=8, vectorise=False)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("not bytes" in str(warn.message) for warn in w))

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_shape_check_failure_vectorised(self):
        """Test check_fx_sanity warns on shape/type check failure for vectorised output."""

        def keystream_fn(i, seed):
            # Wrong shape: should be (num_samples, block_size)
            return np.zeros((self.num_samples, 1), dtype=np.uint8)

        fx = FX(keystream_fn, block_size=8, vectorise=True)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("not a 2D NumPy array" in str(warn.message) for warn in w))

    def test_output_size_inconsistency(self):
        """Test check_fx_sanity warns on output size inconsistency."""

        def keystream_fn(i, seed):
            return b"x" * (8 if i % 2 == 0 else 7)  # Odd indices have wrong length

        fx = FX(keystream_fn, block_size=8, vectorise=False)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("has length" in str(warn.message) for warn in w))

    def test_non_constant_output_detection(self):
        """Test check_fx_sanity warns if output is constant or low-entropy."""

        def keystream_fn(i, seed):
            return b"x" * 8

        fx = FX(keystream_fn, block_size=8, vectorise=False)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("constant" in str(warn.message) in str(warn.message) for warn in w))

    def test_seed_insensitivity_detection(self):
        """Test check_fx_sanity warns if output does not depend on seed."""

        def keystream_fn(i, seed):
            return (i.to_bytes(8, "big") * 1)[:8]

        fx = FX(keystream_fn, block_size=8, vectorise=False)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("does not depend on seed" in str(warn.message) for warn in w))

    def test_uniformity_biased(self):
        """Test check_fx_sanity warns if output is heavily biased."""

        def keystream_fn(i, seed):
            return b"\x00" * 8 if i < self.num_samples - 1 else b"\x01" * 8

        fx = FX(keystream_fn, block_size=8, vectorise=False)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("biased" in str(warn.message) for warn in w))

    def test_uniformity_missing_value(self):
        """Test check_fx_sanity warns if a byte value never appears."""

        def keystream_fn(i, seed):
            return b"\x00" * 8

        fx = FX(keystream_fn, block_size=8, vectorise=False)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("never appears" in str(warn.message) for warn in w))

    def test_avalanche_effect_weak(self):
        """Test check_fx_sanity warns if avalanche effect is weak."""

        def keystream_fn(i, seed):
            # Only the LSB changes, rest is constant
            return (b"\x00" * 7) + bytes([i & 1])

        fx = FX(keystream_fn, block_size=8, vectorise=False)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, self.num_samples)
        self.assertFalse(passed)
        self.assertTrue(any("Avalanche effect weak" in str(warn.message) for warn in w))

    def test_check_default_fx_sanity_scalar(self):
        """Test check_fx_sanity passes for a good scalar default fx."""
        fx = generate_default_fx(vectorise=False)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, 10000)
        self.assertTrue(passed)
        self.assertEqual(len(w), 0)

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_check_default_fx_sanity_vectorised(self):
        """Test check_fx_sanity passes for a good vectorised default fx."""
        fx = generate_default_fx(vectorise=True)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, 10000)
        self.assertTrue(passed)
        self.assertEqual(len(w), 0)

    def test_check_otpfx_sanity(self):
        """Test check_fx_sanity passes for a valid OTPFX instance."""
        block_size = 64
        num_blocks = 21000
        keystream = [VernamVeil.get_initial_seed(num_bytes=block_size) for i in range(num_blocks)]
        fx = OTPFX(keystream, block_size, vectorise=False)
        with warnings.catch_warnings(record=True) as w:
            passed = check_fx_sanity(fx, self.seed, 10000)
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

    def test_load_fx_from_file_file_not_found(self):
        """Test load_fx_from_file raises FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            load_fx_from_file("nonexistent_fx_file.py")

    def test_load_fx_from_file_invalid_python(self):
        """Test load_fx_from_file raises SyntaxError for invalid Python code."""
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write("def keystream_fn(:\n    pass\n")  # Invalid syntax
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(SyntaxError):
                load_fx_from_file(tmp_path)
        finally:
            tmp_path.unlink()

    def test_load_fx_from_file_missing_fx(self):
        """Test load_fx_from_file raises ImportError if 'fx' is missing."""
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write("# no fx here\n")
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(ImportError):
                load_fx_from_file(tmp_path)
        finally:
            tmp_path.unlink()

    def test_load_fx_from_file_fx_wrong_type(self):
        """Test load_fx_from_file raises TypeError if 'fx' is not an FX instance."""
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write("fx = 123\n")
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(TypeError):
                load_fx_from_file(tmp_path)
        finally:
            tmp_path.unlink()

    def test_load_fx_from_file_valid_minimal_fx(self):
        """Test load_fx_from_file works with a minimal valid FX instance."""
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write(
                "from vernamveil._fx_utils import FX\n"
                "def keystream_fn(i, seed):\n"
                "    return b'12345678'\n"
                "fx = FX(keystream_fn, block_size=8, vectorise=False)\n"
            )
            tmp_path = Path(tmp.name)
        try:
            fx_loaded = load_fx_from_file(tmp_path)
            self.assertTrue(callable(fx_loaded))
            self.assertEqual(fx_loaded(1, b"seed"), b"12345678")
        finally:
            tmp_path.unlink()

    def test_scalar_fx_call_and_attributes(self):
        """Test FX scalar mode: __call__, block_size, vectorise, and source_code."""

        def keystream_fn(i, seed):
            return b"abcde123"

        fx = FX(
            keystream_fn,
            block_size=8,
            vectorise=False,
            source_code="def keystream_fn(i, seed): ...",
        )
        self.assertEqual(fx.block_size, 8)
        self.assertFalse(fx.vectorise)
        self.assertEqual(fx.source_code, "def keystream_fn(i, seed): ...")
        result = fx(1, b"seed")
        self.assertEqual(result, b"abcde123")

    @unittest.skipUnless(_HAS_NUMPY, "NumPy not available")
    def test_vectorised_fx_call_and_attributes(self):
        """Test FX vectorised mode: __call__, block_size, vectorise, and source_code."""

        def keystream_fn(i, seed):
            # i is a numpy array
            return np.tile(np.arange(8, dtype=np.uint8), (i.shape[0], 1))

        fx = FX(
            keystream_fn, block_size=8, vectorise=True, source_code="def keystream_fn(i, seed): ..."
        )
        self.assertEqual(fx.block_size, 8)
        self.assertTrue(fx.vectorise)
        self.assertEqual(fx.source_code, "def keystream_fn(i, seed): ...")
        arr = np.arange(5, dtype=np.uint64)
        result = fx(arr, b"seed")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 8))
        np.testing.assert_array_equal(result, np.tile(np.arange(8, dtype=np.uint8), (5, 1)))

    def test_vectorise_flag_and_numpy_requirement(self):
        """Test that FX raises ValueError if vectorise=True but numpy is not available."""

        def keystream_fn(i, seed):
            return b"x" * 8

        with patch("vernamveil._fx_utils._HAS_NUMPY", False):
            with self.assertRaises(ValueError):
                FX(keystream_fn, block_size=8, vectorise=True)

    def test_warns_if_numpy_present_but_vectorise_false(self):
        """Test that FX warns if numpy is present but vectorise is False."""

        def keystream_fn(i, seed):
            return b"x" * 8

        with patch("vernamveil._fx_utils._HAS_NUMPY", True):
            with warnings.catch_warnings(record=True) as w:
                FX(keystream_fn, block_size=8, vectorise=False)
            self.assertTrue(any("NumPy will not be used" in str(warn.message) for warn in w))

    def test_warns_if_numpy_present_but_c_module_missing(self):
        """Test that FX warns if NumPy is present but the C module is not available."""

        def keystream_fn(i, seed):
            import numpy as np

            return np.zeros((1, 8), dtype=np.uint8)

        with (
            patch("vernamveil._fx_utils._HAS_NUMPY", True),
            patch("vernamveil._fx_utils._HAS_C_MODULE", False),
        ):
            with warnings.catch_warnings(record=True) as w:
                FX(keystream_fn, block_size=8, vectorise=True)
            self.assertTrue(any("C module is not available" in str(warn.message) for warn in w))

    def test_otp_fx_source_code_roundtrip(self):
        """Test that OTPFX source code can be saved and loaded, preserving bytes."""

        test_keystream = [b"42", b"99", b"12", b"34", b"56", b"71", b"01"]
        fx = OTPFX(test_keystream, 2, False)
        source_code = fx.source_code

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fx_source.py"
            with open(path, "w") as f:
                f.write(source_code)
            loaded_fx: OTPFX = load_fx_from_file(str(path))
            self.assertEqual(list(loaded_fx.keystream), test_keystream)

    def test_otpfx_keystream_exhaustion(self):
        """Test that OTPFX raises IndexError when the keystream is exhausted."""
        test_keystream = [b"42", b"99"]
        fx = OTPFX(test_keystream, 2, False)
        # Consume all available keystream values
        fx(0, b"seed")
        fx(1, b"seed")
        # Next call should raise IndexError
        with self.assertRaises(IndexError):
            fx(2, b"seed")

    def test_otpfx_large_keystream_roundtrip(self):
        """Test that OTPFX with a large keystream can be saved and loaded, preserving all bytes."""
        # Create a large keystream
        block_size = 64
        num_blocks = 100000
        test_keystream = [bytes([i % 256] * block_size) for i in range(num_blocks)]
        fx = OTPFX(test_keystream, block_size, False)
        source_code = fx.source_code

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fx_large_source.py"
            with open(path, "w") as f:
                f.write(source_code)
            loaded_fx: OTPFX = load_fx_from_file(path)
            # Check that the loaded keystream matches the original
            self.assertEqual(list(loaded_fx.keystream), test_keystream)


if __name__ == "__main__":
    unittest.main()
