import os
import shutil
import sys  # noqa: F401
import tempfile
import unittest
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from vernamveil.cli import main


class TestVernamVeilCLI(unittest.TestCase):
    """Unit tests for the VernamVeil CLI covering all supported scenarios."""

    def setUp(self):
        """Create a temporary directory for each test and an input file."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir_path = Path(self.temp_dir)
        self.infile = self._create_input()
        self.encfile = self.temp_dir_path / "output.enc"
        self.outfile = self.temp_dir_path / "output.txt"
        self.fx_code = """
def fx(i, seed, bound):
    h = i + 1
    return h % bound if bound is not None else h
"""
        self.fx_strong_code = """
import hmac

def fx(i, seed, bound):
    h = int.from_bytes(hmac.new(seed, i.to_bytes(8, "big"), digestmod="blake2b").digest(), "big")
    return h % bound if bound is not None else h
"""

    def tearDown(self):
        """Remove the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def _write_file(self, filename, data, mode="wb"):
        """Helper to write binary or text data to a file in temp_dir."""
        path = self.temp_dir_path / filename
        with path.open(mode) as f:
            f.write(data)
        return path

    def _create_input(self, content=b"test data"):
        """Create an input file with given content."""
        return self._write_file("input.txt", content)

    def _create_fx(self, code=None):
        """Create a fx.py file with given code."""
        code = code if code is not None else self.fx_code
        return self._write_file("fx.py", code.encode(), mode="wb")

    def _create_seed(self, content=b"myseed"):
        """Create a seed.bin file with given content."""
        return self._write_file("seed.bin", content)

    @contextmanager
    def _in_tempdir(self):
        """Context manager to run code in the temp dir, restoring cwd and rethrowing exceptions."""
        cwd = Path.cwd()
        try:
            os.chdir(self.temp_dir_path)
            yield
        finally:
            if Path.cwd() != cwd:
                os.chdir(cwd)

    def _encode(self, infile, outfile, fx_file=None, seed_file=None, extra_args=None):
        """Helper to run the encode CLI command with optional fx and seed files."""
        args = ["encode", "--infile", infile, "--outfile", outfile]
        if fx_file:
            args += ["--fx-file", fx_file]
        if seed_file:
            args += ["--seed-file", seed_file]
        if extra_args:
            args += extra_args

        args = [str(arg) for arg in args]
        if not any(arg.endswith("vectorise") for arg in args):
            args += ["--no-vectorise"]
        main(args)

    def _decode(self, infile, outfile, fx_file, seed_file, extra_args=None):
        """Helper to run the decode CLI command with required fx and seed files."""
        args = [
            "decode",
            "--infile",
            infile,
            "--outfile",
            outfile,
            "--fx-file",
            fx_file,
            "--seed-file",
            seed_file,
        ]
        if extra_args:
            args += extra_args

        args = [str(arg) for arg in args]
        if not any(arg.endswith("vectorise") for arg in args):
            args += ["--no-vectorise"]
        main(args)

    def test_encode_generates_fx_and_seed(self):
        """Test encoding with auto-generated fx and seed."""
        with self._in_tempdir():
            self._encode(self.infile, self.encfile)
        self.assertTrue(self.encfile.exists())
        self.assertTrue((self.temp_dir_path / "fx.py").exists())
        self.assertTrue((self.temp_dir_path / "seed.bin").exists())

    def test_encode_with_custom_fx(self):
        """Test encoding with a user-supplied fx file."""
        with self._in_tempdir():
            self._encode(self.infile, self.encfile, fx_file=self._create_fx())
        self.assertTrue(self.encfile.exists())
        self.assertTrue((self.temp_dir_path / "seed.bin").exists())

    def test_encode_with_custom_seed(self):
        """Test encoding with a user-supplied seed file."""
        with self._in_tempdir():
            self._encode(self.infile, self.encfile, seed_file=self._create_seed())
        self.assertTrue(self.encfile.exists())
        self.assertTrue((self.temp_dir_path / "fx.py").exists())

    def test_encode_with_custom_fx_and_seed(self):
        """Test encoding with both custom fx and seed files."""
        with self._in_tempdir():
            self._encode(
                self.infile, self.encfile, fx_file=self._create_fx(), seed_file=self._create_seed()
            )
        self.assertTrue(self.encfile.exists())

    def test_decode_requires_fx_and_seed(self):
        """Test decoding requires both fx and seed files."""
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(self.infile, self.encfile, fx_file=fx_file, seed_file=seed_file)
            self._decode(self.encfile, self.outfile, fx_file, seed_file)
        self.assertTrue(self.outfile.exists())

    def test_encode_with_check_sanity(self):
        """Test encoding with sanity check for fx and seed enabled."""
        with self._in_tempdir():
            self._encode(self.infile, self.encfile, extra_args=["--check-sanity"])
        self.assertTrue(self.encfile.exists())

    def test_encode_with_check_sanity_fails_fx(self):
        """Test that sanity check fails if fx does not depend on seed."""
        stderr = StringIO()
        with self._in_tempdir(), patch("sys.stderr", stderr), self.assertRaises(SystemExit):
            self._encode(
                self.infile,
                self.encfile,
                fx_file=self._create_fx(),
                extra_args=["--check-sanity"],
            )
        self.assertIn("Error: fx sanity check failed.", stderr.getvalue())

    def test_encode_with_check_sanity_fails_seed(self):
        """Test that sanity check fails if the seed is too short."""
        short_seed = b"short"
        seed_file = self._create_seed(content=short_seed)
        stderr = StringIO()
        with self._in_tempdir(), patch("sys.stderr", stderr), self.assertRaises(SystemExit):
            self._encode(
                self.infile,
                self.encfile,
                seed_file=seed_file,
                extra_args=["--check-sanity"],
            )
        self.assertIn(
            "Error: Seed is too short. It must be at least 16 bytes for security.",
            stderr.getvalue(),
        )

    def _assert_encode_refuses_to_overwrite(self, file_path):
        expected_error = f"Error: {file_path.name} already exists. Refusing to overwrite."
        with file_path.open("rb") as f:
            expected_content = f.read()
        with self._in_tempdir():
            stderr = StringIO()
            with patch("sys.stderr", stderr), self.assertRaises(SystemExit):
                self._encode(self.infile, self.encfile)
            self.assertIn(expected_error, stderr.getvalue())
            with file_path.open("rb") as f:
                self.assertEqual(f.read(), expected_content)

    def test_encode_refuses_to_overwrite_existing_fx(self):
        """Test that encoding refuses to overwrite an existing fx.py file."""
        fx_path = self._create_fx()
        self._assert_encode_refuses_to_overwrite(fx_path)

    def test_encode_refuses_to_overwrite_existing_seed(self):
        """Test that encoding refuses to overwrite an existing seed.bin file."""
        seed_path = self._create_seed()
        self._assert_encode_refuses_to_overwrite(seed_path)

    def test_encode_refuses_to_overwrite_existing_output(self):
        """Test that encoding refuses to overwrite an existing output file."""
        out_path = self._write_file("output.enc", b"original data")
        self._assert_encode_refuses_to_overwrite(out_path)

    def test_decode_refuses_to_overwrite_existing_output(self):
        """Test that decoding refuses to overwrite an existing output file."""
        self._write_file("output.txt", b"original plain")
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            stderr = StringIO()
            with patch("sys.stderr", stderr), self.assertRaises(SystemExit):
                self._decode(self.encfile, self.outfile, fx_file, seed_file)
            self.assertIn(
                "Error: output.txt already exists. Refusing to overwrite.", stderr.getvalue()
            )
            # Ensure output file was not modified
            with (self.temp_dir_path / "output.txt").open("rb") as f:
                self.assertEqual(f.read(), b"original plain")

    def test_verbosity_warning(self):
        """Test that warnings and errors are printed with --verbosity warning (default)."""
        with self._in_tempdir():
            # First, run a successful encode to trigger a warning (fx.py generated)
            stderr = StringIO()
            with patch("sys.stderr", stderr):
                self._encode(self.infile, self.encfile, extra_args=["--verbosity", "warning"])
            self.assertIn("Warning:", stderr.getvalue())

            # Now, run a failed encode to trigger an error (output.enc exists)
            stderr = StringIO()
            with patch("sys.stderr", stderr), self.assertRaises(SystemExit):
                self._encode(self.infile, self.encfile, extra_args=["--verbosity", "warning"])
            self.assertIn("Error:", stderr.getvalue())

    def test_verbosity_error(self):
        """Test that only errors are printed with --verbosity error and that warnings are not."""
        with self._in_tempdir():
            # First, run a successful encode to check warning is NOT present
            stderr = StringIO()
            with patch("sys.stderr", stderr):
                self._encode(self.infile, self.encfile, extra_args=["--verbosity", "error"])
            self.assertNotIn("Warning:", stderr.getvalue())

            # Now, run a failed encode to trigger an error (output.enc exists)
            stderr = StringIO()
            with patch("sys.stderr", stderr), self.assertRaises(SystemExit):
                self._encode(self.infile, self.encfile, extra_args=["--verbosity", "error"])
            self.assertIn("Error:", stderr.getvalue())
            self.assertNotIn("Warning:", stderr.getvalue())

    def test_verbosity_none(self):
        """Test that nothing is printed with --verbosity none, for both warnings and errors."""
        with self._in_tempdir():
            # Test warning (successful encode, should print nothing)
            stderr = StringIO()
            with patch("sys.stderr", stderr):
                self._encode(self.infile, self.encfile, extra_args=["--verbosity", "none"])
            self.assertEqual(stderr.getvalue(), "")

            # Test error (attempt to overwrite output.enc, should print nothing)
            stderr = StringIO()
            with patch("sys.stderr", stderr), self.assertRaises(SystemExit):
                self._encode(self.infile, self.encfile, extra_args=["--verbosity", "none"])
            self.assertEqual(stderr.getvalue(), "")

    def test_decode_fails_with_mismatched_parameters(self):
        """Test that decoding fails with mismatched encoding parameters."""
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            # Encode with a specific chunk size
            self._encode(
                self.infile,
                self.encfile,
                fx_file=fx_file,
                seed_file=seed_file,
                extra_args=["--chunk-size", "2048"],
            )
            # Attempt to decode with a different chunk size
            with self.assertRaises(ValueError) as context:
                self._decode(
                    self.encfile,
                    self.outfile,
                    fx_file=fx_file,
                    seed_file=seed_file,
                    extra_args=["--chunk-size", "1024"],
                )
            self.assertEqual(str(context.exception), "Authentication failed: MAC tag mismatch.")


if __name__ == "__main__":
    unittest.main()
