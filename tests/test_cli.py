import os
import shutil
import sys  # noqa: F401
import tempfile
import unittest
import unittest.mock
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

from vernamveil.cli import main


class TestVernamVeilCLI(unittest.TestCase):
    """Unit tests for the VernamVeil CLI covering all supported scenarios."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir_path = Path(self.temp_dir)
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

    @contextmanager
    def assertDoesNotRaise(self, exc_type):
        """Context manager to assert that no exception is raised."""
        try:
            yield
        except exc_type as e:
            self.fail(f"Unexpected exception raised: {e}")

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
        infile = self._create_input()
        outfile = self.temp_dir_path / "output.enc"
        with self._in_tempdir():
            self._encode(infile, outfile)
        self.assertTrue(outfile.exists())
        self.assertTrue((self.temp_dir_path / "fx.py").exists())
        self.assertTrue((self.temp_dir_path / "seed.bin").exists())

    def test_encode_with_custom_fx(self):
        """Test encoding with a user-supplied fx file."""
        infile = self._create_input()
        outfile = self.temp_dir_path / "output.enc"
        fx_file = self._create_fx()
        with self._in_tempdir():
            self._encode(infile, outfile, fx_file=fx_file)
        self.assertTrue(outfile.exists())
        self.assertTrue((self.temp_dir_path / "seed.bin").exists())

    def test_encode_with_custom_seed(self):
        """Test encoding with a user-supplied seed file."""
        infile = self._create_input()
        outfile = self.temp_dir_path / "output.enc"
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(infile, outfile, seed_file=seed_file)
        self.assertTrue(outfile.exists())
        self.assertTrue((self.temp_dir_path / "fx.py").exists())

    def test_encode_with_custom_fx_and_seed(self):
        """Test encoding with both custom fx and seed files."""
        infile = self._create_input()
        outfile = self.temp_dir_path / "output.enc"
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(infile, outfile, fx_file=fx_file, seed_file=seed_file)
        self.assertTrue(outfile.exists())

    def test_decode_requires_fx_and_seed(self):
        """Test decoding requires both fx and seed files."""
        infile = self._create_input()
        encfile = self.temp_dir_path / "output.enc"
        outfile = self.temp_dir_path / "output.txt"
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(infile, encfile, fx_file=fx_file, seed_file=seed_file)
            self._decode(encfile, outfile, fx_file, seed_file)
        self.assertTrue(outfile.exists())

    def test_encode_with_check_fx_sanity(self):
        """Test encoding with fx sanity check enabled."""
        infile = self._create_input()
        outfile = self.temp_dir_path / "output.enc"
        with self._in_tempdir():
            self._encode(infile, outfile, extra_args=["--check-fx-sanity"])
        self.assertTrue(outfile.exists())

    def test_encode_with_check_fx_sanity_fails(self):
        """Test that fx sanity check fails if fx does not depend on seed."""
        infile = self._create_input()
        encfile = self.temp_dir_path / "output.enc"
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as cm:
                    self._encode(
                        infile,
                        encfile,
                        fx_file=fx_file,
                        seed_file=seed_file,
                        extra_args=["--check-fx-sanity", "--verbosity", "error"],
                    )
            self.assertNotEqual(cm.exception.code, 0)
            err = stderr.getvalue()
            self.assertIn("Error: fx sanity check failed.", err)

    def test_encode_refuses_to_overwrite_existing_fx(self):
        """Test that encoding refuses to overwrite an existing fx.py file."""
        infile = self._create_input()
        outfile = self.temp_dir_path / "output.enc"
        # Pre-create fx.py
        self._create_fx()
        with self._in_tempdir():
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as cm:
                    self._encode(infile, outfile, extra_args=["--verbosity", "error"])
            self.assertNotEqual(cm.exception.code, 0)
            err = stderr.getvalue()
            self.assertIn("Error: fx.py already exists. Refusing to overwrite.", err)
            # Ensure fx.py was not modified (content unchanged)
            with (self.temp_dir_path / "fx.py").open("rb") as f:
                self.assertEqual(f.read(), self.fx_code.encode())

    def test_encode_refuses_to_overwrite_existing_seed(self):
        """Test that encoding refuses to overwrite an existing seed.bin file."""
        infile = self._create_input()
        outfile = self.temp_dir_path / "output.enc"
        # Pre-create seed.bin
        self._create_seed()
        with self._in_tempdir():
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as cm:
                    self._encode(infile, outfile, extra_args=["--verbosity", "error"])
            self.assertNotEqual(cm.exception.code, 0)
            err = stderr.getvalue()
            self.assertIn("Error: seed.bin already exists. Refusing to overwrite.", err)
            # Ensure seed.bin was not modified (content unchanged)
            with (self.temp_dir_path / "seed.bin").open("rb") as f:
                self.assertEqual(f.read(), b"myseed")

    def test_encode_refuses_to_overwrite_existing_output(self):
        """Test that encoding refuses to overwrite an existing output file."""
        infile = self._create_input()
        outfile = self._write_file("output.enc", b"original data")
        with self._in_tempdir():
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as cm:
                    self._encode(infile, outfile, extra_args=["--verbosity", "error"])
            self.assertNotEqual(cm.exception.code, 0)
            err = stderr.getvalue()
            self.assertIn("Error: output.enc already exists. Refusing to overwrite.", err)
            # Ensure output file was not modified
            with (self.temp_dir_path / "output.enc").open("rb") as f:
                self.assertEqual(f.read(), b"original data")

    def test_decode_refuses_to_overwrite_existing_output(self):
        """Test that decoding refuses to overwrite an existing output file."""
        infile = self._create_input()
        encfile = self.temp_dir_path / "output.enc"
        outfile = self._write_file("output.txt", b"original plain")
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(infile, encfile, fx_file=fx_file, seed_file=seed_file)
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as cm:
                    self._decode(
                        encfile, outfile, fx_file, seed_file, extra_args=["--verbosity", "error"]
                    )
            self.assertNotEqual(cm.exception.code, 0)
            err = stderr.getvalue()
            self.assertIn("Error: output.txt already exists. Refusing to overwrite.", err)
            # Ensure output file was not modified
            with (self.temp_dir_path / "output.txt").open("rb") as f:
                self.assertEqual(f.read(), b"original plain")

    def test_verbosity_warning(self):
        """Test that warnings and errors are printed with --verbosity warning (default)."""
        infile = self._create_input()
        outfile = self.temp_dir_path / "output.enc"
        # Pre-create fx.py to trigger warning/error
        self._create_fx()
        with self._in_tempdir():
            # First, run a successful encode to trigger a warning (fx.py generated)
            # Remove fx.py so encode will generate it and print a warning
            os.remove("fx.py")
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertDoesNotRaise(SystemExit) as cm:
                    self._encode(infile, outfile, extra_args=["--verbosity", "warning"])
            out = stderr.getvalue()
            self.assertIn("Warning:", out)

            # Now, run a failed encode to trigger an error (output.enc exists)
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as cm:
                    self._encode(infile, outfile, extra_args=["--verbosity", "warning"])
            self.assertNotEqual(cm.exception.code, 0)
            out = stderr.getvalue()
            self.assertIn("Error: output.enc already exists. Refusing to overwrite.", out)

    def test_verbosity_error(self):
        """Test that only errors are printed with --verbosity error and that warnings appear with warning level."""
        infile = self._create_input()
        outfile = self.temp_dir_path / "output.enc"
        fx_path = self.temp_dir_path / "fx.py"
        seed_path = self.temp_dir_path / "seed.bin"
        if fx_path.exists():
            fx_path.unlink()
        if seed_path.exists():
            seed_path.unlink()
        with self._in_tempdir():
            # First, run with --verbosity warning and check warning is present
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertDoesNotRaise(SystemExit) as cm:
                    self._encode(infile, outfile, extra_args=["--verbosity", "warning"])
            out = stderr.getvalue()
            self.assertIn("Warning:", out)
            # Now, run with --verbosity error and check warning is NOT present
            os.remove("output.enc")
            if fx_path.exists():
                fx_path.unlink()
            if seed_path.exists():
                seed_path.unlink()
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertDoesNotRaise(SystemExit) as cm:
                    self._encode(infile, outfile, extra_args=["--verbosity", "error"])
            out = stderr.getvalue()
            self.assertNotIn("Warning:", out)
            # Also check that error is printed if we try to overwrite
            os.remove("output.enc")
            if fx_path.exists():
                fx_path.unlink()
            if seed_path.exists():
                seed_path.unlink()
            # Create output.enc to trigger the error
            self._write_file("output.enc", b"original data")
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as cm:
                    self._encode(infile, outfile, extra_args=["--verbosity", "error"])
            self.assertNotEqual(cm.exception.code, 0)
            out = stderr.getvalue()
            self.assertIn("Error: output.enc already exists. Refusing to overwrite.", out)

    def test_verbosity_none(self):
        """Test that nothing is printed with --verbosity none, for both warnings and errors."""
        infile = self._create_input()
        outfile = self.temp_dir_path / "output.enc"
        fx_path = self.temp_dir_path / "fx.py"
        seed_path = self.temp_dir_path / "seed.bin"
        # Ensure clean state
        if fx_path.exists():
            fx_path.unlink()
        if seed_path.exists():
            seed_path.unlink()
        with self._in_tempdir():
            # Test warning (successful encode, should print warning if verbosity != none)
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertDoesNotRaise(SystemExit) as cm:
                    self._encode(infile, outfile, extra_args=["--verbosity", "none"])
            out = stderr.getvalue()
            self.assertEqual(out, "")
            # Test error (attempt to overwrite output.enc)
            stderr = StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as cm:
                    self._encode(infile, outfile, extra_args=["--verbosity", "none"])
            self.assertNotEqual(cm.exception.code, 0)
            out = stderr.getvalue()
            self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
