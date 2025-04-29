import os
import tempfile
import shutil
import unittest
from contextlib import contextmanager

from vernamveil.cli import main


class TestVernamVeilCLI(unittest.TestCase):
    """Unit tests for the VernamVeil CLI covering all supported scenarios."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.fx_code = """
def fx(i, seed, bound):
    return (i + 1) % bound
"""
        self.fx_strong_code = """
import hashlib

def fx(i, seed, bound):
    h = int.from_bytes(hashlib.sha256(seed + int(i).to_bytes(4, 'big')).digest(), 'big')
    return h % bound if bound is not None else h
"""

    def tearDown(self):
        """Remove the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def _write_file(self, filename, data, mode="wb"):
        """Helper to write binary or text data to a file in temp_dir."""
        path = os.path.join(self.temp_dir, filename)
        with open(path, mode) as f:
            f.write(data)
        return path

    def _create_input(self, content=b"test data"):
        """Create an input file with given content."""
        return self._write_file("input.txt", content)

    def _create_fx(self, code=None):
        """Create an fx.py file with given code."""
        code = code if code is not None else self.fx_code
        return self._write_file("fx.py", code.encode(), mode="wb")

    def _create_seed(self, content=b"myseed"):
        """Create a seed.bin file with given content."""
        return self._write_file("seed.bin", content)

    @contextmanager
    def _in_tempdir(self):
        """Context manager to run code in the temp dir, restoring cwd and rethrowing exceptions."""
        cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            if os.getcwd() != cwd:
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
        if not any("vectorise" in arg for arg in args):
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
        if not any("vectorise" in arg for arg in args):
            args += ["--no-vectorise"]
        main(args)

    def test_encode_generates_fx_and_seed(self):
        """Test encoding with auto-generated fx and seed."""
        infile = self._create_input()
        outfile = os.path.join(self.temp_dir, "output.enc")
        with self._in_tempdir():
            self._encode(infile, outfile)
        self.assertTrue(os.path.exists(outfile))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "fx.py")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "seed.bin")))

    def test_encode_with_custom_fx(self):
        """Test encoding with a user-supplied fx file."""
        infile = self._create_input()
        outfile = os.path.join(self.temp_dir, "output.enc")
        fx_file = self._create_fx()
        with self._in_tempdir():
            self._encode(infile, outfile, fx_file=fx_file)
        self.assertTrue(os.path.exists(outfile))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "seed.bin")))

    def test_encode_with_custom_seed(self):
        """Test encoding with a user-supplied seed file."""
        infile = self._create_input()
        outfile = os.path.join(self.temp_dir, "output.enc")
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(infile, outfile, seed_file=seed_file)
        self.assertTrue(os.path.exists(outfile))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "fx.py")))

    def test_encode_with_custom_fx_and_seed(self):
        """Test encoding with both custom fx and seed files."""
        infile = self._create_input()
        outfile = os.path.join(self.temp_dir, "output.enc")
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(infile, outfile, fx_file=fx_file, seed_file=seed_file)
        self.assertTrue(os.path.exists(outfile))

    def test_decode_requires_fx_and_seed(self):
        """Test decoding requires both fx and seed files."""
        infile = self._create_input()
        encfile = os.path.join(self.temp_dir, "output.enc")
        outfile = os.path.join(self.temp_dir, "output.txt")
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(infile, encfile, fx_file=fx_file, seed_file=seed_file)
            self._decode(encfile, outfile, fx_file, seed_file)
        self.assertTrue(os.path.exists(outfile))

    def test_encode_with_check_fx_sanity(self):
        """Test encoding with fx sanity check enabled."""
        infile = self._create_input()
        outfile = os.path.join(self.temp_dir, "output.enc")
        with self._in_tempdir():
            self._encode(infile, outfile, extra_args=["--check-fx-sanity"])
        self.assertTrue(os.path.exists(outfile))

    def test_decode_with_check_fx_sanity(self):
        """Test decoding with fx sanity check enabled (handles both scalar and vectorised fx)."""
        infile = self._create_input()
        encfile = os.path.join(self.temp_dir, "output.enc")
        outfile = os.path.join(self.temp_dir, "output.txt")
        fx_file = self._create_fx(self.fx_strong_code)
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(infile, encfile, fx_file=fx_file, seed_file=seed_file)
            self._decode(encfile, outfile, fx_file, seed_file, extra_args=["--check-fx-sanity"])
        self.assertTrue(os.path.exists(outfile))

    def test_decode_with_check_fx_sanity_fails(self):
        """Test that fx sanity check fails if fx does not depend on seed."""
        infile = self._create_input()
        encfile = os.path.join(self.temp_dir, "output.enc")
        outfile = os.path.join(self.temp_dir, "output.txt")
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(infile, encfile, fx_file=fx_file, seed_file=seed_file)
            with self.assertRaises(SystemExit) as cm:
                self._decode(encfile, outfile, fx_file, seed_file, extra_args=["--check-fx-sanity"])
            self.assertNotEqual(cm.exception.code, 0)

    def test_encode_refuses_to_overwrite_existing_fx(self):
        """Test that encoding refuses to overwrite an existing fx.py file."""
        infile = self._create_input()
        outfile = os.path.join(self.temp_dir, "output.enc")
        # Pre-create fx.py
        self._create_fx()
        with self._in_tempdir():
            with self.assertRaises(SystemExit) as cm:
                self._encode(infile, outfile)
            self.assertNotEqual(cm.exception.code, 0)
            # Ensure fx.py was not modified (content unchanged)
            with open("fx.py", "rb") as f:
                self.assertEqual(f.read(), self.fx_code.encode())

    def test_encode_refuses_to_overwrite_existing_seed(self):
        """Test that encoding refuses to overwrite an existing seed.bin file."""
        infile = self._create_input()
        outfile = os.path.join(self.temp_dir, "output.enc")
        # Pre-create seed.bin
        self._create_seed()
        with self._in_tempdir():
            with self.assertRaises(SystemExit) as cm:
                self._encode(infile, outfile)
            self.assertNotEqual(cm.exception.code, 0)
            # Ensure seed.bin was not modified (content unchanged)
            with open("seed.bin", "rb") as f:
                self.assertEqual(f.read(), b"myseed")

    def test_encode_refuses_to_overwrite_existing_output(self):
        """Test that encoding refuses to overwrite an existing output file."""
        infile = self._create_input()
        outfile = self._write_file("output.enc", b"original data")
        with self._in_tempdir():
            with self.assertRaises(SystemExit) as cm:
                self._encode(infile, outfile)
            self.assertNotEqual(cm.exception.code, 0)
            # Ensure output file was not modified
            with open("output.enc", "rb") as f:
                self.assertEqual(f.read(), b"original data")

    def test_decode_refuses_to_overwrite_existing_output(self):
        """Test that decoding refuses to overwrite an existing output file."""
        infile = self._create_input()
        encfile = os.path.join(self.temp_dir, "output.enc")
        outfile = self._write_file("output.txt", b"original plain")
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(infile, encfile, fx_file=fx_file, seed_file=seed_file)
            with self.assertRaises(SystemExit) as cm:
                self._decode(encfile, outfile, fx_file, seed_file)
            self.assertNotEqual(cm.exception.code, 0)
            # Ensure output file was not modified
            with open("output.txt", "rb") as f:
                self.assertEqual(f.read(), b"original plain")


if __name__ == "__main__":
    unittest.main()
