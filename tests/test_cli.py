import os
import shutil
import sys  # noqa: F401
import tempfile
import unittest
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch

from vernamveil.cli import main


class _UnclosableBytesIO(BytesIO):
    """Buffer that prevents closing during the test"""

    def close(self):
        pass


class _FakeStdout:
    """Fake stdout class to simulate a non-tty buffer for testing purposes."""

    def __init__(self, buffer, isatty=False):
        self.buffer = buffer
        self._isatty = isatty

    def isatty(self):
        return self._isatty


class _FakeStdin:
    """Fake stdin class with a buffer attribute for simulating binary input in tests."""

    def __init__(self, buffer):
        self.buffer = buffer


class _BrokenPipeBytesIO(BytesIO):
    """BytesIO that raises BrokenPipeError on write to simulate a broken pipe."""

    def write(self, b):
        raise BrokenPipeError("Simulated broken pipe")


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

    @contextmanager
    def _patch_stdio(self, infile, outfile, stdin_data=None):
        patches = []
        if infile == "-":
            patches.append(patch("sys.stdin", new=_FakeStdin(BytesIO(stdin_data))))
        # Only patch sys.stdout if not already patched
        if not isinstance(sys.stdout, _FakeStdout):
            fake_stdout_buffer = _UnclosableBytesIO()
            fake_stdout = _FakeStdout(fake_stdout_buffer, isatty=False)
            patches.append(patch("sys.stdout", new=fake_stdout))
        else:
            fake_stdout_buffer = getattr(sys.stdout, "buffer", None)
        with contextmanager(lambda: (yield))():
            for p in patches:
                p.start()
            try:
                yield fake_stdout_buffer
            finally:
                for p in reversed(patches):
                    p.stop()

    def _encode(
        self, infile, outfile, fx_file=None, seed_file=None, extra_args=None, stdin_data=None
    ):
        """Helper to run the encode CLI command with optional fx and seed files. Supports stdin/stdout via '-'."""
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
        with self._patch_stdio(infile, outfile, stdin_data) as fake_stdout_buffer:
            main(args)
        return fake_stdout_buffer.getvalue() if fake_stdout_buffer else None

    def _decode(self, infile, outfile, fx_file, seed_file, extra_args=None, stdin_data=None):
        """Helper to run the decode CLI command with required fx and seed files. Supports stdin/stdout via '-'."""
        args = [
            "decode",
            "--infile",
            infile,
            "--outfile",
            outfile,
        ]
        if fx_file:
            args += ["--fx-file", fx_file]
        if seed_file:
            args += ["--seed-file", seed_file]
        if extra_args:
            args += extra_args

        args = [str(arg) for arg in args]
        if not any(arg.endswith("vectorise") for arg in args):
            args += ["--no-vectorise"]
        with self._patch_stdio(infile, outfile, stdin_data) as fake_stdout_buffer:
            main(args)
        return fake_stdout_buffer.getvalue() if fake_stdout_buffer else None

    def _assert_decoded_matches_input(self, original_path, decoded_path):
        """Assert that the decoded output matches the original input."""
        with open(original_path, "rb") as f1, open(decoded_path, "rb") as f2:
            self.assertEqual(f1.read(), f2.read())

    def test_encode_generates_fx_and_seed_file_to_file(self):
        """Test encoding with auto-generated fx and seed, file-to-file mode."""
        with self._in_tempdir():
            self._encode(self.infile, self.encfile)
            self.assertTrue(self.encfile.exists())
            self.assertTrue((self.temp_dir_path / "fx.py").exists())
            self.assertTrue((self.temp_dir_path / "seed.bin").exists())

    def test_encode_generates_fx_and_seed_file_to_stdout(self):
        """Test encoding with auto-generated fx and seed, file-to-stdout mode."""
        with self._in_tempdir():
            out_bytes = self._encode(self.infile, "-")
            self.assertIsInstance(out_bytes, bytes)
            self.assertGreater(len(out_bytes), 0)

    def test_encode_generates_fx_and_seed_stdin_to_file(self):
        """Test encoding with auto-generated fx and seed, stdin-to-file mode."""
        with self._in_tempdir():
            in_bytes = self.infile.read_bytes()
            self._encode("-", self.encfile, stdin_data=in_bytes)
            self.assertTrue(self.encfile.exists())

    def test_encode_generates_fx_and_seed_stdin_to_stdout(self):
        """Test encoding with auto-generated fx and seed, stdin-to-stdout mode."""
        with self._in_tempdir():
            in_bytes = self.infile.read_bytes()
            out_bytes = self._encode("-", "-", stdin_data=in_bytes)
            self.assertIsInstance(out_bytes, bytes)
            self.assertGreater(len(out_bytes), 0)

    def test_encode_with_custom_fx_file_to_file(self):
        """Test encoding with custom fx, file-to-file mode."""
        with self._in_tempdir():
            fx_path = self._create_fx()
            self._encode(self.infile, self.encfile, fx_file=fx_path)
            self.assertTrue(self.encfile.exists())
            self.assertTrue((self.temp_dir_path / "seed.bin").exists())

    def test_encode_with_custom_fx_file_to_stdout(self):
        """Test encoding with custom fx, file-to-stdout mode."""
        with self._in_tempdir():
            fx_path = self._create_fx()
            out_bytes = self._encode(self.infile, "-", fx_file=fx_path)
            self.assertIsInstance(out_bytes, bytes)
            self.assertGreater(len(out_bytes), 0)

    def test_encode_with_custom_fx_stdin_to_file(self):
        """Test encoding with custom fx, stdin-to-file mode."""
        with self._in_tempdir():
            fx_path = self._create_fx()
            in_bytes = self.infile.read_bytes()
            self._encode("-", self.encfile, fx_file=fx_path, stdin_data=in_bytes)
            self.assertTrue(self.encfile.exists())

    def test_encode_with_custom_fx_stdin_to_stdout(self):
        """Test encoding with custom fx, stdin-to-stdout mode."""
        with self._in_tempdir():
            fx_path = self._create_fx()
            in_bytes = self.infile.read_bytes()
            out_bytes = self._encode("-", "-", fx_file=fx_path, stdin_data=in_bytes)
            self.assertIsInstance(out_bytes, bytes)
            self.assertGreater(len(out_bytes), 0)

    def test_encode_with_custom_seed_file_to_file(self):
        """Test encoding with custom seed, file-to-file mode."""
        with self._in_tempdir():
            seed_path = self._create_seed()
            self._encode(self.infile, self.encfile, seed_file=seed_path)
            self.assertTrue(self.encfile.exists())
            self.assertTrue((self.temp_dir_path / "fx.py").exists())

    def test_encode_with_custom_seed_file_to_stdout(self):
        """Test encoding with custom seed, file-to-stdout mode."""
        with self._in_tempdir():
            seed_path = self._create_seed()
            out_bytes = self._encode(self.infile, "-", seed_file=seed_path)
            self.assertIsInstance(out_bytes, bytes)
            self.assertGreater(len(out_bytes), 0)

    def test_encode_with_custom_seed_stdin_to_file(self):
        """Test encoding with custom seed, stdin-to-file mode."""
        with self._in_tempdir():
            seed_path = self._create_seed()
            in_bytes = self.infile.read_bytes()
            self._encode("-", self.encfile, seed_file=seed_path, stdin_data=in_bytes)
            self.assertTrue(self.encfile.exists())

    def test_encode_with_custom_seed_stdin_to_stdout(self):
        """Test encoding with custom seed, stdin-to-stdout mode."""
        with self._in_tempdir():
            seed_path = self._create_seed()
            in_bytes = self.infile.read_bytes()
            out_bytes = self._encode("-", "-", seed_file=seed_path, stdin_data=in_bytes)
            self.assertIsInstance(out_bytes, bytes)
            self.assertGreater(len(out_bytes), 0)

    def test_encode_with_custom_fx_and_seed_file_to_file(self):
        """Test encoding with custom fx and seed, file-to-file mode."""
        with self._in_tempdir():
            fx_path = self._create_fx()
            seed_path = self._create_seed()
            self._encode(self.infile, self.encfile, fx_file=fx_path, seed_file=seed_path)
            self.assertTrue(self.encfile.exists())

    def test_encode_with_custom_fx_and_seed_file_to_stdout(self):
        """Test encoding with custom fx and seed, file-to-stdout mode."""
        with self._in_tempdir():
            fx_path = self._create_fx()
            seed_path = self._create_seed()
            out_bytes = self._encode(self.infile, "-", fx_file=fx_path, seed_file=seed_path)
            self.assertIsInstance(out_bytes, bytes)
            self.assertGreater(len(out_bytes), 0)

    def test_encode_with_custom_fx_and_seed_stdin_to_file(self):
        """Test encoding with custom fx and seed, stdin-to-file mode."""
        with self._in_tempdir():
            fx_path = self._create_fx()
            seed_path = self._create_seed()
            in_bytes = self.infile.read_bytes()
            self._encode(
                "-", self.encfile, fx_file=fx_path, seed_file=seed_path, stdin_data=in_bytes
            )
            self.assertTrue(self.encfile.exists())

    def test_encode_with_custom_fx_and_seed_stdin_to_stdout(self):
        """Test encoding with custom fx and seed, stdin-to-stdout mode."""
        with self._in_tempdir():
            fx_path = self._create_fx()
            seed_path = self._create_seed()
            in_bytes = self.infile.read_bytes()
            out_bytes = self._encode(
                "-", "-", fx_file=fx_path, seed_file=seed_path, stdin_data=in_bytes
            )
            self.assertIsInstance(out_bytes, bytes)
            self.assertGreater(len(out_bytes), 0)

    def test_decode_requires_fx_file(self):
        """Test that decoding without --fx-file produces the correct error message."""
        with self._in_tempdir():
            # Create a dummy input file and a dummy seed file
            encfile = self._write_file("input.enc", b"dummydata")
            seed_file = self._create_seed()
            stderr = StringIO()
            with patch("sys.stderr", stderr), self.assertRaises(SystemExit):
                self._decode(
                    encfile,
                    self.outfile,
                    fx_file=None,
                    seed_file=seed_file,
                    extra_args=["--no-vectorise"],
                )
            self.assertIn("Error: --fx-file must be specified when decoding.", stderr.getvalue())

    def test_decode_requires_seed_file(self):
        """Test that decoding without --seed-file produces the correct error message."""
        with self._in_tempdir():
            # Create a dummy input file and a dummy fx file
            encfile = self._write_file("input.enc", b"dummydata")
            fx_file = self._create_fx()
            stderr = StringIO()
            with patch("sys.stderr", stderr), self.assertRaises(SystemExit):
                self._decode(
                    encfile,
                    self.outfile,
                    fx_file=fx_file,
                    seed_file=None,
                    extra_args=["--no-vectorise"],
                )
            self.assertIn("Error: --seed-file must be specified when decoding.", stderr.getvalue())

    def test_decode_file_to_file(self):
        """Test decoding, file-to-file mode."""
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(self.infile, self.encfile, fx_file=fx_file, seed_file=seed_file)
            self._decode(self.encfile, self.outfile, fx_file, seed_file)
            self.assertTrue(self.outfile.exists())
            self._assert_decoded_matches_input(self.infile, self.outfile)

    def test_decode_file_to_stdout(self):
        """Test decoding, file-to-stdout mode."""
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(self.infile, self.encfile, fx_file=fx_file, seed_file=seed_file)
            out_bytes = self._decode(self.encfile, "-", fx_file, seed_file)
            self.assertIsInstance(out_bytes, bytes)
            self.assertGreater(len(out_bytes), 0)
            self.assertEqual(self.infile.read_bytes(), out_bytes)

    def test_decode_stdin_to_file(self):
        """Test decoding, stdin-to-file mode."""
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(self.infile, self.encfile, fx_file=fx_file, seed_file=seed_file)
            enc_bytes = self.encfile.read_bytes()
            self._decode("-", self.outfile, fx_file, seed_file, stdin_data=enc_bytes)
            self.assertTrue(self.outfile.exists())
            self._assert_decoded_matches_input(self.infile, self.outfile)

    def test_decode_stdin_to_stdout(self):
        """Test decoding, stdin-to-stdout mode."""
        fx_file = self._create_fx()
        seed_file = self._create_seed()
        with self._in_tempdir():
            self._encode(self.infile, self.encfile, fx_file=fx_file, seed_file=seed_file)
            enc_bytes = self.encfile.read_bytes()
            out_bytes = self._decode("-", "-", fx_file, seed_file, stdin_data=enc_bytes)
            self.assertIsInstance(out_bytes, bytes)
            self.assertGreater(len(out_bytes), 0)
            self.assertEqual(self.infile.read_bytes(), out_bytes)

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
            "Error: Seed is too short.",
            stderr.getvalue(),
        )

    def _assert_encode_refuses_to_overwrite(self, file_path):
        """Assert that encoding refuses to overwrite an existing file and preserves its content."""
        expected_error = f"Error: {file_path.resolve()} already exists. Refusing to overwrite."
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
                f"Error: {(self.temp_dir_path / 'output.txt').resolve()} already exists. Refusing to overwrite.",
                stderr.getvalue(),
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
            self.assertIn(
                f"Warning: Generated an fx-file in {(self.temp_dir_path / 'fx.py').resolve()}.",
                stderr.getvalue(),
            )
            self.assertIn(
                f"Warning: Generated a seed-file in {(self.temp_dir_path / 'seed.bin').resolve()}.",
                stderr.getvalue(),
            )

            # Now, run a failed encode to trigger an error (output.enc exists)
            stderr = StringIO()
            with patch("sys.stderr", stderr), self.assertRaises(SystemExit):
                self._encode(self.infile, self.encfile, extra_args=["--verbosity", "warning"])
            self.assertIn(
                f"Error: {(self.temp_dir_path / 'output.enc').resolve()} already exists. Refusing to overwrite.",
                stderr.getvalue(),
            )

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
            self.assertIn(
                f"Error: {(self.temp_dir_path / 'output.enc').resolve()} already exists. Refusing to overwrite.",
                stderr.getvalue(),
            )
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

    def test_warning_binary_output_to_terminal(self):
        """Test that a warning is printed when outputting binary data to a terminal."""

        with self._in_tempdir():
            stderr = StringIO()
            fake_buffer = _UnclosableBytesIO()
            # Patch sys.stdout directly here, so _patch_stdio will not patch it again
            with (
                patch("sys.stdout", new=_FakeStdout(fake_buffer, isatty=True)),
                patch("sys.stderr", stderr),
            ):
                self._encode(self.infile, "-", extra_args=["--verbosity", "warning"])

            self.assertIn(
                "Warning: Writing binary data to a terminal may corrupt your session.",
                stderr.getvalue(),
            )
            self.assertGreater(len(fake_buffer.getvalue()), 0)

    def test_broken_pipe_error_handling(self):
        """Test that BrokenPipeError during output is handled gracefully."""
        with self._in_tempdir():
            # Patch sys.stdout to simulate a broken pipe on write
            fake_stdout = _FakeStdout(_BrokenPipeBytesIO(), isatty=False)
            stderr = StringIO()
            with patch("sys.stdout", fake_stdout), patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit):
                    self._encode(self.infile, "-")
            self.assertIn("Error: I/O error during read/write:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
