import hashlib
import random
import string
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from vernamveil.cypher import _HAS_NUMPY, VernamVeil
from vernamveil.fx_utils import generate_default_fx
from vernamveil.hash_utils import _HAS_C_MODULE


def get_test_modes():
    """Return available test modes based on numpy and C backend availability."""
    modes = [("scalar", False, None)]
    if _HAS_NUMPY:
        # Always test vectorised (force fallback)
        modes.append(("vectorised", True, False))
        # Only test with C if available
        if _HAS_C_MODULE:
            modes.append(("vectorised_with_extension", True, True))
    return modes


class TestVernamVeil(unittest.TestCase):
    """Unit tests for the VernamVeil stream cipher."""

    @classmethod
    def setUpClass(cls):
        """Set up a reusable initial seed for all tests."""
        cls.initial_seed = VernamVeil.get_initial_seed()

    def for_all_modes(self, fx_complexity, test_func):
        """Run the given test function for all supported cipher modes."""
        for mode, vectorise, use_c_backend in get_test_modes():
            with self.subTest(mode=mode):
                print(f"mode={mode}")
                context = (
                    patch("vernamveil.hash_utils._HAS_C_MODULE", use_c_backend)
                    if use_c_backend is not None
                    else nullcontext()
                )
                with context:
                    fx = generate_default_fx(fx_complexity, vectorise=vectorise)
                    cipher = VernamVeil(fx, vectorise=vectorise)
                    test_func(cipher, vectorise)

    def test_single_message_encryption(self):
        """Test that a single message can be encrypted and decrypted correctly."""
        message = (
            "This is a secret message that needs to be protected. Make sure it is hard to break!"
        )

        def test(cipher, _):
            encrypted, _ = cipher.encode(message.encode(), self.initial_seed)
            decrypted, _ = cipher.decode(encrypted, self.initial_seed)
            self.assertEqual(message, decrypted.decode())

        self.for_all_modes(10, test)

    def test_variable_length_encryption(self):
        """Test encryption and decryption for messages of varying lengths."""

        def test(cipher, _):
            for i in range(500):
                msg = "".join(random.choices(string.printable, k=i))
                encrypted, _ = cipher.encode(msg.encode(), self.initial_seed)
                decrypted, _ = cipher.decode(encrypted, self.initial_seed)
                self.assertEqual(msg, decrypted.decode(), f"Failed at length {i}")

        self.for_all_modes(10, test)

    def test_file_encryption(self):
        """Test file encryption and decryption, verifying file integrity via checksum."""
        with (
            tempfile.NamedTemporaryFile(delete=False) as input_tmp,
            tempfile.NamedTemporaryFile(delete=False) as output_tmp,
            tempfile.NamedTemporaryFile(delete=False) as decoded_tmp,
        ):
            input_file = Path(input_tmp.name)
            output_file = Path(output_tmp.name)
            decoded_file = Path(decoded_tmp.name)

            # Write random data to input file
            input_tmp.write(random.randbytes(65536))
            input_tmp.flush()

        def test(_, vectorise):
            fx = generate_default_fx(20, vectorise=vectorise)
            cipher_args = dict(vectorise=vectorise)
            VernamVeil.process_file(
                input_file,
                output_file,
                fx,
                self.initial_seed,
                mode="encode",
                buffer_size=1024,
                chunk_size=128,
                **cipher_args,
            )
            VernamVeil.process_file(
                output_file,
                decoded_file,
                fx,
                self.initial_seed,
                mode="decode",
                buffer_size=1024,
                chunk_size=128,
                **cipher_args,
            )
            with input_file.open("rb") as f1, decoded_file.open("rb") as f2:
                checksum_original = hashlib.blake2b(f1.read()).hexdigest()
                checksum_decoded = hashlib.blake2b(f2.read()).hexdigest()
            self.assertEqual(checksum_original, checksum_decoded)
            output_file.unlink()
            decoded_file.unlink()

        self.for_all_modes(20, test)


if __name__ == "__main__":
    unittest.main()
