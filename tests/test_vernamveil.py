import hashlib
import os
import random
import string
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from vernamveil import VernamVeil, generate_polynomial_fx

try:
    import numpy as np  # noqa: F401
    from npsha256._npsha256 import _HAS_C_MODULE

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def get_test_modes():
    """Return available test modes based on numpy and C backend availability."""
    modes = [("scalar", False, None)]
    if HAS_NUMPY:

        # Always test vectorised (force fallback)
        modes.append(("vectorised", True, False))
        # Only test with C if available
        if _HAS_C_MODULE:
            modes.append(("vectorised_with_npsha256", True, True))
    return modes


class TestVernamVeil(unittest.TestCase):
    """Unit tests for the VernamVeil stream cipher."""

    @classmethod
    def setUpClass(cls):
        """Set up a reusable initial seed for all tests."""
        cls.initial_seed = VernamVeil.get_initial_seed()

    def for_all_modes(self, fx_degree, test_func):
        """Run the given test function for all supported cipher modes."""
        for mode, vectorise, use_c_backend in get_test_modes():
            with self.subTest(mode=mode):
                print(f"mode={mode}")
                context = (
                    patch("npsha256._npsha256._HAS_C_MODULE", use_c_backend)
                    if use_c_backend is not None
                    else nullcontext()
                )
                with context:
                    fx = generate_polynomial_fx(fx_degree, vectorise=vectorise)
                    cipher = VernamVeil(fx, vectorise=vectorise) if vectorise else VernamVeil(fx)
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

    @unittest.skipIf(not os.path.exists("/usr/bin/ls"), "Input file does not exist")
    def test_file_encryption(self):
        """Test file encryption and decryption, verifying file integrity via checksum."""
        input_file = "/usr/bin/ls"
        output_file = "/tmp/ls.encoded"
        decoded_file = "/tmp/ls.decoded"

        def test(_, vectorise):
            fx = generate_polynomial_fx(20, vectorise=vectorise)
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
            with open(input_file, "rb") as f1, open(decoded_file, "rb") as f2:
                checksum_original = hashlib.blake2b(f1.read()).hexdigest()
                checksum_decoded = hashlib.blake2b(f2.read()).hexdigest()
            self.assertEqual(checksum_original, checksum_decoded)
            os.remove(output_file)
            os.remove(decoded_file)

        self.for_all_modes(20, test)


if __name__ == "__main__":
    unittest.main()
