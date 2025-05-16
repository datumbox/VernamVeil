import hashlib
import random
import string
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from vernamveil._cypher import _HAS_NUMPY
from vernamveil._fx_utils import generate_default_fx
from vernamveil._hash_utils import _HAS_C_MODULE
from vernamveil._vernamveil import VernamVeil


class TestVernamVeil(unittest.TestCase):
    """Unit tests for the VernamVeil stream cypher."""

    @classmethod
    def setUpClass(cls):
        """Set up a reusable initial seed for all tests."""
        cls.initial_seed = VernamVeil.get_initial_seed()

    def _for_all_modes(self, test_func, **cypher_kwargs):
        """Run the given test function for all supported cypher modes."""
        modes = [("scalar", False, None)]
        if _HAS_NUMPY:
            # Always test vectorised
            modes.append(("vectorised", True, False))
            # Only test with C if available
            if _HAS_C_MODULE:
                modes.append(("vectorised_with_extension", True, True))

        for mode, vectorise, use_c_backend in modes:
            with self.subTest(mode=mode, **cypher_kwargs):
                print(f"mode={mode}, {cypher_kwargs}")
                context = (
                    patch("vernamveil._hash_utils._HAS_C_MODULE", use_c_backend)
                    if use_c_backend is not None
                    else nullcontext()
                )
                with context:
                    fx = generate_default_fx(vectorise=vectorise)
                    cypher = VernamVeil(fx, **cypher_kwargs)
                    test_func(cypher, vectorise)

    def test_single_message_encryption(self):
        """Test that a single message can be encrypted and decrypted correctly."""
        message = (
            "This is a secret message that needs to be protected. Make sure it is hard to break!"
        )

        def test(cypher, _):
            encrypted, _ = cypher.encode(message.encode(), self.initial_seed)
            decrypted, _ = cypher.decode(encrypted, self.initial_seed)
            self.assertEqual(message, decrypted.decode())

        # Test with all combinations of siv_seed_initialisation and auth_encrypt
        self._for_all_modes(test, siv_seed_initialisation=True, auth_encrypt=True)
        self._for_all_modes(test, siv_seed_initialisation=False, auth_encrypt=True)
        self._for_all_modes(test, siv_seed_initialisation=True, auth_encrypt=False)
        self._for_all_modes(test, siv_seed_initialisation=False, auth_encrypt=False)

        # Test with different padding configurations
        self._for_all_modes(test, padding_range=(0, 0))
        self._for_all_modes(test, padding_range=(0, 1))
        self._for_all_modes(test, padding_range=(5, 5))

    def test_variable_length_encryption(self):
        """Test encryption and decryption for messages of varying lengths."""

        def test(cypher, _):
            for i in range(150):
                msg = "".join(random.choices(string.printable, k=i))
                encrypted, _ = cypher.encode(msg.encode(), self.initial_seed)
                decrypted, _ = cypher.decode(encrypted, self.initial_seed)
                self.assertEqual(msg, decrypted.decode(), f"Failed at length {i}")

        # Test with all combinations of siv_seed_initialisation and auth_encrypt
        self._for_all_modes(test, siv_seed_initialisation=True, auth_encrypt=True)
        self._for_all_modes(test, siv_seed_initialisation=False, auth_encrypt=True)
        self._for_all_modes(test, siv_seed_initialisation=True, auth_encrypt=False)
        self._for_all_modes(test, siv_seed_initialisation=False, auth_encrypt=False)

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

        def test(cypher, _):
            cypher.process_file(
                "encode",
                input_file,
                output_file,
                self.initial_seed,
                buffer_size=1024,
            )
            cypher.process_file(
                "decode",
                output_file,
                decoded_file,
                self.initial_seed,
                buffer_size=1024,
            )
            with input_file.open("rb") as f1, decoded_file.open("rb") as f2:
                checksum_original = hashlib.blake2b(f1.read()).hexdigest()
                checksum_decoded = hashlib.blake2b(f2.read()).hexdigest()
            self.assertEqual(checksum_original, checksum_decoded)
            output_file.unlink()
            decoded_file.unlink()

        # Test with all combinations of siv_seed_initialisation and auth_encrypt
        self._for_all_modes(test, siv_seed_initialisation=True, auth_encrypt=True)
        self._for_all_modes(test, siv_seed_initialisation=False, auth_encrypt=True)
        self._for_all_modes(test, siv_seed_initialisation=True, auth_encrypt=False)
        self._for_all_modes(test, siv_seed_initialisation=False, auth_encrypt=False)

    def test_delimiter_conflict_raises(self):
        """Test that encoding fails if the delimiter appears in the message."""
        cypher = VernamVeil(generate_default_fx())
        cypher._delimiter_size = 1  # override to force a delimiter size of 1 byte
        message = bytes(range(256))  # Message contains every possible byte value
        with self.assertRaises(ValueError) as ctx:
            cypher.encode(message, self.initial_seed)
        self.assertIn("The delimiter appears in the message.", str(ctx.exception))

    def test_avalanche_effect(self):
        """Test that flipping a single bit in the input causes ~50% of output bits to change (avalanche effect)."""
        message = b"This is a test message for avalanche effect!"

        def test(cypher, _):
            modified = bytearray(message)
            byte_idx = len(modified) // 2
            bit_idx = 0
            modified[byte_idx] ^= 1 << bit_idx

            seed = self.initial_seed
            original, _ = cypher.encode(message, seed)
            altered, _ = cypher.encode(modified, seed)

            diff_count = 0
            for o, a in zip(original, altered):
                diff = o ^ a
                diff_count += bin(diff).count("1")

            total_bits = len(original) * 8
            expected_diff = total_bits * 0.5

            self.assertTrue(
                abs(diff_count - expected_diff) / total_bits <= 0.15,
                f"Poor avalanche effect: {diff_count} bits differ (expected ~{expected_diff})",
            )

        # Test with all combinations of siv_seed_initialisation and auth_encrypt
        self._for_all_modes(test, siv_seed_initialisation=True, auth_encrypt=True)
        self._for_all_modes(test, siv_seed_initialisation=False, auth_encrypt=True)
        self._for_all_modes(test, siv_seed_initialisation=True, auth_encrypt=False)
        self._for_all_modes(test, siv_seed_initialisation=False, auth_encrypt=False)


if __name__ == "__main__":
    unittest.main()
