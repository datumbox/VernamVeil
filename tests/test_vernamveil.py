import hashlib
import random
import string
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from vernamveil._fx_utils import OTPFX, generate_default_fx, load_fx_from_file
from vernamveil._types import _HAS_C_MODULE, _HAS_NUMPY
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

        hash_names = ["blake2b", "sha256"]
        if _HAS_C_MODULE:
            hash_names.append("blake3")

        for hash_name in hash_names:
            for mode, vectorise, use_c_backend in modes:
                if hash_name == "blake3" and not use_c_backend:
                    # Skip blake3 if C extension is not available
                    continue
                with self.subTest(mode=mode, hash_name=hash_name, **cypher_kwargs):
                    print(f"mode={mode}, hash_name={hash_name}, {cypher_kwargs}")
                    context = (
                        patch("vernamveil._hash_utils._HAS_C_MODULE", use_c_backend)
                        if use_c_backend is not None
                        else nullcontext()
                    )
                    with context:
                        fx = generate_default_fx(vectorise=vectorise)
                        cypher = VernamVeil(fx, hash_name=hash_name, **cypher_kwargs)
                        test_func(cypher, vectorise)

    def test_single_message_encryption(self):
        """Test that a single message can be encrypted and decrypted correctly."""
        message = (
            "This is a secret message that needs to be protected. Make sure it is hard to break!"
        )

        def test(cypher, _):
            msg = message.encode()
            encrypted, _ = cypher.encode(msg, self.initial_seed)
            decrypted, _ = cypher.decode(encrypted, self.initial_seed)
            self.assertEqual(msg, decrypted)

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
                msg = ("".join(random.choices(string.printable, k=i))).encode()
                encrypted, _ = cypher.encode(msg, self.initial_seed)
                decrypted, _ = cypher.decode(encrypted, self.initial_seed)
                self.assertEqual(msg, decrypted, f"Failed at length {i}")

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

    def test_otpfx_roundtrip_encryption_decryption(self):
        """Test OTPFX: roundtrip encryption/decryption with keystream serialization and reload."""

        # Generate a large enough pseudo-random keystream
        block_size = 64
        keystream = [VernamVeil.get_initial_seed(num_bytes=block_size) for _ in range(100)]

        # Message and cypher configuration
        message = b"Secret message for OTPFX roundtrip test!"
        cypher_kwargs = dict(
            chunk_size=8,
            delimiter_size=4,
            padding_range=(2, 2),
            decoy_ratio=0.5,
            siv_seed_initialisation=True,
            auth_encrypt=True,
        )

        for vectorise in (False, True) if _HAS_NUMPY else (False,):
            # Instantiate the OTPFX with the keystream
            fx = OTPFX(keystream, block_size, vectorise=vectorise)

            # Encrypt the message
            seed = VernamVeil.get_initial_seed()
            cypher = VernamVeil(fx, **cypher_kwargs)
            encrypted, _ = cypher.encode(message, seed)

            # Serialize and reload the OTPFX
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(fx.source_code)
                fx_path = tmp.name
            fx_loaded = load_fx_from_file(fx_path)

            # Decrypt with the reloaded fx
            cypher = VernamVeil(fx_loaded, **cypher_kwargs)
            decrypted, _ = cypher.decode(encrypted, seed)
            self.assertEqual(message, decrypted)

    def test_otpfx_reset_and_clip_encryption_decryption(self):
        """Test OTPFX: encryption/decryption with keystream reset and clip."""

        # Generate a large enough pseudo-random keystream
        block_size = 64
        keystream = [VernamVeil.get_initial_seed(num_bytes=block_size) for _ in range(100)]

        # Message and cypher configuration
        message = b"Secret message for OTPFX reset and clip test!"
        cypher_kwargs = dict(
            chunk_size=8,
            delimiter_size=4,
            padding_range=(2, 2),
            decoy_ratio=0.5,
            siv_seed_initialisation=True,
            auth_encrypt=True,
        )

        for vectorise in (False, True) if _HAS_NUMPY else (False,):
            # Instantiate the OTPFX with the keystream
            fx = OTPFX(keystream, block_size, vectorise=vectorise)

            # Encrypt the message
            seed = VernamVeil.get_initial_seed()
            cypher = VernamVeil(fx, **cypher_kwargs)
            encrypted, _ = cypher.encode(message, seed)

            # Clip the keystream and reset the position
            fx.keystream = fx.keystream[: fx.position]
            fx.position = 0

            # Decrypt the message
            decrypted, _ = cypher.decode(encrypted, seed)
            self.assertEqual(message, decrypted)


if __name__ == "__main__":
    unittest.main()
