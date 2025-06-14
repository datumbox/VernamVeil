import itertools
import tempfile
import unittest
from pathlib import Path

from vernamveil._deniability_utils import forge_plausible_fx
from vernamveil._fx_utils import OTPFX, generate_default_fx, load_fx_from_file
from vernamveil._imports import _HAS_NUMPY
from vernamveil._vernamveil import VernamVeil


class TestDeniabilityUtils(unittest.TestCase):
    """Unit tests for plausible deniability utilities."""

    def _run_deniability_test(
        self,
        real_fx,
        chunk_size,
        delimiter_size,
        padding_range,
        decoy_ratio,
    ):
        """Utility to run a basic deniability test with configurable parameters."""
        cypher = VernamVeil(
            real_fx,
            chunk_size=chunk_size,
            delimiter_size=delimiter_size,
            padding_range=padding_range,
            decoy_ratio=decoy_ratio,
            siv_seed_initialisation=True,
            auth_encrypt=True,
        )
        real_seed = VernamVeil.get_initial_seed()
        secret_message = b"Sensitive data: the launch code is 12345!"
        cyphertext, _ = cypher.encode(secret_message, real_seed)

        decoy_message = (
            b"This message is totally real and not at all a decoy... "
            b"There is nothing worth seeing here, move along!!! "
        )
        plausible_fx, fake_seed = forge_plausible_fx(
            cypher, cyphertext, decoy_message, max_obfuscate_attempts=50000
        )

        fake_cypher = VernamVeil(
            plausible_fx,
            chunk_size=cypher._chunk_size,
            delimiter_size=cypher._delimiter_size,
            padding_range=cypher._padding_range,
            decoy_ratio=cypher._decoy_ratio,
            siv_seed_initialisation=False,
            auth_encrypt=False,
        )
        decoy_out, _ = fake_cypher.decode(cyphertext, fake_seed)
        return decoy_out, decoy_message

    def _combo_name(self, chunk_size, delimiter_size, padding_range, decoy_ratio):
        """Produce a string name for the test combo."""
        return f"chunk{chunk_size}_delim{delimiter_size}_pad{padding_range}_decoy{decoy_ratio}"

    def _run_end_to_end_deniability_disk_io_test(
        self,
        secret_message,
        decoy_message,
        chunk_size=64,
        delimiter_size=8,
        padding_range=(5, 20),
        decoy_ratio=0.1,
        max_obfuscate_attempts=50000,
        vectorise=False,
    ):
        """Helper utility to run end-to-end deniability test with disk I/O."""
        if vectorise:
            if not _HAS_NUMPY:
                self.skipTest("Numpy is required for vectorised FX but is not available.")

        real_fx = generate_default_fx(vectorise=vectorise)
        real_seed = VernamVeil.get_initial_seed()

        cypher = VernamVeil(
            real_fx,
            chunk_size=chunk_size,
            delimiter_size=delimiter_size,
            padding_range=padding_range,
            decoy_ratio=decoy_ratio,
            siv_seed_initialisation=True,
            auth_encrypt=True,
        )
        cyphertext, _ = cypher.encode(secret_message, real_seed)

        try:
            plausible_fx, fake_seed = forge_plausible_fx(
                cypher, cyphertext, decoy_message, max_obfuscate_attempts=max_obfuscate_attempts
            )
        except ValueError as e:
            msg = str(e)
            if (
                "Cannot plausibly forge decoy message of length" in msg
                or "Could not find obfuscated decoy of length" in msg
            ):
                self.skipTest("Could not find a decoy for this configuration.")
            else:
                raise

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Store real fx, plausible fx, real seed, fake seed, cyphertext
            real_fx_path = tmpdir_path / "real_fx.py"
            plausible_fx_path = tmpdir_path / "plausible_fx.py"
            real_seed_path = tmpdir_path / "real_seed.hex"
            fake_seed_path = tmpdir_path / "fake_seed.hex"
            cyphertext_path = tmpdir_path / "cyphertext.bin"

            # Save everything to disk
            with open(real_fx_path, "w") as f:
                f.write(real_fx.source_code)
            with open(plausible_fx_path, "w") as f:
                f.write(plausible_fx.source_code)
            with open(real_seed_path, "wb") as f:
                f.write(real_seed)
            with open(fake_seed_path, "wb") as f:
                f.write(fake_seed)
            with open(cyphertext_path, "wb") as f:
                f.write(cyphertext)

            # Load everything back
            loaded_real_fx = load_fx_from_file(str(real_fx_path))
            loaded_plausible_fx = load_fx_from_file(str(plausible_fx_path))
            with open(real_seed_path, "rb") as f:
                loaded_real_seed = f.read()
            with open(fake_seed_path, "rb") as f:
                loaded_fake_seed = f.read()
            with open(cyphertext_path, "rb") as f:
                loaded_cyphertext = f.read()

            # Decrypt with real fx/seed
            real_cypher = VernamVeil(
                loaded_real_fx,
                chunk_size=chunk_size,
                delimiter_size=delimiter_size,
                padding_range=padding_range,
                decoy_ratio=decoy_ratio,
                siv_seed_initialisation=True,
                auth_encrypt=True,
            )
            real_out, _ = real_cypher.decode(loaded_cyphertext, loaded_real_seed)

            # Decrypt with plausible fx/fake seed
            fake_cypher = VernamVeil(
                loaded_plausible_fx,
                chunk_size=chunk_size,
                delimiter_size=delimiter_size,
                padding_range=padding_range,
                decoy_ratio=decoy_ratio,
                siv_seed_initialisation=False,
                auth_encrypt=False,
            )
            decoy_out, _ = fake_cypher.decode(loaded_cyphertext, loaded_fake_seed)

            self.assertEqual(real_out, secret_message)
            self.assertEqual(decoy_out, decoy_message)

    def test_end_to_end_deniability_disk_io_standard_messages(self):
        """Test end-to-end deniability with disk I/O using standard small messages."""
        secret_message = b"Sensitive data: the launch code is 12345!"
        decoy_message = (
            b"This message is totally real and not at all a decoy... "
            b"There is nothing worth seeing here, move along!!! "
        )
        self._run_end_to_end_deniability_disk_io_test(secret_message, decoy_message)

    def test_end_to_end_deniability_disk_io_large_messages(self):
        """Test end-to-end deniability with disk I/O using large messages."""
        size = 5 * 1024 * 1024
        large_secret_message = VernamVeil.get_initial_seed(size)

        decoy_base_phrase = b"nothing to see here "
        num_repeats = size // len(decoy_base_phrase) + 1
        large_decoy_message = (decoy_base_phrase * num_repeats)[:size]

        self._run_end_to_end_deniability_disk_io_test(
            large_secret_message,
            large_decoy_message,
            chunk_size=512 * 1024,
            delimiter_size=64,
            vectorise=True,
            max_obfuscate_attempts=10000,
        )

    def test_deniability_with_otpfx(self):
        """Test deniability works when the real_fx is OTPFX."""
        block_size = 64
        real_fx = OTPFX(
            [VernamVeil.get_initial_seed(num_bytes=block_size) for _ in range(100)],
            block_size,
            False,
        )
        try:
            decoy_out, decoy_message = self._run_deniability_test(
                real_fx=real_fx,
                chunk_size=64,
                delimiter_size=8,
                padding_range=(5, 20),
                decoy_ratio=0.1,
            )
            self.assertEqual(decoy_out, decoy_message)
        except ValueError as e:
            msg = str(e)
            self.assertTrue(
                "Cannot plausibly forge decoy message of length" in msg
                or "Could not find obfuscated decoy of length" in msg
            )
            self.skipTest("Could not find a decoy for this configuration.")


# Generate all combinations
chunk_sizes = [127, 128, 129]
delimiter_sizes = [7, 8, 9, 63, 64, 65]
combos = list(itertools.product(chunk_sizes, delimiter_sizes))


def make_test_func(chunk_size, delimiter_size):
    """Create a test function for a specific combination of parameters."""

    def test_func(self):
        """Test that the deniability function works correctly for a specific combo."""
        vectorise_options = [True, False] if _HAS_NUMPY else [False]
        for vectorise in vectorise_options:
            with self.subTest(vectorise=vectorise):
                try:
                    decoy_out, decoy_message = self._run_deniability_test(
                        real_fx=generate_default_fx(vectorise=vectorise),
                        chunk_size=chunk_size,
                        delimiter_size=delimiter_size,
                        padding_range=(5, 150),
                        decoy_ratio=0.3,
                    )
                    self.assertEqual(decoy_out, decoy_message)
                except ValueError as e:
                    msg = str(e)
                    self.assertTrue(
                        "Cannot plausibly forge decoy message of length" in msg
                        or "Could not find obfuscated decoy of length" in msg
                    )
                    self.skipTest("Could not find a decoy for this configuration.")

    return test_func


# Dynamically add a test method for each combo except vectorise
for chunk_size, delimiter_size in combos:
    test_name = f"test_deniability_{chunk_size}_{delimiter_size}"
    test_func = make_test_func(chunk_size, delimiter_size)
    setattr(TestDeniabilityUtils, test_name, test_func)

if __name__ == "__main__":
    unittest.main()
