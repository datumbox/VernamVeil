import itertools
import tempfile
import unittest
from pathlib import Path

from vernamveil._cypher import _HAS_NUMPY
from vernamveil._deniability_utils import _PlausibleFX, forge_plausible_fx
from vernamveil._fx_utils import generate_default_fx, load_fx_from_file
from vernamveil._vernamveil import VernamVeil


class TestDeniabilityUtils(unittest.TestCase):
    """Unit tests for plausible deniability utilities."""

    def _run_deniability_test(
        self,
        chunk_size=31,
        delimiter_size=9,
        padding_range=(5, 15),
        decoy_ratio=0.2,
        backend="python",
    ):
        """Utility to run a basic deniability test with configurable parameters."""
        real_fx = generate_default_fx(backend=backend)
        cypher = VernamVeil(
            real_fx,
            chunk_size=chunk_size,
            delimiter_size=delimiter_size,
            padding_range=padding_range,
            decoy_ratio=decoy_ratio,
            siv_seed_initialisation=True,
            auth_encrypt=True,
            backend=backend,
        )
        real_seed = VernamVeil.get_initial_seed()
        secret_message = b"Sensitive data: the launch code is 12345!"
        cyphertext, _ = cypher.encode(secret_message, real_seed)

        decoy_message = (
            b"This message is totally real and not at all a decoy... "
            b"There is nothing worth seeing here, move along!!! "
        )
        plausible_fx, fake_seed = forge_plausible_fx(
            cypher, cyphertext, decoy_message, max_obfuscate_attempts=10000
        )

        fake_cypher = VernamVeil(
            plausible_fx,
            chunk_size=cypher._chunk_size,
            delimiter_size=cypher._delimiter_size,
            padding_range=cypher._padding_range,
            decoy_ratio=cypher._decoy_ratio,
            siv_seed_initialisation=False,
            auth_encrypt=False,
            backend=cypher._backend,
        )
        decoy_out, _ = fake_cypher.decode(cyphertext, fake_seed)
        return decoy_out.decode(errors="replace"), decoy_message.decode(errors="replace")

    def _combo_name(self, chunk_size, delimiter_size, padding_range, decoy_ratio):
        return f"chunk{chunk_size}_delim{delimiter_size}_pad{padding_range}_decoy{decoy_ratio}"

    def test_plausible_fx_source_code_roundtrip(self):
        """Test that _PlausibleFX source code can be saved and loaded, preserving integers."""
        test_ints = [42, 99, 123456, 7, 0]
        fx = _PlausibleFX(test_ints)
        source_code = fx._source_code

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fx_source.py"
            with open(path, "w") as f:
                f.write(source_code)
            loaded_fx = load_fx_from_file(str(path))
            self.assertTrue(hasattr(loaded_fx, "_uint64s"))
            self.assertEqual(list(loaded_fx._uint64s), test_ints)

    def test_end_to_end_deniability_disk_io(self):
        """End-to-end test: store/load all artifacts and verify deniability."""
        real_fx = generate_default_fx(backend="python")
        real_seed = VernamVeil.get_initial_seed()
        secret_message = b"Sensitive data: the launch code is 12345!"
        cypher = VernamVeil(
            real_fx,
            chunk_size=32,
            delimiter_size=8,
            padding_range=(5, 15),
            decoy_ratio=0.2,
            siv_seed_initialisation=True,
            auth_encrypt=True,
            backend="python",
        )
        cyphertext, _ = cypher.encode(secret_message, real_seed)

        decoy_message = (
            b"This message is totally real and not at all a decoy... "
            b"There is nothing worth seeing here, move along!!!"
        )
        try:
            plausible_fx, fake_seed = forge_plausible_fx(cypher, cyphertext, decoy_message)
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
            real_seed_path = tmpdir_path / "real_seed.bin"
            fake_seed_path = tmpdir_path / "fake_seed.bin"
            cyphertext_path = tmpdir_path / "cyphertext.bin"

            # Save everything to disk
            with open(real_fx_path, "w") as f:
                f.write(real_fx._source_code)
            with open(plausible_fx_path, "w") as f:
                f.write(plausible_fx._source_code)
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
                chunk_size=32,
                delimiter_size=8,
                padding_range=(5, 15),
                decoy_ratio=0.2,
                siv_seed_initialisation=True,
                auth_encrypt=True,
                backend="python",
            )
            real_out, _ = real_cypher.decode(loaded_cyphertext, loaded_real_seed)

            # Decrypt with plausible fx/fake seed
            fake_cypher = VernamVeil(
                loaded_plausible_fx,
                chunk_size=32,
                delimiter_size=8,
                padding_range=(5, 15),
                decoy_ratio=0.2,
                siv_seed_initialisation=False,
                auth_encrypt=False,
                backend="python",
            )
            decoy_out, _ = fake_cypher.decode(loaded_cyphertext, loaded_fake_seed)

            self.assertEqual(real_out, secret_message)
            self.assertEqual(decoy_out, decoy_message)


# Generate all combinations
chunk_sizes = [31, 32, 33]
delimiter_sizes = [7, 8, 9]
combos = list(itertools.product(chunk_sizes, delimiter_sizes))


def make_test_func(chunk_size, delimiter_size):
    """Create a test function for a specific combination of parameters."""

    def test_func(self):
        """Test that the deniability function works correctly for a specific combo."""
        backend_options = ["numpy", "python"] if _HAS_NUMPY else ["python"]
        for backend in backend_options:
            with self.subTest(backend=backend):
                try:
                    decoy_out, decoy_message = self._run_deniability_test(
                        chunk_size=chunk_size,
                        delimiter_size=delimiter_size,
                        padding_range=(5, 15),
                        decoy_ratio=0.3,
                        backend=backend,
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


# Dynamically add a test method for each combo except for backend
for chunk_size, delimiter_size in combos:
    test_name = f"test_deniability_{chunk_size}_{delimiter_size}"
    test_func = make_test_func(chunk_size, delimiter_size)
    setattr(TestDeniabilityUtils, test_name, test_func)

if __name__ == "__main__":
    unittest.main()
