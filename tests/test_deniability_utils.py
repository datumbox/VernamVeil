import unittest
import itertools

from vernamveil._deniability_utils import forge_plausible_fx
from vernamveil._fx_utils import generate_default_fx
from vernamveil._vernamveil import VernamVeil


class TestDeniabilityUtils(unittest.TestCase):
    """Unit tests for plausible deniability utilities."""

    def _run_deniability_test(self,
            chunk_size=31,
            delimiter_size=9,
            padding_range=(5, 15),
            decoy_ratio=0.2,
            vectorise=True,
    ):
        """Utility to run a basic deniability test with configurable parameters."""
        real_fx = generate_default_fx(vectorise=vectorise)
        cypher = VernamVeil(
            real_fx,
            chunk_size=chunk_size,
            delimiter_size=delimiter_size,
            padding_range=padding_range,
            decoy_ratio=decoy_ratio,
            siv_seed_initialisation=True,
            auth_encrypt=True,
            vectorise=vectorise,
        )
        real_seed = VernamVeil.get_initial_seed()
        secret_message = b"Sensitive data: the launch code is 12345!"
        cyphertext, _ = cypher.encode(secret_message, real_seed)

        decoy_message = (
            b"This message is totally real and not at all a decoy... "
            b"There is nothing worth seeing here, move along!!! "
        )
        plausible_fx, fake_seed = forge_plausible_fx(cypher, cyphertext, decoy_message)

        fake_cypher = VernamVeil(
            plausible_fx,
            chunk_size=cypher._chunk_size,
            delimiter_size=cypher._delimiter_size,
            padding_range=cypher._padding_range,
            decoy_ratio=cypher._decoy_ratio,
            siv_seed_initialisation=False,
            auth_encrypt=False,
            vectorise=cypher._vectorise,
        )
        decoy_out, _ = fake_cypher.decode(cyphertext, fake_seed)
        return decoy_out.decode(errors="replace"), decoy_message.decode(errors="replace")

    def _combo_name(self, chunk_size, delimiter_size, padding_range, decoy_ratio):
        return (
            f"chunk{chunk_size}_delim{delimiter_size}_pad{padding_range}_decoy{decoy_ratio}"
        )


# Generate all combinations
chunk_sizes = [31, 32, 33]
delimiter_sizes = [7, 8, 9]
combos = list(itertools.product(chunk_sizes, delimiter_sizes))

def make_test_func(chunk_size, delimiter_size):
    """Create a test function for a specific combination of parameters."""
    def test_func(self):
        """Test that the deniability function works correctly for a specific combo."""
        for vectorise in [True, False]:
            with self.subTest(vectorise=vectorise):
                try:
                    decoy_out, decoy_message = self._run_deniability_test(
                        chunk_size=chunk_size,
                        delimiter_size=delimiter_size,
                        padding_range=(5, 15),
                        decoy_ratio=0.3,
                        vectorise=vectorise,
                    )
                    self.assertEqual(decoy_out, decoy_message)
                except ValueError as e:
                    self.assertIn("Could not find obfuscated decoy of length", str(e))
                    self.skipTest("Could not find a decoy for this configuration.")

    return test_func

# Dynamically add a test method for each combo except vectorise
for chunk_size, delimiter_size in combos:
    test_name = (
        f"test_deniability_{chunk_size}_{delimiter_size}"
    )
    test_func = make_test_func(chunk_size, delimiter_size)
    setattr(TestDeniabilityUtils, test_name, test_func)

if __name__ == "__main__":
    unittest.main()

