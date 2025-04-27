import hashlib
import os
import random
import string
import unittest

from vernamveil import VernamVeil, generate_secret_fx

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TestVernamVeil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = generate_secret_fx(10)
        cls.initial_seed = VernamVeil.get_initial_seed()
        cls.cipher = VernamVeil(cls.fx)
        cls.ciphers = [(cls.cipher, "scalar")]

        if HAS_NUMPY:
            cls.fx_vec = generate_secret_fx(10, vectorise=True)
            cls.cipher_vec = VernamVeil(cls.fx_vec, vectorise=True)
            cls.ciphers.append((cls.cipher_vec, "vectorised"))

    def test_single_message_encryption(self):
        message = "This is a secret message that needs to be protected. Make sure it is hard to break!"
        for cipher, mode in self.ciphers:
            with self.subTest(mode=mode):
                encrypted, _ = cipher.encode(message.encode(), self.initial_seed)
                decrypted, _ = cipher.decode(encrypted, self.initial_seed)
                self.assertEqual(message, decrypted.decode())

    def test_variable_length_encryption(self):
        for cipher, mode in self.ciphers:
            with self.subTest(mode=mode):
                for i in range(500):
                    msg = "".join(random.choices(string.printable, k=i))
                    encrypted, _ = cipher.encode(msg.encode(), self.initial_seed)
                    decrypted, _ = cipher.decode(encrypted, self.initial_seed)
                    self.assertEqual(msg, decrypted.decode(), f"Failed at length {i}")

    @unittest.skipIf(not os.path.exists("/usr/bin/ls"), "Input file does not exist")
    def test_file_encryption(self):
        input_file = "/usr/bin/ls"
        output_file = "/tmp/ls.encoded"
        decoded_file = "/tmp/ls.decoded"

        for vectorise, mode in [(False, "scalar")] + ([(True, "vectorised")] if HAS_NUMPY else []):
            with self.subTest(mode=mode):
                pass
                fx = generate_secret_fx(20, vectorise=vectorise)

                # Encrypt and decrypt the file
                VernamVeil.process_file(input_file, output_file, fx, self.initial_seed, mode="encode", buffer_size=1024,
                                        chunk_size=128, vectorise=vectorise)
                VernamVeil.process_file(output_file, decoded_file, fx, self.initial_seed, mode="decode",
                                        buffer_size=1024,
                                        chunk_size=128, vectorise=vectorise)

                # Compare SHA256 checksums
                with open(input_file, "rb") as f1, open(decoded_file, "rb") as f2:
                    sha256_original = hashlib.sha256(f1.read()).hexdigest()
                    sha256_decoded = hashlib.sha256(f2.read()).hexdigest()

                self.assertEqual(sha256_original, sha256_decoded)

                # Clean up test files
                os.remove(output_file)
                os.remove(decoded_file)


if __name__ == "__main__":
    unittest.main()
