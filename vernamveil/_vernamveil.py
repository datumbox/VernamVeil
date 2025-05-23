"""Implements the VernamVeil stream cypher and related utilities.

Defines the main encryption class and core cryptographic operations.
"""

import hashlib
import hmac
import math
import secrets
import time
from typing import Any, Iterator

from vernamveil._cypher import _Cypher, np
from vernamveil._fx_utils import FX
from vernamveil._hash_utils import fold_bytes_to_uint64, hash_numpy

__all__ = ["VernamVeil"]


class VernamVeil(_Cypher):
    """VernamVeil is a modular, symmetric stream cypher.

    Inspired by One-Time Pad principles, it features customisable keystream generation, synthetic IV seed initialisation,
    stateful seed evolution for avalanche effects, authenticated encryption, and layered message obfuscation (chunk
    shuffling, padding, decoy injection). Supports vectorised operations (NumPy) and optional C-backed hashing for
    performance. Designed for educational and experimental use.
    """

    def __init__(
        self,
        fx: FX,
        chunk_size: int = 32,
        delimiter_size: int = 8,
        padding_range: tuple[int, int] = (5, 15),
        decoy_ratio: float = 0.1,
        siv_seed_initialisation: bool = True,
        auth_encrypt: bool = True,
    ):
        """Initialise the VernamVeil encryption cypher with configurable parameters.

        Args:
            fx (FX): A callable object that generates keystream bytes. This function is critical for the
                encryption process and should be carefully designed to ensure cryptographic security.
            chunk_size (int): Size of message chunks. Defaults to 32.
            delimiter_size (int): The delimiter size in bytes used for separating chunks; must be
                at least 4. Defaults to 8.
            padding_range (tuple[int, int]): Range for padding length before and after
                chunks. Defaults to (5, 15).
            decoy_ratio (float): Proportion of decoy chunks to insert. Must not be negative. Defaults to 0.1.
            siv_seed_initialisation (bool): Enables synthetic IV seed initialisation based on the message to
                resist seed reuse. Defaults to True.
            auth_encrypt (bool): Enables authenticated encryption with integrity check. Defaults to True.

        Raises:
            ValueError: If `chunk_size` is less than 8.
            ValueError: If `delimiter_size` is less than 4.
            TypeError: If `padding_range` is not a tuple of two integers.
            ValueError: If `padding_range` values are negative.
            ValueError: If `padding_range` values are not in ascending order.
            ValueError: If `decoy_ratio` is negative.
        """
        # Validate input
        if chunk_size < 8:
            raise ValueError("chunk_size must be at least 8 bytes.")
        if delimiter_size < 4:
            raise ValueError("delimiter_size must be at least 4 bytes.")
        if not (
            isinstance(padding_range, tuple)
            and len(padding_range) == 2
            and all(isinstance(x, int) for x in padding_range)
        ):
            raise TypeError("padding_range must be a tuple of two integers.")
        elif padding_range[0] < 0 or padding_range[1] < 0:
            raise ValueError("padding_range values must be non-negative.")
        elif padding_range[0] > padding_range[1]:
            raise ValueError("padding_range values must be in ascending order.")
        if decoy_ratio < 0:
            raise ValueError("decoy_ratio must not be negative.")

        # Initialise instance variables
        self._fx = fx
        self._chunk_size = chunk_size
        self._delimiter_size = delimiter_size
        self._padding_range = padding_range
        self._decoy_ratio = decoy_ratio
        self._siv_seed_initialisation = siv_seed_initialisation
        self._auth_encrypt = auth_encrypt

    def __str__(self) -> str:
        """Return a string representation of the VernamVeil instance.

        Returns:
            str: A string representation of the VernamVeil instance, including its parameters.
        """
        return (
            f"VernamVeil(chunk_size={self._chunk_size}, "
            f"delimiter_size={self._delimiter_size}, "
            f"padding_range={self._padding_range}, "
            f"decoy_ratio={self._decoy_ratio}, "
            f"siv_seed_initialisation={self._siv_seed_initialisation}, "
            f"auth_encrypt={self._auth_encrypt}"
        )

    @classmethod
    def get_initial_seed(cls, num_bytes: int = 64) -> bytes:
        """Generate a cryptographically secure initial random seed.

        This method uses the `secrets` module to generate a random sequence of bytes
        suitable for cryptographic use. It returns a byte string of the specified length.

        Args:
            num_bytes (int): The number of bytes to generate for the seed.
                Defaults to 64 bytes if not provided.

        Returns:
            bytes: A random byte string of the specified length.

        Raises:
            TypeError: If `num_bytes` is not an integer.
            ValueError: If `num_bytes` is not a positive integer.
        """
        if not isinstance(num_bytes, int):
            raise TypeError("num_bytes must be an integer.")
        if num_bytes <= 0:
            raise ValueError("num_bytes must be a positive integer.")

        return secrets.token_bytes(num_bytes)

    def _hash(
        self,
        key: bytes | bytearray | memoryview,
        msg_list: list[bytes | memoryview],
        use_hmac: bool = False,
    ) -> bytes:
        """Generate a Keyed Hash or Hash-based Message Authentication Code (HMAC).

        Each element in `msg_list` is sequentially fed into the Hash as message data.

        Args:
            key (bytes or bytearray or memoryview): The key for the keyed hash or HMAC.
            msg_list (list of bytes or memoryview): List of message parts to hash with the key.
            use_hmac (bool): If True, the key is used for HMAC; otherwise, it's a keyed hash. Defaults to False.

        Returns:
            bytes: The resulting hash digest.
        """
        if use_hmac:
            hm = hmac.new(key, msg_list[0], digestmod="blake2b")
            for i in range(1, len(msg_list)):
                hm.update(msg_list[i])
            return hm.digest()
        else:
            hasher = hashlib.blake2b(key=key)
            for m in msg_list:
                hasher.update(m)
            return hasher.digest()

    def _determine_shuffled_indices(
        self, seed: bytes, real_count: int, total_count: int
    ) -> list[int]:
        """Implement the Fisher–Yates shuffle algorithm.

        Determines the shuffled positions for real chunks based on a deterministic seed.

        Args:
            seed (bytes): Seed for deterministic shuffling.
            real_count (int): Number of real chunks.
            total_count (int): Total number of chunks (real + decoy).

        Returns:
            list[int]: Shuffled indices for real message chunks.
        """
        # Create a list with all positions
        positions = list(range(total_count))

        hashes: Any
        if self._fx.vectorise:
            # Vectorised: generate all hashes at once
            i_arr = np.arange(1, total_count, dtype=np.uint64)
            hashes = fold_bytes_to_uint64(hash_numpy(i_arr, seed, "blake2b"))
        else:
            # Standard: generate hashes one by one
            hashes = [
                int.from_bytes(self._hash(seed, [i.to_bytes(8, "big")]), "big")
                for i in range(1, total_count)
            ]

        # Shuffle deterministically based on the hashed seed
        for i in range(total_count - 1, 0, -1):
            # Create a random number between 0 and i
            j = hashes[i - 1] % (i + 1)

            # Swap elements at positions i and j
            positions[i], positions[j] = positions[j], positions[i]

        return positions[:real_count]

    def _generate_bytes(self, length: int, seed: bytes) -> memoryview:
        """Produce a byte stream of the given length using the key generator function.

        It samples `fx.block_size` bytes at a time.

        Args:
            length (int): Number of bytes to generate.
            seed (bytes): Seed for key generation.

        Returns:
            memoryview: Generated byte stream.
        """
        if self._fx.vectorise:
            # Vectorised generation using numpy
            # Generate enough uint64s to cover the length
            n_uint64 = math.ceil(length / self._fx.block_size)
            indices = np.arange(1, n_uint64 + 1, dtype=np.uint64)
            # Generate uint8 for bytes
            keystream = self._fx(indices, seed)
            # Flatten the array to 1D and slice to the required length
            keystream = keystream.ravel()
            memview: memoryview = keystream[:length].data
            return memview
        else:
            # Standard generation using python
            result = bytearray()
            i = 1
            total = 0
            while total < length:
                # Generate bytes
                val = self._fx(i, seed)
                result.extend(val)
                total += len(val)
                i += 1
            return memoryview(result)[:length]

    def _generate_chunk_ranges(self, message_len: int) -> Iterator[tuple[int, int]]:
        """Split a message into chunk index ranges based on the configured chunk size.

        Args:
            message_len (int): Length of the message in bytes.

        Returns:
            Iterator[tuple[int, int]]: An iterator with (start, end) indices for each chunk.
        """
        return (
            (i, min(i + self._chunk_size, message_len))
            for i in range(0, message_len, self._chunk_size)
        )

    def _obfuscate(self, cyphertext: bytearray, seed: bytes) -> bytearray:
        """Inject noise and padding into the cyphertext and shuffle the real chunk positions.

        Args:
            cyphertext (bytearray): The cyphertext of the message.
            seed (bytes): Seed for deterministic shuffling of chunks.

        Returns:
            bytearray: Obfuscated cyphertext with shuffled real and decoy chunks.
        """
        # Produce a unique seed for shuffling and to match the order of operations with deobfuscate.
        shuffle_seed = self._hash(seed, [b"shuffle"])

        # Estimate the number of real and fake chunks
        cyphertext_len = len(cyphertext)
        real_count = math.ceil(cyphertext_len / self._chunk_size)
        decoy_count = max(1, int(self._decoy_ratio * real_count)) if self._decoy_ratio > 0 else 0
        total_count = real_count + decoy_count

        # Estimate shuffled positions of real chunks
        shuffled_positions = self._determine_shuffled_indices(shuffle_seed, real_count, total_count)

        # Use the randomness of the positions to shuffle the chunks
        chunk_ranges_iter = self._generate_chunk_ranges(cyphertext_len)
        shuffled_chunk_ranges = [(-1, -1) for _ in range(total_count)]
        for i in shuffled_positions:
            shuffled_chunk_ranges[i] = next(chunk_ranges_iter)

        # Build the noisy cyphertext by combining fake and shuffled real chunks
        noisy_blocks = bytearray()
        pad_min, pad_max = self._padding_range
        view = memoryview(cyphertext)
        for i in range(total_count):
            if shuffled_chunk_ranges[i][0] != -1:  # real chunk location
                start, end = shuffled_chunk_ranges[i]
                chunk: memoryview | bytes = view[start:end]
            else:
                chunk = secrets.token_bytes(self._chunk_size)

            # Pre-pad
            pre_pad_len = (
                secrets.randbelow(pad_max - pad_min + 1) + pad_min
                if pad_max != pad_min
                else pad_min
            )
            if pre_pad_len > 0:
                noisy_blocks.extend(secrets.token_bytes(pre_pad_len))
            delimiter, seed = self._generate_delimiter(seed)
            noisy_blocks.extend(delimiter)
            # Actual data
            noisy_blocks.extend(chunk)
            # Post-pad
            delimiter, seed = self._generate_delimiter(seed)
            noisy_blocks.extend(delimiter)
            post_pad_len = (
                secrets.randbelow(pad_max - pad_min + 1) + pad_min
                if pad_max != pad_min
                else pad_min
            )
            if post_pad_len > 0:
                noisy_blocks.extend(secrets.token_bytes(post_pad_len))

        return noisy_blocks

    def _deobfuscate(self, noisy_cyphertext: bytearray, seed: bytes) -> bytearray:
        """Remove noise and extract real chunks from a shuffled noisy cyphertext.

        Args:
            noisy_cyphertext (bytearray): The obfuscated cyphertext of the message.
            seed (bytes): Seed for deterministic chunk deshuffling.

        Returns:
            bytearray: Original message reconstructed from real chunks.
        """
        # Produce a unique seed for shuffling and to match the order of operations with obfuscate.
        shuffle_seed = self._hash(seed, [b"shuffle"])

        # Estimate the ranges of all chunks
        delimiter_len = self._delimiter_size
        all_chunk_ranges: list[tuple[int, int]] = []
        prev_idx = None  # Tracks the index of the previous delimiter found
        look_start = 0  # Start position for searching the next delimiter
        while True:
            # Generate a new delimiter
            delimiter, seed = self._generate_delimiter(seed)

            # Search for the next occurrence of the delimiter
            idx = noisy_cyphertext.find(delimiter, look_start)
            if idx == -1:
                # No more delimiters found
                break
            if prev_idx is not None:
                # We have found a pair of delimiters:
                # The chunk starts after the previous delimiter and ends at the current one
                all_chunk_ranges.append((prev_idx + delimiter_len, idx))
                prev_idx = None  # Reset to look for the next pair
            else:
                # Store the index of the first delimiter in the pair
                prev_idx = idx
            # Move the search start past the current delimiter
            look_start = idx + delimiter_len

        # Determine the number of real chunks
        total_count = len(all_chunk_ranges)
        if self._decoy_ratio > 0:
            # Approximate guess for real and decoy counts
            real_count = int(total_count / (1 + self._decoy_ratio))
            decoy_count = max(1, int(self._decoy_ratio * real_count))

            # Adjust by at most one step if needed to match the total count
            diff = total_count - real_count - decoy_count
            if diff > 0:
                real_count += 1
            elif diff < 0:
                real_count -= 1
        else:
            real_count = total_count

        # Estimate the shuffled real positions
        # TODO: this call is out of order with the obfuscate call. It won't work for OTP. It needs to go on top but
        #      it is not possible without storing the total_count in the meta-data. Recovering total_count after the
        #      delimiter generation consumes the seed (which is fixable via producing a shuffle seed) but it also
        #      leads to out-of-order calls on fx which messes up the OTP.
        shuffled_positions = self._determine_shuffled_indices(shuffle_seed, real_count, total_count)

        # Reconstruct and unshuffle the message
        message = bytearray()
        view = memoryview(noisy_cyphertext)
        for pos in shuffled_positions:
            start, end = all_chunk_ranges[pos]
            message.extend(view[start:end])

        return message

    def _xor_with_key(
        self, data: memoryview, seed: bytes, is_encode: bool
    ) -> tuple[bytearray, bytes]:
        """Encrypt or decrypt data using XOR with the generated keystream.

        Args:
            data (memoryview): Input data to process.
            seed (bytes): Seed for keystream generation.
            is_encode (bool): True for encryption, False for decryption.

        Returns:
            tuple[bytearray, bytes]: Processed data and the final seed.
        """
        # Preallocate memory and avoid copying when slicing
        data_len = len(data)
        result = bytearray(data_len)
        if self._fx.vectorise:
            # Create a numpy array on top of the bytearray to vectorise and still have access to original bytearray
            processed = np.frombuffer(result, dtype=np.uint8)
        else:
            processed = result

        for start, end in self._generate_chunk_ranges(data_len):
            # Generate a key using fx
            chunk_len = end - start
            keystream = self._generate_bytes(chunk_len, seed)

            # XOR the chunk with the key
            if self._fx.vectorise:
                np.bitwise_xor(data[start:end], keystream, out=processed[start:end])
                plaintext_data = data[start:end] if is_encode else processed[start:end].data
            else:
                for i in range(chunk_len):
                    pos = start + i
                    processed[pos] = data[pos] ^ keystream[i]
                plaintext_data = data[start:end] if is_encode else memoryview(processed)[start:end]

            # Refresh the seed differently for encoding and decoding
            seed = self._hash(seed, [plaintext_data])

        return result, seed

    def _generate_delimiter(self, seed: bytes) -> tuple[memoryview, bytes]:
        """Create a delimiter sequence using the key stream and update the seed.

        Args:
            seed (bytes): Seed used for generating the delimiter.

        Returns:
            tuple[memoryview, bytes]: The delimiter and the refreshed seed.
        """
        delimiter = self._generate_bytes(self._delimiter_size, seed)
        seed = self._hash(seed, [b"delimiter"])
        return delimiter, seed

    def encode(
        self, message: bytes | bytearray | memoryview, seed: bytes
    ) -> tuple[bytearray, bytes]:
        """Encrypt a message.

        Args:
            message (bytes or bytearray or memoryview): Message to encode.
            seed (bytes): Initial seed for encryption.

        Returns:
            tuple[bytearray, bytes]: Encrypted message and final seed.
        """
        # Convert to memoryview for efficient slicing
        if not isinstance(message, memoryview):
            message = memoryview(message)

        # SIV seed initialisation: Encrypt and prepend a synthetic IV (SIV) derived from the seed and message.
        # This prevents deterministic keystreams on the first block and makes the scheme resilient to seed reuse.
        if self._siv_seed_initialisation:
            # Generate the SIV hash from the initial seed, the timestamp and the message
            timestamp = time.time_ns().to_bytes(8, "big")
            siv_hash = self._hash(seed, [message, timestamp])
            # Encrypt the synthetic IV and evolve the seed with it
            encrypted_siv_hash, seed = self._xor_with_key(memoryview(siv_hash), seed, True)
            # Use the encrypted SIV hash bytearray as the output; this puts it in front
            output = encrypted_siv_hash

            # Note: The SIV is not reused for MAC computation, ensuring separation
            # between seed evolution and authentication.
        else:
            output = bytearray()

        # Produce a unique seed for Authenticated Encryption
        auth_seed = self._hash(seed, [b"auth"]) if self._auth_encrypt else b""

        # Produce a unique seed for Obfuscation to avoid reusing the same seed during shuffling and to match the order
        # of operations with decode.
        obfuscate_seed = self._hash(seed, [b"obfuscate"])

        # Encrypt the message
        cyphertext, last_seed = self._xor_with_key(message, seed, True)

        # Add noise and shuffle the cyphertext
        noisy_cyphertext = self._obfuscate(cyphertext, obfuscate_seed)
        output.extend(noisy_cyphertext)

        # Authenticated Encryption
        if self._auth_encrypt:
            # The tag is computed over the noisy cyphertext and the configuration of the cypher
            tag = self._hash(auth_seed, [noisy_cyphertext, str(self).encode()], use_hmac=True)
            output.extend(tag)

        return output, last_seed

    def decode(
        self, cyphertext: bytes | bytearray | memoryview, seed: bytes
    ) -> tuple[bytearray, bytes]:
        """Decrypt an encoded message.

        Args:
            cyphertext (bytes or bytearray or memoryview): Encrypted and obfuscated message.
            seed (bytes): Initial seed for decryption.

        Returns:
            tuple[bytearray, bytes]: Decrypted message and final seed.

        Raises:
            ValueError: If the authentication tag does not match.
        """
        # Convert to memoryview for efficient slicing
        if not isinstance(cyphertext, memoryview):
            cyphertext = memoryview(cyphertext)

        HASH_LENGTH = 64

        # SIV seed initialisation: Decrypt and consume the synthetic IV (SIV) to reconstruct the evolved seed.
        # This ensures the keystream remains unique and prevents deterministic decryption on the first block.
        if self._siv_seed_initialisation:
            # Split the data by taking the first HASH_LENGTH bytes
            encrypted_siv_hash, cyphertext = cyphertext[:HASH_LENGTH], cyphertext[HASH_LENGTH:]
            # Decrypt the SIV hash (throw away) and evolve the seed with it
            _, seed = self._xor_with_key(encrypted_siv_hash, seed, False)

        # Authenticated Encryption
        if self._auth_encrypt:
            # Split the data by taking the last HASH_LENGTH bytes
            encrypted_data, expected_tag = cyphertext[:-HASH_LENGTH], cyphertext[-HASH_LENGTH:]

            # Produce a unique seed for Authenticated Encryption
            auth_seed = self._hash(seed, [b"auth"])

            # Estimate the tag and compare it with the expected
            tag = self._hash(auth_seed, [encrypted_data, str(self).encode()], use_hmac=True)
            if not hmac.compare_digest(tag, expected_tag):
                raise ValueError("Authentication failed: MAC tag mismatch.")
        else:
            encrypted_data = cyphertext

        # Produce a unique seed for Obfuscation to avoid reusing the same seed during unshuffling and to match the order
        # of operations with encode.
        obfuscate_seed = self._hash(seed, [b"obfuscate"])

        # Denoise, Unshuffle and extract the real message
        denoised_cyphertext = self._deobfuscate(bytearray(encrypted_data), obfuscate_seed)

        # Decrypt the message
        message, last_seed = self._xor_with_key(memoryview(denoised_cyphertext), seed, False)

        return message, last_seed
