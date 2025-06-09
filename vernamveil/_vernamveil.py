"""Implements the VernamVeil stream cypher and related utilities.

Defines the main encryption class and core cryptographic operations.
"""

import hashlib
import hmac
import math
import secrets
import time
from functools import partial
from typing import Any, Callable, Iterator, cast

import numpy as np

from vernamveil._bytesearch import find, find_all
from vernamveil._cypher import _Cypher
from vernamveil._fx_utils import FX
from vernamveil._hash_utils import blake3, fold_bytes_to_uint64, hash_numpy
from vernamveil._types import HashType

__all__ = ["VernamVeil"]


class VernamVeil(_Cypher):
    """VernamVeil is a modular, symmetric stream cypher.

    Inspired by One-Time Pad principles, it features customisable keystream generation, synthetic IV seed initialisation,
    stateful seed evolution for avalanche effects, authenticated encryption, and layered message obfuscation (chunk
    shuffling, padding, decoy injection). Supports vectorised operations (via NumPy) and optional C-backed hashing for
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
        hash_name: HashType = "blake2b",
    ) -> None:
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
            hash_name (HashType): Hash function to use ("blake2b", "blake3" or "sha256") for keyed hashing
                and HMAC. The blake3 is only available if the C extension is installed.  Defaults to "blake2b".

        Raises:
            ValueError: If `chunk_size` is less than 8.
            ValueError: If `delimiter_size` is less than 4.
            TypeError: If `padding_range` is not a tuple of two integers.
            ValueError: If `padding_range` values are negative.
            ValueError: If `padding_range` values are not in ascending order.
            ValueError: If `decoy_ratio` is negative.
            ValueError: If `hash_name` is not "blake2b", "blake3" or "sha256".
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
        if hash_name not in ("blake2b", "blake3", "sha256"):
            raise ValueError("hash_name must be either 'blake2b', 'blake3' or 'sha256'.")

        # Initialise instance variables
        self._fx = fx
        self._chunk_size = chunk_size
        self._delimiter_size = delimiter_size
        self._padding_range = padding_range
        self._decoy_ratio = decoy_ratio
        self._siv_seed_initialisation = siv_seed_initialisation
        self._auth_encrypt = auth_encrypt
        self._hash_name = hash_name

        # Constants
        if hash_name == "blake3":
            self._HASH_METHOD: Callable[..., Any] = blake3
        else:
            self._HASH_METHOD = partial(hashlib.new, hash_name)
        self._HASH_LENGTH = self._HASH_METHOD().digest_size

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
            f"auth_encrypt={self._auth_encrypt},"
            f"hash_name={self._hash_name})"
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
        elif num_bytes <= 0:
            raise ValueError("num_bytes must be a positive integer.")

        return secrets.token_bytes(num_bytes)

    def _hash(
        self,
        key: bytes | bytearray | memoryview,
        msg_list: list[bytes | bytearray | memoryview],
        use_hmac: bool = False,
    ) -> bytes | bytearray:
        """Generate a Keyed Hash or Hash-based Message Authentication Code (HMAC).

        Each element in `msg_list` is sequentially fed into the Hash as message data.

        Args:
            key (bytes or bytearray or memoryview): The key for the keyed hash or HMAC.
            msg_list (list of bytes or bytearray or memoryview): List of message parts to hash with the key.
            use_hmac (bool): If True, the key is used for HMAC; otherwise, it's a keyed hash. Defaults to False.

        Returns:
            bytes or bytearray: The resulting hash digest.
        """
        n = len(msg_list)
        key = cast(bytes, key)
        if use_hmac or self._hash_name == "sha256":  # sha256 does not support key argument
            hasher = hmac.new(key, msg=msg_list[0] if n > 0 else None, digestmod=self._HASH_METHOD)
        else:
            hasher = self._HASH_METHOD(msg_list[0] if n > 0 else b"", key=key)
        for i in range(1, n):
            hasher.update(msg_list[i])
        return hasher.digest()

    def _determine_shuffled_indices(
        self, seed: bytes | bytearray, real_count: int, total_count: int
    ) -> list[int]:
        """Implement the Fisher–Yates shuffle algorithm.

        Determines the shuffled positions for real chunks based on a deterministic seed.

        Args:
            seed (bytes or bytearray): Seed for deterministic shuffling.
            real_count (int): Number of real chunks.
            total_count (int): Total number of chunks (real + decoy).

        Returns:
            list[int]: Shuffled indices for real message chunks.
        """
        # Create a list with all positions
        positions = list(range(total_count))

        # Calculate the number of raw hash primitive outputs required.
        # Each uint64 needs 8 bytes. self._HASH_LENGTH is bytes per raw hash output.
        num_uint64_needed = total_count - 1
        num_bytes_needed = 8 * num_uint64_needed
        num_raw_hash_outputs = math.ceil(num_bytes_needed / self._HASH_LENGTH)

        # Generate input indices for these raw hash outputs.
        i_arr = np.arange(num_raw_hash_outputs, dtype=np.uint64)

        # Get the raw bytes from hashing these indices.
        raw_bytes = hash_numpy(i_arr, seed, self._hash_name, self._HASH_LENGTH)

        # Truncate the raw bytes to the exact total number of bytes needed for the uint64s.
        truncated_bytes = raw_bytes.ravel()[:num_bytes_needed].reshape(num_uint64_needed, 8)

        # Fold these bytes into an array of uint64s.
        random_ints = fold_bytes_to_uint64(truncated_bytes)

        # Shuffle deterministically based on the hashed seed
        for i in range(total_count - 1, 0, -1):
            # Create a random number between 0 and i
            j = random_ints[i - 1] % (i + 1)

            # Swap elements at positions i and j
            positions[i], positions[j] = positions[j], positions[i]

        return positions[:real_count]

    def _generate_bytes(self, length: int, seed: bytes | bytearray) -> memoryview:
        """Produce a byte stream of the given length using the key generator function.

        It samples `fx.block_size` bytes at a time.

        Args:
            length (int): Number of bytes to generate.
            seed (bytes or bytearray): Seed for key generation.

        Returns:
            memoryview: Generated byte stream.
        """
        # Generate enough uint64s to cover the length
        n_uint64 = math.ceil(length / self._fx.block_size)
        indices = np.arange(n_uint64, dtype=np.uint64)
        # Generate uint8 for bytes
        keystream = self._fx(indices, seed)
        # Flatten the array to 1D and slice to the required length
        return keystream.ravel()[:length].data

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

    def _obfuscate(
        self, message: memoryview, seed: bytes | bytearray, delimiter: memoryview
    ) -> memoryview:
        """Inject noise and padding into the message and shuffle the real chunk positions.

        Args:
            message (memoryview): Original message.
            seed (bytes or bytearray): Seed for deterministic shuffling of chunks.
            delimiter (memoryview): Chunk delimiter.

        Returns:
            memoryview: Obfuscated message with shuffled real and decoy chunks.
        """
        # Estimate the number of real and fake chunks
        message_len = len(message)
        real_count = math.ceil(message_len / self._chunk_size)
        decoy_count = max(1, int(self._decoy_ratio * real_count)) if self._decoy_ratio > 0 else 0
        total_count = real_count + decoy_count

        # Estimate shuffled positions of real chunks
        shuffled_positions = self._determine_shuffled_indices(seed, real_count, total_count)

        # Use the randomness of the positions to shuffle the chunks
        chunk_ranges_iter = self._generate_chunk_ranges(message_len)
        shuffled_chunk_ranges: list[None | tuple[int, int]] = [None] * total_count
        for i in shuffled_positions:
            shuffled_chunk_ranges[i] = next(chunk_ranges_iter)

        # Precompute all pre/post pad lengths and generate all random bytes
        chunk_size = self._chunk_size
        pad_min, pad_max = self._padding_range
        pad_width = pad_max - pad_min + 1
        pad_count = 2 * total_count
        pad_lens = [
            secrets.randbelow(pad_width) + pad_min if pad_max != pad_min else pad_min
            for _ in range(pad_count)
        ]  # the order of padding lengths is not important; we pop in reverse order
        random_size = sum(pad_lens) + decoy_count * chunk_size
        random_bytes = memoryview(secrets.token_bytes(random_size))

        # Calculate the exact size from the delimiters, message, and random bytes
        exact_size = pad_count * self._delimiter_size + random_size + message_len

        # Build the noisy message by combining fake and shuffled real chunks
        noisy_blocks = np.empty(exact_size, dtype=np.uint8)
        current_rec_loc = 0
        current_rnd_loc = 0
        for chunk_range in shuffled_chunk_ranges:
            record = []

            # Pre-pad
            pre_pad_len = pad_lens.pop()
            if pre_pad_len > 0:
                next_rnd_loc = current_rnd_loc + pre_pad_len
                record.append(random_bytes[current_rnd_loc:next_rnd_loc])
                current_rnd_loc = next_rnd_loc
            record.append(delimiter)

            # Actual data
            if chunk_range is not None:
                start, end = chunk_range
                record.append(message[start:end])
            else:
                next_rnd_loc = current_rnd_loc + chunk_size
                record.append(random_bytes[current_rnd_loc:next_rnd_loc])
                current_rnd_loc = next_rnd_loc

            # Post-pad
            record.append(delimiter)
            post_pad_len = pad_lens.pop()
            if post_pad_len > 0:
                next_rnd_loc = current_rnd_loc + post_pad_len
                record.append(random_bytes[current_rnd_loc:next_rnd_loc])
                current_rnd_loc = next_rnd_loc

            # Add the record to the noisy blocks
            for part in record:
                next_rec_loc = current_rec_loc + len(part)
                noisy_blocks[current_rec_loc:next_rec_loc] = part
                current_rec_loc = next_rec_loc

        return noisy_blocks.data

    def _deobfuscate(
        self, noisy: memoryview, seed: bytes | bytearray, delimiter: memoryview
    ) -> memoryview:
        """Remove noise and extract real chunks from a shuffled noisy message.

        Args:
            noisy (memoryview): Decrypted and obfuscated message.
            seed (bytes or bytearray): Seed for deterministic chunk deshuffling.
            delimiter (memoryview): Delimiter used to detect chunks.

        Returns:
            memoryview: Original message reconstructed from real chunks.
        """
        # Estimate the ranges of all chunks
        delimiter_len = len(delimiter)
        locations = find_all(noisy, delimiter)
        all_chunk_ranges = [
            # Take a pair of delimiter locations.
            # The chunk starts after the first delimiter and ends at the second one
            (locations[i] + delimiter_len, locations[i + 1])
            for i in range(0, len(locations) - 1, 2)
        ]

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
        shuffled_positions = self._determine_shuffled_indices(seed, real_count, total_count)

        # Calculate the exact size from the chunk ranges
        exact_size = sum(
            end - start for start, end in (all_chunk_ranges[pos] for pos in shuffled_positions)
        )

        # Reconstruct and unshuffle the message
        message = np.empty(exact_size, dtype=np.uint8)
        current_loc = 0
        for pos in shuffled_positions:
            start, end = all_chunk_ranges[pos]

            next_loc = current_loc + end - start
            message[current_loc:next_loc] = noisy[start:end]
            current_loc = next_loc

        return message.data

    def _xor_with_key(
        self, data: memoryview, seed: bytes | bytearray, is_encode: bool
    ) -> tuple[memoryview, bytes | bytearray]:
        """Encrypt or decrypt data using XOR with the generated keystream.

        Args:
            data (memoryview): Input data to process.
            seed (bytes or bytearray): Seed for keystream generation.
            is_encode (bool): True for encryption, False for decryption.

        Returns:
            tuple[memoryview, bytes or bytearray]: Processed data and the final seed.
        """
        # Preallocate memory
        data_len = len(data)
        result = np.empty(data_len, dtype=np.uint8)

        for start, end in self._generate_chunk_ranges(data_len):
            # Generate a key using fx
            chunk_len = end - start
            keystream = self._generate_bytes(chunk_len, seed)

            # XOR the chunk with the key
            # Store the slicing to avoid duplicate ops
            data_slice = data[start:end]
            processed_slice = result[start:end]

            # Writing to slices modifies the original data
            np.bitwise_xor(data_slice, keystream, out=processed_slice)
            plaintext_data = data_slice if is_encode else processed_slice.data

            # Refresh the seed differently for encoding and decoding
            seed = self._hash(seed, [plaintext_data])

        return result.data, seed

    def _generate_delimiter(self, seed: bytes | bytearray) -> tuple[memoryview, bytes | bytearray]:
        """Create a delimiter sequence using the key stream and update the seed.

        Args:
            seed (bytes or bytearray): Seed used for generating the delimiter.

        Returns:
            tuple[memoryview, bytes or bytearray]: The delimiter and the refreshed seed.
        """
        delimiter = self._generate_bytes(self._delimiter_size, seed)
        seed = self._hash(seed, [delimiter])
        return delimiter, seed

    def encode(
        self, message: bytes | bytearray | memoryview, seed: bytes | bytearray
    ) -> tuple[memoryview, bytes | bytearray]:
        """Encrypt a message.

        Args:
            message (bytes or bytearray or memoryview): Message to encode.
            seed (bytes or bytearray): Initial seed for encryption.

        Returns:
            tuple[memoryview, bytes or bytearray]: Encrypted message and final seed.

        Raises:
            ValueError: If the delimiter appears in the message.
        """
        # Convert to memoryview for efficient slicing
        if isinstance(message, (bytes, bytearray)):
            message = memoryview(message)

        # Store the output parts
        output: list[memoryview] = []

        # SIV seed initialisation: Encrypt and prepend a synthetic IV (SIV) derived from the seed and message.
        # This prevents deterministic keystreams on the first block and makes the scheme resilient to seed reuse.
        if self._siv_seed_initialisation:
            # Generate the SIV hash from the initial seed, the timestamp and the message
            timestamp = time.time_ns().to_bytes(8, "big")
            siv_hash = self._hash(seed, [timestamp, message])
            # Encrypt the synthetic IV and evolve the seed with it
            encrypted_siv_hash, seed = self._xor_with_key(memoryview(siv_hash), seed, True)
            # Use the encrypted SIV hash bytearray as the output; this puts it in front
            output.append(encrypted_siv_hash)

            # Note: The SIV is not reused for MAC computation, ensuring separation
            # between seed evolution and authentication.

        # Produce a unique seed for Authenticated Encryption
        auth_seed = self._hash(seed, [b"auth"]) if self._auth_encrypt else b""

        # Generate the delimiter
        delimiter, seed = self._generate_delimiter(seed)

        # Delimiter conflict check
        if find(message, delimiter) != -1:
            raise ValueError(
                "The delimiter appears in the message. Consider increasing the delimiter size."
            )

        # Produce a unique seed for Obfuscation to avoid reusing the same seed during shuffling and to match the order
        # of operations with decode.
        shuffle_seed = self._hash(seed, [b"shuffle"])

        # Add noise and shuffle the message
        noisy = self._obfuscate(message, shuffle_seed, delimiter)

        # Encrypt the noisy message
        cyphertext, last_seed = self._xor_with_key(noisy, seed, True)
        output.append(cyphertext)

        # Authenticated Encryption
        if self._auth_encrypt:
            # The tag is computed over the configuration of the cypher and the cyphertext.
            tag = self._hash(auth_seed, [str(self).encode(), cyphertext], use_hmac=True)
            output.append(memoryview(tag))

        # Concatenate all parts into a single bytearray
        result = np.empty(sum(len(part) for part in output), dtype=np.uint8)
        current_loc = 0
        for part in output:
            next_loc = current_loc + len(part)
            result[current_loc:next_loc] = part
            current_loc = next_loc

        return result.data, last_seed

    def decode(
        self, cyphertext: bytes | bytearray | memoryview, seed: bytes | bytearray
    ) -> tuple[memoryview, bytes | bytearray]:
        """Decrypt an encoded message.

        Args:
            cyphertext (bytes or bytearray or memoryview): Encrypted and obfuscated message.
            seed (bytes or bytearray): Initial seed for decryption.

        Returns:
            tuple[memoryview, bytes or bytearray]: Decrypted message and final seed.

        Raises:
            ValueError: If the authentication tag does not match.
        """
        # Convert to memoryview for efficient slicing
        if not isinstance(cyphertext, memoryview):
            cyphertext = memoryview(cyphertext)

        # SIV seed initialisation: Decrypt and consume the synthetic IV (SIV) to reconstruct the evolved seed.
        # This ensures the keystream remains unique and prevents deterministic decryption on the first block.
        if self._siv_seed_initialisation:
            # Split the data by taking the first bytes
            encrypted_siv_hash, cyphertext = (
                cyphertext[: self._HASH_LENGTH],
                cyphertext[self._HASH_LENGTH :],
            )
            # Decrypt the SIV hash (throw away) and evolve the seed with it
            _, seed = self._xor_with_key(encrypted_siv_hash, seed, False)

        # Authenticated Encryption
        if self._auth_encrypt:
            # Split the data by taking the last bytes
            encrypted_data, expected_tag = (
                cyphertext[: -self._HASH_LENGTH],
                cyphertext[-self._HASH_LENGTH :],
            )

            # Produce a unique seed for Authenticated Encryption
            auth_seed = self._hash(seed, [b"auth"])

            # Estimate the tag and compare it with the expected
            tag = self._hash(auth_seed, [str(self).encode(), encrypted_data], use_hmac=True)
            if not hmac.compare_digest(tag, expected_tag):
                raise ValueError("Authentication failed: MAC tag mismatch.")
        else:
            encrypted_data = cyphertext

        # Generate the delimiter
        delimiter, seed = self._generate_delimiter(seed)

        # Produce a unique seed for Obfuscation to avoid reusing the same seed during unshuffling and to match the order
        # of operations with encode.
        shuffle_seed = self._hash(seed, [b"shuffle"])

        # Decrypt the noisy message
        decrypted, last_seed = self._xor_with_key(encrypted_data, seed, False)

        # Denoise, Unshuffle and extract the real message
        message = self._deobfuscate(decrypted, shuffle_seed, delimiter)

        return message, last_seed
