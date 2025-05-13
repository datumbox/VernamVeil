"""Implements the VernamVeil stream cypher and related utilities.

Defines the main encryption class and core cryptographic operations.
"""

import hashlib
import hmac
import math
import secrets
import time
import warnings
from typing import Any, Callable, Iterator

from vernamveil._cypher import _Cypher
from vernamveil._hash_utils import _UINT64_BOUND, fold_bytes_to_uint64, hash_numpy

np: Any
_IntOrArray: Any
try:
    import numpy
    from numpy.typing import NDArray

    np = numpy
    _IntOrArray = int | NDArray
    _HAS_NUMPY = True
except ImportError:
    np = None
    _IntOrArray = int
    _HAS_NUMPY = False


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
        fx: Callable[[_IntOrArray, bytes, int | None], _IntOrArray],
        chunk_size: int = 32,
        delimiter_size: int = 8,
        padding_range: tuple[int, int] = (5, 15),
        decoy_ratio: float = 0.1,
        siv_seed_initialisation: bool = True,
        auth_encrypt: bool = True,
        vectorise: bool = False,
    ):
        """Initialise the VernamVeil encryption cypher with configurable parameters.

        Args:
            fx (Callable): Key stream generator accepting (int | np.ndarray, bytes, int | None) and returning an
                int or np.ndarray. This function is critical for the encryption process and should be carefully
                designed to ensure cryptographic security.
            chunk_size (int): Size of message chunks. Defaults to 32.
            delimiter_size (int): The delimiter size in bytes used for separating chunks; must be
                at least 4. Defaults to 8.
            padding_range (tuple[int, int]): Range for padding length before and after
                chunks. Defaults to (5, 15).
            decoy_ratio (float): Proportion of decoy chunks to insert. Must not be negative. Defaults to 0.1.
            siv_seed_initialisation (bool): Enables synthetic IV seed initialisation based on the message to
                resist seed reuse. Defaults to True.
            auth_encrypt (bool): Enables authenticated encryption with integrity check. Defaults to True.
            vectorise (bool): Whether to use numpy for vectorised operations. If True, numpy must be
                installed and `fx` must support numpy arrays. Defaults to False.

        Raises:
            ValueError: If `chunk_size` is less than 8.
            ValueError: If `delimiter_size` is less than 4.
            TypeError: If `padding_range` is not a tuple of two integers.
            ValueError: If `padding_range` values are negative.
            ValueError: If `padding_range` values are not in ascending order.
            ValueError: If `decoy_ratio` is negative.
            ValueError: If `vectorise` is True but numpy is not installed.
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
        if vectorise and not _HAS_NUMPY:
            raise ValueError("NumPy is required for vectorised mode but is not installed.")
        elif not vectorise and _HAS_NUMPY:
            warnings.warn(
                "vectorise is False, NumPy will not be used. Consider setting it to True for better performance."
            )

        # Initialise instance variables
        self._fx = fx
        self._chunk_size = chunk_size
        self._delimiter_size = delimiter_size
        self._padding_range = padding_range
        self._decoy_ratio = decoy_ratio
        self._siv_seed_initialisation = siv_seed_initialisation
        self._auth_encrypt = auth_encrypt
        self._vectorise = vectorise

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
            f"auth_encrypt={self._auth_encrypt}, "
            f"vectorise={self._vectorise})"
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

    @property
    def _hmac_length(self) -> int:
        """Return the length of the HMAC digest used in the VernamVeil class.

        This is a constant value representing the size of the hash output from the BLAKE2b algorithm.

        Returns:
            int: The length of the HMAC digest in bytes.
        """
        return 64

    @staticmethod
    def _hmac(
        key: bytes | bytearray | memoryview, msg_list: list[bytes | memoryview] | None = None
    ) -> bytes:
        """Generate a hash-based message authentication code (HMAC) using the Blake2b algorithm.

        If `msg_list` is provided, each element is sequentially fed into the HMAC as message data.
        If `msg_list` is `None`, only the key is hashed.

        Args:
            key (bytes or bytearray or memoryview): The key for HMAC or Hash.
            msg_list (list of bytes or memoryview, optional): List of message parts to hash with the key.
                If None, only the key is hashed.

        Returns:
            bytes: The resulting hash digest.
        """
        if msg_list is not None:
            hm = hmac.new(key, msg_list.pop(0), digestmod="blake2b")
            for m in msg_list:
                hm.update(m)
            return hm.digest()
        else:
            return hashlib.blake2b(key).digest()

    def _determine_shuffled_indices(
        self, seed: bytes, real_count: int, total_count: int
    ) -> list[int]:
        """Implement the Fisher–Yates shuffle algorithm.

        Determines the shuffled positions for real chunks based on a deterministic seed.

        Uses `hash_numpy` for vectorised hashing if available.

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
        if self._vectorise:
            # Vectorised: generate all hashes at once
            i_arr = np.arange(1, total_count, dtype=np.uint64)
            hashes = fold_bytes_to_uint64(hash_numpy(i_arr, seed, "blake2b"))
        else:
            # Standard: generate hashes one by one
            hashes = [
                int.from_bytes(self._hmac(seed, [i.to_bytes(8, "big")]), "big")
                for i in range(1, total_count)
            ]

        # Shuffle deterministically based on the hashed seed
        for i in range(total_count - 1, 0, -1):
            # Create a random number between 0 and i
            j = int(hashes[i - 1]) % (i + 1)

            # Swap elements at positions i and j
            positions[i], positions[j] = positions[j], positions[i]

        return positions[:real_count]

    def _generate_bytes(self, length: int, seed: bytes) -> memoryview:
        """Produce a byte stream of the given length using the key generator function.

        In vectorised mode, uses numpy for efficient batch generation if available and supported by `fx`.

        It samples 8 bytes at a time from the generator function, which is expected to return a Python int
        or an uint64 NumPy array.

        Args:
            length (int): Number of bytes to generate.
            seed (bytes): Seed for key generation.

        Returns:
            memoryview: Generated byte stream.
        """
        if self._vectorise:
            # Vectorised generation using numpy
            # Generate enough uint64s to cover the length
            n_uint64 = math.ceil(length / 8)
            indices = np.arange(1, n_uint64 + 1, dtype=np.uint64)
            # Unbounded, get the full uint64 range
            keystream = self._fx(indices, seed, None)
            # Ensure output is a numpy array of integers in [0, 255]
            memview: memoryview = keystream.view(np.uint8)[:length].data
            return memview
        else:
            # Standard generation using python
            result = bytearray()
            i = 1
            while len(result) < length:
                # Still bound it to 8 bytes
                val: int = self._fx(i, seed, _UINT64_BOUND)
                result.extend(val.to_bytes(8, "big"))
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

    def _obfuscate(self, message: memoryview, seed: bytes, delimiter: memoryview) -> memoryview:
        """Inject noise and padding into the message and shuffle the real chunk positions.

        Args:
            message (memoryview): Original message.
            seed (bytes): Seed for deterministic shuffling of chunks.
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
        shuffled_chunk_ranges = [(-1, -1) for _ in range(total_count)]
        for i in shuffled_positions:
            shuffled_chunk_ranges[i] = next(chunk_ranges_iter)

        # Build the noisy message by combining fake and shuffled real chunks
        noisy_blocks = bytearray()
        pad_min, pad_max = self._padding_range
        for i in range(total_count):
            if shuffled_chunk_ranges[i][0] != -1:  # real chunk location
                start, end = shuffled_chunk_ranges[i]
                chunk: memoryview | bytes = message[start:end]
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
            noisy_blocks.extend(delimiter)
            # Actual data
            noisy_blocks.extend(chunk)
            # Post-pad
            noisy_blocks.extend(delimiter)
            post_pad_len = (
                secrets.randbelow(pad_max - pad_min + 1) + pad_min
                if pad_max != pad_min
                else pad_min
            )
            if post_pad_len > 0:
                noisy_blocks.extend(secrets.token_bytes(post_pad_len))

        return memoryview(noisy_blocks)

    def _deobfuscate(self, noisy: bytearray, seed: bytes, delimiter: memoryview) -> bytearray:
        """Remove noise and extract real chunks from a shuffled noisy message.

        Args:
            noisy (bytearray): Encrypted and obfuscated message.
            seed (bytes): Seed for deterministic chunk deshuffling.
            delimiter (memoryview): Delimiter used to detect chunks.

        Returns:
            bytearray: Original message reconstructed from real chunks.
        """
        # Locate all positions of the delimiter
        delimiter_len = len(delimiter)
        delimiter_indices = []
        look_start = 0
        while True:
            idx = noisy.find(delimiter, look_start)
            if idx == -1:
                break
            delimiter_indices.append(idx)
            look_start = idx + delimiter_len

        # Each chunk is framed by consecutive delimiters
        all_chunk_ranges: list[tuple[int, int]] = []
        for i in range(0, len(delimiter_indices) - 1, 2):
            start = delimiter_indices[i] + delimiter_len
            end = delimiter_indices[i + 1]
            all_chunk_ranges.append((start, end))

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

        # Reconstruct and unshuffle the message
        message = bytearray()
        view = memoryview(noisy)
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
        if self._vectorise:
            arr = np.frombuffer(data, dtype=np.uint8)
            # Create a numpy array on top of the bytearray to vectorise and still have access to original bytearray
            processed = np.frombuffer(result, dtype=np.uint8)
        else:
            arr = data
            processed = result

        for start, end in self._generate_chunk_ranges(data_len):
            # Generate a key using fx
            chunk_len = end - start
            keystream = self._generate_bytes(chunk_len, seed)

            # XOR the chunk with the key
            if self._vectorise:
                np.bitwise_xor(arr[start:end], keystream, out=processed[start:end])
                seed_data = (arr[start:end] if is_encode else processed[start:end]).data
            else:
                for i in range(chunk_len):
                    pos = start + i
                    processed[pos] = arr[pos] ^ keystream[i]
                seed_data = arr[start:end] if is_encode else memoryview(processed)[start:end]

            # Refresh the seed differently for encoding and decoding
            seed = self._hmac(seed, [seed_data])

        return result, seed

    def _generate_delimiter(self, seed: bytes) -> tuple[memoryview, bytes]:
        """Create a delimiter sequence using the key stream and update the seed.

        Args:
            seed (bytes): Seed used for generating the delimiter.

        Returns:
            tuple[memoryview, bytes]: The delimiter and the refreshed seed.
        """
        delimiter = self._generate_bytes(self._delimiter_size, seed)
        seed = self._hmac(seed, [b"delimiter"])
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

        Raises:
            ValueError: If the delimiter appears in the message.
        """
        # Convert to memoryview for efficient slicing
        if not isinstance(message, memoryview):
            msg_bytes = message
            message = memoryview(message)
        else:
            # Accessing memoryview.obj can be unsafe if the memoryview is a slice, but in this library, inputs are
            # always bytes. However, callers might still provide a sliced memoryview over bytes. This code is safe
            # because msg_bytes is only used to check for the delimiter, so at worst, the check is performed on the
            # entire underlying array. The expensive tobytes() copy is almost always avoided, unless the caller
            # provides a memoryview is not backed by a bytes or bytearray object.
            msg_bytes = (
                message.obj if isinstance(message.obj, (bytes, bytearray)) else message.tobytes()
            )

        # SIV seed initialisation: Encrypt and prepend a synthetic IV (SIV) derived from the seed and message.
        # This prevents deterministic keystreams on the first block and makes the scheme resilient to seed reuse.
        if self._siv_seed_initialisation:
            # Generate the SIV hash from the initial seed, the timestamp and the message
            timestamp = time.time_ns().to_bytes(8, "big")
            siv_hash = self._hmac(seed, [message, timestamp])
            # Encrypt the synthetic IV and evolve the seed with it
            encrypted_siv_hash, seed = self._xor_with_key(memoryview(siv_hash), seed, True)
            # Use the encrypted SIV hash bytearray as the output; this puts it in front
            output = encrypted_siv_hash

            # Note: The SIV is not reused for MAC computation, ensuring separation
            # between seed evolution and authentication.
        else:
            output = bytearray()

        # Produce a unique seed for Authenticated Encryption
        # This ensures integrity by generating a MAC tag for the cyphertext
        auth_seed = self._hmac(seed, [b"auth"]) if self._auth_encrypt else b""

        # Generate the delimiter
        delimiter, seed = self._generate_delimiter(seed)

        # Delimiter conflict check
        if msg_bytes.find(delimiter) != -1:
            raise ValueError(
                "The delimiter appears in the message. Consider increasing the delimiter size."
            )

        # Produce a unique seed for Obfuscation to avoid reusing the same seed during shuffling and to match the order
        # of operations with decode.
        shuffle_seed = self._hmac(seed, [b"shuffle"])

        # Add noise and shuffle the message
        noisy = self._obfuscate(message, shuffle_seed, delimiter)

        # Encrypt the noisy message
        cyphertext, last_seed = self._xor_with_key(noisy, seed, True)
        output.extend(cyphertext)

        # Authenticated Encryption
        if self._auth_encrypt:
            # The tag is computed over the cyphertext and the configuration of the cypher
            tag = self._hmac(auth_seed, [cyphertext, str(self).encode()])
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

        HMAC_LENGTH = self._hmac_length

        # SIV seed initialisation: Decrypt and consume the synthetic IV (SIV) to reconstruct the evolved seed.
        # This ensures the keystream remains unique and prevents deterministic decryption on the first block.
        if self._siv_seed_initialisation:
            # Split the data by taking the first HMAC_LENGTH bytes
            encrypted_siv_hash, cyphertext = cyphertext[:HMAC_LENGTH], cyphertext[HMAC_LENGTH:]
            # Decrypt the SIV hash (throw away) and evolve the seed with it
            _, seed = self._xor_with_key(encrypted_siv_hash, seed, False)

        # Authenticated Encryption
        if self._auth_encrypt:
            # Split the data by taking the last HMAC_LENGTH bytes
            encrypted_data, expected_tag = cyphertext[:-HMAC_LENGTH], cyphertext[-HMAC_LENGTH:]

            # Produce a unique seed for Authenticated Encryption
            auth_seed = self._hmac(seed, [b"auth"])

            # Estimate the tag and compare it with the expected
            tag = self._hmac(auth_seed, [encrypted_data, str(self).encode()])
            if not hmac.compare_digest(tag, expected_tag):
                raise ValueError("Authentication failed: MAC tag mismatch.")
        else:
            encrypted_data = cyphertext

        # Generate the delimiter
        delimiter, seed = self._generate_delimiter(seed)

        # Produce a unique seed for Obfuscation to avoid reusing the same seed during unshuffling and to match the order
        # of operations with encode.
        shuffle_seed = self._hmac(seed, [b"shuffle"])

        # Decrypt the noisy message
        decrypted, last_seed = self._xor_with_key(encrypted_data, seed, False)

        # Denoise, Unshuffle and extract the real message
        message = self._deobfuscate(decrypted, shuffle_seed, delimiter)

        return message, last_seed
