"""
Implements the VernamVeil stream cypher and related utilities.
Defines the main encryption class and core cryptographic operations.
"""

import hashlib
import hmac
import math
import secrets
import warnings
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, TypeAlias

from .hash_utils import _UINT64_BOUND, hash_numpy

try:
    import numpy as np
    from numpy.typing import NDArray

    _IntOrArray: TypeAlias = int | NDArray[np.uint64]
    _HAS_NUMPY = True
except ImportError:
    np = None

    _IntOrArray: TypeAlias = int  # type: ignore[misc, no-redef]
    _HAS_NUMPY = False

__all__ = ["VernamVeil"]


class VernamVeil:
    """
    VernamVeil is a modular, symmetric stream cypher inspired by One-Time Pad principles. It features customisable
    keystream generation, synthetic IV seed initialisation, stateful seed evolution for avalanche effects,
    authenticated encryption, and layered message obfuscation (chunk shuffling, padding, decoy injection). Supports
    vectorised operations (NumPy) and optional C-backed hashing for performance. Designed for educational and
    experimental use.
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
        """
        Initialise the VernamVeil encryption cypher with configurable parameters.

        Args:
            fx (Callable): Key stream generator accepting (int | np.ndarray, bytes, int | None) and returning an
                int or np.ndarray. This function is critical for the encryption process and should be carefully
                designed to ensure cryptographic security.
            chunk_size (int, optional): Size of message chunks. Defaults to 32.
            delimiter_size (int, optional): The delimiter size in bytes used for separating chunks; must be
                at least 4. Defaults to 8.
            padding_range (tuple[int, int], optional): Range for padding length before and after
                chunks. Defaults to (5, 15).
            decoy_ratio (float, optional): Proportion of decoy chunks to insert. Must not be negative. Defaults to 0.1.
            siv_seed_initialisation (bool, optional): Enables synthetic IV seed initialisation based on the message to
                resist seed reuse. Defaults to True.
            auth_encrypt (bool, optional): Enables authenticated encryption with integrity check. Defaults to True.
            vectorise (bool, optional): Whether to use numpy for vectorised operations. If True, numpy must be
                installed and `fx` must support numpy arrays. Defaults to False.

        Raises:
            ValueError: If `delimiter_size` is less than 4.
            ValueError: If `padding_range` is not a tuple of two integers.
            ValueError: If `decoy_ratio` is negative.
            ValueError: If `vectorise` is True but numpy is not installed.
        """
        # Validate input
        if delimiter_size < 4:
            raise ValueError("delimiter_size must be at least 4 bytes.")
        if not (
            isinstance(padding_range, tuple)
            and len(padding_range) == 2
            and all(isinstance(x, int) for x in padding_range)
        ):
            raise ValueError("padding_range must be a tuple of two integers.")
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

    @property
    def _hmac_length(self) -> int:
        """
        Returns the length of the HMAC digest used in the VernamVeil class.

        This is a constant value representing the size of the hash output from the BLAKE2b algorithm.
        """
        return 64

    @staticmethod
    def _hmac(key: bytes | bytearray | memoryview, msg: bytes | memoryview | None = None) -> bytes:
        """
        Generates a hash-based message authentication code (HMAC) using the Blake2b algorithm.
        Often used for refreshing the seed or generating a unique hash for a message.

        Args:
            key (bytes or bytearray or memoryview): The key to hash.
            msg (bytes or memoryview, optional): The data to hash with the key. If None, the key is hashed alone.

        Returns:
            bytes: A hash digest of the key and message.
        """
        if msg is not None:
            return hmac.new(key, msg, digestmod="blake2b").digest()
        else:
            return hashlib.blake2b(key).digest()

    def _determine_shuffled_indices(
        self, seed: bytes, real_count: int, total_count: int
    ) -> list[int]:
        """
        Implements the Fisher–Yates shuffle algorithm, to determine the shuffled positions for real
        chunks based on a deterministic seed.

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

        if self._vectorise:
            # Vectorised: generate all hashes at once
            i_arr = np.arange(1, total_count, dtype=np.uint64)
            hashes: NDArray[np.uint64] = hash_numpy(i_arr, seed, "blake2b")
        else:
            # Standard: generate hashes one by one
            hashes: list[int] = [  # type: ignore[no-redef]
                int.from_bytes(self._hmac(seed, i.to_bytes(8, "big")), "big")
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
        """
        Produces a byte stream of the given length using the key generator function.

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
            keystream: NDArray[np.uint64] = self._fx(indices, seed, None)
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

    def _generate_delimiter(self, seed: bytes) -> tuple[memoryview, bytes]:
        """
        Creates a delimiter sequence using the key stream and updates the seed.

        Args:
            seed (bytes): Seed used for generating the delimiter.

        Returns:
            tuple[memoryview, bytes]: The delimiter and the refreshed seed.
        """
        delimiter = self._generate_bytes(self._delimiter_size, seed)
        seed = self._hmac(seed, b"delimiter")
        return delimiter, seed

    def _generate_chunk_ranges(self, message_len: int) -> Iterator[tuple[int, int]]:
        """
        Splits a message into chunk index ranges based on the configured chunk size.

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
        """
        Injects noise and padding into the message and shuffles the real chunk positions.

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
            if shuffled_chunk_ranges[i] != (-1, -1):
                start, end = shuffled_chunk_ranges[i]
                chunk: memoryview = message[start:end]
            else:
                chunk: bytes = secrets.token_bytes(self._chunk_size)  # type: ignore[no-redef]

            # Pre-pad
            pre_pad_len = secrets.randbelow(pad_max - pad_min + 1) + pad_min
            noisy_blocks.extend(secrets.token_bytes(pre_pad_len))
            noisy_blocks.extend(delimiter)
            # Actual data
            noisy_blocks.extend(chunk)
            # Post-pad
            noisy_blocks.extend(delimiter)
            post_pad_len = secrets.randbelow(pad_max - pad_min + 1) + pad_min
            noisy_blocks.extend(secrets.token_bytes(post_pad_len))

        return memoryview(noisy_blocks)

    def _deobfuscate(self, noisy: bytearray, seed: bytes, delimiter: memoryview) -> bytearray:
        """
        Removes noise and extracts real chunks from a shuffled noisy message.

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
    ) -> tuple[memoryview, bytes]:
        """
        Encrypts or decrypts data using XOR with the generated keystream.

        Args:
            data (memoryview): Input data to process.
            seed (bytes): Seed for keystream generation.
            is_encode (bool): True for encryption, False for decryption.

        Returns:
            tuple[memoryview, bytes]: Processed data and the final seed.
        """
        # Preallocate memory and avoid copying when slicing
        data_len = len(data)
        if self._vectorise:
            arr: NDArray[np.uint8] = np.frombuffer(data, dtype=np.uint8)
            processed: NDArray[np.uint8] = np.empty_like(arr)
        else:
            arr: memoryview = data  # type: ignore[no-redef]
            processed: bytearray = bytearray(data_len)  # type: ignore[no-redef]

        for start, end in self._generate_chunk_ranges(data_len):
            # Generate a key using fx
            chunk_len = end - start
            keystream = self._generate_bytes(chunk_len, seed)

            # XOR the chunk with the key
            if self._vectorise:
                np.bitwise_xor(arr[start:end], keystream, out=processed[start:end])
                seed_data: memoryview = (arr[start:end] if is_encode else processed[start:end]).data
            else:
                for i in range(chunk_len):
                    pos = start + i
                    processed[pos] = arr[pos] ^ keystream[i]
                seed_data: memoryview = arr[start:end] if is_encode else memoryview(processed)[start:end]  # type: ignore[no-redef]

            # Refresh the seed differently for encoding and decoding
            seed = self._hmac(seed, seed_data)

        if self._vectorise:
            result: memoryview = processed.data
        else:
            result = memoryview(processed)

        return result, seed

    def encode(self, message: bytes | memoryview, seed: bytes) -> tuple[bytearray, bytes]:
        """
        Encrypts a message.

        Args:
            message (bytes or memoryview): Message to encode.
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
            # Generate the SIV hash from the initial seed and the message
            siv_hash = self._hmac(seed, message)
            # Encrypt the synthetic IV and evolve the seed with it
            encrypted_siv_hash, seed = self._xor_with_key(memoryview(siv_hash), seed, True)
            # Put the encrypted SIV hash at the start of the output
            output = bytearray(encrypted_siv_hash)

            # Note: The SIV is not reused for MAC computation, ensuring separation
            # between seed evolution and authentication.
        else:
            output = bytearray()

        # Produce a unique seed for Authenticated Encryption
        # This ensures integrity by generating a MAC tag for the cyphertext
        auth_seed = self._hmac(seed, b"auth") if self._auth_encrypt else b""

        # Generate the delimiter
        delimiter, seed = self._generate_delimiter(seed)

        # Produce a unique seed for Obfuscation to avoid reusing the same seed during shuffling and to match the order
        # of operations with decode.
        shuffle_seed = self._hmac(seed, b"shuffle")

        # Add noise and shuffle the message
        noisy = self._obfuscate(message, shuffle_seed, delimiter)

        # Encrypt the noisy message
        cyphertext, last_seed = self._xor_with_key(noisy, seed, True)
        output.extend(cyphertext)

        # Authenticated Encryption
        if self._auth_encrypt:
            encrypted_hash = self._hmac(cyphertext)
            tag, _ = self._xor_with_key(memoryview(encrypted_hash), auth_seed, True)
            output.extend(tag)

        return output, last_seed

    def decode(self, cyphertext: bytes | memoryview, seed: bytes) -> tuple[bytearray, bytes]:
        """
        Decrypts an encoded message.

        Args:
            cyphertext (bytes or memoryview): Encrypted and obfuscated message.
            seed (bytes): Initial seed for decryption.

        Returns:
            tuple[bytearray, bytes]: Decrypted message and final seed.
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
            auth_seed = self._hmac(seed, b"auth")

            # Estimate the tag and compare it with the expected
            encrypted_hash = self._hmac(encrypted_data)
            tag, _ = self._xor_with_key(memoryview(encrypted_hash), auth_seed, True)
            if not hmac.compare_digest(tag, expected_tag):
                raise ValueError("Authentication failed: MAC tag mismatch.")
        else:
            encrypted_data = cyphertext

        # Generate the delimiter
        delimiter, seed = self._generate_delimiter(seed)

        # Produce a unique seed for Obfuscation to avoid reusing the same seed during unshuffling and to match the order
        # of operations with encode.
        shuffle_seed = self._hmac(seed, b"shuffle")

        # Decrypt the noisy message
        decrypted, last_seed = self._xor_with_key(encrypted_data, seed, False)

        # Convert to bytearray based on the type of the memoryview
        if isinstance(decrypted.obj, bytearray):
            decrypted_bytearray = decrypted.obj
        else:
            decrypted_bytearray = bytearray(decrypted)

        # Denoise, Unshuffle and extract the real message
        message = self._deobfuscate(decrypted_bytearray, shuffle_seed, delimiter)

        return message, last_seed

    @staticmethod
    def get_initial_seed(num_bytes: int = 64) -> bytes:
        """
        Generates a cryptographically secure initial random seed.

        This method uses the `secrets` module to generate a random sequence of bytes
        suitable for cryptographic use. It returns a byte string of the specified length.

        Args:
            num_bytes (int, optional): The number of bytes to generate for the seed.
                Defaults to 64 bytes if not provided.

        Returns:
            bytes: A random byte string of the specified length.

        Raises:
            ValueError: If `num_bytes` is not a positive integer.
        """
        if num_bytes <= 0:
            raise ValueError("num_bytes must be a positive integer.")

        return secrets.token_bytes(num_bytes)

    @staticmethod
    def process_file(
        input_file: str | Path,
        output_file: str | Path,
        fx: Callable[[_IntOrArray, bytes, int | None], _IntOrArray],
        seed: bytes,
        buffer_size: int = 1024 * 1024,
        mode: Literal["encode", "decode"] = "encode",
        **vernamveil_kwargs: dict[str, Any],
    ) -> None:
        """
        Processes a file in blocks using VernamVeil encryption or decryption.

        Args:
            input_file (str | Path): Path to the input file.
            output_file (str | Path): Path to write the output.
            fx (Callable): Key stream generator function.
            seed (bytes): Initial seed for processing.
            buffer_size (int, optional): Bytes to read from the file at a time. Defaults to 1MB.
            mode (Literal["encode", "decode"], optional): Operation mode. Defaults to "encode".
            **vernamveil_kwargs: Additional parameters for VernamVeil configuration.

        Raises:
            ValueError: If `mode` is not "encode" or "decode".
            ValueError: If the end of file is reached in decode mode and a block is incomplete (missing delimiter).
        """
        # Define default VernamVeil parameters suitable for large files
        defaults: dict[str, Any] = {
            "chunk_size": 4096,
            "delimiter_size": 8,
            "padding_range": (10, 20),
            "decoy_ratio": 0.05,
            "siv_seed_initialisation": True,
            "auth_encrypt": True,
        }

        # Update defaults with any user-specified overrides
        defaults.update(vernamveil_kwargs)

        # Initialise the VernamVeil object
        cypher = VernamVeil(fx, **defaults)

        # Convert to Path if necessary
        input_path = Path(input_file)
        output_path = Path(output_file)

        # Open the input and output files
        with input_path.open("rb") as infile, output_path.open("wb") as outfile:
            current_seed = seed

            # Unencrypted fixed-size delimiter to separate encoded blocks
            # This ensures that variable-sized blocks can be identified during decoding
            block_delimiter = b"|END_OF_BLOCK|"
            block_delimiter_size = len(block_delimiter)

            if mode == "encode":
                while True:
                    # Read from the file
                    block = infile.read(buffer_size)
                    if not block:
                        break  # End of file

                    # Encode the content block
                    processed_block, current_seed = cypher.encode(block, current_seed)

                    # Write the processed block to the output file
                    outfile.write(processed_block)

                    # Write a fixed delimiter to mark the end of the block
                    outfile.write(block_delimiter)
            elif mode == "decode":
                buffer = bytearray()
                while True:
                    block = infile.read(buffer_size)
                    if not block and not buffer:
                        break  # End of file and nothing left to process

                    buffer.extend(block)
                    while True:
                        delim_index = buffer.find(block_delimiter)
                        if delim_index == -1:
                            break  # No complete block in buffer yet

                        # Extract the complete block up to the delimiter
                        complete_block = memoryview(buffer)[:delim_index]

                        # Decode the complete block
                        processed_block, current_seed = cypher.decode(complete_block, current_seed)
                        outfile.write(processed_block)

                        # Remove the processed block and delimiter from the buffer
                        buffer = buffer[delim_index + block_delimiter_size :]

                    if not block:
                        # No more data to read, but there may be leftover data without a delimiter
                        if buffer:
                            raise ValueError("Incomplete block at end of file: missing delimiter.")
                        break

            else:
                raise ValueError("Invalid mode. Use 'encode' or 'decode'.")
