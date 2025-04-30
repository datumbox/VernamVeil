import hashlib
import hmac
import math
import secrets
import warnings
from typing import Callable, Iterator, Literal

try:
    import numpy as np

    _IntOrArray = int | np.ndarray
except ImportError:
    np = None
    _IntOrArray = int


class VernamVeil:
    """
    VernamVeil is a modular, symmetric cipher inspired by One-Time Pad principles, featuring customizable keystream
    generation, layered obfuscation, and authenticated encryption. Stateful seed evolution ensures avalanche effects,
    while chunk shuffling, padding, and decoy injection enhance message secrecy. Designed for educational use.
    """

    _UINT64_BOUND = 2**64

    def __init__(
        self,
        fx: Callable[[_IntOrArray, bytes, int | None], _IntOrArray],
        chunk_size: int = 32,
        delimiter_size: int = 8,
        padding_range: tuple[int, int] = (5, 15),
        decoy_ratio: float = 0.1,
        auth_encrypt: bool = True,
        vectorise: bool = False,
    ):
        """
        Initialises the VernamVeil encryption cipher with configurable parameters.

        Args:
            fx (Callable): Key stream generator accepting (int, bytes, int | None) and returning int.
                This function is critical for the encryption process and should be carefully designed
                to ensure unpredictability.
            chunk_size (int, optional): Size of message chunks. Defaults to 32.
            delimiter_size (int, optional): The delimiter size in bytes used for separating chunks; must be
                at least 4. Defaults to 8.
            padding_range (tuple[int, int], optional): Range for padding length before and after
                chunks. Defaults to (5, 15).
            decoy_ratio (float, optional): Proportion of decoy chunks to insert. Must not be negative. Defaults to 0.1.
            auth_encrypt (bool, optional): Enables authenticated encryption with integrity check. Defaults to True.
            vectorise (bool, optional): Whether to use numpy for vectorised operations. If True, numpy must be
                installed and `fx` must support numpy arrays. Defaults to False.

        Raises:
            ValueError: If `padding_range` is not a tuple of two integers.
            ValueError: If `decoy_ratio` is negative.
            ImportError: If `vectorise` is True but numpy is not installed.
        """
        # Validate input
        if not (
            isinstance(padding_range, tuple)
            and len(padding_range) == 2
            and all(isinstance(x, int) for x in padding_range)
        ):
            raise ValueError("padding_range must be a tuple of two integers.")
        if decoy_ratio < 0:
            raise ValueError("decoy_ratio must not be negative.")
        if vectorise and np is None:
            raise ImportError("NumPy is required for vectorised mode but is not installed.")
        if delimiter_size < 4:
            raise ValueError("delimiter_size must be at least 4 bytes.")
        if not vectorise and np is not None:
            warnings.warn(
                "vectorise is False, NumPy will not be used. Consider setting it to True for better performance."
            )

        # Initialise instance variables
        self._fx = fx
        self._chunk_size = chunk_size
        self._delimiter_size = delimiter_size
        self._padding_range = padding_range
        self._decoy_ratio = decoy_ratio
        self._auth_encrypt = auth_encrypt
        self._vectorise = vectorise

    @staticmethod
    def _refresh_seed(seed: bytes, data: bytes | None = None) -> bytes:
        """
        Rehashes the current seed with optional data to produce a new seed.

        Args:
            seed (bytes): Original seed.
            data (bytes, optional): Additional data to influence seed update.

        Returns:
            bytes: A new refreshed seed.
        """
        m = hashlib.blake2b(seed, digest_size=len(seed))
        if data is not None:
            m.update(data)
        return m.digest()

    @staticmethod
    def _determine_shuffled_indices(seed: bytes, real_count: int, total_count: int) -> list[int]:
        """
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

        # Shuffle deterministically based on the hashed seed
        seed_len = len(seed)
        for i in range(len(positions) - 1, 0, -1):
            # Create a random number between 0 and i
            j = int.from_bytes(
                hashlib.blake2b(seed + i.to_bytes(4, "big"), digest_size=seed_len).digest(),
                "big",
            ) % (i + 1)

            # Swap elements at positions i and j
            positions[i], positions[j] = positions[j], positions[i]

        return positions[:real_count]

    def _generate_bytes(self, length: int, seed: bytes) -> bytes:
        """
        Produces a byte stream of the given length using the key generator function.

        In vectorised mode, uses numpy for efficient batch generation if available and supported by `fx`.

        It samples 8 bytes at a time from the generator function, which is expected to return a Python int
        or an uint64 NumPy array.

        Args:
            length (int): Number of bytes to generate.
            seed (bytes): Seed for key generation.

        Returns:
            bytes: Generated byte stream (always bytes, even if vectorised).
        """
        if self._vectorise:
            # Vectorised generation using numpy
            # Generate enough uint64s to cover the length
            n_uint64 = math.ceil(length / 8)
            indices = np.arange(1, n_uint64 + 1, dtype=np.uint64)
            # Unbounded, get the full uint64 range
            keystream = self._fx(indices, seed, None)
            # Ensure output is a numpy array of integers in [0, 255]
            keystream_bytes = keystream.view(np.uint8)[:length].tobytes()
            return keystream_bytes
        else:
            # Standard generation using python
            result = bytearray()
            i = 1
            while len(result) < length:
                # Still bound it to 8 bytes
                val = self._fx(i, seed, self._UINT64_BOUND)
                result.extend(val.to_bytes(8, "big"))
                i += 1
            return result[:length]

    def _generate_delimiter(self, seed: bytes) -> tuple[bytes, bytes]:
        """
        Creates a delimiter sequence using the key stream and updates the seed.

        Args:
            seed (bytes): Seed used for generating the delimiter.

        Returns:
            tuple[bytes, bytes]: The delimiter and the refreshed seed.
        """
        delimiter = self._generate_bytes(self._delimiter_size, seed)
        seed = self._refresh_seed(seed, b"delimiter")
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

    def _obfuscate(self, message: bytes, seed: bytes, delimiter: bytes) -> bytes:
        """
        Injects noise and padding into the message and shuffles the real chunk positions.

        Args:
            message (bytes): Original message.
            seed (bytes): Seed for deterministic shuffling of chunks.
            delimiter (bytes): Chunk delimiter.

        Returns:
            bytes: Obfuscated message with shuffled real and decoy chunks.
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
        shuffled_chunk_ranges = [None] * total_count
        for i in shuffled_positions:
            shuffled_chunk_ranges[i] = next(chunk_ranges_iter)

        # Build the noisy message by combining fake and shuffled real chunks
        noisy_blocks = bytearray()
        view = memoryview(message)
        pad_min, pad_max = self._padding_range
        for i in range(total_count):
            if shuffled_chunk_ranges[i] is not None:
                start, end = shuffled_chunk_ranges[i]
                chunk = view[start:end]
            else:
                chunk = secrets.token_bytes(self._chunk_size)

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

        return noisy_blocks

    def _deobfuscate(self, noisy: bytes, seed: bytes, delimiter: bytes) -> bytes:
        """
        Removes noise and extracts real chunks from a shuffled noisy message.

        Args:
            noisy (bytes): Encrypted and obfuscated message.
            seed (bytes): Seed for deterministic chunk deshuffling.
            delimiter (bytes): Delimiter used to detect chunks.

        Returns:
            bytes: Original message reconstructed from real chunks.
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
        all_chunk_ranges = []
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

    def _xor_with_key(self, data: bytes, seed: bytes, is_encode: bool) -> tuple[bytes, bytes]:
        """
        Encrypts or decrypts data using XOR with the generated keystream.

        Args:
            data (bytes): Input data to process.
            seed (bytes): Seed for keystream generation.
            is_encode (bool): True for encryption, False for decryption.

        Returns:
            tuple[bytes, bytes]: Processed data and the final seed.
        """
        # Preallocate memory and avoid copying when slicing
        data_len = len(data)
        if self._vectorise:
            arr = np.frombuffer(data, dtype=np.uint8)
            processed = np.empty_like(arr)
        else:
            arr = memoryview(data)
            processed = bytearray(data_len)

        for start, end in self._generate_chunk_ranges(data_len):
            # Generate a key using fx
            chunk_len = end - start
            keystream = self._generate_bytes(chunk_len, seed)

            # XOR the chunk with the key
            if self._vectorise:
                ks_arr = np.frombuffer(keystream, dtype=np.uint8)
                processed[start:end] = np.bitwise_xor(arr[start:end], ks_arr)
                seed_data = (
                    arr[start:end].tobytes() if is_encode else processed[start:end].tobytes()
                )
            else:
                for i in range(chunk_len):
                    pos = start + i
                    processed[pos] = arr[pos] ^ keystream[i]
                seed_data = arr[start:end] if is_encode else memoryview(processed)[start:end]

            # Refresh the seed differently for encoding and decoding
            seed = self._refresh_seed(seed, seed_data)

        if self._vectorise:
            processed = processed.tobytes()

        return processed, seed

    def encode(self, message: bytes, seed: bytes) -> tuple[bytes, bytes]:
        """
        Encrypts a message by shuffling, padding, and applying keystream XOR.

        Args:
            message (bytes): Message to encode.
            seed (bytes): Initial seed for encryption.

        Returns:
            tuple[bytes, bytes]: Encrypted message and final seed.
        """
        # Produce a unique seed for Authenticated Encryption
        # This ensures integrity by generating a MAC tag for the ciphertext
        auth_seed = self._refresh_seed(seed, b"MAC") if self._auth_encrypt else None

        # Generate the delimiter
        delimiter, seed = self._generate_delimiter(seed)

        # Produce a unique seed for Obfuscation to avoid reusing the same seed during shuffling and to match the order
        # of operations with decode.
        shuffle_seed = self._refresh_seed(seed, b"shuffle")

        # Add noise and shuffle the message
        noisy = self._obfuscate(message, shuffle_seed, delimiter)

        # Encrypt the noisy message
        ciphertext, last_seed = self._xor_with_key(noisy, seed, True)

        # Authenticated Encryption
        if self._auth_encrypt:
            blake_hash = hashlib.blake2b(ciphertext).digest()
            tag, _ = self._xor_with_key(blake_hash, auth_seed, True)
            result = bytearray(ciphertext)
            result.extend(tag)

        return result, last_seed

    def decode(self, ciphertext: bytes, seed: bytes) -> tuple[bytes, bytes]:
        """
        Decrypts an encoded message and extracts the original content.

        Args:
            ciphertext (bytes): Encrypted and obfuscated message.
            seed (bytes): Initial seed for decryption.

        Returns:
            tuple[bytes, bytes]: Decrypted message and final seed.
        """
        # Authenticated Encryption
        if self._auth_encrypt:
            # Split the data
            data = memoryview(ciphertext)
            blake2b_len = 64
            encrypted_data, expected_tag = data[:-blake2b_len], data[-blake2b_len:]

            # Produce a unique seed for Authenticated Encryption
            auth_seed = self._refresh_seed(seed, b"MAC")

            # Estimate the tag and compare it with the expected
            blake_hash = hashlib.blake2b(encrypted_data).digest()
            tag, _ = self._xor_with_key(blake_hash, auth_seed, True)
            if not hmac.compare_digest(tag, expected_tag):
                raise ValueError("Authentication failed: MAC tag mismatch.")
        else:
            encrypted_data = ciphertext

        # Generate the delimiter
        delimiter, seed = self._generate_delimiter(seed)

        # Produce a unique seed for Obfuscation to avoid reusing the same seed during unshuffling and to match the order
        # of operations with encode.
        shuffle_seed = self._refresh_seed(seed, b"shuffle")

        # Decrypt the noisy message
        decrypted, last_seed = self._xor_with_key(encrypted_data, seed, False)

        # Denoise, Unshuffle and extract the real message
        message = self._deobfuscate(decrypted, shuffle_seed, delimiter)

        return message, last_seed

    @staticmethod
    def get_initial_seed(num_bytes: int = 32) -> bytes:
        """
        Generates a cryptographically secure initial random seed.

        This method uses the `secrets` module to generate a random sequence of bytes
        suitable for cryptographic use. It returns a byte string of the specified length.

        Args:
            num_bytes (int, optional): The number of bytes to generate for the seed.
                Defaults to 32 bytes if not provided.

        Returns:
            bytes: A random byte string of the specified length.
        """
        if num_bytes <= 0:
            raise ValueError("num_bytes must be a positive integer.")

        return secrets.token_bytes(num_bytes)

    @staticmethod
    def process_file(
        input_file: str,
        output_file: str,
        fx: Callable[[_IntOrArray, bytes, int | None], _IntOrArray],
        seed: bytes,
        buffer_size: int = 1024 * 1024,
        mode: Literal["encode", "decode"] = "encode",
        **vernamveil_kwargs,
    ):
        """
        Processes a file in blocks using VernamVeil encryption or decryption.

        Args:
            input_file (str): Path to the input file.
            output_file (str): Path to write the output.
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
        defaults = {
            "chunk_size": 4096,
            "delimiter_size": 8,
            "padding_range": (10, 20),
            "decoy_ratio": 0.05,
            "auth_encrypt": True,
        }

        # Update defaults with any user-specified overrides
        defaults.update(vernamveil_kwargs)

        # Initialise the VernamVeil object
        cipher = VernamVeil(fx, **defaults)

        # Open the input and output files
        with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
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
                    processed_block, current_seed = cipher.encode(block, current_seed)

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
                        processed_block, current_seed = cipher.decode(complete_block, current_seed)
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
