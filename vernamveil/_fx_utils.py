"""Key stream function utilities for library.

This module provides utilities for generating, loading, and checking the sanity of key stream functions (fx)
used by the VernamVeil cypher.
"""

import importlib.util
import secrets
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vernamveil._types import _HAS_C_MODULE, HashType

__all__ = [
    "FX",
    "OTPFX",
    "check_fx_sanity",
    "generate_default_fx",
    "generate_keyed_hash_fx",
    "generate_polynomial_fx",
    "load_fx_from_file",
]


class FX:
    """A generic callable wrapper for key stream generator functions used in VernamVeil.

    This class wraps any user-supplied or generated keystream function, providing a consistent interface
    for use in the VernamVeil cypher. The wrapped function must be deterministic, seed-sensitive, and type-correct.

    Attributes:
        keystream_fn (Callable): Keystream function
            accepting `(int | np.ndarray[tuple[int], np.dtype[np.uint64]], bytes | bytearray)` and
            returning `np.ndarray[tuple[int, int], np.dtype[np.uint8]]`.
        block_size (int): The number of bytes returned per call.
        source_code (str): The source code of the keystream function.

    Example:

    .. code-block:: python

        fx = FX(keystream_fn, block_size=64)
        keystream_bytes = fx(42, b"mysecretseed")
    """

    def __init__(
        self,
        keystream_fn: Callable[
            [np.ndarray[tuple[int], np.dtype[np.uint64]], bytes | bytearray],
            np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        ],
        block_size: int,
        source_code: str = "",
    ) -> None:
        """Initialise the FX wrapper.

        Args:
            keystream_fn (Callable): Keystream function accepting `(np.ndarray[tuple[int], np.dtype[np.uint64]], bytes | bytearray)`
                and returning `np.ndarray[tuple[int, int], np.dtype[np.uint8]]`.
            block_size (int): The number of bytes returned per call.
            source_code (str): The source code of the keystream function.
        """
        super().__init__()
        if not _HAS_C_MODULE:
            warnings.warn("The C module is not available. Performance will be suboptimal.")

        self.keystream_fn = keystream_fn
        self.block_size = block_size
        self.source_code = source_code

    def __call__(
        self, i: np.ndarray[tuple[int], np.dtype[np.uint64]], seed: bytes | bytearray
    ) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
        """Generate the keystream for a given index and seed.

        Args:
            i (np.ndarray[tuple[int], np.dtype[np.uint64]]): The index or array of indices to generate the keystream for.
            seed (bytes or bytearray): The seed used for generating the keystream.

        Returns:
            np.ndarray[tuple[int, int], np.dtype[np.uint8]]: The generated keystream bytes or array of bytes.
        """
        return self.keystream_fn(i, seed)


class OTPFX(FX):
    """A callable class for one-time-pad (OTP) keystreams used in VernamVeil.

    This class wraps a user-supplied keystream (as a list of byte blocks), providing a consistent interface
    for OTP encryption and decryption. The keystream must be at least as long as the message to be encrypted,
    and each keystream must be used only once for cryptographic security.

    Attributes:
        keystream (list[bytes]): The list of keystream blocks (each of length `block_size`).
        position (int): The current position in the keystream.
        block_size (int): The number of bytes returned per call.
        source_code (str): The source code to reconstruct this OTPFX instance.

    **Security Warning:**
        One-time-pad keystreams must never be reused for different messages. Reusing a keystream
        completely breaks the security of the encryption.

    Example:

    .. code-block:: python

        def get_true_random_bytes(n: int) -> bytes:
            # Replace with a function that returns n bytes from a true random source.
            # For real OTP, use a true random source (e.g., hardware RNG, quantum RNG, etc.)
            # Using `secrets` or `os.urandom` is not truly random and does not provide the same guarantees.
            raise NotImplementedError()

        # Generate a long enough keystream
        block_size = 64
        keystream = [get_true_random_bytes(block_size) for _ in range(100)]

        # Create a cypher with the OTPFX instance
        fx = OTPFX(keystream, block_size=block_size)
        cypher = VernamVeil(fx)

        # Encrypt a message
        initial_seed = VernamVeil.get_initial_seed()  # remember to store this securely
        encrypted_message = cypher.encrypt(b"some message", initial_seed)

        # Optionally clip the keystream to the used portion
        fx.keystream = fx.keystream[:fx.position]  # remember to store this securely

        # Reset the pointer for decryption
        fx.position = 0

        # Decrypt the message
        decrypted_message = cypher.decrypt(encrypted_message, initial_seed)

    .. warning::
        Only reset `fx.position` to 0 for decryption of the same cyphertext. Never reuse the keystream for a new message.

    """

    def __init__(self, keystream: list[bytes], block_size: int) -> None:
        """Initialises the OTPFX instance.

        Args:
            keystream (list[bytes]): A list of bytes representing the keystream, split in equal block_size bytes.
            block_size (int): The block size for the keystream.

        Raises:
            ValueError: If the keystream blocks are not of the same size.
        """
        for idx, block in enumerate(keystream):
            if len(block) != block_size:
                raise ValueError(
                    f"Keystream block at index {idx} has length {len(block)} (expected {block_size})"
                )

        self.keystream = keystream
        self.position = 0
        source_code = f"from vernamveil import OTPFX\nfx = OTPFX({keystream}, {block_size})"
        super().__init__(self.__call__, block_size, source_code=source_code)

    def __call__(
        self, i: np.ndarray[tuple[int], np.dtype[np.uint64]], _: bytes | bytearray
    ) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
        """Generates the next value in the keystream.

        Args:
            i (np.ndarray[tuple[int], np.dtype[np.uint64]]): The index or array of indices to generate the keystream for.
            _ (bytes or bytearray): Unused parameter for compatibility.

        Returns:
            np.ndarray[tuple[int, int], np.dtype[np.uint8]]: The next value in the keystream.

        Raises:
            IndexError: If the keystream is exhausted and no more values are available.
        """
        n = len(i)
        vals = []
        for __ in range(n):
            if self.position >= len(self.keystream):
                raise IndexError("Keystream exhausted. No more values available.")
            vals.append(self.keystream[self.position])
            self.position += 1

        out = np.empty((n, self.block_size), dtype=np.uint8)
        for idx, chunk in enumerate(vals):
            out[idx] = np.frombuffer(chunk, dtype=np.uint8)
        return out


def generate_keyed_hash_fx(
    hash_name: HashType = "blake2b",
    block_size: int | None = None,
) -> FX:
    """Generate a standard keyed hash-based pseudorandom function (PRF) using BLAKE2b, BLAKE3 or SHA256.

    This is the recommended secure default `fx` for the VernamVeil cypher.

    .. note::
        For performance reasons, this function does not use an HMAC construction but instead concatenates the seed
        with the index. This is safe in this context because the inputs are tightly controlled by the cypher and always
        have fixed lengths.

    Args:
        hash_name (HashType): Hash function to use ("blake2b", "blake3" or "sha256"). The blake3 is only
            available if the C extension is installed. Defaults to "blake2b".
        block_size (int, optional): Size of the hash output in bytes. Should be 64 for blake2b, larger than 0 for blake3
            and 32 for sha256. If None, the default size for the selected hash algorithm is used. Defaults to None.

    Returns:
        FX: A callable that returns pseudo-random bytes from a keyed hash-based function.

    Raises:
        ValueError: If `hash_name` is not "blake2b", "blake3" or "sha256".
        ValueError: If `hash_name` is "blake3" but the C extension is not available.
        ValueError: If the hash_size is not 64 for blake2b, larger than 0 for blake3 or 32 for sha256.
    """
    if hash_name not in ("blake2b", "blake3", "sha256"):
        raise ValueError("hash_name must be either 'blake2b', 'blake3' or 'sha256'.")
    if hash_name == "blake3" and not _HAS_C_MODULE:
        raise ValueError("blake3 requires the C extension.")

    if block_size is None:
        if hash_name == "blake2b":
            block_size = 64
        elif hash_name == "blake3":
            block_size = 32
        elif hash_name == "sha256":
            block_size = 32
    elif hash_name == "blake2b" and block_size != 64:
        raise ValueError("blake2b block_size must be 64 bytes.")
    elif hash_name == "blake3" and block_size <= 0:
        raise ValueError("blake3 block_size must be larger than 0 bytes.")
    elif hash_name == "sha256" and block_size != 32:
        raise ValueError("sha256 block_size must be 32 bytes.")

    # Dynamically generate the function code for keyed hash-based PRF
    function_code = f"""
import numpy as np
from vernamveil import FX, hash_numpy


def keystream_fn(i: np.ndarray, seed: bytes) -> np.ndarray:
    # The secure default `fx` of the VernamVeil cypher.
    # Implements a standard keyed hash-based pseudorandom function (PRF) using {hash_name}.
    # The output is deterministically derived from the input index `i` and the secret `seed`.
    # Security relies entirely on the secrecy of the seed and the cryptographic strength of the keyed hash.

    # Hash using {hash_name}
    return hash_numpy(i, seed, "{hash_name}", hash_size={block_size})  # uses C module if available, else NumPy fallback


fx = FX(keystream_fn, block_size={block_size})
"""

    # Load the fx function from source code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(function_code)
        tmp_path = Path(tmp.name)

    fx = load_fx_from_file(tmp_path)
    tmp_path.unlink(missing_ok=True)

    # Attach the code string directly to the function object for later reference
    fx.source_code = function_code

    return fx


def generate_polynomial_fx(degree: int = 10, max_weight: int = 10**5) -> FX:
    """Generate a random polynomial-based secret function to act as a deterministic key stream generator.

    The transformed input index is passed to a cryptographic hash function.

    Args:
        degree (int): Degree of the polynomial. Defaults to 10.
        max_weight (int): Maximum value for polynomial coefficients. Defaults to `10 ** 5`.

    Returns:
        FX: A callable that returns pseudo-random bytes from the polynomial-based function.

    Raises:
        TypeError: If `degree` is not an integer.
        ValueError: If `degree` is not positive.
        TypeError: If `max_weight` is not an integer.
        ValueError: If `max_weight` is not positive.
    """
    if not isinstance(degree, int):
        raise TypeError("degree must be an integer.")
    elif degree <= 0:
        raise ValueError("degree must be a positive integer.")
    if not isinstance(max_weight, int):
        raise TypeError("max_weight must be an integer.")
    elif max_weight <= 0:
        raise ValueError("max_weight must be a positive integer.")

    # Generate random weights for each term in the polynomial including the constant term
    weights = [max(1, secrets.randbelow(max_weight + 1)) for _ in range(degree + 1)]

    # Dynamically generate the function code to allow flexibility in testing different polynomial configurations
    function_code = f"""
import numpy as np
from vernamveil import FX, hash_numpy


def make_keystream_fn():
    # Create a closure to capture the weights and initialise them only once
    weights = np.array([{", ".join(str(w) for w in weights)}], dtype=np.uint64)
    degrees = np.arange({degree + 1}, dtype=np.uint64)

    def keystream_fn(i: np.ndarray, seed: bytes) -> np.ndarray:
        # Implements a customisable fx function based on a {degree}-degree polynomial transformation of the index,
        # followed by a cryptographically secure keyed hash (BLAKE2b) output.
        # Note: The security of `fx` relies entirely on the secrecy of the seed and the strength of the keyed hash.
        # The polynomial transformation adds uniqueness to each fx instance but does not contribute additional entropy.

        # Transform index i using a polynomial function to introduce uniqueness on fx
        # Compute all powers: shape (len(i), degree + 1)
        powers = np.power.outer(i, degrees)
        # Weighted sum (polynomial evaluation)
        result = np.dot(powers, weights)

        # Hash using BLAKE2b
        return hash_numpy(result, seed, "blake2b")  # uses C module if available, else NumPy fallback

    return keystream_fn


fx = FX(make_keystream_fn(), block_size=64)
"""

    # Load the fx function from source code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(function_code)
        tmp_path = Path(tmp.name)

    fx = load_fx_from_file(tmp_path)
    tmp_path.unlink(missing_ok=True)

    # Attach the code string directly to the function object for later reference
    fx.source_code = function_code

    return fx


# Default function for key stream generation
generate_default_fx = generate_keyed_hash_fx


def load_fx_from_file(path: str | Path) -> FX:
    """Load the fx function from a Python file.

    This uses `importlib` internally to import the `fx`. Never use this with
    files from untrusted sources, as it can run arbitrary code on your system.

    Args:
        path (str or Path): Path to the Python file containing fx.

    Returns:
        FX: The loaded fx function.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If the module could not be loaded or no `fx` was found.
        TypeError: If the loaded `fx` is not an instance of FX.
    """
    # Check if the path is a valid file
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    # Load the module using importlib
    spec: Any = importlib.util.spec_from_file_location("fx_module", path.resolve())
    module = importlib.util.module_from_spec(spec)
    if module is None:
        raise ImportError(f"Could not load module from {path.resolve()}")
    spec.loader.exec_module(module)

    # Fetch the fx function from the module
    if not hasattr(module, "fx"):
        raise ImportError(f"Module {path.resolve()} does not contain an 'fx'.")
    fx: FX = module.fx
    if not isinstance(fx, FX):
        raise TypeError(f"fx in {path.resolve()} is not an instance of FX.")
    return fx


def check_fx_sanity(
    fx: FX,
    seed: bytes,
    num_samples: int = 1000,
) -> bool:
    """Perform basic sanity checks on a user-supplied fx function for use as a key stream generator.

    Checks performed:
        1. Type and output size check: All outputs should be `np.ndarray[tuple[int, int], np.dtype[np.uint8]]` of shape (num_samples, fx.block_size).
        2. Non-constant output: fx should return diverse values for varying i.
        3. Seed sensitivity: fx output should change if the seed changes.
        4. Basic uniformity: No single byte value should dominate.
        5. Avalanche effect: Flipping a bit in the input should significantly change the output.

    Args:
        fx (FX): The function to test.
        seed (bytes): The seed to use for testing.
        num_samples (int): Number of samples to test.

    Returns:
        bool: True if all checks pass, False otherwise. Issues are reported as warnings.
    """
    passed = True

    # 1. Type and output size check
    indices = np.arange(num_samples, dtype=np.uint64)
    outputs = fx(indices, seed)
    if not (
        isinstance(outputs, np.ndarray)
        and outputs.dtype == np.uint8
        and outputs.ndim == 2
        and outputs.shape == (num_samples, fx.block_size)
    ):
        warnings.warn(
            f"fx output is not a 2D NumPy array of uint8 with shape (num_samples, fx.block_size): got {type(outputs)}, "
            f"dtype={getattr(outputs, 'dtype', None)}, shape={getattr(outputs, 'shape', None)}"
        )
        passed = False

    # 2. Non-constant output for varying i
    unique_rows = np.unique(outputs, axis=0)
    if unique_rows.shape[0] < (num_samples // 10):
        warnings.warn("fx may be constant or low-entropy for varying i.")
        passed = False

    # 3. Seed sensitivity
    alt_seed = bytes((b ^ 0xAA) for b in seed)
    outputs_alt_seed = fx(indices, alt_seed)
    if not isinstance(fx, OTPFX) and np.array_equal(outputs, outputs_alt_seed):
        warnings.warn("fx output does not depend on seed.")
        passed = False

    # 4. Basic uniformity
    # Use np.bincount to get the frequency of each byte value (0-255).
    all_byte_counts = np.bincount(outputs.flatten(), minlength=256)
    min_count = np.min(all_byte_counts)
    max_count = np.max(all_byte_counts)

    # Check for heavy bias.
    if max_count > 4 * min_count:
        warnings.warn(
            "fx output is heavily biased: some byte values appear much more frequently than others."
        )
        passed = False
    # Check for missing byte values.
    if min_count == 0:
        warnings.warn("At least one byte value never appears in fx output.")
        passed = False

    # 5. Avalanche effect
    # Flip a bit in a single input index and check that the output changes significantly.
    test_idx = 42  # A sample index.
    input_arr_orig_np = np.array(test_idx, dtype=np.uint64).reshape(1)
    input_arr_flipped_np = np.array(test_idx ^ 1, dtype=np.uint64).reshape(1)

    # Call fx and get the full 2D outputs
    output_row_orig = fx(input_arr_orig_np, seed)
    output_row_flipped = fx(input_arr_flipped_np, seed)

    # Calculate Hamming distance (number of differing bits)
    xor_result = np.bitwise_xor(output_row_orig, output_row_flipped)
    # np.unpackbits converts each byte in xor_result to its 8 constituent bits.
    # The sum of these bits across all bytes gives the total Hamming distance.
    hamming_distance_bits = np.sum(np.unpackbits(xor_result))
    if hamming_distance_bits < fx.block_size * 2:  # expect at least 2 bits per byte to flip
        warnings.warn(
            f"Avalanche effect weak: flipping a bit in input changed only {hamming_distance_bits} bits out "
            f"of {8 * fx.block_size}."
        )
        passed = False

    return passed
