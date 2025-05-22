"""Key stream function utilities for library.

This module provides utilities for generating, loading, and checking the sanity of key stream functions (fx)
used by the VernamVeil cypher.
"""

import importlib.util
import secrets
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Literal

from vernamveil._cypher import _HAS_NUMPY, _Bytes, _Integer, np
from vernamveil._hash_utils import _HAS_C_MODULE

__all__ = [
    "FX",
    "OTPFX",
    "check_fx_sanity",
    "generate_default_fx",
    "generate_keyed_hash_fx",
    "generate_polynomial_fx",
    "generate_prf_fx",
    "load_fx_from_file",
]


class FX:
    """A generic callable wrapper for key stream generator functions used in VernamVeil.

    This class wraps any user-supplied or generated keystream function, providing a consistent interface
    for use in the VernamVeil cypher. The wrapped function must be deterministic, seed-sensitive, and type-correct.

    Attributes:
        keystream_fn (Callable): Keystream function accepting `(int | np.ndarray[np.uint64], bytes)` and returning
            `bytes` or `np.ndarray[np.uint8]`.
        block_size (int): The number of bytes returned per call.
        vectorise (bool): Whether the keystream function performs vectorised operations.
        source_code (str): The source code of the keystream function.

    Example:

    .. code-block:: python

        fx = FX(keystream_fn, block_size=64, vectorise=False)
        keystream_bytes = fx(42, b"mysecretseed")
    """

    def __init__(
        self,
        keystream_fn: Callable[[_Integer, bytes], _Bytes],
        block_size: int,
        vectorise: bool,
        source_code: str = "",
    ) -> None:
        """Initialise the FX wrapper.

        Args:
            keystream_fn (Callable): Keystream function accepting `(int | np.ndarray[np.uint64], bytes)` and returning
                `bytes` or `np.ndarray[np.uint8]`.
            block_size (int): The number of bytes returned per call.
            vectorise (bool): Whether the keystream function performs vectorised operations.
            source_code (str): The source code of the keystream function.

        Raises:
            ValueError: If `vectorise` is True but numpy is not installed.
        """
        super().__init__()
        if vectorise:
            if not _HAS_NUMPY:
                raise ValueError("NumPy is required for vectorised mode but is not installed.")
            elif not _HAS_C_MODULE:
                warnings.warn(
                    "NumPy is installed but the C module is not available. Performance will be suboptimal."
                )
        elif not vectorise and _HAS_NUMPY:
            warnings.warn(
                "vectorise is False, NumPy will not be used. Consider setting it to True for better performance."
            )

        self.keystream_fn = keystream_fn
        self.block_size = block_size
        self.vectorise = vectorise
        self.source_code = source_code

    def __call__(self, i: _Integer, seed: bytes) -> _Bytes:
        """Generate the keystream for a given index and seed.

        Args:
            i (_Integer): The index or array of indices to generate the keystream for.
            seed (bytes): The seed used for generating the keystream.

        Returns:
            _Bytes: The generated keystream bytes or array of bytes.
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
        vectorise (bool): Whether the keystream is used in vectorised mode.
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
        fx = OTPFX(keystream, block_size=block_size, vectorise=False)
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

    def __init__(self, keystream: list[bytes], block_size: int, vectorise: bool):
        """Initializes the OTPFX instance.

        Args:
            keystream (list[bytes]): A list of bytes representing the keystream, split in equal block_size bytes.
            block_size (int): The block size for the keystream.
            vectorise (bool): Whether to use vectorised operations.

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
        source_code = (
            f"from vernamveil import OTPFX\nfx = OTPFX({keystream}, {block_size}, {vectorise})"
        )
        super().__init__(self.__call__, block_size, vectorise, source_code=source_code)

    def __call__(self, i: _Integer, _: bytes) -> _Bytes:
        """Generates the next value in the keystream.

        Args:
            i (_Integer): The index or array of indices to generate the keystream for.
            _ (bytes): Unused parameter for compatibility.

        Returns:
            _Bytes: The next value in the keystream.

        Raises:
            IndexError: If the keystream is exhausted and no more values are available.
        """
        n = len(i) if self.vectorise else 1
        vals = []
        for __ in range(n):
            if self.position >= len(self.keystream):
                raise IndexError("Keystream exhausted. No more values available.")
            vals.append(self.keystream[self.position])
            self.position += 1

        if not self.vectorise:
            return vals[0]
        return np.fromiter((b for chunk in vals for b in chunk), dtype=np.uint8)


def generate_keyed_hash_fx(
    hash_name: Literal["blake2b", "sha256"] = "blake2b",
    vectorise: bool = False,
) -> FX:
    """Generate a standard keyed hash-based pseudorandom function (PRF) using Blake2b or SHA256.

    This is the recommended secure default `fx` for the VernamVeil cypher.

    Args:
        hash_name (Literal["blake2b", "sha256"]): Hash function to use ("blake2b" or "sha256"). Defaults to "blake2b".
        vectorise (bool): If True, uses numpy arrays as input for vectorised operations. Defaults to False.

    Returns:
        FX: A callable that returns pseudo-random bytes from a keyed hash-based function.

    Raises:
        ValueError: If `hash_name` is not "blake2b" or "sha256".
        TypeError: If `vectorise` is not a boolean.
        ValueError: If `vectorise` is True but numpy is not installed.
    """
    if vectorise and np is None:
        raise ValueError("NumPy is required for vectorised mode but is not installed.")
    if hash_name not in ("blake2b", "sha256"):
        raise ValueError("hash_name must be either 'blake2b' or 'sha256'.")
    if not isinstance(vectorise, bool):
        raise TypeError("vectorise must be a boolean.")

    # Dynamically generate the function code for scalar or vectorised keyed hash-based PRF
    if vectorise:
        function_code = f"""
import numpy as np
from vernamveil import FX, hash_numpy


def keystream_fn(i: np.ndarray, seed: bytes) -> np.ndarray:
    # The secure default `fx` of the VernamVeil cypher.
    # Implements a standard keyed hash-based pseudorandom function (PRF) using {hash_name}.
    # The output is deterministically derived from the input index `i` and the secret `seed`.
    # Security relies entirely on the secrecy of the seed and the cryptographic strength of the keyed hash.

    # Cryptographic keyed hash using {hash_name}
    return hash_numpy(i, seed, "{hash_name}")  # uses C module if available, else NumPy fallback
"""
    else:
        function_code = f"""
import hashlib
from vernamveil import FX


def keystream_fn(i: int, seed: bytes) -> bytes:
    # The secure default `fx` of the VernamVeil cypher.
    # Implements a standard keyed hash-based pseudorandom function (PRF) using {hash_name}.
    # The output is deterministically derived from the input index `i` and the secret `seed`.
    # Security relies entirely on the secrecy of the seed and the cryptographic strength of the keyed hash.

    # Cryptographic keyed hash using {hash_name}
    hasher = hashlib.new("{hash_name}", key=seed)
    hasher.update(i.to_bytes(8, "big"))
    return hasher.digest()
"""

    function_code += f"""


fx = FX(keystream_fn, block_size={64 if hash_name == "blake2b" else 32}, vectorise={vectorise})
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


def generate_polynomial_fx(
    degree: int = 10, max_weight: int = 10**5, vectorise: bool = False
) -> FX:
    """Generate a random polynomial-based secret function to act as a deterministic key stream generator.

    The transformed input index is passed to a cryptographic hash function.

    Args:
        degree (int): Degree of the polynomial. Defaults to 10.
        max_weight (int): Maximum value for polynomial coefficients. Defaults to `10 ** 5`.
        vectorise (bool): If True, uses numpy arrays as input for vectorised operations. Defaults to False.

    Returns:
        FX: A callable that returns pseudo-random bytes from the polynomial-based function.

    Raises:
        TypeError: If `degree` is not an integer.
        ValueError: If `degree` is not positive.
        TypeError: If `max_weight` is not an integer.
        ValueError: If `max_weight` is not positive.
        TypeError: If `vectorise` is not a boolean.
        ValueError: If `vectorise` is True but numpy is not installed.
    """
    if vectorise and np is None:
        raise ValueError("NumPy is required for vectorised mode but is not installed.")
    if not isinstance(degree, int):
        raise TypeError("degree must be an integer.")
    elif degree <= 0:
        raise ValueError("degree must be a positive integer.")
    if not isinstance(max_weight, int):
        raise TypeError("max_weight must be an integer.")
    elif max_weight <= 0:
        raise ValueError("max_weight must be a positive integer.")
    if not isinstance(vectorise, bool):
        raise TypeError("vectorise must be a boolean.")

    # Generate random weights for each term in the polynomial including the constant term
    weights = [max(1, secrets.randbelow(max_weight + 1)) for _ in range(degree + 1)]

    # Dynamically generate the function code to allow flexibility in testing different polynomial configurations
    if vectorise:
        function_code = f"""
import numpy as np
from vernamveil import FX, hash_numpy


def make_keystream_fn():
    # Create a closure to capture the weights and initialise them only once
    weights = np.array([{", ".join(str(w) for w in weights)}], dtype=np.uint64)
    degrees = np.arange({degree + 1}, dtype=np.uint64)

    def keystream_fn(i: np.ndarray, seed: bytes) -> np.ndarray:
        # Implements a customisable fx function based on a {degree}-degree polynomial transformation of the index,
        # followed by a cryptographically secure keyed hash (Blake2b) output.
        # Note: The security of `fx` relies entirely on the secrecy of the seed and the strength of the keyed hash.
        # The polynomial transformation adds uniqueness to each fx instance but does not contribute additional entropy.

        # Transform index i using a polynomial function to introduce uniqueness on fx
        # Compute all powers: shape (len(i), degree + 1)
        powers = np.power.outer(i, degrees)
        # Weighted sum (polynomial evaluation)
        result = np.dot(powers, weights)

        # Cryptographic keyed hash using Blake2b
        return hash_numpy(result, seed, "blake2b")  # uses C module if available, else NumPy fallback

    return keystream_fn
"""
    else:
        function_code = f"""
import hashlib
from vernamveil import FX


def make_keystream_fn():
    # Create a closure to capture the weights and initialise them only once
    weights = [{", ".join(str(w) for w in weights)}]

    def keystream_fn(i: int, seed: bytes) -> bytes:
        # Implements a customisable fx function based on a {degree}-degree polynomial transformation of the index,
        # followed by a cryptographically secure keyed hash (Blake2b) output.
        # Note: The security of `fx` relies entirely on the secrecy of the seed and the strength of the keyed hash.
        # The polynomial transformation adds uniqueness to each fx instance but does not contribute additional entropy.

        # Transform index i using a polynomial function to introduce uniqueness on fx
        current_pow = 1
        result = 0
        for weight in weights:
            result = (result + weight * current_pow) & 0xFFFFFFFFFFFFFFFF
            current_pow = (current_pow * i) & 0xFFFFFFFFFFFFFFFF

        # Cryptographic keyed hash using Blake2b
        hasher = hashlib.blake2b(seed)
        hasher.update(i.to_bytes(8, "big"))
        return hasher.digest()

    return keystream_fn
"""

    function_code += f"""


fx = FX(make_keystream_fn(), block_size=64, vectorise={vectorise})
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


def generate_prf_fx(
    hash_name: Literal["blake2b", "sha256"] = "blake2b",
    vectorise: bool = False,
) -> FX:
    """Generate a pseudorandom function (PRF) for keystream generation using HMAC or HKDF-Expand.

    This function returns an `FX` instance that produces cryptographically secure keystream bytes
    from a secret seed and an input index. In scalar (non-vectorised) mode, it uses HMAC with the
    specified hash function. In vectorised mode, it uses HKDF-Expand to generate longer keystreams
    efficiently.

    Args:
        hash_name (Literal["blake2b", "sha256"]): Hash function to use ("blake2b" or "sha256"). Defaults to "blake2b".
        vectorise (bool): If True, uses numpy arrays as input for vectorised operations. Defaults to False.

    Returns:
        FX: A callable PRF suitable for use as a keystream generator.

    Raises:
        ValueError: If `hash_name` is not "blake2b" or "sha256".
        TypeError: If `vectorise` is not a boolean.
        ValueError: If `vectorise` is True but numpy is not installed.
    """
    if vectorise and np is None:
        raise ValueError("NumPy is required for vectorised mode but is not installed.")
    if hash_name not in ("blake2b", "sha256"):
        raise ValueError("hash_name must be either 'blake2b' or 'sha256'.")
    if not isinstance(vectorise, bool):
        raise TypeError("vectorise must be a boolean.")

    block_size = 64 if hash_name == "blake2b" else 32

    # Dynamically generate the function code for scalar or vectorised PRF
    if vectorise:
        function_code = f"""
import numpy as np
from vernamveil import FX, hkdf_numpy


def keystream_fn(i: np.ndarray, seed: bytes) -> np.ndarray:
    # Implements a HKDF-Expand using {hash_name}.
    # The output is deterministically derived from the input index `i` and the secret `seed`.
    # The `info` parameter is varied (using a counter) for each chunk to expand the keystream
    # beyond the HKDF per-call output limit (255 * hash_size), as allowed by RFC 5869.
    # Security relies entirely on the secrecy of the seed and the cryptographic strength of the HKDF.

    total_bytes = len(i) * {block_size}
    max_per_call = {255 * block_size}
    output = np.empty(total_bytes, dtype=np.uint8)

    # Generate the keystream in chunks to avoid exceeding the HKDF per-call output limit
    for ctr, chunk_start in enumerate(range(0, total_bytes, max_per_call)):
        chunk_len = min(max_per_call, total_bytes - chunk_start)
        info = ctr.to_bytes(8, "big")

        # HKDF using {hash_name}
        output[chunk_start:chunk_start + chunk_len] = hkdf_numpy(seed, info, chunk_len, "{hash_name}")  # uses C module if available, else Python fallback

    return output.reshape((-1, {block_size}), copy=False)
"""
    else:
        function_code = f"""
import hashlib
import hmac
from vernamveil import FX


def keystream_fn(i: int, seed: bytes) -> bytes:
    # Implements a HMAC using {hash_name}.
    # The output is deterministically derived from the input index `i` and the secret `seed`.
    # Security relies entirely on the secrecy of the seed and the cryptographic strength of the HMAC.

    # HMAC using {hash_name}
    return hmac.new(seed, i.to_bytes(8, "big"), hashlib.{hash_name}).digest()
"""

    function_code += f"""


fx = FX(keystream_fn, block_size={block_size}, vectorise={vectorise})
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
        path (str | Path): Path to the Python file containing fx.

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
        1. Type and output size check: All outputs should be `bytes` of length `fx.block_size` (scalar) or `np.ndarray[np.uint8]` of shape (num_samples, fx.block_size) (vectorised).
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
    if fx.vectorise:
        arr = np.arange(1, num_samples + 1, dtype=np.uint64)
        outputs = fx(arr, seed)
        if not (
            isinstance(outputs, np.ndarray)
            and outputs.dtype == np.uint8
            and outputs.ndim == 2
            and outputs.shape == (num_samples, fx.block_size)
        ):
            warnings.warn(
                f"fx output is not a 2D NumPy array of uint8 with shape (num_samples, fx.block_size): got {type(outputs)}, dtype={getattr(outputs, 'dtype', None)}, shape={getattr(outputs, 'shape', None)}"
            )
            passed = False
        # Each row must have fx.block_size columns
        if outputs.shape[1] != fx.block_size:
            warnings.warn("Each row of fx output must have fx.block_size columns.")
            passed = False
        outputs_list = [bytes(row) for row in outputs]
    else:
        outputs_list = [fx(i, seed) for i in range(1, num_samples + 1)]
        if not all(isinstance(o, (bytes, bytearray)) for o in outputs_list):
            warnings.warn("fx output is not bytes or bytearray.")
            passed = False
        else:
            # Each output must have fx.block_size length
            for idx, o in enumerate(outputs_list):
                if len(o) != fx.block_size:
                    warnings.warn(
                        f"fx output at index {idx} has length {len(o)} (expected {fx.block_size})."
                    )
                    passed = False

    # 2. Non-constant output for varying i
    # Check that not all outputs are identical
    if len(set(outputs_list)) < num_samples // 10:
        warnings.warn("fx may be constant or low-entropy for varying i.")
        passed = False

    # 3. Seed sensitivity
    alt_seed = bytes((b ^ 0xAA) for b in seed)
    if fx.vectorise:
        arr = np.arange(1, num_samples + 1, dtype=np.uint64)
        outputs_alt = fx(arr, alt_seed)
        outputs_alt_list = [bytes(row) for row in outputs_alt]
    else:
        outputs_alt_list = [fx(i, alt_seed) for i in range(1, num_samples + 1)]
    if outputs_list == outputs_alt_list:
        warnings.warn("fx output does not depend on seed.")
        passed = False

    # 4. Basic uniformity
    # Concatenate all output bytes and count the frequency of each byte value (0-255)
    all_bytes = b"".join([bytes(o) for o in outputs_list])
    if len(all_bytes) == 0:
        warnings.warn("fx produced no output bytes for uniformity check.")
        passed = False
    else:
        byte_counts = Counter(all_bytes)
        # Ensure all 256 possible byte values are checked, not just those present
        min_count = min(byte_counts.get(i, 0) for i in range(256))
        max_count = max(byte_counts.get(i, 0) for i in range(256))
        if max_count > 4 * min_count:
            warnings.warn(
                "fx output is heavily biased: some byte values appear much more frequently than others."
            )
            passed = False
        if min_count == 0:
            warnings.warn("At least one byte value never appears in fx output.")
            passed = False

    # 5. Avalanche effect
    # Flip a bit in the input and check that the output changes significantly (Hamming distance)
    try:
        test_idx = 42
        if fx.vectorise:
            arr = np.array([test_idx], dtype=np.uint64)
            orig = fx(arr, seed)[0]
            # Flip the least significant bit of the index
            arr_flipped = np.array([test_idx ^ 1], dtype=np.uint64)
            flipped = fx(arr_flipped, seed)[0]
        else:
            orig = fx(test_idx, seed)
            flipped = fx(test_idx ^ 1, seed)
        # Compute Hamming distance
        if len(orig) == len(flipped):
            hamming = sum(bin(a ^ b).count("1") for a, b in zip(orig, flipped))
            if hamming < len(orig) * 2:  # expect at least 2 bits per byte to flip
                warnings.warn(
                    f"Avalanche effect weak: flipping a bit in input changed only {hamming} bits out of {len(orig) * 8}."
                )
                passed = False
        else:
            warnings.warn("Avalanche effect check failed: output lengths differ.")
            passed = False
    except Exception as e:
        warnings.warn(f"Avalanche effect check could not be performed: {e}")
        passed = False

    return passed
