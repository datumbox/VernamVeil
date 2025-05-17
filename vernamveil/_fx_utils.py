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
    "check_fx_sanity",
    "generate_default_fx",
    "generate_hmac_fx",
    "generate_polynomial_fx",
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


def generate_hmac_fx(
    hash_name: Literal["blake2b", "sha256"] = "blake2b",
    vectorise: bool = False,
) -> FX:
    """Generate a standard HMAC-based pseudorandom function (PRF) using Blake2b or SHA256.

    Args:
        hash_name (Literal["blake2b", "sha256"]): Hash function to use ("blake2b" or "sha256"). Defaults to "blake2b".
        vectorise (bool): If True, uses numpy arrays as input for vectorised operations. Defaults to False.

    Returns:
        FX: A callable that returns pseudo-random bytes from HMAC-based function.

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

    # Dynamically generate the function code for scalar or vectorised HMAC-based PRF
    if vectorise:
        function_code = f"""
import numpy as np
from vernamveil import FX, hash_numpy


def keystream_fn(i: np.ndarray, seed: bytes) -> np.ndarray:
    # Implements a standard HMAC-based pseudorandom function (PRF) using {hash_name}.
    # The output is deterministically derived from the input index `i` and the secret `seed`.
    # Security relies entirely on the secrecy of the seed and the cryptographic strength of HMAC.

    # Cryptographic HMAC using {hash_name}
    return hash_numpy(i, seed, "{hash_name}")  # uses C module if available, else NumPy fallback
"""
    else:
        function_code = f"""
import hmac
from vernamveil import FX


def keystream_fn(i: int, seed: bytes) -> bytes:
    # Implements a standard HMAC-based pseudorandom function (PRF) using {hash_name}.
    # The output is deterministically derived from the input index `i` and the secret `seed`.
    # Security relies entirely on the secrecy of the seed and the cryptographic strength of HMAC.

    # Cryptographic HMAC using {hash_name}
    return hmac.new(seed, i.to_bytes(8, "big"), digestmod="{hash_name}").digest()
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

    The transformed input index is passed to a cryptographic hash function (HMAC).

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
        # followed by a cryptographically secure HMAC-Blake2b output.
        # Note: The security of `fx` relies entirely on the secrecy of the seed and the strength of the HMAC.
        # The polynomial transformation adds uniqueness to each fx instance but does not contribute additional entropy.

        # Transform index i using a polynomial function to introduce uniqueness on fx
        # Compute all powers: shape (len(i), degree)
        powers = np.power.outer(i, degrees)
        # Weighted sum (polynomial evaluation)
        result = np.dot(powers, weights)

        # Cryptographic HMAC using Blake2b
        return hash_numpy(result, seed, "blake2b")  # uses C module if available, else NumPy fallback

    return keystream_fn
"""
    else:
        function_code = f"""
import hmac
from vernamveil import FX


def make_keystream_fn():
    # Create a closure to capture the weights and initialise them only once
    weights = [{", ".join(str(w) for w in weights)}]

    def keystream_fn(i: int, seed: bytes) -> bytes:
        # Implements a customisable fx function based on a {degree}-degree polynomial transformation of the index,
        # followed by a cryptographically secure HMAC-Blake2b output.
        # Note: The security of `fx` relies entirely on the secrecy of the seed and the strength of the HMAC.
        # The polynomial transformation adds uniqueness to each fx instance but does not contribute additional entropy.

        # Transform index i using a polynomial function to introduce uniqueness on fx
        current_pow = 1
        result = 0
        for weight in weights:
            result = (result + weight * current_pow) & 0xFFFFFFFFFFFFFFFF
            current_pow = (current_pow * i) & 0xFFFFFFFFFFFFFFFF

        # Cryptographic HMAC using Blake2b
        return hmac.new(seed, result.to_bytes(8, "big"), digestmod="blake2b").digest()

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


# Default function for key stream generation
generate_default_fx = generate_polynomial_fx


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
                    f"Avalanche effect weak: flipping a bit in input changed only {hamming} bits out of {len(orig)*8}."
                )
                passed = False
        else:
            warnings.warn("Avalanche effect check failed: output lengths differ.")
            passed = False
    except Exception as e:
        warnings.warn(f"Avalanche effect check could not be performed: {e}")
        passed = False

    return passed
