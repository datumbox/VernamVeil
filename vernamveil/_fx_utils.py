"""Key stream function utilities for library.

This module provides utilities for generating, loading, and checking the sanity of key stream functions (fx)
used by the VernamVeil cypher.
"""

import hmac
import secrets
import warnings
from pathlib import Path
from typing import Any, Callable, Literal

from vernamveil._cypher import _HAS_NUMPY, _Bytes, _Integer, np
from vernamveil._hash_utils import hash_numpy

__all__ = [
    "check_fx_sanity",
    "generate_default_fx",
    "generate_hmac_fx",
    "generate_polynomial_fx",
    "load_fx_from_file",
]


class FX:
    """A generic callable wrapper for key stream generator functions used in VernamVeil.

    This class wraps any user-supplied or generated keystream function, providing a consistent interface
    for use in the VernamVeil cipher. The wrapped function must be deterministic, seed-sensitive, and type-correct.

    Attributes:
        keystream_fn (Callable): Keystream function accepting (int | np.ndarray[np.uint64], bytes) and returning
            bytes or np.ndarray[np.uint8].
        block_size (int): The number of bytes returned per call.
        vectorise (bool): Whether the keystream function supports vectorised operation.
        source_code (str): The source code of the keystream function.

    Example::
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
            keystream_fn (Callable): Keystream function accepting (int | np.ndarray[np.uint64], bytes) and returning
                bytes or np.ndarray[np.uint8].
            block_size (int): The number of bytes returned per call.
            vectorise (bool): Whether the keystream function supports vectorised operation.
            source_code (str): The source code of the keystream function.

        Raises:
            ValueError: If `vectorise` is True but numpy is not installed.
        """
        super().__init__()
        if vectorise and not _HAS_NUMPY:
            raise ValueError("NumPy is required for vectorised mode but is not installed.")
        elif not vectorise and _HAS_NUMPY:
            warnings.warn(
                "vectorise is False, NumPy will not be used. Consider setting it to True for better performance."
            )

        self.keystream_fn = keystream_fn
        self.block_size = block_size
        self.vectorise = vectorise
        self.source_code = source_code

    def __call__(self, i: _Integer, seed: bytes) -> _Bytes:
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
        FX: An callable that returns pseudo-random bytes from HMAC-based function.

    Raises:
        ValueError: If `vectorise` is True but numpy is not installed.
        ValueError: If `hash_name` is not "blake2b" or "sha256".
    """
    if vectorise and np is None:
        raise ValueError("NumPy is required for vectorised mode but is not installed.")
    if hash_name not in ("blake2b", "sha256"):
        raise ValueError("hash_name must be either 'blake2b' or 'sha256'.")

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


def keystream_fn(i: int, seed: bytes) -> int:
    # Implements a standard HMAC-based pseudorandom function (PRF) using {hash_name}.
    # The output is deterministically derived from the input index `i` and the secret `seed`.
    # Security relies entirely on the secrecy of the seed and the cryptographic strength of HMAC.

    # Cryptographic HMAC using {hash_name}
    return hmac.new(seed, i.to_bytes(8, "big"), digestmod="{hash_name}").digest()
"""

    function_code += f"""


fx = FX(keystream_fn, block_size={64 if hash_name == "blake2b" else 32}, vectorise={vectorise})
"""

    # Execute the string to define fx in a local namespace
    local_vars: dict[str, Any] = {}
    exec(
        function_code,
        {
            "hash_numpy": hash_numpy,
            "hmac": hmac,
            "np": np,
        },
        local_vars,
    )

    # Attach the code string directly to the function object for later reference
    fx: FX = local_vars["fx"]
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
        FX: An callable that returns pseudo-random bytes from the polynomial-based function.

    Raises:
        ValueError: If `vectorise` is True but numpy is not installed.
        TypeError: If `degree` is not an integer.
        ValueError: If `degree` is not positive.
        TypeError: If `max_weight` is not an integer.
        ValueError: If `max_weight` is not positive.
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

    # Generate random weights for each term in the polynomial including the constant term
    weights = [secrets.randbelow(max_weight + 1) for _ in range(degree + 1)]

    # Dynamically generate the function code to allow flexibility in testing different polynomial configurations
    if vectorise:
        function_code = f"""
import numpy as np
from vernamveil import FX, hash_numpy


def keystream_fn(i: np.ndarray, seed: bytes) -> np.ndarray:
    # Implements a customisable fx function based on a {degree}-degree polynomial transformation of the index,
    # followed by a cryptographically secure HMAC-Blake2b output.
    # Note: The security of `fx` relies entirely on the secrecy of the seed and the strength of the HMAC.
    # The polynomial transformation adds uniqueness to each fx instance but does not contribute additional entropy.
    weights = np.array([{", ".join(str(w) for w in weights)}], dtype=np.uint64)

    # Transform index i using a polynomial function to introduce uniqueness on fx
    # Compute all powers: shape (i.size, degree)
    powers = np.power.outer(i, np.arange({degree + 1}, dtype=np.uint64))
    # Weighted sum (polynomial evaluation)
    result = np.dot(powers, weights)

    # Cryptographic HMAC using Blake2b
    return hash_numpy(result, seed, "blake2b")  # uses C module if available, else NumPy fallback
"""
    else:
        function_code = f"""
import hmac
from vernamveil import FX


def keystream_fn(i: int, seed: bytes) -> int:
    # Implements a customisable fx function based on a {degree}-degree polynomial transformation of the index,
    # followed by a cryptographically secure HMAC-Blake2b output.
    # Note: The security of `fx` relies entirely on the secrecy of the seed and the strength of the HMAC.
    # The polynomial transformation adds uniqueness to each fx instance but does not contribute additional entropy.
    weights = [{", ".join(str(w) for w in weights)}]

    # Transform index i using a polynomial function to introduce uniqueness on fx
    current_pow = 1
    result = 0
    for weight in weights:
        result = (result + weight * current_pow) & 0xFFFFFFFFFFFFFFFF
        current_pow = (current_pow * i) & 0xFFFFFFFFFFFFFFFF

    # Cryptographic HMAC using Blake2b
    return hmac.new(seed, result.to_bytes(8, "big"), digestmod="blake2b").digest()
"""

    function_code += f"""


fx = FX(keystream_fn, block_size=64, vectorise={vectorise})
"""

    # Execute the string to define fx in a local namespace
    local_vars: dict[str, Any] = {}
    exec(
        function_code,
        {
            "hash_numpy": hash_numpy,
            "hmac": hmac,
            "np": np,
        },
        local_vars,
    )

    # Attach the code string directly to the function object for later reference
    fx: FX = local_vars["fx"]
    fx.source_code = function_code

    return fx


# Default function for key stream generation
generate_default_fx = generate_polynomial_fx


def load_fx_from_file(path: str | Path) -> FX:
    """Load the fx function from a Python file.

    Args:
        path (str | Path): Path to the Python file containing fx.

    Returns:
        FX: The loaded fx function.
    """
    global_vars: dict[str, Any] = {}
    path_obj = Path(path)
    code = path_obj.read_text()
    exec(code, global_vars)
    fx: FX = global_vars["fx"]
    return fx


def check_fx_sanity(
    fx: FX,
    seed: bytes,
    num_samples: int = 1000,
) -> bool:
    """Perform basic sanity checks on a user-supplied fx function for use as a key stream generator.

    Checks performed:
        1. Non-constant output: fx should return diverse values for varying i.
        2. Seed sensitivity: fx output should change if the seed changes.
        3. Basic uniformity: No single output value should dominate.
        4. Type check: All outputs should be bytes (for scalar) or np.ndarray[np.uint8] (for vectorised).

    Args:
        fx (FX): The function to test.
        seed (bytes): The seed to use for testing.
        num_samples (int): Number of samples to test.

    Returns:
        bool: True if all checks pass, False otherwise. Issues are reported as warnings.
    """
    passed = True

    if fx.vectorise:
        arr = np.arange(1, num_samples + 1, dtype=np.uint64)
        outputs = fx(arr, seed)
        if not (
            isinstance(outputs, np.ndarray) and outputs.dtype == np.uint8 and outputs.ndim == 2
        ):
            warnings.warn("fx output is not a 2D NumPy array of uint8.")
            passed = False
        # Flatten each row to bytes for comparison
        outputs_list = [bytes(row) for row in outputs]
    else:
        outputs_list = [fx(i, seed) for i in range(1, num_samples + 1)]

    # 1. Non-constant output for varying i
    if len(set(outputs_list)) < num_samples // 10:
        warnings.warn("fx may be constant or low-entropy for varying i.")
        passed = False

    # 2. Seed sensitivity: output should change if the seed changes
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

    # 3. Basic uniformity: no single output value should dominate
    counts: dict[bytes, int] = {}
    for o in outputs_list:
        counts[o] = counts.get(o, 0) + 1
    if max(counts.values()) > 4 * min(counts.values()):
        warnings.warn("fx output is heavily biased.")
        passed = False

    # 4. Type check: all outputs should be bytes (scalar) or np.uint8 (vectorised)
    if fx.vectorise:
        if not all(isinstance(row, (bytes, bytearray)) for row in outputs_list):
            warnings.warn("fx output rows are not bytes-like objects.")
            passed = False
    else:
        if not all(isinstance(o, (bytes, bytearray)) for o in outputs_list):
            warnings.warn("fx output is not bytes.")
            passed = False

    return passed
