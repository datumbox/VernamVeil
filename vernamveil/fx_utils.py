"""
Key stream function utilities for VernamVeil.

This module provides utilities for generating, loading, and checking the sanity of key stream functions (fx)
used by the VernamVeil cypher.
"""

import hmac
import secrets
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, Tuple, cast

try:
    import numpy as np
except ImportError:
    np = None


from .cypher import _IntOrArray
from .hash_utils import _UINT64_BOUND, hash_numpy

__all__ = [
    "check_fx_sanity",
    "generate_default_fx",
    "generate_hmac_fx",
    "generate_polynomial_fx",
    "load_fx_from_file",
]


def generate_hmac_fx(
    *args: Tuple[Any, ...],
    hash_name: Literal["blake2b", "sha256"] = "blake2b",
    vectorise: bool = False,
) -> Callable[[_IntOrArray, bytes, int | None], _IntOrArray]:
    """
    Generate a standard HMAC-based pseudorandom function (PRF) using Blake2b or SHA256.

    Args:
        *args (tuple): Additional positional arguments (ignored; present for API compatibility).
        hash_name (str, optional): Hash function to use ("blake2b" or "sha256"). Defaults to "blake2b".
        vectorise (bool, optional): If True, uses numpy arrays as input for vectorised operations.

    Returns:
        Callable[[int | np.ndarray, bytes, int | None], int | np.ndarray]: A function that returns pseudo-random
            integers from HMAC-based function.

    Raises:
        ValueError: If `vectorise` is True but numpy is not installed.
        ValueError: If `hash_name` is not "blake2b" or "sha256".
    """
    if vectorise and np is None:
        raise ValueError("NumPy is required for vectorised mode but is not installed.")
    if not isinstance(hash_name, str) or hash_name not in ("blake2b", "sha256"):
        raise ValueError("hash_name must be either 'blake2b' or 'sha256'.")

    # Dynamically generate the function code for scalar or vectorised HMAC-based PRF
    if vectorise:
        function_code = f"""
from vernamveil import hash_numpy
import numpy as np


def fx(i: np.ndarray, seed: bytes, bound: int | None) -> np.ndarray:
    # Implements a standard HMAC-based pseudorandom function (PRF) using {hash_name}.
    # The output is deterministically derived from the input index `i` and the secret `seed`.
    # Security relies entirely on the secrecy of the seed and the cryptographic strength of HMAC.

    # Cryptographic HMAC using {hash_name}
    result = hash_numpy(i, seed, "{hash_name}")  # uses C module if available, else NumPy fallback

    # Modulo the result with the bound to ensure it's always within the requested range
    if bound is not None:
        np.remainder(result, bound, out=result)

    return result
"""
    else:
        function_code = f"""
import hmac


def fx(i: int, seed: bytes, bound: int | None) -> int:
    # Implements a standard HMAC-based pseudorandom function (PRF) using {hash_name}.
    # The output is deterministically derived from the input index `i` and the secret `seed`.
    # Security relies entirely on the secrecy of the seed and the cryptographic strength of HMAC.

    # Cryptographic HMAC using {hash_name}
    result = int.from_bytes(hmac.new(seed, i.to_bytes(8, "big"), digestmod="{hash_name}").digest(), "big")

    # Modulo the result with the bound to ensure it's always within the requested range
    if bound is not None:
        result %= bound

    return result
"""

    # Execute the string to define fx in a local namespace
    local_vars: dict[str, Any] = {}
    exec(
        function_code,
        {"np": np, "hash_numpy": hash_numpy, "hmac": hmac},
        local_vars,
    )

    # Attach the code string directly to the function object for later reference
    fx = local_vars["fx"]
    fx._source_code = function_code  # type: ignore[attr-defined, unused-ignore]

    return cast(Callable[[_IntOrArray, bytes, int | None], _IntOrArray], fx)


def generate_polynomial_fx(
    complexity: int, max_weight: int = 10**5, vectorise: bool = False
) -> Callable[[_IntOrArray, bytes, int | None], _IntOrArray]:
    """
    Generate a random polynomial-based secret function to act as a deterministic key stream generator.
    The transformed input index is passed to a cryptographic hash function (HMAC) and bounded to the requested range.
    Though any mathematical function with domain the positive integers can be used, this utility only supports
    polynomials and is used for testing.

    Args:
        complexity (int): Degree of the polynomial.
        max_weight (int, optional): Maximum value for polynomial coefficients. Defaults to 10 ** 5.
        vectorise (bool, optional): If True, uses numpy arrays as input for vectorised operations.

    Returns:
        Callable[[int | np.ndarray, bytes, int | None], int | np.ndarray]: A function that returns pseudo-random
            integers from polynomial evaluation.

    Raises:
        ValueError: If `vectorise` is True but numpy is not installed.
        TypeError: If `complexity` is not an integer.
        ValueError: If `complexity` is negative.
        TypeError: If `max_weight` is not an integer.
        ValueError: If `max_weight` is not positive.
    """
    if vectorise and np is None:
        raise ValueError("NumPy is required for vectorised mode but is not installed.")
    if not isinstance(complexity, int):
        raise TypeError("complexity must be an integer.")
    elif complexity <= 0:
        raise ValueError("complexity must be a positive integer.")
    if not isinstance(max_weight, int):
        raise TypeError("max_weight must be an integer.")
    elif max_weight <= 0:
        raise ValueError("max_weight must be a positive integer.")

    # Generate random weights for each term in the polynomial including the constant term
    weights = [secrets.randbelow(max_weight + 1) for _ in range(complexity + 1)]

    # Dynamically generate the function code to allow flexibility in testing different polynomial configurations
    if vectorise:
        function_code = f"""
from vernamveil import hash_numpy
import numpy as np


def fx(i: np.ndarray, seed: bytes, bound: int | None) -> np.ndarray:
    # Implements a customisable fx function based on a {complexity}-degree polynomial transformation of the index,
    # followed by a cryptographically secure HMAC-Blake2b output.
    # Note: The security of `fx` relies entirely on the secrecy of the seed and the strength of the HMAC.
    # The polynomial transformation adds uniqueness to each fx instance but does not contribute additional entropy.
    weights = np.array([{", ".join(str(w) for w in weights)}], dtype=np.uint64)

    # Transform index i using a polynomial function to introduce uniqueness on fx
    # Compute all powers: shape (i.size, degree)
    powers = np.power.outer(i, np.arange({complexity + 1}, dtype=np.uint64))
    # Weighted sum (polynomial evaluation)
    result = np.dot(powers, weights)

    # Cryptographic HMAC using Blake2b
    result = hash_numpy(result, seed, "blake2b")  # uses C module if available, else NumPy fallback

    # Modulo the result with the bound to ensure it's always within the requested range
    if bound is not None:
        np.remainder(result, bound, out=result)

    return result
"""
    else:
        function_code = f"""
import hmac


def fx(i: int, seed: bytes, bound: int | None) -> int:
    # Implements a customisable fx function based on a {complexity}-degree polynomial transformation of the index,
    # followed by a cryptographically secure HMAC-Blake2b output.
    # Note: The security of `fx` relies entirely on the secrecy of the seed and the strength of the HMAC.
    # The polynomial transformation adds uniqueness to each fx instance but does not contribute additional entropy.
    weights = [{", ".join(str(w) for w in weights)}]
    interim_modulus = {_UINT64_BOUND}

    # Transform index i using a polynomial function to introduce uniqueness on fx
    current_pow = 1
    result = 0
    for weight in weights:
        result = (result + weight * current_pow) % interim_modulus
        current_pow = (current_pow * i) % interim_modulus  # Avoid large power growth

    # Cryptographic HMAC using Blake2b
    result = int.from_bytes(hmac.new(seed, result.to_bytes(8, "big"), digestmod="blake2b").digest(), "big")

    # Modulo the result with the bound to ensure it's always within the requested range
    if bound is not None:
        result %= bound

    return result
"""

    # Execute the string to define fx in a local namespace
    local_vars: dict[str, Any] = {}
    exec(
        function_code,
        {"np": np, "hash_numpy": hash_numpy, "hmac": hmac},
        local_vars,
    )
    fx = local_vars["fx"]

    # Attach the code string directly to the function object for later reference
    fx._source_code = function_code  # type: ignore[attr-defined, unused-ignore]

    return cast(Callable[[_IntOrArray, bytes, int | None], _IntOrArray], fx)


# Default function for key stream generation
generate_default_fx = generate_polynomial_fx


def load_fx_from_file(path: str | Path) -> Callable[[_IntOrArray, bytes, int | None], _IntOrArray]:
    """
    Load the fx function from a Python file.

    Args:
        path (str | Path): Path to the Python file containing fx.

    Returns:
        Callable[[int | np.ndarray, bytes, int | None], int | np.ndarray]: The loaded fx function.
    """
    global_vars: dict[str, Any] = {}
    path_obj = Path(path)
    code = path_obj.read_text()
    exec(code, global_vars)
    fx = global_vars["fx"]
    return cast(Callable[[_IntOrArray, bytes, int | None], _IntOrArray], fx)


def check_fx_sanity(
    fx: Callable[[_IntOrArray, bytes, int | None], _IntOrArray],
    seed: bytes,
    bound: int = 256,
    num_samples: int = 1000,
) -> bool:
    """
    Perform basic sanity checks on a user-supplied fx function for use as a key stream generator.
    Automatically detects if fx is vectorised (NumPy) or scalar (int) and tests accordingly.

    Checks performed:
        1. Non-constant output: fx should return diverse values for varying i.
        2. Seed sensitivity: fx output should change if the seed changes.
        3. Bound respect: All outputs should be in [0, bound).
        4. Basic uniformity: No single output value should dominate.
        5. Type check: All outputs should be int.

    Args:
        fx: The function to test.
        seed: The seed to use for testing.
        bound: The upper bound for output values.
        num_samples: Number of samples to test.

    Returns:
        bool: True if all checks pass, False otherwise. Issues are reported as warnings.
    """
    passed = True

    # Try to detect if fx supports numpy arrays
    try:
        test_input = np.arange(num_samples, dtype=np.uint64)
        test_output = fx(test_input, seed, bound)
        is_vectorised = isinstance(test_output, np.ndarray)
    except Exception:
        is_vectorised = False

    if is_vectorised:
        arr = np.arange(num_samples, dtype=np.uint64)
        outputs = fx(arr, seed, bound)
        if isinstance(outputs, np.ndarray):
            if outputs.dtype != np.uint64:
                warnings.warn("fx output is not an uint64 NumPy array.")
                passed = False
            outputs_list = outputs.tolist()
        else:
            warnings.warn("fx output is not a NumPy array.")
            passed = False
            outputs_list = list(outputs)  # type: ignore[arg-type]
    else:
        outputs_list = [fx(i, seed, bound) for i in range(num_samples)]

    # 1. Non-constant output for varying i
    if len(set(outputs_list)) < num_samples // 10:
        warnings.warn("fx may be constant or low-entropy for varying i.")
        passed = False

    # 2. Seed sensitivity
    alt_seed = bytes((b ^ 0xAA) for b in seed)
    if is_vectorised:
        arr = np.arange(num_samples, dtype=np.uint64)
        outputs_alt = fx(arr, alt_seed, bound)
        if isinstance(outputs_alt, np.ndarray):
            outputs_alt_list = outputs_alt.tolist()
        else:
            outputs_alt_list = list(outputs_alt)  # type: ignore[arg-type]
    else:
        outputs_alt_list = [fx(i, alt_seed, bound) for i in range(num_samples)]
    if outputs_list == outputs_alt_list:
        warnings.warn("fx output does not depend on seed.")
        passed = False

    # 3. Bound respect
    if not all(0 <= o < bound for o in outputs_list):
        warnings.warn("fx output does not respect the bound.")
        passed = False

    # 4. Basic uniformity
    counts = [0 for _ in range(bound)]
    for o in outputs_list:
        if 0 <= o < bound:
            counts[o] += 1
    if max(counts) > num_samples * 0.2:
        warnings.warn("fx output is heavily biased.")
        passed = False

    # 5. Type check
    if not all(isinstance(o, int) for o in outputs_list):
        warnings.warn("fx output is not int.")
        passed = False

    return passed
