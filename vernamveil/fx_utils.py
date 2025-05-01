import hmac
import hashlib
import secrets
import warnings
from typing import Callable
from pathlib import Path


try:
    import numpy as np
except ImportError:
    np = None


from .hash_utils import hash_numpy, _UINT64_BOUND
from .cypher import _IntOrArray


def generate_polynomial_fx(
    complexity: int, max_weight: int = 10**5, vectorise: bool = False
) -> Callable[[_IntOrArray, bytes, int | None], _IntOrArray]:
    """
    Generates a polynomial-based secret function to act as a deterministic key stream generator. Though any
    mathematical function with domain the positive integers can be used, this utility only supports polynomials and is
    used for testing.

    Args:
        complexity (int): Degree of the polynomial.
        max_weight (int, optional): Maximum value for polynomial coefficients. Defaults to 10 ** 5.
        vectorise (bool, optional): If True, uses numpy arrays as input for vectorised operations.

    Returns:
        Callable[[int | np.ndarray, bytes, int | None], int | np.ndarray]: A function that returns pseudo-random
            integers from polynomial evaluation.

    Raises:
        ImportError: If `vectorise` is True but numpy is not installed.
    """
    if vectorise and np is None:
        raise ImportError("NumPy is required for vectorised mode but is not installed.")

    # Generate random weights for each term in the polynomial including the constant term
    weights = [secrets.randbelow(max_weight + 1) for _ in range(complexity + 1)]

    # Dynamically generate the function code to allow flexibility in testing different polynomial configurations
    if vectorise:
        function_code = f"""
def fx(i: np.ndarray, seed: bytes, bound: int | None) -> np.ndarray:
    # Implements a customizable fx function based on a {complexity}-degree polynomial transformation of the index,
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
def fx(i: int, seed: bytes, bound: int | None) -> int:
    # Implements a customizable fx function based on a {complexity}-degree polynomial transformation of the index,
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
    result = int.from_bytes(hmac.new(seed, result.to_bytes(8, "big"), hashlib.blake2b).digest(), "big")

    # Modulo the result with the bound to ensure it's always within the requested range
    if bound is not None:
        result %= bound

    return result
"""

    # Execute the string to define fx in a local namespace
    local_vars = {}
    exec(
        function_code,
        {"hashlib": hashlib, "np": np, "hash_numpy": hash_numpy, "hmac": hmac},
        local_vars,
    )
    fx = local_vars["fx"]

    # Attach the code string directly to the function object for later reference
    fx._source_code = function_code

    return fx


# Default function for key stream generation
generate_default_fx = generate_polynomial_fx


def load_fx_from_file(path: str | Path) -> Callable[[_IntOrArray, bytes, int | None], _IntOrArray]:
    """
    Loads the fx function from a Python file.

    Args:
        path (str | Path): Path to the Python file containing fx.

    Returns:
        Callable[[int | np.ndarray, bytes, int | None], int | np.ndarray]: The loaded fx function.
    """
    global_vars = {}
    path_obj = Path(path)
    code = path_obj.read_text()
    exec(code, global_vars)
    return global_vars["fx"]


def check_fx_sanity(
    fx: Callable[[_IntOrArray, bytes, int | None], _IntOrArray],
    seed: bytes,
    bound: int = 256,
    num_samples: int = 1000,
) -> bool:
    """
    Performs basic sanity checks on a user-supplied fx function for use as a key stream generator.
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
            outputs = outputs.tolist()
        else:
            warnings.warn("fx output is not a NumPy array.")
            passed = False
            outputs = list(outputs)
    else:
        outputs = [fx(i, seed, bound) for i in range(num_samples)]

    # 1. Non-constant output for varying i
    if len(set(outputs)) < num_samples // 10:
        warnings.warn("fx may be constant or low-entropy for varying i.")
        passed = False

    # 2. Seed sensitivity
    alt_seed = bytes((b ^ 0xAA) for b in seed)
    if is_vectorised:
        arr = np.arange(num_samples, dtype=np.uint64)
        outputs_alt = fx(arr, alt_seed, bound)
        if isinstance(outputs_alt, np.ndarray):
            outputs_alt = outputs_alt.tolist()
        else:
            outputs_alt = list(outputs_alt)
    else:
        outputs_alt = [fx(i, alt_seed, bound) for i in range(num_samples)]
    if outputs == outputs_alt:
        warnings.warn("fx output does not depend on seed.")
        passed = False

    # 3. Bound respect
    if not all(0 <= o < bound for o in outputs):
        warnings.warn("fx output does not respect the bound.")
        passed = False

    # 4. Basic uniformity
    counts = [0] * bound
    for o in outputs:
        if 0 <= o < bound:
            counts[o] += 1
    if max(counts) > num_samples * 0.2:
        warnings.warn("fx output is heavily biased.")
        passed = False

    # 5. Type check
    if not all(isinstance(o, int) for o in outputs):
        warnings.warn("fx output is not int.")
        passed = False

    return passed
