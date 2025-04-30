import hashlib
import secrets
import warnings
from typing import Callable
from pathlib import Path


try:
    import numpy as np
except ImportError:
    np = None


from npsha256 import numpy_sha256
from .cypher import _IntOrArray


def generate_polynomial_fx(
    n: int, max_weight: int = 10**5, base_modulus: int = 10**9, vectorise: bool = False
) -> Callable[[_IntOrArray, bytes, int | None], _IntOrArray]:
    """
    Generates a polynomial-based secret function to act as a deterministic key stream generator. Though any
    mathematical function with domain the positive integers can be used, this utility only supports polynomials and is
    used for testing.

    Args:
        n (int): Degree of the polynomial.
        max_weight (int, optional): Maximum value for polynomial coefficients. Defaults to 10 ** 5.
        base_modulus (int, optional): Modulus to prevent large intermediate values. Defaults to 10 ** 9.
        vectorise (bool, optional): If True, uses numpy arrays as input for vectorised operations.

    Returns:
        Callable[[int | np.ndarray, bytes, int | None], int | np.ndarray]: A function that returns pseudo-random
            integers from polynomial evaluation.

    Raises:
        ImportError: If `vectorise` is True but numpy is not installed.
    """
    if vectorise and np is None:
        raise ImportError("NumPy is required for vectorised mode but is not installed.")

    # Generate random weights for each term in the polynomial
    weights = [secrets.randbelow(max_weight + 1) for _ in range(n)]

    # Dynamically generate the function code to allow flexibility in testing different polynomial configurations
    if vectorise:
        function_code = f"""
def fx(i: np.ndarray, seed: bytes, bound: int | None) -> np.ndarray:
    # Implements a polynomial of {n} degree
    weights = np.array([{", ".join(str(w) for w in weights)}], dtype=np.uint64)
    base_modulus = {base_modulus}

    # Hash the input with the seed to get entropy
    entropy = numpy_sha256(i, seed)  # uses C module if available, else NumPy fallback
    base = i + entropy
    np.remainder(base, base_modulus, out=base)  # in-place modulus, avoids copy

    # Compute all powers in one go
    powers = np.power.outer(base, np.arange(1, len(weights) + 1, dtype=np.uint64))

    # Weighted sum for each element
    result = base
    np.remainder(result, 99991, out=result)
    np.add(result, np.dot(powers, weights), out=result)

    # Modulo the result with the bound to ensure it's always within the requested range
    if bound is not None:
        np.remainder(result, bound, out=result)

    return result
"""
    else:
        function_code = f"""
def fx(i: int, seed: bytes, bound: int | None) -> int:
    # Implements a polynomial of {n} degree
    weights = [{", ".join(str(w) for w in weights)}]
    base_modulus = {base_modulus}

    # Hash the input with the seed to get entropy
    entropy = int.from_bytes(hashlib.sha256(seed + i.to_bytes(4, "big")).digest(), "big")
    base = (i + entropy) % base_modulus

    # Combine terms of the polynomial using weights and powers of the base
    result = base % 99991
    for power, weight in enumerate(weights, start=1):
        result += weight * pow(base, power)

    # Modulo the result with the bound to ensure it's always within the requested range
    if bound is not None:
        result %= bound

    return result
"""

    # Execute the string to define fx in a local namespace
    local_vars = {}
    exec(
        function_code,
        {"hashlib": hashlib, "np": np, "numpy_sha256": numpy_sha256},
        local_vars,
    )
    fx = local_vars["fx"]

    # Attach the code string directly to the function object for later reference
    fx._source_code = function_code

    return fx


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
