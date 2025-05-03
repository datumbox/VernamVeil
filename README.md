# 🔐 VernamVeil: A Function-Based Stream Cypher

[![CI](https://github.com/datumbox/VernamVeil/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/datumbox/VernamVeil/actions) [![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://datumbox.github.io/VernamVeil/) [![License](https://img.shields.io/:license-apache-brightgreen.svg)](./LICENSE)

---

> ⚠️ **DISCLAIMER:** This is an educational encryption prototype and **not** meant for real-world use. It has **not** been audited or reviewed by cryptography experts, and **should not** be used to store, transmit, or protect sensitive data.

## 🚀 Quick Start

Minimal Installation (without vectorisation or C module):
```bash
pip install .
```

Minimal Example:
```python
from vernamveil import VernamVeil, generate_default_fx

fx = generate_default_fx(4)
initial_seed = VernamVeil.get_initial_seed()
cipher = VernamVeil(fx)
encrypted, _ = cipher.encode(b"Hello!", initial_seed)
decrypted, _ = cipher.decode(encrypted, initial_seed)
assert decrypted == b"Hello!"
```

---

## 🔎 Overview

**VernamVeil** is an experimental cypher inspired by the **One-Time Pad (OTP)** developed in Python. The name honours **Gilbert Vernam**, who is credited with the theoretical foundation of the OTP.

Instead of using a static key, VernamVeil allows the key to be represented by a function `fx(i: int | np.ndarray, seed: bytes, bound: int | None) -> int | np.ndarray`:
- `i`: the index of the bytes in the message; a scalar integer or an uint64 NumPy array with a continuous enumeration for vectorised operations
- `seed`: a byte string that provides context and state; should be kept secret
- `bound`: an optional integer used to modulo the function output into the desired range (usually `2**64` because we sample 8 bytes at a time)
- **Output**: an integer or an uint64 NumPy array representing the key stream values

_Note: `numpy` is an optional dependency, used to accelerate vectorised operations when available._

```python
from vernamveil import VernamVeil


# Step 1: Define a custom key stream function; remember to store this securely
def fx(i: int, seed: bytes, bound: int | None) -> int:
    # Simple but cryptographically unsafe fx; see below for a more complex example
    b = seed[i % len(seed)]
    result = ((i ** 2 + i * b + b ** 2) * (i + 7))
    if bound is not None:
        result %= bound
    return result


# Step 2: Generate a random initial seed for encryption
initial_seed = VernamVeil.get_initial_seed()  # remember to store this securely

# Step 3: Encrypt and decrypt a single message
cipher = VernamVeil(fx)
encrypted, _ = cipher.encode(b"Hello!", initial_seed)
decrypted, _ = cipher.decode(encrypted, initial_seed)
```

This approach enables novel forms of key generation, especially for those who enjoy playing with maths and code. While this is not a secure implementation by cryptographic standards, it offers a fun and flexible way to experiment with function-based encryption.

If you're curious about how encryption works, or just want to mess with maths and code in a cool way, this project is a fun starting point. For more information, read the accompanying [blog post](https://blog.datumbox.com/vernamveil-a-fresh-take-on-function-based-encryption/).

---

## 💡 Why VernamVeil?

- **Using a function as a key** is appealing compared to fixed-size keys that repeat, which create vulnerabilities. While an infinitely long key would be uncrackable, a weak function introduces similar risks to repeating keys.
- Its **modular structure** means anyone can build their own `fx` functions with creative mathematical expressions or external data.
- **Inspired by the OTP**, VernamVeil supports long, non-repeating key streams, as long as your function and seed combination allows it.

---

## ✨ Characteristics

- **Function-Based Key Stream:** The key stream is dynamically generated using a user-defined function `fx` and an `initial_seed`, both of which should be kept secret.
- **Symmetric Encryption:** The same secrets are used for both encryption and decryption, ensuring the process is fully reversible with identical parameters for both operations.
- **One-Time Pad Inspired**: The keystream is derived in a manner loosely reminiscent of OTPs, with potential for non-repeating, functionally generated keys. Initial seeds are intended for single use.
- **Modular Keystream Design**: The `fx` function can be swapped to explore different styles of pseudorandom generation, including custom PRNGs or cryptographic hashes.
- **Obfuscation Techniques**: The cypher injects decoy chunks into ciphertext, pads real chunks with dummy bytes, and shuffles output to obscure chunk boundaries. Note that chunk delimiters are randomly generated, encrypted and not exposed.
- **Avalanche Effect**: Through hash-based seed refreshing, small changes in input result in large changes in output, enhancing unpredictability.
- **Authenticated Encryption**: Supports message authentication using MAC-before-decryption to detect tampering and prevent padding oracle-style issues.
- **Highly Configurable**: The implementation allows the user to adjust key parameters such as `chunk_size`, `delimiter_size`, `padding_range`, `decoy_ratio`, and `auth_encrypt`, offering flexibility to tailor the encryption to specific needs. These parameters must be aligned between encoding and decoding, otherwise the MAC check will fail.
- **Vectorisation**: All operations are vectorised using `numpy` if `vectorise=True`. Pure Python mode can be used as a fallback when `numpy` is unavailable by setting `vectorise=False`, but it is slower.
- **Optional C-backed Fast Hashing**: For even faster vectorised `fx` functions, an optional C module (`nphash`) is provided. When installed (with `cffi` and system dependencies), it enables high-performance BLAKE2b and SHA-256 hashing for NumPy-based key stream generation. This can be used directly in user-defined `fx` methods or is automatically leveraged by helpers like `generate_default_fx`. See [`nphash/README.md`](nphash/README.md) for build and usage details.
---

## ⚠️ Caveats & Best Practices

- **Not Secure for Real Use**: This is an educational tool and experimental toy, not production-ready cryptography.
- **Use Strong `fx` Functions**: The entire system's unpredictability hinges on the entropy and behaviour of your `fx`. Avoid anything guessable or biased and the use of periodic mathematical functions which can lead to predictable or repeating outputs.
- **Block Delimiters Leak**: When encrypting multiple messages to the same file, plaintext delimiters remain visible. Encrypt the entire blob if full confidentiality is needed.
- **Use Secure Randomness**: For generating initial seeds favour `VernamVeil.get_initial_seed()` over `random.randbytes()`.
- **Do not reuse seeds**: Treat each `initial_seed` as a one-time-use context. It's recommended to use a fresh initial seed for every encode/decode session. During the same session, the API returns the next seed you should use for the following call.
- **The seed replaces the need for IVs/nonces**: While this implementation doesn’t use a nonce or IV in the traditional cryptographic sense, the seed serves a similar role by maintaining state between operations. Each new seed evolves from the previous one, ensuring unique keystreams and preventing key stream reuse.
- **Message Ordering & Replay**: If transmitting encrypted chunks over time, ensure that any external metadata (e.g., message order, timestamps) is securely handled. While the evolving seed prevents keystream reuse, maintaining proper ordering and anti-replay mechanisms might still be necessary in some cases.

---

## 📝 Examples

### ✉️ Encrypting and Decrypting Multiple Messages

```python
from vernamveil import VernamVeil, generate_default_fx


# Step 1: Generate a random custom fx using the helper
fx = generate_default_fx(10)  # remember to store fx._source_code securely

# Step 2: Generate a random initial seed for encryption
initial_seed = VernamVeil.get_initial_seed()  # remember to store this securely

# Step 3: Initialise VernamVeil with the custom fx and parameters
cipher = VernamVeil(fx, chunk_size=64, decoy_ratio=0.2)

# Step 4: Encrypt multiple messages in one session
messages = [
    "This is a secret message!",
    "another one",
    "and another one"
]
encrypted = []
seed = initial_seed
for msg in messages:
    # Each message evolves the seed for the next one
    enc, seed = cipher.encode(msg.encode(), seed)
    encrypted.append(enc)

# Step 5: Decrypt multiple messages in one session
seed = initial_seed
for original, enc in zip(messages, encrypted):
    # Each message evolves the seed for the next one
    dec, seed = cipher.decode(enc, seed)
    assert dec.decode() == original
```

### 📂 Encrypting and Decrypting Files

```python
from vernamveil import VernamVeil, generate_default_fx


# Step 1: Generate a random custom fx using the helper
fx = generate_default_fx(10)  # remember to store fx._source_code securely

# Step 2: Generate a random initial seed for encryption
initial_seed = VernamVeil.get_initial_seed()  # remember to store this securely

# Step 3: Encrypt a file
VernamVeil.process_file("plain.txt", "encrypted.dat", fx, initial_seed, mode="encode")

# Step 4: Decrypt the file
VernamVeil.process_file("encrypted.dat", "decrypted.txt", fx, initial_seed, mode="decode")
```

### 🧠 A marginally stronger `fx`

```python
import hmac


def fx(i: int, seed: bytes, bound: int | None) -> int:
    # Implements a customisable fx function based on a 10-degree polynomial transformation of the index,
    # followed by a cryptographically secure HMAC-Blake2b output. 
    # Note: The security of `fx` relies entirely on the secrecy of the seed and the strength of the HMAC.
    # The polynomial transformation adds uniqueness to each fx instance but does not contribute additional entropy.
    weights = [24242, 68652, 77629, 55585, 32284, 78741, 70249, 39611, 54080, 73198, 12426]
    interim_modulus = 18446744073709551616
    
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
```

### 🏎️ A fast `fx` that uses NumPy vectorisation and the `nphash` C module

```python
from vernamveil import hash_numpy
import numpy as np


def fx(i: np.ndarray, seed: bytes, bound: int | None) -> np.ndarray:
    # Implements a customisable fx function based on a 10-degree polynomial transformation of the index,
    # followed by a cryptographically secure HMAC-Blake2b output. 
    # Note: The security of `fx` relies entirely on the secrecy of the seed and the strength of the HMAC.
    # The polynomial transformation adds uniqueness to each fx instance but does not contribute additional entropy.
    weights = np.array([24242, 68652, 77629, 55585, 32284, 78741, 70249, 39611, 54080, 73198, 12426], dtype=np.uint64)
    
    # Transform index i using a polynomial function to introduce uniqueness on fx
    # Compute all powers: shape (i.size, degree)
    powers = np.power.outer(i, np.arange(11, dtype=np.uint64))
    # Weighted sum (polynomial evaluation)
    result = np.dot(powers, weights)
    
    # Cryptographic HMAC using Blake2b
    result = hash_numpy(result, seed, "blake2b")  # uses C module if available, else NumPy fallback
    
    # Modulo the result with the bound to ensure it's always within the requested range
    if bound is not None:
        np.remainder(result, bound, out=result)
    
    return result
```

---

## 🧰 Provided `fx` Utilities

VernamVeil includes helper tools to make working with key stream functions easier:

- `generate_default_fx`: Quickly create deterministic `fx` functions for testing or experimentation. Supports both scalar and vectorised (NumPy) modes.
- `check_fx_sanity`: Run basic sanity checks on your custom `fx` to ensure it produces diverse, seed-sensitive, and well-bounded outputs.
- `load_fx_from_file`: Load a custom `fx` function from a Python file. This is useful for testing and validating your own implementations.

These utilities help you prototype and validate your own key stream functions before using them in encryption.

Example:

```python
from vernamveil import generate_default_fx, check_fx_sanity


# Generate a vectorised fx function of degree 4
fx = generate_default_fx(4, max_weight=1000, vectorise=True)

# Show the generated function's source code
print("Generated fx source code:\n", fx._source_code)

# Check if the generated fx passes basic sanity checks
seed = b"mysecretseed"
bound = 256
num_samples = 100
passed = check_fx_sanity(fx, seed, bound, num_samples)
print("Sanity check passed:", passed)
```

---

## 🖥️ Command-Line Interface (CLI)

VernamVeil provides a convenient CLI for file encryption and decryption. The CLI supports both encoding (encryption) and decoding (decryption) operations, allowing you to specify custom key stream functions (`fx`) and seeds, or have them generated automatically.

### ⚙️ Features

- **Encrypt and decrypt files** using a user-defined or auto-generated `fx` function and seed.
- **Auto-generate `fx.py` and `seed.bin`** during encoding if not provided; these files are saved in the current working directory.
- **Custom `fx` and seed support**: Supply your own `fx.py` and `seed.bin` for both encoding and decoding.
- **Configurable parameters**: Adjust chunk size, delimiter size, padding, decoy ratio, and more.
- **Sanity checks**: Optionally verify that your `fx` function is suitable for cryptographic use.

### 💻 Usage

```commandline
# Encrypt a file with auto-generated fx and seed
vernamveil encode --infile plain.txt --outfile encrypted.dat

# Encrypt a file with a custom fx function and seed
vernamveil encode --infile plain.txt --outfile encrypted.dat --fx-file my_fx.py --seed-file my_seed.bin

# Decrypt a file (requires the same fx and seed used for encryption)
vernamveil decode --infile encrypted.dat --outfile decrypted.txt --fx-file my_fx.py --seed-file my_seed.bin

# Enable fx sanity check during encoding or decoding
vernamveil encode --infile plain.txt --outfile encrypted.dat --fx-file my_fx.py --seed-file my_seed.bin --check-fx-sanity
vernamveil decode --infile encrypted.dat --outfile decrypted.txt --fx-file my_fx.py --seed-file my_seed.bin --check-fx-sanity
```

> ⚠️ **Warning: CLI Parameter Consistency**
>
> When decoding, you **must** use the exact same parameters (such as `--chunk-size`, `--delimiter-size`, `--padding-range`, `--decoy-ratio`, `--auth-encrypt`, and `--vectorise`) as you did during encoding.
>
> For example, the following will **fail** with a `MAC tag mismatch` error because the `--chunk-size` parameter differs between encoding and decoding:
>
> ```commandline
> vernamveil encode --infile plain.txt --outfile encrypted.dat --chunk-size 2048
> vernamveil decode --infile encrypted.dat --outfile decrypted.txt --chunk-size 1024 --fx-file fx.py --seed-file seed.bin
> ```
>
> **Always use identical parameters for both encoding and decoding.** Any mismatch will result in decryption failure.

### 🗄️ File Handling

- When encoding **without** `--fx-file` or `--seed-file`, the CLI generates `fx.py` and `seed.bin` in the current directory. **Store these files securely**; they are required for decryption.
- When decoding, you **must** provide both `--fx-file` and `--seed-file` pointing to the originals used for encryption.
- For safety, the CLI will **not overwrite** existing output files, `fx.py`, or `seed.bin`. If these files already exist, you must delete or rename them manually before running the command.

See `vernamveil encode --help` and `vernamveil decode --help` for all available options.

---

## 🛠️ Technical Details

- **Compact Implementation**: The cipher implementation is less than 300 lines of code, excluding comments and documentation.
- **External Dependencies**: Built using only Python's standard library, with NumPy being optional for vectorisation.
- **Optional C Module for Fast Hashing**: Includes an optional C module (`nphash`) built with [cffi](https://cffi.readthedocs.io/), enabling fast BLAKE2b and SHA-256 estimations for vectorised `fx` functions. See the [`nphash` README](nphash/README.md) for details.
- **Tested with**: Python 3.10 and NumPy 2.2.5.

### 🔧 Installation

To install the library with all optional dependencies (development tools, NumPy for vectorisation, and cffi for the C module):
```
    pip install .[dev,numpy,cffi]
```

- The `[dev]` extra installs development and testing dependencies.
- The `[numpy]` extra enables fast vectorised operations.
- The `[cffi]` extra builds the `nphash` C extension for accelerated BLAKE2b and SHA-256 in NumPy-based `fx` functions.

### ⚡ Fast Vectorised `fx` Functions

If you want to use fast vectorised key stream functions, install with both `numpy` and `cffi` enabled. The included `nphash` C module provides high-performance BLAKE2b and SHA-256 estimators for NumPy arrays, which are automatically used by `generate_default_fx(..., vectorise=True)` when available. If not present, a slower pure NumPy fallback is used.

For more details on the C module and its usage, see [`nphash/README.md`](nphash/README.md).

---

## 📚 Documentation

Full API and usage docs are available at: [https://datumbox.github.io/VernamVeil/](https://datumbox.github.io/VernamVeil/)

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or pull request on [GitHub](https://github.com/datumbox/VernamVeil).

---

## 📄 Copyright & License

Copyright (C) 2025 [Vasilis Vryniotis](http://blog.datumbox.com/author/bbriniotis/).

The code is licensed under the [Apache License, Version 2.0](./LICENSE).
