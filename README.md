# 🔐 VernamVeil: A Function-Based Stream Cypher

[![CI](https://github.com/datumbox/VernamVeil/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/datumbox/VernamVeil/actions) [![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://datumbox.github.io/VernamVeil/) [![License](https://img.shields.io/:license-apache-brightgreen.svg)](./LICENSE)

---

> ⚠️ **DISCLAIMER:** This is an educational encryption prototype and **not** meant for real-world use. It has **not** been audited or reviewed by cryptography experts, and **should not** be used to store, transmit, or protect sensitive data.

## 🚀 Quick Start

Minimal Installation (without vectorisation or C extension support):
```bash
pip install .
```

Minimal Example:
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
cypher = VernamVeil(fx)
encrypted, _ = cypher.encode(b"Hello!", initial_seed)
decrypted, _ = cypher.decode(encrypted, initial_seed)
```

---

## 🔎 Overview

**VernamVeil** is an experimental cypher inspired by the **One-Time Pad (OTP)** developed in Python. The name honours **Gilbert Vernam**, who is credited with the theoretical foundation of the OTP.

Instead of using a static key, VernamVeil allows the key to be represented by a function `fx(i: int | np.ndarray, seed: bytes, bound: int | None) -> int | np.ndarray`:
- `i`: the index of the bytes in the message; a scalar integer or an uint64 NumPy array with a continuous enumeration for vectorised operations.
- `seed`: a byte string that provides context and state; should be kept secret.
- `bound`: an optional integer used to modulo the function output into the desired range (usually `2**64` because we sample 8 bytes at a time).
- **Output**: an integer or an uint64 NumPy array representing the key stream values.

_Note: `numpy` is an optional but highly recommended dependency, used to accelerate vectorised operations when available._

This approach enables novel forms of key generation, especially for those who enjoy playing with maths and code. While this is not a secure implementation by cryptographic standards, it offers a fun and flexible way to experiment with function-based encryption. If you're curious about how encryption works, or just want to mess with maths and code in a cool way, this project is a fun starting point. For more information, read the accompanying [blog post](https://blog.datumbox.com/vernamveil-a-fresh-take-on-function-based-encryption/).

---

## 💡 Why VernamVeil?

- **Using a function as a key** is appealing compared to fixed-size keys that repeat, which create vulnerabilities. While an infinitely long key would be uncrackable, a weak function introduces similar risks to repeating keys.
- Its **modular structure** means anyone can build their own `fx` functions with creative mathematical expressions or external data.
- **Inspired by the OTP**, VernamVeil supports long, non-repeating key streams, as long as your function and seed combination allows it.

---

## ✨ Cryptographic Characteristics

1. **Function-Based, Symmetric, OTP-Inspired Cypher**: VernamVeil uses a user-defined function `fx` and an `initial_seed` (both secret) to dynamically generate the key stream. This approach is symmetric, identical secrets and encryption configuration are required for both encryption and decryption, making the process fully reversible. The design is inspired by the One-Time Pad, supporting non-repeating, functionally generated keys.
2. **Synthetic IV Seed Initialisation, Stateful Seed Evolution & Avalanche Effects**: Instead of a traditional nonce, the first internal seed is derived using a Synthetic IV computed as an HMAC of the user-provided initial seed and the full plaintext (inspired by [RFC 5297](https://datatracker.ietf.org/doc/html/rfc5297)). For each chunk, the seed is further evolved by HMACing the previous seed with the chunk's plaintext, maintaining state between operations. This hash-based seed refreshing ensures each keystream is unique, prevents keystream reuse, provides resilience against seed reuse and deterministic output, and produces an avalanche effect: small changes in input result in large, unpredictable changes in output. The scheme does not allow backward derivation of seeds, if a current seed is leaked, past messages remain secure (backward secrecy is preserved).
3. **Message Obfuscation, Zero Metadata & Authenticated Encryption**: The cypher injects decoy chunks, pads real chunks with dummy bytes, and shuffles output to obscure chunk boundaries, complicating cryptanalysis methods such as traffic analysis or block boundary detection. All delimiters are randomly generated, encrypted, and not exposed; file delimiters are refreshed for every block. The cyphertext contains no embedded metadata, minimizing the risk of attackers identifying recurring patterns or structural information. All encryption details are deterministically recovered from the `fx`, except for the configuration parameters (e.g., chunk and delimiter sizes) which must be provided and matched exactly during decryption, or the MAC check will fail. During encryption, Message Authentication is enforced using a standard encrypt-then-MAC (EtM) construction. During decryption verification-before-decryption is used to detect tampering and prevent padding oracle-style issues.
4. **Modular & Configurable Keystream Design**: The `fx` function can be swapped to explore different styles of pseudorandom generation, including custom PRNGs or cryptographic hashes. The implementation also allows full adjustment of configuration, offering flexibility to tailor encryption to specific needs. 
5. **Vectorisation & Optional C-backed Fast Hashing**: All operations are vectorised using `numpy` when `vectorise=True`, with a slower pure Python fallback available. For even faster vectorised `fx` functions, an optional C module (`nphash`) can be installed (with `cffi` and system dependencies), enabling high-performance BLAKE2b and SHA-256 hashing for NumPy-based key stream generation. This is supported both in user-defined `fx` methods and automatically by helpers like `generate_default_fx`. See [`nphash/README.md`](nphash/README.md) for details.

---

## ⚠️ Caveats & Best Practices

- **Not Secure for Real Use**: This is an educational tool and experimental toy, not production-ready cryptography.
- **Use Strong `fx` Functions**: The entire system's unpredictability hinges on the entropy and behaviour of your `fx`. Avoid anything guessable or biased and the use of periodic mathematical functions which can lead to predictable or repeating outputs.
- **Use Secure Seeds & Avoid Reuse**: Generate initial seeds using the provided `VernamVeil.get_initial_seed()` method which is cryptographically safe. Treat each `initial_seed` as a one-time-use context and use a fresh initial seed for every encode/decode session. During the same session, the API returns the next seed you should use for the following call.
- **Message Ordering & Replay**: VernamVeil is designed to be nonce-free by evolving the seed with each message or chunk, ensuring keystream uniqueness as long as each session uses a distinct `initial_seed`. The Synthetic IV mechanism provides additional resilience against accidental seed reuse for the first message. However, the cypher itself does not enforce message ordering or replay protection; these must be handled by the application. For most use cases, careful state management and unique seeds are sufficient, but applications with strict anti-replay or ordering requirements should implement explicit mechanisms at a higher layer.

---

## 📝 Examples

### ✉️ Encrypting and Decrypting Multiple Messages

```python
from vernamveil import VernamVeil, generate_default_fx


# Step 1: Generate a random custom fx using the helper
fx = generate_default_fx()  # remember to store fx._source_code securely

# Step 2: Generate a random initial seed for encryption
initial_seed = VernamVeil.get_initial_seed()  # remember to store this securely

# Step 3: Initialise VernamVeil with the custom fx and parameters
cypher = VernamVeil(fx, chunk_size=64, decoy_ratio=0.2)

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
    enc, seed = cypher.encode(msg.encode(), seed)
    encrypted.append(enc)

# Step 5: Decrypt multiple messages in one session
seed = initial_seed
for original, enc in zip(messages, encrypted):
    # Each message evolves the seed for the next one
    dec, seed = cypher.decode(enc, seed)
    assert dec.decode() == original
```

### 📂 Encrypting and Decrypting Files

```python
from vernamveil import VernamVeil, generate_default_fx


# Step 1: Generate a random custom fx using the helper
fx = generate_default_fx()  # remember to store fx._source_code securely

# Step 2: Generate a random initial seed for encryption
initial_seed = VernamVeil.get_initial_seed()  # remember to store this securely

# Step 3: Initialise VernamVeil with the custom fx and parameters
cypher = VernamVeil(fx, chunk_size=64, decoy_ratio=0.2)

# Step 4: Encrypt a file
cypher.process_file("encode", "plain.txt", "encrypted.dat", initial_seed)

# Step 5: Decrypt the file
cypher.process_file("decode", "encrypted.dat", "decrypted.txt", initial_seed)
```

> **Note:**
> The `process_file` method uses background threads and queues to perform asynchronous I/O for both reading and writing, enabling efficient processing of large files without blocking the main thread.

---

## 🧪💻 How to Design a Custom `fx`

> ⚠️ **Warning: Designing cryptographic functions is difficult and risky**
>
> Creating your own cryptographic methods is a major undertaking, and even small mistakes can introduce severe vulnerabilities. The greatest weakness of this cypher is that it allows users to supply their own `fx` functions: a non-expert can easily "shoot themselves in the foot" by designing a function that is predictable, biased, or otherwise insecure, potentially making the encryption trivial to break. This project is strictly educational and not intended for real-world security. The following section provides some basic principles for designing `fx` functions, but it is not a comprehensive guide to cryptographic engineering.

When creating your own key stream function (`fx`), it is essential to follow best practices to ensure the unpredictability and security of your cypher. Poorly designed functions can introduce vulnerabilities, bias, or even make the encryption reversible by attackers. Use the following guidelines:

- **Uniform & Non-Constant Output**: Your `fx` should produce diverse, unpredictable outputs for different input indices. Avoid constant, biased, low-entropy, or periodic mathematical functions. The distribution of outputs should be as uniform as possible.
- **Seed Sensitivity**: The output of `fx` must depend on the secret seed. Changing the seed should result in completely different outputs.
- **Respect the Bound**: Always ensure that the output of `fx` is within the range `[0, bound)`, where `bound` is provided as an argument.
- **Type Correctness**: The function must return an `int` (or a NumPy `uint64` array in vectorised mode).
- **Determinism**: `fx` must be deterministic for the same inputs. Do not use external state or randomness inside your function.
- **Avoid Data-Dependent Branching or Timing**: Do not introduce data-dependent branching or timing in your `fx`, as this can lead to side-channel attacks.
- **Performance**: Complex or slow `fx` functions will slow down encryption and decryption. Test performance if speed is important for your use case.

**Recommended approach:**  
Apply a unique, high-entropy transformation to the input index using a function that incorporates constant but randomly sampled parameters to make each `fx` instance unpredictable. Then, combine the result with the secret seed using a cryptographically secure method such as HMAC. This ensures your keystream is both unpredictable and securely bound to your secret.

**Always test your custom `fx`** with the provided `check_fx_sanity` utility before using it for encryption. Note that this method only performs very basic checks and cannot guarantee cryptographic security; it may catch common mistakes, but passing all checks does not mean your function is secure.

Below we provide some example `fx` methods to illustrate these principles in practice:

### 🧠 A more robust, HMAC-based scalar `fx` (not cryptographically standard)

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

### 🏎️ A fast version of the above `fx` that uses NumPy vectorisation and the `nphash` C module

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

### 🛡️ A cryptographically strong HMAC-SHA256 `fx` (vectorised & C-accelerated)

```python
from vernamveil import hash_numpy
import numpy as np


def fx(i: np.ndarray, seed: bytes, bound: int | None) -> np.ndarray:
    # Implements a standard HMAC-based pseudorandom function (PRF) using sha256.
    # The output is deterministically derived from the input index `i` and the secret `seed`.
    # Security relies entirely on the secrecy of the seed and the cryptographic strength of HMAC.

    # Cryptographic HMAC using sha256
    result = hash_numpy(i, seed, "sha256")  # uses C module if available, else NumPy fallback

    # Modulo the result with the bound to ensure it's always within the requested range
    if bound is not None:
        np.remainder(result, bound, out=result)

    return result
```

---

## 🧰 Provided `fx` Utilities

VernamVeil includes helper tools to make working with key stream functions easier:

- `check_fx_sanity`: Runs basic sanity checks on your custom `fx` to ensure it produces diverse, seed-sensitive, and well-bounded outputs.
- `generate_default_fx` (same as `generate_polynomial_fx`): Generates a random `fx` function that first transforms the index using a polynomial with random weights, then applies HMAC (Blake2b) for cryptographic output. Supports both scalar and vectorised (NumPy) modes.
- `generate_hmac_fx`: Generates a deterministic `fx` function that applies HMAC (using a specified hash algorithm, e.g., BLAKE2b or SHA-256) directly to the index and seed. The seed is the only secret key but HMAC is a cryptographically strong and proven `fx`. Supports both scalar and vectorised (NumPy) modes.
- `load_fx_from_file`: Loads a custom `fx` function from a Python file. This is useful for testing and validating your own implementations. This uses `exec` internally to execute the file's code. **Never use this with files from untrusted sources, as it can run arbitrary code on your system.**

These utilities help you prototype and validate your own key stream functions before using them in encryption.

Example:

```python
from vernamveil import generate_default_fx, check_fx_sanity


# Generate a vectorised fx function
fx = generate_default_fx(vectorise=True)

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

- **Encrypt and decrypt files or streams** using a user-defined or auto-generated `fx` function and seed.
- **Auto-generate `fx.py` and `seed.bin`** during encoding if not provided; these files are saved in the current working directory.
- **Custom `fx` and seed support**: Supply your own `fx.py` and `seed.bin` for both encoding and decoding.
- **Configurable parameters**: Adjust chunk size, delimiter size, padding, decoy ratio, and more. Set `--verbosity info` to see progress information (off by default).
- **Sanity checks**: Optionally verify that your `fx` function is suitable for cryptographic use.

### 💻 Usage

```commandline
# Encrypt a file with auto-generated fx and seed
vernamveil encode --infile plain.txt --outfile encrypted.dat

# Encrypt a file with a custom fx function and seed
vernamveil encode --infile plain.txt --outfile encrypted.dat --fx-file fx.py --seed-file seed.bin

# Decrypt a file (requires the same fx and seed used for encryption)
vernamveil decode --infile encrypted.dat --outfile decrypted.txt --fx-file fx.py --seed-file seed.bin

# Encrypt and Decrypt from stdin to stdout (using - or omitting the argument)
vernamveil encode --infile - --outfile - --fx-file fx.py --seed-file seed.bin < plain.txt > encrypted.dat
vernamveil decode --infile - --outfile - --fx-file fx.py --seed-file seed.bin < encrypted.dat > decrypted.txt

# Enable sanity check for fx and seed during encryption
vernamveil encode --infile plain.txt --outfile encrypted.dat --fx-file fx.py --seed-file seed.bin --check-sanity
```

> ⚠️ **Warning: CLI Parameter Consistency**
>
> When decoding, you **must** use the exact same parameters (such as `--chunk-size`, `--delimiter-size`, `--padding-range`, `--decoy-ratio`, `--siv-seed-initialisation`, `--auth-encrypt`, and `--vectorise`) as you did during encoding.
>
> For example, the following will **fail** with a `Authentication failed: MAC tag mismatch.` error because the `--chunk-size` parameter differs between encoding and decoding:
>
> ```commandline
> vernamveil encode --infile plain.txt --outfile encrypted.dat --chunk-size 2048
> vernamveil decode --infile encrypted.dat --outfile decrypted.txt --chunk-size 1024 --fx-file fx.py --seed-file seed.bin
> ```
>
> **Always use identical parameters for both encoding and decoding.** Any mismatch will result in decryption failure. The only exception is the `--buffer-size` parameter, which can be different for encoding and decoding.

### 🗄️ File Handling

- For both `--infile` and `--outfile`, passing `-` or omitting the argument means `stdin`/`stdout` will be used. This allows for piping and streaming data directly.
- When encoding **without** `--fx-file` or `--seed-file`, the CLI generates `fx.py` and `seed.bin` in the current working directory. The absolute paths to these files are displayed after generation. **Store these files securely**; they are required for decryption.
- When decoding, you **must** provide both `--fx-file` and `--seed-file` pointing to the originals used for encryption.
- For safety, the CLI will **not overwrite** existing output files, `fx.py`, or `seed.bin`. If these files already exist, you must delete or rename them manually before running the command. Overwrite protection does **not** apply when outputting to `stdout`.

> ⚠️ **Warning: Binary Output to Terminals**
>
> If you use `-` or omit `--outfile`, output will be written to `stdout` in binary mode. Writing binary data directly to a terminal may corrupt your session. Only redirect binary output to files or pipes, not to an interactive terminal.

See `vernamveil encode --help` and `vernamveil decode --help` for all available options.

---

## 🛠️ Technical Details

- **Compact Implementation**: The core algorithm implementation is about 200 lines of code, excluding comments, documentation and empty lines.
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

If you want to use fast vectorised key stream functions, install with both `numpy` and `cffi` enabled. The included `nphash` C module provides high-performance BLAKE2b and SHA-256 estimators for NumPy arrays, which are automatically used by `generate_default_fx(vectorise=True)` when available. If not present, a slower pure NumPy fallback is used.

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
