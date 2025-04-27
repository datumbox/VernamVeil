# 🔐 VernamVeil: A Function-Based Stream Cipher

> ⚠️ **DISCLAIMER:** This is an educational encryption prototype and **not** meant for real-world use. It has **not** been audited or reviewed by cryptography experts, and **should not** be used to store, transmit, or protect sensitive data.

---

## 🔎 Overview

**VernamVeil** is an experimental cipher inspired by the **One-Time Pad (OTP)** developed in Python. The name honors **Gilbert Vernam**, who is credited with the theoretical foundation of the OTP.

Instead of using a static key, VernamVeil allows the key to be represented by a function `fx(i: int | np.array, seed: bytes, bound: int | None) -> int | np.array`:
- `i`: the index of the byte in the stream  
- `seed`: a byte string that provides context and state  
- `bound`: an optional integer used to modulo the function output into the desired range (usually 256; 1 byte)
- **Output**: an integer or Numpy array representing the key stream value

_Note: `numpy` is an optional dependency, used to accelerate vectorised operations when available._

```python
from vernamveil import VernamVeil


def fx(i: int, seed: bytes, bound: int | None) -> int:
    b = seed[i % len(seed)]
    result = ((i ** 2 + i * b + b ** 2) * (i + 7))
    if bound is not None:
        result %= bound
    return result


cipher = VernamVeil(fx)
seed = cipher.get_initial_seed()
encrypted, _ = cipher.encode(b"Hello!", seed)
decrypted, _ = cipher.decode(encrypted, seed)
```

This approach enables novel forms of key generation, especially for those who enjoy playing with math and code. While this is not a secure implementation by cryptographic standards, it offers a fun and flexible way to experiment with function-based encryption.

If you're curious about how encryption works, or just want to mess with math and code in a cool way, this project is a fun starting point. For more information, read the accompanying [blog post](https://blog.datumbox.com/vernamveil-a-fresh-take-on-function-based-encryption/).

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
- **Obfuscation Techniques**:
  - Injects decoy chunks into ciphertext
  - Pads real chunks with dummy bytes
  - Shuffles output to obscure chunk boundaries
  - Chunk delimiters are randomly generated, encrypted and not exposed
- **Avalanche Effect**: Through hash-based seed refreshing, small changes in input result in large changes in output, enhancing unpredictability.
- **Authenticated Encryption**: Supports message authentication using MAC-before-decryption to detect tampering.
- **Highly Configurable**: The implementation allows the user to adjust key parameters such as `chunk_size`, `delimiter_size`, `padding_range`, `decoy_ratio`, and `auth_encrypt`, offering flexibility to tailor the encryption to specific needs or security requirements. These parameters must be aligned between encoding and decoding.
- **Vectorisation**: Some operations are vectorised using `numpy` if `vectorise=True`. Pure Python mode can be used as a fallback when `numpy` is unavailable by setting `vectorise=False`, but it is slower.
---

## ⚠️ Caveats & Best Practices

- **Not Secure for Real Use**: This is an educational tool and experimental toy, not production-ready cryptography.
- **Use Strong `fx` Functions**: The entire system's unpredictability hinges on the entropy and behavior of your `fx`. Avoid anything guessable or biased.
- **Block Delimiters Leak**: When encrypting multiple messages to the same file, plaintext delimiters remain visible. Encrypt the entire blob if full confidentiality is needed.
- **Use Secure Randomness**: For generating initial seeds favour `VernamVeil.get_initial_seed()` over `random.randbytes()`.
- **Do not reuse seeds**: Treat each `initial_seed` as a one-time-use context. It's recommended to use a fresh initial seed for every encode/decode session. During the same session, the API returns the next seed you should use for the following call.
- **The seed replaces the need for IVs/nonces**: While this implementation doesn’t use a nonce or IV in the traditional cryptographic sense, the seed serves a similar role by maintaining state between operations. Each new seed evolves from the previous one, ensuring unique keystreams and preventing key stream reuse.
- **MAC Limitations**: Tampering is detected before decryption using a message authentication code (MAC) derived from the initial seed. While this improves safety by preventing padding oracle-style issues, its overall cryptographic design remains experimental.
- **Message Ordering & Replay**: If transmitting encrypted chunks over time, ensure that any external metadata (e.g., message order, timestamps) is securely handled. While the evolving seed prevents keystream reuse, maintaining proper ordering and anti-replay mechanisms might still be necessary in some cases.
- **Performance is not optimised**: The algorithm is relatively slow due to lack of threading. Use `vectorise=True` if `numpy` is available to speed up operations.

---

## 📝 Examples

### ✉️ Encrypting and Decrypting Multiple Messages

```python
from vernamveil import VernamVeil


# Step 1: Define a custom key stream function
def fx(i: int, seed: bytes, bound: int | None) -> int:
    b = seed[i % len(seed)]
    result = ((i ** 2 + i * b + b ** 2) * (i + 7))
    if bound is not None:
        result %= bound
    return result


# Step 2: Generate a random initial seed for encryption
initial_seed = VernamVeil.get_initial_seed()  # remember to store this securely

# Step 3: Initialise VernamVeil with the custom fx and parameters
cipher = VernamVeil(fx, chunk_size=64, decoy_ratio=0.2)

# Step 4: Encrypt messages
messages = [
    "This is a secret message!",
    "another one",
    "and another one"
]
encrypted = []
seed = initial_seed
for msg in messages:
    enc, seed = cipher.encode(msg.encode(), seed)
    encrypted.append(enc)

# Step 5: Decrypt messages
seed = initial_seed
for original, enc in zip(messages, encrypted):
    dec, seed = cipher.decode(enc, seed)
    assert dec.decode() == original
```

### 📂 Encrypting and Decrypting Files

```python
from vernamveil import VernamVeil


# Step 1: Define a custom key stream function
def fx(i: int, seed: bytes, bound: int | None) -> int:
    b = seed[i % len(seed)]
    result = ((i ** 2 + i * b + b ** 2) * (i + 7))
    if bound is not None:
        result %= bound
    return result


# Step 2: Generate a random initial seed for encryption
initial_seed = VernamVeil.get_initial_seed()  # remember to store this securely

# Step 3: Encrypt a file
VernamVeil.process_file("plain.txt", "encrypted.dat", fx, initial_seed, mode="encode")

# Step 4: Decrypt the file
VernamVeil.process_file("encrypted.dat", "decrypted.txt", fx, initial_seed, mode="decode")
```

### 🧠 A marginally stronger `fx`

```python
import hashlib

def fx(i: int, seed: bytes, bound: int | None) -> int:
    # Implements a polynomial of 10 degree
    weights = [21663, 5116, -83367, -80908, 61353, -54860, 47252, 67022, 41229, 45510]
    base_modulus = 1000000000
    
    # Hash the input with the seed to get entropy
    seed_len = len(seed)
    entropy = int.from_bytes(hashlib.blake2b(seed + i.to_bytes(4, "big"), digest_size=seed_len).digest(), "big")
    base = (i + entropy) % base_modulus
    
    # Combine terms of the polynomial using weights and powers of the base
    combined_result = 0
    for power, weight in enumerate(weights, start=1):
        combined_result += weight * pow(base, power)
    result = combined_result + (base % 99991)
    
    # Modulo the result with the bound to ensure it's always within the requested range
    if bound is not None:
        result %= bound
    
    return result
```

### 🏎️ A vectorised `fx` that uses Numpy

```python
import hashlib
import numpy as np

def fx(i: np.array, seed: bytes, bound: int | None) -> np.array:
    # Implements a polynomial of 10 degree
    weights = [21663, 5116, -83367, -80908, 61353, -54860, 47252, 67022, 41229, 45510]
    base_modulus = 1000000000
    
    # Hash the input with the seed to get entropy
    int64_bound = 9223372036854775808
    seed_len = len(seed)
    entropy = np.vectorize(lambda x: int.from_bytes(hashlib.blake2b(seed + int(x).to_bytes(4, "big"), digest_size=seed_len).digest(), "big") % int64_bound)(i)
    entropy = entropy.astype(np.int64)
    base = i + entropy
    np.remainder(base, base_modulus, out=base)  # in-place modulus, avoids copy
    
    # Compute all powers in one go: shape (len(i), n)
    powers = np.power.outer(base, np.arange(1, len(weights) + 1))
    
    # Weighted sum for each element
    combined_result = np.dot(powers, weights)
    result = np.add(combined_result, np.remainder(base, 99991), dtype=np.int64)
    
    # Modulo the result with the bound to ensure it's always within the requested range
    if bound is not None:
        np.remainder(result, bound, out=result)
    
    return result
```

---

## 🛠️ Technical Details

- **Compact Implementation**: Less than 300 lines of code, excluding comments and documentation.
- **External Dependencies**: Built using only Python's standard library with Numpy being optional for vectorisation.
- **Tested with**: Python 3.10 and Numpy 2.2.5.

