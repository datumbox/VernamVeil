"""Utility for plausible deniability with VernamVeil encrypted messages.

Given a VernamVeil-encrypted byte string and a user-supplied decoy plaintext (plus the cypher configuration),
this tool constructs a plausible fx function and a fake seed, so that decrypting the cyphertext yields the decoy message.

The utility reuses as many private methods of VernamVeil as possible to ensure compatibility with future updates.
"""

import copy
import math

from vernamveil._fx_utils import FX, OTPFX
from vernamveil._vernamveil import VernamVeil

__all__ = [
    "forge_plausible_fx",
]


def _find_obfuscated_decoy_message(
    cypher: VernamVeil,
    decoy_message: bytes,
    target_len: int,
    max_attempts: int,
) -> tuple[bytes, bytes, memoryview]:
    """Tries to produce an obfuscated version of the decoy message with the exact desired target length.

    Args:
        cypher (VernamVeil): The VernamVeil instance used for obfuscation.
        decoy_message (bytes): The decoy message to obfuscate.
        target_len (int): The desired length of the obfuscated message.
        max_attempts (int): The maximum number of attempts to find a valid obfuscated message.

    Returns:
        tuple[bytes, bytes, memoryview]: A tuple containing the obfuscated message, the fake seed,
        and the delimiter.

    Raises:
        ValueError: If a valid obfuscated decoy of the desired length cannot be found within the maximum attempts.
    """
    decoy_view = memoryview(decoy_message)
    for _ in range(max_attempts):
        # Generate random fake seed
        fake_seed = cypher.get_initial_seed()

        # Generate delimiter and evolve the seed
        delimiter, seed = cypher._generate_delimiter(fake_seed)

        # Generate shuffle seed for obfuscation
        shuffle_seed = cypher._hash(seed, [b"shuffle"])

        # Obfuscate the decoy message
        obfuscated = cypher._obfuscate(decoy_view, shuffle_seed, delimiter)

        # Check if the obfuscated message has the desired length
        if len(obfuscated) == target_len:
            return obfuscated.tobytes(), fake_seed, delimiter

    raise ValueError(
        f"Could not find obfuscated decoy of length {target_len} in {max_attempts} attempts. "
        f"Try different decoy length or adjust cypher parameters."
    )


def _estimate_obfuscated_length(cypher: VernamVeil, message_len: int, pad_val: int) -> int:
    """Estimates the obfuscated cyphertext length for a decoy message, given a fixed padding value.

    Args:
        cypher (VernamVeil): The VernamVeil instance used for obfuscation.
        message_len (int): The length of the decoy message.
        pad_val (int): The padding value to use for the estimation.

    Returns:
        int: The estimated length of the obfuscated cyphertext.
    """
    chunk_size = cypher._chunk_size
    delimiter_size = cypher._delimiter_size
    decoy_ratio = cypher._decoy_ratio

    real_count = math.ceil(message_len / chunk_size)
    decoy_count = max(1, int(decoy_ratio * real_count)) if decoy_ratio > 0 else 0
    total_count = real_count + decoy_count

    return 2 * total_count * (delimiter_size + pad_val) + message_len + decoy_count * chunk_size


def forge_plausible_fx(
    cypher: VernamVeil,
    cyphertext: bytes,
    decoy_message: bytes,
    max_obfuscate_attempts: int = 1_000,
) -> tuple[FX, bytes]:
    """Generates a fake keystream and seed to plausibly decrypt a cyphertext to a decoy message.

    This function enables plausible deniability: it lets you demonstrate that an encrypted file
    could plausibly contain a harmless message, by generating the necessary cryptographic
    parameters to make the decryption appear valid. The original encryption remains secure,
    but you can provide a decoy message and matching decryption parameters to anyone demanding
    access, without revealing the true content.

    Args:
        cypher (VernamVeil): The VernamVeil instance used for encryption.
        cyphertext (bytes): The encrypted cyphertext.
        decoy_message (bytes): The decoy message to forge the keystream for.
        max_obfuscate_attempts (int): The maximum number of attempts to find a valid obfuscated decoy message.
            Defaults to 1,000.

    Returns:
        tuple[FX, bytes]: A tuple containing the plausible fx function and the fake seed.

    Raises:
        ValueError: If the decoy message cannot plausibly fit the cyphertext length given the cypher parameters.
    """
    # 1. Prepare a cypher with SIV and MAC off
    cypher = copy.deepcopy(cypher)
    cypher._siv_seed_initialisation = False
    cypher._auth_encrypt = False

    # 2. Estimate the plausible boundaries for the cyphertext length
    cyphertext_len = len(cyphertext)
    decoy_message_len = len(decoy_message)
    pad_min, pad_max = cypher._padding_range
    lower_bound = _estimate_obfuscated_length(cypher, decoy_message_len, pad_min)
    upper_bound = _estimate_obfuscated_length(cypher, decoy_message_len, pad_max)

    if not (lower_bound <= cyphertext_len <= upper_bound):
        raise ValueError(
            f"Cannot plausibly forge decoy message of length {decoy_message_len} for cyphertext of length {cyphertext_len} "
            f"with the current cypher parameters. The expected cyphertext length for this decoy is between "
            f"{lower_bound} and {upper_bound} bytes. Please adjust the decoy message length or cypher settings."
        )

    # 3. Find an obfuscated decoy message of the right length
    obfuscated, fake_seed, delimiter = _find_obfuscated_decoy_message(
        cypher, decoy_message, cyphertext_len, max_attempts=max_obfuscate_attempts
    )

    # 3. Generate the delimiter bytes and make sure they are a multiple of block_size
    block_size = cypher._fx.block_size
    delimiter_len = math.ceil(len(delimiter) / block_size) * block_size
    delimiter_bytes = delimiter.tobytes()
    padding_len = delimiter_len - len(delimiter)
    if padding_len > 0:
        delimiter_bytes += VernamVeil.get_initial_seed(num_bytes=padding_len)

    # 4. Prepend the delimiter bytes to the keystream_values
    keystream_values = [
        delimiter_bytes[i : i + block_size] for i in range(0, delimiter_len, block_size)
    ]

    # 5. Recover the keystream: keystream = cyphertext ^ obfuscated
    # We need to recover the chunk ranges for the obfuscated message to handle the case where
    # the chunk_size is not a multiple of block_size. This can lead to the fx sampling the wrong bytes.
    for start, end in cypher._generate_chunk_ranges(cyphertext_len):
        for block_start in range(start, end, block_size):
            ct_block = cyphertext[block_start : block_start + block_size]
            obf_block = obfuscated[block_start : block_start + block_size]

            # Pad to block_size bytes if needed using random bytes
            padding_len = block_size - len(ct_block)
            if padding_len > 0:
                ct_block += VernamVeil.get_initial_seed(num_bytes=padding_len)
                obf_block += VernamVeil.get_initial_seed(num_bytes=padding_len)

            ks = bytes(a ^ b for a, b in zip(ct_block, obf_block))
            keystream_values.append(ks)

    # 6. Generate the fx function
    plausible_fx = OTPFX(keystream_values, block_size, cypher._fx.vectorise)

    return plausible_fx, fake_seed
