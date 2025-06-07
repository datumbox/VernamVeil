from vernamveil._types import _kmpffi


def find_all(haystack: bytes | bytearray, needle: bytes | bytearray | memoryview) -> list[int]:
    """Finds all occurrences of pattern_bytes in text_bytes using KMP.

    Args:
        haystack (bytes or bytearray): The bytes object to search within.
        needle (bytes or bytearray or memoryview): The bytes object to search for.

    Returns:
        list[int]: A list of 0-based starting indices of all occurrences. Returns an empty list if
            no occurrences are found or if the pattern is empty or longer than the text.
    """
    result_indices: list[int] = []
    n = len(haystack)
    m = len(needle)

    if m == 0 or n == 0 or m > n:
        return result_indices

    if _kmpffi is None:
        # Fallback to Python implementation if C library is not available
        look_start = 0  # Start position for searching the next needle
        while look_start < n:
            # Search for the next occurrence of the delimiter
            idx = haystack.find(needle, look_start)
            if idx == -1:
                # No more needle found
                break
            # Append the found index to the result
            result_indices.append(idx)
            # Move the search start past the current needle
            look_start = idx + m
    else:
        # Use the C extension for KMP search
        ffi = _kmpffi.ffi

        count_ptr = ffi.new("size_t *")
        indices_ptr = _kmpffi.lib.find_all_kmp(
            ffi.from_buffer(haystack), n, ffi.from_buffer(needle), m, count_ptr
        )
        count = count_ptr[0]

        if indices_ptr is not ffi.NULL and count > 0:
            # convert the C array of size_t to a Python list in one go
            result_indices = ffi.unpack(indices_ptr, count)
            _kmpffi.lib.free_indices_kmp(indices_ptr)

    return result_indices
