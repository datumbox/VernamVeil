"""Provides fast byte search functionalities.

Offers `find` and `find_all` operations, utilising a C extension for performance
when available, with a Python fallback mechanism.
"""

from vernamveil._imports import _HAS_C_MODULE, _bytesearchffi

__all__ = ["find", "find_all"]


def find(
    haystack: bytes | bytearray | memoryview,
    needle: bytes | bytearray | memoryview,
    start: int = 0,
    end: int | None = None,
) -> int:
    """Finds the first occurrence of needle in haystack[start:end] using the fast C byte search, or Python fallback.

    Args:
        haystack (bytes or bytearray or memoryview): The bytes object to search within.
        needle (bytes or bytearray or memoryview): The bytes object to search for.
        start (int): The starting index to search from. Defaults to 0.
        end (int, optional): The ending index (exclusive) to search to. Defaults to None (end of haystack).

    Returns:
        int: The 0-based starting index of the first occurrence, or -1 if not found.
    """
    if not _HAS_C_MODULE:
        # Fallback to Python implementation if C library is not available
        bytes_haystack = haystack.tobytes() if isinstance(haystack, memoryview) else haystack
        idx = bytes_haystack.find(needle, start, end)
        return idx
    else:
        # Use the C extension for byte search

        # Validate input types
        n = len(haystack)

        if start < 0:
            start = max(n + start, 0)
        if end is None:
            end = n
        elif end < 0:
            end = max(n + end, 0)

        m = len(needle)
        if m == 0:
            # Python's behavior for empty needle
            if start > n:
                return -1
            return start
        sub_n = end - start
        if sub_n <= 0 or m > sub_n:
            return -1

        ffi = _bytesearchffi.ffi

        view = haystack if isinstance(haystack, memoryview) else memoryview(haystack)
        idx = _bytesearchffi.lib.find(
            ffi.from_buffer(view[start:end]), sub_n, ffi.from_buffer(needle), m
        )
        if idx == -1:
            return -1
        return int(idx) + start


def find_all(
    haystack: bytes | bytearray | memoryview, needle: bytes | bytearray | memoryview
) -> list[int]:
    """Finds all occurrences of needle in haystack using a fast byte search algorithm.

    Args:
        haystack (bytes or bytearray or memoryview): The bytes object to search within.
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

    if not _HAS_C_MODULE:
        # Fallback to Python implementation if C library is not available
        bytes_haystack = haystack.tobytes() if isinstance(haystack, memoryview) else haystack

        look_start = 0  # Start position for searching the next needle
        while look_start < n:
            # Search for the next occurrence of the delimiter
            idx = bytes_haystack.find(needle, look_start)
            if idx == -1:
                # No more needle found
                break
            # Append the found index to the result
            result_indices.append(idx)
            # Move the search start past the current needle
            look_start = idx + m
    else:
        # Use the C extension for byte search
        ffi = _bytesearchffi.ffi

        count_ptr = ffi.new("size_t *")
        indices_ptr = _bytesearchffi.lib.find_all(
            ffi.from_buffer(haystack), n, ffi.from_buffer(needle), m, count_ptr, 0
        )
        count = count_ptr[0]

        if indices_ptr is not ffi.NULL and count > 0:
            # convert the C array to a Python list in one go
            result_indices = ffi.unpack(indices_ptr, count)
            _bytesearchffi.lib.free_indices(indices_ptr)

    return result_indices
