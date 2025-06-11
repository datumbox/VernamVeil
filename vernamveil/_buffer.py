"""Implements a dynamic buffer for bytearray or NumPy arrays.

Provides a specialized data structure for efficient, dynamic management of
bytearray or NumPy array buffers, supporting pre-allocation and optimized
extend operations.
"""

from vernamveil._types import np


class _Buffer:
    """A dynamic buffer that can use either a bytearray or a NumPy array.

    This class manages an internal buffer that can be pre-allocated to a certain
    size. It supports extending the buffer with new data. If the new data fits
    within the current capacity, it's written using slice assignment. If the
    buffer needs to grow, it uses `bytearray.extend` or `numpy.concatenate`.
    """

    def __init__(self, size: int = 0, use_numpy: bool = False) -> None:
        """Initialise the _DynamicBuffer.

        Args:
            size (int): The initial size to pre-allocate for the buffer.
                Defaults to 0 (an empty buffer).
            use_numpy (bool): If True, a NumPy array (np.uint8) is used as the
                internal buffer. If False, a bytearray is used. Defaults to False.

        Raises:
            ValueError: If `size` is not a non-negative integer.
            TypeError: If `vectorised` is not a boolean.
        """
        if not isinstance(size, int) or size < 0:
            raise ValueError("size must be a non-negative integer.")
        if not isinstance(use_numpy, bool):
            raise TypeError("use_numpy must be a boolean.")

        self._current_pos: int = 0
        self._capacity: int = size

        self._buffer: np.ndarray[tuple[int], np.dtype[np.uint8]] | bytearray
        if use_numpy:
            self._buffer = np.empty(size, dtype=np.uint8)
        else:
            self._buffer = bytearray(size)

    def extend(self, data_to_add: bytes | bytearray | memoryview) -> None:
        """Extend the buffer with the given data.

        If the data fits within the current capacity, it's written using slice
        assignment. Otherwise, the buffer is grown.

        Args:
            data_to_add (bytes | bytearray | memoryview): The data to append.
        """
        data_len = len(data_to_add)
        if data_len == 0:
            return

        next_pos = self._current_pos + data_len
        if next_pos <= self._capacity:
            # Data fits into the current physical capacity
            self._buffer[self._current_pos : next_pos] = (
                data_to_add if isinstance(data_to_add, memoryview) else memoryview(data_to_add)
            )
            self._current_pos = next_pos
        else:
            if isinstance(self._buffer, np.ndarray):
                buffer = (
                    self._buffer
                    if self._capacity == self._current_pos
                    else self._buffer[: self._current_pos]
                )
                self._buffer = np.concatenate((buffer, np.frombuffer(data_to_add, dtype=np.uint8)))
            else:
                if self._capacity == self._current_pos:
                    self._buffer.extend(data_to_add)
                else:
                    view = (
                        data_to_add
                        if isinstance(data_to_add, memoryview)
                        else memoryview(data_to_add)
                    )
                    remainder = self._capacity - self._current_pos
                    self._buffer[self._current_pos : self._capacity] = view[:remainder]
                    self._buffer.extend(view[remainder:])
            self._current_pos = next_pos
            self._capacity = self._current_pos

    @property
    def data(self) -> memoryview:
        """Returns a memoryview of the used portion of the buffer.

        Returns:
            memoryview: A memoryview of the used portion of the buffer.
        """
        data: memoryview = (
            self._buffer.data if isinstance(self._buffer, np.ndarray) else memoryview(self._buffer)
        )
        if self._capacity == self._current_pos:
            return data
        else:
            return data[: self._current_pos]

    def __len__(self) -> int:
        """Returns the length of the used portion of the buffer.

        Returns:
            int: The length of the used portion of the buffer.
        """
        return self._current_pos

    def __str__(self) -> str:
        """Returns a string representation of the _DynamicBuffer instance.

        Returns:
            str: A string representation of the _DynamicBuffer instance.
        """
        return (
            f"_DynamicBuffer(current_pos={self._current_pos}, "
            f"capacity={self._capacity}, "
            f"type={type(self._buffer).__name__})"
        )

    def truncate(self, start: int, end: int | None = None) -> None:
        """Truncates the buffer in-place to keep only its logical content from `start` to `end`.

        The buffer's content is replaced with a copy of the specified slice of its
        previous logical content.

        Args:
            start (int): The starting logical index of the content to keep.
            end (int | None): The ending logical index (exclusive) of the content to keep.
                              If None, keeps content until the current logical end.
                              Defaults to None.

        Raises:
            ValueError: If `start` or `end` indices are invalid (e.g., negative, out of bounds of the current logical content, or end < start).
        """
        if not isinstance(start, int) or start < 0:
            raise ValueError("start must be a non-negative integer.")
        elif start > self._current_pos:
            raise ValueError("start index is out of bounds of current logical content.")
        if end is None:
            end = self._current_pos
        elif not isinstance(end, int) or end < 0:
            raise ValueError("end must be a non-negative integer if provided.")
        elif end > self._current_pos:
            raise ValueError("end index is out of bounds of current logical content.")
        elif end < start:
            raise ValueError("end index must be greater than or equal to start index.")

        # Slice the buffer to keep only the specified range.
        self._buffer = self._buffer[start:end]

        # Update current position and capacity to reflect the new content.
        self._current_pos = len(self._buffer)
        self._capacity = self._current_pos
