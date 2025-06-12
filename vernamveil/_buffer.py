"""Implements a dynamic buffer for bytearray or NumPy arrays.

Provides a specialised data structure for efficient, dynamic management of
bytearray or NumPy array buffers, supporting pre-allocation and optimised
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
        """Initialise the _Buffer.

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

        self._buffer = self.build_array(size, use_numpy)
        self._capacity = size
        self._current_pos = 0

    def __len__(self) -> int:
        """Returns the length of the used portion of the buffer.

        Returns:
            int: The length of the used portion of the buffer.
        """
        return self._current_pos

    def __str__(self) -> str:
        """Returns a string representation of the _Buffer instance.

        Returns:
            str: A string representation of the _Buffer instance.
        """
        return (
            f"_Buffer(capacity={self._capacity}, "
            f"current_pos={self._current_pos}, "
            f"type={type(self._buffer).__name__})"
        )

    @classmethod
    def build_array(
        cls, size: int = 0, use_numpy: bool = False
    ) -> "np.ndarray[tuple[int], np.dtype[np.uint8]] | bytearray":
        """Builds a bytearray or NumPy array of the specified size.

        Args:
            size (int): The initial size to pre-allocate for the buffer.
                Defaults to 0 (an empty buffer).
            use_numpy (bool): If True, a NumPy array (np.uint8) is used as the
                internal buffer. If False, a bytearray is used. Defaults to False.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.uint8]] | bytearray: The created array.
        """
        return np.empty(size, dtype=np.uint8) if use_numpy else bytearray(size)

    def extend(self, block: bytes | bytearray | memoryview) -> None:
        """Extend the buffer with the given block of data.

        If the data fits within the current capacity, it's written using slice
        assignment. Otherwise, the buffer is grown.

        Args:
            block (bytes | bytearray | memoryview): The data to append.
        """
        block_len = len(block)
        if block_len == 0:
            return

        next_pos = self._current_pos + block_len
        if next_pos <= self._capacity:
            # Data fits into the current physical capacity
            self._buffer[self._current_pos : next_pos] = (
                block if isinstance(block, memoryview) else memoryview(block)
            )
        else:
            if isinstance(self._buffer, bytearray):
                if self._capacity == self._current_pos:
                    # If no space left, extend the bytearray
                    self._buffer.extend(block)
                else:
                    # If there is space left, fill the remaining space
                    view = block if isinstance(block, memoryview) else memoryview(block)
                    remainder = self._capacity - self._current_pos
                    self._buffer[self._current_pos : self._capacity] = view[:remainder]
                    if remainder < block_len:
                        # If there is more data, extend the bytearray
                        self._buffer.extend(view[remainder:])
            else:
                # For NumPy arrays, first truncate the existing buffer if needed
                buffer = (
                    self._buffer
                    if self._capacity == self._current_pos
                    else self._buffer[: self._current_pos]
                )
                # Then concatenate the new block
                self._buffer = np.concatenate((buffer, np.frombuffer(block, dtype=np.uint8)))
            self._capacity = next_pos
        self._current_pos = next_pos

    @property
    def data(self) -> memoryview:
        """Returns a memoryview of the used portion of the buffer.

        Returns:
            memoryview: A memoryview of the used portion of the buffer.
        """
        data: memoryview = (
            memoryview(self._buffer) if isinstance(self._buffer, bytearray) else self._buffer.data
        )
        if self._capacity == self._current_pos:
            return data
        else:
            return data[: self._current_pos]

    def truncate(self, start: int, end: int | None = None) -> None:
        """Truncate the buffer in-place, keeping only the data from `start` to `end`.

        This operation discards data before `start` and after `end` (exclusive), shifting
        the logical start of the buffer. The buffer's capacity and logical position are
        updated accordingly. If `end` is None, the buffer is truncated up to its current
        physical capacity.

        Args:
            start (int): The starting index (inclusive) of the data to keep, relative to the buffer.
            end (int | None): The ending index (exclusive) of the data to keep. If None, keeps up to the buffer's capacity.

        Raises:
            ValueError: If `start` or `end` are negative, out of bounds, or if `end` < `start`.
        """
        if not isinstance(start, int) or start < 0:
            raise ValueError("start must be a non-negative integer.")
        elif start > self._capacity:
            raise ValueError("start index is out of bounds of buffer capacity.")
        if end is None:
            end = self._capacity
        elif not isinstance(end, int) or end < 0:
            raise ValueError("end must be a non-negative integer if provided.")
        elif end > self._capacity:
            raise ValueError("end index is out of bounds of buffer capacity.")
        elif end < start:
            raise ValueError("end index must be greater than or equal to start index.")

        self._buffer = self._buffer[start:end]
        self._capacity = end - start
        self._current_pos = max(0, min(self._current_pos - start, self._capacity))
