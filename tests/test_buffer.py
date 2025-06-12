import unittest

from vernamveil._buffer import _Buffer
from vernamveil._types import _HAS_NUMPY, np


class TestBuffer(unittest.TestCase):
    """Unit tests for the _Buffer class."""

    def _run_test_for_buffer_types(self, test_func, *args, **kwargs):
        """Helper to run a test for both bytearray and (if available) numpy buffers."""
        with self.subTest(buffer_type="bytearray"):
            test_func(use_numpy=False, *args, **kwargs)

        if _HAS_NUMPY:
            with self.subTest(buffer_type="numpy"):
                test_func(use_numpy=True, *args, **kwargs)
        elif kwargs.get("use_numpy") is True:
            self.skipTest("NumPy not available or not installed.")

    def test_initialisation(self):
        """Test buffer initialisation with various configurations."""

        def _test(use_numpy):
            # Default initialisation
            buf = _Buffer(use_numpy=use_numpy)
            self.assertEqual(len(buf), 0)
            self.assertEqual(buf._capacity, 0)
            self.assertIsInstance(buf.data, memoryview)
            self.assertEqual(len(buf.data), 0)
            if use_numpy:
                self.assertIsInstance(buf._buffer, np.ndarray)
            else:
                self.assertIsInstance(buf._buffer, bytearray)

            # initialisation with size
            buf = _Buffer(size=100, use_numpy=use_numpy)
            self.assertEqual(len(buf), 0)
            self.assertEqual(buf._capacity, 100)
            self.assertEqual(len(buf._buffer), 100)
            if use_numpy:
                self.assertIsInstance(buf._buffer, np.ndarray)
            else:
                self.assertIsInstance(buf._buffer, bytearray)

        self._run_test_for_buffer_types(_test)

    def test_initialisation_invalid_args(self):
        """Test buffer initialisation with invalid arguments."""
        with self.assertRaisesRegex(ValueError, "size must be a non-negative integer."):
            _Buffer(size=-1)
        with self.assertRaisesRegex(ValueError, "size must be a non-negative integer."):
            _Buffer(size="abc")  # type: ignore
        with self.assertRaisesRegex(TypeError, "use_numpy must be a boolean."):
            _Buffer(use_numpy="True")  # type: ignore

    def test_build_array_classmethod(self):
        """Test the build_array classmethod."""
        # Test bytearray
        arr_ba = _Buffer.build_array(size=10, use_numpy=False)
        self.assertIsInstance(arr_ba, bytearray)
        self.assertEqual(len(arr_ba), 10)

        arr_ba_zero = _Buffer.build_array(size=0, use_numpy=False)
        self.assertIsInstance(arr_ba_zero, bytearray)
        self.assertEqual(len(arr_ba_zero), 0)

        if _HAS_NUMPY:
            # Test numpy array
            arr_np = _Buffer.build_array(size=20, use_numpy=True)
            self.assertIsInstance(arr_np, np.ndarray)
            self.assertEqual(arr_np.dtype, np.uint8)
            self.assertEqual(len(arr_np), 20)

            arr_np_zero = _Buffer.build_array(size=0, use_numpy=True)
            self.assertIsInstance(arr_np_zero, np.ndarray)
            self.assertEqual(len(arr_np_zero), 0)

    def test_extend_empty_buffer(self):
        """Test extending an empty buffer."""

        def _test(use_numpy):
            buf = _Buffer(use_numpy=use_numpy)
            data_to_add = b"hello"
            buf.extend(data_to_add)
            self.assertEqual(len(buf), 5)
            self.assertEqual(buf.data.tobytes(), data_to_add)
            self.assertEqual(buf._current_pos, 5)
            self.assertEqual(buf._capacity, 5)

        self._run_test_for_buffer_types(_test)

    def test_extend_with_empty_block(self):
        """Test extending with an empty block of data."""

        def _test(use_numpy):
            buf = _Buffer(size=10, use_numpy=use_numpy)
            buf.extend(b"hello")
            len_before = len(buf)
            capacity_before = buf._capacity
            current_pos_before = buf._current_pos

            buf.extend(b"")  # Extend with empty
            self.assertEqual(len(buf), len_before)
            self.assertEqual(buf._capacity, capacity_before)
            self.assertEqual(buf._current_pos, current_pos_before)
            self.assertEqual(buf.data.tobytes(), b"hello")

        self._run_test_for_buffer_types(_test)

    def test_extend_fits_in_capacity(self):
        """Test extending when data fits within current capacity."""

        def _test(use_numpy):
            # Scenario 1: Fits exactly in remaining capacity (no prior data)
            buf = _Buffer(size=5, use_numpy=use_numpy)
            data_to_add = b"hello"
            buf.extend(data_to_add)
            self.assertEqual(len(buf), 5)
            self.assertEqual(buf.data.tobytes(), data_to_add)
            self.assertEqual(buf._current_pos, 5)
            self.assertEqual(buf._capacity, 5)  # Capacity remains initial size

            # Scenario 2: Fits with slack
            buf = _Buffer(size=10, use_numpy=use_numpy)
            buf.extend(b"abc")
            data_to_add_2 = b"de"
            buf.extend(data_to_add_2)
            self.assertEqual(len(buf), 5)
            self.assertEqual(buf.data.tobytes(), b"abcde")
            self.assertEqual(buf._current_pos, 5)
            self.assertEqual(buf._capacity, 10)  # Capacity remains initial size

        self._run_test_for_buffer_types(_test)

    def test_extend_exceeds_capacity(self):
        """Test extending when data exceeds current capacity."""

        def _test(use_numpy):
            # Scenario 1: Extending an empty buffer (size 0 initially)
            buf = _Buffer(use_numpy=use_numpy)
            data_to_add = b"longdata"
            buf.extend(data_to_add)
            self.assertEqual(len(buf), len(data_to_add))
            self.assertEqual(buf.data.tobytes(), data_to_add)
            self.assertEqual(buf._current_pos, len(data_to_add))
            self.assertEqual(buf._capacity, len(data_to_add))  # Capacity grows

            # Scenario 2: Extending a buffer that has some initial capacity and data
            buf = _Buffer(size=5, use_numpy=use_numpy)
            buf.extend(b"abc")  # current_pos=3, capacity=5
            data_to_add_2 = b"defgh"  # length 5. 2 fit in slack, 3 need growth
            buf.extend(data_to_add_2)
            expected_data = b"abcdefgh"
            self.assertEqual(len(buf), len(expected_data))
            self.assertEqual(buf.data.tobytes(), expected_data)
            self.assertEqual(buf._current_pos, len(expected_data))
            self.assertEqual(buf._capacity, len(expected_data))  # Capacity grows

            # Scenario 3: Extending a full buffer (current_pos == capacity)
            buf = _Buffer(size=3, use_numpy=use_numpy)
            buf.extend(b"xyz")  # current_pos=3, capacity=3
            data_to_add_3 = b"123"
            buf.extend(data_to_add_3)
            expected_data_3 = b"xyz123"
            self.assertEqual(len(buf), len(expected_data_3))
            self.assertEqual(buf.data.tobytes(), expected_data_3)
            self.assertEqual(buf._current_pos, len(expected_data_3))
            self.assertEqual(buf._capacity, len(expected_data_3))

        self._run_test_for_buffer_types(_test)

    def test_extend_with_various_input_types(self):
        """Test extending with bytes, bytearray, and memoryview inputs."""
        data_bytes = b"bytes"
        data_bytearray = bytearray(b"bytearray")
        data_memview = memoryview(b"memview")

        def _test_input_type(input_data, use_numpy):
            buf = _Buffer(use_numpy=use_numpy)
            buf.extend(input_data)
            self.assertEqual(len(buf), len(input_data))
            # For memoryview, tobytes() is needed for comparison if it's from bytes/bytearray
            if isinstance(input_data, bytes):
                expected_bytes = input_data
            elif isinstance(input_data, bytearray):
                expected_bytes = bytes(input_data)  # Convert bytearray to bytes
            else:  # memoryview
                expected_bytes = input_data.tobytes()
            self.assertEqual(buf.data.tobytes(), expected_bytes)

        for input_data_item in [data_bytes, data_bytearray, data_memview]:
            with self.subTest(input_type=type(input_data_item).__name__):
                self._run_test_for_buffer_types(
                    lambda use_numpy: _test_input_type(input_data_item, use_numpy)
                )

    def test_data_property(self):
        """Test the data property."""

        def _test(use_numpy):
            # Empty buffer
            buf = _Buffer(use_numpy=use_numpy)
            self.assertIsInstance(buf.data, memoryview)
            self.assertEqual(len(buf.data), 0)

            # Partially filled buffer
            buf = _Buffer(size=10, use_numpy=use_numpy)
            buf.extend(b"hello")
            self.assertIsInstance(buf.data, memoryview)
            self.assertEqual(buf.data.tobytes(), b"hello")
            self.assertEqual(len(buf.data), 5)  # Only used portion

            # Full buffer (current_pos == capacity)
            buf = _Buffer(size=5, use_numpy=use_numpy)
            buf.extend(b"world")
            self.assertIsInstance(buf.data, memoryview)
            self.assertEqual(buf.data.tobytes(), b"world")
            self.assertEqual(len(buf.data), 5)

        self._run_test_for_buffer_types(_test)

    def test_len_dunder(self):
        """Test the __len__ dunder method."""

        def _test(use_numpy):
            buf = _Buffer(use_numpy=use_numpy)
            self.assertEqual(len(buf), 0)

            buf.extend(b"abc")
            self.assertEqual(len(buf), 3)

            buf = _Buffer(size=10, use_numpy=use_numpy)
            self.assertEqual(len(buf), 0)
            buf.extend(b"hello")
            self.assertEqual(len(buf), 5)
            buf.extend(b"world")
            self.assertEqual(len(buf), 10)

        self._run_test_for_buffer_types(_test)

    def test_str_dunder(self):
        """Test the __str__ dunder method."""

        def _test(use_numpy):
            buf = _Buffer(size=10, use_numpy=use_numpy)
            buf.extend(b"data")
            expected_type = "ndarray" if use_numpy else "bytearray"
            self.assertEqual(
                str(buf),
                f"_Buffer(capacity=10, current_pos=4, type={expected_type})",  # Swapped order
            )

            buf_empty = _Buffer(use_numpy=use_numpy)
            self.assertEqual(
                str(buf_empty),
                f"_Buffer(capacity=0, current_pos=0, type={expected_type})",  # Swapped order
            )

        self._run_test_for_buffer_types(_test)

    def test_truncate_empty_buffer(self):
        """Test truncating an empty buffer."""

        def _test(use_numpy):
            buf = _Buffer(use_numpy=use_numpy)
            # Truncating an empty buffer to be empty should not fail
            buf.truncate(0, 0)
            self.assertEqual(len(buf), 0)
            self.assertEqual(buf._capacity, 0)

            # Other truncations on empty buffer should fail due to bounds
            with self.assertRaisesRegex(
                ValueError, "start index is out of bounds of buffer capacity."
            ):
                buf.truncate(1, 1)  # start=1 > capacity=0
            with self.assertRaisesRegex(
                ValueError, "end index is out of bounds of buffer capacity."
            ):
                buf.truncate(0, 1)  # end=1 > capacity=0

        self._run_test_for_buffer_types(_test)

    def test_truncate_to_empty(self):
        """Test truncating a buffer to become empty."""

        def _test(use_numpy):
            buf = _Buffer(size=10, use_numpy=use_numpy)
            buf.extend(b"hello")
            buf.truncate(0, 0)
            self.assertEqual(len(buf), 0)
            self.assertEqual(buf.data.tobytes(), b"")
            self.assertEqual(buf._current_pos, 0)
            self.assertEqual(buf._capacity, 0)  # Capacity shrinks

        self._run_test_for_buffer_types(_test)

    def test_truncate_from_start(self):
        """Test truncating data from the start of the buffer."""

        def _test(use_numpy):
            buf = _Buffer(size=10, use_numpy=use_numpy)
            original_data = b"abcdefghij"
            buf.extend(original_data)

            buf.truncate(start=3)  # Keep from index 3 to end
            expected_data = original_data[3:]
            self.assertEqual(len(buf), len(expected_data))
            self.assertEqual(buf.data.tobytes(), expected_data)
            self.assertEqual(buf._current_pos, len(expected_data))
            self.assertEqual(buf._capacity, len(expected_data))

        self._run_test_for_buffer_types(_test)

    def test_truncate_at_end(self):
        """Test truncating data at the end of the buffer."""

        def _test(use_numpy):
            buf = _Buffer(size=10, use_numpy=use_numpy)
            original_data = b"abcdefghij"
            buf.extend(original_data)

            buf.truncate(start=0, end=7)  # Keep from index 0 up to (not including) 7
            expected_data = original_data[:7]
            self.assertEqual(len(buf), len(expected_data))
            self.assertEqual(buf.data.tobytes(), expected_data)
            self.assertEqual(buf._current_pos, len(expected_data))
            self.assertEqual(buf._capacity, len(expected_data))

        self._run_test_for_buffer_types(_test)

    def test_truncate_middle_segment(self):
        """Test truncating to keep a middle segment of the buffer."""

        def _test(use_numpy):
            buf = _Buffer(size=10, use_numpy=use_numpy)
            original_data = b"abcdefghij"
            buf.extend(original_data)

            buf.truncate(start=2, end=8)  # Keep original_data[2:8]
            expected_data = original_data[2:8]
            self.assertEqual(len(buf), len(expected_data))
            self.assertEqual(buf.data.tobytes(), expected_data)
            self.assertEqual(buf._current_pos, len(expected_data))
            self.assertEqual(buf._capacity, len(expected_data))

        self._run_test_for_buffer_types(_test)

    def test_truncate_end_is_none(self):
        """Test truncate when end parameter is None."""

        def _test(use_numpy):
            buf = _Buffer(size=10, use_numpy=use_numpy)
            original_data = b"abcdefghij"
            buf.extend(original_data)

            buf.truncate(start=5, end=None)  # Same as truncate(start=5)
            expected_data = original_data[5:]
            self.assertEqual(len(buf), len(expected_data))
            self.assertEqual(buf.data.tobytes(), expected_data)
            # With new truncate: start=5, end=original_capacity (10)
            # new_capacity = 10 - 5 = 5
            # new_current_pos = max(0, min(10 - 5, 5)) = 5
            self.assertEqual(buf._current_pos, 5)
            self.assertEqual(buf._capacity, 5)

        self._run_test_for_buffer_types(_test)

    def test_truncate_full_segment(self):
        """Test truncating to keep the full current logical content."""

        def _test(use_numpy):
            buf = _Buffer(size=10, use_numpy=use_numpy)
            original_data = b"abcdef"  # len 6
            buf.extend(original_data)

            # Truncate to current logical content
            buf.truncate(start=0, end=len(buf))
            self.assertEqual(len(buf), len(original_data))
            self.assertEqual(buf.data.tobytes(), original_data)
            self.assertEqual(buf._current_pos, len(original_data))
            # Capacity becomes current_pos after truncate
            self.assertEqual(buf._capacity, len(original_data))

            # Truncate with end=None when buffer is not full capacity
            buf = _Buffer(size=20, use_numpy=use_numpy)
            buf.extend(original_data)  # len 6, capacity 20, current_pos 6
            # truncate(start=0, end=None) means start=0, end=20 (original capacity)
            buf.truncate(start=0, end=None)
            self.assertEqual(len(buf), len(original_data))  # current_pos should be 6
            self.assertEqual(buf.data.tobytes(), original_data)
            self.assertEqual(
                buf._current_pos, len(original_data)
            )  # current_pos = max(0, min(6-0, 20)) = 6
            # new capacity = end - start = 20 - 0 = 20
            self.assertEqual(buf._capacity, 20)

        self._run_test_for_buffer_types(_test)

    def test_truncate_invalid_args(self):
        """Test truncate with invalid start/end arguments."""

        def _test(use_numpy):
            buf = _Buffer(size=10, use_numpy=use_numpy)
            buf.extend(b"hello")  # len=5 (_current_pos=5), _capacity=10

            # Invalid start (negative)
            with self.assertRaisesRegex(ValueError, "start must be a non-negative integer."):
                buf.truncate(start=-1)

            # Valid start, but was previously considered invalid by some tests
            # start=6 is < _capacity=10. This is now a valid operation.
            # It should not raise "start index is out of bounds..."
            # Let's test the behavior of this valid truncation:
            buf_copy = _Buffer(size=10, use_numpy=use_numpy)
            buf_copy.extend(b"hello")
            buf_copy.truncate(start=6)  # end=None (becomes 10)
            # new_capacity = 10 - 6 = 4
            # new_current_pos = max(0, min(5 - 6, 4)) = max(0, min(-1, 4)) = 0
            self.assertEqual(len(buf_copy), 0)
            self.assertEqual(buf_copy._capacity, 4)
            self.assertEqual(buf_copy.data.tobytes(), b"")

            # Test start out of *capacity* bounds
            with self.assertRaisesRegex(
                ValueError, "start index is out of bounds of buffer capacity."
            ):
                buf.truncate(start=11)  # 11 > _capacity (10)

            # Invalid end (negative)
            with self.assertRaisesRegex(
                ValueError, "end must be a non-negative integer if provided."
            ):
                buf.truncate(start=0, end=-1)

            # Test end out of *capacity* bounds
            with self.assertRaisesRegex(
                ValueError, "end index is out of bounds of buffer capacity."
            ):
                buf.truncate(start=0, end=11)  # 11 > _capacity (10)

            # end < start
            with self.assertRaisesRegex(
                ValueError, "end index must be greater than or equal to start index."
            ):
                buf.truncate(start=3, end=2)

        self._run_test_for_buffer_types(_test)


if __name__ == "__main__":
    unittest.main()
