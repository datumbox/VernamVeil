import unittest
from unittest.mock import patch

from vernamveil._bytesearch import find, find_all
from vernamveil._types import _HAS_C_MODULE, _HAS_NUMPY, np


class TestByteSearch(unittest.TestCase):
    """Unit tests for the bytesearch methods in _find.py."""

    def _get_checks(self):
        """Utility to get the checks for _HAS_C_MODULE."""
        checks = [False]
        if _HAS_C_MODULE:
            checks.append(True)
        return checks

    def _run_find(self, haystack, needle, has_c_module, start=0, end=None):
        """Utility to run find with or without C module."""
        with (
            patch("vernamveil._types._HAS_C_MODULE", has_c_module),
            patch("vernamveil._bytesearch._HAS_C_MODULE", has_c_module),
        ):
            return find(haystack, needle, start, end)

    def _run_find_all(self, haystack, needle, has_c_module):
        """Utility to run find_all with or without C module."""
        with (
            patch("vernamveil._types._HAS_C_MODULE", has_c_module),
            patch("vernamveil._bytesearch._HAS_C_MODULE", has_c_module),
        ):
            return find_all(haystack, needle)

    def test_find_all_empty_needle(self):
        """Should return empty list if needle is empty."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"abc", b"", has_c), [])

    def test_find_all_empty_haystack(self):
        """Should return empty list if haystack is empty."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"", b"abc", has_c), [])

    def test_find_all_needle_longer_than_haystack(self):
        """Should return empty list if needle is longer than haystack."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"ab", b"abc", has_c), [])

    def test_find_all_no_occurrences(self):
        """Should return empty list if needle is not found."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"abcdef", b"gh", has_c), [])

    def test_find_all_single_occurrence(self):
        """Should return correct index for a single occurrence."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"abcdef", b"cd", has_c), [2])

    def test_find_all_multiple_occurrences(self):
        """Should return all indices for multiple non-overlapping occurrences."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"abcabcabc", b"abc", has_c), [0, 3, 6])

    def test_find_all_overlapping_occurrences(self):
        """Should not return overlapping matches (Python's find behavior)."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"aaaa", b"aa", has_c), [0, 2])

    def test_find_all_memoryview_and_bytearray(self):
        """Should work with memoryview and bytearray types for both haystack and needle."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(memoryview(b"abcabc"), b"abc", has_c), [0, 3])
                self.assertEqual(self._run_find_all(b"abcabc", memoryview(b"abc"), has_c), [0, 3])
                self.assertEqual(
                    self._run_find_all(bytearray(b"abcabc"), bytearray(b"abc"), has_c), [0, 3]
                )

    def test_find_all_full_haystack_match(self):
        """Should return [0] if needle matches the entire haystack."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"abc", b"abc", has_c), [0])

    def test_find_all_single_byte_needle(self):
        """Should find all occurrences of a single byte needle."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"banana", b"a", has_c), [1, 3, 5])

    def test_find_all_large_input(self):
        """Should work correctly for large haystack and needle."""
        haystack = b"ab" * 1000 + b"cd" + b"ab" * 1000 + b"cd"
        needle = b"cd"
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(haystack, needle, has_c), [2000, 4002])

    def test_find_all_numpy_memoryview_support(self):
        """Should support numpy arrays wrapped with memoryview for search if _HAS_NUMPY is True."""
        if not _HAS_NUMPY:
            self.skipTest("Numpy not available")
        arr_haystack = np.array([97, 98, 99, 97, 98, 99], dtype=np.uint8)  # b"abcabc"
        arr_needle = np.array([97, 98, 99], dtype=np.uint8)  # b"abc"
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(arr_haystack.data, b"abc", has_c), [0, 3])
                self.assertEqual(self._run_find_all(b"abcabc", arr_needle.data, has_c), [0, 3])
                self.assertEqual(
                    self._run_find_all(arr_haystack.data, arr_needle.data, has_c), [0, 3]
                )

    def test_find_empty_needle(self):
        """Should handle empty needle correctly."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"abc", b"", has_c), 0)
                self.assertEqual(self._run_find(b"abc", b"", has_c, start=1), 1)
                self.assertEqual(self._run_find(b"abc", b"", has_c, start=4), -1)
                self.assertEqual(self._run_find(b"abc", b"", has_c, start=-1), 2)
                self.assertEqual(self._run_find(b"abc", b"", has_c, start=-4), 0)

    def test_find_empty_haystack(self):
        """Should return -1 if haystack is empty."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"", b"abc", has_c), -1)

    def test_find_needle_longer_than_haystack(self):
        """Should return -1 if needle is longer than haystack."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"ab", b"abc", has_c), -1)

    def test_find_no_occurrences(self):
        """Should return -1 if needle is not found in haystack."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"abcdef", b"gh", has_c), -1)

    def test_find_single_occurrence(self):
        """Should return the index of the first occurrence of the needle."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"abcdef", b"cd", has_c), 2)

    def test_find_multiple_occurrences(self):
        """Should return the index of the first occurrence in case of multiple non-overlapping occurrences."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c), 0)

    def test_find_overlapping_occurrences(self):
        """Should return the index of the first occurrence in case of overlapping matches."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"aaaa", b"aa", has_c), 0)

    def test_find_memoryview_and_bytearray(self):
        """Should work with memoryview and bytearray types for both haystack and needle."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(memoryview(b"abcabc"), b"abc", has_c), 0)
                self.assertEqual(self._run_find(b"abcabc", memoryview(b"abc"), has_c), 0)
                self.assertEqual(self._run_find(bytearray(b"abcabc"), bytearray(b"abc"), has_c), 0)

    def test_find_full_haystack_match(self):
        """Should return 0 if needle matches the entire haystack."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"abc", b"abc", has_c), 0)

    def test_find_single_byte_needle(self):
        """Should find the first occurrence of a single byte needle."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"banana", b"a", has_c), 1)

    def test_find_large_input(self):
        """Should return the index of the first occurrence in a large haystack."""
        haystack = b"ab" * 1000 + b"cd" + b"ab" * 1000 + b"cd"
        needle = b"cd"
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(haystack, needle, has_c), 2000)

    def test_find_numpy_memoryview_support(self):
        """Should support numpy arrays wrapped with memoryview for search if _HAS_NUMPY is True."""
        if not _HAS_NUMPY:
            self.skipTest("Numpy not available")
        arr_haystack = np.array([97, 98, 99, 97, 98, 99], dtype=np.uint8)  # b"abcabc"
        arr_needle = np.array([97, 98, 99], dtype=np.uint8)  # b"abc"
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(arr_haystack.data, b"abc", has_c), 0)
                self.assertEqual(self._run_find(b"abcabc", arr_needle.data, has_c), 0)
                self.assertEqual(self._run_find(arr_haystack.data, arr_needle.data, has_c), 0)

    # --- Start/end parameter tests ---
    def test_find_start_parameter(self):
        """Should return the index of the first occurrence starting from the specified index."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=1), 3)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=4), 6)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=7), -1)

    def test_find_end_parameter(self):
        """Should return the index of the first occurrence up to the specified end index."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, end=6), 0)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, end=3), 0)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, end=2), -1)

    def test_find_start_and_end_parameters(self):
        """Should return the index of the first occurrence within the specified start and end indices."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=3, end=8), 3)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=4, end=8), -1)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=7, end=8), -1)

    def test_find_negative_start_end(self):
        """Should handle negative start and end indices correctly."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=-6), 3)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=-3), 6)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, end=-3), 0)
                self.assertEqual(self._run_find(b"abcabcabc", b"cab", has_c, start=-4, end=-1), 5)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=-2, end=-1), -1)

    def test_find_start_greater_than_end(self):
        """Should return -1 if start index is greater than end index."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=5, end=2), -1)

    def test_find_start_end_out_of_bounds(self):
        """Should handle out-of-bounds start and end indices correctly."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=100), -1)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, end=100), 0)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=3, end=100), 3)
                self.assertEqual(self._run_find(b"abcabcabc", b"abc", has_c, start=-100, end=2), -1)
                self.assertEqual(
                    self._run_find(b"abcabcabc", b"abc", has_c, start=-100, end=100), 0
                )


if __name__ == "__main__":
    unittest.main()
