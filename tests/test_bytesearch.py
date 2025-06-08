import unittest
from unittest.mock import patch

from vernamveil._bytesearch import find_all
from vernamveil._types import _HAS_C_MODULE
from vernamveil._types import _HAS_NUMPY, np


class TestFindAll(unittest.TestCase):
    """Unit tests for the find_all function in _find.py."""

    def _get_checks(self):
        """Utility to get the checks for _HAS_C_MODULE."""
        checks = [False]
        if _HAS_C_MODULE:
            checks.append(True)
        return checks

    def _run_find_all(self, haystack, needle, has_c_module):
        """Utility to run find_all with or without C module."""
        with (
            patch("vernamveil._types._HAS_C_MODULE", has_c_module),
            patch("vernamveil._bytesearch._HAS_C_MODULE", has_c_module),
        ):
            return find_all(haystack, needle)

    def test_empty_needle(self):
        """Should return empty list if needle is empty."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"abc", b"", has_c), [])

    def test_empty_haystack(self):
        """Should return empty list if haystack is empty."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"", b"abc", has_c), [])

    def test_needle_longer_than_haystack(self):
        """Should return empty list if needle is longer than haystack."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"ab", b"abc", has_c), [])

    def test_no_occurrences(self):
        """Should return empty list if needle is not found."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"abcdef", b"gh", has_c), [])

    def test_single_occurrence(self):
        """Should return correct index for a single occurrence."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"abcdef", b"cd", has_c), [2])

    def test_multiple_occurrences(self):
        """Should return all indices for multiple non-overlapping occurrences."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"abcabcabc", b"abc", has_c), [0, 3, 6])

    def test_overlapping_occurrences(self):
        """Should not return overlapping matches (Python's find behavior)."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"aaaa", b"aa", has_c), [0, 2])

    def test_memoryview_and_bytearray(self):
        """Should work with memoryview and bytearray types for both haystack and needle."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(memoryview(b"abcabc"), b"abc", has_c), [0, 3])
                self.assertEqual(self._run_find_all(b"abcabc", memoryview(b"abc"), has_c), [0, 3])
                self.assertEqual(
                    self._run_find_all(bytearray(b"abcabc"), bytearray(b"abc"), has_c), [0, 3]
                )

    def test_full_haystack_match(self):
        """Should return [0] if needle matches the entire haystack."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"abc", b"abc", has_c), [0])

    def test_single_byte_needle(self):
        """Should find all occurrences of a single byte needle."""
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(b"banana", b"a", has_c), [1, 3, 5])

    def test_large_input(self):
        """Should work correctly for large haystack and needle."""
        haystack = b"ab" * 1000 + b"cd" + b"ab" * 1000 + b"cd"
        needle = b"cd"
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(self._run_find_all(haystack, needle, has_c), [2000, 4002])

    def test_numpy_memoryview_support(self):
        """Should support numpy arrays wrapped with memoryview for search if _HAS_NUMPY is True."""
        if not _HAS_NUMPY:
            self.skipTest("Numpy not available")
        arr_haystack = np.array([97, 98, 99, 97, 98, 99], dtype=np.uint8)  # b"abcabc"
        arr_needle = np.array([97, 98, 99], dtype=np.uint8)  # b"abc"
        for has_c in self._get_checks():
            with self.subTest(_HAS_C_MODULE=has_c):
                self.assertEqual(
                    self._run_find_all(arr_haystack.data, b"abc", has_c), [0, 3]
                )
                self.assertEqual(
                    self._run_find_all(b"abcabc", arr_needle.data, has_c), [0, 3]
                )
                self.assertEqual(
                    self._run_find_all(arr_haystack.data, arr_needle.data, has_c), [0, 3]
                )


if __name__ == "__main__":
    unittest.main()
