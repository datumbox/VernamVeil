import unittest
from unittest.mock import patch

from vernamveil._bytesearch import find_all
from vernamveil._types import _HAS_C_MODULE


class TestFindAll(unittest.TestCase):
    """Unit tests for the find_all function in _find.py."""

    def _get_checks(self):
        return [False, True]

    def _run_find_all(self, haystack, needle, has_c_module):
        """Utility to run find_all with or without C module."""
        with (
            patch("vernamveil._types._HAS_C_MODULE", has_c_module),
            patch("vernamveil._find._HAS_C_MODULE", has_c_module),
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


if __name__ == "__main__":
    unittest.main()
