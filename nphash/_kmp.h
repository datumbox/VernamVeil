#ifndef NPKMP_H
#define NPKMP_H

#include <stddef.h>

// Searches for all occurrences of 'pattern' in 'text' using Knuth-Morris-Pratt.
// Parameters:
//   text: The text to search within.
//   n: The length of the text.
//   pattern: The pattern to search for.
//   m: The length of the pattern.
//   count_ptr: Out-parameter, will be filled with the number of occurrences found.
//   allow_overlap: If non-zero, allows overlapping occurrences of the pattern.
// Returns:
//   A dynamically allocated array of integers containing the 0-based starting
//   indices of all occurrences. The caller is responsible for freeing this array
//   using free_indices. Returns NULL if no occurrences are found,
//   if m > n, or if memory allocation fails.
size_t* find_all(const unsigned char *text, size_t n,
                                   const unsigned char *pattern, size_t m,
                                   size_t *count_ptr,
                                   int allow_overlap);

// Frees the array of indices allocated by find_all.
void free_indices(size_t *indices_ptr);

#endif // NPKMP_H
