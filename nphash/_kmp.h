#ifndef NPKMP_H
#define NPKMP_H

#include <stddef.h> // For size_t

// Searches for all occurrences of 'pattern' in 'text' using Knuth-Morris-Pratt.
// Parameters:
//   text: The text to search within.
//   n: The length of the text.
//   pattern: The pattern to search for.
//   m: The length of the pattern.
//   count_ptr: Out-parameter, will be filled with the number of occurrences found.
// Returns:
//   A dynamically allocated array of integers containing the 0-based starting
//   indices of all occurrences. The caller is responsible for freeing this array
//   using free_indices_kmp. Returns NULL if no occurrences are found,
//   if m > n, or if memory allocation fails.
size_t* find_all_kmp(const unsigned char *text, size_t n,
                                   const unsigned char *pattern, size_t m,
                                   size_t *count_ptr);

// Frees the array of indices allocated by find_all_kmp.
void free_indices_kmp(size_t *indices_ptr);

#endif // NPKMP_H
