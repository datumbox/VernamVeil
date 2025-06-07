#include <stdlib.h>
#include <stddef.h>
#include "_kmp.h"

// Computes the Longest Proper Prefix which is also Suffix (LPS) array.
// This is a standard KMP preprocessing step.
// Marked static to limit visibility to this file.
static void compute_lps_array_kmp(const unsigned char *pattern, size_t m, size_t *lps) {
    size_t length = 0; // Length of the previous longest prefix suffix
    lps[0] = 0;       // lps[0] is always 0
    size_t i = 1;

    // The loop calculates lps[i] for i = 1 to m-1
    while (i < m) {
        if (pattern[i] == pattern[length]) {
            lps[i++] = ++length;
        } else {
            // This is tricky. Consider the example.
            // AAACAAAA and i = 7. The idea is similar to search step.
            if (length != 0) {
                length = lps[length - 1];
                // Also, note that we do not increment i here
            } else {
                lps[i++] = 0;
            }
        }
    }
}

// Searches for all occurrences of 'pattern' in 'text' using KMP.
// Returns a dynamically allocated array of indices, and sets count_ptr.
// Caller must free the returned array using free_indices_kmp.
size_t* find_all_kmp(const unsigned char *text, size_t n,
                                const unsigned char *pattern, size_t m,
                                size_t *count_ptr) {
    *count_ptr = 0;

    // Allocate LPS array
    size_t *lps = (size_t *)malloc(sizeof(size_t) * m);
    if (lps == NULL) {
        return NULL;
    }
    compute_lps_array_kmp(pattern, m, lps);

    size_t i = 0; // Index for text[]
    size_t j = 0; // Index for pattern[]

    // Initial allocation for indices. We can reallocate if more are found.
    size_t capacity = 1000;
    size_t *indices = (size_t *)malloc(sizeof(size_t) * capacity);
    if (indices == NULL) {
        free(lps);
        return NULL;
    }

    while (i < n) {
        if (pattern[j] == text[i]) {
            ++i;
            ++j;
        }

        if (j == m) {
            // Found pattern at index (i - j)
            if (*count_ptr >= capacity) {
                capacity *= 2;
                size_t *new_indices = (size_t *)realloc(indices, sizeof(size_t) * capacity);
                if (new_indices == NULL) {
                    free(lps);
                    free(indices);
                    *count_ptr = 0; // Indicate failure
                    return NULL;
                }
                indices = new_indices;
            }
            indices[*count_ptr] = i - j;
            (*count_ptr)++;
            j = lps[j - 1]; // Continue search for more occurrences
        } else if (i < n && pattern[j] != text[i]) {
            // Mismatch after j matches
            // Do not match lps[0..lps[j-1]] characters, they will match anyway
            if (j != 0) {
                j = lps[j - 1];
            } else {
                ++i;
            }
        }
    }

    free(lps);

    if (*count_ptr == 0) {
        free(indices);
        return NULL;
    }
    // Shrink to fit if desired, or leave as is for simplicity
    // size_t *final_indices = (size_t *)realloc(indices, sizeof(size_t) * (*count_ptr));
    // if (final_indices == NULL && *count_ptr > 0) { /* handle error, though unlikely if shrinking */ }
    // else if (*count_ptr > 0) { indices = final_indices; }

    return indices;
}

// Frees the array of indices allocated by find_all_kmp.
void free_indices_kmp(size_t *indices_ptr) {
    if (indices_ptr != NULL) {
        free(indices_ptr);
    }
}
