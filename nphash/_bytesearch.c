#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include "_bytesearch.h"
#include "_twoway.h"

// Searches for all occurrences of 'pattern' in 'text'. It supports byte-like objects such as bytes, bytearray, and memoryview in Python.
// Returns a dynamically allocated array of indices, and sets count_ptr.
// Caller must free the returned array using free_indices.
size_t* find_all(const unsigned char *text, size_t n, const unsigned char *pattern, size_t m, size_t *count_ptr, int allow_overlap) {
    *count_ptr = 0;
    // Not necessary as we check on Python side.
    // if (m == 0 || n == 0 || m > n) return NULL;

    // Pass 'm' (size_t) as ptrdiff_t.
    prework p;
    preprocess(pattern, (ptrdiff_t)m, &p);

    size_t capacity = 1000; // Initial allocation for indices. We can reallocate if more are found.
    size_t *indices = (size_t *)malloc(sizeof(size_t) * capacity);
    if (indices == NULL) {
        return NULL;
    }

    size_t i = 0;
    while (i <= n - m) { // Ensure there's enough space for the pattern
        // Pass 'n-i' (size_t) as ptrdiff_t.
        ptrdiff_t result = two_way(text + i, (ptrdiff_t)(n - i), &p);
        if (result == -1) {
            break;
        }
        size_t found = i + (size_t)result;
        if (*count_ptr >= capacity) {
            capacity *= 2;
            size_t *new_indices = (size_t *)realloc(indices, sizeof(size_t) * capacity);
            if (new_indices == NULL) {
                free(indices);
                *count_ptr = 0;
                return NULL;
            }
            indices = new_indices;
        }
        indices[*count_ptr] = found;
        (*count_ptr)++;

        if (allow_overlap) {
            // If m is 0, found + 1 could still lead to issues if not handled above.
            // But we return NULL for m=0 now.
            i = found + 1;
        } else {
            // If m is 0, found + m is found. Infinite loop if not handled above.
            i = found + m;
        }
    }

    if (*count_ptr == 0) {
        free(indices);
        return NULL;
    }

    // Shrink to fit, optional because Python side handles this.
    // size_t *final_indices = (size_t *)realloc(indices, sizeof(size_t) * (*count_ptr));
    // if (final_indices == NULL && *count_ptr > 0) { /* handle error */ }
    // else if (*count_ptr > 0) { indices = final_indices; }

    return indices;
}

// Frees the array of indices allocated by find_all.
void free_indices(size_t *indices_ptr) {
    if (indices_ptr != NULL) {
        free(indices_ptr);
    }
}
