#include <string.h>

// MEMMEM availability check
#ifndef USE_MEMMEM
  #if defined(_GNU_SOURCE) && defined(__GLIBC__)
      // GNU/Linux systems
      #define USE_MEMMEM 1
  #elif defined(__has_include)
      // Compiler with __has_include support
      #if __has_include(<string.h>) && __has_builtin(memmem)
          #define USE_MEMMEM 1
      #else
          #define USE_MEMMEM 0
      #endif
  #else
      // Check for POSIX.1-2008 compliance which includes memmem
      #if (_POSIX_C_SOURCE >= 200809L) || defined(_POSIX_VERSION) && (_POSIX_VERSION >= 200809L)
          #define USE_MEMMEM 1
      #else
          #define USE_MEMMEM 0
      #endif
  #endif
#endif

#if USE_MEMMEM == 1
// This declaration ensures the compiler is aware of the correct signature for `memmem`
// (specifically, its `void*` return type) when `USE_MEMMEM` is active.
// Without this, if system headers fail to provide the prototype (e.g., due to
// missing feature-test macros like `_GNU_SOURCE`), the compiler might make an
// implicit declaration, typically assuming `memmem` returns an `int`. This can
// lead to incorrect behaviour, particularly on 64-bit systems where pointer
// sizes differ from `int`, as the returned pointer address could be truncated
// or misinterpreted.
void *memmem(const void *haystack, size_t haystacklen, const void *needle, size_t needlelen);
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "_bmh.h"
#include "_bytesearch.h"

// Searches for the first occurrence of 'pattern' in 'text'. Returns the index or -1 if not found.
ptrdiff_t find(const unsigned char * restrict text, size_t n, const unsigned char * restrict pattern, size_t m) {
    // We don't do the check here because on Python we handle it for all corner-cases.
    // if (m == 0 || n == 0 || m > n) return -1;
#if USE_MEMMEM == 1
    const unsigned char *found = memmem(text, n, pattern, m);
    if (!found) return -1;
    return (ptrdiff_t)(found - text);
#else
    bmh_prework_t p_bmh;
    bmh_preprocess(pattern, (ptrdiff_t)m, &p_bmh);
    return bmh_search(text, (ptrdiff_t)n, &p_bmh);
#endif
}

// Searches for all occurrences of 'pattern' in 'text'. It supports byte-like objects such as bytes, bytearray, and memoryview in Python.
// Returns a dynamically allocated array of indices, and sets count_ptr.
// Caller must free the returned array using the free_indices() function.
size_t* find_all(const unsigned char * restrict text, size_t n, const unsigned char * restrict pattern, size_t m, size_t * restrict count_ptr, int allow_overlap) {
    *count_ptr = 0;
    // Not necessary as we check on Python side.
    // if (m == 0 || n == 0 || m > n) return NULL;

#if USE_MEMMEM == 0
    // Preprocess pattern once for BMH
    bmh_prework_t p_bmh;
    bmh_preprocess(pattern, (ptrdiff_t)m, &p_bmh);
#endif

    size_t capacity = 512; // Initial allocation for indices. We reallocate if more are found.
    size_t *indices = malloc(sizeof(size_t) * capacity);
    if (indices == NULL) {
        return NULL;
    }

    size_t i = 0;
    while (i <= n - m) { // Ensure there's enough space for the pattern
#if USE_MEMMEM == 1
        const unsigned char * restrict found = memmem(text + i, n - i, pattern, m);
        if (!found) {
            break;
        }
        size_t match_idx = (size_t)(found - text);
#else
        ptrdiff_t found = bmh_search(text + i, (ptrdiff_t)(n - i), &p_bmh);
        if (found == -1) {
            break;
        }
        size_t match_idx = i + (size_t)found;
#endif
        if (*count_ptr >= capacity) {
            capacity *= 2;
            size_t *new_indices = realloc(indices, sizeof(size_t) * capacity);
            if (new_indices == NULL) {
                free(indices);
                *count_ptr = 0;
                return NULL;
            }
            indices = new_indices;
        }
        indices[(*count_ptr)++] = match_idx;

        if (allow_overlap) {
            i = match_idx + 1;
        } else {
            i = match_idx + m;
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
