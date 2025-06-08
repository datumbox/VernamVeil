#include <string.h>

// MEMMEM availability check
#if defined(_GNU_SOURCE) && defined(__GLIBC__)
    // GNU/Linux systems
    #define HAVE_MEMMEM 1
#elif defined(__APPLE__)
    // Apple/macOS systems
    #include <AvailabilityMacros.h>
    #if defined(MAC_OS_X_VERSION_10_7) && MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_7
        // macOS has memmem since 10.7
        #define HAVE_MEMMEM 1
    #else
        #define HAVE_MEMMEM 0
    #endif
#elif defined(__has_include)
    // Compiler with __has_include support
    #if __has_include(<string.h>) && __has_builtin(memmem)
        #define HAVE_MEMMEM 1
    #else
        #define HAVE_MEMMEM 0
    #endif
#else
    // Check for POSIX.1-2008 compliance which includes memmem
    #if (_POSIX_C_SOURCE >= 200809L) || defined(_POSIX_VERSION) && (_POSIX_VERSION >= 200809L)
        #define HAVE_MEMMEM 1
    #else
        #define HAVE_MEMMEM 0
    #endif
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include "_bytesearch.h"

#undef HAVE_MEMMEM
#define HAVE_MEMMEM 0 //TODO: Force disable memmem for this implementation

#define ALPHABET_SIZE 256

// Structure for BMH preprocessing data
typedef struct {
    ptrdiff_t skip_table[ALPHABET_SIZE];
    const unsigned char *pattern_ptr;
    size_t pattern_len;
} bmh_prework_t;

// Preprocesses the pattern for BMH search.
static void bmh_preprocess(const unsigned char *pattern, size_t m, bmh_prework_t *p) {
    // m is 0 is handled by Python side. it is assumed that the caller ensures m >= 0.
    p->pattern_ptr = pattern;
    p->pattern_len = m;

    // Initialize skip table: default shift is pattern length m
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        p->skip_table[i] = (ptrdiff_t)m;
    }
    // For characters in pattern (except the last one), set specific shifts
    // The shift is m - 1 - k, where k is the index of the character pattern[k]
    const size_t m_minus_1 = m - 1;
    for (size_t k = 0; k < m_minus_1; ++k) {
        p->skip_table[pattern[k]] = (ptrdiff_t)(m_minus_1 - k);
    }
}

// Searches for pattern in text using BMH algorithm with precomputed data.
static ptrdiff_t bmh_search(const unsigned char *text, size_t n, const bmh_prework_t *p) {
    const unsigned char *pattern = p->pattern_ptr;
    size_t m = p->pattern_len;

    // n is 0 is handled by Python side. it is assumed that the caller ensures n >= 0.
    // if (m == 0) return 0; // Python side handles m=0, bmh_preprocess also handles m=0 for skip_table.
    // if (m > n) return -1; // Python side handles m > n.

    // Pre-calculate values that are constant within the loop
    const size_t n_minus_m = n - m;
    const ptrdiff_t m_minus_1 = (ptrdiff_t)m - 1;

    size_t i = 0; // Current alignment of pattern's start in text

    while (i <= n_minus_m) {
        ptrdiff_t k = m_minus_1; // Index for pattern (from m-1 down to 0)
        while (k >= 0 && pattern[k] == text[i + k]) {
            --k;
        }

        if (k < 0) {
            return (ptrdiff_t)i; // Match found at text[i]
        } else {
            // Mismatch. Calculate shift using the character in text aligned with the *last* char of pattern.
            // This is text[i + m - 1].
            unsigned char char_to_skip_by = text[i + m_minus_1];
            i += p->skip_table[char_to_skip_by];
        }
    }
    return -1; // No match found
}


// Searches for the first occurrence of 'pattern' in 'text'. Returns the index or -1 if not found.
ptrdiff_t find(const unsigned char * restrict text, size_t n, const unsigned char * restrict pattern, size_t m) {
    // We don't do the check here because on Python we handle it for all corner-cases.
    // if (m == 0 || n == 0 || m > n) return -1;
#if HAVE_MEMMEM == 1
    const unsigned char *found = memmem(text, n, pattern, m);
    if (!found) return -1;
    return (ptrdiff_t)(found - text);
#else
    bmh_prework_t p_bmh;
    bmh_preprocess(pattern, m, &p_bmh);
    ptrdiff_t found = bmh_search(text, n, &p_bmh);
    return found;
#endif
}

// Searches for all occurrences of 'pattern' in 'text'. It supports byte-like objects such as bytes, bytearray, and memoryview in Python.
// Returns a dynamically allocated array of indices, and sets count_ptr.
// Caller must free the returned array using free_indices.
size_t* find_all(const unsigned char * restrict text, size_t n, const unsigned char * restrict pattern, size_t m, size_t * restrict count_ptr, int allow_overlap) {
    *count_ptr = 0;
    // Not necessary as we check on Python side.
    // if (m == 0 || n == 0 || m > n) return NULL;

#if HAVE_MEMMEM == 0
    // Preprocess pattern once for BMH
    bmh_prework_t p_bmh;
    bmh_preprocess(pattern, m, &p_bmh);
#endif

    size_t capacity = 1000; // Initial allocation for indices. We can reallocate if more are found.
    size_t *indices = malloc(sizeof(size_t) * capacity);
    if (indices == NULL) {
        return NULL;
    }

    size_t i = 0;
    while (i <= n - m) { // Ensure there's enough space for the pattern
#if HAVE_MEMMEM == 1
        const unsigned char *found = memmem(text + i, n - i, pattern, m);
        if (!found) {
            break;
        }
        size_t match_idx = (size_t)(found - text);
#else
        ptrdiff_t found = bmh_search(text + i, n - i, &p_bmh);
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
        indices[*count_ptr] = match_idx;
        ++(*count_ptr);

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
