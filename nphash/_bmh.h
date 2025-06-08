#ifndef NPHASH_BMH_H
#define NPHASH_BMH_H

#include <stddef.h> // For size_t, ptrdiff_t
#include <stdint.h> // For standard integer types (though ptrdiff_t is usually in stddef.h)
#include <stdbool.h> // For bool

#define ALPHABET_SIZE 256
#define BMH_GOOD_SUFFIX_STACK_MAX_M 128

// Structure for BMH preprocessing data
typedef struct {
    ptrdiff_t skip_table[ALPHABET_SIZE];
    ptrdiff_t good_suffix_shifts_arr[BMH_GOOD_SUFFIX_STACK_MAX_M + 1];
    ptrdiff_t *good_suffix_shifts; // Points to good_suffix_shifts_arr or is NULL
    const unsigned char *pattern_ptr;
    size_t pattern_len;
} bmh_prework_t;

// Preprocesses the pattern for BMH search.
static inline void bmh_preprocess(const unsigned char * restrict pattern, size_t m, bmh_prework_t * restrict p) {
    // m is 0 is handled by Python side. it is assumed that the caller ensures m >= 0.
    p->pattern_ptr = pattern;
    p->pattern_len = m;

    // Initialize skip table: default shift is pattern length m
    const ptrdiff_t m_ptrdiff_t = (ptrdiff_t)m;
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        p->skip_table[i] = m_ptrdiff_t;
    }

    // For characters in pattern (except the last one), set specific shifts
    // The shift is m - 1 - k, where k is the index of the character pattern[k]
    const size_t m_minus_1 = m - 1;
    for (size_t k = 0; k < m_minus_1; ++k) {
        p->skip_table[pattern[k]] = (ptrdiff_t)(m_minus_1 - k);
    }

    // Initialize: GS rule off by default or if m is too large
    p->good_suffix_shifts = NULL;
    if (m > BMH_GOOD_SUFFIX_STACK_MAX_M) {
        return; // GS rule not used if m exceeds stack allocation limit
    }

    // Good Suffix Rule preprocessing can use stack arrays.
    p->good_suffix_shifts = p->good_suffix_shifts_arr; // Point to the stack array in the struct
    ptrdiff_t *gs = p->good_suffix_shifts;
    for (size_t i = 0; i <= m; ++i) {
        gs[i] = m_ptrdiff_t; // Default shift is pattern length
    }

    ptrdiff_t f_arr[BMH_GOOD_SUFFIX_STACK_MAX_M + 1]; // fixed-size array for MSVC compatibility
    ptrdiff_t *f = f_arr;   // Use f as a pointer to the stack array f_arr

    // Phase 1 (compute f array - KMP-like borders for reversed pattern)
    // f[i] stores the starting position of the widest border of pattern[i..m-1]
    ptrdiff_t j_f = m_ptrdiff_t + 1;
    f[m] = j_f;
    for (ptrdiff_t i_f = m_ptrdiff_t - 1; i_f >= 0; --i_f) {
        while (0 < j_f && j_f <= m_ptrdiff_t && pattern[i_f] != pattern[j_f - 1]) {
            if (gs[j_f] == m_ptrdiff_t) {
                gs[j_f] = j_f - 1 - i_f;
            }
            j_f = f[j_f];
        }
        f[i_f] = --j_f;
    }

    // Phase 2 (populate remaining gs entries)
    for (ptrdiff_t k_gs = 0; k_gs <= m_ptrdiff_t; ++k_gs) {
        if (gs[k_gs] == m_ptrdiff_t) {
            gs[k_gs] = j_f;
        }
        if (k_gs == j_f) {
            j_f = f[j_f];
        }
    }
}

// Searches for pattern in text using Boyer-Moore-Horspool algorithm with precomputed data.
static inline ptrdiff_t bmh_search(const unsigned char * restrict text, size_t n, const bmh_prework_t * restrict p) {
    // if (n == 0) return -1; // Python side handles n >= 0.
    // if (m == 0) return 0; // Python side handles m=0, bmh_preprocess also handles m=0 for skip_table.
    // if (m > n) return -1; // Python side handles m > n.

    // Pre-calculate values that are constant within the loop
    const unsigned char * restrict pattern = p->pattern_ptr;
    const size_t m = p->pattern_len;
    const size_t n_minus_m = n - m;
    const ptrdiff_t m_minus_1 = (ptrdiff_t)m - 1;
    const bool has_good_suffix = (p->good_suffix_shifts != NULL);

    size_t i = 0; // Current alignment of pattern's start in text
    while (i <= n_minus_m) {
        ptrdiff_t k = m_minus_1; // Index for pattern (from m-1 down to 0)
        while (k >= 0 && pattern[k] == text[i + k]) {
            --k;
        }

        if (k < 0) {
            return (ptrdiff_t)i; // Match found at text[i]
        } else {
            ptrdiff_t bad_char_shift = p->skip_table[text[i + m_minus_1]];

            // k is the index of mismatch in pattern P[0...m-1]
            // Good suffix is P[k+1 ... m-1]
            // gs table is indexed by the start of the good suffix (k+1 here)
            // if no good suffix is precomputed, we use a default shift of 1
            ptrdiff_t good_suffix_shift = has_good_suffix ? p->good_suffix_shifts[k + 1] : 1;

            // Take the maximum of the two shifts
            i += (bad_char_shift > good_suffix_shift) ? bad_char_shift : good_suffix_shift;
        }
    }
    return -1; // No match found
}

#endif // NPHASH_BMH_H

