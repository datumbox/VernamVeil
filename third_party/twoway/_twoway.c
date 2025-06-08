/*
 * This file is derived from Python's fastsearch.h implementation:
 * https://github.com/python/cpython/blob/8fdbbf8b18f1405abe677d0e04874c1c86ccdb4a/Objects/stringlib/fastsearch.h#L370
 *
 * Copyright (c) 2001-2023 Python Software Foundation.
 * All Rights Reserved.
 *
 * Licensed under the PSF License Agreement; you may not use this file except
 * in compliance with the License.
 *
 * See the Python Software Foundation License Version 2 at:
 * https://github.com/python/cpython/blob/main/LICENSE
 */

#include "_twoway.h"
#include <string.h>
#include <stddef.h>

// Helper macros for MIN and MAX
#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif
#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

static inline ptrdiff_t lex_search(const unsigned char *needle, ptrdiff_t len_needle, ptrdiff_t *return_period, int invert_alphabet) {
    ptrdiff_t max_suffix = 0;
    ptrdiff_t candidate = 1;
    ptrdiff_t k = 0;
    ptrdiff_t period = 1;
    while (candidate + k < len_needle) {
        unsigned char a = needle[candidate + k];
        unsigned char b = needle[max_suffix + k];
        if (invert_alphabet ? (b < a) : (a < b)) {
            candidate += k + 1;
            k = 0;
            period = candidate - max_suffix;
        }
        else if (a == b) {
            if (k + 1 != period) {
                ++k;
            }
            else {
                candidate += period;
                k = 0;
            }
        }
        else {
            max_suffix = candidate;
            ++candidate;
            k = 0;
            period = 1;
        }
    }
    *return_period = period;
    return max_suffix;
}

static inline ptrdiff_t factorize(const unsigned char *needle, ptrdiff_t len_needle, ptrdiff_t *return_period) {
    ptrdiff_t cut1, period1, cut2, period2, cut, period;
    cut1 = lex_search(needle, len_needle, &period1, 0);
    cut2 = lex_search(needle, len_needle, &period2, 1);
    if (cut1 > cut2) {
        period = period1;
        cut = cut1;
    }
    else {
        period = period2;
        cut = cut2;
    }
    *return_period = period;
    return cut;
}

void preprocess(const unsigned char *needle, ptrdiff_t len_needle, prework *p) {
    p->needle = needle;
    p->len_needle = len_needle;
    p->cut = factorize(needle, len_needle, &(p->period));
    // Ensure that period + cut is within bounds before memcmp
    if (p->period + p->cut > len_needle && len_needle > 0) { // Added len_needle > 0 to prevent issues with empty needles if factorize behaves unexpectedly
         // This case should ideally not be hit if factorize works as expected
         // and needle is not empty. If it is, it implies an issue with factorization
         // or that the needle is too short for this factorization logic.
         // For very short needles (e.g. len_needle=1), cut might be 0 or 1, period might be 1.
         // If len_needle=1, cut=0, period=1. p->period + p->cut = 1. needle+p->period is needle+1.
         // memcmp(needle, needle+1, 0) is okay.
         // If len_needle=1, cut=1, period=1. p->period + p->cut = 2. This is > len_needle.
         // The assert(p->period + p->cut <= len_needle) would fire.
         // For safety, if this condition is met, we can assume not periodic.
         p->is_periodic = 0;
    } else if (len_needle == 0) { // Handle empty needle explicitly
        p->is_periodic = 0; // Or handle as an error/edge case as appropriate
    }
    else {
        // assert(p->period + p->cut <= len_needle);
        p->is_periodic = (0 == memcmp(needle, needle + p->period, p->cut));
    }

    if (p->is_periodic) {
        // assert(p->cut <= len_needle/2);
        // assert(p->cut < p->period);
    }
    else {
        p->period = MAX(p->cut, len_needle - p->cut) + 1;
    }
    p->gap = len_needle;
    if (len_needle > 0) { // Check to prevent reading needle[-1] for empty needle
        unsigned char last = needle[len_needle - 1] & TABLE_MASK;
        for (ptrdiff_t i = len_needle - 2; i >= 0; --i) {
            if ((needle[i] & TABLE_MASK) == last) {
                p->gap = len_needle - 1 - i;
                break;
            }
        }
    }

    ptrdiff_t not_found_shift = MIN(len_needle, (ptrdiff_t)MAX_SHIFT);
    for (ptrdiff_t i = 0; i < (ptrdiff_t)TABLE_SIZE; ++i) {
        p->table[i] = (SHIFT_TYPE)(not_found_shift);
    }
    // Ensure len_needle is positive before this loop to avoid issues with len_needle - not_found_shift
    if (len_needle > 0) {
        for (ptrdiff_t i = len_needle - not_found_shift; i < len_needle; ++i) {
            SHIFT_TYPE shift = (SHIFT_TYPE)(len_needle - 1 - i);
            p->table[needle[i] & TABLE_MASK] = shift;
        }
    }
}

ptrdiff_t two_way(const unsigned char *haystack, ptrdiff_t len_haystack, prework *p) {
    const ptrdiff_t len_needle = p->len_needle;

    // If needle is empty, it's found at the beginning of any haystack
    // Or, if this is an error condition, handle accordingly.
    // Python's find behavior: "" in "abc" is 0. "" in "" is 0.
    if (len_needle == 0) {
        return 0;
    }
    // If haystack is shorter than needle, no match is possible
    if (len_haystack < len_needle) {
        return -1;
    }

    const ptrdiff_t cut = p->cut;
    ptrdiff_t period = p->period;
    const unsigned char *const needle = p->needle;
    const unsigned char *window_last = haystack + len_needle - 1;
    const unsigned char *const haystack_end = haystack + len_haystack;
    const SHIFT_TYPE *table = p->table; // Changed to const SHIFT_TYPE *
    const unsigned char *window;
    const ptrdiff_t gap = p->gap; // Added const
    const ptrdiff_t gap_jump_end = MIN(len_needle, cut + gap); // Added const


    if (p->is_periodic) {
        ptrdiff_t memory = 0;
      periodicwindowloop:
        while (window_last < haystack_end) {
            for (;;) {
                ptrdiff_t shift = table[(*window_last) & TABLE_MASK];
                window_last += shift;
                if (shift == 0) {
                    break;
                }
                if (window_last >= haystack_end) {
                    return -1;
                }
            }
          no_shift:
            window = window_last - len_needle + 1;
            ptrdiff_t i = MAX(cut, memory);
            for (; i < len_needle; ++i) {
                if (needle[i] != window[i]) {
                    if (i < gap_jump_end) {
                        window_last += gap;
                    }
                    else {
                        window_last += i - cut + 1;
                    }
                    memory = 0;
                    goto periodicwindowloop;
                }
            }
            for (i = memory; i < cut; ++i) {
                if (needle[i] != window[i]) {
                    window_last += period;
                    memory = len_needle - period;
                    if (window_last >= haystack_end) {
                        return -1;
                    }
                    ptrdiff_t shift = table[(*window_last) & TABLE_MASK];
                    if (shift) {
                        ptrdiff_t mem_jump = MAX(cut, memory) - cut + 1;
                        memory = 0;
                        window_last += MAX(shift, mem_jump);
                        goto periodicwindowloop;
                    }
                    goto no_shift;
                }
            }
            return window - haystack;
        }
    }
    else {
        period = MAX(gap, period);
      windowloop:
        while (window_last < haystack_end) {
            for (;;) {
                ptrdiff_t shift = table[(*window_last) & TABLE_MASK];
                window_last += shift;
                if (shift == 0) {
                    break;
                }
                if (window_last >= haystack_end) {
                    return -1;
                }
            }
            window = window_last - len_needle + 1;
            ptrdiff_t i = cut;
            for (; i < len_needle; ++i) {
                if (needle[i] != window[i]) {
                    if (i < gap_jump_end) {
                        window_last += gap;
                    }
                    else {
                        window_last += i - cut + 1;
                    }
                    goto windowloop;
                }
            }
            for (ptrdiff_t k = 0; k < cut; ++k) {
                if (needle[k] != window[k]) {
                    window_last += period;
                    goto windowloop;
                }
            }
            return window - haystack;
        }
    }
    return -1;
}
