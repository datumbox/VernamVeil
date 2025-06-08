#ifndef TWOWAY_H
#define TWOWAY_H

#include <stddef.h>
#include <stdint.h>
#include <limits.h>

// Constants for the Two-Way algorithm table
#define SHIFT_TYPE uint8_t
#define MAX_SHIFT UINT8_MAX
#define TABLE_SIZE_BITS 6u
#define TABLE_SIZE (1U << TABLE_SIZE_BITS) // 64
#define TABLE_MASK (TABLE_SIZE - 1U)       // 63

typedef struct {
    const unsigned char *needle;
    ptrdiff_t len_needle;
    ptrdiff_t cut;
    ptrdiff_t period;
    ptrdiff_t gap;
    int is_periodic;
    SHIFT_TYPE table[TABLE_SIZE];
} prework;

void preprocess(const unsigned char *needle, ptrdiff_t len_needle, prework *p);
ptrdiff_t two_way(const unsigned char *haystack, ptrdiff_t len_haystack, prework *p);

#endif // TWOWAY_H

