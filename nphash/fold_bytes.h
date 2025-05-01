#ifndef FOLD_BYTES_H
#define FOLD_BYTES_H

#include <stdint.h>
#include <stddef.h>

// Helper: Convert a hash buffer to a 64-bit integer (big-endian, folding all bytes)
// This ensures the output is a single uint64 value, using all entropy from the hash.
static inline uint64_t fold_bytes_to_uint64(const unsigned char* restrict hash, const size_t hash_len) {
    uint64_t val = 0;
    #if defined(__clang__)
    #pragma unroll
    #elif defined(__GNUC__)
    #pragma GCC unroll 64
    #endif
    for (size_t j = 0; j < hash_len; ++j) {
        val = (val << 8) | hash[j];
    }
    return val;
}

#endif // FOLD_BYTES_H
