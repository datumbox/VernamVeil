#ifndef _NPBLAKE3_H
#define _NPBLAKE3_H

#include <stdint.h>
#include <stddef.h>

// Hashes an array of uint64 elements with a seed using BLAKE3, outputs variable-length hashes
void numpy_blake3(const uint64_t* arr, size_t n, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size);

// Hashes multiple data chunks with BLAKE3, outputs variable-length hash
void bytes_multi_chunk_blake3(
    const uint8_t* const* data_chunks, // Array of pointers to data buffers
    const size_t* data_lengths,        // Array of lengths for each buffer
    size_t num_chunks,                 // Number of chunks
    const char* seed,                  // Key for keyed hashing (optional)
    size_t seedlen,                    // Key length
    uint8_t* out,                      // Output buffer for the digest
    size_t hash_size                   // Desired digest length
);

#endif // _NPBLAKE3_H

