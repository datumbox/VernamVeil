#ifndef _NPBLAKE3_H
#define _NPBLAKE3_H

#include <stdint.h>
#include <stddef.h>

void numpy_blake3(const uint64_t* arr, size_t n, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size);
void bytes_blake3(const uint8_t* data, size_t datalen, const char* seed, size_t seedlen, uint8_t* out, size_t hash_size);

#endif // _NPBLAKE3_H

