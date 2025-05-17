#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>

// #define __NO_SIMD__  // Uncomment to disable SIMD optimisations

#if !defined(__NO_SIMD__)

// SIMD intrinsics headers and alignment macro
#if defined(__AVX512F__)
#include <immintrin.h>
#define NP_SIMD_AVX512 1
#define NP_SIMD_BATCH 8
#define NP_SIMD_ALIGN 64
#elif defined(__AVX2__)
#include <immintrin.h>
#define NP_SIMD_AVX2 1
#define NP_SIMD_BATCH 4
#define NP_SIMD_ALIGN 32
#elif defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || _M_IX86_FP >= 2))
#include <emmintrin.h>
#define NP_SIMD_SSE2 1
#define NP_SIMD_BATCH 2
#define NP_SIMD_ALIGN 16
#endif

// Alignment macro for SIMD types
#if defined(_MSC_VER)
#define ALIGNED(N) __declspec(align(N))
#else
#define ALIGNED(N) __attribute__((aligned(N)))
#endif

#endif

#ifdef _OPENMP
// Enable parallelisation with OpenMP for multi-core performance
#include <omp.h>
#endif

#define BLOCK_SIZE 8  // Each input element is an uint64 block
#define HASH_SIZE 64  // BLAKE2b output size in bytes


// Utility: HMAC BLAKE2b for a single block
static inline void hmac_blake2b(const char* seed, int seedlen, const unsigned char* block, uint8_t* out) {
    HMAC(EVP_blake2b512(), seed, seedlen, block, BLOCK_SIZE, out, NULL);
}

// Utility: Hash BLAKE2b for a single block
static inline void hash_blake2b(const unsigned char* block, uint8_t* out) {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    int ok = (ctx != NULL);
    ok &= EVP_DigestInit_ex(ctx, EVP_blake2b512(), NULL) == 1;
    ok &= EVP_DigestUpdate(ctx, block, BLOCK_SIZE) == 1;
    ok &= EVP_DigestFinal_ex(ctx, out, NULL) == 1;
    EVP_MD_CTX_free(ctx);
}

// HMACs or hashes an array of uint64 elements with a seed using BLAKE2b, outputs 64-byte hashes
// - If a seed is provided, HMAC is used for cryptographic safety.
// - If no seed is provided, a plain hash is used (not recommended for security).
// - Output is a 2D uint8 array of shape n x 64 (n rows, 64 columns)
void numpy_blake2b(const uint64_t* restrict arr, const size_t n, const char* restrict seed, const size_t seedlen, uint8_t* restrict out) {
    // Input: native-endian uint64_t array; each element is hashed as an 8-byte block
    // When passing to hash/HMAC, use (const unsigned char *)&arr[i] and BLOCK_SIZE
    const unsigned char (*arr8)[BLOCK_SIZE] = (const unsigned char (*)[BLOCK_SIZE])arr;
    const int n_int = (int)n;
    int i = 0;

    if (seed != NULL && seedlen > 0) {
    #if defined(NP_SIMD_AVX512) || defined(NP_SIMD_AVX2) || defined(NP_SIMD_SSE2)
        // SIMD path: batch process with AVX512, AVX2 or SSE2
        const int batch = NP_SIMD_BATCH;
        #ifdef _OPENMP
        // Parallelise the loop with OpenMP to use multiple CPU cores
        #pragma omp parallel for schedule(static)
        #endif
        for (i = 0; i <= n_int - batch; i += batch) {
            ALIGNED(NP_SIMD_ALIGN) unsigned char tmp[NP_SIMD_BATCH][BLOCK_SIZE];
        #if NP_SIMD_AVX512
            __m512i v = _mm512_loadu_si512((const void*)arr8[i]);
            _mm512_store_si512((void*)tmp, v);
        #elif NP_SIMD_AVX2
            __m256i v = _mm256_loadu_si256((const __m256i*)arr8[i]);
            _mm256_store_si256((__m256i*)tmp, v);
        #elif NP_SIMD_SSE2
            __m128i v = _mm_loadu_si128((const __m128i*)arr8[i]);
            _mm_store_si128((__m128i*)tmp, v);
        #endif
            // Write HMAC output directly to the output buffer
            for (int j = 0; j < batch; ++j) {
                hmac_blake2b(seed, (int)seedlen, tmp[j], &out[(i + j) * HASH_SIZE]);
            }
        }
        // the scalar fallback and the non-SIMD branch are unified and follow the above ifdef
    #endif
        #ifdef _OPENMP
        // Enable parallelisation with OpenMP for multi-core performance
        #pragma omp parallel for schedule(static)
        #endif
        for (int k = i; k < n_int; ++k) {
            hmac_blake2b(seed, (int)seedlen, arr8[k], &out[k * HASH_SIZE]);
        }
    } else {
        // Hash branch
    #if defined(NP_SIMD_AVX512) || defined(NP_SIMD_AVX2) || defined(NP_SIMD_SSE2)
        // SIMD path: batch process with AVX512, AVX2 or SSE2
        const int batch = NP_SIMD_BATCH;
        #ifdef _OPENMP
        // Parallelise the loop with OpenMP to use multiple CPU cores
        #pragma omp parallel for schedule(static)
        #endif
        for (i = 0; i <= n_int - batch; i += batch) {
            ALIGNED(NP_SIMD_ALIGN) unsigned char tmp[NP_SIMD_BATCH][BLOCK_SIZE];
        #if NP_SIMD_AVX512
            __m512i v = _mm512_loadu_si512((const void*)arr8[i]);
            _mm512_store_si512((void*)tmp, v);
        #elif NP_SIMD_AVX2
            __m256i v = _mm256_loadu_si256((const __m256i*)arr8[i]);
            _mm256_store_si256((__m256i*)tmp, v);
        #elif NP_SIMD_SSE2
            __m128i v = _mm_loadu_si128((const __m128i*)arr8[i]);
            _mm_store_si128((__m128i*)tmp, v);
        #endif
            // Create a new digest context for each hash computation
            for (int j = 0; j < batch; ++j) {
                hash_blake2b(tmp[j], &out[(i + j) * HASH_SIZE]);
            }
        }
        // the scalar fallback and the non-SIMD branch are unified and follow the above ifdef
    #endif
        #ifdef _OPENMP
        // Enable parallelisation with OpenMP for multi-core performance
        #pragma omp parallel for schedule(static)
        #endif
        for (int k = i; k < n_int; ++k) {
            hash_blake2b(arr8[k], &out[k * HASH_SIZE]);
        }
    }
}
