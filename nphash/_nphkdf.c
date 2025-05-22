#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <openssl/evp.h>
#include <openssl/kdf.h>

// HKDF Expand-only: key = seed, info = info, digest = blake2b/sha256, out = keystream
// Returns 0 on success, -1 on error
int numpy_hkdf(const unsigned char* key,
               const size_t keylen,
               const char* digest_name,
               const unsigned char* info,
               size_t infolen,
               size_t outlen,
               uint8_t* restrict out) {
    // Select the digest algorithm based on the input string
    const EVP_MD* md =
        (strcmp(digest_name, "blake2b") == 0) ? EVP_blake2b512() :
        (strcmp(digest_name, "sha256") == 0) ? EVP_sha256() : NULL;

    // Fail early if digest is not available
    if (!md)
        return -1;

    // Create a new HKDF context for key derivation
    EVP_PKEY_CTX* pctx = EVP_PKEY_CTX_new_id(EVP_PKEY_HKDF, NULL);

    // Check if the context was created successfully
    bool ok = pctx;

    // Perform HKDF key derivation
    ok &= EVP_PKEY_derive_init(pctx) > 0;
    ok &= EVP_PKEY_CTX_set_hkdf_mode(pctx, EVP_PKEY_HKDEF_MODE_EXPAND_ONLY) > 0;
    ok &= EVP_PKEY_CTX_set_hkdf_md(pctx, md) > 0;
    ok &= EVP_PKEY_CTX_set1_hkdf_key(pctx, key, (int)keylen) > 0;
    ok &= EVP_PKEY_CTX_add1_hkdf_info(pctx, info, (int)infolen) > 0;
    ok &= EVP_PKEY_derive(pctx, out, &outlen) > 0;

    // Free the HKDF context to avoid memory leaks
    EVP_PKEY_CTX_free(pctx);

    // Return 0 on success, -1 on failure
    return ok ? 0 : -1;
}
