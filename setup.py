from setuptools import setup

setup(
    cffi_modules=[
        "nphash._build_utils._ffi_builders:_get_bytesearch_ffi",
        "nphash._build_utils._ffi_builders:_get_blake2b_ffi",
        "nphash._build_utils._ffi_builders:_get_blake3_ffi",
        "nphash._build_utils._ffi_builders:_get_sha256_ffi",
    ],
    zip_safe=False
)
