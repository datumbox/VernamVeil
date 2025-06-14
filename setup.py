from setuptools import setup

setup(
    cffi_modules=[
        "nphash/_build_utils/_ffi_builders.py:_get_bytesearch_ffi",
        "nphash/_build_utils/_ffi_builders.py:_get_npblake2b_ffi",
        "nphash/_build_utils/_ffi_builders.py:_get_npblake3_ffi",
        "nphash/_build_utils/_ffi_builders.py:_get_npsha256_ffi",
    ],
    zip_safe=False,
)
