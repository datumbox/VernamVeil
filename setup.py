import os
from setuptools import setup

try:
    # Attempt to import the dummy package to check if C extensions should be built
    import nphash_c_ext_trigger
    c_ext = True
except ImportError:
    c_ext = False

if c_ext:
    os.environ["SETUPTOOLS_BUILD"] = "1"
    cffi_modules_list = [
        "nphash/_build_utils/_ffi_builders.py:_get_bytesearch_ffi",
        "nphash/_build_utils/_ffi_builders.py:_get_npblake2b_ffi",
        "nphash/_build_utils/_ffi_builders.py:_get_npblake3_ffi",
        "nphash/_build_utils/_ffi_builders.py:_get_npsha256_ffi",
    ]
else:
    cffi_modules_list = []
    if "SETUPTOOLS_BUILD" in os.environ:
        del os.environ["SETUPTOOLS_BUILD"]

setup(cffi_modules=cffi_modules_list, zip_safe=False)
