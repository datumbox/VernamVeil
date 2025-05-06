import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "VernamVeil"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"

autodoc_inherit_docstrings = True
