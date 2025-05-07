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
autoclass_content = "both"

html_theme_options = {
    "collapse_navigation": False,
    "vcs_pageview_mode": "blob",
}

html_context = {
    "display_github": True,
    "github_user": "datumbox",
    "github_repo": "VernamVeil",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
