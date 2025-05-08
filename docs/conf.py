import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

project = "VernamVeil"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

exclude_patterns = []

html_theme = "sphinx_rtd_theme"

autodoc_inherit_docstrings = True
autoclass_content = "both"

html_theme_options = {
    "vcs_pageview_mode": "blob",
}

html_context = {
    "display_github": True,
    "github_user": "datumbox",
    "github_repo": "VernamVeil",
    "github_version": "main",
    "conf_py_path": "/docs/",
}


external_readmes = [
    "README.md",
    "nphash/README.md",
]


def copy_docs(app):
    for rel_path in external_readmes:
        src = Path(app.confdir).parent / rel_path
        dst = Path(app.confdir) / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


def cleanup_docs(app, exception):
    docs_dir = Path(app.confdir)
    for rel_path in external_readmes:
        path = docs_dir / rel_path
        if path.exists():
            path.unlink()
            parent = path.parent
            if parent != docs_dir and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()


def setup(app):
    app.connect("builder-inited", copy_docs)
    app.connect("build-finished", cleanup_docs)
