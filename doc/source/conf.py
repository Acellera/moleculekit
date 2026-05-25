# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from pathlib import Path

# Drop the timestamp from moleculekit log lines in rendered tutorial output.
os.environ.setdefault("MOLECULEKIT_LOG_FORMAT", "%(name)s - %(levelname)s - %(message)s")

# -- Project information -----------------------------------------------------

project = "MoleculeKit"
copyright = "2026, Acellera"
author = "Acellera"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "myst_nb",
    "sphinxarg.ext",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = ["build", "**/.ipynb_checkpoints"]

# -- MyST / MyST-NB ----------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3  # register H1-H3 anchors as cross-doc xref targets

nb_execution_mode = "cache"  # execute tutorials on first build; cache per-cell
nb_execution_in_temp = True  # run each notebook in a temp dir so writes don't pollute source/
nb_execution_timeout = 300  # some system-prep cells take ~60s
nb_merge_streams = True  # combine consecutive stream outputs into a single block

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True  # per-page Markdown source link (LLM-friendly)

html_theme_options = {
    "header_links_before_dropdown": 5,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "use_edit_page_button": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Acellera/moleculekit",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_sidebars = {
    "**": ["sidebar-nav-bs.html"],
}

# -- LLM full-corpus artifact ------------------------------------------------

def _emit_llms_full_txt(app, exception):
    """build-finished hook: concatenate every rendered page source into llms-full.txt."""
    if exception is not None:
        return
    srcdir = Path(app.srcdir)
    output = srcdir / "llms-full.txt"
    parts = []
    for path in sorted(srcdir.rglob("*.md")):
        if "build" in path.parts:
            continue
        rel = path.relative_to(srcdir)
        parts.append(f"# === {rel} ===\n\n{path.read_text(encoding='utf-8')}\n")
    for path in sorted(srcdir.rglob("*.rst")):
        if "build" in path.parts:
            continue
        rel = path.relative_to(srcdir)
        parts.append(f"# === {rel} ===\n\n{path.read_text(encoding='utf-8')}\n")
    output.write_text("\n".join(parts), encoding="utf-8")


def setup(app):
    app.connect("build-finished", _emit_llms_full_txt)
    return {"version": "1.0", "parallel_read_safe": True}
