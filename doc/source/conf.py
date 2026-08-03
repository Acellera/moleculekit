# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from pathlib import Path

from acellera_docs_theme import apply

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

exclude_patterns = ["build", "**/.ipynb_checkpoints"]

# Publish llms.txt, the index that points retrieval pipelines at llms-full.txt.
# It is a committed source file rather than hook output, and ".txt" is not a
# source_suffix, so without this Sphinx never copies it into the build dir -- and
# docs.yml deploys by rsyncing only doc/build/html.
#
# llms-full.txt must NOT be listed here. Extra files are copied during
# builder.finish(), which runs before the build-finished hook below writes it, so
# listing it would publish a stale copy or fail on a clean build.
html_extra_path = ["llms.txt"]

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

# -- Acellera unified branding ----------------------------------------------

apply(
    globals(),
    project_name="MoleculeKit",
    github_repo="Acellera/moleculekit",
)

# -- LLM full-corpus artifact ------------------------------------------------


def _emit_llms_full_txt(app, exception):
    """build-finished hook: concatenate every doc page source into llms-full.txt.

    Written to two places, both of which matter:

    - ``outdir`` so it is published. docs.yml deploys by rsyncing only
      ``doc/build/html`` to the docs bucket, so a file written solely into srcdir
      can never reach https://software.acellera.com/moleculekit/llms-full.txt.
      Writing here is safe from build-finished: Sphinx's copy phase is already
      done, so nothing will overwrite it.
    - ``srcdir`` for local retrieval pipelines that read it straight out of the
      checkout (playmoleculeAI's vectordb/config/collections.yaml points at
      ``doc/source/llms-full.txt``).
    """
    if exception is not None:
        return
    srcdir = Path(app.srcdir)
    parts = []
    for pattern in ("*.md", "*.rst"):
        for path in sorted(srcdir.rglob(pattern)):
            if "build" in path.parts:
                continue
            rel = path.relative_to(srcdir)
            parts.append(f"# === {rel} ===\n\n{path.read_text(encoding='utf-8')}\n")
    corpus = "\n".join(parts)
    for output in (srcdir / "llms-full.txt", Path(app.outdir) / "llms-full.txt"):
        output.write_text(corpus, encoding="utf-8")


def setup(app):
    app.connect("build-finished", _emit_llms_full_txt)
    return {"version": "1.0", "parallel_read_safe": True}
