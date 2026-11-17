"""Sphinx configuration for the drcurves docs site.

Build locally with::

    pip install -e ".[docs]"
    sphinx-build -W -b html docs docs/_build/html
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import drcurves

DOCS = Path(__file__).parent
ROOT = DOCS.parent

project = "drcurves"
author = "Ryan O'Sullivan"
copyright = "2026, Ryan O'Sullivan"
release = drcurves.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
]

# The walkthrough article (article/index.html, built by tools/build_article.py) is a
# single file with its charts inline. Copy it into _extra/, which Sphinx copies to the
# site root as-is, so it is served at raz1470.github.io/continuous-dr-learner/walkthrough.html.
(DOCS / "_extra").mkdir(exist_ok=True)
shutil.copyfile(ROOT / "article" / "index.html", DOCS / "_extra" / "walkthrough.html")
html_extra_path = ["_extra"]

# The example notebooks live in examples/. Copy them in at build time and render their
# saved outputs; executing them would add minutes to every build (the weekly notebooks
# workflow checks they still run).
(DOCS / "examples").mkdir(exist_ok=True)
for _notebook in sorted((ROOT / "examples").glob("*.ipynb")):
    shutil.copyfile(_notebook, DOCS / "examples" / _notebook.name)
nb_execution_mode = "off"

exclude_patterns = ["_build", "**.ipynb_checkpoints", "_extra"]

myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3

# API pages: signatures from the code, sections from the numpydoc docstrings.
autodoc_default_options = {"member-order": "bysource"}
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_xref_param_type = False
add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
# Set DOCS_OFFLINE=1 to build without fetching the inventories above.
if os.environ.get("DOCS_OFFLINE"):
    intersphinx_mapping = {}

html_theme = "pydata_sphinx_theme"
html_title = "drcurves"
html_theme_options = {
    "github_url": "https://github.com/raz1470/continuous-dr-learner",
    "navbar_align": "left",
    "show_toc_level": 2,
    "external_links": [
        {"name": "Walkthrough", "url": "https://raz1470.github.io/continuous-dr-learner/walkthrough.html"},
    ],
}
toc_object_entries_show_parents = "hide"
html_show_sourcelink = False
