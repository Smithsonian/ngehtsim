"""Sphinx configuration for the ngehtsim documentation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "ngehtsim"
copyright = "2026, Smithsonian Institution"
author = "Dom Pesce"

from ngehtsim import __version__

version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
