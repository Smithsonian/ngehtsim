# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./ngehtsim/'))
import versioneer

# -- Project information -----------------------------------------------------

project = 'ngehtsim'
copyright = '2022, Dom Pesce'
author = 'Dom Pesce'

# The full version, including alpha/beta/rc tags
version = versioneer.get_version()
release = version
show_authors = False

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.duration',
              'sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.ifconfig',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosectionlabel']

source_suffix = '.rst'
master_doc = 'docs/source/index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['docs/source/_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['docs/source/_static']

# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'https://docs.python.org/': None}
intersphinx_mapping = {'python': ('https://docs.python.org', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
                       'matplotlib': ('https://matplotlib.org/', None),
                       'ehtim': ('https://achael.github.io/eht-imaging/', None),
                       'ngEHTforecast': ('https://aeb.github.io/ngEHTforecast/html/', None),
                       'ngehtsim': ('https://smithsonian.github.io/ngehtsim/html/', None)
                       }
