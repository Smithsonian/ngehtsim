"""
This is a set of tools for simulating ngEHT datasets.
Tools may be accessed via
>>> import ngehtsim as ns
Available subpackages
---------------------
metrics
    Some functions for computing array quality metrics
obs
    Tools for generating observations
"""

__author__ = "Dom Pesce"
__bibtex__ = r"""@Article{TBD,
  %%% Fill in from ADS!
}"""

__all__ = ['obs', 'metrics', 'const_def', 'weather']
from . import *


from . import _version
__version__ = _version.get_versions()['version']
