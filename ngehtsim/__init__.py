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

__all__ = ['obs', 'metrics', 'const_def', 'weather', 'calibration']


def __getattr__(name):
    """Import public subpackages only when they are requested."""

    if name in __all__:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


from . import _version
__version__ = _version.get_versions()['version']
