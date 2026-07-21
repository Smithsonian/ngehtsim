"""
Tools for generating and plotting observations.
"""

__author__ = "Dom Pesce"

__all__ = [
    'fringe_selection',
    'ehtfits',
    'obs_generator',
    'obs_plotter',
    'simulation_result',
    'uvfits',
    'visibility_dataset',
]


def __getattr__(name):
    """Import observation submodules only when they are requested."""

    if name in __all__:
        import importlib

        module = importlib.import_module("{0}.{1}".format(__name__, name))
        globals()[name] = module
        return module
    raise AttributeError("module {0!r} has no attribute {1!r}".format(__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
