Introduction and examples
========================

The ngehtsim package is intended to provide a fast and flexible way to generate synthetic interferometric data.  It inherits much of its core functionality from the `ehtim <https://github.com/achael/eht-imaging>`_ library but adds to it a number of relevant data corruption effects and fringe-fitting emulation.  The ngehtsim package also provides substantial flexibility to generate data from a highly heterogeneous array, with properties such as dish size, surface accuracy, bandwidth, and receiver noise temperatures being specifiable at a per-site level.

Basic framework
------------------------

The primary functionality provided by the ngehtsim package is synthetic data generation, whose properties are determined by an ``obs_generator`` object.  The ``obs_generator`` object is initialized with a number of settings, many of which come with reasonable default values but several of which the user will typically want to specify.  To initialize an ``obs_generator`` object, simply call the initialization function::
   
   import ngehtsim.obs.obs_generator as og
   obsgen = og.obs_generator()

When called with no arguments, this function will return an ``obs_generator`` with the default settings, which are stored in `obsgen.settings`.  To specify non-default settings during initializion, a user can either provide a dictionary or the path to a YAML file to the initialization function.









