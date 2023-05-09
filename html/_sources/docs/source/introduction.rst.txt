Introduction
========================

The ngehtsim package is intended to provide a fast and flexible way to generate synthetic interferometric data.  It inherits much of its core functionality from the `ehtim <https://github.com/achael/eht-imaging>`_ library but adds to it a number of relevant data corruption effects and fringe-fitting emulation.  The ngehtsim package also provides substantial flexibility to generate data from a highly heterogeneous array, with properties such as dish size, surface accuracy, bandwidth, and receiver noise temperatures being specifiable at a per-site level.

Basic framework
------------------------

The primary functionality provided by the ngehtsim package is synthetic data generation, whose properties are determined by an ``obs_generator`` object.  The ``obs_generator`` object is initialized with a number of settings, many of which come with reasonable default values but several of which the user will typically want to specify.  To initialize an ``obs_generator`` object, simply call the initialization function::
   
   import ngehtsim.obs.obs_generator as og
   obsgen = og.obs_generator()

When called with no arguments, this function will return an ``obs_generator`` object with the default settings, which are stored in ``obsgen.settings``.  To specify non-default settings during initializion, a user can either provide path to a YAML file or else pass a dictionary to the initialization function.  For instance, to specify the right ascension and declination of a target source while also aiming to observe with only a small array consisting of the ALMA, LMT, and SMT telescopes, we could pass a settings dictionary::

   settings = {'RA': 12.5137,
               'DEC': 12.3911,
               'sites': ['ALMA', 'LMT', 'SMT']}
   obsgen = og.obs_generator(settings)

These specified settings will override the corresponding default values.

To generate a synthetic observation using this ``obs_generator`` object, we need to first supply the emission structure of the source that is to be observed.  The ngehtsim package supports a number of different formats for the specified emission structure.  For instance, we could use `ehtim <https://github.com/achael/eht-imaging>`_ to create a ring-like emission model::
   
   import ehtim as eh
   mod = eh.model.Model()
   mod = mod.add_ring(F0=1.5, d=40.*eh.RADPERUAS)

Generating a dataset using the array settings specified in the ``obs_generator`` object and the source properties contained in the ring-like emission model then simply requires calling the ``make_obs()`` function::

   obs = obsgen.make_obs(mod)

The ``make_obs()`` function returns an `ehtim <https://github.com/achael/eht-imaging>`_-compatible ``obsdata`` object, which can be saved, imaged, or otherwise manipulated using all of the tools in the `ehtim <https://github.com/achael/eht-imaging>`_ library.  For instance, to save the observation as a uvfits file, we simply call::

   obs.save_uvfits('synthetic_observation.uvfits')

Sites
------------------------

The ngehtsim package contains a library of existing and potential telescope sites whose locations and atmospheric properties have been tabulated.







