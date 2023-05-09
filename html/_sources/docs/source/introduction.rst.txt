Introduction
========================

The ngehtsim package is intended to provide a fast and flexible way to generate synthetic interferometric data.  It inherits much of its core functionality from the `ehtim <https://github.com/achael/eht-imaging>`_ [Chael2016]_[Chael2018]_ library but adds to it a number of relevant data corruption effects and fringe-fitting emulation.  The ngehtsim package also provides substantial flexibility to generate data from a highly heterogeneous array, with properties such as dish size, surface accuracy, bandwidth, and receiver noise temperatures being specifiable at a per-site level.

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

The ``make_obs()`` function returns an `ehtim <https://github.com/achael/eht-imaging>`_-compatible ``obsdata`` object, which can be saved, imaged, or otherwise manipulated using any of the tools in the `ehtim <https://github.com/achael/eht-imaging>`_ library.  For instance, to save the observation as a uvfits file, we simply call::

   obs.save_uvfits('synthetic_observation.uvfits')

Site information
------------------------

The ngehtsim package contains a library of existing and potential telescope sites whose locations and atmospheric properties are used during synthetic data generation.  The list of available sites can be accessed using::

   import ngehtsim.obs.obs_generator as og
   site_list = og.get_site_list()

For sites with existing telescopes, such as the LMT, the telescope information is inherited from the `ngehtutil <https://github.com/Smithsonian/ngehtutil>`_ library.  We can pull up the associated `ngehtutil <https://github.com/Smithsonian/ngehtutil>`_ ``Station`` object using::

   import ngehtutil as ng
   LMT = ng.Station.from_name('LMT')

The `ngehtutil <https://github.com/Smithsonian/ngehtutil>`_ library provides information such as the dish size, surface accuracy, location, and level of local infrastructure at the selected site.

The ngehtsim package also contains historical weather information at every site that has been tabulated from the `MERRA-2 <https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/>`_ database [Rienecker2011]_[Molod2015]_[Gelaro2017]_.



References
------------------------

.. [Chael2016] Chael, A. A. et al. *High-resolution Linear Polarimetric Imaging for the Event Horizon Telescope*  2016, ApJ, 829, 11

.. [Chael2018] Chael, A. A. et al. *Interferometric Imaging Directly with Closure Phases and Closure Amplitudes*  2018, ApJ, 857, 23

.. [Gelaro2017] Gelaro, R. et al. *The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2)*  2017, Journal of Climate, 30, 5419

.. [Molod2015] Molod, A. et al. *Development of the GEOS-5 atmospheric general circulation model: evolution from MERRA to MERRA2*  2015, Geoscientific Model Development, 8, 1339

.. [Rienecker2011] Rienecker, M. M. et al. *MERRA: NASA’s Modern-Era Retrospective Analysis for Research and Applications*  2011, Journal of Climate, 24, 3624

