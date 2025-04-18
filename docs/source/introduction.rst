========================
Introduction
========================

The ngehtsim package is intended to provide a fast and flexible way to generate synthetic interferometric data.  It inherits much of its core functionality from the `ehtim <https://github.com/achael/eht-imaging>`_ [#Chael2016]_ [#Chael2018]_ library but adds to it a number of relevant data corruption effects and fringe-fitting emulation.  The ngehtsim package also provides substantial flexibility to generate data from a highly heterogeneous array, with properties such as dish size, surface accuracy, bandwidth, and receiver noise temperatures being specifiable at a per-site level.

Basic framework
========================

The primary functionality provided by the ngehtsim package is synthetic data generation, whose properties are determined by an ``obs_generator`` object.  The ``obs_generator`` object is initialized with a number of settings, many of which come with reasonable default values but several of which the user will typically want to specify.  To initialize an ``obs_generator`` object, simply call the initialization function::
   
   import ngehtsim.obs.obs_generator as og
   obsgen = og.obs_generator()

When called with no arguments, this function will return an ``obs_generator`` object with the default settings, which are stored in ``obsgen.settings``.  To specify non-default settings during initialization, a user can either provide a path to a YAML file or else pass a dictionary to the initialization function.  For instance, to specify the right ascension and declination of a target source while also aiming to observe with only a small array consisting of the ALMA, LMT, and SMT telescopes, we could pass a settings dictionary::

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
========================

The ngehtsim package contains a library of existing and potential telescope sites whose locations and atmospheric properties are used during synthetic data generation.  The list of available sites can be accessed using::

   import ngehtsim.obs.obs_generator as og
   site_list = og.get_site_list()

For sites with existing telescopes, such as the LMT, various pieces of telescope information -- such as the dish size, surface accuracy, and coordinate location -- are already tabulated within ngehtsim.

The ngehtsim package also contains historical weather information at every site that has been tabulated from the `MERRA-2 <https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/>`_ database [#Rienecker2011]_ [#Molod2015]_ [#Gelaro2017]_.  The atmospheric state data from MERRA-2 have been processed through the *am* radiative transfer code [#Paine2022]_ to determine opacities and brightness temperatures as a function of time and observing frequency (any frequency in the range from 0 to 2 THz can be specified).  This information is used internally by ngehtsim to determine the sensitivity of each site during observation generation.

We can access the available weather information for an individual site using the ``ngehtsim.weather`` subpackage.  For instance, to access the 230 GHz zenith opacity at the LMT site on 2017 April 10, we can call::

   import ngehtsim.weather.weather as nw
   tau = nw.opacity('LMT', freq=230.0, day=10, month='Apr', year=2017)

The ngehtsim library also contains similar functions for accessing the atmospheric pressure, temperature, level of precipitable water vapor (PWV), and ground windspeed at each site.

References
========================

.. [#Chael2016] Chael, A. A. et al. (2016) *High-resolution Linear Polarimetric Imaging for the Event Horizon Telescope*, ApJ, 829, 11, DOI: `10.3847/0004-637X/829/1/11 <https://iopscience.iop.org/article/10.3847/0004-637X/829/1/11>`_

.. [#Chael2018] Chael, A. A. et al. (2018) *Interferometric Imaging Directly with Closure Phases and Closure Amplitudes*, ApJ, 857, 23, DOI: `10.3847/1538-4357/aab6a8 <https://iopscience.iop.org/article/10.3847/1538-4357/aab6a8>`_

.. [#Rienecker2011] Rienecker, M. M. et al. (2011) *MERRA: NASA’s Modern-Era Retrospective Analysis for Research and Applications*, Journal of Climate, 24, 3624, DOI: `10.1175/JCLI-D-11-00015.1 <https://journals.ametsoc.org/view/journals/clim/24/14/jcli-d-11-00015.1.xml>`_

.. [#Molod2015] Molod, A. et al. (2015) *Development of the GEOS-5 atmospheric general circulation model: evolution from MERRA to MERRA2*, Geoscientific Model Development, 8, 1339, DOI: `10.5194/gmd-8-1339-2015 <https://gmd.copernicus.org/articles/8/1339/2015/>`_

.. [#Gelaro2017] Gelaro, R. et al. (2017) *The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2)*, Journal of Climate, 30, 5419, DOI: `10.1175/JCLI-D-16-0758.1 <https://journals.ametsoc.org/view/journals/clim/30/14/jcli-d-16-0758.1.xml>`_

.. [#Paine2022] Paine, S. (2022) *The am atmospheric model*, 12.2, Zenodo, DOI: `10.5281/zenodo.6774378 <https://zenodo.org/record/6774378>`_
