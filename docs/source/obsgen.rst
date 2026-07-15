.. module:: ngehtsim.obs

Observation utilities
=========================

Observation generation
-------------------------

Definition of the obs_generator interface.

To use an externally distributed Zarr weather release, open the local dataset
explicitly and pass the resulting store to the generator. Omitting
``weather_store`` retains the packaged binary-weather behavior.

.. code-block:: python

   from ngehtsim.obs import obs_generator
   from ngehtsim.weather.zarr_store import ZarrWeatherStore

   store = ZarrWeatherStore("/path/to/ngehtsim-weather-merra2-3hour-v0.1.0.zarr")
   obsgen = obs_generator.obs_generator(settings=settings, weather_store=store)

By default, this continues to use the release's daily weather aggregates. To
use linearly interpolated three-hour weather during observation generation,
opt in explicitly:

.. code-block:: python

   obsgen = obs_generator.obs_generator(
       settings=settings,
       weather_store=store,
       weather_cadence="native",
   )

Native weather varies the opacity, atmospheric and ground temperatures, and
wind effects at each observation timestamp. SYMBA's static ``.antennas``
format cannot represent this time dependence and is not available in native
weather mode.

.. automodule:: ngehtsim.obs.obs_generator
   :members:

Observation plotting
-------------------------

Definition of the obs_plotter interface.

.. automodule:: ngehtsim.obs.obs_plotter
   :members:
