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

   store = ZarrWeatherStore("/path/to/ngehtsim-weather-merra2-3hour-v0.2.0.zarr")
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

Raster Source Transform Backend
-------------------------------

Rasterized ``ehtim.Image`` and ``ehtim.Movie`` sources are sampled by
ngehtsim's native raster sampler. The default setting uses FINUFFT for the
native ground-array simulation route, removing the pyNFFT runtime dependency.
The source object is treated as input-only: ngehtsim does not overwrite its
source coordinates, frequency, or metadata.

``transform_backend`` accepts three values:

* ``"auto"`` is the default. It selects FINUFFT for native simulation and the
  exact direct transform for ``ehtim.Obsdata`` compatibility routes.
* ``"finufft"`` requests the native FINUFFT backend explicitly. It is only
  available for native ground-array simulation.
* ``"direct"`` uses ngehtsim's exact discrete Fourier-transform reference
  implementation. It is useful for validation and is the compatibility-route
  default.

``raster_tolerance`` specifies the requested FINUFFT relative accuracy. Its
default, ``1e-12``, is appropriate for synthetic visibility generation:

.. code-block:: python

   settings = {
       "transform_backend": "finufft",
       "raster_tolerance": 1.0e-12,
   }
   obsgen = obs_generator.obs_generator(settings=settings)

The removed v1 settings ``ttype`` and ``fft_pad_factor`` are not accepted in
v2. FINUFFT supports both even- and odd-sized raster dimensions.

.. automodule:: ngehtsim.obs.obs_generator
   :members:

Observation plotting
-------------------------

Definition of the obs_plotter interface.

.. automodule:: ngehtsim.obs.obs_plotter
   :members:
