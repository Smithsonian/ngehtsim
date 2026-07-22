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

Mixed-Receptor Native Simulation
--------------------------------

The native simulator can generate a different receptor inventory at every
station. Configure this through constructor arguments, rather than a YAML
setting, so feed and calibration definitions remain explicit Python data:

.. code-block:: python

   obsgen = obs_generator.obs_generator(
       settings=settings,
       station_receptors={
           "ALMA": ("X", "Y"),
           "APEX": ("R", "L"),
           "CUSTOM": ("R", "X", "Y"),
           "SINGLE": ("Y",),
       },
   )

Standard ``R``, ``L``, ``X``, and ``Y`` labels receive their conventional
Jones rows in a common circular sky basis. Each recorded baseline contains the
full Cartesian product of the two stations' available voltage streams, so a
single-feed or three-feed station does not require a special data shape.

For a non-standard calibrated signal path, provide a feed mapping and its
explicit Jones row, relative SEFD, and fixed gain:

.. code-block:: python

   import numpy as np

   obsgen = obs_generator.obs_generator(
       settings=settings,
       station_receptors={"CUSTOM": ({"feed_id": "P"},)},
       station_signal_paths={
           "CUSTOM": {
               "P": {
                   "jones_vector": (1.0, 0.3j),
                   "sefd_scale": 1.15,
                   "gain_scale": 0.98 * np.exp(0.1j),
               },
           },
       },
   )

Native fringe selection reconstructs Stokes I when the available feed products
have rank two. A rank-deficient baseline, such as one containing a single-feed
station, uses its strongest individual product as the fringe-detection proxy.
This follows the HOPS fringe-group criterion without treating a particular
polarization basis as privileged. FPT uses the same evidence at target and
reference frequency; it remains a detectability proxy and does not modify
visibility phases.

Mixed-receptor datasets can be written to FITS-EHT. UVFITS and
``ehtim.Obsdata`` export remain intentionally limited to their representable
uniform-polarization layouts.

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
