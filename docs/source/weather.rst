.. module:: ngehtsim.weather

Weather utilities
=========================

Definition of the weather interface.

.. automodule:: ngehtsim.weather.weather
   :members:

Versioned external datasets
---------------------------

The versioned MERRA-2 weather releases are distributed separately from the
Python package. After downloading a release to a local directory, install the
optional Zarr dependency and open it with an explicit path:

.. code-block:: python

   from ngehtsim.weather.zarr_store import ZarrWeatherStore

   store = ZarrWeatherStore("/path/to/ngehtsim-weather-merra2-3hour-v0.1.0.zarr")
   april_weather = store.read_partition("ALMA", "Apr", cadence="daily")

The established weather functions can use that store directly. Without the
``weather_store`` argument, they continue to use the packaged binary data:

.. code-block:: python

   from ngehtsim.weather import weather

   opacity = weather.opacity(
       "ALMA", form="exact", month="Apr", day=11, year=2017, freq=230.0,
       weather_store=store,
   )

Native three-hour weather can be sampled at arbitrary UTC-hour offsets with
linear interpolation between consecutive stored records. ``form`` accepts
``"exact"``, ``"mean"``, ``"median"``, ``"good"``, and ``"bad"``; summary
forms are evaluated independently at each stored UTC time before interpolation.
The result always has a leading sample dimension, including for one requested
time:

.. code-block:: python

   import numpy as np

   samples = store.sample_native(
       "ALMA", year=2017, month="Apr", day=11, utc_hours=[0.0, 1.5, 3.0]
   )
   opacity_at_230_ghz = np.interp(230.0, store.frequency_ghz, samples.opacity[1])

The reader only opens local filesystem paths. Synchronization and download are
deliberately outside of ngehtsim, so the same interface works with a manually
downloaded release, a network filesystem, or a cloud-synchronized directory.

.. autoclass:: ngehtsim.weather.zarr_store.ZarrWeatherStore
   :members:

.. autoclass:: ngehtsim.weather.zarr_store.NativeWeatherSamples
