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

The reader only opens local filesystem paths. Synchronization and download are
deliberately outside of ngehtsim, so the same interface works with a manually
downloaded release, a network filesystem, or a cloud-synchronized directory.

.. autoclass:: ngehtsim.weather.zarr_store.ZarrWeatherStore
   :members:
