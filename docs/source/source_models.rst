Source Models
=============

ngehtsim accepts ``ehtim.Image``, ``ehtim.Movie``, and ``ehtim.Model`` objects
as source-structure inputs. The adapter layer samples them onto either an
``ehtim.Obsdata`` boundary object or a native ``VisibilityDataset``. It is an
interoperability boundary, not ngehtsim's final source-model representation.

Image and Movie inputs use ngehtsim's own documented raster sampler rather
than ehtim's pyNFFT path. The ``transform_backend`` and ``raster_tolerance``
settings are described in :doc:`obsgen`; neither adapter mutates the source
object supplied by the caller.

For native datasets, the adapters sample one spectral channel into arbitrary
station/feed products through an explicit Jones measurement equation. The
source coherency is sampled in a common circular sky basis, then projected into
the configured R/L, X/Y, mixed, single-feed, or custom receptor paths. The
Image and Movie raster-transform backend is described in :doc:`obsgen`.

The lower-level functions below are useful when building integrations around
the native data model. Typical users should call ``obs_generator.make_obs()``.

.. currentmodule:: ngehtsim.obs.source_models

.. autoclass:: EhtimImageAdapter
   :members:

.. autoclass:: EhtimMovieAdapter
   :members:

.. autoclass:: EhtimModelAdapter
   :members:

.. autofunction:: adapter_for

.. autofunction:: observe_source

.. autofunction:: observe_source_dataset
