Source Models
=============

ngehtsim accepts ``ehtim.Image``, ``ehtim.Movie``, and ``ehtim.Model`` objects
as source-structure inputs. The adapter layer samples them onto either an
``ehtim.Obsdata`` boundary object or a native ``VisibilityDataset``. It is an
interoperability boundary, not ngehtsim's final source-model representation.

For native datasets, the current adapter supports one-channel circular RR,
LL, RL, LR products. Mixed-receptor source sampling requires the forthcoming
native source-structure model and is rejected rather than assigned an implicit
polarization conversion. The Image and Movie raster-transform backend is
described in :doc:`obsgen`.

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
