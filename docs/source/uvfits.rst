Native UVFITS I/O
=================

The native visibility representation can read and write standard AIPS UVFITS
random-groups files without constructing an ``ehtim.Obsdata`` object.  This
supports multiple spectral channels and IF spectral windows for datasets with
a single global circular or linear correlation basis.

.. code-block:: python

   from ngehtsim.obs.uvfits import read_uvfits, write_uvfits

   dataset = read_uvfits("observation.uvfits")
   write_uvfits(dataset, "round_trip.uvfits")

The same adapters are available on ``VisibilityDataset`` as
``from_uvfits()`` and ``to_uvfits()``.  UVFITS has one global STOKES axis and
cannot losslessly store arbitrary per-baseline mixed-feed layouts.  Such
datasets remain native ``VisibilityDataset`` objects until an explicit output
basis conversion is selected.

.. automodule:: ngehtsim.obs.uvfits
   :members:
