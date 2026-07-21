FITS-EHT I/O
============

FITS-EHT is ngehtsim's lossless FITS-based native visibility format. It is a
project-owned convention, not a FITS-IDI variant that existing FITS-IDI readers
can interpret. It uses FITS binary tables and retains the familiar station,
frequency, source, and UV-data organization, while adding explicit receptor and
correlation-product metadata needed for mixed-polarization arrays.

Use the module functions or the matching ``VisibilityDataset`` methods:

.. code-block:: python

   from ngehtsim.obs.ehtfits import read_ehtfits, write_ehtfits

   dataset = read_ehtfits("observation.ehtfits")
   write_ehtfits(dataset, "round_trip.ehtfits")

Each receiver signal path has an ``EHT_RECEPTORS`` row that identifies its
station, local feed identifier, polarization label, and basis. ``UV_DATA``
records reference ordered pairs of those receptors through variable-length
product IDs. This permits, for example, a two-receptor linear station, a
two-receptor circular station, and a three-receptor mixed station to coexist
without inventing a global polarization axis or a fixed four-product layout.

Visibility payloads contain real and imaginary components, ``SIGMA`` values in
Jy, and explicit flags. ``SIGMA`` is the one-standard-deviation uncertainty of
each real or imaginary component. FITS-EHT has no weight field. An unflagged
sample requires a finite positive uncertainty; a flagged sample may have
``NaN`` uncertainty when the originating format did not provide one.

``to_uvfits()`` remains available for standard uniformly circular or linear
datasets. It converts ``sigma_jy`` to UVFITS's required inverse-variance field
only at that export boundary. Mixed-receptor datasets should use FITS-EHT.

.. automodule:: ngehtsim.obs.ehtfits
   :members:
