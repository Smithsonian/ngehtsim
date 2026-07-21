Native Visibility Data
======================

``VisibilityDataset`` is ngehtsim's native in-memory representation for
visibility data. It is independent of ``ehtim.Obsdata`` and can represent
multiple spectral channels, explicitly flagged samples, and arrays whose
stations have different receptor layouts.

Data Model
----------

A dataset has baseline-time *rows*, frequency *channels*, and dense in-memory
*product slots*.  Each populated slot maps through ``row_product_id`` to an
ordered pair of entries in ``CorrelationProductTable``. Each product entry, in
turn, identifies one receptor at ``antenna1`` and one receptor at ``antenna2``.
This representation supports arbitrary station-local receptor inventories:
for example, circular R/L, linear X/Y, a single Y receptor, or a custom
three-receptor station can coexist in one dataset.

The data arrays ``visibilities``, ``sigma_jy``, and ``flags`` have shape
``(row, channel, product_slot)``. ``sigma_jy`` is the one-standard-deviation
uncertainty, in Jy, of each real or imaginary component of a complex
visibility. Native datasets deliberately do not store statistical weights.

Unused dense product slots are represented by ``row_product_id == -1``. They
are always flagged and have ``NaN`` uncertainty; use ``sample_present`` to
distinguish padding from an actual flagged sample.

Boundary Adapters
-----------------

``from_ehtim_obsdata()`` and ``to_ehtim_obsdata()`` are deliberately narrow
boundary adapters. The latter requires one unflagged circular-polarization
channel because those are the constraints of ``ehtim.Obsdata``. Use FITS-EHT
for lossless interchange of native mixed-receptor data.

.. currentmodule:: ngehtsim.obs.visibility_dataset

.. autoclass:: StationTable
   :members:

.. autoclass:: ReceptorTable
   :members:

.. autoclass:: CorrelationProductTable
   :members:

.. autofunction:: standard_products_for_rows

.. autoclass:: VisibilityDataset
   :members:
