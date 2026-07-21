Native Simulation Results
=========================

Ground-array simulations retain their full native result until an explicit
export boundary is requested. This preserves flagged rows and supports data
that cannot be expressed as ``ehtim.Obsdata``.

``SimulationResult`` couples a ``VisibilityDataset`` with the station terms
used to generate it. The values in ``station_terms`` are implementation
products of a particular simulation, rather than a stable on-disk calibration
format. FITS-EHT currently archives the visibility dataset; an archive format
for station-term realizations will be introduced only after the public
station-corruption configuration has been redesigned.

Ground Geometry and Station Terms
---------------------------------

The native ground-array geometry kernel produces baseline rows, elevation
selection masks, and a circular visibility template without constructing an
``ehtim.Obsdata`` object. Spacecraft stations retain the legacy geometry path.
The station-term kernel then evaluates weather, SEFD, uptime, gains, leakage,
and feed-rotation quantities on those rows. Its current corruption application
is circular and one-channel; mixed-receptor station corruptions are a planned
native extension rather than an implicit conversion.

.. currentmodule:: ngehtsim.obs.observation_geometry

.. autoclass:: GroundGeometry
   :members:

.. autoclass:: StationGeometry
   :members:

.. autofunction:: ground_geometry

.. autofunction:: ground_visibility_template

.. autofunction:: visibility_dataset_elevation_mask

.. autofunction:: apply_visibility_dataset_elevation_limits

.. currentmodule:: ngehtsim.obs.station_observation

.. autofunction:: station_metadata_for_dataset

.. autofunction:: station_terms_for_dataset

Fringe and Corruption Primitives
--------------------------------

The fringe functions are deterministic detectability and fringe-selection
primitives. Frequency phase transfer (FPT) is currently a selection proxy; it
does not alter visibility phases. Circular corruption functions currently
require a one-channel dataset containing exactly RR, LL, RL, and LR products.

.. currentmodule:: ngehtsim.obs.simulation_result

.. autoclass:: SimulationResult
   :members:

.. currentmodule:: ngehtsim.obs.fringe_selection

.. autoclass:: FringeRows
   :members:

.. autofunction:: fringe_group_mask

.. autofunction:: fpt_fringe_group_mask

.. currentmodule:: ngehtsim.obs.instrumental_corruptions

.. autofunction:: apply_circular_leakage

.. autofunction:: apply_circular_corruptions
