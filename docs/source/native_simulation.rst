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
The station-term kernel evaluates weather, SEFD, uptime, and station geometry
on those rows. ``StationCorruptionModel`` separately declares a station-common
gain ``G`` and optional two-feed gain ratios ``R = G_A / G_B``. The one-channel
corruption kernel uses a circular-sky Jones RIME and projects the result into
every configured station-local receptor path. This supports circular, linear,
mixed, single-feed, and over-complete feed inventories without converting the
archived data to an assumed global basis.

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

.. autofunction:: template_station_terms_for_dataset

.. currentmodule:: ngehtsim.obs.station_effects

.. autoclass:: RealizationCadence
   :members:

.. autoclass:: GainModel
   :members:

.. autoclass:: GainRatioModel
   :members:

.. autoclass:: LeakageModel
   :members:

.. autoclass:: StationCorruptionModel
   :members:

Observation Templates
---------------------

.. currentmodule:: ngehtsim.obs.observation_template

.. autoclass:: ObservationTemplate
   :members:

.. autofunction:: read_observation_template

.. autofunction:: simulate_observation_template

Fringe and Corruption Primitives
--------------------------------

The fringe functions are deterministic detectability and fringe-selection
primitives. Frequency phase transfer (FPT) is currently a selection proxy; it
does not alter visibility phases. Native fringe evidence reconstructs Stokes I
from any rank-two receptor layout, with a strongest-product fallback when a
baseline cannot constrain the full coherency. This is independent of whether
the recorded feeds are circular, linear, or mixed.

.. currentmodule:: ngehtsim.obs.simulation_result

.. autoclass:: SimulationResult
   :members:

.. currentmodule:: ngehtsim.obs.fringe_selection

.. autoclass:: FringeRows
   :members:

.. autofunction:: fringe_group_mask

.. autofunction:: fpt_fringe_group_mask

.. autofunction:: receptor_fringe_snr

.. currentmodule:: ngehtsim.obs.instrumental_corruptions

.. autofunction:: apply_circular_leakage

.. autofunction:: apply_circular_corruptions

.. autofunction:: apply_receptor_corruptions

.. autofunction:: receptor_rows_for_station_terms
