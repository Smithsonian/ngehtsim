Observation-Template Simulations
================================

``ObservationTemplate`` provides a native equivalent of ehtim's
``observe_same`` workflow.  It reads an existing UVFITS or FITS-EHT data set,
retains its actual baseline-time rows and reported measurement uncertainties,
then samples a different source model through ngehtsim's native RIME.  It does
not reconstruct an array schedule and does not evaluate the weather database.

Loading an EHT-like UVFITS dataset
----------------------------------

Start by reading a UVFITS file through the native adapter.  No ehtim
``Obsdata`` object is created in this workflow::

   from ngehtsim.obs.observation_template import ObservationTemplate

   template = ObservationTemplate.from_uvfits(
       "SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits"
   )

The template retains the station labels written in the file.  ngehtsim does
not assume that a label such as ``AA`` means ALMA.  This prevents a simulation
from applying a station-specific convention merely because a file happened to
use a familiar two-letter code.

The stored rows may include scan metadata through a UVFITS NX table.  A
per-scan gain model requires that metadata.  When a file lacks it, provide
the scan intervals from the observing schedule explicitly; do not infer them
from timestamp gaps::

   import numpy as np

   template = template.with_scans(
       scan_start_mjd=np.array([57849.0500, 57849.1050]),
       scan_stop_mjd=np.array([57849.0950, 57849.1500]),
   )

The interval values above are illustrative only.  Supply the actual scan
boundaries for the imported observation.  A manual fixed-time cadence is also
available through ``RealizationCadence.interval(seconds)`` when that is the
intended physical model.

Replacing the source structure
------------------------------

The source phase centre and observing frequency remain those of the imported
template, so the stored UVW coordinates remain valid.  Substitute only the
underlying brightness structure::

   import ehtim as eh
   from ngehtsim.obs.station_effects import StationCorruptionModel

   model = eh.model.Model().add_circ_gauss(
       F0=0.6,
       FWHM=40.0 * eh.RADPERUAS,
   )
   effects = StationCorruptionModel(
       thermal_noise=True,
       station_gain=None,
       feed_rotation=False,
       leakage=None,
       flag_wind=False,
       flag_daylight=False,
       flag_sun=False,
   )
   result = template.simulate(
       model,
       effects=effects,
       source_name="Synthetic M87",
       random_seed=17,
   )

With an unchanged receptor layout, ``result.dataset`` has exactly the input
``time_mjd``, ``integration_time_s``, UVW coordinates, flags, and ``sigma_jy``
arrays.  Thermal noise remains independent for every integration and
correlation product, with its one-standard-deviation scale drawn from the
imported ``sigma_jy`` value.  This preserves the supplied thermal-noise
properties rather than recalculating SEFD from weather and station models.

Station gains and feed rotation
-------------------------------

For ordinary two-feed stations, configure the common gain ``G`` and gain
ratio ``R = G_A / G_B`` directly.  The simulator applies
``G_A = G sqrt(R)`` and ``G_B = G / sqrt(R)``.  Each amplitude and phase is a
separate real-valued stochastic process; complex output visibilities contain
the usual wrapped phase naturally::

   from ngehtsim.obs.station_effects import (
       GainModel,
       GainRatioModel,
       LeakageModel,
       RealizationCadence,
       StationCorruptionModel,
   )

   effects = StationCorruptionModel(
       thermal_noise=True,
       station_gain=GainModel(
           amplitude_sigma_dex=0.04,
           phase_distribution="normal",
           phase_sigma_rad=0.30,
           amplitude_cadence=RealizationCadence.scan(),
           phase_cadence=RealizationCadence.scan(),
       ),
       gain_ratio=GainRatioModel(
           amplitude_sigma_dex=0.02,
           phase_distribution="normal",
           phase_sigma_rad=0.10,
       ),
       gain_ratio_overrides={
           "AA": GainRatioModel(
               "R", "L",
               amplitude_sigma_dex=0.04,
               phase_distribution="normal",
               phase_sigma_rad=0.20,
           ),
       },
       leakage=LeakageModel(component_sigma=0.02),
       feed_rotation=True,
       flag_wind=False,
       flag_daylight=False,
       flag_sun=False,
   )

   station_resolver = {
       "AA": "ALMA",
       "AP": "APEX",
       "AZ": "SMT",
       "JC": "JCMT",
       "LM": "LMT",
       "PV": "IRAM",
       "SM": "SMA",
       "SR": "SPT",
   }
   result = template.simulate(
       model,
       effects=effects,
       station_resolver=station_resolver,
       random_seed=18,
   )

The non-override models apply at every station. The station-specific mappings
replace those defaults; an override value of ``None`` disables that corruption
at one station. A default ``GainRatioModel()`` applies to every two-feed
station using its declared local feed order, so it works across mixed R/L and
X/Y arrays. Single-feed stations have no gain ratio. Supply ``feed_a`` and
``feed_b`` only when a particular ratio model must name its feed ordering.

The resolver is entirely caller supplied and only looks up mount/feed-angle
metadata.  It does not rename output stations.  Alternatively, set
``mount_types`` and ``feed_angles_deg`` directly with the labels stored in the
template.  A station with one feed has only the common gain; requesting a
gain ratio for it is an error.  Gain ratios are intentionally limited to two
feed stations until a general multi-feed parameterization is introduced.

Mixed-feed output and FITS-EHT
------------------------------

The output can change its station-local receptor inventory.  For example,
ALMA can be represented with native X/Y feeds while another station retains
R/L feeds::

   result = template.simulate(
       model,
       effects=effects,
       station_receptors={"AA": ("X", "Y")},
       station_resolver=station_resolver,
       random_seed=19,
   )
   result.dataset.to_ehtfits("synthetic-mixed-feed.ehtfits", overwrite=True)

Changing the layout creates every station-feed product available on a row.
Because an imported R/L product has no one-to-one uncertainty counterpart in
an X/Y or mixed basis, ngehtsim assigns every available output product the
median valid input ``sigma_jy`` for that baseline-time row.  FITS-EHT retains
these explicit products, flags, and uncertainties.  UVFITS and
``ehtim.Obsdata`` should only be used at an export boundary when the final
layout is representable by their global polarization conventions.
