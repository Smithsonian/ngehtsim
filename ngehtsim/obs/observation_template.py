"""Native re-simulation of an imported observation's sampling template.

An :class:`ObservationTemplate` retains an existing observation's time rows,
baseline coordinates, integration lengths, flags, and uncertainty estimates.
It can then sample a different source model and apply native station Jones
corruptions without rebuilding an array schedule or querying weather data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ngehtsim.obs import instrumental_corruptions, source_models, station_observation
from ngehtsim.obs.receptor_configuration import resolve_receptor_configuration
from ngehtsim.obs.simulation_result import SimulationResult
from ngehtsim.obs.station_effects import StationCorruptionModel
from ngehtsim.obs.visibility_dataset import (
    VisibilityDataset,
    receptor_products_for_rows,
)


@dataclass(frozen=True)
class ObservationTemplate:
    """Immutable native geometry, uncertainty, and flag template.

    Parameters
    ----------
    dataset : VisibilityDataset
        Imported native data whose row times, station labels, baseline
        geometry, integration lengths, sample flags, and ``sigma_jy`` values
        define the template.

    Notes
    -----
    The class does not infer station identities from UVFITS codes.  Callers
    that want ngehtsim station metadata, such as a feed mount type, must pass
    an explicit mapping to :meth:`simulate`.
    """

    dataset: VisibilityDataset

    def __post_init__(self):
        if not isinstance(self.dataset, VisibilityDataset):
            raise TypeError("dataset must be a VisibilityDataset instance.")

    @classmethod
    def from_dataset(cls, dataset):
        """Create a template from an already-native dataset.

        Parameters
        ----------
        dataset : VisibilityDataset
            Native dataset whose sampling and uncertainty metadata are to be
            retained.

        Returns
        -------
        ObservationTemplate
            Immutable wrapper around ``dataset``.
        """

        return cls(dataset)

    @classmethod
    def from_uvfits(cls, path):
        """Read a UVFITS template through ngehtsim's native adapter.

        Parameters
        ----------
        path : str or pathlib.Path
            UVFITS file to use as the sampling and uncertainty template.
        """

        return cls(VisibilityDataset.from_uvfits(path))

    @classmethod
    def from_ehtfits(cls, path):
        """Read a FITS-EHT template while retaining mixed-feed products.

        Parameters
        ----------
        path : str or pathlib.Path
            FITS-EHT archive to use as the sampling and uncertainty template.
        """

        return cls(VisibilityDataset.from_ehtfits(path))

    def with_scans(self, scan_start_mjd, scan_stop_mjd):
        """Return a copy with caller-supplied explicit scan intervals.

        Parameters
        ----------
        scan_start_mjd, scan_stop_mjd : array_like, shape (scan,)
            UTC MJD interval boundaries.  Every row must belong to exactly one
            interval before a scan-cadence corruption can be realized.

        Returns
        -------
        ObservationTemplate
            Template with validated scan metadata attached.

        Notes
        -----
        This method intentionally accepts intervals rather than inferring them
        from time gaps.  Scan boundaries are an observing decision, not a
        universally recoverable property of visibility timestamps.
        """

        return type(self)(replace(
            self.dataset,
            scan_start_mjd=scan_start_mjd,
            scan_stop_mjd=scan_stop_mjd,
        ))

    def simulate(self, input_model, **kwargs):
        """Sample ``input_model`` on this template and apply native effects.

        Parameters
        ----------
        input_model : ehtim.Image, ehtim.Model, or ehtim.Movie
            Source structure to substitute onto this template's sampling.
        **kwargs
            Keyword arguments accepted by
            :func:`simulate_observation_template`, including ``effects``,
            receptor layout overrides, explicit mount metadata, and random
            state settings.

        Returns
        -------
        SimulationResult
            Native source-substituted data and its station-term provenance.

        Notes
        -----
        This is a method form of :func:`simulate_observation_template`; see
        that function for the full parameter contract.
        """

        return simulate_observation_template(input_model, self, **kwargs)


def read_observation_template(path, format="auto"):
    """Read a UVFITS or FITS-EHT observation template.

    Parameters
    ----------
    path : str or pathlib.Path
        Input archive path.
    format : {"auto", "uvfits", "ehtfits"}, optional
        Archive reader to use.  ``"auto"`` selects FITS-EHT only for a
        ``.ehtfits`` suffix and otherwise selects UVFITS.

    Returns
    -------
    ObservationTemplate
        Immutable imported template.
    """

    format = str(format).lower()
    if format == "auto":
        format = "ehtfits" if Path(path).suffix.lower() == ".ehtfits" else "uvfits"
    if format == "uvfits":
        return ObservationTemplate.from_uvfits(path)
    if format == "ehtfits":
        return ObservationTemplate.from_ehtfits(path)
    raise ValueError("format must be 'auto', 'uvfits', or 'ehtfits'.")


def simulate_observation_template(
    input_model,
    template,
    *,
    effects=None,
    station_receptors=None,
    station_signal_paths=None,
    station_resolver=None,
    mount_types=None,
    feed_angles_deg=None,
    source_name=None,
    transform_backend="auto",
    raster_tolerance=1.0e-12,
    random_seed=None,
    rng=None,
):
    """Generate a source substitution on an imported observation template.

    The returned dataset retains the input row timestamps, station labels,
    UVW coordinates, integration durations, channel frequency, and sample
    uncertainty scale.  With an unchanged receptor layout, its input flags
    and ``sigma_jy`` are preserved exactly.  A requested receptor-layout
    change creates all station-feed products on each row; because no direct
    product correspondence then exists, each available output product uses
    the median valid input uncertainty for that baseline-time row.

    Parameters
    ----------
    input_model : ehtim.Image, ehtim.Model, or ehtim.Movie
        Different source structure to sample at the template's UV points and
        phase centre.  Native one-channel source adapters are used.
    template : ObservationTemplate or VisibilityDataset
        Imported sampling template.  A raw :class:`VisibilityDataset` is
        accepted as a convenience.
    effects : StationCorruptionModel, optional
        Native station corruptions.  The default adds independent thermal
        noise at every integration using the template's ``sigma_jy`` values,
        but no gains, leakage, feed rotation, weather, or availability flags.
    station_receptors, station_signal_paths : mapping, optional
        Receptor layout and Jones-row configuration for the output.  Omit
        both to retain the imported layout.  Supplying a different layout,
        such as ``{"AA": ("X", "Y")}``, can create a mixed-feed output that
        should be stored as FITS-EHT rather than UVFITS.
    station_resolver : mapping, optional
        Explicit ``{template_label: ngehtsim_station_name}`` mapping used only
        to look up mount/feed-angle metadata.  It never changes the output
        station labels and no code-to-station mapping is applied implicitly.
    mount_types, feed_angles_deg : mapping, optional
        Per-template-station feed-rotation overrides.  A mount type is needed
        only when ``effects.feed_rotation`` is true; supplied values take
        precedence over resolver-derived metadata.
    source_name : str, optional
        Replacement output source label.  The template phase centre remains
        unchanged so its stored UVW coordinates stay physically meaningful.
    transform_backend : {"auto", "direct", "finufft"}, optional
        Native raster transform backend for ``ehtim.Image`` and Movie inputs.
    raster_tolerance : float, optional
        Requested relative FINUFFT tolerance when that backend is selected.
    random_seed : int, optional
        Seed used to create a fresh :class:`numpy.random.Generator`.  Mutually
        exclusive with ``rng``.
    rng : numpy.random.Generator, optional
        Random generator for all requested corruption and thermal-noise draws.

    Returns
    -------
    SimulationResult
        Native simulated dataset and row-aligned corruption provenance.

    Raises
    ------
    ValueError
        If a multi-channel template is supplied, scans are needed for a
        scan-cadence process but absent, or a feed-rotation mount is not
        explicitly supplied or resolved.

    Notes
    -----
    Template re-simulation intentionally does not re-evaluate weather,
    opacity, or thermal sensitivity.  Feed rotation is evaluated per input
    integration from the template timestamps and geometry.  The native
    multi-channel simulation path will extend this interface in a later
    change.
    """

    dataset = _template_dataset(template)
    if dataset.channel_count != 1:
        raise ValueError(
            "Template source substitution currently supports exactly one channel."
        )
    if effects is None:
        effects = _default_template_effects()
    if not isinstance(effects, StationCorruptionModel):
        raise TypeError("effects must be a StationCorruptionModel instance.")
    rng = _resolve_rng(random_seed, rng)

    configuration = _output_receptor_configuration(
        dataset,
        station_receptors,
        station_signal_paths,
    )
    output_template = _template_with_receptors(dataset, configuration)
    effects.validate_receptors(output_template.stations.names, output_template.receptors)
    context = _source_context(
        output_template,
        configuration,
        source_name,
        transform_backend,
        raster_tolerance,
    )
    sampled, _, sky_coherency = source_models.observe_source_dataset(
        input_model,
        output_template,
        context,
        return_coherency=True,
    )
    station_terms, stations = station_observation.template_station_terms_for_dataset(
        sampled,
        rng,
        effects,
        station_resolver=station_resolver,
        mount_types=mount_types,
        feed_angles_deg=feed_angles_deg,
    )
    corrupted = instrumental_corruptions.apply_receptor_corruptions(
        sampled,
        sky_coherency,
        station_terms,
        stations,
        configuration,
        effects,
        rng,
        uncertainty_mode="template",
    )
    return SimulationResult(corrupted, station_terms)


def _template_dataset(template):
    """Return the native dataset held by a public template input."""

    if isinstance(template, ObservationTemplate):
        return template.dataset
    if isinstance(template, VisibilityDataset):
        return template
    raise TypeError("template must be an ObservationTemplate or VisibilityDataset.")


def _default_template_effects():
    """Return default effects that retain a template's reported noise budget."""

    return StationCorruptionModel(
        thermal_noise=True,
        opacity_calibrated=True,
        feed_rotation=False,
        station_gain=None,
        leakage=None,
        flag_wind=False,
        flag_daylight=False,
        flag_sun=False,
    )


def _resolve_rng(random_seed, rng):
    """Resolve the public random-state inputs without hidden global state."""

    if random_seed is not None and rng is not None:
        raise ValueError("random_seed and rng cannot both be supplied.")
    if rng is not None:
        return rng
    return np.random.default_rng(random_seed)


def _output_receptor_configuration(dataset, station_receptors, station_signal_paths):
    """Resolve requested output paths while retaining omitted template feeds."""

    definitions = _template_receptor_definitions(dataset)
    if station_receptors is not None:
        supplied = dict(station_receptors)
        unknown = set(supplied) - set(dataset.stations.names)
        if unknown:
            raise ValueError(
                "station_receptors references unknown template stations: {0}.".format(
                    ", ".join(sorted(unknown)),
                )
            )
        definitions.update(supplied)
    return resolve_receptor_configuration(
        dataset.stations.names,
        definitions,
        station_signal_paths,
    )


def _template_receptor_definitions(dataset):
    """Convert a native receptor table to resolver-compatible feed mappings."""

    return _receptor_definitions(dataset.stations.names, dataset.receptors)


def _receptor_definitions(station_names, receptors):
    """Convert station-local receptors to public feed specifications."""

    definitions = {}
    for station_index, station in enumerate(station_names):
        receptor_indices = np.flatnonzero(receptors.station_index == station_index)
        definitions[station] = tuple(
            {
                "feed_id": receptors.feed_id[index],
                "polarization_label": receptors.polarization_label[index],
                "basis": receptors.basis[index],
            }
            for index in receptor_indices
        )
    return definitions


def _same_receptor_layout(left, right):
    """Return whether two receptor tables define the same output inventory."""

    return (
        np.array_equal(left.station_index, right.station_index)
        and left.feed_id == right.feed_id
        and left.polarization_label == right.polarization_label
        and left.basis == right.basis
    )


def _template_with_receptors(dataset, configuration):
    """Retain or rebuild a template's product layout for output receptors."""

    if _same_receptor_layout(dataset.receptors, configuration.receptors):
        return dataset

    products, row_product_id = receptor_products_for_rows(
        configuration.receptors,
        dataset.antenna1,
        dataset.antenna2,
    )
    rows, slots = row_product_id.shape
    visibilities = np.zeros((rows, 1, slots), dtype=complex)
    sigma_jy = np.full((rows, 1, slots), np.nan, dtype=float)
    flags = np.ones((rows, 1, slots), dtype=bool)
    for row, product_ids in enumerate(row_product_id):
        present = product_ids >= 0
        source_present = dataset.sample_present[row, 0]
        source_sigma = dataset.sigma_jy[row, 0]
        source_flags = dataset.flags[row, 0]
        valid = (
            source_present
            & ~source_flags
            & np.isfinite(source_sigma)
            & (source_sigma > 0.0)
        )
        if np.any(valid):
            sigma_jy[row, 0, present] = np.median(source_sigma[valid])
            flags[row, 0, present] = False
    return replace(
        dataset,
        receptors=configuration.receptors,
        correlation_products=products,
        row_product_id=row_product_id,
        visibilities=visibilities,
        sigma_jy=sigma_jy,
        flags=flags,
    )


def _source_context(dataset, configuration, source_name, transform_backend, raster_tolerance):
    """Build source-adapter metadata without relying on an observation generator."""

    station_receptors = _receptor_definitions(
        dataset.stations.names,
        configuration.receptors,
    )
    station_signal_paths = {}
    for index, (station_index, feed_id) in enumerate(zip(
        configuration.receptors.station_index,
        configuration.receptors.feed_id,
    )):
        station = dataset.stations.names[station_index]
        station_signal_paths.setdefault(station, {})[feed_id] = {
            "jones_vector": configuration.response_circular[index],
            "sefd_scale": configuration.sefd_scale[index],
            "gain_scale": configuration.gain_scale[index],
        }
    return {
        "source": dataset.source if source_name is None else str(source_name),
        "ra": dataset.ra_hours,
        "dec": dataset.dec_degrees,
        "mjd": float(np.floor(np.min(dataset.time_mjd))),
        "rf": float(dataset.channel_frequency_hz[0]),
        "transform_backend": transform_backend,
        "raster_tolerance": raster_tolerance,
        "verbosity": 0,
        "station_receptors": station_receptors,
        "station_signal_paths": station_signal_paths,
    }
