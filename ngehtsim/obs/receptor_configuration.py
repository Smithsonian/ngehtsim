"""Validated station-receptor configuration for native RIME simulations.

The native simulator represents every recorded voltage stream with a Jones
row in a common circular sky basis.  Standard ``R``, ``L``, ``X``, and ``Y``
labels have conventional rows; arbitrary signal paths must provide one
explicitly.  This keeps the on-disk data model permissive while preventing
the simulator from inventing physics for an unfamiliar feed label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ngehtsim.obs.visibility_dataset import ReceptorTable


STANDARD_JONES_ROWS = {
    "R": np.array((1.0, 0.0), dtype=complex),
    "L": np.array((0.0, 1.0), dtype=complex),
    "X": np.array((1.0, 1.0), dtype=complex) / np.sqrt(2.0),
    "Y": np.array((-1.0j, 1.0j), dtype=complex) / np.sqrt(2.0),
}


@dataclass(frozen=True)
class ReceptorConfiguration:
    """Resolved station-local signal paths for one native simulation.

    Parameters
    ----------
    receptors : ReceptorTable
        Explicit station/feed inventory carried by the output dataset.
    response_circular : numpy.ndarray, shape (receptor, 2)
        Uncorrupted Jones row for each voltage stream in the common circular
        ``(R, L)`` sky basis.
    sefd_scale : numpy.ndarray, shape (receptor,)
        Positive per-receptor multiplier applied to weather-derived station
        SEFD values when calculating thermal uncertainty.
    gain_scale : numpy.ndarray, shape (receptor,)
        Fixed complex per-receptor gain. This is applied in addition to the
        simulated stochastic station gain when ``addgains=True``.

    Notes
    -----
    The response row is deliberately general: it can describe ordinary
    circular or linear feeds, a single-feed station, an over-complete feed
    inventory, or a custom calibrated voltage path.
    """

    receptors: ReceptorTable
    response_circular: np.ndarray
    sefd_scale: np.ndarray
    gain_scale: np.ndarray

    def __post_init__(self):
        response = np.array(self.response_circular, dtype=complex, copy=True)
        sefd_scale = np.array(self.sefd_scale, dtype=float, copy=True)
        gain_scale = np.array(self.gain_scale, dtype=complex, copy=True)
        count = self.receptors.count
        if response.shape != (count, 2):
            raise ValueError("response_circular must have shape (receptor, 2).")
        if sefd_scale.shape != (count,) or gain_scale.shape != (count,):
            raise ValueError("Receptor path scales must have one value per receptor.")
        if not np.all(np.isfinite(response)) or not np.all(np.isfinite(gain_scale)):
            raise ValueError("Receptor response and gain values must be finite.")
        if np.any(sefd_scale <= 0.0) or not np.all(np.isfinite(sefd_scale)):
            raise ValueError("Receptor SEFD scales must be positive and finite.")
        if np.any(np.linalg.norm(response, axis=1) == 0.0):
            raise ValueError("Receptor Jones rows must be non-zero.")
        response.setflags(write=False)
        sefd_scale.setflags(write=False)
        gain_scale.setflags(write=False)
        object.__setattr__(self, "response_circular", response)
        object.__setattr__(self, "sefd_scale", sefd_scale)
        object.__setattr__(self, "gain_scale", gain_scale)


def resolve_receptor_configuration(station_names, station_receptors=None,
                                   station_signal_paths=None):
    """Resolve public station/feed settings into immutable native paths.

    Parameters
    ----------
    station_names : iterable of str
        Ordered native station names.
    station_receptors : mapping, optional
        Maps station names to an ordered sequence of standard labels or feed
        specifications. Omitting a station uses the conventional ``("R", "L")``
        inventory. A feed specification is a mapping with ``feed_id`` and
        optional ``polarization_label`` keys.
    station_signal_paths : mapping, optional
        Optional nested mapping ``{station: {feed_id: specification}}``. Each
        specification may set ``jones_vector`` (two complex circular-basis
        components), ``sefd_scale`` (positive float), and ``gain_scale``
        (complex). It supplements, rather than replaces, ``station_receptors``.

    Returns
    -------
    ReceptorConfiguration
        Resolved receptors and their physical signal-path properties.

    Raises
    ------
    ValueError
        If a station is unknown, a feed is duplicated, a standard response is
        unavailable, or a supplied path specification is invalid.
    """

    names = tuple(str(name) for name in station_names)
    receptor_map = {} if station_receptors is None else dict(station_receptors)
    path_map = {} if station_signal_paths is None else dict(station_signal_paths)
    unknown = (set(receptor_map) | set(path_map)) - set(names)
    if unknown:
        raise ValueError("Receptor settings reference unknown stations: {0}.".format(
            ", ".join(sorted(unknown))
        ))

    station_index = []
    feed_ids = []
    labels = []
    bases = []
    responses = []
    sefd_scales = []
    gain_scales = []
    for index, station in enumerate(names):
        feeds = receptor_map.get(station, ("R", "L"))
        if isinstance(feeds, str):
            raise ValueError("Station receptor inventories must be sequences, not strings.")
        seen = set()
        for feed in feeds:
            spec = _feed_specification(feed)
            feed_id = spec["feed_id"]
            if feed_id in seen:
                raise ValueError("Station {0} repeats feed_id {1!r}.".format(station, feed_id))
            seen.add(feed_id)
            path_spec = dict(path_map.get(station, {}).get(feed_id, {}))
            path_spec.update({key: value for key, value in spec.items() if key != "feed_id"})
            label = str(path_spec.pop("polarization_label", feed_id)).upper()
            basis = str(path_spec.pop("basis", _basis_for_label(label))).upper()
            response = _response_for_path(label, path_spec.pop("jones_vector", None))
            sefd_scale = float(path_spec.pop("sefd_scale", 1.0))
            gain_scale = complex(path_spec.pop("gain_scale", 1.0 + 0.0j))
            if path_spec:
                raise ValueError(
                    "Unsupported signal-path keys for {0}/{1}: {2}.".format(
                        station, feed_id, ", ".join(sorted(path_spec))
                    )
                )
            station_index.append(index)
            feed_ids.append(feed_id)
            labels.append(label)
            bases.append(basis)
            responses.append(response)
            sefd_scales.append(sefd_scale)
            gain_scales.append(gain_scale)
        if not seen:
            raise ValueError("Station {0} must define at least one receptor.".format(station))

    return ReceptorConfiguration(
        receptors=ReceptorTable(
            station_index=np.asarray(station_index, dtype=np.intp),
            feed_id=tuple(feed_ids),
            polarization_label=tuple(labels),
            basis=tuple(bases),
        ),
        response_circular=np.asarray(responses, dtype=complex),
        sefd_scale=np.asarray(sefd_scales, dtype=float),
        gain_scale=np.asarray(gain_scales, dtype=complex),
    )


def response_rows_for_dataset(dataset, configuration):
    """Return response rows aligned to a dataset's receptor table.

    A native geometry template and its configuration must describe precisely
    the same station/feed inventory.  This strict check prevents accidental
    application of a response to the wrong voltage stream.
    """

    receptors = dataset.receptors
    configured = configuration.receptors
    if (
        not np.array_equal(receptors.station_index, configured.station_index)
        or receptors.feed_id != configured.feed_id
        or receptors.polarization_label != configured.polarization_label
    ):
        raise ValueError("Dataset receptors do not match the resolved signal-path configuration.")
    return configuration.response_circular, configuration.sefd_scale, configuration.gain_scale


def configuration_for_dataset(dataset, station_receptors=None, station_signal_paths=None):
    """Resolve or infer the signal paths needed by an existing dataset.

    Parameters
    ----------
    dataset : VisibilityDataset
        Native dataset whose receptor table defines the required station/feed
        inventory.
    station_receptors, station_signal_paths : mapping, optional
        Public configuration inputs accepted by
        :func:`resolve_receptor_configuration`. When both are omitted, the
        existing dataset's standard R/L/X/Y labels are used to infer paths.

    Returns
    -------
    ReceptorConfiguration
        Configuration guaranteed to match ``dataset.receptors``.

    Raises
    ------
    ValueError
        If a custom-label dataset has no explicit Jones-row configuration or
        supplied settings do not match the dataset receptor inventory.
    """

    if station_receptors is None and station_signal_paths is None:
        inferred = {}
        for station_index, station in enumerate(dataset.stations.names):
            receptor_indices = np.flatnonzero(dataset.receptors.station_index == station_index)
            inferred[station] = tuple(
                {
                    "feed_id": dataset.receptors.feed_id[index],
                    "polarization_label": dataset.receptors.polarization_label[index],
                    "basis": dataset.receptors.basis[index],
                }
                for index in receptor_indices
            )
        configuration = resolve_receptor_configuration(dataset.stations.names, inferred)
    else:
        configuration = resolve_receptor_configuration(
            dataset.stations.names,
            station_receptors,
            station_signal_paths,
        )
    response_rows_for_dataset(dataset, configuration)
    return configuration


def _feed_specification(feed):
    if isinstance(feed, str):
        return {"feed_id": str(feed)}
    if not isinstance(feed, dict):
        raise TypeError("Each station receptor must be a label string or mapping.")
    if "feed_id" not in feed:
        raise ValueError("Receptor mappings must define feed_id.")
    return dict(feed)


def _basis_for_label(label):
    if label in ("R", "L"):
        return "CIRCULAR"
    if label in ("X", "Y"):
        return "LINEAR"
    return "CUSTOM"


def _response_for_path(label, supplied):
    if supplied is None:
        try:
            return np.array(STANDARD_JONES_ROWS[label], copy=True)
        except KeyError as exc:
            raise ValueError(
                "Receptor label {0!r} requires an explicit jones_vector.".format(label)
            ) from exc
    response = np.asarray(supplied, dtype=complex)
    if response.shape != (2,):
        raise ValueError("jones_vector must contain exactly two circular-basis components.")
    return np.array(response, copy=True)
