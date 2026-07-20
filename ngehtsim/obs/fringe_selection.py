"""Deterministic fringe-detection selection primitives.

The routines here implement ngehtsim's published detectability proxy: at a
given timestamp, strong baselines form a station graph and every baseline
within a connected component is retained.  This is deliberately not a full
implementation of the HOPS fringe-fitting or phase-calibration pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FringeRows:
    """Visibility rows required for fringe-detection selection."""

    time: np.ndarray
    station1: np.ndarray
    station2: np.ndarray
    integration_time_s: np.ndarray
    rr: np.ndarray
    ll: np.ndarray
    rr_sigma: np.ndarray
    ll_sigma: np.ndarray

    def __post_init__(self):
        arrays = {
            "time": self.time,
            "station1": self.station1,
            "station2": self.station2,
            "integration_time_s": self.integration_time_s,
            "rr": self.rr,
            "ll": self.ll,
            "rr_sigma": self.rr_sigma,
            "ll_sigma": self.ll_sigma,
        }
        normalized = {}
        length = None
        for name, values in arrays.items():
            values = np.asarray(values)
            if values.ndim != 1:
                raise ValueError("{0} must be one-dimensional.".format(name))
            if length is None:
                length = len(values)
            elif len(values) != length:
                raise ValueError("All FringeRows fields must have the same length.")
            normalized[name] = values
        for name, values in normalized.items():
            object.__setattr__(self, name, values)

        if np.any(self.integration_time_s <= 0.0):
            raise ValueError("Integration times must be positive.")
        if np.any(self.rr_sigma <= 0.0) or np.any(self.ll_sigma <= 0.0):
            raise ValueError("Parallel-hand uncertainties must be positive.")

    @property
    def row_count(self):
        """Number of visibility rows."""
        return len(self.time)

    @property
    def stokes_i_snr(self):
        """Aligned parallel-hand Stokes-I SNR for each row.

        The proxy assumes the RR and LL products can be phase aligned before
        being stacked.  It therefore combines their amplitudes and propagates
        their independent thermal uncertainties.
        """
        amplitude = 0.5 * (np.abs(self.rr) + np.abs(self.ll))
        sigma = 0.5 * np.hypot(self.rr_sigma, self.ll_sigma)
        return amplitude / sigma


def fringe_group_mask(rows, snr_threshold, tint_reference_s, available_sites=None):
    """Return rows in timestamp-local components of strong baselines."""
    _validate_thresholds(snr_threshold, tint_reference_s)
    available = _availability_mask(rows.station1, rows.station2, available_sites)
    strong = available & (
        rows.stokes_i_snr
        >= snr_threshold * np.sqrt(rows.integration_time_s / tint_reference_s)
    )
    return connected_component_mask(
        rows.time,
        rows.station1,
        rows.station2,
        strong,
        rows.time,
        rows.station1,
        rows.station2,
        query_available=available,
    )


def fpt_fringe_group_mask(target_rows, reference_rows, reference_snr_threshold,
                          tint_reference_s, reference_to_target_ratio,
                          target_available_sites=None, reference_available_sites=None):
    """Return target rows detectable through native or FPT-supported fringes.

    ``reference_snr_threshold`` is the strong-baseline threshold at the
    reference frequency.  The corresponding target threshold is that value
    multiplied by ``reference_to_target_ratio``.  Reference and target rows
    are intentionally independent: the station graph is keyed by timestamp
    and never by row position.
    """
    _validate_thresholds(reference_snr_threshold, tint_reference_s)
    if not np.isfinite(reference_to_target_ratio) or reference_to_target_ratio <= 0.0:
        raise ValueError("reference_to_target_ratio must be positive and finite.")

    target_available = _availability_mask(
        target_rows.station1,
        target_rows.station2,
        target_available_sites,
    )
    reference_available = _availability_mask(
        reference_rows.station1,
        reference_rows.station2,
        reference_available_sites,
    )
    target_strong = target_available & (
        target_rows.stokes_i_snr
        >= reference_snr_threshold
        * reference_to_target_ratio
        * np.sqrt(target_rows.integration_time_s / tint_reference_s)
    )
    reference_strong = reference_available & (
        reference_rows.stokes_i_snr
        >= reference_snr_threshold
        * np.sqrt(reference_rows.integration_time_s / tint_reference_s)
    )

    return connected_component_mask(
        np.concatenate((target_rows.time, reference_rows.time)),
        np.concatenate((target_rows.station1, reference_rows.station1)),
        np.concatenate((target_rows.station2, reference_rows.station2)),
        np.concatenate((target_strong, reference_strong)),
        target_rows.time,
        target_rows.station1,
        target_rows.station2,
        query_available=target_available,
    )


def connected_component_mask(graph_time, graph_station1, graph_station2, graph_edges,
                             query_time, query_station1, query_station2,
                             query_available=None):
    """Select query rows whose stations share a graph component per timestamp."""
    graph_time = np.asarray(graph_time)
    graph_station1 = np.asarray(graph_station1)
    graph_station2 = np.asarray(graph_station2)
    graph_edges = np.asarray(graph_edges, dtype=bool)
    query_time = np.asarray(query_time)
    query_station1 = np.asarray(query_station1)
    query_station2 = np.asarray(query_station2)

    graph_length = len(graph_time)
    if any(len(values) != graph_length for values in (graph_station1, graph_station2, graph_edges)):
        raise ValueError("All graph fields must have the same length.")
    query_length = len(query_time)
    if any(len(values) != query_length for values in (query_station1, query_station2)):
        raise ValueError("All query fields must have the same length.")
    if query_available is None:
        query_available = np.ones(query_length, dtype=bool)
    else:
        query_available = np.asarray(query_available, dtype=bool)
        if query_available.shape != (query_length,):
            raise ValueError("query_available must have one value per query row.")

    selected = np.zeros(query_length, dtype=bool)
    for timestamp in np.unique(query_time):
        graph_indices = np.flatnonzero((graph_time == timestamp) & graph_edges)
        if not len(graph_indices):
            continue
        parent = {}

        def find(station):
            parent.setdefault(station, station)
            if parent[station] != station:
                parent[station] = find(parent[station])
            return parent[station]

        for index in graph_indices:
            station1 = graph_station1[index]
            station2 = graph_station2[index]
            root1 = find(station1)
            root2 = find(station2)
            if root1 != root2:
                parent[root2] = root1

        for index in np.flatnonzero(query_time == timestamp):
            station1 = query_station1[index]
            station2 = query_station2[index]
            if (
                query_available[index]
                and station1 in parent
                and station2 in parent
                and find(station1) == find(station2)
            ):
                selected[index] = True
    return selected


def _availability_mask(station1, station2, available_sites):
    if available_sites is None:
        return np.ones(len(station1), dtype=bool)
    available_sites = tuple(available_sites)
    return np.isin(station1, available_sites) & np.isin(station2, available_sites)


def _validate_thresholds(snr_threshold, tint_reference_s):
    if not np.isfinite(snr_threshold) or snr_threshold < 0.0:
        raise ValueError("snr_threshold must be finite and non-negative.")
    if not np.isfinite(tint_reference_s) or tint_reference_s <= 0.0:
        raise ValueError("tint_reference_s must be positive and finite.")
