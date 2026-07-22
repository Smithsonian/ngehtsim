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
    """Visibility rows required for fringe-detection selection.

    Parameters
    ----------
    time : array_like, shape (row,)
        Timestamp keys. Values are compared exactly, so callers should use a
        common time representation for all rows passed to one selection call.
    station1, station2 : array_like, shape (row,)
        Ordered station identifiers for each baseline.
    integration_time_s : array_like, shape (row,)
        Positive integration durations in seconds.
    rr, ll : array_like, shape (row,), optional
        Legacy parallel-hand circular visibility samples. They remain accepted
        for the historical Stokes-I convenience path.
    rr_sigma, ll_sigma : array_like, shape (row,), optional
        Positive one-sigma uncertainties corresponding to ``rr`` and ``ll``.
    fringe_snr : array_like, shape (row,), optional
        Basis-agnostic scalar fringe evidence. When supplied, fringe and FPT
        selection use this value directly instead of the legacy R/L estimate.

    Notes
    -----
    This compact structure intentionally contains only the information needed
    for the published detectability proxy. It is not a calibrated visibility
    data model.
    """

    time: np.ndarray
    station1: np.ndarray
    station2: np.ndarray
    integration_time_s: np.ndarray
    rr: np.ndarray | None = None
    ll: np.ndarray | None = None
    rr_sigma: np.ndarray | None = None
    ll_sigma: np.ndarray | None = None
    fringe_snr: np.ndarray | None = None

    def __post_init__(self):
        arrays = {
            "time": self.time,
            "station1": self.station1,
            "station2": self.station2,
            "integration_time_s": self.integration_time_s,
        }
        legacy_values = (self.rr, self.ll, self.rr_sigma, self.ll_sigma)
        if any(value is not None for value in legacy_values):
            if any(value is None for value in legacy_values):
                raise ValueError("Legacy RR/LL fringe inputs must be supplied together.")
            arrays.update({
                "rr": self.rr,
                "ll": self.ll,
                "rr_sigma": self.rr_sigma,
                "ll_sigma": self.ll_sigma,
            })
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
        if self.rr_sigma is not None and (
            np.any(self.rr_sigma <= 0.0) or np.any(self.ll_sigma <= 0.0)
        ):
            raise ValueError("Parallel-hand uncertainties must be positive.")
        if self.fringe_snr is not None:
            fringe_snr = np.asarray(self.fringe_snr, dtype=float)
            if fringe_snr.shape != (length,) or not np.all(np.isfinite(fringe_snr)):
                raise ValueError("fringe_snr must be finite with one value per row.")
            if np.any(fringe_snr < 0.0):
                raise ValueError("fringe_snr must be non-negative.")
            object.__setattr__(self, "fringe_snr", fringe_snr)
        elif self.rr is None:
            raise ValueError("Provide either fringe_snr or complete RR/LL inputs.")

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
        if self.rr is None:
            raise ValueError("Stokes-I SNR is unavailable without legacy RR/LL inputs.")
        amplitude = 0.5 * (np.abs(self.rr) + np.abs(self.ll))
        sigma = 0.5 * np.hypot(self.rr_sigma, self.ll_sigma)
        return amplitude / sigma

    @property
    def detectability_snr(self):
        """Return generic fringe evidence or the legacy Stokes-I proxy."""

        return self.stokes_i_snr if self.fringe_snr is None else self.fringe_snr


def receptor_fringe_snr(visibilities, sigma_jy, left_rows, right_rows, present):
    """Estimate basis-agnostic fringe SNR from arbitrary receptor products.

    When the available products constrain the full two-polarization coherency,
    this solves a weighted linear measurement equation and returns the
    reconstructed Stokes-I SNR.  For rank-deficient rows, such as a baseline
    containing a single-feed station, it returns the largest valid individual
    product SNR.  The latter is the HOPS fringe-group criterion: one reliable
    polarization-product fringe is sufficient to constrain a station edge.

    Parameters
    ----------
    visibilities, sigma_jy : array_like, shape (row, product_slot)
        Complex product samples and independent one-sigma uncertainties.
    left_rows, right_rows : array_like, shape (row, product_slot, 2)
        Effective Jones rows for the first and second receptor of each
        product, expressed in a common circular sky basis.
    present : array_like of bool, shape (row, product_slot)
        True for unflagged real product samples.

    Returns
    -------
    numpy.ndarray
        One non-negative scalar fringe SNR per row.
    """

    visibilities = np.asarray(visibilities, dtype=complex)
    sigma_jy = np.asarray(sigma_jy, dtype=float)
    left_rows = np.asarray(left_rows, dtype=complex)
    right_rows = np.asarray(right_rows, dtype=complex)
    present = np.asarray(present, dtype=bool)
    if visibilities.ndim != 2:
        raise ValueError("Fringe products must have shape (row, product_slot).")
    shape = visibilities.shape
    if any(values.shape != shape for values in (sigma_jy, present)):
        raise ValueError("Fringe product arrays must have matching shapes.")
    if left_rows.shape != shape + (2,) or right_rows.shape != shape + (2,):
        raise ValueError("Receptor rows must have shape (row, product_slot, 2).")

    snr = np.zeros(shape[0], dtype=float)
    trace = np.array((0.5, 0.0, 0.0, 0.5), dtype=complex)
    for row in range(shape[0]):
        usable = present[row] & np.isfinite(sigma_jy[row]) & (sigma_jy[row] > 0.0)
        if not np.any(usable):
            continue
        values = visibilities[row, usable]
        uncertainties = sigma_jy[row, usable]
        left = left_rows[row, usable]
        right = right_rows[row, usable]
        design = np.column_stack((
            left[:, 0] * np.conj(right[:, 0]),
            left[:, 0] * np.conj(right[:, 1]),
            left[:, 1] * np.conj(right[:, 0]),
            left[:, 1] * np.conj(right[:, 1]),
        ))
        weights = 1.0 / uncertainties**2
        normal = design.conj().T @ (weights[:, np.newaxis] * design)
        if np.linalg.matrix_rank(normal) == 4:
            covariance = np.linalg.inv(normal)
            estimate = covariance @ (design.conj().T @ (weights * values))
            sigma_i = np.sqrt(np.real(trace.conj() @ covariance @ trace))
            if np.isfinite(sigma_i) and sigma_i > 0.0:
                snr[row] = np.abs(trace @ estimate) / sigma_i
                continue
        snr[row] = np.max(np.abs(values) / uncertainties)
    return snr


def fringe_group_mask(rows, snr_threshold, tint_reference_s, available_sites=None):
    """Return rows in timestamp-local components of strong baselines.

    Parameters
    ----------
    rows : FringeRows
        Target-frequency parallel-hand visibility rows.
    snr_threshold : float
        Strong-baseline SNR threshold at ``tint_reference_s``.
    tint_reference_s : float
        Positive reference integration time in seconds.
    available_sites : iterable, optional
        Restrict both graph edges and selected rows to baselines whose stations
        are in this set.

    Returns
    -------
    numpy.ndarray of bool, shape (row,)
        Rows whose stations lie in a connected component built from strong
        baselines at the same timestamp.

    Notes
    -----
    This is a detectability/fringe-selection proxy, not fringe fitting.
    """
    _validate_thresholds(snr_threshold, tint_reference_s)
    available = _availability_mask(rows.station1, rows.station2, available_sites)
    strong = available & (
        rows.detectability_snr
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
                          target_available_sites=None, reference_available_sites=None,
                          target_row_available=None, reference_row_available=None):
    """Return target rows detectable through native or FPT-supported fringes.

    ``reference_snr_threshold`` is the strong-baseline threshold at the
    reference frequency.  The corresponding target threshold is that value
    multiplied by ``reference_to_target_ratio``.  Reference and target rows
    are intentionally independent: the station graph is keyed by timestamp
    and never by row position.  Optional row-availability masks exclude
    station-flagged data before either graph is assembled.

    Parameters
    ----------
    target_rows, reference_rows : FringeRows
        Target- and reference-frequency visibility rows. Their row ordering and
        counts may differ.
    reference_snr_threshold : float
        Strong-baseline threshold at the reference frequency and reference
        integration time.
    tint_reference_s : float
        Positive reference integration time in seconds.
    reference_to_target_ratio : float
        Positive multiplier mapping the reference SNR threshold to the target
        frequency.
    target_available_sites, reference_available_sites : iterable, optional
        Site-level availability restrictions for the two datasets.
    target_row_available, reference_row_available : array_like of bool, optional
        Additional row-level availability masks.

    Returns
    -------
    numpy.ndarray of bool, shape (target_row,)
        Target rows whose stations are connected by strong target or reference
        baselines at the same timestamp.

    Notes
    -----
    FPT selection does not transfer or correct visibility phases.
    """
    _validate_thresholds(reference_snr_threshold, tint_reference_s)
    if not np.isfinite(reference_to_target_ratio) or reference_to_target_ratio <= 0.0:
        raise ValueError("reference_to_target_ratio must be positive and finite.")

    target_available = _availability_mask(
        target_rows.station1,
        target_rows.station2,
        target_available_sites,
    )
    target_available &= _row_availability_mask(
        target_row_available,
        target_rows.row_count,
        "target_row_available",
    )
    reference_available = _availability_mask(
        reference_rows.station1,
        reference_rows.station2,
        reference_available_sites,
    )
    reference_available &= _row_availability_mask(
        reference_row_available,
        reference_rows.row_count,
        "reference_row_available",
    )
    target_strong = target_available & (
        target_rows.detectability_snr
        >= reference_snr_threshold
        * reference_to_target_ratio
        * np.sqrt(target_rows.integration_time_s / tint_reference_s)
    )
    reference_strong = reference_available & (
        reference_rows.detectability_snr
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


def _row_availability_mask(values, count, name):
    if values is None:
        return np.ones(count, dtype=bool)
    values = np.asarray(values, dtype=bool)
    if values.shape != (count,):
        raise ValueError("{0} must have one value per row.".format(name))
    return values


def _validate_thresholds(snr_threshold, tint_reference_s):
    if not np.isfinite(snr_threshold) or snr_threshold < 0.0:
        raise ValueError("snr_threshold must be finite and non-negative.")
    if not np.isfinite(tint_reference_s) or tint_reference_s <= 0.0:
        raise ValueError("tint_reference_s must be positive and finite.")
