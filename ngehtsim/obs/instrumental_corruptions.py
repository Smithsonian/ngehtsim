"""Pure circular-basis visibility corruption kernels."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import ngehtsim.const_def as const
from ngehtsim.obs.visibility_dataset import StationTable, VisibilityDataset
from ngehtsim.obs.receptor_configuration import response_rows_for_dataset
from ngehtsim.obs.station_effects import StationCorruptionModel


def apply_circular_leakage(visibilities, leakage1_r, leakage1_l, leakage2_r,
                           leakage2_l):
    """Apply ``D_1 V D_2^H`` to circular RR, LL, RL, LR visibilities.

    The final axis of ``visibilities`` must use the order ``RR, LL, RL, LR``.
    The result is a new array so every output correlation is calculated from
    the same, unmodified input coherency matrix.

    Parameters
    ----------
    visibilities : array_like, shape (..., 4)
        Circular-basis coherency products ordered RR, LL, RL, LR.
    leakage1_r, leakage1_l, leakage2_r, leakage2_l : complex or array_like
        Station 1 and station 2 circular D-terms, broadcast against the
        leading visibility dimensions.

    Returns
    -------
    numpy.ndarray
        Corrupted visibility products after applying ``D_1 V D_2^H``.
    """

    visibilities = np.asarray(visibilities, dtype=complex)
    if visibilities.ndim < 1 or visibilities.shape[-1] != 4:
        raise ValueError(
            "Circular visibility data must have a final RR, LL, RL, LR axis."
        )

    rr = visibilities[..., 0]
    ll = visibilities[..., 1]
    rl = visibilities[..., 2]
    lr = visibilities[..., 3]
    output = np.empty_like(visibilities)
    output[..., 0] = (
        rr
        + leakage1_r * lr
        + np.conj(leakage2_r) * rl
        + leakage1_r * np.conj(leakage2_r) * ll
    )
    output[..., 1] = (
        ll
        + leakage1_l * rl
        + np.conj(leakage2_l) * lr
        + leakage1_l * np.conj(leakage2_l) * rr
    )
    output[..., 2] = (
        rl
        + leakage1_r * ll
        + np.conj(leakage2_l) * rr
        + leakage1_r * np.conj(leakage2_l) * lr
    )
    output[..., 3] = (
        lr
        + leakage1_l * rr
        + np.conj(leakage2_r) * ll
        + leakage1_l * np.conj(leakage2_r) * rl
    )
    return output


def apply_circular_corruptions(dataset, station_terms, stations, rng, addnoise=True,
                               addgains=True, opacitycal=True, addFR=True,
                               addleakage=False):
    """Apply circular station terms to a native single-channel dataset.

    Flagged rows remain in the returned dataset. Use ``select_rows()`` before
    converting an eligible result to ``ehtim.Obsdata``.

    Parameters
    ----------
    dataset : VisibilityDataset
        One-channel native dataset with exactly RR, LL, RL, LR products.
    station_terms : mapping
        Row-aligned output from ``station_terms_for_dataset()``. Required terms
        include station names, opacity, SEFD, bandwidth, feed angles, and
        availability; optional gain and leakage terms are required only when
        their corresponding switches are enabled.
    stations : StationTable
        Station metadata updated by the station-term calculation.
    rng : numpy.random.Generator
        Random generator used for thermal noise.
    addnoise, addgains, opacitycal, addFR, addleakage : bool, optional
        Enable thermal noise, gain errors, opacity calibration state, feed
        rotation, and leakage corruption, respectively.

    Returns
    -------
    VisibilityDataset
        New dataset containing corrupted visibilities, propagated ``sigma_jy``
        values, flags, and calibration-state metadata.

    Notes
    -----
    This is currently a circular single-channel kernel. It rejects mixed
    receptor layouts rather than applying an implicit basis conversion.
    """

    if not isinstance(dataset, VisibilityDataset):
        raise TypeError("dataset must be a VisibilityDataset instance.")
    if not isinstance(stations, StationTable):
        raise TypeError("stations must be a StationTable instance.")
    if dataset.channel_count != 1:
        raise ValueError("Circular corruption currently requires exactly one channel.")
    try:
        circular_slots = dataset.circular_product_slots()
    except ValueError as exc:
        raise ValueError(
            "Circular corruption requires exactly RR, LL, RL, LR correlations for every row."
        ) from exc

    count = dataset.row_count
    terms = {
        name: _term_array(station_terms, name, count)
        for name in (
            "t1", "t2", "tau1", "tau2", "SEFD1", "SEFD2", "bw1", "bw2",
            "f_el1", "f_el2", "f_par1", "f_par2", "el1", "el2", "par1",
            "par2", "phi_off1", "phi_off2", "uptime_mask",
        )
    }
    if addgains:
        terms.update({
            name: _term_array(station_terms, name, count)
            for name in (
                "gainamp1R", "gainamp2R", "gainphase1R", "gainphase2R",
                "gainamp1L", "gainamp2L", "gainphase1L", "gainphase2L",
            )
        })
    if addleakage:
        terms.update({
            name: _term_array(station_terms, name, count)
            for name in ("leak1R", "leak1L", "leak2R", "leak2L")
        })
    expected_t1 = np.asarray(dataset.stations.names)[dataset.antenna1]
    expected_t2 = np.asarray(dataset.stations.names)[dataset.antenna2]
    if not np.array_equal(terms["t1"], expected_t1) or not np.array_equal(terms["t2"], expected_t2):
        raise ValueError("Station terms do not correspond to the dataset row order.")

    row_index = np.arange(count)[:, np.newaxis]
    visibilities = np.array(dataset.visibilities, copy=True)
    circular = np.array(visibilities[row_index, 0, circular_slots], copy=True)
    tau1 = np.asarray(terms["tau1"], dtype=float)
    tau2 = np.asarray(terms["tau2"], dtype=float)
    if np.any(tau1 < 0.0) or np.any(tau2 < 0.0):
        raise ValueError("Station opacity terms must be non-negative.")

    if addFR:
        feed_rotation1 = (
            terms["f_par1"] * terms["par1"]
            + terms["f_el1"] * terms["el1"]
            + (np.pi / 180.0) * terms["phi_off1"]
        )
        feed_rotation2 = (
            terms["f_par2"] * terms["par2"]
            + terms["f_el2"] * terms["el2"]
            + (np.pi / 180.0) * terms["phi_off2"]
        )
        feed_rotation1 = np.array(feed_rotation1, copy=True)
        feed_rotation2 = np.array(feed_rotation2, copy=True)
        feed_rotation1[terms["t1"] == "space"] = 0.0
        feed_rotation2[terms["t2"] == "space"] = 0.0
        circular[:, 0] *= np.exp(-1.0j * feed_rotation1) * np.exp(1.0j * feed_rotation2)
        circular[:, 1] *= np.exp(1.0j * feed_rotation1) * np.exp(-1.0j * feed_rotation2)
        circular[:, 2] *= np.exp(-1.0j * feed_rotation1) * np.exp(-1.0j * feed_rotation2)
        circular[:, 3] *= np.exp(1.0j * feed_rotation1) * np.exp(1.0j * feed_rotation2)

    if addleakage:
        leakage_terms = {
            name: _term_array(station_terms, name, count)
            for name in ("leak1R", "leak1L", "leak2R", "leak2L")
        }
        circular[:] = apply_circular_leakage(
            circular,
            leakage_terms["leak1R"],
            leakage_terms["leak1L"],
            leakage_terms["leak2R"],
            leakage_terms["leak2L"],
        )

    sigma = _baseline_sigmas(
        terms["SEFD1"],
        terms["SEFD2"],
        np.minimum(terms["bw1"], terms["bw2"]),
        dataset.integration_time_s,
        tau1,
        tau2,
        opacitycal,
    )
    if addgains:
        gain_terms = {
            name: _term_array(station_terms, name, count)
            for name in (
                "gainamp1R", "gainamp1L", "gainamp2R", "gainamp2L",
                "gainphase1R", "gainphase1L", "gainphase2R", "gainphase2L",
            )
        }
        gain1r = gain_terms["gainamp1R"] * np.exp(1.0j * gain_terms["gainphase1R"])
        gain1l = gain_terms["gainamp1L"] * np.exp(1.0j * gain_terms["gainphase1L"])
        gain2r = gain_terms["gainamp2R"] * np.exp(1.0j * gain_terms["gainphase2R"])
        gain2l = gain_terms["gainamp2L"] * np.exp(1.0j * gain_terms["gainphase2L"])
        gain_products = np.column_stack((
            gain1r * np.conj(gain2r),
            gain1l * np.conj(gain2l),
            gain1r * np.conj(gain2l),
            gain1l * np.conj(gain2r),
        ))
        circular *= gain_products
        sigma *= np.abs(gain_products)

    if not opacitycal:
        circular *= np.sqrt(np.exp(-tau1 - tau2))[:, np.newaxis]

    if addnoise:
        for index in range(4):
            circular[:, index] += sigma[:, index] * (
                rng.normal(0.0, 1.0, count)
                + 1.0j * rng.normal(0.0, 1.0, count)
            )

    flags = np.array(dataset.flags, copy=True)
    flagged_sites = set(station_terms["flagsites"])
    row_mask = (
        ~np.isin(terms["t1"], tuple(flagged_sites))
        & ~np.isin(terms["t2"], tuple(flagged_sites))
        & np.asarray(terms["uptime_mask"], dtype=bool)
    )
    flags[row_index, 0, circular_slots] |= ~row_mask[:, np.newaxis]
    sigma_jy = np.array(dataset.sigma_jy, copy=True)
    sigma_jy[row_index, 0, circular_slots] = sigma
    visibilities[row_index, 0, circular_slots] = circular
    return replace(
        dataset,
        stations=stations,
        tau1=tau1,
        tau2=tau2,
        visibilities=visibilities,
        sigma_jy=sigma_jy,
        flags=flags,
        ampcal=not addgains,
        phasecal=not addgains,
        opacitycal=opacitycal,
        dcal=not addleakage,
        frcal=not addFR,
    )


def apply_receptor_corruptions(dataset, sky_coherency, station_terms, stations,
                               receptor_configuration, effects, rng,
                               uncertainty_mode="sefd"):
    """Apply a one-channel Jones RIME to arbitrary station-feed products.

    ``sky_coherency`` is sampled in the common circular sky basis and has one
    ``2 x 2`` matrix per visibility row.  Every stored product is then formed
    as ``e_p J_1 B J_2^H e_q^H``, where ``e`` is the configured receptor row
    and ``J`` contains the simulated station gain, leakage, and feed-rotation
    terms. This formulation supports circular, linear, mixed, single-feed,
    and over-complete receptor inventories without relabelling products.

    Parameters
    ----------
    dataset : VisibilityDataset
        One-channel native visibility layout to populate.
    sky_coherency : array_like, shape (row, 2, 2)
        Source coherency matrices in the circular ``(R, L)`` sky basis.
    station_terms, stations, receptor_configuration, effects, rng
        Native station-term realization, updated station table, resolved
        receptor paths, station-effect model, and random generator.
    uncertainty_mode : {"sefd", "template"}, optional
        ``"sefd"`` calculates product uncertainty from the simulated station
        SEFD terms. ``"template"`` retains the supplied native
        ``dataset.sigma_jy`` values, which is used when re-simulating an
        imported observation with its reported uncertainty budget.

    Returns
    -------
    VisibilityDataset
        Corrupted products with thermal uncertainty and flag state propagated.
    """

    if not isinstance(dataset, VisibilityDataset):
        raise TypeError("dataset must be a VisibilityDataset instance.")
    if not isinstance(stations, StationTable):
        raise TypeError("stations must be a StationTable instance.")
    if not isinstance(effects, StationCorruptionModel):
        raise TypeError("effects must be a StationCorruptionModel instance.")
    if uncertainty_mode not in ("sefd", "template"):
        raise ValueError("uncertainty_mode must be either 'sefd' or 'template'.")
    if dataset.channel_count != 1:
        raise ValueError("Native receptor corruption requires exactly one channel.")
    sky_coherency = np.asarray(sky_coherency, dtype=complex)
    if sky_coherency.shape != (dataset.row_count, 2, 2):
        raise ValueError("sky_coherency must have shape (row, 2, 2).")
    response, sefd_scale, gain_scale = response_rows_for_dataset(
        dataset,
        receptor_configuration,
    )

    count = dataset.row_count
    terms = {
        name: _term_array(station_terms, name, count)
        for name in (
            "t1", "t2", "tau1", "tau2", "SEFD1", "SEFD2", "bw1", "bw2",
            "f_el1", "f_el2", "f_par1", "f_par2", "el1", "el2", "par1",
            "par2", "phi_off1", "phi_off2", "uptime_mask",
        )
    }
    expected_t1 = np.asarray(dataset.stations.names)[dataset.antenna1]
    expected_t2 = np.asarray(dataset.stations.names)[dataset.antenna2]
    if not np.array_equal(terms["t1"], expected_t1) or not np.array_equal(terms["t2"], expected_t2):
        raise ValueError("Station terms do not correspond to the dataset row order.")
    tau1 = np.asarray(terms["tau1"], dtype=float)
    tau2 = np.asarray(terms["tau2"], dtype=float)
    if np.any(tau1 < 0.0) or np.any(tau2 < 0.0):
        raise ValueError("Station opacity terms must be non-negative.")

    jones1, jones2 = _native_station_jones(terms, count, station_terms, effects)
    output_coherency = jones1 @ sky_coherency @ np.swapaxes(np.conj(jones2), -1, -2)
    if not effects.opacity_calibrated:
        output_coherency *= np.sqrt(np.exp(-tau1 - tau2))[:, np.newaxis, np.newaxis]
    gain_ratio_factors = _gain_ratio_factor_array(
        station_terms,
        count,
        dataset.receptors.count,
    )

    visibilities = np.array(dataset.visibilities, copy=True)
    sigma_jy = np.array(dataset.sigma_jy, copy=True)
    flags = np.array(dataset.flags, copy=True)
    products = dataset.correlation_products
    if uncertainty_mode == "sefd":
        base_sigma = _baseline_sigmas(
            terms["SEFD1"],
            terms["SEFD2"],
            np.minimum(terms["bw1"], terms["bw2"]),
            dataset.integration_time_s,
            tau1,
            tau2,
            effects.opacity_calibrated,
        )[:, 0]
    else:
        base_sigma = None
    flagged_sites = set(station_terms["flagsites"])
    row_available = (
        ~np.isin(terms["t1"], tuple(flagged_sites))
        & ~np.isin(terms["t2"], tuple(flagged_sites))
        & np.asarray(terms["uptime_mask"], dtype=bool)
    )
    for row, product_ids in enumerate(dataset.row_product_id):
        for slot, product_id in enumerate(product_ids):
            if product_id < 0:
                continue
            receptor1 = products.receptor1_id[product_id]
            receptor2 = products.receptor2_id[product_id]
            first = (
                gain_ratio_factors[row, receptor1]
                * gain_scale[receptor1]
                * response[receptor1]
            )
            second = (
                gain_ratio_factors[row, receptor2]
                * gain_scale[receptor2]
                * response[receptor2]
            )
            value = first @ output_coherency[row] @ np.conj(second)
            if uncertainty_mode == "sefd":
                gain_magnitude = (
                    np.abs(station_terms["common_gain1"][row])
                    * np.abs(gain_ratio_factors[row, receptor1] * gain_scale[receptor1])
                    * np.abs(station_terms["common_gain2"][row])
                    * np.abs(gain_ratio_factors[row, receptor2] * gain_scale[receptor2])
                )
                sigma = (
                    base_sigma[row]
                    * np.sqrt(sefd_scale[receptor1] * sefd_scale[receptor2])
                    * gain_magnitude
                )
            else:
                sigma = sigma_jy[row, 0, slot]
                if not np.isfinite(sigma) or sigma <= 0.0:
                    flags[row, 0, slot] = True
                    continue
            if effects.thermal_noise:
                value += sigma * (rng.normal() + 1.0j * rng.normal())
            visibilities[row, 0, slot] = value
            sigma_jy[row, 0, slot] = sigma
            flags[row, 0, slot] |= not row_available[row]

    return replace(
        dataset,
        stations=stations,
        tau1=tau1,
        tau2=tau2,
        visibilities=visibilities,
        sigma_jy=sigma_jy,
        flags=flags,
        ampcal=not effects.has_gain_corruption,
        phasecal=not effects.has_gain_corruption,
        opacitycal=effects.opacity_calibrated,
        dcal=not effects.has_leakage_corruption,
        frcal=not effects.feed_rotation,
    )


def _native_station_jones(terms, count, station_terms, effects):
    """Build native common-frame station Jones matrices from generic terms."""

    identity = np.broadcast_to(np.eye(2, dtype=complex), (count, 2, 2)).copy()
    jones1 = np.array(identity, copy=True)
    jones2 = np.array(identity, copy=True)
    if effects.feed_rotation:
        rotation1 = (
            terms["f_par1"] * terms["par1"]
            + terms["f_el1"] * terms["el1"]
            + (np.pi / 180.0) * terms["phi_off1"]
        )
        rotation2 = (
            terms["f_par2"] * terms["par2"]
            + terms["f_el2"] * terms["el2"]
            + (np.pi / 180.0) * terms["phi_off2"]
        )
        rotation1 = np.array(rotation1, copy=True)
        rotation2 = np.array(rotation2, copy=True)
        rotation1[terms["t1"] == "space"] = 0.0
        rotation2[terms["t2"] == "space"] = 0.0
        jones1[:, 0, 0] = np.exp(-1.0j * rotation1)
        jones1[:, 1, 1] = np.exp(1.0j * rotation1)
        jones2[:, 0, 0] = np.exp(-1.0j * rotation2)
        jones2[:, 1, 1] = np.exp(1.0j * rotation2)
    jones1 = _matrix_term(station_terms, "leakage_matrix1", count) @ jones1
    jones2 = _matrix_term(station_terms, "leakage_matrix2", count) @ jones2
    gain1 = _term_array(station_terms, "common_gain1", count)
    gain2 = _term_array(station_terms, "common_gain2", count)
    return gain1[:, np.newaxis, np.newaxis] * jones1, gain2[:, np.newaxis, np.newaxis] * jones2


def _matrix_term(station_terms, name, count):
    """Return one validated row-aligned two-by-two station matrix term."""

    try:
        values = np.asarray(station_terms[name], dtype=complex)
    except KeyError as exc:
        raise ValueError("Missing station term: {0}.".format(name)) from exc
    if values.shape != (count, 2, 2):
        raise ValueError("Station term {0} must have shape (row, 2, 2).".format(name))
    return values


def _gain_ratio_factor_array(station_terms, count, receptor_count):
    """Return validated row/receptor factors derived from gain ratios."""

    try:
        values = np.asarray(station_terms["gain_ratio_factors"], dtype=complex)
    except KeyError as exc:
        raise ValueError("Missing station term: gain_ratio_factors.") from exc
    if values.shape != (count, receptor_count):
        raise ValueError("gain_ratio_factors must have shape (row, receptor).")
    return values


def receptor_rows_for_station_terms(dataset, station_terms, receptor_configuration, effects):
    """Return effective receptor Jones rows for generic fringe selection.

    The rows include the same simulated station gain, leakage, and feed
    rotation applied by :func:`apply_receptor_corruptions`.  They permit a
    fringe-evidence estimator to reconstruct Stokes I in a common sky basis
    rather than treating a particular recorded feed basis as special.

    Parameters
    ----------
    dataset : VisibilityDataset
        One-channel native dataset whose product layout is being evaluated.
    station_terms : mapping
        Row-aligned native station realization returned by
        :func:`station_terms_for_dataset`.
    receptor_configuration : ReceptorConfiguration
        Resolved station-local receptor paths matching ``dataset.receptors``.
    effects : StationCorruptionModel
        Native station-effect configuration that produced the realization.

    Returns
    -------
    tuple of numpy.ndarray
        Effective first- and second-station Jones rows, each with shape
        ``(row, product_slot, 2)`` in the common circular sky basis.
    """

    if dataset.channel_count != 1:
        raise ValueError("Native receptor fringe selection requires one channel.")
    if not isinstance(effects, StationCorruptionModel):
        raise TypeError("effects must be a StationCorruptionModel instance.")
    response, _, gain_scale = response_rows_for_dataset(dataset, receptor_configuration)
    count = dataset.row_count
    required = (
        "t1", "t2", "f_el1", "f_el2", "f_par1", "f_par2", "el1", "el2",
        "par1", "par2", "phi_off1", "phi_off2",
    )
    terms = {name: _term_array(station_terms, name, count) for name in required}
    jones1, jones2 = _native_station_jones(terms, count, station_terms, effects)
    gain_ratio_factors = _gain_ratio_factor_array(
        station_terms,
        count,
        dataset.receptors.count,
    )
    left = np.zeros((dataset.row_count, dataset.visibilities.shape[2], 2), dtype=complex)
    right = np.zeros_like(left)
    products = dataset.correlation_products
    for row, product_ids in enumerate(dataset.row_product_id):
        for slot, product_id in enumerate(product_ids):
            if product_id < 0:
                continue
            receptor1 = products.receptor1_id[product_id]
            receptor2 = products.receptor2_id[product_id]
            left[row, slot] = (
                gain_ratio_factors[row, receptor1]
                * gain_scale[receptor1]
                * (response[receptor1] @ jones1[row])
            )
            right[row, slot] = (
                gain_ratio_factors[row, receptor2]
                * gain_scale[receptor2]
                * (response[receptor2] @ jones2[row])
            )
    return left, right


def _term_array(station_terms, name, count):
    try:
        values = np.asarray(station_terms[name])
    except KeyError as exc:
        raise ValueError("Missing station term: {0}.".format(name)) from exc
    if values.shape != (count,):
        raise ValueError(
            "Station term {0} must have one value per visibility row.".format(name)
        )
    return values


def _baseline_sigmas(sefd1, sefd2, bandwidth_hz, integration_time_s, tau1, tau2,
                     opacitycal):
    bandwidth_hz = np.asarray(bandwidth_hz, dtype=float)
    if np.any(bandwidth_hz <= 0.0):
        raise ValueError("Station bandwidths must be positive.")
    opacity_factor = np.exp(tau1 + tau2) if opacitycal else 1.0
    sigma = np.sqrt(
        (np.asarray(sefd1, dtype=float) * np.asarray(sefd2, dtype=float) * opacity_factor)
        / (2.0 * bandwidth_hz * integration_time_s)
    ) / const.quant_eff
    return np.broadcast_to(sigma[:, np.newaxis], (len(sigma), 4)).copy()
