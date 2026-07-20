"""Pure circular-basis visibility corruption kernels."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import ngehtsim.const_def as const
from ngehtsim.obs.visibility_dataset import CIRCULAR_CORRELATIONS, StationTable, VisibilityDataset


def apply_circular_leakage(visibilities, leakage1_r, leakage1_l, leakage2_r,
                           leakage2_l):
    """Apply ``D_1 V D_2^H`` to circular RR, LL, RL, LR visibilities.

    The final axis of ``visibilities`` must use the order ``RR, LL, RL, LR``.
    The result is a new array so every output correlation is calculated from
    the same, unmodified input coherency matrix.
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
    """

    if not isinstance(dataset, VisibilityDataset):
        raise TypeError("dataset must be a VisibilityDataset instance.")
    if not isinstance(stations, StationTable):
        raise TypeError("stations must be a StationTable instance.")
    if dataset.channel_count != 1:
        raise ValueError("Circular corruption currently requires exactly one channel.")
    if any(
        dataset.correlation_layouts[index] != CIRCULAR_CORRELATIONS
        for index in dataset.row_layout_id
    ):
        raise ValueError(
            "Circular corruption requires RR, LL, RL, LR correlations for every row."
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

    visibilities = np.array(dataset.visibilities, copy=True)
    circular = visibilities[:, 0, :]
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
    flags[:, 0, :] |= ~row_mask[:, np.newaxis]
    weights = np.array(dataset.weights, copy=True)
    weights[:, 0, :] = 1.0 / np.square(sigma)
    return replace(
        dataset,
        stations=stations,
        tau1=tau1,
        tau2=tau2,
        visibilities=visibilities,
        weights=weights,
        flags=flags,
        ampcal=not addgains,
        phasecal=not addgains,
        opacitycal=opacitycal,
        dcal=not addleakage,
        frcal=not addFR,
    )


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
