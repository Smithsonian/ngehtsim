"""Regression tests for native mixed-receptor simulation and fringe evidence."""

from __future__ import annotations

import numpy as np
import ehtim as eh

import ngehtsim.obs.fringe_selection as fringe_selection
import ngehtsim.obs.instrumental_corruptions as instrumental_corruptions
import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.source_models as source_models
from ngehtsim.obs.receptor_configuration import (
    STANDARD_JONES_ROWS,
    resolve_receptor_configuration,
)


SETTINGS = {
    "source": "M87",
    "sites": ["ALMA", "APEX", "LMT", "SMT"],
    "weather": "typical",
    "t_start": 0.0,
    "dt": 1.0,
    "t_int": 600.0,
    "t_rest": 1200.0,
    "fringe_finder": ["fringegroups", [0.0, 10.0]],
    "random_seed": 1,
    "transform_backend": "direct",
}


MIXED_RECEPTORS = {
    "ALMA": ("X", "Y"),
    "APEX": ("R", "L"),
    "LMT": ("Y",),
    "SMT": ("R", "X", "Y"),
}


def _polarized_model():
    model = eh.model.Model()
    return model.add_circ_gauss(
        F0=1.0,
        FWHM=40.0 * eh.RADPERUAS,
        pol_frac=0.25,
        pol_evpa=30.0 * eh.DEGREE,
        cpol_frac=0.10,
    )


def _product_snr(left, right, coherency, sigma=1.0):
    values = np.array([
        first @ coherency @ np.conj(second)
        for first in left for second in right
    ], dtype=complex)[np.newaxis, :]
    sigma_jy = np.full(values.shape, sigma, dtype=float)
    left_rows = np.array([
        first for first in left for _ in right
    ], dtype=complex)[np.newaxis, :, :]
    right_rows = np.array([
        second for _ in left for second in right
    ], dtype=complex)[np.newaxis, :, :]
    return fringe_selection.receptor_fringe_snr(
        values,
        sigma_jy,
        left_rows,
        right_rows,
        np.ones(values.shape, dtype=bool),
    )[0]


def test_stokes_i_fringe_evidence_is_invariant_under_feed_basis_conversion():
    coherency = np.array(
        [[1.2 + 0.3j, -0.25 + 0.4j], [0.1 - 0.2j, 0.7 - 0.15j]],
        dtype=complex,
    )
    circular = (STANDARD_JONES_ROWS["R"], STANDARD_JONES_ROWS["L"])
    linear = (STANDARD_JONES_ROWS["X"], STANDARD_JONES_ROWS["Y"])

    circular_snr = _product_snr(circular, circular, coherency)
    linear_snr = _product_snr(linear, linear, coherency)
    mixed_snr = _product_snr(linear, circular, coherency)

    assert np.allclose((circular_snr, linear_snr, mixed_snr), circular_snr)


def test_rank_deficient_fringe_evidence_uses_strongest_product():
    coherency = np.array([[1.0 + 0.5j, 0.2j], [0.3, 0.5 - 0.1j]], dtype=complex)
    left = (STANDARD_JONES_ROWS["Y"],)
    right = (STANDARD_JONES_ROWS["R"],)
    expected = np.abs(left[0] @ coherency @ np.conj(right[0])) / 2.0

    assert _product_snr(left, right, coherency, sigma=2.0) == expected


def test_standard_circular_rime_matches_the_legacy_circular_kernel():
    """The generic Jones path retains the established circular calculation."""

    generator = og.obs_generator(settings=SETTINGS)
    result = generator.simulate(
        _polarized_model(),
        addnoise=False,
        addgains=True,
        addFR=True,
        addleakage=True,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )
    sampled, _ = source_models.observe_source_dataset(
        _polarized_model(),
        result.dataset,
        generator.source_context(),
    )
    legacy = instrumental_corruptions.apply_circular_corruptions(
        sampled,
        result.station_terms,
        result.dataset.stations,
        np.random.default_rng(2),
        addnoise=False,
        addgains=True,
        addFR=True,
        addleakage=True,
    )

    assert np.allclose(result.dataset.visibilities, legacy.visibilities, atol=1.0e-12)
    assert np.allclose(result.dataset.sigma_jy, legacy.sigma_jy, atol=1.0e-12)
    assert np.array_equal(result.dataset.flags, legacy.flags)


def test_mixed_receptor_rime_matches_explicit_effective_jones_rows():
    generator = og.obs_generator(
        settings=SETTINGS,
        station_receptors=MIXED_RECEPTORS,
    )
    result = generator.simulate(
        _polarized_model(),
        addnoise=False,
        addgains=True,
        addFR=True,
        addleakage=True,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )
    dataset = result.dataset
    configuration = resolve_receptor_configuration(
        dataset.stations.names,
        generator.station_receptors,
        generator.station_signal_paths,
    )
    _, _, sky_coherency = source_models.observe_source_dataset(
        _polarized_model(),
        dataset,
        generator.source_context(),
        return_coherency=True,
    )
    left, right = instrumental_corruptions.receptor_rows_for_station_terms(
        dataset,
        result.station_terms,
        configuration,
    )

    expected = np.zeros_like(dataset.visibilities[:, 0])
    for row, product_ids in enumerate(dataset.row_product_id):
        for slot, product_id in enumerate(product_ids):
            if product_id >= 0:
                expected[row, slot] = left[row, slot] @ sky_coherency[row] @ np.conj(right[row, slot])

    assert dataset.receptors.polarization_label == ("X", "Y", "R", "L", "Y", "R", "X", "Y")
    assert np.allclose(dataset.visibilities[:, 0], expected, atol=1.0e-12)


def test_mixed_receptor_fringegroups_uses_generic_stokes_i_evidence():
    generator = og.obs_generator(
        settings=SETTINGS,
        station_receptors=MIXED_RECEPTORS,
    )
    result = generator.make_dataset(
        _polarized_model(),
        addnoise=False,
        addgains=False,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert result.dataset.row_count > 0
    assert np.any(~result.dataset.flags)


def test_direct_mixed_fringegroups_dataset_infers_standard_feed_responses():
    """The public helper must not require private simulator state for R/L/X/Y."""

    generator = og.obs_generator(
        settings=SETTINGS,
        station_receptors=MIXED_RECEPTORS,
    )
    result = generator.make_dataset(
        _polarized_model(),
        addnoise=False,
        addgains=False,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    direct = og.fringegroups_dataset(generator, result.dataset, 0.0, 10.0)
    with_terms = og.fringegroups_dataset(
        generator,
        result.dataset,
        0.0,
        10.0,
        station_terms=result.station_terms,
        receptor_configuration=resolve_receptor_configuration(
            result.dataset.stations.names,
            generator.station_receptors,
            generator.station_signal_paths,
        ),
    )

    assert np.array_equal(direct, with_terms)


def test_direct_uncalibrated_mixed_fringegroups_requires_station_terms():
    """Unknown Jones terms must not be silently ignored by Stokes-I recovery."""

    generator = og.obs_generator(
        settings=SETTINGS,
        station_receptors=MIXED_RECEPTORS,
    )
    result = generator.make_dataset(
        _polarized_model(),
        addnoise=False,
        addgains=True,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "Mixed-receptor fringe selection needs station_terms",
    ):
        og.fringegroups_dataset(generator, result.dataset, 0.0, 10.0)


def test_mixed_receptor_fpt_uses_the_same_generic_fringe_evidence():
    settings = dict(SETTINGS)
    settings["fringe_finder"] = ["fpt", [0.0, 10.0, 86.0, None]]
    generator = og.obs_generator(
        settings=settings,
        station_receptors=MIXED_RECEPTORS,
    )
    result = generator.make_dataset(
        _polarized_model(),
        addnoise=False,
        addgains=False,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert result.dataset.row_count > 0
    assert np.any(~result.dataset.flags)
