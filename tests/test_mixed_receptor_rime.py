"""Regression tests for native mixed-receptor simulation and fringe evidence."""

from __future__ import annotations

from dataclasses import replace

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
from ngehtsim.obs.station_effects import (
    GainModel,
    GainRatioModel,
    LeakageModel,
    StationCorruptionModel,
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


def _effects(*, thermal_noise=False, station_gain=False, feed_rotation=False,
             leakage=False, flag_wind=False, flag_daylight=False, flag_sun=False):
    """Build the compact native effect configurations used in this module."""

    return StationCorruptionModel(
        thermal_noise=thermal_noise,
        station_gain=(
            GainModel(amplitude_sigma_dex=0.04, phase_distribution="uniform")
            if station_gain else None
        ),
        feed_rotation=feed_rotation,
        leakage=LeakageModel(component_sigma=0.1) if leakage else None,
        flag_wind=flag_wind,
        flag_daylight=flag_daylight,
        flag_sun=flag_sun,
    )


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


def test_standard_circular_rime_uses_generic_station_terms():
    """Native circular output no longer exposes hand-specific state fields."""

    generator = og.obs_generator(settings=SETTINGS)
    result = generator.simulate(
        _polarized_model(),
        effects=_effects(station_gain=True, feed_rotation=True, leakage=True),
    )
    assert {
        "common_gain1",
        "common_gain2",
        "leakage_feed_matrix1",
        "leakage_feed_matrix2",
        "gain_ratio_factors",
    } <= set(result.station_terms)
    assert not any(name.startswith(("gainamp", "gainphase", "leak1", "leak2")) for name in result.station_terms)


def test_mixed_receptor_rime_matches_explicit_effective_jones_rows():
    generator = og.obs_generator(
        settings=SETTINGS,
        station_receptors=MIXED_RECEPTORS,
    )
    effects = StationCorruptionModel(
        thermal_noise=False,
        station_gain=GainModel(
            amplitude_sigma_dex=0.04,
            phase_distribution="uniform",
        ),
        feed_rotation=True,
        leakage_overrides={
            "ALMA": LeakageModel(component_sigma=0.1),
            "APEX": LeakageModel(component_sigma=0.1),
        },
        gain_ratio_overrides={
            "ALMA": GainRatioModel("X", "Y", amplitude_sigma_dex=0.02),
        },
        flag_wind=False,
        flag_daylight=False,
        flag_sun=False,
    )
    result = generator.simulate(
        _polarized_model(),
        effects=effects,
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
        effects,
    )

    expected = np.zeros_like(dataset.visibilities[:, 0])
    for row, product_ids in enumerate(dataset.row_product_id):
        for slot, product_id in enumerate(product_ids):
            if product_id >= 0:
                expected[row, slot] = left[row, slot] @ sky_coherency[row] @ np.conj(right[row, slot])

    assert dataset.receptors.polarization_label == ("X", "Y", "R", "L", "Y", "R", "X", "Y")
    assert np.allclose(dataset.visibilities[:, 0], expected, atol=1.0e-12)
    alma_x = next(
        index
        for index, (station, feed_id) in enumerate(zip(
            dataset.receptors.station_index,
            dataset.receptors.feed_id,
        ))
        if dataset.stations.names[station] == "ALMA" and feed_id == "X"
    )
    alma_y = next(
        index
        for index, (station, feed_id) in enumerate(zip(
            dataset.receptors.station_index,
            dataset.receptors.feed_id,
        ))
        if dataset.stations.names[station] == "ALMA" and feed_id == "Y"
    )
    gain_ratios = result.station_terms["gain_ratio_factors"]
    assert not np.allclose(gain_ratios[:, alma_x], 1.0)
    assert np.allclose(gain_ratios[:, alma_x] * gain_ratios[:, alma_y], 1.0)


def test_local_feed_leakage_matches_explicit_mixed_basis_jones_matrices():
    """X/Y and R/L stations must receive independent local-feed D-terms."""

    settings = dict(SETTINGS)
    settings["sites"] = ["ALMA", "APEX"]
    receptor_layout = {"ALMA": ("X", "Y"), "APEX": ("R", "L")}
    d_alma = np.array(((1.0, 0.02 + 0.03j), (-0.04 + 0.01j, 1.0)), dtype=complex)
    d_apex = np.array(((1.0, -0.05 + 0.02j), (0.06 - 0.01j, 1.0)), dtype=complex)
    effects = StationCorruptionModel(
        thermal_noise=False,
        station_gain=None,
        feed_rotation=False,
        leakage_overrides={
            "ALMA": LeakageModel(
                "X",
                "Y",
                leakage_a_mean=d_alma[0, 1],
                leakage_b_mean=d_alma[1, 0],
            ),
            "APEX": LeakageModel(
                "R",
                "L",
                leakage_a_mean=d_apex[0, 1],
                leakage_b_mean=d_apex[1, 0],
            ),
        },
        flag_wind=False,
        flag_daylight=False,
        flag_sun=False,
    )
    generator = og.obs_generator(settings=settings, station_receptors=receptor_layout)
    result = generator.simulate(_polarized_model(), effects=effects)
    dataset = result.dataset
    configuration = resolve_receptor_configuration(dataset.stations.names, receptor_layout)
    _, _, sky_coherency = source_models.observe_source_dataset(
        _polarized_model(),
        dataset,
        generator.source_context(),
        return_coherency=True,
    )
    response = configuration.response_circular
    station_receptors = {
        station: np.flatnonzero(dataset.receptors.station_index == station_index)
        for station_index, station in enumerate(dataset.stations.names)
    }
    local_index = {
        receptor: index
        for indices in station_receptors.values()
        for index, receptor in enumerate(indices)
    }
    local_d = {"ALMA": d_alma, "APEX": d_apex}
    expected = np.zeros_like(dataset.visibilities[:, 0])
    products = dataset.correlation_products
    for row, product_ids in enumerate(dataset.row_product_id):
        first_station = dataset.stations.names[dataset.antenna1[row]]
        second_station = dataset.stations.names[dataset.antenna2[row]]
        first_indices = station_receptors[first_station]
        second_indices = station_receptors[second_station]
        first_rows = local_d[first_station] @ response[first_indices]
        second_rows = local_d[second_station] @ response[second_indices]
        for slot, product_id in enumerate(product_ids):
            if product_id < 0:
                continue
            receptor1 = products.receptor1_id[product_id]
            receptor2 = products.receptor2_id[product_id]
            expected[row, slot] = (
                first_rows[local_index[receptor1]]
                @ sky_coherency[row]
                @ np.conj(second_rows[local_index[receptor2]])
            )

    for endpoint, matrix_name in (("t1", "leakage_feed_matrix1"), ("t2", "leakage_feed_matrix2")):
        endpoint_stations = np.asarray(result.station_terms[endpoint])
        for station, leakage_matrix in local_d.items():
            mask = endpoint_stations == station
            if np.any(mask):
                assert np.allclose(result.station_terms[matrix_name][mask], leakage_matrix)
    assert np.allclose(dataset.visibilities[:, 0], expected, atol=1.0e-12)


def test_mixed_receptor_fringegroups_uses_generic_stokes_i_evidence():
    generator = og.obs_generator(
        settings=SETTINGS,
        station_receptors=MIXED_RECEPTORS,
    )
    result = generator.make_dataset(
        _polarized_model(),
        effects=_effects(),
    )

    assert result.dataset.row_count > 0
    assert np.any(~result.dataset.flags)


def test_mixed_receptor_fringegroups_subsets_row_aligned_jones_terms():
    """Fringe selection must subset Jones matrices with unavailable rows."""

    settings = dict(SETTINGS)
    settings["sites"] = ["ALMA", "APEX", "LMT"]
    generator = og.obs_generator(
        settings=settings,
        station_receptors={
            "ALMA": ("X", "Y"),
            "APEX": ("R", "L"),
            "LMT": ("Y",),
        },
    )
    effects = _effects(station_gain=True, feed_rotation=True, leakage=True)
    raw = generator.simulate(_polarized_model(), effects=effects)

    assert raw.dataset.row_count > 1
    flags = np.array(raw.dataset.flags, copy=True)
    flags[0] = True
    selection = generator._native_selection_mask(
        replace(raw.dataset, flags=flags),
        raw.station_terms,
        effects,
    )

    assert selection.shape == (raw.dataset.row_count,)
    assert not selection[0]
    assert np.any(selection[1:])


def test_direct_mixed_fringegroups_dataset_infers_standard_feed_responses():
    """The public helper must not require private simulator state for R/L/X/Y."""

    generator = og.obs_generator(
        settings=SETTINGS,
        station_receptors=MIXED_RECEPTORS,
    )
    result = generator.make_dataset(
        _polarized_model(),
        effects=_effects(),
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
        effects=_effects(),
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
        effects=_effects(station_gain=True),
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
        effects=_effects(),
    )

    assert result.dataset.row_count > 0
    assert np.any(~result.dataset.flags)
