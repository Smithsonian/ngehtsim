"""Measurement-equation regression tests for circular synthetic data."""

import numpy as np
import pytest
import ehtim as eh

import ngehtsim.const_def as const
import ngehtsim.obs.instrumental_corruptions as instrumental_corruptions
import ngehtsim.obs.obs_generator as og
from ngehtsim.obs.receptor_configuration import resolve_receptor_configuration
from ngehtsim.obs.station_effects import GainModel, LeakageModel, StationCorruptionModel
from ngehtsim.obs.visibility_dataset import VisibilityDataset


SETTINGS = {
    "source": "M87",
    "sites": ["ALMA", "APEX", "LMT", "SMT"],
    "weather": "typical",
    "t_start": 0.0,
    "dt": 2.0,
    "t_int": 600.0,
    "t_rest": 1200.0,
    "fringe_finder": ["naive", 0.0],
    "random_seed": 1,
    "transform_backend": "direct",
}


class FixedRng:
    """Return deterministic draws that expose visibility/sigma mismatches."""

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return 1.0
        return np.ones(size, dtype=float)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return 0.0
        return np.zeros(size, dtype=float)

    def choice(self, values, size=None, replace=True, p=None):
        return np.zeros(size, dtype=int)


def _effects(*, thermal_noise=False, common_gain=False, opacity_calibrated=True,
             feed_rotation=False, leakage=False, flag_wind=False,
             flag_daylight=False, flag_sun=False):
    """Build native station effects without legacy simulation keywords."""

    return StationCorruptionModel(
        thermal_noise=thermal_noise,
        opacity_calibrated=opacity_calibrated,
        feed_rotation=feed_rotation,
        common_gain=(
            GainModel(amplitude_sigma_dex=0.04, phase_distribution="uniform")
            if common_gain else None
        ),
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


def _polarized_image():
    return _polarized_model().make_image(160.0 * eh.RADPERUAS, 64)


def _station_jones(common_gain, leakage, feed_rotation):
    jones = np.zeros((len(feed_rotation), 2, 2), dtype=complex)
    jones[:, 0, 0] = np.exp(-1.0j * feed_rotation)
    jones[:, 1, 1] = np.exp(1.0j * feed_rotation)
    return common_gain[:, np.newaxis, np.newaxis] * leakage @ jones


def test_circular_leakage_matches_jones_matrix():
    visibilities = np.array(
        [
            [1.2 + 0.4j, 0.5 + 0.1j, -0.3 + 0.7j, 0.9 - 0.2j],
            [0.7 - 0.5j, -0.1 + 0.3j, 0.4 + 0.6j, 0.2 - 0.8j],
        ]
    )
    leakage1_r = np.array([0.13 - 0.08j, -0.06 + 0.04j])
    leakage1_l = np.array([-0.05 + 0.11j, 0.08 + 0.02j])
    leakage2_r = np.array([-0.09 + 0.06j, 0.03 - 0.07j])
    leakage2_l = np.array([0.07 + 0.04j, -0.02 + 0.09j])

    actual = instrumental_corruptions.apply_circular_leakage(
        visibilities,
        leakage1_r,
        leakage1_l,
        leakage2_r,
        leakage2_l,
    )
    coherency = np.empty((len(visibilities), 2, 2), dtype=complex)
    coherency[:, 0, 0] = visibilities[:, 0]
    coherency[:, 0, 1] = visibilities[:, 2]
    coherency[:, 1, 0] = visibilities[:, 3]
    coherency[:, 1, 1] = visibilities[:, 1]
    d1 = np.zeros_like(coherency)
    d2 = np.zeros_like(coherency)
    d1[:, 0, 0] = d1[:, 1, 1] = 1.0
    d2[:, 0, 0] = d2[:, 1, 1] = 1.0
    d1[:, 0, 1] = leakage1_r
    d1[:, 1, 0] = leakage1_l
    d2[:, 0, 1] = leakage2_r
    d2[:, 1, 0] = leakage2_l
    expected_coherency = d1 @ coherency @ np.swapaxes(np.conj(d2), -1, -2)
    expected = np.column_stack((
        expected_coherency[:, 0, 0],
        expected_coherency[:, 1, 1],
        expected_coherency[:, 0, 1],
        expected_coherency[:, 1, 0],
    ))

    assert np.allclose(actual, expected, atol=1.0e-12)
    assert np.allclose(visibilities, np.array([
        [1.2 + 0.4j, 0.5 + 0.1j, -0.3 + 0.7j, 0.9 - 0.2j],
        [0.7 - 0.5j, -0.1 + 0.3j, 0.4 + 0.6j, 0.2 - 0.8j],
    ]))


def test_circular_generator_matches_composed_station_jones_matrices():
    generator = og.obs_generator(settings=SETTINGS, weight=1)
    effects = _effects(common_gain=True, feed_rotation=True, leakage=True)
    simulated = generator.simulate(
        _polarized_model(),
        effects=effects,
    )
    dataset = simulated.dataset
    terms = simulated.station_terms
    sky_coherency = np.empty((dataset.row_count, 2, 2), dtype=complex)
    row_index = np.arange(dataset.row_count, dtype=float)
    sky_coherency[:, 0, 0] = 1.0 + (0.01 * row_index) + 0.1j
    sky_coherency[:, 0, 1] = -0.2 + 0.3j
    sky_coherency[:, 1, 0] = 0.15 - 0.25j
    sky_coherency[:, 1, 1] = 0.8 - (0.02 * row_index) - 0.05j
    corrupted = instrumental_corruptions.apply_receptor_corruptions(
        dataset,
        sky_coherency,
        terms,
        dataset.stations,
        resolve_receptor_configuration(
            dataset.stations.names,
            generator.station_receptors,
            generator.station_signal_paths,
        ),
        effects,
        generator.rng,
    )

    jones1 = _station_jones(
        terms["common_gain1"],
        terms["leakage_matrix1"],
        (terms["f_par1"] * terms["par1"])
        + (terms["f_el1"] * terms["el1"])
        + ((np.pi / 180.0) * terms["phi_off1"]),
    )
    jones2 = _station_jones(
        terms["common_gain2"],
        terms["leakage_matrix2"],
        (terms["f_par2"] * terms["par2"])
        + (terms["f_el2"] * terms["el2"])
        + ((np.pi / 180.0) * terms["phi_off2"]),
    )
    expected = (
        jones1
        @ sky_coherency
        @ np.swapaxes(np.conj(jones2), -1, -2)
    )

    corrupted_visibility = corrupted.visibilities[:, 0]
    corrupted_coherency = np.empty_like(expected)
    corrupted_coherency[:, 0, 0] = corrupted_visibility[:, 0]
    corrupted_coherency[:, 0, 1] = corrupted_visibility[:, 2]
    corrupted_coherency[:, 1, 0] = corrupted_visibility[:, 3]
    corrupted_coherency[:, 1, 1] = corrupted_visibility[:, 1]
    assert np.allclose(corrupted_coherency, expected, atol=1.0e-12)
    assert corrupted.ampcal is False
    assert corrupted.phasecal is False
    assert corrupted.opacitycal is True
    assert corrupted.dcal is False
    assert corrupted.frcal is False


def test_thermal_noise_uses_reported_gain_corrupted_sigmas():
    clean_generator = og.obs_generator(settings=SETTINGS)
    clean_generator.rng = FixedRng()
    clean = clean_generator.simulate(
        _polarized_model(),
        effects=_effects(common_gain=True),
    )
    noisy_generator = og.obs_generator(settings=SETTINGS)
    noisy_generator.rng = FixedRng()
    noisy = noisy_generator.simulate(
        _polarized_model(),
        effects=_effects(thermal_noise=True, common_gain=True),
    )

    clean_sigma = clean.dataset.sigma_jy
    noisy_sigma = noisy.dataset.sigma_jy
    assert np.allclose(noisy_sigma, clean_sigma)
    assert np.allclose(
        noisy.dataset.visibilities - clean.dataset.visibilities,
        (1.0 + 1.0j) * noisy_sigma,
    )


def test_generator_marks_uncalibrated_opacity_in_output_metadata():
    obs = og.obs_generator(settings=SETTINGS).make_obs(
        _polarized_model(),
        effects=_effects(opacity_calibrated=False),
    )

    assert obs.ampcal is True
    assert obs.phasecal is True
    assert obs.opacitycal is False
    assert obs.dcal is True
    assert obs.frcal is True


def test_native_receptor_corruptions_do_not_require_obsdata_conversion(monkeypatch):
    def unexpected_obsdata_conversion(*args, **kwargs):
        raise AssertionError("Native corruptions must not construct an Obsdata object.")

    monkeypatch.setattr(VisibilityDataset, "to_ehtim_obsdata", unexpected_obsdata_conversion)
    result = og.obs_generator(settings=SETTINGS).simulate(
        _polarized_image(),
        effects=_effects(opacity_calibrated=False),
    )

    assert isinstance(result.dataset, VisibilityDataset)
    assert result.dataset.opacitycal is False
    assert np.all(np.isfinite(result.dataset.sigma_jy))


def test_native_effects_reject_legacy_corruption_keywords():
    obsgen = og.obs_generator(settings=SETTINGS)

    with pytest.raises(TypeError, match="unexpected keyword argument 'addnoise'"):
        obsgen.simulate(_polarized_model(), addnoise=False)
    with pytest.raises(TypeError, match=r"Native observe\(\) accepts effects"):
        obsgen.observe(_polarized_model(), addnoise=False)
    with pytest.raises(TypeError, match="only supported by the native"):
        obsgen.make_obs(
            _polarized_model(),
            effects=_effects(),
            backend="legacy",
        )


def test_station_registry_marks_roen_as_linear():
    assert const.known_polbases["ROEN"] == "linear"
