"""Measurement-equation regression tests for circular synthetic data."""

import numpy as np
import pytest
import ehtim as eh

import ngehtsim.const_def as const
import ngehtsim.obs.instrumental_corruptions as instrumental_corruptions
import ngehtsim.obs.obs_generator as og


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


def _polarized_model():
    model = eh.model.Model()
    return model.add_circ_gauss(
        F0=1.0,
        FWHM=40.0 * eh.RADPERUAS,
        pol_frac=0.25,
        pol_evpa=30.0 * eh.DEGREE,
        cpol_frac=0.10,
    )


def _coherency_matrices(obs):
    matrices = np.empty((len(obs.data), 2, 2), dtype=complex)
    matrices[:, 0, 0] = obs.data["rrvis"]
    matrices[:, 0, 1] = obs.data["rlvis"]
    matrices[:, 1, 0] = obs.data["lrvis"]
    matrices[:, 1, 1] = obs.data["llvis"]
    return matrices


def _station_jones(gain_r, gain_l, leakage_r, leakage_l, feed_rotation):
    jones = np.empty((len(feed_rotation), 2, 2), dtype=complex)
    jones[:, 0, 0] = gain_r * np.exp(-1.0j * feed_rotation)
    jones[:, 0, 1] = gain_r * leakage_r * np.exp(1.0j * feed_rotation)
    jones[:, 1, 0] = gain_l * leakage_l * np.exp(-1.0j * feed_rotation)
    jones[:, 1, 1] = gain_l * np.exp(1.0j * feed_rotation)
    return jones


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
    clean_generator = og.obs_generator(settings=SETTINGS)
    clean = clean_generator.make_obs(
        _polarized_model(),
        addnoise=False,
        addgains=False,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )
    corrupted_generator = og.obs_generator(settings=SETTINGS, weight=1)
    corrupted = corrupted_generator.make_obs(
        _polarized_model(),
        addnoise=False,
        addgains=True,
        addFR=True,
        addleakage=True,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert np.array_equal(corrupted.data["t1"], clean.data["t1"])
    assert np.array_equal(corrupted.data["t2"], clean.data["t2"])
    assert np.allclose(corrupted.data["time"], clean.data["time"])

    jones1 = _station_jones(
        corrupted_generator.station_gains1R,
        corrupted_generator.station_gains1L,
        corrupted_generator.station_leakage1R,
        corrupted_generator.station_leakage1L,
        corrupted_generator.fa_1,
    )
    jones2 = _station_jones(
        corrupted_generator.station_gains2R,
        corrupted_generator.station_gains2L,
        corrupted_generator.station_leakage2R,
        corrupted_generator.station_leakage2L,
        corrupted_generator.fa_2,
    )
    expected = jones1 @ _coherency_matrices(clean) @ np.swapaxes(np.conj(jones2), -1, -2)

    assert np.allclose(_coherency_matrices(corrupted), expected, atol=1.0e-12)


def test_thermal_noise_uses_reported_gain_corrupted_sigmas():
    clean_generator = og.obs_generator(settings=SETTINGS)
    clean_generator.rng = FixedRng()
    clean = clean_generator.make_obs(
        _polarized_model(),
        addnoise=False,
        addgains=True,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )
    noisy_generator = og.obs_generator(settings=SETTINGS)
    noisy_generator.rng = FixedRng()
    noisy = noisy_generator.make_obs(
        _polarized_model(),
        addnoise=True,
        addgains=True,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    for visibility_field, sigma_field in (
        ("rrvis", "rrsigma"),
        ("llvis", "llsigma"),
        ("rlvis", "rlsigma"),
        ("lrvis", "lrsigma"),
    ):
        assert np.allclose(noisy.data[sigma_field], clean.data[sigma_field])
        assert np.allclose(
            noisy.data[visibility_field] - clean.data[visibility_field],
            (1.0 + 1.0j) * noisy.data[sigma_field],
        )


def test_mixed_basis_output_is_rejected_until_native_support_exists():
    obsgen = og.obs_generator(settings=SETTINGS)

    with pytest.raises(NotImplementedError, match="Mixed-polarization output"):
        obsgen.make_obs(_polarized_model(), allow_mixed_basis=True)


def test_station_registry_marks_roen_as_linear():
    assert const.known_polbases["ROEN"] == "linear"
