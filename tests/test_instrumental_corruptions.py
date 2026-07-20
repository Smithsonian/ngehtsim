"""Measurement-equation regression tests for circular synthetic data."""

import numpy as np
import pytest
import ehtim as eh

import ngehtsim.const_def as const
import ngehtsim.obs.instrumental_corruptions as instrumental_corruptions
import ngehtsim.obs.observation_geometry as observation_geometry
import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.source_models as source_models
import ngehtsim.obs.station_observation as station_observation
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


def _polarized_image():
    return _polarized_model().make_image(160.0 * eh.RADPERUAS, 64)


def _elevation_limited(dataset):
    geometry = observation_geometry.station_geometry_from_rows(
        dataset.stations.position_itrs_m,
        dataset.time_mjd,
        dataset.antenna1,
        dataset.antenna2,
        dataset.ra_hours,
        dataset.dec_degrees,
    )
    mask = (
        (np.rad2deg(geometry.elevation1_rad) > const.el_min)
        & (np.rad2deg(geometry.elevation1_rad) < const.el_max)
        & (np.rad2deg(geometry.elevation2_rad) > const.el_min)
        & (np.rad2deg(geometry.elevation2_rad) < const.el_max)
    )
    return dataset.select_rows(mask)


def _coherency_matrices(obs):
    matrices = np.empty((len(obs.data), 2, 2), dtype=complex)
    matrices[:, 0, 0] = obs.data["rrvis"]
    matrices[:, 0, 1] = obs.data["rlvis"]
    matrices[:, 1, 0] = obs.data["lrvis"]
    matrices[:, 1, 1] = obs.data["llvis"]
    return matrices


def _native_row_order(dataset):
    names = np.asarray(dataset.stations.names)
    return np.lexsort((
        names[dataset.antenna2],
        names[dataset.antenna1],
        dataset.time_mjd,
    ))


def _obsdata_row_order(obs):
    return np.lexsort((obs.data["t2"], obs.data["t1"], obs.data["time"]))


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
    assert clean.ampcal is True
    assert clean.phasecal is True
    assert clean.opacitycal is True
    assert clean.dcal is True
    assert clean.frcal is True
    assert corrupted.ampcal is False
    assert corrupted.phasecal is False
    assert corrupted.opacitycal is True
    assert corrupted.dcal is False
    assert corrupted.frcal is False


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


def test_generator_marks_uncalibrated_opacity_in_output_metadata():
    obs = og.obs_generator(settings=SETTINGS).make_obs(
        _polarized_model(),
        addnoise=False,
        addgains=False,
        opacitycal=False,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert obs.ampcal is True
    assert obs.phasecal is True
    assert obs.opacitycal is False
    assert obs.dcal is True
    assert obs.frcal is True


def _native_dataset_and_terms(generator, image, **kwargs):
    template = observation_geometry.ground_visibility_template(
        generator.arr,
        generator.geometry_context(),
    )
    sampled, F0 = source_models.observe_source_dataset(
        image,
        template,
        generator.source_context(),
    )
    sampled = _elevation_limited(sampled)
    station_keys = (
        "gainamp",
        "leakamp",
        "addgains",
        "addleakage",
        "flagwind",
        "flagday",
        "flagsun",
    )
    station_terms, stations = station_observation.station_terms_for_dataset(
        sampled,
        F0,
        generator.station_context((sampled.time_mjd - generator.mjd) * 24.0),
        generator.rng,
        solar_angle=generator.solar_angle,
        windspeed_sefd_modifier=og.windspeed_SEFD_modification,
        reference_mjd=generator.mjd,
        cache=generator.station_term_cache,
        **{key: kwargs[key] for key in station_keys if key in kwargs},
    )
    return sampled, station_terms, stations


def _native_corrupted_dataset(generator, image, **kwargs):
    sampled, station_terms, stations = _native_dataset_and_terms(generator, image, **kwargs)
    corruption_keys = ("addnoise", "addgains", "opacitycal", "addFR", "addleakage")
    return instrumental_corruptions.apply_circular_corruptions(
        sampled,
        station_terms,
        stations,
        generator.rng,
        **{key: kwargs[key] for key in corruption_keys if key in kwargs},
    )


def test_native_circular_corruptions_match_legacy_generator_without_noise(monkeypatch):
    kwargs = {
        "addnoise": False,
        "addgains": True,
        "gainamp": 0.04,
        "leakamp": 0.1,
        "opacitycal": True,
        "addFR": True,
        "addleakage": True,
        "flagwind": False,
        "flagday": False,
        "flagsun": False,
    }
    native_generator = og.obs_generator(settings=SETTINGS)
    native_generator.rng = FixedRng()
    native = _native_corrupted_dataset(native_generator, _polarized_image(), **kwargs)

    legacy_generator = og.obs_generator(settings=SETTINGS)
    legacy_generator.rng = FixedRng()
    legacy_template = observation_geometry.ground_visibility_template(
        legacy_generator.arr,
        legacy_generator.geometry_context(),
    )
    legacy_empty = _elevation_limited(legacy_template).to_ehtim_obsdata()

    def fixed_template(*args, **ignored_kwargs):
        return legacy_empty, "native-test-template", {}, legacy_empty.copy()

    monkeypatch.setattr(observation_geometry, "observation_template", fixed_template)
    legacy = legacy_generator.observe(_polarized_image(), **kwargs)
    unflagged = native.select_rows(~np.any(native.flags[:, 0, :], axis=1))
    native_order = _native_row_order(unflagged)
    legacy_order = _obsdata_row_order(legacy)
    native_names = np.asarray(unflagged.stations.names)
    native_visibilities = unflagged.visibilities[:, 0, :]
    native_sigma = 1.0 / np.sqrt(unflagged.weights[:, 0, :])
    legacy_visibilities = np.column_stack((
        legacy.data["rrvis"],
        legacy.data["llvis"],
        legacy.data["rlvis"],
        legacy.data["lrvis"],
    ))
    legacy_sigma = np.column_stack((
        legacy.data["rrsigma"],
        legacy.data["llsigma"],
        legacy.data["rlsigma"],
        legacy.data["lrsigma"],
    ))

    assert np.array_equal(
        native_names[unflagged.antenna1][native_order],
        legacy.data["t1"][legacy_order],
    )
    assert np.array_equal(
        native_names[unflagged.antenna2][native_order],
        legacy.data["t2"][legacy_order],
    )
    assert np.allclose(
        (unflagged.time_mjd[native_order] - legacy.mjd) * 24.0,
        legacy.data["time"][legacy_order],
        atol=1.0e-12,
    )
    assert np.allclose(
        unflagged.integration_time_s[native_order],
        legacy.data["tint"][legacy_order],
        atol=1.0e-12,
    )
    assert np.allclose(unflagged.tau1[native_order], legacy.data["tau1"][legacy_order], atol=1.0e-12)
    assert np.allclose(unflagged.tau2[native_order], legacy.data["tau2"][legacy_order], atol=1.0e-12)
    assert np.allclose(native_visibilities[native_order], legacy_visibilities[legacy_order], atol=1.0e-12)
    assert np.allclose(native_sigma[native_order], legacy_sigma[legacy_order], atol=1.0e-12)
    assert unflagged.ampcal is legacy.ampcal is False
    assert unflagged.phasecal is legacy.phasecal is False
    assert unflagged.opacitycal is legacy.opacitycal is True
    assert unflagged.dcal is legacy.dcal is False
    assert unflagged.frcal is legacy.frcal is False


def test_native_circular_corruptions_do_not_require_obsdata_conversion(monkeypatch):
    generator = og.obs_generator(settings=SETTINGS)
    sampled, station_terms, stations = _native_dataset_and_terms(
        generator,
        _polarized_image(),
        addgains=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    def unexpected_obsdata_conversion(*args, **kwargs):
        raise AssertionError("Native corruptions must not construct an Obsdata object.")

    monkeypatch.setattr(
        VisibilityDataset,
        "to_ehtim_obsdata",
        unexpected_obsdata_conversion,
    )
    corrupted = instrumental_corruptions.apply_circular_corruptions(
        sampled,
        station_terms,
        stations,
        generator.rng,
        addnoise=False,
        addgains=False,
        opacitycal=False,
        addFR=False,
        addleakage=False,
    )

    assert isinstance(corrupted, VisibilityDataset)
    assert corrupted.opacitycal is False
    assert np.all(np.isfinite(corrupted.weights))


def test_native_circular_noise_uses_reported_sigmas():
    clean_generator = og.obs_generator(settings=SETTINGS)
    clean_generator.rng = FixedRng()
    clean = _native_corrupted_dataset(
        clean_generator,
        _polarized_image(),
        addnoise=False,
        addgains=True,
        gainamp=0.04,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )
    noisy_generator = og.obs_generator(settings=SETTINGS)
    noisy_generator.rng = FixedRng()
    noisy = _native_corrupted_dataset(
        noisy_generator,
        _polarized_image(),
        addnoise=True,
        addgains=True,
        gainamp=0.04,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    sigma = 1.0 / np.sqrt(noisy.weights)
    assert np.allclose(noisy.weights, clean.weights)
    assert np.allclose(noisy.visibilities - clean.visibilities, (1.0 + 1.0j) * sigma)


def test_native_circular_corruptions_flag_rows_and_support_selection():
    generator = og.obs_generator(settings=SETTINGS)
    sampled, station_terms, stations = _native_dataset_and_terms(
        generator,
        _polarized_image(),
        addgains=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )
    station_terms = dict(station_terms)
    station_terms["uptime_mask"] = np.ones(sampled.row_count, dtype=bool)
    station_terms["uptime_mask"][:2] = False
    station_terms["flagsites"] = [station_terms["t1"][2]]

    corrupted = instrumental_corruptions.apply_circular_corruptions(
        sampled,
        station_terms,
        stations,
        generator.rng,
        addnoise=False,
        addgains=False,
        opacitycal=True,
        addFR=False,
        addleakage=False,
    )

    expected_flagged = (
        ~station_terms["uptime_mask"]
        | (station_terms["t1"] == station_terms["flagsites"][0])
        | (station_terms["t2"] == station_terms["flagsites"][0])
    )
    assert np.array_equal(np.all(corrupted.flags[:, 0, :], axis=1), expected_flagged)
    selected = corrupted.select_rows(~expected_flagged)
    assert selected.row_count == np.count_nonzero(~expected_flagged)
    assert not np.any(selected.flags)


def test_mixed_basis_output_is_rejected_until_native_support_exists():
    obsgen = og.obs_generator(settings=SETTINGS)

    with pytest.raises(NotImplementedError, match="Mixed-polarization output"):
        obsgen.make_obs(_polarized_model(), allow_mixed_basis=True)


def test_station_registry_marks_roen_as_linear():
    assert const.known_polbases["ROEN"] == "linear"
