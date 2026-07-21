import numpy as np
import pytest
import ehtim as eh

import ngehtsim.const_def as const
from ngehtsim.obs.obs_generator import obs_generator
import ngehtsim.obs.observation_geometry as observation_geometry
from ngehtsim.obs.obs_generator import make_array
import ngehtsim.obs.source_models as source_models
from ngehtsim.obs.visibility_dataset import CIRCULAR_CORRELATIONS, VisibilityDataset


@pytest.fixture
def array_and_context():
    obsgen = obs_generator(settings={"weather": "exact", "source": "M87"})
    return obsgen.arr, obsgen.geometry_context()


def geometry_context(array_name):
    return {
        "sites": tuple(const.known_arrays[array_name]),
        "ra": 12.5,
        "dec": 12.4,
        "rf": 230.0e9,
        "bandwidth_hz": 2.0e9,
        "t_int": 600.0,
        "t_rest": 1800.0,
        "t_start": 0.0,
        "t_stop": 3.0,
        "mjd": 57855,
    }


def expected_scans(context, time_hours):
    scan_half_width_hours = 0.5 * context["t_int"] / 3600.0
    return np.column_stack((
        time_hours - scan_half_width_hours,
        time_hours + scan_half_width_hours,
    ))


@pytest.mark.parametrize("array_name", ["EHT2017", "ngEHT"])
def test_ground_geometry_preserves_legacy_rows_and_corrects_scan_metadata(array_name):
    array = make_array(const.known_arrays[array_name])
    context = geometry_context(array_name)

    legacy = observation_geometry._legacy_empty_observation(array, context)
    internal = observation_geometry.make_empty_observation(array, context)

    assert np.array_equal(internal.data["t1"], legacy.data["t1"])
    assert np.array_equal(internal.data["t2"], legacy.data["t2"])
    for field in ("time", "tint", "tau1", "tau2", "u", "v", "rrsigma", "llsigma", "rlsigma", "lrsigma"):
        assert np.allclose(internal.data[field], legacy.data[field], rtol=1.0e-9, atol=1.0e-12)
    assert np.allclose(internal.scans, expected_scans(context, np.unique(internal.data["time"])))
    assert not np.allclose(internal.scans, legacy.scans)


@pytest.mark.parametrize("array_name", ["EHT2017", "ngEHT"])
def test_ground_visibility_template_preserves_legacy_visibility_rows(array_name):
    array = make_array(const.known_arrays[array_name])
    context = geometry_context(array_name)

    legacy = observation_geometry._legacy_empty_observation(array, context)
    template = observation_geometry.ground_visibility_template(array, context)
    adapted = template.to_ehtim_obsdata()

    assert isinstance(template, VisibilityDataset)
    assert template.channel_count == 1
    assert template.correlation_layouts == (CIRCULAR_CORRELATIONS,)
    assert np.allclose(template.time_mjd, legacy.mjd + (legacy.data["time"] / 24.0))
    geometry = observation_geometry.ground_geometry(array, context)
    assert np.array_equal(template.antenna1, geometry.station1_indices)
    assert np.array_equal(template.antenna2, geometry.station2_indices)
    assert np.allclose(template.uvw_m, geometry.uvw_m)
    for field in ("time", "tint", "tau1", "tau2", "u", "v", "rrsigma", "llsigma", "rlsigma", "lrsigma"):
        assert np.allclose(adapted.data[field], legacy.data[field], rtol=1.0e-9, atol=1.0e-12)
    assert np.allclose(adapted.scans, expected_scans(context, np.unique(geometry.time_hours)))


@pytest.mark.parametrize("array_name", ["EHT2017", "ngEHT"])
def test_native_visibility_template_matches_legacy_elevation_selection(array_name):
    array = make_array(const.known_arrays[array_name])
    context = geometry_context(array_name)
    _, _, _, legacy = observation_geometry.observation_template(
        None,
        None,
        {},
        array,
        context,
        el_min=const.el_min,
        el_max=const.el_max,
    )
    _, _, _, native = observation_geometry.native_visibility_template(
        None,
        None,
        {},
        array,
        context,
        el_min=const.el_min,
        el_max=const.el_max,
    )
    names = np.asarray(native.stations.names)

    assert np.array_equal(names[native.antenna1], legacy.data["t1"])
    assert np.array_equal(names[native.antenna2], legacy.data["t2"])
    assert np.allclose(
        (native.time_mjd - legacy.mjd) * 24.0,
        legacy.data["time"],
        atol=1.0e-12,
    )
    assert np.allclose(native.integration_time_s, legacy.data["tint"], atol=1.0e-12)


def test_native_visibility_template_reuses_cached_selection(array_and_context):
    array, context = array_and_context
    template, cached_key, template_cache, limited = observation_geometry.native_visibility_template(
        None,
        None,
        {},
        array,
        context,
        el_min=0.0,
        el_max=90.0,
    )
    template_again, cached_key_again, template_cache, limited_again = observation_geometry.native_visibility_template(
        template,
        cached_key,
        template_cache,
        array,
        context,
        el_min=0.0,
        el_max=90.0,
    )

    assert template_again is template
    assert cached_key_again == cached_key
    assert limited_again is limited
    assert len(template_cache) == 1


def test_ground_geometry_scan_groups_preserve_individual_snapshots():
    array = make_array(const.known_arrays["EHT2017"])
    context = geometry_context("EHT2017")
    obs = observation_geometry.make_empty_observation(array, context)

    scan_groups = obs.tlist(scan_gather=True)
    scan_averaged = obs.avg_coherent(0.0, scan_avg=True)

    assert len(scan_groups) == len(np.unique(obs.data["time"]))
    assert len(scan_averaged.data) == len(obs.data)


@pytest.mark.parametrize("array_name", ["EHT2017", "ngEHT"])
def test_ground_station_geometry_matches_ehtim_unpack(array_name):
    array = make_array(const.known_arrays[array_name])
    context = geometry_context(array_name)
    obs = observation_geometry.make_empty_observation(array, context)

    geometry = observation_geometry.ground_station_geometry(obs)
    expected = obs.unpack(
        ["el1", "el2", "par_ang1", "par_ang2"],
        ang_unit="rad",
    )

    assert geometry is not None
    assert np.allclose(geometry.elevation1_rad, expected["el1"], atol=1.0e-10)
    assert np.allclose(geometry.elevation2_rad, expected["el2"], atol=1.0e-10)
    assert np.allclose(
        geometry.parallactic_angle1_rad,
        expected["par_ang1"],
        atol=1.0e-10,
    )
    assert np.allclose(
        geometry.parallactic_angle2_rad,
        expected["par_ang2"],
        atol=1.0e-10,
    )
    assert not geometry.elevation1_rad.flags.writeable


def test_obs_generator_outputs_scan_averaging_ready_data():
    obsgen = obs_generator(
        settings={
            "weather": "exact",
            "source": "M87",
            "array": "EHT2017",
            "dt": 3.0,
            "t_int": 600.0,
            "t_rest": 1800.0,
            "fringe_finder": ["naive", 0.0],
            "random_seed": 1,
        }
    )
    model = eh.model.Model().add_circ_gauss(F0=1.0, FWHM=40.0 * eh.RADPERUAS)

    obs = obsgen.make_obs(
        model,
        addnoise=False,
        addgains=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )
    scan_averaged = obs.avg_coherent(0.0, scan_avg=True)

    assert len(obs.tlist(scan_gather=True)) == 6
    assert len(scan_averaged.data) == len(obs.data)
    assert np.allclose(obs.scans[:, 1] - obs.scans[:, 0], 600.0 / 3600.0)


def test_ground_visibility_template_preserves_rows_across_utc_midnight():
    array = make_array(const.known_arrays["EHT2017"])
    context = geometry_context("EHT2017")
    context.update(t_start=22.0, t_stop=2.0)

    template = observation_geometry.ground_visibility_template(array, context)
    adapted = template.to_ehtim_obsdata()

    assert np.max(template.time_mjd) - np.min(template.time_mjd) > 0.0
    assert np.any(adapted.data["time"] >= 24.0)


def test_ground_geometry_path_does_not_call_legacy_obsdata(array_and_context, monkeypatch):
    array, context = array_and_context
    original_template = observation_geometry.ground_visibility_template
    template_calls = []

    def unexpected_legacy_path(*args, **kwargs):
        raise AssertionError("Ground arrays must use internal geometry.")

    def capture_template(*args, **kwargs):
        template_calls.append((args, kwargs))
        return original_template(*args, **kwargs)

    monkeypatch.setattr(observation_geometry, "_legacy_empty_observation", unexpected_legacy_path)
    monkeypatch.setattr(observation_geometry, "ground_visibility_template", capture_template)
    obs = observation_geometry.make_empty_observation(array, context)

    assert len(obs.data) > 0
    assert len(template_calls) == 1


def test_ground_geometry_preserves_ehtim_model_visibilities():
    array = make_array(const.known_arrays["EHT2017"])
    context = geometry_context("EHT2017")
    source_context = {
        "ra": context["ra"],
        "dec": context["dec"],
        "mjd": context["mjd"],
        "source": "M87",
        "rf": context["rf"],
        "ttype": "nfft",
        "fft_pad_factor": 2,
        "verbosity": 0,
    }
    model = eh.model.Model().add_circ_gauss(F0=1.0, FWHM=40.0 * eh.RADPERUAS)

    legacy, _ = source_models.observe_source(
        model,
        observation_geometry._legacy_empty_observation(array, context),
        source_context,
    )
    internal, _ = source_models.observe_source(
        model,
        observation_geometry.make_empty_observation(array, context),
        source_context,
    )

    for field in ("rrvis", "llvis", "rlvis", "lrvis"):
        assert np.allclose(internal.data[field], legacy.data[field], rtol=1.0e-9, atol=1.0e-12)


def test_space_station_uses_legacy_geometry_fallback(array_and_context, monkeypatch):
    array, context = array_and_context
    array.tarr[0]["x"] = 0.0
    array.tarr[0]["y"] = 0.0
    array.tarr[0]["z"] = 0.0
    original_legacy_path = observation_geometry._legacy_empty_observation
    calls = []

    def capture_legacy_path(*args, **kwargs):
        calls.append((args, kwargs))
        return original_legacy_path(*args, **kwargs)

    monkeypatch.setattr(observation_geometry, "_legacy_empty_observation", capture_legacy_path)
    with np.errstate(invalid="ignore"):
        obs = observation_geometry.make_empty_observation(array, context)

    assert len(calls) == 1
    assert np.allclose(obs.scans, expected_scans(context, np.unique(obs.data["time"])))
    assert observation_geometry.ground_station_geometry(obs) is None


def test_geometry_cache_key_changes_when_geometry_context_changes(array_and_context):
    _, context = array_and_context

    original_key = observation_geometry.geometry_cache_key(context)

    changed_context = dict(context)
    changed_context["rf"] = context["rf"] * 2

    assert observation_geometry.geometry_cache_key(changed_context) != original_key


def test_observation_template_reuses_cached_empty_observation(array_and_context):
    array, context = array_and_context

    cached_obs, cached_key, template_cache, limited_obs = observation_geometry.observation_template(
        None,
        None,
        {},
        array,
        context,
        el_min=0,
        el_max=90,
    )

    cached_obs_again, cached_key_again, template_cache, limited_obs_again = observation_geometry.observation_template(
        cached_obs,
        cached_key,
        template_cache,
        array,
        context,
        el_min=0,
        el_max=90,
    )

    assert cached_obs_again is cached_obs
    assert cached_key_again == cached_key
    assert len(template_cache) == 1
    assert limited_obs_again is not limited_obs
    assert len(limited_obs_again.data) == len(limited_obs.data)


def test_observation_template_rebuilds_when_frequency_changes(array_and_context):
    array, context = array_and_context

    cached_obs, cached_key, template_cache, _ = observation_geometry.observation_template(
        None,
        None,
        {},
        array,
        context,
        el_min=0,
        el_max=90,
    )

    changed_context = dict(context)
    changed_context["rf"] = context["rf"] * 2

    rebuilt_obs, rebuilt_key, template_cache, _ = observation_geometry.observation_template(
        cached_obs,
        cached_key,
        template_cache,
        array,
        changed_context,
        el_min=0,
        el_max=90,
    )

    assert rebuilt_obs is not cached_obs
    assert rebuilt_key != cached_key
    assert len(template_cache) == 1
    assert observation_geometry.elevation_cache_key(rebuilt_key, 0, 90) in template_cache


def test_observation_template_rebuilds_when_timing_changes(array_and_context):
    array, context = array_and_context

    cached_obs, cached_key, template_cache, _ = observation_geometry.observation_template(
        None,
        None,
        {},
        array,
        context,
        el_min=0,
        el_max=90,
    )

    changed_context = dict(context)
    changed_context["t_stop"] = context["t_stop"] + 1

    rebuilt_obs, rebuilt_key, template_cache, _ = observation_geometry.observation_template(
        cached_obs,
        cached_key,
        template_cache,
        array,
        changed_context,
        el_min=0,
        el_max=90,
    )

    assert rebuilt_obs is not cached_obs
    assert rebuilt_key != cached_key
    assert len(template_cache) == 1
    assert observation_geometry.elevation_cache_key(rebuilt_key, 0, 90) in template_cache


def test_observation_template_caches_distinct_elevation_limits(array_and_context):
    array, context = array_and_context

    cached_obs, cached_key, template_cache, limited_obs = observation_geometry.observation_template(
        None,
        None,
        {},
        array,
        context,
        el_min=0,
        el_max=90,
    )

    cached_obs_again, cached_key_again, template_cache, stricter_obs = observation_geometry.observation_template(
        cached_obs,
        cached_key,
        template_cache,
        array,
        context,
        el_min=50,
        el_max=80,
    )

    assert cached_obs_again is cached_obs
    assert cached_key_again == cached_key
    assert len(template_cache) == 2
    assert len(stricter_obs.data) <= len(limited_obs.data)


def test_elevation_limits_do_not_mutate_cached_empty_observation(array_and_context):
    array, context = array_and_context

    cached_obs, cached_key = observation_geometry.ensure_empty_observation(
        None,
        None,
        array,
        context,
    )
    original_len = len(cached_obs.data)

    limited_obs = observation_geometry.apply_elevation_limits(
        cached_obs,
        el_min=50,
        el_max=80,
    )

    assert len(cached_obs.data) == original_len
    assert len(limited_obs.data) <= original_len
    assert limited_obs is not cached_obs
    assert cached_key == observation_geometry.geometry_cache_key(context)
