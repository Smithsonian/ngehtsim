import numpy as np
import pytest
import ehtim as eh

import ngehtsim.const_def as const
from ngehtsim.obs.obs_generator import obs_generator
import ngehtsim.obs.observation_geometry as observation_geometry
from ngehtsim.obs.obs_generator import make_array
import ngehtsim.obs.source_models as source_models


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


@pytest.mark.parametrize("array_name", ["EHT2017", "ngEHT"])
def test_ground_geometry_matches_legacy_ehtim_template(array_name):
    array = make_array(const.known_arrays[array_name])
    context = geometry_context(array_name)

    legacy = observation_geometry._legacy_empty_observation(array, context)
    internal = observation_geometry.make_empty_observation(array, context)

    assert np.array_equal(internal.data["t1"], legacy.data["t1"])
    assert np.array_equal(internal.data["t2"], legacy.data["t2"])
    for field in ("time", "tint", "tau1", "tau2", "u", "v", "rrsigma", "llsigma", "rlsigma", "lrsigma"):
        assert np.allclose(internal.data[field], legacy.data[field], rtol=1.0e-9, atol=1.0e-12)
    assert np.allclose(internal.scans, legacy.scans)


def test_ground_geometry_path_does_not_call_legacy_obsdata(array_and_context, monkeypatch):
    array, context = array_and_context

    def unexpected_legacy_path(*args, **kwargs):
        raise AssertionError("Ground arrays must use internal geometry.")

    monkeypatch.setattr(observation_geometry, "_legacy_empty_observation", unexpected_legacy_path)
    obs = observation_geometry.make_empty_observation(array, context)

    assert len(obs.data) > 0


def test_ground_geometry_preserves_ehtim_model_visibilities():
    array = make_array(const.known_arrays["EHT2017"])
    context = geometry_context("EHT2017")
    source_context = {
        "ra": context["ra"],
        "dec": context["dec"],
        "mjd": context["mjd"],
        "source": "M87",
        "rf": context["rf"],
        "ttype": "fast",
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
    sentinel = object()

    monkeypatch.setattr(
        observation_geometry,
        "_legacy_empty_observation",
        lambda array, context: sentinel,
    )

    assert observation_geometry.make_empty_observation(array, context) is sentinel


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
