#######################################################
# imports

import pytest
import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.observation_geometry as observation_geometry

#######################################################
# helpers

COMPACT_OBS_SETTINGS = {
    "source": "M87",
    "sites": ["ALMA", "APEX", "LMT", "SMT"],
    "weather": "typical",
    "t_start": 0.0,
    "dt": 6.0,
    "t_int": 600.0,
    "t_rest": 1200.0,
    "random_seed": 1,
}


@pytest.fixture(scope="module")
def array_and_context():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    return obsgen.arr, obsgen.geometry_context()

#######################################################
# tests


def test_geometry_cache_key_changes_when_geometry_context_changes(array_and_context):
    array, context = array_and_context

    key = observation_geometry.geometry_cache_key(context)

    new_context = dict(context)
    new_context["t_stop"] = context["t_stop"] + 1.0

    new_key = observation_geometry.geometry_cache_key(new_context)

    assert new_key != key


def test_observation_template_reuses_cached_empty_observation(array_and_context):
    array, context = array_and_context

    cached_obs, cached_key, limited_obs = observation_geometry.observation_template(
        None,
        None,
        array,
        context,
        el_min=0.0,
        el_max=90.0,
    )

    cached_obs_again, cached_key_again, limited_obs_again = observation_geometry.observation_template(
        cached_obs,
        cached_key,
        array,
        context,
        el_min=0.0,
        el_max=90.0,
    )

    assert cached_obs_again is cached_obs
    assert cached_key_again == cached_key
    assert limited_obs is not cached_obs
    assert limited_obs_again is not cached_obs


def test_observation_template_rebuilds_when_frequency_changes(array_and_context):
    array, context = array_and_context

    cached_obs, cached_key, _ = observation_geometry.observation_template(
        None,
        None,
        array,
        context,
        el_min=0.0,
        el_max=90.0,
    )

    new_context = dict(context)
    new_context["rf"] = 345.0e9

    rebuilt_obs, rebuilt_key, _ = observation_geometry.observation_template(
        cached_obs,
        cached_key,
        array,
        new_context,
        el_min=0.0,
        el_max=90.0,
    )

    assert rebuilt_obs is not cached_obs
    assert rebuilt_obs.rf == new_context["rf"]
    assert rebuilt_key != cached_key


def test_observation_template_rebuilds_when_timing_changes(array_and_context):
    array, context = array_and_context

    cached_obs, cached_key, _ = observation_geometry.observation_template(
        None,
        None,
        array,
        context,
        el_min=0.0,
        el_max=90.0,
    )

    new_context = dict(context)
    new_context["t_stop"] = context["t_stop"] + 1.0

    rebuilt_obs, rebuilt_key, _ = observation_geometry.observation_template(
        cached_obs,
        cached_key,
        array,
        new_context,
        el_min=0.0,
        el_max=90.0,
    )

    assert rebuilt_obs is not cached_obs
    assert rebuilt_key != cached_key


def test_elevation_limits_do_not_mutate_cached_empty_observation(array_and_context):
    array, context = array_and_context

    cached_obs, cached_key, strict_obs = observation_geometry.observation_template(
        None,
        None,
        array,
        context,
        el_min=50.0,
        el_max=80.0,
    )

    original_row_count = len(cached_obs.data)

    assert cached_key == observation_geometry.geometry_cache_key(context)
    assert len(strict_obs.data) < original_row_count
    assert len(cached_obs.data) == original_row_count
