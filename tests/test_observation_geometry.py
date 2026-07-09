import pytest

from ngehtsim.obs.obs_generator import obs_generator
import ngehtsim.obs.observation_geometry as observation_geometry


@pytest.fixture
def array_and_context():
    obsgen = obs_generator(settings={"weather": "exact", "source": "M87"})
    return obsgen.arr, obsgen.geometry_context()


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
