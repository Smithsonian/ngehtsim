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

    context = {
        "ra": obsgen.RA,
        "dec": obsgen.DEC,
        "rf": obsgen.freq,
        "bandwidth_hz": (1.0e9)*float(obsgen.settings["bandwidth"]),
        "t_int": obsgen.settings["t_int"],
        "t_rest": obsgen.settings["t_rest"],
        "t_start": obsgen.settings["t_start"],
        "t_stop": obsgen.settings["t_start"] + obsgen.settings["dt"],
        "mjd": obsgen.mjd,
    }

    return obsgen.arr, context

#######################################################
# tests


def test_observation_template_reuses_cached_empty_observation(array_and_context):
    array, context = array_and_context

    cached_obs, limited_obs = observation_geometry.observation_template(
        None,
        array,
        context,
        el_min=0.0,
        el_max=90.0,
    )

    cached_obs_again, limited_obs_again = observation_geometry.observation_template(
        cached_obs,
        array,
        context,
        el_min=0.0,
        el_max=90.0,
    )

    assert cached_obs_again is cached_obs
    assert limited_obs is not cached_obs
    assert limited_obs_again is not cached_obs


def test_observation_template_rebuilds_when_frequency_changes(array_and_context):
    array, context = array_and_context

    cached_obs, _ = observation_geometry.observation_template(
        None,
        array,
        context,
        el_min=0.0,
        el_max=90.0,
    )

    new_context = dict(context)
    new_context["rf"] = 345.0e9

    rebuilt_obs, _ = observation_geometry.observation_template(
        cached_obs,
        array,
        new_context,
        el_min=0.0,
        el_max=90.0,
    )

    assert rebuilt_obs is not cached_obs
    assert rebuilt_obs.rf == new_context["rf"]


def test_elevation_limits_do_not_mutate_cached_empty_observation(array_and_context):
    array, context = array_and_context

    cached_obs, strict_obs = observation_geometry.observation_template(
        None,
        array,
        context,
        el_min=50.0,
        el_max=80.0,
    )

    original_row_count = len(cached_obs.data)

    assert len(strict_obs.data) < original_row_count
    assert len(cached_obs.data) == original_row_count
