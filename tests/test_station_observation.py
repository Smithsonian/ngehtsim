#######################################################
# imports

import numpy as np
import ehtim as eh

import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.observation_geometry as observation_geometry
import ngehtsim.obs.source_models as source_models
import ngehtsim.obs.station_observation as station_observation

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


def _compact_model():
    model = eh.model.Model()
    return model.add_circ_gauss(F0=1.0, FWHM=40.0 * eh.RADPERUAS)


def _source_observation(obsgen):
    obsgen.obs_empty, obsgen.obs_empty_key, obsgen.obs_template_cache, obs_empty = observation_geometry.observation_template(
        obsgen.obs_empty,
        obsgen.obs_empty_key,
        obsgen.obs_template_cache,
        obsgen.arr,
        obsgen.geometry_context(),
        el_min=0.0,
        el_max=90.0,
    )
    return source_models.observe_source(_compact_model(), obs_empty, obsgen.source_context())


def _station_terms(obsgen, obs, F0, **kwargs):
    station_kwargs = dict(
        addgains=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
        allow_mixed_basis=False,
        solar_angle=obsgen.solar_angle,
        verbosity=obsgen.verbosity,
        windspeed_sefd_modifier=og.windspeed_SEFD_modification,
    )
    station_kwargs.update(kwargs)
    return station_observation.station_terms(
        obs,
        F0,
        obsgen.station_context(),
        obsgen.arr,
        obsgen.rng,
        **station_kwargs,
    )

#######################################################
# tests


def test_station_metadata_reuses_cached_geometry_terms():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    obs, _ = _source_observation(obsgen)
    cache = {}
    context = obsgen.station_context()

    metadata = station_observation.station_metadata(obs, context, cache=cache)
    metadata_again = station_observation.station_metadata(obs, context, cache=cache)

    assert metadata_again is metadata
    assert len(cache) == 1


def test_station_metadata_cache_key_changes_with_station_context():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    obs, _ = _source_observation(obsgen)
    cache = {}
    context = obsgen.station_context()

    metadata = station_observation.station_metadata(obs, context, cache=cache)

    changed_context = dict(context)
    changed_context["bandwidth_hz"] = dict(context["bandwidth_hz"])
    changed_site = obsgen.sites[0]
    if context["bandwidth_hz"][changed_site] is None:
        changed_context["bandwidth_hz"][changed_site] = 1.0
    else:
        changed_context["bandwidth_hz"][changed_site] = context["bandwidth_hz"][changed_site]*2.0

    changed_metadata = station_observation.station_metadata(obs, changed_context, cache=cache)

    assert changed_metadata is not metadata
    assert len(cache) == 2


def test_station_terms_populates_metadata_cache():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    obs, F0 = _source_observation(obsgen)
    cache = {}

    _station_terms(obsgen, obs, F0, cache=cache)
    _station_terms(obsgen, obs, F0, cache=cache)

    assert len(cache) == 1


def test_station_terms_populates_weather_arrays_without_corruptions():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    obs, F0 = _source_observation(obsgen)

    terms = _station_terms(obsgen, obs, F0)

    data_len = len(obs.data)
    for key in [
        "tau1",
        "tau2",
        "Tb1",
        "Tb2",
        "Tsys1",
        "Tsys2",
        "SEFD1",
        "SEFD2",
        "bw1",
        "bw2",
        "f_el1",
        "f_el2",
        "f_par1",
        "f_par2",
        "phi_off1",
        "phi_off2",
    ]:
        assert len(terms[key]) == data_len

    assert terms["flagsites"] == []
    assert np.all(terms["uptime_mask"])
    assert np.all(terms["SEFD1"] > 0.0)
    assert np.all(terms["SEFD2"] > 0.0)
    assert np.all(terms["bw1"] > 0.0)
    assert np.all(terms["bw2"] > 0.0)
    assert "gainamp1R" not in terms
    assert "leak1R" not in terms


def test_station_terms_populates_gain_and_leakage_arrays_when_enabled():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    obs, F0 = _source_observation(obsgen)

    terms = _station_terms(
        obsgen,
        obs,
        F0,
        addgains=True,
        addleakage=True,
    )

    data_len = len(obs.data)
    for key in [
        "gainamp1R",
        "gainamp2R",
        "gainphase1R",
        "gainphase2R",
        "gainamp1L",
        "gainamp2L",
        "gainphase1L",
        "gainphase2L",
        "leak1R",
        "leak2R",
        "leak1L",
        "leak2L",
    ]:
        assert len(terms[key]) == data_len

    assert np.any(terms["gainamp1R"] != 0.0)
    assert np.any(terms["gainamp2R"] != 0.0)
    assert np.iscomplexobj(terms["leak1R"])
    assert np.iscomplexobj(terms["leak2R"])
