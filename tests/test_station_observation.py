#######################################################
# imports

import copy

import numpy as np
import ehtim as eh

import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.observation_geometry as observation_geometry
import ngehtsim.obs.source_models as source_models
import ngehtsim.obs.station_observation as station_observation
from ngehtsim.obs.visibility_dataset import VisibilityDataset

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


def _time_varying_weather_context(obsgen, obs):
    context = obsgen.station_context()
    count = len(obs.data)
    context = dict(context)
    for quantity in ("tau", "Tatm", "Tgnd", "windspeed"):
        context[quantity] = {
            site: np.full(count, value, dtype=float)
            for site, value in context[quantity].items()
        }
    return context

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


def test_station_metadata_ground_path_does_not_call_ehtim_unpack(monkeypatch):
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    obs, _ = _source_observation(obsgen)

    def unexpected_unpack(*args, **kwargs):
        raise AssertionError("Ground station metadata must use native geometry.")

    monkeypatch.setattr(obs, "unpack", unexpected_unpack)
    metadata = station_observation.station_metadata(obs, obsgen.station_context())

    assert len(metadata["el1"]) == len(obs.data)
    assert len(metadata["par1"]) == len(obs.data)


def test_native_station_metadata_matches_ehtim_fallback(monkeypatch):
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    obs, _ = _source_observation(obsgen)
    context = obsgen.station_context()

    native = station_observation.station_metadata(obs, context)
    monkeypatch.setattr(
        station_observation.observation_geometry,
        "ground_station_geometry",
        lambda obs: None,
    )
    legacy = station_observation.station_metadata(obs, context)

    for field in ("el1", "el2", "par1", "par2"):
        assert np.allclose(native[field], legacy[field], atol=1.0e-10)


def test_native_station_geometry_preserves_generated_observations(monkeypatch):
    settings = dict(COMPACT_OBS_SETTINGS)
    settings["fringe_finder"] = ["naive", 0.0]

    native_generator = og.obs_generator(settings=settings)
    native_obs = native_generator.make_obs(
        _compact_model(),
        addnoise=False,
        addgains=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
        addFR=True,
    )

    monkeypatch.setattr(
        station_observation.observation_geometry,
        "ground_station_geometry",
        lambda obs: None,
    )
    fallback_generator = og.obs_generator(settings=settings)
    fallback_obs = fallback_generator.make_obs(
        _compact_model(),
        addnoise=False,
        addgains=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
        addFR=True,
    )

    assert np.array_equal(native_obs.data["t1"], fallback_obs.data["t1"])
    assert np.array_equal(native_obs.data["t2"], fallback_obs.data["t2"])
    for field in (
        "time",
        "u",
        "v",
        "tau1",
        "tau2",
        "rrvis",
        "llvis",
        "rlvis",
        "lrvis",
        "rrsigma",
        "llsigma",
        "rlsigma",
        "lrsigma",
    ):
        assert np.allclose(native_obs.data[field], fallback_obs.data[field], atol=1.0e-10)


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


def test_station_terms_accepts_per_observation_weather_values():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    obs, F0 = _source_observation(obsgen)
    context = _time_varying_weather_context(obsgen, obs)
    context["tau"]["ALMA"] = np.linspace(0.1, 0.2, len(obs.data))
    context["Tatm"]["ALMA"] = np.linspace(240.0, 260.0, len(obs.data))
    context["Tgnd"]["ALMA"] = np.linspace(250.0, 270.0, len(obs.data))

    terms = station_observation.station_terms(
        obs,
        F0,
        context,
        obsgen.arr,
        obsgen.rng,
        addgains=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
        solar_angle=obsgen.solar_angle,
        windspeed_sefd_modifier=og.windspeed_SEFD_modification,
    )

    ind1 = terms["t1"] == "ALMA"
    ind2 = terms["t2"] == "ALMA"
    assert np.allclose(
        terms["tau1"][ind1] * np.cos((np.pi/2.0) - terms["el1"][ind1]),
        context["tau"]["ALMA"][ind1],
    )
    assert np.allclose(
        terms["tau2"][ind2] * np.cos((np.pi/2.0) - terms["el2"][ind2]),
        context["tau"]["ALMA"][ind2],
    )


def test_station_terms_flags_time_varying_wind_per_timestamp():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    obs, F0 = _source_observation(obsgen)
    context = _time_varying_weather_context(obsgen, obs)
    high_wind_time = np.max(obs.data["time"])
    context["windspeed"]["ALMA"][obs.data["time"] == high_wind_time] = 30.0

    terms = station_observation.station_terms(
        obs,
        F0,
        context,
        obsgen.arr,
        obsgen.rng,
        addgains=False,
        addleakage=False,
        flagwind=True,
        flagday=False,
        flagsun=False,
        solar_angle=obsgen.solar_angle,
        windspeed_sefd_modifier=og.windspeed_SEFD_modification,
    )

    expected_flags = (
        ((terms["t1"] == "ALMA") | (terms["t2"] == "ALMA"))
        & (terms["times"] == high_wind_time)
    )
    assert terms["flagsites"] == []
    assert np.array_equal(~terms["uptime_mask"], expected_flags)


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


def test_native_station_terms_match_obsdata_terms_and_station_table():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    obs, F0 = _source_observation(obsgen)
    context = _time_varying_weather_context(obsgen, obs)
    context["tau"]["ALMA"] = np.linspace(0.1, 0.2, len(obs.data))
    context["windspeed"]["ALMA"] = np.linspace(1.0, 8.0, len(obs.data))
    kwargs = {
        "gainamp": 0.04,
        "leakamp": 0.1,
        "addgains": True,
        "addleakage": True,
        "flagwind": True,
        "flagday": False,
        "flagsun": False,
        "solar_angle": obsgen.solar_angle,
        "windspeed_sefd_modifier": og.windspeed_SEFD_modification,
    }

    legacy_array = copy.deepcopy(obsgen.arr)
    legacy = station_observation.station_terms(
        obs.copy(),
        F0,
        context,
        legacy_array,
        np.random.default_rng(17),
        **kwargs,
    )
    native, native_stations = station_observation.station_terms_for_dataset(
        VisibilityDataset.from_ehtim_obsdata(obs),
        F0,
        context,
        np.random.default_rng(17),
        reference_mjd=obs.mjd,
        **kwargs,
    )

    assert np.array_equal(native["t1"], legacy["t1"])
    assert np.array_equal(native["t2"], legacy["t2"])
    assert native["flagsites"] == legacy["flagsites"]
    for field in (
        "times",
        "el1",
        "el2",
        "par1",
        "par2",
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
    ):
        assert np.allclose(native[field], legacy[field], atol=1.0e-10)
    assert np.array_equal(native["uptime_mask"], legacy["uptime_mask"])
    assert np.allclose(native_stations.sefd_r_jy, legacy_array.tarr["sefdr"])
    assert np.allclose(native_stations.sefd_l_jy, legacy_array.tarr["sefdl"])
    assert np.allclose(native_stations.leakage_r, legacy_array.tarr["dr"])
    assert np.allclose(native_stations.leakage_l, legacy_array.tarr["dl"])


def test_native_station_metadata_uses_cache_without_obsdata_conversion():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    obs, _ = _source_observation(obsgen)
    dataset = VisibilityDataset.from_ehtim_obsdata(obs)
    cache = {}

    metadata = station_observation.station_metadata_for_dataset(
        dataset,
        obsgen.station_context(),
        reference_mjd=obs.mjd,
        cache=cache,
    )
    metadata_again = station_observation.station_metadata_for_dataset(
        dataset,
        obsgen.station_context(),
        reference_mjd=obs.mjd,
        cache=cache,
    )

    assert metadata_again is metadata
    assert len(cache) == 1
    assert np.array_equal(
        metadata["_rows"].t1,
        np.asarray(dataset.stations.names)[dataset.antenna1],
    )
