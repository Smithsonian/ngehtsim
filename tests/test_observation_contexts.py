#######################################################
# imports

import numpy as np

import ngehtsim.const_def as const
import ngehtsim.obs.obs_generator as og

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

#######################################################
# tests


def test_geometry_context_contains_observation_template_inputs():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)

    context = obsgen.geometry_context()

    assert context["sites"] == tuple(obsgen.sites)
    assert context["ra"] == obsgen.RA
    assert context["dec"] == obsgen.DEC
    assert context["rf"] == obsgen.freq
    assert context["bandwidth_hz"] == (1.0e9)*float(obsgen.settings["bandwidth"])
    assert context["t_int"] == obsgen.settings["t_int"]
    assert context["t_rest"] == obsgen.settings["t_rest"]
    assert context["t_start"] == obsgen.settings["t_start"]
    assert context["t_stop"] == obsgen.settings["t_start"] + obsgen.settings["dt"]
    assert context["mjd"] == obsgen.mjd


def test_source_context_contains_source_adapter_inputs():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)

    context = obsgen.source_context()

    assert context["ra"] == obsgen.RA
    assert context["dec"] == obsgen.DEC
    assert context["mjd"] == obsgen.mjd
    assert context["source"] == obsgen.settings["source"]
    assert context["rf"] == obsgen.freq
    assert context["transform_backend"] == obsgen.settings["transform_backend"]
    assert context["raster_tolerance"] == obsgen.settings["raster_tolerance"]
    assert context["verbosity"] == obsgen.verbosity


def test_station_context_contains_weather_receiver_and_noise_inputs():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)

    context = obsgen.station_context()

    assert context["sites"] == tuple(obsgen.sites)

    for site in obsgen.sites:
        band = obsgen.bands[site]

        assert context["tau"][site] == obsgen.tau_dict[site]
        assert context["Tatm"][site] == obsgen.Tatm_dict[site]
        assert context["Tgnd"][site] == obsgen.Tgnd_dict[site]
        assert context["windspeed"][site] == obsgen.windspeed_dict[site]
        assert context["bands"][site] == band
        assert context["wind_loading"][site] == obsgen.wind_loading_dict[site]
        assert context["solar_avoidance"][site] == obsgen.solar_avoidance_dict[site]

        expected_area = (np.pi/4.0)*obsgen.eta_dict[site]*((obsgen.D_dict[site])**2)
        assert np.isclose(context["effective_area"][site], expected_area)

        if band is None:
            assert context["receiver_temperature"][site] == 0.0
            assert context["sideband_ratio"][site] == 0.0
            assert context["bandwidth_hz"][site] is None
        else:
            assert context["receiver_temperature"][site] == obsgen.receivers[site][band]["T_R"]
            assert context["sideband_ratio"][site] == obsgen.receivers[site][band]["SSR"]

            if band in obsgen.bandwidth_setup[site]:
                assert context["bandwidth_hz"][site] == obsgen.bandwidth_setup[site][band]*(1.0e9)
            else:
                assert context["bandwidth_hz"][site] is None


def test_station_context_resolves_station_defaults():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)

    context = obsgen.station_context()

    for site in obsgen.sites:
        assert context["mount_type"][site] == const.known_mount_types.get(site, const.mount_type)
        assert context["feed_angle"][site] == const.known_feed_angles.get(site, const.feed_angle)
        assert context["polarization_basis"][site] == const.known_polbases.get(site)
        assert context["station_uptimes"] == obsgen.station_uptimes
