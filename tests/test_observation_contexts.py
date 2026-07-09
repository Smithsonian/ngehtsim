#######################################################
# imports

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
    assert context["ttype"] == obsgen.settings["ttype"]
    assert context["fft_pad_factor"] == obsgen.settings["fft_pad_factor"]
    assert context["verbosity"] == obsgen.verbosity
