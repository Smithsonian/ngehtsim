#######################################################
# imports

import inspect
import pytest
import ehtim as eh
import ngehtsim.calibration.calibration as nc
import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.source_models as source_models
from ngehtsim.obs.station_effects import StationCorruptionModel

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


def _observe_without_corruptions(obsgen, input_model, **kwargs):
    effects = StationCorruptionModel(
        thermal_noise=False,
        station_gain=None,
        feed_rotation=False,
        flag_wind=False,
        flag_daylight=False,
        flag_sun=False,
    )
    return obsgen.observe(input_model, effects=effects, **kwargs)

#######################################################
# tests


@pytest.mark.parametrize(
    "callable_obj",
    [
        og.obs_generator.__init__,
        nc.apriorical,
        nc.write_dlist,
    ],
)
def test_public_api_defaults_are_not_mutable(callable_obj):
    signature = inspect.signature(callable_obj)

    for parameter in signature.parameters.values():
        if parameter.default is inspect.Parameter.empty:
            continue

        assert not isinstance(parameter.default, (dict, list, set))


def test_observe_elevation_cuts_do_not_mutate_cached_empty_observation():
    model = _compact_model()

    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)

    strict_obs = _observe_without_corruptions(
        obsgen,
        model,
        el_min=50.0,
        el_max=80.0,
    )

    relaxed_after_strict_obs = _observe_without_corruptions(
        obsgen,
        model,
        el_min=0.0,
        el_max=90.0,
    )

    fresh_obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    relaxed_fresh_obs = _observe_without_corruptions(
        fresh_obsgen,
        model,
        el_min=0.0,
        el_max=90.0,
    )

    assert len(strict_obs.data) < len(relaxed_fresh_obs.data)
    assert len(relaxed_after_strict_obs.data) == len(relaxed_fresh_obs.data)


def test_unsupported_model_raises_type_error_when_ngEHTforecast_is_missing(monkeypatch):
    monkeypatch.setattr(source_models, "fp", None)

    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)

    with pytest.raises(TypeError, match="input_model must be"):
        _observe_without_corruptions(obsgen, object())
