#######################################################
# imports

import numpy as np
import pytest
import ngehtsim.weather.weather as nw

#######################################################
# constants

MONTH_ALIASES = ["Apr", "04", "4", 4]

SCALAR_WEATHER_FUNCTIONS = [
    (nw.opacity, {"freq": 230.0}),
    (nw.brightness_temperature, {"freq": 230.0}),
    (nw.pressure, {}),
    (nw.temperature, {}),
    (nw.PWV, {}),
    (nw.windspeed, {}),
]

SPECTRUM_WEATHER_FUNCTIONS = [
    nw.opacity_spectrum,
    nw.brightness_temperature_spectrum,
]

#######################################################
# tests


@pytest.mark.parametrize("month", MONTH_ALIASES)
@pytest.mark.parametrize("weather_function, extra_kwargs", SCALAR_WEATHER_FUNCTIONS)
def test_exact_scalar_weather_month_aliases(weather_function, extra_kwargs, month):
    kwargs = dict(site="ALMA", form="exact", day=11, year=2017)
    kwargs.update(extra_kwargs)

    reference = weather_function(month="Apr", **kwargs)
    result = weather_function(month=month, **kwargs)

    assert np.isclose(result, reference, equal_nan=True)


@pytest.mark.parametrize("month", MONTH_ALIASES)
@pytest.mark.parametrize("weather_function", SPECTRUM_WEATHER_FUNCTIONS)
def test_exact_spectrum_weather_month_aliases(weather_function, month):
    kwargs = dict(site="ALMA", form="exact", day=11, year=2017)

    reference = weather_function(month="Apr", **kwargs)
    result = weather_function(month=month, **kwargs)

    assert np.allclose(result, reference, equal_nan=True)


@pytest.mark.parametrize(
    ("weather_function", "reconstructor_name"),
    [
        (nw.opacity_spectrum, "reconstruct_spectrum_tau"),
        (nw.brightness_temperature_spectrum, "reconstruct_spectrum_Tb"),
    ],
)
def test_exact_spectrum_reconstructs_only_the_requested_legacy_record(
    weather_function, reconstructor_name, monkeypatch
):
    original_reconstructor = getattr(nw, reconstructor_name)
    calls = []

    def count_reconstructions(coefficients):
        calls.append(np.asarray(coefficients))
        return original_reconstructor(coefficients)

    monkeypatch.setattr(nw, reconstructor_name, count_reconstructions)

    result = weather_function("ALMA", form="exact", month="Apr", day=11, year=2017)

    assert len(calls) == 1
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("weather_function, extra_kwargs", SCALAR_WEATHER_FUNCTIONS)
def test_february_scalar_integer_month_matches_named_month(weather_function, extra_kwargs):
    kwargs = dict(site="ALMA", form="median")
    kwargs.update(extra_kwargs)

    reference = weather_function(month="Feb", **kwargs)
    result = weather_function(month=2, **kwargs)

    assert np.allclose(result, reference, equal_nan=True)


@pytest.mark.parametrize("weather_function", SPECTRUM_WEATHER_FUNCTIONS)
def test_february_spectrum_integer_month_matches_named_month(weather_function):
    kwargs = dict(site="ALMA", form="median")

    reference = weather_function(month="Feb", **kwargs)
    result = weather_function(month=2, **kwargs)

    assert np.allclose(result, reference, equal_nan=True)


@pytest.mark.parametrize("bad_month", [0, 13, "Foo", "", None])
def test_invalid_month_rejected(bad_month):
    with pytest.raises(ValueError, match="Specified month not recognized"):
        nw.opacity(site="ALMA", form="median", month=bad_month, freq=230.0)
