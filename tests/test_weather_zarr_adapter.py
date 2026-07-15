"""Unit tests for the optional Zarr-backed public weather API."""

import numpy as np
import pytest

import ngehtsim.weather.weather as weather
from ngehtsim.weather.zarr_store import ZarrWeatherStore


@pytest.fixture
def store(weather_dataset):
    return ZarrWeatherStore(weather_dataset)


@pytest.mark.parametrize(
    "weather_function, extra_kwargs, expected",
    [
        (weather.opacity_spectrum, {}, [10.0, 1000.0, 10.0]),
        (weather.brightness_temperature_spectrum, {}, [11.0, 22.0, 33.0]),
        (weather.opacity, {"freq": 200.0}, 1000.0),
        (weather.brightness_temperature, {"freq": 200.0}, 22.0),
        (weather.pressure, {}, 500.0),
        (weather.temperature, {}, 250.0),
        (weather.PWV, {}, 1.0),
        (weather.windspeed, {}, 3.0),
    ],
)
def test_exact_weather_api_reads_zarr_store(weather_function, extra_kwargs, expected, store):
    result = weather_function(
        "ALMA",
        form="exact",
        month="Apr",
        day=11,
        year=2017,
        weather_store=store,
        **extra_kwargs,
    )

    assert np.allclose(result, expected)


@pytest.mark.parametrize("form", ["all", "mean", "median", "good", "bad"])
@pytest.mark.parametrize(
    "weather_function, extra_kwargs",
    [
        (weather.opacity_spectrum, {}),
        (weather.brightness_temperature_spectrum, {}),
        (weather.opacity, {"freq": 200.0}),
        (weather.brightness_temperature, {"freq": 200.0}),
        (weather.pressure, {}),
        (weather.temperature, {}),
        (weather.PWV, {}),
        (weather.windspeed, {}),
    ],
)
def test_summary_weather_api_reads_zarr_store(weather_function, extra_kwargs, form, store):
    result = weather_function(
        "ALMA", form=form, month=4, weather_store=store, **extra_kwargs
    )

    if form == "all":
        assert np.shape(result)[0] == 2
    else:
        assert np.all(np.isfinite(result))


def test_zarr_weather_api_does_not_load_legacy_pca_bases(monkeypatch, store):
    def fail_if_called():
        raise AssertionError("The legacy PCA basis must not be loaded for Zarr weather.")

    monkeypatch.setattr(weather, "_legacy_pca_bases", fail_if_called)

    assert weather.opacity(
        "ALMA", form="exact", month=4, day=11, year=2017, freq=200.0, weather_store=store
    ) == 1000.0
    assert weather.pressure(
        "ALMA", form="exact", month=4, day=11, year=2017, weather_store=store
    ) == 500.0


@pytest.mark.parametrize("weather_function", [weather.opacity, weather.pressure])
def test_weather_api_rejects_invalid_store(weather_function):
    with pytest.raises(TypeError, match="weather_store must be a ZarrWeatherStore"):
        weather_function("ALMA", form="mean", weather_store="not-a-weather-store")
