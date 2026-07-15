"""Opt-in acceptance test for an externally distributed Zarr weather release."""

import os

import numpy as np
import pytest

import ngehtsim.weather.weather as weather
from ngehtsim.weather.zarr_store import SCHEMA_VERSION, ZarrWeatherStore


WEATHER_FUNCTIONS = (
    (weather.opacity_spectrum, {}),
    (weather.brightness_temperature_spectrum, {}),
    (weather.opacity, {"freq": 230.0}),
    (weather.brightness_temperature, {"freq": 230.0}),
    (weather.pressure, {}),
    (weather.temperature, {}),
    (weather.PWV, {}),
    (weather.windspeed, {}),
)


@pytest.fixture(scope="module")
def external_store():
    path = os.environ.get("NGEHTSIM_WEATHER_ZARR")
    if path is None:
        pytest.skip("Set NGEHTSIM_WEATHER_ZARR to run the external weather release test.")
    return ZarrWeatherStore(path)


@pytest.mark.external_weather
def test_local_weather_release_has_expected_schema_and_alma_coverage(external_store):
    assert external_store.attributes["schema_version"] == SCHEMA_VERSION
    assert external_store.dataset_id == "ngehtsim-weather-merra2-3hour-v0.1.0"
    assert len(external_store.sites) == 141

    daily = external_store.read_partition("ALMA", "Apr", cadence="daily")
    native = external_store.read_partition("ALMA", "Apr", cadence="native")
    assert daily.record_count * 8 == native.record_count
    assert np.all(np.isfinite(external_store.reconstruct_tau_spectra(daily)))
    assert np.all(np.isfinite(external_store.reconstruct_tb_spectra(daily)))


@pytest.mark.external_weather
@pytest.mark.parametrize("month, year, day", [("Apr", 2017, 11), ("Feb", 2017, 11)])
@pytest.mark.parametrize("form", ["exact", "all", "mean", "median", "good", "bad"])
@pytest.mark.parametrize("weather_function, extra_kwargs", WEATHER_FUNCTIONS)
def test_zarr_weather_api_matches_packaged_daily_weather(
    external_store, month, year, day, form, weather_function, extra_kwargs
):
    kwargs = {"form": form, "month": month, "day": day, "year": year}
    kwargs.update(extra_kwargs)

    legacy = weather_function("ALMA", **kwargs)
    zarr = weather_function("ALMA", weather_store=external_store, **kwargs)

    assert np.allclose(zarr, legacy, equal_nan=True)
