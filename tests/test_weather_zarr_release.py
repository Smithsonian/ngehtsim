"""Opt-in acceptance test for an externally distributed Zarr weather release."""

import os

import numpy as np
import pytest

from ngehtsim.weather.zarr_store import SCHEMA_VERSION, ZarrWeatherStore


@pytest.mark.external_weather
def test_local_weather_release_has_expected_schema_and_alma_coverage():
    path = os.environ.get("NGEHTSIM_WEATHER_ZARR")
    if path is None:
        pytest.skip("Set NGEHTSIM_WEATHER_ZARR to run the external weather release test.")

    store = ZarrWeatherStore(path)
    assert store.attributes["schema_version"] == SCHEMA_VERSION
    assert store.dataset_id == "ngehtsim-weather-merra2-3hour-v0.1.0"
    assert len(store.sites) == 141

    daily = store.read_partition("ALMA", "Apr", cadence="daily")
    native = store.read_partition("ALMA", "Apr", cadence="native")
    assert daily.record_count * 8 == native.record_count
    assert np.all(np.isfinite(store.reconstruct_tau_spectra(daily)))
    assert np.all(np.isfinite(store.reconstruct_tb_spectra(daily)))
