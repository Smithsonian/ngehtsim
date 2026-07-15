"""Unit tests for the local versioned Zarr weather reader."""

import numpy as np
import pytest
import zarr

from ngehtsim.weather.zarr_store import WeatherStoreError, ZarrWeatherStore

def test_store_exposes_validated_metadata(weather_dataset):
    store = ZarrWeatherStore(weather_dataset)

    assert store.dataset_id == "test-weather-v0.1.0"
    assert store.attributes["schema_version"] == "0.1.0"
    assert store.sites == ("ALMA",)
    assert np.array_equal(store.frequency_ghz, [100.0, 200.0, 300.0])
    assert not store.frequency_ghz.flags.writeable


@pytest.mark.parametrize("month", ["Apr", "04", "4", 4])
def test_store_reads_daily_partition_for_month_aliases(weather_dataset, month):
    partition = ZarrWeatherStore(weather_dataset).read_partition("ALMA", month)

    assert partition.month == 4
    assert partition.cadence == "daily"
    assert partition.time_index is None
    assert partition.record_count == 2
    assert np.array_equal(partition.year, [2017, 2017])
    assert np.array_equal(partition.day, [11, 12])
    assert np.array_equal(partition.tau_coefficients, [[1.0, 2.0], [2.0, 1.0]])
    assert not partition.year.flags.writeable


def test_store_reads_native_partition(weather_dataset):
    partition = ZarrWeatherStore(weather_dataset).read_partition(
        "ALMA", "Apr", cadence="native"
    )

    assert np.array_equal(partition.time_index, [0, 1])


def test_store_caches_validated_partitions(weather_dataset):
    store = ZarrWeatherStore(weather_dataset)

    assert store.read_partition("ALMA", "Apr") is store.read_partition("ALMA", 4)


def test_store_reconstructs_tau_and_tb_spectra(weather_dataset):
    store = ZarrWeatherStore(weather_dataset)
    partition = store.read_partition("ALMA", "Apr")

    assert np.allclose(
        store.reconstruct_tau_spectra(partition),
        [[10.0, 1000.0, 10.0], [100.0, 100.0, 1.0]],
    )
    assert np.allclose(
        store.reconstruct_tb_spectra(partition),
        [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]],
    )


@pytest.mark.parametrize("month", [0, 13, "Foo", "", None])
def test_store_rejects_invalid_month(weather_dataset, month):
    store = ZarrWeatherStore(weather_dataset)

    with pytest.raises(ValueError, match="Specified month not recognized"):
        store.read_partition("ALMA", month)


def test_store_rejects_unknown_site_and_cadence(weather_dataset):
    store = ZarrWeatherStore(weather_dataset)

    with pytest.raises(KeyError, match="does not contain site"):
        store.read_partition("APEX", "Apr")
    with pytest.raises(ValueError, match="Unsupported weather cadence"):
        store.read_partition("ALMA", "Apr", cadence="hourly")


def test_store_rejects_incompatible_schema(weather_dataset):
    root = zarr.open_group(weather_dataset, mode="r+")
    root.attrs["schema_version"] = "99.0.0"

    with pytest.raises(WeatherStoreError, match="Unsupported Zarr weather schema"):
        ZarrWeatherStore(weather_dataset)


def test_store_rejects_nonfinite_partition_values(weather_dataset):
    root = zarr.open_group(weather_dataset, mode="r+")
    root["sites/ALMA/months/04/daily/tau_coefficients"][0, 0] = np.nan
    store = ZarrWeatherStore(weather_dataset)

    with pytest.raises(WeatherStoreError, match="contains non-finite values"):
        store.read_partition("ALMA", "Apr")


def test_store_rejects_invalid_calendar_dates(weather_dataset):
    root = zarr.open_group(weather_dataset, mode="r+")
    root["sites/ALMA/months/04/daily/day"][0] = 31
    store = ZarrWeatherStore(weather_dataset)

    with pytest.raises(WeatherStoreError, match="contains invalid calendar dates"):
        store.read_partition("ALMA", "Apr")


def test_store_rejects_invalid_native_time_index(weather_dataset):
    root = zarr.open_group(weather_dataset, mode="r+")
    root["sites/ALMA/months/04/native/time_index"][0] = 8
    store = ZarrWeatherStore(weather_dataset)

    with pytest.raises(WeatherStoreError, match="contains invalid native time indices"):
        store.read_partition("ALMA", "Apr", cadence="native")
