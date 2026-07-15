"""Unit tests for the local versioned Zarr weather reader."""

import numpy as np
import pytest
import zarr

from ngehtsim.weather.zarr_store import WeatherStoreError, ZarrWeatherStore


@pytest.fixture
def weather_dataset(tmp_path):
    path = tmp_path / "weather.zarr"
    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "schema_version": "0.1.0",
            "dataset_id": "test-weather-v0.1.0",
            "native_time_step_hours": 3,
            "native_samples_per_day": 8,
        }
    )
    root.create_array("frequency_ghz", data=np.array([100.0, 200.0, 300.0]))
    root.create_array("pca/tau/mean", data=np.array([0.0, 1.0, 2.0]))
    root.create_array(
        "pca/tau/components", data=np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    )
    root.create_array("pca/tb/mean", data=np.array([10.0, 20.0, 30.0]))
    root.create_array(
        "pca/tb/components", data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )

    daily = root.create_group("sites/ALMA/months/04/daily")
    _write_records(daily, include_time_index=False)
    native = root.create_group("sites/ALMA/months/04/native")
    _write_records(native, include_time_index=True)
    return path


def _write_records(group, include_time_index):
    group.create_array("year", data=np.array([2017, 2017], dtype=np.int16))
    group.create_array("day", data=np.array([11, 12], dtype=np.int8))
    group.create_array(
        "tau_coefficients", data=np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float16)
    )
    group.create_array(
        "tb_coefficients", data=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    )
    group.create_array("pwv_mm", data=np.array([1.0, 2.0]))
    group.create_array("wind_speed_m_s", data=np.array([3.0, 4.0]))
    group.create_array("surface_pressure_mbar", data=np.array([500.0, 501.0]))
    group.create_array("surface_temperature_k", data=np.array([250.0, 251.0]))
    if include_time_index:
        group.create_array("time_index", data=np.array([0, 1], dtype=np.int8))


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
