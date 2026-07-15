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

    assert np.array_equal(partition.time_index, np.tile(np.arange(8), 2))


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


def test_store_linearly_samples_exact_native_weather(weather_dataset):
    samples = ZarrWeatherStore(weather_dataset).sample_native(
        "ALMA", year=2017, month="Apr", day=11, utc_hours=[0.0, 1.5, 3.0]
    )

    assert np.allclose(samples.surface_pressure_mbar, [500.0, 500.5, 501.0])
    assert np.allclose(samples.surface_temperature_k, [250.0, 250.5, 251.0])
    assert np.allclose(samples.wind_speed_m_s, [3.0, 3.5, 4.0])
    assert np.allclose(samples.pwv_mm, [1.0, 1.5, 2.0])
    assert np.allclose(samples.opacity[:, 0], [1.0, 5.5, 10.0])
    assert np.allclose(samples.brightness_temperature[:, 0], [10.0, 10.5, 11.0])
    assert not samples.opacity.flags.writeable


def test_store_linearly_samples_across_native_day_boundary(weather_dataset):
    samples = ZarrWeatherStore(weather_dataset).sample_native(
        "ALMA", year=2017, month=4, day=11, utc_hours=[21.0, 22.5, 24.0]
    )

    assert np.allclose(samples.surface_pressure_mbar, [507.0, 507.5, 508.0])


@pytest.mark.parametrize(
    "form,reducer",
    [
        ("mean", np.nanmean),
        ("median", np.nanmedian),
        ("good", lambda values: np.nanpercentile(values, 15.87)),
        ("bad", lambda values: np.nanpercentile(values, 84.13)),
    ],
)
def test_store_summarizes_native_weather_by_utc_time(weather_dataset, form, reducer):
    samples = ZarrWeatherStore(weather_dataset).sample_native(
        "ALMA", year=2017, month="Apr", day=11, utc_hours=1.5, form=form
    )

    assert samples.surface_pressure_mbar.shape == (1,)
    expected = (reducer([500.0, 508.0]) + reducer([501.0, 509.0])) / 2.0
    assert np.allclose(samples.surface_pressure_mbar, [expected])


@pytest.mark.parametrize("form", ["all", "random", "unknown"])
def test_store_rejects_unsupported_native_weather_forms(weather_dataset, form):
    with pytest.raises(ValueError, match="Unsupported native weather form"):
        ZarrWeatherStore(weather_dataset).sample_native(
            "ALMA", year=2017, month="Apr", day=11, utc_hours=0.0, form=form
        )


def test_store_rejects_native_interpolation_across_missing_records(weather_dataset):
    with pytest.raises(WeatherStoreError, match="interpolation endpoints are unavailable"):
        ZarrWeatherStore(weather_dataset).sample_native(
            "ALMA", year=2017, month="Apr", day=11, utc_hours=48.0
        )


def test_store_returns_exact_native_samples_without_adjacent_records(weather_dataset):
    root = zarr.open_group(weather_dataset, mode="r+")
    native = root["sites/ALMA/months/04/native"]
    native["time_index"][6] = 7
    native["day"][6] = 13

    store = ZarrWeatherStore(weather_dataset)
    samples = store.sample_native(
        "ALMA", year=2017, month="Apr", day=11, utc_hours=21.0
    )

    assert np.allclose(samples.surface_pressure_mbar, [507.0])
    with pytest.raises(WeatherStoreError, match="would cross missing records"):
        store.sample_native("ALMA", year=2017, month="Apr", day=11, utc_hours=19.5)


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
