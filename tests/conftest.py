#######################################################
# imports

import os

import numpy as np
import pytest

#######################################################
# configuration testing

SLOW_EXAMPLE_TESTS = [
    "test_SYMBA_export.py",
    "test_basic.py",
    "test_flux_calibration.py",
    "test_generate_observation.py",
    "test_generate_observation_flagging.py",
    "test_generate_observation_instrumental_corruptions.py",
    "test_generate_observation_multi-frequency.py",
    "test_generate_observation_space.py",
    "test_generate_observation_with_FPT.py",
    "test_weather.py",
]

OPTIONAL_TESTS = [
    "test_generate_observation_using_ngEHTforecast.py",
]

collect_ignore = []

if os.environ.get("NGEHTSIM_RUN_SLOW_TESTS") != "1":
    collect_ignore.extend(SLOW_EXAMPLE_TESTS)

if os.environ.get("NGEHTSIM_RUN_OPTIONAL_TESTS") != "1":
    collect_ignore.extend(OPTIONAL_TESTS)


@pytest.fixture
def weather_dataset(tmp_path):
    """Create a minimal valid Zarr weather release for unit tests."""

    zarr = pytest.importorskip("zarr")
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
    _write_weather_records(daily, include_time_index=False)
    native = root.create_group("sites/ALMA/months/04/native")
    _write_weather_records(native, include_time_index=True)
    return path


@pytest.fixture
def weather_dataset_v02(weather_dataset):
    """Add schema-v0.2 physical native-summary products to the test release."""

    root = pytest.importorskip("zarr").open_group(weather_dataset, mode="r+")
    root.attrs.update(
        {
            "schema_version": "0.2.0",
            "dataset_id": "test-weather-v0.2.0",
            "native_summary_forms": ["mean", "median", "good", "bad"],
        }
    )
    native = root["sites/ALMA/months/04/native"]
    tau_coefficients = np.asarray(native["tau_coefficients"][:], dtype=float)
    tb_coefficients = np.asarray(native["tb_coefficients"][:], dtype=float)
    values = {
        "opacity": np.power(
            10.0,
            np.asarray(root["pca/tau/mean"][:])
            + tau_coefficients @ np.asarray(root["pca/tau/components"][:]),
        ),
        "brightness_temperature": (
            np.asarray(root["pca/tb/mean"][:])
            + tb_coefficients @ np.asarray(root["pca/tb/components"][:])
        ),
        "pwv_mm": np.asarray(native["pwv_mm"][:]),
        "wind_speed_m_s": np.asarray(native["wind_speed_m_s"][:]),
        "surface_pressure_mbar": np.asarray(native["surface_pressure_mbar"][:]),
        "surface_temperature_k": np.asarray(native["surface_temperature_k"][:]),
    }
    reducers = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "good": lambda source, axis: np.nanpercentile(source, 15.87, axis=axis),
        "bad": lambda source, axis: np.nanpercentile(source, 84.13, axis=axis),
    }
    time_index = np.asarray(native["time_index"][:])
    for form, reducer in reducers.items():
        group = root.create_group("sites/ALMA/months/04/native_summary/{0}".format(form))
        for name, source in values.items():
            group.create_array(
                name,
                data=np.asarray(
                    [reducer(source[time_index == index], axis=0) for index in range(8)]
                ),
            )
    return weather_dataset


def _write_weather_records(group, include_time_index):
    if include_time_index:
        time_index = np.tile(np.arange(8, dtype=np.int8), 2)
        sample_index = np.arange(len(time_index), dtype=float)
        group.create_array("year", data=np.full(len(time_index), 2017, dtype=np.int16))
        group.create_array("day", data=np.repeat([11, 12], 8).astype(np.int8))
        group.create_array(
            "tau_coefficients",
            data=np.column_stack((sample_index, np.zeros(len(sample_index)))).astype(np.float16),
        )
        group.create_array(
            "tb_coefficients",
            data=np.column_stack((sample_index, np.zeros(len(sample_index)))).astype(np.float16),
        )
        group.create_array("pwv_mm", data=1.0 + sample_index)
        group.create_array("wind_speed_m_s", data=3.0 + sample_index)
        group.create_array("surface_pressure_mbar", data=500.0 + sample_index)
        group.create_array("surface_temperature_k", data=250.0 + sample_index)
        group.create_array("time_index", data=time_index)
        return

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
