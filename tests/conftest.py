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


def _write_weather_records(group, include_time_index):
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
