"""Tests for opt-in weather-backend benchmark configuration."""

import sys
from pathlib import Path

import pytest


BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

import benchmark_synthetic_data as bench


def test_legacy_weather_backend_does_not_require_a_zarr_release():
    backends = bench.prepare_weather_backends(None, None)

    assert len(backends) == 1
    assert backends[0]["name"] == "legacy"
    assert backends[0]["store_path"] is None
    assert backends[0]["obs_generator_kwargs"] == {}


def test_zarr_weather_backend_requires_an_explicit_release_path():
    with pytest.raises(SystemExit, match="--weather-store"):
        bench.prepare_weather_backends(["zarr-native"], None)


def test_expanded_benchmark_scenario_is_json_serializable():
    backends = bench.prepare_weather_backends(None, None)
    scenario = bench.expand_scenarios([bench.SCENARIOS[0]], backends)[0]
    definition = bench.scenario_definition(scenario)

    assert scenario["name"] == "eht2017_model_clean__legacy"
    assert scenario["_obs_generator_kwargs"] == {}
    assert definition["weather_backend"] == "legacy"
    assert "_obs_generator_kwargs" not in definition


def test_benchmark_defaults_to_nfft_and_supports_direct_override():
    scenario = bench.SCENARIOS[0]

    overridden = bench.with_transform_backend([scenario], "direct")

    assert scenario["settings"]["ttype"] == "nfft"
    assert overridden[0]["settings"]["ttype"] == "direct"
    assert scenario["settings"]["ttype"] == "nfft"
