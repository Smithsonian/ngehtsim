"""Opt-in acceptance test for an externally distributed Zarr weather release."""

import os

import ehtim as eh
import numpy as np
import pytest

import ngehtsim.obs.obs_generator as obs_generator_module
from ngehtsim.obs.station_effects import StationCorruptionModel
import ngehtsim.weather.weather as weather
from ngehtsim.weather.zarr_store import SUPPORTED_SCHEMA_VERSIONS, ZarrWeatherStore


DEFAULT_DATASET_ID = "ngehtsim-weather-merra2-3hour-v0.1.0"
EXPECTED_DATASET_ID_ENVIRONMENT_VARIABLE = "NGEHTSIM_WEATHER_ZARR_DATASET_ID"

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

OBS_GENERATOR_SETTINGS = {
    "source": "M87",
    "sites": ["ALMA"],
    "weather": "exact",
    "month": "Apr",
    "day": 11,
    "year": 2017,
    "weather_day": 11,
    "weather_year": 2017,
    "frequency": 230.0,
    "t_start": 0.0,
    "dt": 3.0,
    "t_int": 600.0,
    "t_rest": 1200.0,
    "random_seed": 1,
}


@pytest.fixture(scope="module")
def external_store():
    path = os.environ.get("NGEHTSIM_WEATHER_ZARR")
    if path is None:
        pytest.skip("Set NGEHTSIM_WEATHER_ZARR to run the external weather release test.")
    return ZarrWeatherStore(path)


@pytest.mark.external_weather
def test_local_weather_release_has_expected_schema_and_alma_coverage(external_store):
    assert external_store.attributes["schema_version"] in SUPPORTED_SCHEMA_VERSIONS
    expected_dataset_id = os.environ.get(
        EXPECTED_DATASET_ID_ENVIRONMENT_VARIABLE,
        DEFAULT_DATASET_ID,
    )
    assert external_store.dataset_id == expected_dataset_id
    assert len(external_store.sites) == 141

    daily = external_store.read_partition("ALMA", "Apr", cadence="daily")
    native = external_store.read_partition("ALMA", "Apr", cadence="native")
    assert daily.record_count * 8 == native.record_count
    assert np.all(np.isfinite(external_store.reconstruct_tau_spectra(daily)))
    assert np.all(np.isfinite(external_store.reconstruct_tb_spectra(daily)))


@pytest.mark.external_weather
def test_schema_v02_reads_precomputed_native_summaries(external_store, monkeypatch):
    if external_store.attributes["schema_version"] != "0.2.0":
        pytest.skip("Precomputed native summaries were introduced in schema v0.2.0.")

    external_store._native_summary_cache.clear()

    def fail_if_legacy_summary_fallback_is_used(*args, **kwargs):
        raise AssertionError("Schema v0.2.0 must not reconstruct native summary products.")

    monkeypatch.setattr(
        external_store,
        "_summarize_native_partition",
        fail_if_legacy_summary_fallback_is_used,
    )
    samples = external_store.sample_native(
        "ALMA",
        year=2017,
        month="Apr",
        day=11,
        utc_hours=[0.0, 1.5, 3.0],
        form="median",
    )

    assert samples.opacity.shape == (3, len(external_store.frequency_ghz))
    assert np.all(np.isfinite(samples.opacity))


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


@pytest.mark.external_weather
def test_obs_generator_zarr_weather_matches_packaged_daily_weather(external_store):
    legacy = obs_generator_module.obs_generator(settings=OBS_GENERATOR_SETTINGS)
    zarr = obs_generator_module.obs_generator(
        settings=OBS_GENERATOR_SETTINGS, weather_store=external_store
    )

    for attribute in ("tau_dict", "Tatm_dict", "Tb_dict", "windspeed_dict", "Tgnd_dict"):
        assert np.isclose(getattr(zarr, attribute)["ALMA"], getattr(legacy, attribute)["ALMA"])


@pytest.mark.external_weather
def test_obs_generator_uses_native_weather_for_observation_terms(external_store):
    settings = dict(OBS_GENERATOR_SETTINGS)
    settings.pop("sites")
    settings["array"] = "EHT2017"
    model = eh.model.Model().add_circ_gauss(F0=1.0, FWHM=40.0 * eh.RADPERUAS)
    obsgen = obs_generator_module.obs_generator(
        settings=settings,
        weather_store=external_store,
        weather_cadence="native",
        weight=1,
    )

    obs = obsgen.make_obs(
        model,
        effects=StationCorruptionModel(
            thermal_noise=False,
            common_gain=None,
            flag_wind=False,
            flag_sun=False,
        ),
    )

    assert len(obs.data) > 0
    assert np.all(np.isfinite(obsgen.tau1))
    assert np.all(np.isfinite(obsgen.SEFD1))


@pytest.mark.external_weather
def test_native_sampling_interpolates_across_month_boundary(external_store):
    april = external_store.read_partition("ALMA", "Apr", cadence="native")
    may = external_store.read_partition("ALMA", "May", cadence="native")
    april_mask = (april.year == 2017) & (april.day == 30) & (april.time_index == 7)
    may_mask = (may.year == 2017) & (may.day == 1) & (may.time_index == 0)
    april_tau = external_store.reconstruct_tau_spectra(april)[april_mask][0]
    may_tau = external_store.reconstruct_tau_spectra(may)[may_mask][0]

    samples = external_store.sample_native(
        "ALMA", year=2017, month="Apr", day=30, utc_hours=[21.0, 22.5, 24.0]
    )

    assert np.allclose(samples.opacity[0], april_tau)
    assert np.allclose(samples.opacity[1], (april_tau + may_tau) / 2.0)
    assert np.allclose(samples.opacity[2], may_tau)
    assert np.all(np.isfinite(samples.surface_pressure_mbar))
