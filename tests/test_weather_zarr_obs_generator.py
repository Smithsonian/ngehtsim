"""Tests for using a Zarr weather store in observation generation."""

import numpy as np
import pytest

import ngehtsim.const_def as const
import ngehtsim.obs.obs_generator as obs_generator_module
import ngehtsim.weather.weather as weather
from ngehtsim.weather.zarr_store import ZarrWeatherStore


ZARR_OBS_SETTINGS = {
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


@pytest.fixture
def store(weather_dataset):
    return ZarrWeatherStore(weather_dataset)


def test_obs_generator_uses_zarr_weather_store(store):
    obsgen = obs_generator_module.obs_generator(
        settings=ZARR_OBS_SETTINGS, weather_store=store
    )
    weather_kwargs = {
        "form": "exact",
        "month": "Apr",
        "day": 11,
        "year": 2017,
        "weather_store": store,
    }
    tau = weather.opacity("ALMA", freq=230.0, **weather_kwargs)
    tb = weather.brightness_temperature("ALMA", freq=230.0, **weather_kwargs)
    windspeed = weather.windspeed("ALMA", **weather_kwargs)
    temperature = weather.temperature("ALMA", **weather_kwargs)
    atmospheric_temperature = (
        tb - (const.T_CMB * np.exp(-tau))
    ) / (1.0 - np.exp(-tau))

    assert obsgen.weather_store is store
    assert obsgen.weather_cadence == "daily"
    assert np.isclose(obsgen.tau_dict["ALMA"], tau)
    assert np.isclose(obsgen.Tb_dict["ALMA"], tb)
    assert np.isclose(obsgen.windspeed_dict["ALMA"], windspeed)
    assert np.isclose(obsgen.Tgnd_dict["ALMA"], temperature)
    assert np.isclose(obsgen.Tatm_dict["ALMA"], atmospheric_temperature)


def test_obs_generator_samples_native_weather_at_observation_times(store):
    obsgen = obs_generator_module.obs_generator(
        settings=ZARR_OBS_SETTINGS,
        weather_store=store,
        weather_cadence="native",
    )
    times = np.array([0.0, 1.5, 3.0])
    context = obsgen.station_context(times)
    samples = store.sample_native(
        "ALMA", year=2017, month="Apr", day=11, utc_hours=times
    )
    expected_tau = np.array([
        np.interp(230.0, store.frequency_ghz, spectrum)
        for spectrum in samples.opacity
    ])
    expected_tb = np.array([
        np.interp(230.0, store.frequency_ghz, spectrum)
        for spectrum in samples.brightness_temperature
    ])
    expected_tatm = (
        expected_tb - (const.T_CMB * np.exp(-expected_tau))
    ) / (1.0 - np.exp(-expected_tau))

    assert np.allclose(context["tau"]["ALMA"], expected_tau)
    assert np.allclose(context["Tatm"]["ALMA"], expected_tatm)
    assert np.allclose(context["Tgnd"]["ALMA"], [250.0, 250.5, 251.0])
    assert np.allclose(context["windspeed"]["ALMA"], [3.0, 3.5, 4.0])


def test_native_weather_defers_static_weather_tables(store, monkeypatch):
    def unexpected_static_lookup(*args, **kwargs):
        raise AssertionError("Native weather initialization must not tabulate static weather.")

    for name in ("opacity", "brightness_temperature", "windspeed", "temperature"):
        monkeypatch.setattr(obs_generator_module.nw, name, unexpected_static_lookup)

    obsgen = obs_generator_module.obs_generator(
        settings=ZARR_OBS_SETTINGS,
        weather_store=store,
        weather_cadence="native",
    )

    assert not obsgen._weather_tables_ready
    context = obsgen.station_context(np.array([0.0, 1.5, 3.0]))
    assert np.all(np.isfinite(context["tau"]["ALMA"]))
    assert not obsgen._weather_tables_ready


def test_native_weather_static_tables_remain_lazily_compatible(store):
    daily = obs_generator_module.obs_generator(
        settings=ZARR_OBS_SETTINGS,
        weather_store=store,
    )
    native = obs_generator_module.obs_generator(
        settings=ZARR_OBS_SETTINGS,
        weather_store=store,
        weather_cadence="native",
    )

    assert not native._weather_tables_ready
    for attribute in ("tau_dict", "Tatm_dict", "Tb_dict", "windspeed_dict", "Tgnd_dict"):
        assert np.isclose(getattr(native, attribute)["ALMA"], getattr(daily, attribute)["ALMA"])
    assert native._weather_tables_ready


def test_obs_generator_rejects_invalid_weather_store():
    with pytest.raises(TypeError, match="weather_store must be a ZarrWeatherStore"):
        obs_generator_module.obs_generator(
            settings=ZARR_OBS_SETTINGS, weather_store="weather.zarr"
        )


@pytest.mark.parametrize("weather_cadence", ["hourly", "three-hourly", None])
def test_obs_generator_rejects_invalid_weather_cadence(weather_cadence):
    with pytest.raises(ValueError, match="weather_cadence"):
        obs_generator_module.obs_generator(
            settings=ZARR_OBS_SETTINGS, weather_cadence=weather_cadence
        )


def test_native_weather_cadence_requires_zarr_store():
    with pytest.raises(ValueError, match="requires a Zarr weather_store"):
        obs_generator_module.obs_generator(
            settings=ZARR_OBS_SETTINGS, weather_cadence="native"
        )


def test_native_weather_station_context_requires_observation_times(store):
    obsgen = obs_generator_module.obs_generator(
        settings=ZARR_OBS_SETTINGS,
        weather_store=store,
        weather_cadence="native",
    )

    with pytest.raises(ValueError, match="require observation times"):
        obsgen.station_context()


def test_symba_export_uses_obs_generator_weather_store(store, tmp_path):
    obsgen = obs_generator_module.obs_generator(
        settings=ZARR_OBS_SETTINGS, weather_store=store
    )
    output = tmp_path / "obsgen.antennas"

    obs_generator_module.export_SYMBA_antennas(
        obsgen, output_filename=output, use_two_letter=False
    )

    fields = output.read_text().splitlines()[1].split()
    assert fields[0] == "ALMA"
    assert np.isclose(float(fields[2]), 1.0)
    assert np.isclose(float(fields[3]), 500.0)
    assert np.isclose(float(fields[4]), 250.0)


def test_symba_export_rejects_native_weather_cadence(store, tmp_path):
    obsgen = obs_generator_module.obs_generator(
        settings=ZARR_OBS_SETTINGS,
        weather_store=store,
        weather_cadence="native",
    )

    with pytest.raises(ValueError, match="do not support native time-varying weather"):
        obs_generator_module.export_SYMBA_antennas(
            obsgen, output_filename=tmp_path / "obsgen.antennas"
        )
