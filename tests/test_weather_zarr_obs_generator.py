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
    assert np.isclose(obsgen.tau_dict["ALMA"], tau)
    assert np.isclose(obsgen.Tb_dict["ALMA"], tb)
    assert np.isclose(obsgen.windspeed_dict["ALMA"], windspeed)
    assert np.isclose(obsgen.Tgnd_dict["ALMA"], temperature)
    assert np.isclose(obsgen.Tatm_dict["ALMA"], atmospheric_temperature)


def test_obs_generator_rejects_invalid_weather_store():
    with pytest.raises(TypeError, match="weather_store must be a ZarrWeatherStore"):
        obs_generator_module.obs_generator(
            settings=ZARR_OBS_SETTINGS, weather_store="weather.zarr"
        )


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
