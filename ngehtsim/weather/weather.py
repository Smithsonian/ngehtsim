###################################################
# imports

from functools import lru_cache
import os
import struct

import numpy as np

import ngehtsim.const_def as const
from ngehtsim.weather.zarr_store import ZarrWeatherStore


###################################################
# function definitions


def read_binary_atm(filename, Ncomps=const.number_of_components):
    """
    Read a stored weather data file containing either opacity or brightness temperature info.

    Args:
      filename (str): The name of the weather data file
      Ncomps (int): The number of PCA component coefficients that have been stored

    Returns:
      (numpy.ndarray): Several arrays containing the dates/times and PCA component coefficients
    """

    with open(filename, "rb") as binary_file:
        contents = bytearray(binary_file.read())

    linelength = int(contents[0:2][0])
    prelength = linelength - (2 * Ncomps)
    nlines = int((len(contents) - 2) / linelength)

    years = np.zeros(nlines)
    months = np.zeros(nlines)
    days = np.zeros(nlines)
    if prelength > 4:
        times = np.zeros(nlines)
    coeffs = np.zeros((nlines, Ncomps))
    for index in range(nlines):
        start = 2 + (linelength * index)
        stop = start + linelength
        line = contents[start:stop]

        years[index] = int(struct.unpack("<h", line[0:2])[0])
        months[index] = int(struct.unpack("b", line[2:3])[0])
        days[index] = int(struct.unpack("b", line[3:4])[0])
        if prelength > 4:
            times[index] = int(struct.unpack("b", line[4:5])[0])
        coeffs[index, :] = np.array(
            struct.unpack("<" + "e" * Ncomps, line[prelength:])
        ).astype(float)

    if prelength > 4:
        return years, months, days, times, coeffs
    return years, months, days, coeffs


def read_binary_weather(filename):
    """
    Read a stored weather data file containing pressure, temperature, wind, or PWV info.

    Args:
      filename (str): The name of the weather data file

    Returns:
      (numpy.ndarray): Several arrays containing the dates/times and weather values read from the file
    """

    with open(filename, "rb") as binary_file:
        contents = bytearray(binary_file.read())

    linelength = int(contents[0:2][0])
    prelength = linelength - 8
    nlines = int((len(contents) - 2) / linelength)

    years = np.zeros(nlines)
    months = np.zeros(nlines)
    days = np.zeros(nlines)
    if prelength > 4:
        times = np.zeros(nlines)
    values = np.zeros(nlines)
    for index in range(nlines):
        start = 2 + (linelength * index)
        stop = start + linelength
        line = contents[start:stop]

        years[index] = int(struct.unpack("<h", line[0:2])[0])
        months[index] = int(struct.unpack("b", line[2:3])[0])
        days[index] = int(struct.unpack("b", line[3:4])[0])
        if prelength > 4:
            times[index] = int(struct.unpack("b", line[4:5])[0])
        values[index] = float(struct.unpack("<d", line[prelength:])[0])

    if prelength > 4:
        return years, months, days, times, values
    return years, months, days, values


@lru_cache(maxsize=1)
def _legacy_pca_bases():
    """Load binary-weather PCA bases only when the legacy backend is used."""

    mean_tau = np.loadtxt(const.path_to_eigenspectra + "/spectrum_mean.txt", unpack=True)
    mean_tb = np.loadtxt(const.path_to_eigenspectra + "_Tb/spectrum_mean.txt", unpack=True)
    tau_components = tuple(
        np.loadtxt(
            const.path_to_eigenspectra + "/spectrum_" + str(index).zfill(4) + ".txt",
            unpack=True,
        )
        for index in range(const.number_of_components)
    )
    tb_components = tuple(
        np.loadtxt(
            const.path_to_eigenspectra + "_Tb/spectrum_" + str(index).zfill(4) + ".txt",
            unpack=True,
        )
        for index in range(const.number_of_components)
    )
    return mean_tau, mean_tb, tau_components, tb_components


def reconstruct_spectrum_tau(coeffs):
    """
    Reconstruct an opacity spectrum from PCA component coefficients.

    Args:
      coeffs (numpy.ndarray): Array containing the PCA component coefficients

    Returns:
      (numpy.ndarray): Array containing the opacity spectrum
    """

    mean_tau, _, tau_components, _ = _legacy_pca_bases()
    reconstructed_spectrum = np.array(mean_tau, dtype=float, copy=True)
    for coefficient, eigenspectrum in zip(coeffs, tau_components):
        reconstructed_spectrum += coefficient * eigenspectrum
    return 10.0**reconstructed_spectrum


def reconstruct_spectrum_Tb(coeffs):
    """
    Reconstruct a brightness temperature spectrum from PCA component coefficients.

    Args:
      coeffs (numpy.ndarray): Array containing the PCA component coefficients

    Returns:
      (numpy.ndarray): Array containing the brightness temperature spectrum
    """

    _, mean_tb, _, tb_components = _legacy_pca_bases()
    reconstructed_spectrum = np.array(mean_tb, dtype=float, copy=True)
    for coefficient, eigenspectrum in zip(coeffs, tb_components):
        reconstructed_spectrum += coefficient * eigenspectrum
    return reconstructed_spectrum


def _parse_month(month):
    monthnums = ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12")
    monthnams = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

    if month in monthnams:
        return monthnums[monthnams.index(month)], month
    monthnum = str(month).zfill(2)
    if monthnum in monthnums:
        return monthnum, monthnams[monthnums.index(monthnum)]
    raise ValueError(
        "Specified month not recognized; please use either a three-letter abbreviation "
        "(e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10)."
    )


def _remove_false_february_entries(years, days, values, monthnam):
    if monthnam != "Feb":
        return years, days, values

    keep = np.ones(len(days), dtype=bool)
    for index in range(1, len(days)):
        if days[index] == days[index - 1]:
            keep[index] = False
    return years[keep], days[keep], values[keep]


def _legacy_records(site, monthnum, monthnam, quantity, path_to_weather):
    filenames = {
        "tau": "tau.txt",
        "tb": "Tb.txt",
        "pressure": "Pbase.txt",
        "temperature": "Tbase.txt",
        "pwv": "PWV.txt",
        "windspeed": "windspeed.txt",
    }
    filename = os.path.join(str(path_to_weather), site, monthnum + monthnam, filenames[quantity])

    if quantity in ("tau", "tb"):
        years, _, days, coefficients = read_binary_atm(filename)
        reconstruct = reconstruct_spectrum_tau if quantity == "tau" else reconstruct_spectrum_Tb
        values = np.array([reconstruct(coefficients[index]) for index in range(len(coefficients))])
    else:
        years, _, days, values = read_binary_weather(filename)

    return _remove_false_february_entries(years, days, values, monthnam)


def _zarr_records(store, site, month, quantity):
    partition = store.read_partition(site, month, cadence="daily")
    if quantity == "tau":
        values = store.reconstruct_tau_spectra(partition)
    elif quantity == "tb":
        values = store.reconstruct_tb_spectra(partition)
    else:
        field_names = {
            "pressure": "surface_pressure_mbar",
            "temperature": "surface_temperature_k",
            "pwv": "pwv_mm",
            "windspeed": "wind_speed_m_s",
        }
        values = getattr(partition, field_names[quantity])
    return partition.year, partition.day, values


def _weather_records(site, month, quantity, path_to_weather, weather_store):
    if weather_store is None:
        monthnum, monthnam = _parse_month(month)
        return _legacy_records(site, monthnum, monthnam, quantity, path_to_weather)
    if not isinstance(weather_store, ZarrWeatherStore):
        raise TypeError("weather_store must be a ZarrWeatherStore instance or None.")
    return _zarr_records(weather_store, site, month, quantity)


def _select_weather_values(values, years, days, form, day, year):
    if form == "exact":
        matches = (years == int(year)) & (days == int(day))
        if not np.any(matches):
            raise Exception("No weather on file for the selected date.")
        return values[matches][0]
    if form == "all":
        return values
    if form == "mean":
        return np.nanmean(values, axis=0)
    if form == "median":
        return np.nanmedian(values, axis=0)
    if form == "good":
        return np.nanpercentile(values, 15.87, axis=0)
    if form == "bad":
        return np.nanpercentile(values, 84.13, axis=0)
    raise ValueError("Weather form must be exact, mean, median, good, bad, or all.")


def _spectrum(site, form, month, day, year, path_to_weather, weather_store, quantity):
    years, days, spectra = _weather_records(
        site, month, quantity, path_to_weather, weather_store
    )
    return _select_weather_values(spectra, years, days, form, day, year)


def _spectrum_frequency(weather_store):
    if weather_store is None:
        return const.spectrum_frequency
    return weather_store.frequency_ghz


def _scalar(site, form, month, day, year, path_to_weather, weather_store, quantity):
    years, days, values = _weather_records(
        site, month, quantity, path_to_weather, weather_store
    )
    return _select_weather_values(values, years, days, form, day, year)


def opacity_spectrum(
    site,
    form="exact",
    month="Apr",
    day=15,
    year=2015,
    path_to_weather=const.path_to_weather,
    weather_store=None,
):
    """Retrieve zenith opacity as a function of frequency.

    ``weather_store`` may be a :class:`ZarrWeatherStore`; when it is omitted,
    this function reads the packaged legacy binary weather data.
    """

    return _spectrum(site, form, month, day, year, path_to_weather, weather_store, "tau")


def opacity(
    site,
    form="exact",
    month="Apr",
    day=15,
    year=2015,
    freq=230.0,
    path_to_weather=const.path_to_weather,
    weather_store=None,
):
    """Retrieve zenith opacity at ``freq`` GHz."""

    if (freq < 0.0) | (freq > 2000.0):
        raise Exception("Specified frequency is outside of the acceptable range (0, 2000) GHz.")

    spectrum = opacity_spectrum(
        site,
        form=form,
        month=month,
        day=day,
        year=year,
        path_to_weather=path_to_weather,
        weather_store=weather_store,
    )
    frequency_ghz = _spectrum_frequency(weather_store)
    if form != "all":
        return np.interp(freq, frequency_ghz, spectrum)
    return np.array([np.interp(freq, frequency_ghz, values) for values in spectrum])


def brightness_temperature_spectrum(
    site,
    form="exact",
    month="Apr",
    day=15,
    year=2015,
    path_to_weather=const.path_to_weather,
    weather_store=None,
):
    """Retrieve zenith brightness temperature as a function of frequency."""

    return _spectrum(site, form, month, day, year, path_to_weather, weather_store, "tb")


def brightness_temperature(
    site,
    form="exact",
    month="Apr",
    day=15,
    year=2015,
    freq=230.0,
    path_to_weather=const.path_to_weather,
    weather_store=None,
):
    """Retrieve zenith brightness temperature at ``freq`` GHz."""

    if (freq < 0.0) | (freq > 2000.0):
        raise Exception("Specified frequency is outside of the acceptable range (0, 2000) GHz.")

    spectrum = brightness_temperature_spectrum(
        site,
        form=form,
        month=month,
        day=day,
        year=year,
        path_to_weather=path_to_weather,
        weather_store=weather_store,
    )
    frequency_ghz = _spectrum_frequency(weather_store)
    if form != "all":
        return np.interp(freq, frequency_ghz, spectrum)
    return np.array([np.interp(freq, frequency_ghz, values) for values in spectrum])


def pressure(
    site,
    form="exact",
    month="Apr",
    day=15,
    year=2015,
    path_to_weather=const.path_to_weather,
    weather_store=None,
):
    """Retrieve surface pressure in mbar."""

    return _scalar(site, form, month, day, year, path_to_weather, weather_store, "pressure")


def temperature(
    site,
    form="exact",
    month="Apr",
    day=15,
    year=2015,
    path_to_weather=const.path_to_weather,
    weather_store=None,
):
    """Retrieve surface temperature in K."""

    return _scalar(site, form, month, day, year, path_to_weather, weather_store, "temperature")


def PWV(
    site,
    form="exact",
    month="Apr",
    day=15,
    year=2015,
    path_to_weather=const.path_to_weather,
    weather_store=None,
):
    """Retrieve precipitable water vapor in mm."""

    return _scalar(site, form, month, day, year, path_to_weather, weather_store, "pwv")


def windspeed(
    site,
    form="exact",
    month="Apr",
    day=15,
    year=2015,
    path_to_weather=const.path_to_weather,
    weather_store=None,
):
    """Retrieve wind speed in m/s."""

    return _scalar(site, form, month, day, year, path_to_weather, weather_store, "windspeed")
