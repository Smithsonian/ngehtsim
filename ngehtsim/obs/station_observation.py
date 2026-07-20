###################################################
# imports

from dataclasses import dataclass

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun

import ngehtsim.const_def as const
import ngehtsim.obs.observation_geometry as observation_geometry
from ngehtsim.obs.visibility_dataset import StationTable, VisibilityDataset

###################################################
# helpers


def _cache_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return tuple(_cache_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_cache_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _cache_value(item)) for key, item in value.items()))
    return value


def _station_weather_values(value, count, name):
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        return np.full(count, values, dtype=float), True
    if values.shape != (count,):
        raise ValueError(
            "Station {0} weather must be a scalar or one value per observation row.".format(name)
        )
    return values, False


@dataclass(frozen=True)
class _StationRows:
    t1: np.ndarray
    t2: np.ndarray
    times: np.ndarray
    reference_mjd: float
    bandwidth_hz: float
    elevation1_rad: np.ndarray
    elevation2_rad: np.ndarray
    parallactic_angle1_rad: np.ndarray
    parallactic_angle2_rad: np.ndarray


def _station_context_key(station_context):
    return tuple(
        (
            site,
            _cache_value(station_context["bandwidth_hz"][site]),
            _cache_value(station_context["mount_type"][site]),
            _cache_value(station_context["feed_angle"][site]),
            _cache_value(station_context["polarization_basis"][site]),
        )
        for site in tuple(station_context["sites"])
    )


def _station_metadata_key(source, ra, dec, rf, bandwidth_hz, mjd, times, t1, t2,
                          station_context):
    return (
        _cache_value(source),
        _cache_value(ra),
        _cache_value(dec),
        _cache_value(rf),
        _cache_value(bandwidth_hz),
        _cache_value(mjd),
        _cache_value(times),
        _cache_value(t1),
        _cache_value(t2),
        _station_context_key(station_context),
    )


def station_metadata_cache_key(obs, station_context):
    return _station_metadata_key(
        getattr(obs, "source", None),
        getattr(obs, "ra", None),
        getattr(obs, "dec", None),
        getattr(obs, "rf", None),
        getattr(obs, "bw", None),
        getattr(obs, "mjd", None),
        obs.data["time"],
        obs.data["t1"],
        obs.data["t2"],
        station_context,
    )


def _station_rows_from_obsdata(obs):
    geometry = observation_geometry.ground_station_geometry(obs)
    if geometry is None:
        els = obs.unpack(["el1", "el2"], ang_unit="rad")
        pars = obs.unpack(["par_ang1", "par_ang2"], ang_unit="rad")
        elevation1 = els["el1"]
        elevation2 = els["el2"]
        parallactic_angle1 = pars["par_ang1"]
        parallactic_angle2 = pars["par_ang2"]
    else:
        elevation1 = geometry.elevation1_rad
        elevation2 = geometry.elevation2_rad
        parallactic_angle1 = geometry.parallactic_angle1_rad
        parallactic_angle2 = geometry.parallactic_angle2_rad

    return _StationRows(
        t1=np.asarray(obs.data["t1"]),
        t2=np.asarray(obs.data["t2"]),
        times=np.asarray(obs.data["time"], dtype=float),
        reference_mjd=float(obs.mjd),
        bandwidth_hz=float(obs.bw),
        elevation1_rad=np.asarray(elevation1, dtype=float),
        elevation2_rad=np.asarray(elevation2, dtype=float),
        parallactic_angle1_rad=np.asarray(parallactic_angle1, dtype=float),
        parallactic_angle2_rad=np.asarray(parallactic_angle2, dtype=float),
    )


def _dataset_reference_mjd(dataset, reference_mjd):
    if reference_mjd is not None:
        return float(reference_mjd)
    if not dataset.row_count:
        raise ValueError("Native station terms require at least one visibility row.")
    return float(np.floor(np.min(dataset.time_mjd)))


def _dataset_station_names(dataset):
    names = np.asarray(dataset.stations.names)
    return names[dataset.antenna1], names[dataset.antenna2]


def _station_rows_from_dataset(dataset, reference_mjd):
    if not isinstance(dataset, VisibilityDataset):
        raise TypeError("dataset must be a VisibilityDataset instance.")
    if dataset.channel_count != 1:
        raise ValueError("Native station terms currently require exactly one spectral channel.")

    reference_mjd = _dataset_reference_mjd(dataset, reference_mjd)
    geometry = observation_geometry.station_geometry_from_rows(
        dataset.stations.position_itrs_m,
        dataset.time_mjd,
        dataset.antenna1,
        dataset.antenna2,
        dataset.ra_hours,
        dataset.dec_degrees,
    )
    if geometry is None:
        raise ValueError("Native station terms do not yet support spacecraft stations.")

    t1, t2 = _dataset_station_names(dataset)
    return _StationRows(
        t1=t1,
        t2=t2,
        times=(np.asarray(dataset.time_mjd, dtype=float) - reference_mjd) * 24.0,
        reference_mjd=reference_mjd,
        bandwidth_hz=float(dataset.channel_bandwidth_hz[0]),
        elevation1_rad=geometry.elevation1_rad,
        elevation2_rad=geometry.elevation2_rad,
        parallactic_angle1_rad=geometry.parallactic_angle1_rad,
        parallactic_angle2_rad=geometry.parallactic_angle2_rad,
    )


def _station_metadata_from_rows(rows, station_context):
    t1 = rows.t1
    t2 = rows.t2
    sites_obs = np.unique(np.concatenate((t1, t2)))
    elevation1 = rows.elevation1_rad
    elevation2 = rows.elevation2_rad
    feed_elevation1 = np.zeros_like(elevation1)
    feed_elevation2 = np.zeros_like(elevation2)
    feed_parallactic1 = np.zeros_like(elevation1)
    feed_parallactic2 = np.zeros_like(elevation2)
    feed_offset1 = np.zeros_like(elevation1)
    feed_offset2 = np.zeros_like(elevation2)
    bandwidth1 = np.zeros_like(elevation1)
    bandwidth2 = np.zeros_like(elevation2)
    site_masks = {}

    for site in sites_obs:
        index1 = t1 == site
        index2 = t2 == site
        site_masks[site] = (index1, index2)
        bandwidth = station_context["bandwidth_hz"][site]
        if bandwidth is None:
            bandwidth = rows.bandwidth_hz
        bandwidth1[index1] = bandwidth
        bandwidth2[index2] = bandwidth

        mount_type = station_context["mount_type"][site]
        feed_angle = station_context["feed_angle"][site]
        feed_elevation1[index1] = const.mount_type_dict[mount_type]["f_el"]
        feed_elevation2[index2] = const.mount_type_dict[mount_type]["f_el"]
        feed_parallactic1[index1] = const.mount_type_dict[mount_type]["f_par"]
        feed_parallactic2[index2] = const.mount_type_dict[mount_type]["f_par"]
        feed_offset1[index1] = feed_angle
        feed_offset2[index2] = feed_angle

    return {
        "_rows": rows,
        "sites_obs": sites_obs,
        "site_masks": site_masks,
        "el1": elevation1,
        "el2": elevation2,
        "par1": rows.parallactic_angle1_rad,
        "par2": rows.parallactic_angle2_rad,
        "tuniq": np.unique(rows.times),
        "bw1": bandwidth1,
        "bw2": bandwidth2,
        "f_el1": feed_elevation1,
        "f_el2": feed_elevation2,
        "f_par1": feed_parallactic1,
        "f_par2": feed_parallactic2,
        "phi_off1": feed_offset1,
        "phi_off2": feed_offset2,
    }


def station_metadata(obs, station_context, cache=None):
    key = station_metadata_cache_key(obs, station_context)
    if cache is not None and key in cache:
        return cache[key]
    metadata = _station_metadata_from_rows(
        _station_rows_from_obsdata(obs),
        station_context,
    )
    if cache is not None:
        cache[key] = metadata
    return metadata


def station_metadata_for_dataset(dataset, station_context, reference_mjd=None, cache=None):
    if not isinstance(dataset, VisibilityDataset):
        raise TypeError("dataset must be a VisibilityDataset instance.")
    if dataset.channel_count != 1:
        raise ValueError("Native station terms currently require exactly one spectral channel.")

    reference_mjd = _dataset_reference_mjd(dataset, reference_mjd)
    t1, t2 = _dataset_station_names(dataset)
    times = (np.asarray(dataset.time_mjd, dtype=float) - reference_mjd) * 24.0
    key = _station_metadata_key(
        dataset.source,
        dataset.ra_hours,
        dataset.dec_degrees,
        dataset.channel_frequency_hz[0],
        dataset.channel_bandwidth_hz[0],
        reference_mjd,
        times,
        t1,
        t2,
        station_context,
    )
    if cache is not None and key in cache:
        return cache[key]
    metadata = _station_metadata_from_rows(
        _station_rows_from_dataset(dataset, reference_mjd),
        station_context,
    )
    if cache is not None:
        cache[key] = metadata
    return metadata


def _updated_station_table(stations, sefd_r_jy, sefd_l_jy, leakage_r, leakage_l):
    return StationTable(
        names=stations.names,
        position_itrs_m=stations.position_itrs_m,
        sefd_r_jy=sefd_r_jy,
        sefd_l_jy=sefd_l_jy,
        leakage_r=leakage_r,
        leakage_l=leakage_l,
        feed_rotation_par=stations.feed_rotation_par,
        feed_rotation_elev=stations.feed_rotation_elev,
        feed_rotation_offset_deg=stations.feed_rotation_offset_deg,
    )


def _station_terms_from_rows(rows, metadata, F0, station_context, stations, rng,
                             gainamp=0.04, leakamp=0.1, addgains=True,
                             addleakage=False, flagwind=True, flagday=False,
                             flagsun=True, solar_angle=None, verbosity=0,
                             windspeed_sefd_modifier=None):
    t1 = rows.t1
    t2 = rows.t2
    times = rows.times
    sites_obs = metadata["sites_obs"]
    site_masks = metadata["site_masks"]
    elevation1 = metadata["el1"]
    elevation2 = metadata["el2"]
    parallactic_angle1 = metadata["par1"]
    parallactic_angle2 = metadata["par2"]
    unique_times = metadata["tuniq"]

    tau1 = np.zeros_like(elevation1)
    tau2 = np.zeros_like(elevation2)
    brightness1 = np.zeros_like(elevation1)
    brightness2 = np.zeros_like(elevation2)
    system_temperature1 = np.zeros_like(elevation1)
    system_temperature2 = np.zeros_like(elevation2)
    sefd1 = np.zeros_like(elevation1)
    sefd2 = np.zeros_like(elevation2)
    if addgains:
        gain_amplitude1_r = np.zeros_like(elevation1)
        gain_amplitude2_r = np.zeros_like(elevation2)
        gain_phase1_r = np.zeros_like(elevation1)
        gain_phase2_r = np.zeros_like(elevation2)
        gain_amplitude1_l = np.zeros_like(elevation1)
        gain_amplitude2_l = np.zeros_like(elevation2)
        gain_phase1_l = np.zeros_like(elevation1)
        gain_phase2_l = np.zeros_like(elevation2)
    if addleakage:
        leakage1_r = np.zeros_like(elevation1, dtype=complex)
        leakage2_r = np.zeros_like(elevation2, dtype=complex)
        leakage1_l = np.zeros_like(elevation1, dtype=complex)
        leakage2_l = np.zeros_like(elevation2, dtype=complex)

    flagged_sites = []
    uptime_mask = np.ones(len(times), dtype=bool)
    sefd_r_jy = np.array(stations.sefd_r_jy, copy=True)
    sefd_l_jy = np.array(stations.sefd_l_jy, copy=True)
    leakage_r = np.array(stations.leakage_r, copy=True)
    leakage_l = np.array(stations.leakage_l, copy=True)
    station_index = {name: index for index, name in enumerate(stations.names)}

    for site in sites_obs:
        tau_zenith, _ = _station_weather_values(
            station_context["tau"][site], len(times), "opacity"
        )
        atmospheric_temperature, _ = _station_weather_values(
            station_context["Tatm"][site], len(times), "atmospheric temperature"
        )
        ground_temperature, _ = _station_weather_values(
            station_context["Tgnd"][site], len(times), "ground temperature"
        )
        wind_speed, windspeed_is_scalar = _station_weather_values(
            station_context["windspeed"][site], len(times), "wind speed"
        )
        effective_area = station_context["effective_area"][site]
        wind_loading = station_context["wind_loading"][site]
        site_rows = (t1 == site) | (t2 == site)

        if flagwind:
            high_wind = site_rows & (wind_speed > wind_loading["shutdown"])
            if np.any(high_wind) and windspeed_is_scalar:
                flagged_sites.append(site)
                if verbosity > 0:
                    print(site + " cannot observe because of high wind.")
            elif np.any(high_wind):
                uptime_mask[high_wind] = False
                if verbosity > 0:
                    print(site + " cannot observe at some timestamps because of high wind.")

        if flagday and site != "space":
            location = EarthLocation.from_geodetic(
                const.known_longitudes[site],
                const.known_latitudes[site],
                height=const.known_elevations[site],
            )
            timehere = Time(
                rows.reference_mjd + 2400000.5 + (times / 24.0),
                format="jd",
            )
            sun_altaz = get_sun(timehere).transform_to(AltAz(obstime=timehere, location=location))
            uptime_mask[site_rows & (sun_altaz.alt.value > 0.0)] = False

        if flagsun and solar_angle < station_context["solar_avoidance"][site]:
            flagged_sites.append(site)
            if verbosity > 0:
                print(site + " cannot observe because the source is too close to the Sun.")

        if site in station_context["station_uptimes"]:
            start, stop = station_context["station_uptimes"][site]
            uptime_mask[site_rows & (times < start)] = False
            uptime_mask[site_rows & (times > stop)] = False

        index1, index2 = site_masks[site]
        if site != "space":
            tau1[index1] = tau_zenith[index1] / np.cos((np.pi / 2.0) - elevation1[index1])
            tau2[index2] = tau_zenith[index2] / np.cos((np.pi / 2.0) - elevation2[index2])
        else:
            tau1[index1] = 0.0
            tau2[index2] = 0.0

        source_temperature = (F0 * effective_area) / (2.0 * const.k)
        if site != "space":
            brightness1[index1] = (
                (const.T_CMB + source_temperature) * np.exp(-tau1[index1])
                + atmospheric_temperature[index1] * (1.0 - np.exp(-tau1[index1]))
            )
            brightness2[index2] = (
                (const.T_CMB + source_temperature) * np.exp(-tau2[index2])
                + atmospheric_temperature[index2] * (1.0 - np.exp(-tau2[index2]))
            )
        else:
            brightness1[index1] = const.T_CMB + source_temperature
            brightness2[index2] = const.T_CMB + source_temperature

        receiver_temperature = station_context["receiver_temperature"][site]
        sideband_ratio = station_context["sideband_ratio"][site]
        system_temperature1[index1] = (
            receiver_temperature
            + const.eta_ff * brightness1[index1]
            + (1.0 - const.eta_ff) * ground_temperature[index1]
        ) * (1.0 + sideband_ratio)
        system_temperature2[index2] = (
            receiver_temperature
            + const.eta_ff * brightness2[index2]
            + (1.0 - const.eta_ff) * ground_temperature[index2]
        ) * (1.0 + sideband_ratio)
        sefd1[index1] = (2.0 * const.k * system_temperature1[index1]) / effective_area
        sefd2[index2] = (2.0 * const.k * system_temperature2[index2]) / effective_area
        if flagwind:
            sefd1[index1] *= windspeed_sefd_modifier(
                wind_speed[index1], wind_loading["v0"], wind_loading["w"]
            )
            sefd2[index2] *= windspeed_sefd_modifier(
                wind_speed[index2], wind_loading["v0"], wind_loading["w"]
            )

        try:
            station_here = station_index[str(site)]
        except KeyError:
            raise ValueError("Station terms reference unknown station: {0}".format(site))
        sefd_here = np.mean(np.concatenate((sefd1[index1], sefd2[index2])))
        sefd_r_jy[station_here] = sefd_here
        sefd_l_jy[station_here] = sefd_here

        if addgains:
            for time in unique_times:
                index1here = (times == time) & (t1 == site)
                index2here = (times == time) & (t2 == site)
                gain_amplitude = 10.0 ** (gainamp * rng.normal(0.0, 1.0))
                gain_phase = rng.uniform(-np.pi, np.pi)
                gain_amplitude1_r[index1here] = gain_amplitude
                gain_amplitude2_r[index2here] = gain_amplitude
                gain_phase1_r[index1here] = gain_phase
                gain_phase2_r[index2here] = gain_phase
                gain_amplitude1_l[index1here] = gain_amplitude
                gain_amplitude2_l[index2here] = gain_amplitude
                gain_phase1_l[index1here] = gain_phase
                gain_phase2_l[index2here] = gain_phase

        if addleakage:
            leakage_r_here = leakamp * rng.normal(0.0, 1.0) + 1.0j * leakamp * rng.normal(0.0, 1.0)
            leakage_l_here = leakamp * rng.normal(0.0, 1.0) + 1.0j * leakamp * rng.normal(0.0, 1.0)
            leakage1_r[index1] = leakage_r_here
            leakage2_r[index2] = leakage_r_here
            leakage1_l[index1] = leakage_l_here
            leakage2_l[index2] = leakage_l_here
            if site != "space":
                leakage_r[station_here] = leakage_r_here
                leakage_l[station_here] = leakage_l_here

    terms = {
        "t1": t1,
        "t2": t2,
        "el1": elevation1,
        "el2": elevation2,
        "par1": parallactic_angle1,
        "par2": parallactic_angle2,
        "times": times,
        "tau1": tau1,
        "tau2": tau2,
        "Tb1": brightness1,
        "Tb2": brightness2,
        "Tsys1": system_temperature1,
        "Tsys2": system_temperature2,
        "SEFD1": sefd1,
        "SEFD2": sefd2,
        "bw1": metadata["bw1"],
        "bw2": metadata["bw2"],
        "f_el1": metadata["f_el1"],
        "f_el2": metadata["f_el2"],
        "f_par1": metadata["f_par1"],
        "f_par2": metadata["f_par2"],
        "phi_off1": metadata["phi_off1"],
        "phi_off2": metadata["phi_off2"],
        "flagsites": flagged_sites,
        "uptime_mask": uptime_mask,
    }
    if addgains:
        terms.update({
            "gainamp1R": gain_amplitude1_r,
            "gainamp2R": gain_amplitude2_r,
            "gainphase1R": gain_phase1_r,
            "gainphase2R": gain_phase2_r,
            "gainamp1L": gain_amplitude1_l,
            "gainamp2L": gain_amplitude2_l,
            "gainphase1L": gain_phase1_l,
            "gainphase2L": gain_phase2_l,
        })
    if addleakage:
        terms.update({
            "leak1R": leakage1_r,
            "leak2R": leakage2_r,
            "leak1L": leakage1_l,
            "leak2L": leakage2_l,
        })

    return terms, _updated_station_table(
        stations,
        sefd_r_jy,
        sefd_l_jy,
        leakage_r,
        leakage_l,
    )


def _apply_station_table_to_array(array, stations):
    station_index = {str(name): index for index, name in enumerate(stations.names)}
    for field, values in (
        ("sefdr", stations.sefd_r_jy),
        ("sefdl", stations.sefd_l_jy),
        ("dr", stations.leakage_r),
        ("dl", stations.leakage_l),
    ):
        updated = np.array(array.tarr[field], copy=True)
        for index, site in enumerate(array.tarr["site"]):
            if str(site) in station_index:
                updated[index] = values[station_index[str(site)]]
        array.tarr[field] = updated


def station_terms(obs, F0, station_context, array, rng, gainamp=0.04, leakamp=0.1,
                  addgains=True, addleakage=False, flagwind=True, flagday=False,
                  flagsun=True, allow_mixed_basis=False, solar_angle=None,
                  verbosity=0, windspeed_sefd_modifier=None, cache=None):
    """Calculate legacy Obsdata station terms through the native row kernel."""

    if allow_mixed_basis:
        raise NotImplementedError(
            "Mixed-polarization station terms require VisibilityDataset support."
        )

    metadata = station_metadata(obs, station_context, cache=cache)
    terms, stations = _station_terms_from_rows(
        metadata["_rows"],
        metadata,
        F0,
        station_context,
        StationTable.from_ehtim_tarr(array.tarr),
        rng,
        gainamp=gainamp,
        leakamp=leakamp,
        addgains=addgains,
        addleakage=addleakage,
        flagwind=flagwind,
        flagday=flagday,
        flagsun=flagsun,
        solar_angle=solar_angle,
        verbosity=verbosity,
        windspeed_sefd_modifier=windspeed_sefd_modifier,
    )
    _apply_station_table_to_array(array, stations)
    return terms


def station_terms_for_dataset(dataset, F0, station_context, rng, gainamp=0.04,
                              leakamp=0.1, addgains=True, addleakage=False,
                              flagwind=True, flagday=False, flagsun=True,
                              solar_angle=None, verbosity=0,
                              windspeed_sefd_modifier=None, reference_mjd=None,
                              cache=None):
    """Calculate native station terms and return an updated StationTable."""

    metadata = station_metadata_for_dataset(
        dataset,
        station_context,
        reference_mjd=reference_mjd,
        cache=cache,
    )
    return _station_terms_from_rows(
        metadata["_rows"],
        metadata,
        F0,
        station_context,
        dataset.stations,
        rng,
        gainamp=gainamp,
        leakamp=leakamp,
        addgains=addgains,
        addleakage=addleakage,
        flagwind=flagwind,
        flagday=flagday,
        flagsun=flagsun,
        solar_angle=solar_angle,
        verbosity=verbosity,
        windspeed_sefd_modifier=windspeed_sefd_modifier,
    )
