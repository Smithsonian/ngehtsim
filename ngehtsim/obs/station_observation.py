###################################################
# imports

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun

import ngehtsim.const_def as const
import ngehtsim.obs.observation_geometry as observation_geometry

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
            'Station {0} weather must be a scalar or one value per observation row.'.format(name)
        )
    return values, False


def station_metadata_cache_key(obs, station_context):
    sites = tuple(station_context["sites"])
    station_key = tuple(
        (
            site,
            _cache_value(station_context["bandwidth_hz"][site]),
            _cache_value(station_context["mount_type"][site]),
            _cache_value(station_context["feed_angle"][site]),
            _cache_value(station_context["polarization_basis"][site]),
        )
        for site in sites
    )

    return (
        _cache_value(getattr(obs, "source", None)),
        _cache_value(getattr(obs, "ra", None)),
        _cache_value(getattr(obs, "dec", None)),
        _cache_value(getattr(obs, "rf", None)),
        _cache_value(getattr(obs, "bw", None)),
        _cache_value(getattr(obs, "mjd", None)),
        _cache_value(obs.data["time"]),
        _cache_value(obs.data["t1"]),
        _cache_value(obs.data["t2"]),
        station_key,
    )


def station_metadata(obs, station_context, cache=None):
    key = station_metadata_cache_key(obs, station_context)
    if cache is not None and key in cache:
        return cache[key]

    t1 = obs.data["t1"]
    t2 = obs.data["t2"]
    sites_obs = np.unique(np.concatenate((t1, t2)))
    geometry = observation_geometry.ground_station_geometry(obs)
    if geometry is None:
        els = obs.unpack(["el1", "el2"], ang_unit="rad")
        pars = obs.unpack(["par_ang1", "par_ang2"], ang_unit="rad")
        el1 = els["el1"]
        el2 = els["el2"]
        par1 = pars["par_ang1"]
        par2 = pars["par_ang2"]
    else:
        el1 = geometry.elevation1_rad
        el2 = geometry.elevation2_rad
        par1 = geometry.parallactic_angle1_rad
        par2 = geometry.parallactic_angle2_rad
    times = obs.data["time"]
    tuniq = np.unique(times)

    bw1 = np.zeros_like(el1)
    bw2 = np.zeros_like(el2)
    f_el1 = np.zeros_like(el1)
    f_el2 = np.zeros_like(el2)
    f_par1 = np.zeros_like(el1)
    f_par2 = np.zeros_like(el2)
    phi_off1 = np.zeros_like(el1)
    phi_off2 = np.zeros_like(el2)
    site_masks = {}

    for site in sites_obs:
        ind1 = (t1 == site)
        ind2 = (t2 == site)
        site_masks[site] = (ind1, ind2)

        valhere = station_context["bandwidth_hz"][site]
        if valhere is None:
            valhere = obs.bw
        bw1[ind1] = valhere
        bw2[ind2] = valhere

        mttyp = station_context["mount_type"][site]
        fdang = station_context["feed_angle"][site]
        f_el1[ind1] = const.mount_type_dict[mttyp]["f_el"]
        f_el2[ind2] = const.mount_type_dict[mttyp]["f_el"]
        f_par1[ind1] = const.mount_type_dict[mttyp]["f_par"]
        f_par2[ind2] = const.mount_type_dict[mttyp]["f_par"]
        phi_off1[ind1] = fdang
        phi_off2[ind2] = fdang

    metadata = {
        "sites_obs": sites_obs,
        "site_masks": site_masks,
        "el1": el1,
        "el2": el2,
        "par1": par1,
        "par2": par2,
        "tuniq": tuniq,
        "bw1": bw1,
        "bw2": bw2,
        "f_el1": f_el1,
        "f_el2": f_el2,
        "f_par1": f_par1,
        "f_par2": f_par2,
        "phi_off1": phi_off1,
        "phi_off2": phi_off2,
    }

    if cache is not None:
        cache[key] = metadata

    return metadata


def station_terms(obs, F0, station_context, array, rng, gainamp=0.04, leakamp=0.1,
                  addgains=True, addleakage=False, flagwind=True, flagday=False,
                  flagsun=True, allow_mixed_basis=False, solar_angle=None,
                  verbosity=0, windspeed_sefd_modifier=None, cache=None):
    t1 = obs.data["t1"]
    t2 = obs.data["t2"]
    times = obs.data["time"]
    metadata = station_metadata(obs, station_context, cache=cache)
    sites_obs = metadata["sites_obs"]
    site_masks = metadata["site_masks"]
    el1 = metadata["el1"]
    el2 = metadata["el2"]
    par1 = metadata["par1"]
    par2 = metadata["par2"]
    tuniq = metadata["tuniq"]

    tau1 = np.zeros_like(el1)
    tau2 = np.zeros_like(el2)
    Tb1 = np.zeros_like(el1)
    Tb2 = np.zeros_like(el2)
    Tsys1 = np.zeros_like(el1)
    Tsys2 = np.zeros_like(el2)
    SEFD1 = np.zeros_like(el1)
    SEFD2 = np.zeros_like(el2)
    bw1 = metadata["bw1"]
    bw2 = metadata["bw2"]
    f_el1 = metadata["f_el1"]
    f_el2 = metadata["f_el2"]
    f_par1 = metadata["f_par1"]
    f_par2 = metadata["f_par2"]
    phi_off1 = metadata["phi_off1"]
    phi_off2 = metadata["phi_off2"]

    if addgains:
        gainamp1R = np.zeros_like(el1)
        gainamp2R = np.zeros_like(el2)
        gainphase1R = np.zeros_like(el1)
        gainphase2R = np.zeros_like(el2)
        gainamp1L = np.zeros_like(el1)
        gainamp2L = np.zeros_like(el2)
        gainphase1L = np.zeros_like(el1)
        gainphase2L = np.zeros_like(el2)

    if addleakage:
        leak1R = np.zeros_like(el1, dtype=complex)
        leak2R = np.zeros_like(el2, dtype=complex)
        leak1L = np.zeros_like(el1, dtype=complex)
        leak2L = np.zeros_like(el2, dtype=complex)

    flagsites = list()
    uptime_mask = np.ones(len(obs.data), dtype=bool)
    for site in sites_obs:
        tau_z, _ = _station_weather_values(
            station_context["tau"][site], len(obs.data), 'opacity'
        )
        Tatm, _ = _station_weather_values(
            station_context["Tatm"][site], len(obs.data), 'atmospheric temperature'
        )
        Tgnd, _ = _station_weather_values(
            station_context["Tgnd"][site], len(obs.data), 'ground temperature'
        )
        ws, windspeed_is_scalar = _station_weather_values(
            station_context["windspeed"][site], len(obs.data), 'wind speed'
        )
        Aeff = station_context["effective_area"][site]
        wind_loading = station_context["wind_loading"][site]

        if flagwind:
            site_rows = ((t1 == site) | (t2 == site))
            high_wind = site_rows & (ws > wind_loading["shutdown"])
            if np.any(high_wind) and windspeed_is_scalar:
                flagsites.append(site)
                if verbosity > 0:
                    print(site + " cannot observe because of high wind.")
            elif np.any(high_wind):
                uptime_mask[high_wind] = False
                if verbosity > 0:
                    print(site + " cannot observe at some timestamps because of high wind.")

        if flagday:
            if site != "space":
                lon = const.known_longitudes[site]
                lat = const.known_latitudes[site]
                elev = const.known_elevations[site]
                location = EarthLocation.from_geodetic(lon, lat, height=elev)

                jd = obs.mjd + 2400000.5 + (times/24.0)
                timehere = Time(jd, format="jd")
                altazframe = AltAz(obstime=timehere, location=location)
                sun_altaz = get_sun(timehere).transform_to(altazframe)

                ind_daytime = (((t1 == site) | (t2 == site)) & (sun_altaz.alt.value > 0.0))
                uptime_mask[ind_daytime] = False

        if flagsun:
            if solar_angle < station_context["solar_avoidance"][site]:
                flagsites.append(site)
                if verbosity > 0:
                    print(site + " cannot observe because the source is too close to the Sun.")

        if site in station_context["station_uptimes"]:
            ind_too_early = (((t1 == site) | (t2 == site)) & (times < station_context["station_uptimes"][site][0]))
            ind_too_late = (((t1 == site) | (t2 == site)) & (times > station_context["station_uptimes"][site][1]))
            uptime_mask[ind_too_early] = False
            uptime_mask[ind_too_late] = False

        ind1, ind2 = site_masks[site]

        if allow_mixed_basis:
            if station_context["polarization_basis"][site] == "linear":
                tform_mat1 = np.zeros((ind1.sum(), 2, 2), dtype=complex)
                tform_mat2 = np.zeros((ind2.sum(), 2, 2), dtype=complex)
                tform_mat1[:] = const.circ_to_lin
                tform_mat2[:] = np.conj(const.circ_to_lin).T

                coh_mat1 = np.zeros((ind1.sum(), 2, 2), dtype=complex)
                coh_mat1[:, 0, 0] = obs.data["rrvis"][ind1]
                coh_mat1[:, 0, 1] = obs.data["rlvis"][ind1]
                coh_mat1[:, 1, 0] = obs.data["lrvis"][ind1]
                coh_mat1[:, 1, 1] = obs.data["llvis"][ind1]
                coh_mat2 = np.zeros((ind2.sum(), 2, 2), dtype=complex)
                coh_mat2[:, 0, 0] = obs.data["rrvis"][ind2]
                coh_mat2[:, 0, 1] = obs.data["rlvis"][ind2]
                coh_mat2[:, 1, 0] = obs.data["lrvis"][ind2]
                coh_mat2[:, 1, 1] = obs.data["llvis"][ind2]

                coh_mat_tformed1 = np.matmul(tform_mat1, coh_mat1)
                coh_mat_tformed2 = np.matmul(coh_mat2, tform_mat2)

                obs.data["rrvis"][ind1] = coh_mat_tformed1[:, 0, 0]
                obs.data["rlvis"][ind1] = coh_mat_tformed1[:, 0, 1]
                obs.data["lrvis"][ind1] = coh_mat_tformed1[:, 1, 0]
                obs.data["llvis"][ind1] = coh_mat_tformed1[:, 1, 1]
                obs.data["rrvis"][ind2] = coh_mat_tformed2[:, 0, 0]
                obs.data["rlvis"][ind2] = coh_mat_tformed2[:, 0, 1]
                obs.data["lrvis"][ind2] = coh_mat_tformed2[:, 1, 0]
                obs.data["llvis"][ind2] = coh_mat_tformed2[:, 1, 1]

        if site != "space":
            tau1[ind1] = tau_z[ind1] / np.cos((np.pi/2.0) - el1[ind1])
            tau2[ind2] = tau_z[ind2] / np.cos((np.pi/2.0) - el2[ind2])
        else:
            tau1[ind1] = 0.0
            tau2[ind2] = 0.0

        Tsource = (F0*Aeff)/(2.0*const.k)

        if site != "space":
            Tb1[ind1] = ((const.T_CMB + Tsource)*np.exp(-tau1[ind1])) + (Tatm[ind1]*(1.0 - np.exp(-tau1[ind1])))
            Tb2[ind2] = ((const.T_CMB + Tsource)*np.exp(-tau2[ind2])) + (Tatm[ind2]*(1.0 - np.exp(-tau2[ind2])))
        else:
            Tb1[ind1] = const.T_CMB + Tsource
            Tb2[ind2] = const.T_CMB + Tsource

        T_R = station_context["receiver_temperature"][site]
        sideband_ratio = station_context["sideband_ratio"][site]

        Tsys1[ind1] = (T_R + (const.eta_ff*Tb1[ind1]) + ((1.0 - const.eta_ff)*Tgnd[ind1]))*(1.0 + sideband_ratio)
        Tsys2[ind2] = (T_R + (const.eta_ff*Tb2[ind2]) + ((1.0 - const.eta_ff)*Tgnd[ind2]))*(1.0 + sideband_ratio)

        SEFD1[ind1] = (2.0*const.k*Tsys1[ind1])/Aeff
        SEFD2[ind2] = (2.0*const.k*Tsys2[ind2])/Aeff

        if flagwind:
            SEFD1[ind1] *= windspeed_sefd_modifier(
                ws[ind1], wind_loading["v0"], wind_loading["w"]
            )
            SEFD2[ind2] *= windspeed_sefd_modifier(
                ws[ind2], wind_loading["v0"], wind_loading["w"]
            )

        sefdind = (array.tarr["site"] == site)
        sefdr_arr = np.copy(array.tarr["sefdr"])
        sefdl_arr = np.copy(array.tarr["sefdl"])
        sefdhere = np.mean(np.concatenate((SEFD1[ind1], SEFD2[ind2])))
        sefdr_arr[sefdind] = sefdhere
        sefdl_arr[sefdind] = sefdhere
        array.tarr["sefdr"] = sefdr_arr
        array.tarr["sefdl"] = sefdl_arr

        if addgains:
            for t in tuniq:
                ind1here = ((times == t) & (t1 == site))
                ind2here = ((times == t) & (t2 == site))
                gainamphere = 10.0**(gainamp*rng.normal(0.0, 1.0))
                gainphasehere = rng.uniform(-np.pi, np.pi)
                gainamp1R[ind1here] = gainamphere
                gainamp2R[ind2here] = gainamphere
                gainphase1R[ind1here] = gainphasehere
                gainphase2R[ind2here] = gainphasehere
                gainamp1L[ind1here] = gainamphere
                gainamp2L[ind2here] = gainamphere
                gainphase1L[ind1here] = gainphasehere
                gainphase2L[ind2here] = gainphasehere

        if addleakage:
            leakRhere = (leakamp*rng.normal(0.0, 1.0)) + ((1.0j)*leakamp*rng.normal(0.0, 1.0))
            leakLhere = (leakamp*rng.normal(0.0, 1.0)) + ((1.0j)*leakamp*rng.normal(0.0, 1.0))
            leak1R[ind1] = leakRhere
            leak2R[ind2] = leakRhere
            leak1L[ind1] = leakLhere
            leak2L[ind2] = leakLhere

            if site != "space":
                tarrind = (array.tarr["site"] == site)
                dr_arr = np.copy(array.tarr["dr"])
                dl_arr = np.copy(array.tarr["dl"])
                dr_arr[tarrind] = leakRhere
                dl_arr[tarrind] = leakLhere
                array.tarr["dr"] = dr_arr
                array.tarr["dl"] = dl_arr

    terms = {
        "t1": t1,
        "t2": t2,
        "el1": el1,
        "el2": el2,
        "par1": par1,
        "par2": par2,
        "times": times,
        "tau1": tau1,
        "tau2": tau2,
        "Tb1": Tb1,
        "Tb2": Tb2,
        "Tsys1": Tsys1,
        "Tsys2": Tsys2,
        "SEFD1": SEFD1,
        "SEFD2": SEFD2,
        "bw1": bw1,
        "bw2": bw2,
        "f_el1": f_el1,
        "f_el2": f_el2,
        "f_par1": f_par1,
        "f_par2": f_par2,
        "phi_off1": phi_off1,
        "phi_off2": phi_off2,
        "flagsites": flagsites,
        "uptime_mask": uptime_mask,
    }

    if addgains:
        terms.update({
            "gainamp1R": gainamp1R,
            "gainamp2R": gainamp2R,
            "gainphase1R": gainphase1R,
            "gainphase2R": gainphase2R,
            "gainamp1L": gainamp1L,
            "gainamp2L": gainamp2L,
            "gainphase1L": gainphase1L,
            "gainphase2L": gainphase2L,
        })

    if addleakage:
        terms.update({
            "leak1R": leak1R,
            "leak2R": leak2R,
            "leak1L": leak1L,
            "leak2L": leak2L,
        })

    return terms
