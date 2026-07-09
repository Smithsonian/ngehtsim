###################################################
# imports

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun

import ngehtsim.const_def as const

###################################################
# helpers


def station_terms(obs, F0, station_context, array, rng, gainamp=0.04, leakamp=0.1,
                  addgains=True, addleakage=False, flagwind=True, flagday=False,
                  flagsun=True, allow_mixed_basis=False, solar_angle=None,
                  verbosity=0, windspeed_sefd_modifier=None):
    t1 = obs.data["t1"]
    t2 = obs.data["t2"]
    sites_obs = np.unique(np.concatenate((t1, t2)))
    els = obs.unpack(["el1", "el2"], ang_unit="rad")
    pars = obs.unpack(["par_ang1", "par_ang2"], ang_unit="rad")
    el1 = els["el1"]
    el2 = els["el2"]
    par1 = pars["par_ang1"]
    par2 = pars["par_ang2"]
    times = obs.data["time"]
    tuniq = np.unique(times)

    tau1 = np.zeros_like(el1)
    tau2 = np.zeros_like(el2)
    Tb1 = np.zeros_like(el1)
    Tb2 = np.zeros_like(el2)
    Tsys1 = np.zeros_like(el1)
    Tsys2 = np.zeros_like(el2)
    SEFD1 = np.zeros_like(el1)
    SEFD2 = np.zeros_like(el2)
    bw1 = np.zeros_like(el1)
    bw2 = np.zeros_like(el2)
    f_el1 = np.zeros_like(el1)
    f_el2 = np.zeros_like(el2)
    f_par1 = np.zeros_like(el1)
    f_par2 = np.zeros_like(el2)
    phi_off1 = np.zeros_like(el1)
    phi_off2 = np.zeros_like(el2)

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
        tau_z = station_context["tau"][site]
        Tatm = station_context["Tatm"][site]
        Tgnd = station_context["Tgnd"][site]
        ws = station_context["windspeed"][site]
        Aeff = station_context["effective_area"][site]
        wind_loading = station_context["wind_loading"][site]

        if (ws > wind_loading["shutdown"]):
            if flagwind:
                flagsites.append(site)
                if verbosity > 0:
                    print(site + " cannot observe because of high wind.")

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

        ind1 = (t1 == site)
        ind2 = (t2 == site)

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
            tau1[ind1] = tau_z / np.cos((np.pi/2.0) - el1[ind1])
            tau2[ind2] = tau_z / np.cos((np.pi/2.0) - el2[ind2])
        else:
            tau1[ind1] = 0.0
            tau2[ind2] = 0.0

        Tsource = (F0*Aeff)/(2.0*const.k)

        if site != "space":
            Tb1[ind1] = ((const.T_CMB + Tsource)*np.exp(-tau1[ind1])) + (Tatm*(1.0 - np.exp(-tau1[ind1])))
            Tb2[ind2] = ((const.T_CMB + Tsource)*np.exp(-tau2[ind2])) + (Tatm*(1.0 - np.exp(-tau2[ind2])))
        else:
            Tb1[ind1] = const.T_CMB + Tsource
            Tb2[ind2] = const.T_CMB + Tsource

        T_R = station_context["receiver_temperature"][site]
        sideband_ratio = station_context["sideband_ratio"][site]

        Tsys1[ind1] = (T_R + (const.eta_ff*Tb1[ind1]) + ((1.0 - const.eta_ff)*Tgnd))*(1.0 + sideband_ratio)
        Tsys2[ind2] = (T_R + (const.eta_ff*Tb2[ind2]) + ((1.0 - const.eta_ff)*Tgnd))*(1.0 + sideband_ratio)

        SEFD1[ind1] = (2.0*const.k*Tsys1[ind1])/Aeff
        SEFD2[ind2] = (2.0*const.k*Tsys2[ind2])/Aeff

        if flagwind:
            SEFD_factor = windspeed_sefd_modifier(ws, wind_loading["v0"], wind_loading["w"])
            SEFD1[ind1] *= SEFD_factor
            SEFD2[ind2] *= SEFD_factor

        sefdind = (array.tarr["site"] == site)
        sefdr_arr = np.copy(array.tarr["sefdr"])
        sefdl_arr = np.copy(array.tarr["sefdl"])
        sefdhere = np.mean(np.concatenate((SEFD1[ind1], SEFD2[ind2])))
        sefdr_arr[sefdind] = sefdhere
        sefdl_arr[sefdind] = sefdhere
        array.tarr["sefdr"] = sefdr_arr
        array.tarr["sefdl"] = sefdl_arr

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
