###################################################
# imports

import numpy as np
import ehtim as eh
import ngehtsim.obs.obs_generator as og
import ngehtsim.const_def as const
from astropy.time import Time
import copy
try:
    from eat.io.hops import read_alist
except ImportError:
    print('Warning: eat not installed! Cannot use calibration functionality.')
import pandas as pd

###################################################
# define various dictionaries

# station code conversion dictionary
known_station_dict = {'A': 'ALMA',
                      'G': 'GLT',
                      'J': 'JCMT',
                      'K': 'KP',
                      'L': 'LMT',
                      'N': 'NOEMA',
                      'P': 'IRAM',
                      'S': 'SMA',
                      'X': 'APEX',
                      'Y': 'SPT',
                      'Z': 'SMT'}

###################################################
# function definitions

def argunique(array):
    """
    Return the indices associated with the unique elements of an array.
    Analogous in spirit to functions like argmin and argsort.

    Args:
    array (numpy.array): The array for which to return the unique-element arguments

    Returns:
    (numpy.array): An array of arguments for each unique element
    

    """
    arr_unique = np.unique(array)
    ind_unique = np.zeros(len(arr_unique),dtype=int)
    for i in range(len(arr_unique)):
        ind_unique[i] = np.min(np.nonzero(array == arr_unique[i]))
    return ind_unique


def apriorical(filename,sourcename,bandwidth,debias=True,remove_autocorr=True,
               SNR_cut=0.0,station_codes={},return_coords=False,**kwargs):
    """
    Read an alist data file and carry out a priori flux density calibration.

    Args:
      filename (str): The name of the input alist data file
      sourcename (str): The name of the source whose data to calibrate
      bandwidth (float): The bandwidth over which the data have been averaged, in GHz
      debias (bool): Flag for whether to debias the amplitudes stored in the alist file
      remove_autocorr (bool): Flag for whether to remove autocorrelations during calibration
      SNR_cut (float): SNR cut to apply
      station_codes (dict): Dictionary containing conversions between single- and multi-letter station codes
      return_coords (bool): Flag for whether to return RA and DEC of the source

    Returns:
      (pandas.DataFrame): pandas DataFrame containing the calibrated data and associated metainfo
    """

    ############################################
    # check inputs
    
    # update the station codes if needed
    station_dict = copy.deepcopy(known_station_dict)
    station_dict.update(station_codes)

    ############################################
    # parse the alist file

    # read in the alist file using eat
    df = read_alist(filename)

    # extract the relevant quantities
    datetime_orig = df.datetime
    timetag_orig = np.array(df.timetag)
    source_orig = np.array(df.source)
    bl_orig = np.array(df.baseline)
    pl_orig = np.array(df.polarization)
    tint_orig = np.array(df.duration, dtype=float)
    amp_orig = np.array(df.amp, dtype=float)
    snr_orig = np.array(df.snr, dtype=float)
    phase_orig = np.array(df.resid_phas, dtype=float)
    elev1_orig = np.array(df.ref_elev, dtype=float)
    elev2_orig = np.array(df.rem_elev, dtype=float)
    az1_orig = np.array(df.ref_az, dtype=float)
    az2_orig = np.array(df.rem_az, dtype=float)
    u_orig = np.array(df.u, dtype=float)
    v_orig = np.array(df.v, dtype=float)
    freq_orig = np.array(df.ref_freq, dtype=float)
    ra_orig = np.array(df.ra_hrs, dtype=float)
    dec_orig = np.array(df.dec_deg, dtype=float)

    # debias SNR if desired
    if debias:
        snr_orig = np.sqrt((snr_orig**2.0) - 1.0)

    # determine uncertainties
    amp_err_orig = amp_orig / snr_orig

    # determine time in hours
    t_orig = np.zeros(len(datetime_orig))
    day_orig = np.zeros(len(datetime_orig))
    for i, dt in enumerate(datetime_orig):
        t_orig[i] = dt.hour + (dt.minute/60.0) + (dt.second/3600.0)
        day_orig[i] = float(timetag_orig[i].split('-')[0])

    # convert (u,v) to lambda
    u_orig *= (1.0e6)
    v_orig *= (1.0e6)

    # convert frequency to GHz
    freq_orig /= (1.0e3)

    ############################################
    # extract only the data associated with the desired source

    ind = (source_orig == sourcename)
    datetime = datetime_orig[ind]
    datetime.index = np.arange(len(datetime))
    timetag = timetag_orig[ind]
    t_hr = t_orig[ind]
    t = t_hr + (day_orig[ind]*24.0)
    day = day_orig[ind]
    bl = bl_orig[ind]
    pl = pl_orig[ind]
    u = u_orig[ind]
    v = v_orig[ind]
    freq = freq_orig[ind]
    amp = amp_orig[ind]
    phase = phase_orig[ind] * (np.pi/180.0)
    snr = snr_orig[ind]
    elev1 = elev1_orig[ind] * (np.pi/180.0)
    elev2 = elev2_orig[ind] * (np.pi/180.0)
    az1 = az1_orig[ind] * (np.pi/180.0)
    az2 = az2_orig[ind] * (np.pi/180.0)
    ra = ra_orig[ind]
    dec = dec_orig[ind]
    amp_err = amp_err_orig[ind]
    tint = tint_orig[ind]

    # construct complex visibilities
    vis = amp*np.exp((1j)*phase)

    # time info
    days = np.empty(len(datetime),dtype=int)
    months = np.empty(len(datetime),dtype=int)
    years = np.empty(len(datetime),dtype=int)
    for i in range(len(datetime)):
        days[i] = datetime[i].day
        months[i] = datetime[i].month
        years[i] = datetime[i].year

    ############################################
    # remove autocorrelations

    if remove_autocorr:

        ind = np.ones(len(timetag),dtype=bool)
        for i in range(len(timetag)):
            if (bl[i][0] == bl[i][1]):
                ind[i] = False
        datetime = datetime[ind]
        datetime.index = np.arange(len(datetime))
        days = days[ind]
        months = months[ind]
        years = years[ind]
        timetag = timetag[ind]
        t = t[ind]
        t_hr = t_hr[ind]
        day = day[ind]
        bl = bl[ind]
        pl = pl[ind]
        u = u[ind]
        v = v[ind]
        freq = freq[ind]
        amp = amp[ind]
        phase = phase[ind]
        snr = snr[ind]
        elev1 = elev1[ind]
        elev2 = elev2[ind]
        az1 = az1[ind]
        az2 = az2[ind]
        ra = ra[ind]
        dec = dec[ind]
        amp_err = amp_err[ind]
        tint = tint[ind]
        vis = vis[ind]

    if SNR_cut > 0.0:

        ind = (snr >= SNR_cut)
        datetime = datetime[ind]
        datetime.index = np.arange(len(datetime))
        days = days[ind]
        months = months[ind]
        years = years[ind]
        timetag = timetag[ind]
        t = t[ind]
        t_hr = t_hr[ind]
        day = day[ind]
        bl = bl[ind]
        pl = pl[ind]
        u = u[ind]
        v = v[ind]
        freq = freq[ind]
        amp = amp[ind]
        phase = phase[ind]
        snr = snr[ind]
        elev1 = elev1[ind]
        elev2 = elev2[ind]
        az1 = az1[ind]
        az2 = az2[ind]
        ra = ra[ind]
        dec = dec[ind]
        amp_err = amp_err[ind]
        tint = tint[ind]
        vis = vis[ind]

    # flag if there's more than one RA/DEC
    if ((len(np.unique(ra)) > 1) | (len(np.unique(dec)) > 1)):
        raise Exception('More than one RA and/or DEC value for this source is listed!  Check the alist file.')
    RA = np.mean(ra)
    DEC = np.mean(dec)

    ############################################
    # simulate the observation with ngehtsim

    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    # make dummy input model
    mod = eh.model.Model()
    mod = mod.add_point(1.0)

    # loop through scans
    t_uniq = np.unique(t)
    t_uniq_arg = argunique(t)
    datetime_uniq = datetime[t_uniq_arg]
    datetime_uniq.index = np.arange(len(datetime_uniq))
    vis_corrected = np.zeros_like(vis)
    amp_corrected = np.zeros_like(amp)
    amp_err_corrected = np.zeros_like(amp_err)
    for itime in range(len(t_uniq)):

        there = t_uniq[itime]
        ind = (t == there)
        there -= (np.mean(day[ind])*24.0)

        # determine the sites that are observing
        bl_here = bl[ind]
        sites = list()
        for b in bl_here:
            if (b[0] not in station_dict.keys()):
                raise Exception('Unrecognized station with single-letter code '+b[0]+'; please specify this station using the station_codes keyword argument.')
            if (b[1] not in station_dict.keys()):
                raise Exception('Unrecognized station with single-letter code '+b[1]+'; please specify this station using the station_codes keyword argument.') 
            sites.append(station_dict[b[0]])
            sites.append(station_dict[b[1]])
        sites = list(np.unique(sites))

        # other relevant quantities
        freq_here = np.mean(freq[ind])
        pl_here = pl[ind]
        vis_here = vis[ind]
        amp_here = amp[ind]
        amp_err_here = amp_err[ind]
        tint_here = tint[ind]
        dt = np.max(tint_here)
        if len(np.unique(tint_here)) > 1:
            print('Warning: Some baselines in the scan at time '+str(there)+' have different integration times than others.  Assumption here is to use a single integration time corresponding to the maximum value in the scan, but some baselines may be poorly-calibrated as a result.')
        
        # settings
        settings = {'source': sourcename,
                    'RA': RA,
                    'DEC': DEC,
                    'frequency': freq_here,
                    'bandwidth': bandwidth,
                    'day': str(datetime_uniq[itime].day),
                    'month': months[datetime_uniq[itime].month-1],
                    'year': str(datetime_uniq[itime].year),
                    't_start': there,
                    'dt': (dt/3600.0),
                    't_int': dt,
                    't_rest': dt,
                    'fringe_finder': ['naive', 0.0],
                    'sites': sites,
                    'weather': 'exact'}

        # simulate this scan, tracking SEFDs
        obsgen = og.obs_generator(settings,weight=1,**kwargs)
        obs = obsgen.make_obs(mod,addnoise=False,addgains=False,flagwind=False,el_min=0.0,el_max=90.0)
        obs = obs.switch_polrep('circ')

        # flag any unwanted times
        UT_start_hour = there-0.0001
        UT_stop_hour = there+0.0001
        UT_mask = obs.unpack('time')['time'] <= UT_start_hour
        UT_mask = UT_mask + (obs.unpack('time')['time'] >= UT_stop_hour)
        what_mask = np.array([False for j in range(len(UT_mask))])
        mask = UT_mask | what_mask
        SEFD1 = obsgen.SEFD1[~mask]
        SEFD2 = obsgen.SEFD2[~mask]
        with eh.parloop.HiddenPrints():
            obs = obs.flag_UT_range(UT_start_hour, UT_stop_hour, output='flagged')

        # determine amplitude scaling factor
        sites_arr = np.array(sites)
        bw = bandwidth * (1.0e9)
        t1 = obs.data['t1']
        t2 = obs.data['t2']
        scale_factor = np.zeros(len(amp_here))
        for j in range(len(amp_here)):
            ind_here = ((t1 == station_dict[bl_here[j][0]]) & (t2 == station_dict[bl_here[j][1]]))
            ind_here |= ((t2 == station_dict[bl_here[j][0]]) & (t1 == station_dict[bl_here[j][1]]))
            scale_factor[j] = (1.0e-4)*np.sqrt(SEFD1[ind_here]*SEFD2[ind_here])

        # apply the scaling factor
        vis_corrected[ind] = vis_here * scale_factor
        amp_corrected[ind] = amp_here * scale_factor
        amp_err_corrected[ind] = amp_err_here * scale_factor

    ############################################
    # return the calibrated data

    struct = {'freq': freq,
              't': t,
              't_hr': t_hr,
              'bl': bl,
              'pl': pl,
              'u': u,
              'v': v,
              'elev1': elev1,
              'elev2': elev2,
              'az1': az1,
              'az2': az2,
              'datetime': datetime,
              'vis': vis_corrected,
              'err': amp_err_corrected,
              'snr': snr}

    df_out = pd.DataFrame(struct)

    if return_coords:
        return df_out, RA, DEC
    else:
        return df_out


def write_dlist(filename,sourcename,bandwidth,outname,debias=True,remove_autocorr=True,
                SNR_cut=0.0,station_codes={},**kwargs):
    """
    Write a "dlist" data file from an "alist" file.

    Args:
      filename (str): The name of the input alist data file
      sourcename (str): The name of the source whose data to calibrate
      bandwidth (float): The bandwidth over which the data have been averaged, in GHz
      outname (str): The name of the output dlist data file
      debias (bool): Flag for whether to debias the amplitudes stored in the alist file
      remove_autocorr (bool): Flag for whether to remove autocorrelations during calibration
      SNR_cut (float): SNR cut to apply
      station_codes (dict): Dictionary containing conversions between single- and multi-letter station codes

    Returns:
      Writes a dlist file to disk
    """

    ############################################
    # check inputs

    # update the station codes if needed
    station_dict = copy.deepcopy(known_station_dict)
    station_dict.update(station_codes)

    ############################################
    # carry out the flux density calibration
        
    df, RA, DEC = apriorical(filename,sourcename,bandwidth,debias=debias,remove_autocorr=remove_autocorr,station_codes=station_codes,return_coords=True,**kwargs)
    freq = df.freq
    t = df.t
    t_hr = df.t_hr
    bl = df.bl
    pl = df.pl
    u = df.u
    v = df.v
    elev1 = df.elev1
    elev2 = df.elev2
    az1 = df.az1
    az2 = df.az2
    datetime = df.datetime
    vis_corrected = df.vis
    err_corrected = df.err
    snr = df.snr

    ############################################
    # write reformatted "dlist" table

    # convert source RA and DEC to radians
    RA_rad = RA*15.0*(np.pi/180.0)
    DEC_rad = DEC*(np.pi/180.0)

    # write file
    with open(outname,'w') as f:

        header = ''
        header += 'Freq'.ljust(24)
        header += 'Time'.ljust(24)
        header += 'Station1'.ljust(12)
        header += 'Station2'.ljust(12)
        header += 'Pol1'.ljust(6)
        header += 'Pol2'.ljust(6)
        header += 'u'.ljust(24)
        header += 'v'.ljust(24)
        header += 'elev1'.ljust(24)
        header += 'elev2'.ljust(24)
        header += 'parang1'.ljust(24)
        header += 'parang2'.ljust(24)
        header += 'Re(vis)'.ljust(24)
        header += 'Im(vis)'.ljust(24)
        header += 'sigma' + '\n'
        f.write(header)
        f.write('-'*292 + '\n')

        for i in range(len(vis_corrected)):

            # populate table entries
            strhere = ''
            strhere += str(freq[i]).ljust(24)
            strhere += str(np.round(t[i],12)).ljust(24)
            strhere += station_dict[bl[i][0]].ljust(12)
            strhere += station_dict[bl[i][1]].ljust(12)
            strhere += pl[i][0].ljust(6)
            strhere += pl[i][1].ljust(6)
            strhere += str(np.round(u[i],10)).ljust(24)
            strhere += str(np.round(v[i],10)).ljust(24)
            strhere += str(np.round(elev1[i],10)).ljust(24)
            strhere += str(np.round(elev2[i],10)).ljust(24)

            # compute parallactic angle
            timestr = str(datetime[i].year) + '-' + str(datetime[i].month).zfill(2) + '-' + str(datetime[i].day).zfill(2) + ' ' + str(datetime[i].hour).zfill(2) + ':' + str(datetime[i].minute).zfill(2) + ':' + str(datetime[i].second).zfill(2)
            timeobj = Time(timestr,format='iso')
            MJD = np.floor(timeobj.mjd)
            gst = (np.pi/12.0)*eh.observing.obs_helpers.utc_to_gmst(t_hr[i],MJD)        
            lat1 = const.known_latitudes[station_dict[bl[i][0]]]*(np.pi/180.0)
            lat2 = const.known_latitudes[station_dict[bl[i][1]]]*(np.pi/180.0)
            lon1 = const.known_longitudes[station_dict[bl[i][0]]]*(np.pi/180.0)
            lon2 = const.known_longitudes[station_dict[bl[i][1]]]*(np.pi/180.0)
            hr_angle1 = eh.observing.obs_helpers.hr_angle(gst,lon1,RA_rad)
            hr_angle2 = eh.observing.obs_helpers.hr_angle(gst,lon2,RA_rad)
            par_angle1 = eh.observing.obs_helpers.par_angle(hr_angle1, lat1, DEC_rad)
            par_angle2 = eh.observing.obs_helpers.par_angle(hr_angle2, lat2, DEC_rad)

            # continue populating table entries
            strhere += str(np.round(par_angle1,10)).ljust(24)
            strhere += str(np.round(par_angle2,10)).ljust(24)
            strhere += str(np.round(np.real(vis_corrected[i]),12)).ljust(24)
            strhere += str(np.round(np.imag(vis_corrected[i]),12)).ljust(24)
            strhere += str(np.round(err_corrected[i],12))
            strhere += '\n'

            f.write(strhere)
