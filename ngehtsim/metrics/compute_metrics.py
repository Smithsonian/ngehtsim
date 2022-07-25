###################################################
# imports

import numpy as np
import ehtim as eh
import ngehtutil as ng

import ngehtsim.metrics.fill_fracs as ff
import ngehtsim.metrics.bfill_fracs as bff
import ngehtsim.metrics.lcg_metric as lcg
import ngehtsim.const_def as const

###################################################
# fullobs metric computations


def calc_ff(obs, longest_BL=None, fov=100.0, fillpix=10):
    """
    Calculate the :math:`(u,v)`-filling fraction (FF) metric.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      longest_BL (float): length of the bounding baseline, dimensionless (i.e., in :math:`\\lambda`)
      fov (float): field of view for computing FF, in :math:`\\mu\\rm{as}`
      fillpix (int): number of resolution elements across a convolving kernel in FF

    Returns:
      (float): the FF value for this observation

    """

    # deal with empty observation
    if (len(obs.data) == 0):
        return 0.0

    if (longest_BL is None):
        longest_BL = const.D_Earth/(const.c/obs.rf)

    fill = ff.obs_fill(obs, longest=longest_BL, fov=fov, N=fillpix)

    return fill


def calc_bff(obs, longest_BL=None, fov=100.0, fillpix=10, logmid=1.5, logwid=0.525, stokes='I'):
    """
    Calculate the "bolstered" :math:`(u,v)`-filling fraction (BFF) metric.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      longest_BL (float): length of the bounding baseline, dimensionless (i.e., in :math:`\\lambda`)
      fov (float): field of view for computing FF, in :math:`\\mu\\rm{as}`
      fillpix (int): number of resolution elements across a convolving kernel in FF
      logmid (float): the logarithmic midpoint of the BFF SNR mapping function
      logwid (float): the logarithmic width of the BFF SNR mapping function
      stokes (str): Stokes parameter for which to compute the BFF metric; can be 'I', 'Q', 'U', 'V'

    Returns:
      (float): the BFF value for this observation

    """

    # deal with empty observation
    if (len(obs.data) == 0):
        return 0.0

    if (longest_BL is None):
        longest_BL = const.D_Earth/(const.c/obs.rf)

    bff_out = bff.obs_fill(obs, fov=fov, longest=longest_BL, N=fillpix, logmid=logmid, logwid=logwid, stokes=stokes)

    return bff_out


def calc_lcg(obs, dummy_circ_res=None):
    """
    Calculate the largest circular gap (LCG) metric.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      dummy_circ_res (float): resolution of "dummy circle", in :math:`\\mu\\rm{as}`

    Returns:
      (float): the LCG value for this observation

    """

    # deal with empty observation
    if (len(obs.data) == 0):
        return 0.0

    if (dummy_circ_res is None):
        dummy_circ_res = ((const.c/obs.rf) / const.D_Earth) / eh.RADPERUAS

    lcg_out = lcg.LCG_metric(obs, method='analytic', tavg=None, scan_avg=False, dummy_circ=True,
                             dummy_circ_res=dummy_circ_res, plot_solution=False, niter=1000, specify_x0=None)

    return lcg_out


def calc_pss(obs):
    """
    Calculate the point source sensitivity (PSS) metric.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object

    Returns:
      (float): the PSS value for this observation

    """

    # deal with empty observation
    if (len(obs.data) == 0):
        return np.inf

    pss_out = 1.0/np.sqrt(np.sum(1.0/obs.data['sigma']**2.0))

    return pss_out


def beam_shape(obs, weighting='natural', robust=0.0):
    """
    Calculate the beam shape.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      weighting (str): :math:`(u,v)`-weighting scheme for AR metric; can be 'natural', 'uniform', 'Briggs', 'robust'
      robust (float): the robust parameter for Briggs weighting in the AR metric

    Returns:
      (float), (float), (float): the minor and major beam axes (in :math:`\\mu\\rm{as}`),
                                 and the PA measured from the major axis (in degrees East of North)

    """

    # deal with empty observation
    if (len(obs.data) == 0):
        return np.inf, np.inf, np.inf

    # (u,v) coordinates
    u = obs.data['u']
    v = obs.data['v']

    # ehtim conventions
    if (weighting == 'natural'):
        weights = 1.0/obs.data['sigma']**2.0
    elif (weighting == 'uniform'):
        weights = np.ones_like(u)
    elif ((weighting == 'Briggs') | (weighting == 'robust')):
        wtav = np.mean(1.0/obs.data['sigma']**2.0)
        S2 = ((5.0*(10.0**(-robust)))**2.0) / wtav
        weights = 1.0 / (S2 + 2.0*(obs.data['sigma']**2.0))

    # second moment matrix
    u2 = np.average(u**2.0, weights=weights)
    v2 = np.average(v**2.0, weights=weights)
    uv = np.average(u*v, weights=weights)

    # compute (u,v) eigenvalues
    minor_uv = 0.5*(u2 + v2 - np.sqrt((u2**2.0) - (2.0*u2*v2) + (4.0*(uv**2.0)) + (v2**2.0)))
    major_uv = 0.5*(u2 + v2 + np.sqrt((u2**2.0) - (2.0*u2*v2) + (4.0*(uv**2.0)) + (v2**2.0)))

    # compute (u,v) eigenvectors
    ecomp1 = (-(1.0/(2.0*uv))*(v2 - u2 - np.sqrt((u2**2.0) - (2.0*u2*v2) + (4.0*(uv**2.0)) + (v2**2.0))))
    vec_major_uv = np.array([ecomp1, 1.0])
    vec_major_uv /= np.sqrt((vec_major_uv[0]**2.0) + (vec_major_uv[1]**2.0))

    # compute (u,v) position angle
    theta_uv = np.arctan2(vec_major_uv[0], vec_major_uv[1])

    # convert to image-domain
    minor = (0.5/np.sqrt(major_uv)) / eh.RADPERUAS
    major = (0.5/np.sqrt(minor_uv)) / eh.RADPERUAS
    theta = (180.0/np.pi)*(theta_uv + (np.pi/2.0))

    return minor, major, theta


def calc_ar(obs, artype='mean', weighting='natural', robust=0.0):
    """
    Calculate the angular resolution (AR) metric.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      artype (str): what measure of the beam shape to use; can be 'mean', 'minor', 'major', 'PA', 'angle'
      weighting (str): :math:`(u,v)`-weighting scheme for AR metric; can be 'natural', 'uniform', 'Briggs', 'robust'
      robust (float): the robust parameter for Briggs weighting in the AR metric

    Returns:
      (float): the LCG value for this observation, in :math:`\\mu\\rm{as}`

    """

    # deal with empty observation
    if (len(obs.data) == 0):
        return np.inf

    # get the beam shape parameters
    ar_list = beam_shape(obs, weighting=weighting, robust=robust)

    # compute the desired measure of angular resolution
    if (artype == 'minor'):
        ar_out = ar_list[0]
    if (artype == 'major'):
        ar_out = ar_list[1]
    if ((artype == 'PA') | (artype == 'angle')):
        ar_out = ar_list[2]
    if (artype == 'mean'):
        ar_out = np.sqrt(ar_list[0]*ar_list[1])

    return ar_out

###################################################
# snapshot metric computations


def calc_ff_continuous(obs, longest_BL=None, fov=100.0, fillpix=10, start_time=0.0, end_time=24.0, snapshot_interval=600.0):
    """
    Calculate the :math:`(u,v)`-filling fraction (FF) metric on snapshots.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      longest_BL (float): length of the bounding baseline, dimensionless (i.e., in :math:`\\lambda`)
      fov (float): field of view for computing FF, in :math:`\\mu\\rm{as}`
      fillpix (int): number of resolution elements across a convolving kernel in FF
      start_time (float): starting time of first snapshot, in hours
      end_time (float): ending time of last snapshot, in hours
      snapshot_interval (float): length of a single snapshot, in seconds

    Returns:
      (numpy.ndarray), (numpy.ndarray): the segmentation times and FF values for each snapshot in this observation

    """

    if (longest_BL is None):
        longest_BL = const.D_Earth/(const.c/obs.rf)

    # observation info
    times_obs = obs.data['time']
    datatable = obs.data.copy()

    # make a small blank copy to use for snapshots
    obs_blank = obs.copy()
    obs_blank.data = None

    # compute fill fractions
    times_temp = np.arange(start_time, end_time, snapshot_interval/3600.0)
    times = np.concatenate((times_temp, [2.0*times_temp[-1]-times_temp[-2]]))
    fills = np.zeros(len(times)-1)
    for itime in range(len(times)-1):
        UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
        if (UT_mask.sum() > 0):
            obs_snapshot = obs_blank.copy()
            obs_snapshot.data = datatable[UT_mask]
            fill_snapshot = ff.obs_fill(obs_snapshot, fov=fov, longest=longest_BL, N=fillpix)
            fills[itime] = fill_snapshot

    t = 0.5*(times[1:] + times[0:-1])

    return t, fills


def calc_bff_continuous(obs, longest_BL=None, fov=100.0, fillpix=10, logmid=1.5, logwid=0.525, stokes='I',
                        start_time=0.0, end_time=24.0, snapshot_interval=600.0):
    """
    Calculate the "bolstered" :math:`(u,v)`-filling fraction (BFF) metric on snapshots.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      longest_BL (float): length of the bounding baseline, dimensionless (i.e., in :math:`\\lambda`)
      fov (float): field of view for computing FF, in :math:`\\mu\\rm{as}`
      fillpix (int): number of resolution elements across a convolving kernel in FF
      logmid (float): the logarithmic midpoint of the BFF SNR mapping function
      logwid (float): the logarithmic width of the BFF SNR mapping function
      stokes (str): Stokes parameter for which to compute the BFF metric; can be 'I', 'Q', 'U', 'V'
      start_time (float): starting time of first snapshot, in hours
      end_time (float): ending time of last snapshot, in hours
      snapshot_interval (float): length of a single snapshot, in seconds

    Returns:
      (numpy.ndarray), (numpy.ndarray): the segmentation times and BFF values for each snapshot in this observation

    """

    if (longest_BL is None):
        longest_BL = const.D_Earth/(const.c/obs.rf)

    # observation info
    times_obs = obs.data['time']
    datatable = obs.data.copy()

    # make a small blank copy to use for snapshots
    obs_blank = obs.copy()
    obs_blank.data = None

    # compute fill fractions
    times_temp = np.arange(start_time, end_time, snapshot_interval/3600.0)
    times = np.concatenate((times_temp, [2.0*times_temp[-1]-times_temp[-2]]))
    bffs = np.zeros(len(times)-1)
    for itime in range(len(times)-1):
        UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
        if (UT_mask.sum() > 0):
            obs_snapshot = obs_blank.copy()
            obs_snapshot.data = datatable[UT_mask]
            bff_snapshot = bff.obs_fill(obs_snapshot, fov=fov, longest=longest_BL, N=fillpix,
                                        logmid=logmid, logwid=logwid, stokes=stokes)
            bffs[itime] = bff_snapshot

    t = 0.5*(times[1:] + times[0:-1])

    return t, bffs


def calc_lcg_continuous(obs, dummy_circ_res=None, start_time=0.0, end_time=24.0, snapshot_interval=600.0):
    """
    Calculate the largest circular gap (LCG) metric on snapshots.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      dummy_circ_res (float): resolution of "dummy circle", in :math:`\\mu\\rm{as}`
      start_time (float): starting time of first snapshot, in hours
      end_time (float): ending time of last snapshot, in hours
      snapshot_interval (float): length of a single snapshot, in seconds

    Returns:
      (numpy.ndarray), (numpy.ndarray): the segmentation times and LCG values for each snapshot in this observation

    """

    if (dummy_circ_res is None):
        dummy_circ_res = ((const.c/obs.rf) / const.D_Earth) / eh.RADPERUAS

    # observation info
    times_obs = obs.data['time']
    datatable = obs.data.copy()

    # make a small blank copy to use for snapshots
    obs_blank = obs.copy()
    obs_blank.data = None

    # compute LCG in snapshots
    times_temp = np.arange(start_time, end_time, snapshot_interval/3600.0)
    times = np.concatenate((times_temp, [2.0*times_temp[-1]-times_temp[-2]]))
    lcgs = np.zeros(len(times)-1)
    for itime in range(len(times)-1):
        UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
        if (UT_mask.sum() > 0):
            obs_snapshot = obs_blank.copy()
            obs_snapshot.data = datatable[UT_mask]
            lcg_snapshot = lcg.LCG_metric(obs_snapshot, method='analytic', tavg=None, scan_avg=False,
                                          dummy_circ=True, dummy_circ_res=dummy_circ_res, plot_solution=False,
                                          niter=1000, specify_x0=None)
            lcgs[itime] = lcg_snapshot

    # remove zero-valued points
    t = 0.5*(times[1:] + times[0:-1])
    index = (lcgs != 0.0)
    t = t[index]
    lcgs = lcgs[index]

    return t, lcgs


def calc_pss_continuous(obs, start_time=0.0, end_time=24.0, snapshot_interval=600.0):
    """
    Calculate the point source sensitivity (PSS) metric on snapshots.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      start_time (float): starting time of first snapshot, in hours
      end_time (float): ending time of last snapshot, in hours
      snapshot_interval (float): length of a single snapshot, in seconds

    Returns:
      (numpy.ndarray), (numpy.ndarray): the segmentation times and PSS values for each snapshot in this observation

    """

    # observation info
    times_obs = obs.data['time']
    datatable = obs.data.copy()

    # make a small blank copy to use for snapshots
    obs_blank = obs.copy()
    obs_blank.data = None

    # compute PSS in snapshots
    times_temp = np.arange(start_time, end_time, snapshot_interval/3600.0)
    times = np.concatenate((times_temp, [2.0*times_temp[-1]-times_temp[-2]]))
    psss = np.zeros(len(times)-1)
    for itime in range(len(times)-1):
        UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
        if (UT_mask.sum() > 0):
            obs_snapshot = obs_blank.copy()
            obs_snapshot.data = datatable[UT_mask]
            pss_snapshot = 1.0/np.sqrt(np.sum(1.0/obs_snapshot.data['sigma']**2.0))
            psss[itime] = pss_snapshot

    # remove zero-valued points
    t = 0.5*(times[1:] + times[0:-1])
    index = (psss != 0.0)
    t = t[index]
    psss = psss[index]

    return t, psss


def calc_ar_continuous(obs, artype='mean', weighting='natural', robust=0.0, start_time=0.0,
                       end_time=24.0, snapshot_interval=600.0):
    """
    Calculate the angular resolution (AR) metric on snapshots.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      artype (str): what measure of the beam shape to use; can be 'mean', 'minor', 'major', 'PA', 'angle'
      weighting (str): :math:`(u,v)`-weighting scheme for AR metric; can be 'natural', 'uniform', 'Briggs', 'robust'
      robust (float): the robust parameter for Briggs weighting in the AR metric
      start_time (float): starting time of first snapshot, in hours
      end_time (float): ending time of last snapshot, in hours
      snapshot_interval (float): length of a single snapshot, in seconds

    Returns:
      (numpy.ndarray), (numpy.ndarray): the segmentation times and AR values for each snapshot in this observation

    """

    # observation info
    times_obs = obs.data['time']
    datatable = obs.data.copy()

    # make a small blank copy to use for snapshots
    obs_blank = obs.copy()
    obs_blank.data = None

    # compute angular resolution in snapshots
    times_temp = np.arange(start_time, end_time, snapshot_interval/3600.0)
    times = np.concatenate((times_temp, [2.0*times_temp[-1]-times_temp[-2]]))
    ars = np.zeros(len(times)-1)
    for itime in range(len(times)-1):
        UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
        if (UT_mask.sum() > 0):
            obs_snapshot = obs_blank.copy()
            obs_snapshot.data = datatable[UT_mask]

            # get the beam shape parameters
            ar_list_snapshot = beam_shape(obs_snapshot, weighting=weighting, robust=robust)

            # compute the desired measure of angular resolution
            if (artype == 'minor'):
                ar_snapshot = ar_list_snapshot[0]
            if (artype == 'major'):
                ar_snapshot = ar_list_snapshot[1]
            if ((artype == 'PA') | (artype == 'angle')):
                ar_snapshot = ar_list_snapshot[2]
            if (artype == 'mean'):
                ar_snapshot = np.sqrt(ar_list_snapshot[0]*ar_list_snapshot[1])

            ars[itime] = ar_snapshot

    # remove zero-valued points
    t = 0.5*(times[1:] + times[0:-1])
    index = (ars != 0.0)
    t = t[index]
    ars = ars[index]

    return t, ars

###################################################
# other metrics


def calc_cost(obs, observations_per_year=1, days_per_observation=1, hours_per_observation=24):
    """
    Calculate the cost of an array.

    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      observations_per_year (int): number of observating runs to be carried out per year
      days_per_observation (int): number of days per observing run
      hours_per_observation (float): number of hours in a single observation

    Returns:
      (float), (float): the total capital and annual operating costs of the array, in dollars

    """

    # deal with empty observation
    if (len(obs.data) == 0):
        return 0.0

    # set configuration
    config = ng.cost.CostConfig(observations_per_year=observations_per_year,
                                days_per_observation=days_per_observation,
                                hours_per_observation=hours_per_observation)

    # make list of stations
    stations = list()
    for t in obs.tarr['site']:
        stations.append(ng.Station.from_name(t))

    # populate array object
    arr = ng.Array('ngEHT', stations)

    costs = ng.cost.calculate_costs(config, arr.stations())

    return costs[0]['TOTAL CAPEX'], costs[0]['ANNUAL OPEX']
