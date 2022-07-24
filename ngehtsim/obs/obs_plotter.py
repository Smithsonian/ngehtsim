###################################################
# imports

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ehtim as eh

import ngehtsim.metrics.compute_metrics as cm
import ngehtsim.const_def as const

###################################################
# plotting functions

def plot_uv(obs,filename='uvplot.png',umax=10):
    """
    Create and save a :math:`(u,v)`-coverage plot.
    
    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      filename (str): file name for output plot
      umax (float): maximum baseline length for plot axes, in :math:`\\rm{G}\\lambda`
    
    """

    # unpack data
    u = obs.data['u'] / (1.0e9)
    v = obs.data['v'] / (1.0e9)

    # make plot
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(u,v,'b.',markersize=2,alpha=0.2)
    ax.plot(-u,-v,'b.',markersize=2,alpha=0.2)
    ax.set_xlabel(r'$u$ (G$\lambda$)')
    ax.set_ylabel(r'$v$ (G$\lambda$)')
    ax.set_xlim(umax,-umax)
    ax.set_ylim(-umax,umax)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_amp(obs,filename='ampplot.png',xlim=(0,10),ylim=(0.01,3)):
    """
    Create and save a plot of visibility amplitude vs :math:`(u,v)`-distance.
    
    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      filename (str): file name for output plot
      xlim (tuple): x-axis range, in :math:`\\rm{G}\\lambda`
      ylim (tuple): y-axis range, in Jy

    """

    # unpack data
    u = obs.data['u'] / (1.0e9)
    v = obs.data['v'] / (1.0e9)
    uvdist = np.sqrt(u**2.0 + v**2.0)
    amp = np.abs(obs.data['vis'])

    # make plot
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(uvdist,amp,'b.',markersize=2,alpha=0.2)
    ax.semilogy()
    ax.set_xlabel(r'$(u,v)$-distance (G$\lambda$)')
    ax.set_ylabel(r'Visibility amplitude (Jy)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_phase(obs,filename='phaseplot.png',xlim=(0,10),ylim=(-180,180)):
    """
    Create and save a plot of visibility phase vs :math:`(u,v)`-distance.
    
    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      filename (str): file name for output plot
      xlim (tuple): x-axis range, in :math:`\\rm{G}\\lambda`
      ylim (tuple): y-axis range, in degrees

    """

    # unpack data
    u = obs.data['u'] / (1.0e9)
    v = obs.data['v'] / (1.0e9)
    uvdist = np.sqrt(u**2.0 + v**2.0)
    phase = np.angle(obs.data['vis'])*(180.0/np.pi)

    # make plot
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(uvdist,phase,'b.',markersize=2,alpha=0.2)
    ax.set_xlabel(r'$(u,v)$-distance (G$\lambda$)')
    ax.set_ylabel(r'Visibility phase (degrees)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_snr(obs,filename='snrplot.png',xlim=(0,10),ylim=(1e0,1e4)):
    """
    Create and save a plot of SNR vs :math:`(u,v)`-distance.
    
    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      filename (str): file name for output plot
      xlim (tuple): x-axis range, in :math:`\\rm{G}\\lambda`
      ylim (tuple): y-axis range

    """

    # unpack data
    u = obs.data['u'] / (1.0e9)
    v = obs.data['v'] / (1.0e9)
    uvdist = np.sqrt(u**2.0 + v**2.0)
    snr = np.abs(obs.data['vis']) / obs.data['sigma']

    # make plot
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(uvdist,snr,'b.',markersize=2,alpha=0.2)
    ax.semilogy()
    ax.set_xlabel(r'$(u,v)$-distance (G$\lambda$)')
    ax.set_ylabel(r'SNR')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_snapshot(obs,obsgen,metric,filename='metric.png',timetype='UTC',fov=100.0,fillpix=10,logmid=1.5,logwid=0.525,stokes='I',artype='mean',weighting='natural',robust=0.0,ylim=None):
    """
    Create and save a plot of the chosen metric on snapshots.
    
    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object
      obsgen (ngehtsim.obs.obs_generator): an input obs_generator object
      metric (str): selected metric to plot; can be 'ff', 'bff', lcg', 'ar'
      filename (str): file name for output plot
      timetype (str): eht-imaging recognized timetype; can be 'UTC' or 'GMST'
      fov (float): field of view for computing FF, in :math:`\\mu\\rm{as}`
      logmid (float): the logarithmic midpoint of the BFF SNR mapping function
      logwid (float): the logarithmic width of the BFF SNR mapping function
      stokes (str): Stokes parameter for which to compute the BFF metric; can be 'I', 'Q', 'U', 'V'
      artype (str): what measure of the beam shape to use for AR metric; can be 'mean', 'minor', 'major', 'PA', 'angle'
      weighting (str): :math:`(u,v)`-weighting scheme for AR metric; can be 'natural', 'uniform', 'Briggs', 'robust'
      robust (float): the robust parameter for Briggs weighting in the AR metric
      ylim (tuple): y-axis range

    """

    # sort out the observing cadence
    start_time = obsgen.settings['t_start']
    end_time = start_time + obsgen.settings['dt']
    snapshot_interval = obsgen.settings['t_rest']

    # ensure the observation has the correct timetype
    obs = obs.switch_timetype(timetype_out=timetype)

    # compute metric
    if (metric.lower() == 'ff'):
        longest_BL = const.D_Earth/(const.c/obs.rf)
        t, y = cm.calc_ff_continuous(obs,longest_BL,start_time=start_time,end_time=end_time,snapshot_interval=snapshot_interval,fov=fov,fillpix=fillpix)
    if (metric.lower() == 'bff'):
        longest_BL = const.D_Earth/(const.c/obs.rf)
        t, y = cm.calc_bff_continuous(obs,longest_BL,start_time=start_time,end_time=end_time,snapshot_interval=snapshot_interval,fov=fov,fillpix=fillpix,logmid=logmid,logwid=logwid,stokes=stokes)
    if (metric.lower() == 'lcg'):
        dummy_circ_res = ((const.c/obs.rf) / const.D_Earth) / eh.RADPERUAS
        t, y = cm.calc_lcg_continuous(obs,start_time=start_time,end_time=end_time,snapshot_interval=snapshot_interval,dummy_circ_res=dummy_circ_res)
    if (metric.lower() == 'pss'):
        t, y = cm.calc_pss_continuous(obs,start_time=start_time,end_time=end_time,snapshot_interval=snapshot_interval)
    if (metric.lower() == 'ar'):
        t, y = cm.calc_ar_continuous(obs,start_time=start_time,end_time=end_time,snapshot_interval=snapshot_interval,artype=artype,weighting=weighting,robust=robust)

    # initialize figure
    fig = plt.figure(figsize=(4,8))
    ax1 = fig.add_axes([0.1,0.6,0.8,0.35])
    ax2 = fig.add_axes([0.1,0.1,0.8,0.5])

    # plot metric
    ax1.plot(t,y,'b-')

    # plot stations
    t1 = obs.data['t1']
    t2 = obs.data['t2']
    stations = np.sort(np.unique(np.concatenate((t1,t2))))[::-1]
    time = obs.data['time']
    yticks2 = []
    for istat, station in enumerate(stations):
        
        # plot indicator line
        ax2.plot([start_time,end_time],[istat,istat],'k--',alpha=0.2,linewidth=0.5)

        # get the timestamps for this station
        index = ((t1 == station) | (t2 == station))
        timehere = np.sort(time[index])

        # plot the timestamps
        ax2.plot(timehere,[istat]*len(timehere),'k.',markersize=2,linewidth=0)

        yticks2.append(istat)

    # clean up plot
    ax1.set_xlim(start_time,end_time)
    ax2.set_xlim(start_time,end_time)
    if ylim is None:
        ax1.set_ylim(0.0,1.1*np.nanmax(y))
    else:
        ax1.set_ylim(ylim)
    ax2.set_ylim(np.min(yticks2)-1,np.max(yticks2)+1.5)

    ax1.tick_params(axis='x',which='both',direction='inout',bottom=True)
    ax2.tick_params(axis='x',which='both',direction='inout',bottom=True,top=True)

    ax1.set_ylabel(metric.upper()+' metric value')
    
    if (timetype == 'UTC'):
        ax2.set_xlabel('UT (hr)')
    if (timetype == 'GMST'):
        ax2.set_xlabel('GMST (hr)')
    
    ax1.set_xticklabels([])
    ax2.set_yticks(yticks2)
    ax2.set_yticklabels(stations,fontsize=10)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
