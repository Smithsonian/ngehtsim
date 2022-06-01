###################################################
# imports

import numpy as np
import ehtim as eh
import ngehtutil as ng
import scipy.stats as stats
from scipy.special import erf, erfinv
import yaml
import glob
import sys
from collections import defaultdict
from astropy.time import Time
import time

from . import const_def as const

###################################################
# class definition

class obs_generator(object):

    # initialize class instantiation
    def __init__(self,settings_file):

        # load settings file
        self.settings = {}
        self.settings_file = settings_file
        self.load_settings()
        print("========= Loaded settings from {0}".format(settings_file))

        # check for issues, fix some easy ones, complain about the others
        if self.settings['frequency'] not in ['86','230','345']:
            raise ValueError('Input frequency needs to be one of 86, 230, or 345.')
        if (self.settings['nbands'] < 1):
            self.settings['nbands'] = 1
            raise Warning('Input nbands must be at least 1; setting to 1.')
        if ((self.settings['nbands'] > 1) & (self.settings['rf_offset'] < self.settings['bandwidth'])):
            raise Exception('Input rf_offset must be greater than or equal to input bandwidth when nbands > 1.')
        if (self.settings['path_to_weather'][-1] != '/'):
            self.settings['path_to_weather'] += '/'

        if self.settings['random_seed'] is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(self.settings['random_seed'])

        # extract commonly-used settings
        self.sites = self.settings['sites']
        self.model_file = self.settings['model_file']
        self.freq = float(self.settings['frequency'])*(1.0e9)
        self.determine_mjd()
        self.make_array()
        self.tabulate_weather()

        # load input image
        self.load_image()

        # other
        self.python_version = sys.version_info.major

    # load and store settings
    def load_settings(self):
        loader = yaml.SafeLoader
        with open(self.settings_file, 'r') as fi:
            self.settings.update(yaml.load(fi, Loader=loader))

    # load and store image
    def load_image(self):
        im_tmp = eh.image.load_image(self.model_file)
        im_tmp.rf = np.float(np.round(im_tmp.rf))
        self.im = im_tmp

    # store ngeht-util and ehtim array objects
    def make_array(self):
        sitelist = self.sites
        stations = list()
        for site in sitelist:
            stationhere = ng.Station.from_name(site)
            if stationhere.name in ['BAJA','CNI','LAS']:
                stationhere.dishes = [ng.station.Dish(diameter=6.1)]
            elif stationhere.name in ['OVRO']:
                stationhere.dishes = [ng.station.Dish(diameter=10.4)]
            elif (stationhere.existing_dish == False):
                stationhere.dishes = [ng.station.Dish(diameter=self.settings['D_new'])]
            stations.append(stationhere)
        self.array = ng.Array('test_array',stations)
        self.arr = self.array.to_ehtim_array(self.freq/(1.0e9))

    # compute aperture efficiency
    def eta_dish(self,sigma,offset):
        # sigma : surface RMS, in meters
        # offset : focus offset, in meters
        eta_dish = np.exp(-((4*np.pi*np.sqrt((sigma)**2+(offset)**2))/(const.c/self.freq))**2)
        return eta_dish

    # store receiver temperature
    def set_TR(self):
        if (self.settings['frequency'] == '230'):
            self.T_R = const.T_R_230
        if (self.settings['frequency'] == '345'):
            self.T_R = const.T_R_345

    # generate dictionaries of telescope properties
    def initialize_dicts(self):

        # eta_new = self.eta_dish(const.sigma_surface,const.focus_offset)
        
        # # aperture efficiencies of existing telescopes
        # eta_existing_dict = {'ALMA': self.eta_dish(const.sigma_existing_dict['ALMA'],0.),
        #                      'APEX': self.eta_dish(const.sigma_existing_dict['APEX'],0.),
        #                      'GAM':  self.eta_dish(const.sigma_existing_dict['GAM'],const.focus_offset),
        #                      'GLT':  self.eta_dish(const.sigma_existing_dict['GLT'],0.),
        #                      'HAY':  self.eta_dish(const.sigma_existing_dict['HAY'],const.focus_offset)*0.5,
        #                      'JCMT': self.eta_dish(const.sigma_existing_dict['JCMT'],0.),
        #                      'KP':   self.eta_dish(const.sigma_existing_dict['KP'],0.),
        #                      'KVNYS':self.eta_dish(const.sigma_existing_dict['KVNYS'],0.),
        #                      'LAS':  self.eta_dish(const.sigma_existing_dict['LAS'],const.focus_offset),
        #                      'LMT':  self.eta_dish(const.sigma_existing_dict['LMT'],0.),
        #                      'NOB':  self.eta_dish(const.sigma_existing_dict['NOB'],const.focus_offset),
        #                      'NOEMA':self.eta_dish(const.sigma_existing_dict['NOEMA'],0.),
        #                      'OVRO': self.eta_dish(const.sigma_existing_dict['OVRO'],const.focus_offset),
        #                      'PV':   self.eta_dish(const.sigma_existing_dict['PV'],0.),
        #                      'SMA':  self.eta_dish(const.sigma_existing_dict['SMA'],0.),
        #                      'SMT':  self.eta_dish(const.sigma_existing_dict['SMT'],0.),
        #                      'SPT':  self.eta_dish(const.sigma_existing_dict['SPT'],0.),
        #                      'SUF':  self.eta_dish(const.sigma_existing_dict['SUF'],const.focus_offset)}

        # D_dict = {}
        # FWHM_beam = {}
        # eta_dict = {}
        # for site in sites:
        #     if site in const.D_existing_dict.keys():
        #         D_dict[site] = const.D_existing_dict[site]
        #     else:
        #         D_dict[site] = self.settings['D_new']
        #     FWHM_beam[site] = (const.c/self.freq)/D_dict[site]*(60.*60.*180./np.pi)    #arc-seconds

        #     if site in eta_existing_dict.keys():
        #         eta_dict[site] = eta_existing_dict[site]
        #     else:
        #         eta_dict[site] = eta_new

        # aperture efficiency of new dishes
        eta_new = self.eta_dish(const.sigma_surface,const.focus_offset)

        D_dict = {}
        FWHM_beam = {}
        eta_dict = {}
        for station in self.array.stations():
            D_dict[station.name] = station.diameter()
            FWHM_beam[station.name] = (const.c/self.freq)/D_dict[station.name]*(60.*60.*180./np.pi)    #arc-seconds
            eta_dict[station.name] = eta_new

        self.D_dict = D_dict
        self.FWHM_beam_dict = FWHM_beam
        self.eta_dict = eta_dict

    # segment the observation into timestamps
    def get_obs_times(self):
        t_first = self.settings['t_start']
        N_obs = int(np.ceil(self.settings['dt']/(self.settings['t_rest']/3600.)))
        t_last = t_first+float(N_obs-1)*(self.settings['t_rest']/3600.)
        self.t_seg_times = np.linspace(t_first,t_last,N_obs)
        print("========= Source: {0}".format(self.settings['source']))
        print("========= Number of timestamps: {0}".format(N_obs))
        print("========= Beginning of first integration: {0}".format(t_first))
        print("========= Beginning of last integration: {0}".format(t_last))

    # determine which sites will randomly fail technical readiness
    def get_unready_sites(self,sites_in_observ):
        p = self.settings['tech_readiness']
        index = np.random.choice([0, 1], size=(len(sites_in_observ)), p=[p,1-p]).astype(bool)
        sites_to_drop = sites_in_observ[index]
        return(sites_to_drop)

    # defines a specific MJD associated with an observing month
    def determine_mjd(self):
        if (self.settings['month'] == 'Jan'):
            t = Time(self.settings['year']+'-01-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Feb'):
            t = Time(self.settings['year']+'-02-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Mar'):
            t = Time(self.settings['year']+'-03-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Apr'):
            t = Time(self.settings['year']+'-04-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'May'):
            t = Time(self.settings['year']+'-05-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Jun'):
            t = Time(self.settings['year']+'-06-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Jul'):
            t = Time(self.settings['year']+'-07-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Aug'):
            t = Time(self.settings['year']+'-08-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Sep'):
            t = Time(self.settings['year']+'-09-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Oct'):
            t = Time(self.settings['year']+'-10-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Nov'):
            t = Time(self.settings['year']+'-11-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Dec'):
            t = Time(self.settings['year']+'-12-15T00:00:00',format='isot',scale='utc')
        else:
            print('This month abbreviation is not recognized; should be one of: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec')

        self.mjd = t.mjd
        
    # extract the opacity and Tb information from weather tables
    def tabulate_weather(self):

        # list of sites
        sites = self.arr.tarr['site']

        # pick a random past date from which to pull the weather
        self.randyear = np.random.randint(const.year_min,const.year_max+1)
        if (self.settings['month'] == 'Feb'):
            self.randday = np.random.randint(1,29)
        elif (self.settings['month'] in ['Apr','Jun','Sep','Nov']):
            self.randday = np.random.randint(1,31)
        else:
            self.randday = np.random.randint(1,32)

        # initialize dictionaries
        tau_dict = defaultdict(dict)
        Tatm_dict = defaultdict(dict)
        Tb_dict = defaultdict(dict)

        # read in the weather info and store it
        monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
        monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        monthnum = monthnums[monthnams == self.settings['month']][0]
        broken = False
        for isite, site in enumerate(sites):

            # determine which table to read
            pathhere = self.settings['path_to_weather']
            pathhere += site + '/'
            pathhere += monthnum + self.settings['month'] + '/'
            pathhere += 'mean_SEFD_info_' + self.settings['frequency'] + '.csv'
            
            # read in the table
            year, monthdum, day, tau, Tb = np.loadtxt(pathhere,skiprows=7,unpack=True,delimiter=',')

            # pull out the info for the selected random past date
            index = ((year == self.randyear) & (day == self.randday))
            if (index.sum() == 0):
                broken = True
                break
            tau_here = tau[index][0]
            Tb_here = Tb[index][0]

            # divide out the opacity term to get the actual atmospheric temperature
            Tatm = (Tb_here - (const.T_CMB_AM*np.exp(-tau_here))) / (1.0 - np.exp(-tau_here))

            # store the info in the dictionaries
            tau_dict[site] = tau_here
            Tatm_dict[site] = Tatm
            Tb_dict[site] = Tb_here

        # store the dictionaries, but only if everything executed successfully
        if broken:
            print('Something went wrong; retabulating weather...')
            self.tabulate_weather()
        else:
            self.tau_dict = tau_dict
            self.Tatm_dict = Tatm_dict
            self.Tb_dict = Tb_dict

    # generate an observation that folds in opacity effects
    def observe(self,im,addgains=True,gainamp=0.04,opacitycal=True,fft_pad_factor=2,apply_pointing_errors=False):

        # generate empty obsdata object
        obs_temp = self.arr.obsdata(im.ra,
                                    im.dec,
                                    self.freq,
                                    (1.0e9)*float(self.settings['bandwidth']),
                                    self.settings['t_int'],
                                    self.settings['t_rest'],
                                    self.settings['t_start'],
                                    self.settings['t_start'] + self.settings['dt'],
                                    mjd = self.mjd,
                                    polrep = im.polrep,
                                    tau = 0.0,
                                    timetype = 'UTC',
                                    elevmin = const.el_min,
                                    elevmax = const.el_max,
                                    fix_theta_GMST = False)

        # ensure that some relevant image properties are properly set
        im.rf = self.freq
        im.mjd = self.mjd
        im.source = self.settings['source']

        # observe the source
        obs = im.observe_same_nonoise(obs_temp,ttype=self.settings['ttype'],fft_pad_factor=fft_pad_factor)
        
        # make sure we're in a circular basis
        obs = obs.switch_polrep(polrep_out='circ')

        # extract elevation information
        t1 = obs.data['t1']
        t2 = obs.data['t2']
        sites_obs = np.unique(np.concatenate((t1,t2)))
        els = obs.unpack(['el1','el2'],ang_unit='rad')
        el1 = els['el1']
        el2 = els['el2']
        times = obs.data['time']
        tuniq = np.unique(times)

        # initialize various arrays
        tau1 = np.zeros_like(el1)
        tau2 = np.zeros_like(el2)
        Tb1 = np.zeros_like(el1)
        Tb2 = np.zeros_like(el2)
        Tsys1 = np.zeros_like(el1)
        Tsys2 = np.zeros_like(el2)
        SEFD1 = np.zeros_like(el1)
        SEFD2 = np.zeros_like(el2)

        if addgains:
            gainamp1R = np.zeros_like(el1)
            gainamp2R = np.zeros_like(el2)
            gainphase1R = np.zeros_like(el1)
            gainphase2R = np.zeros_like(el2)
            gainamp1L = np.zeros_like(el1)
            gainamp2L = np.zeros_like(el2)
            gainphase1L = np.zeros_like(el1)
            gainphase2L = np.zeros_like(el2)
        
        # loop through the sites in the array
        for isite, site in enumerate(sites_obs):

            # zenith opacity and atmospheric temperature
            tau_z = self.tau_dict[site]
            Tatm = self.Tatm_dict[site]

            # indices for this site
            ind1 = (t1 == site)
            ind2 = (t2 == site)

            # get opacities at each timestamp
            tau1[ind1] = tau_z / np.cos((np.pi/2.0) - el1[ind1])
            tau2[ind2] = tau_z / np.cos((np.pi/2.0) - el2[ind2])

            # get Tb contributions at each timestamp
            Tb1[ind1] = (const.T_CMB*np.exp(-tau1[ind1])) + (Tatm*(1.0 - np.exp(-tau1[ind1])))
            Tb2[ind2] = (const.T_CMB*np.exp(-tau2[ind2])) + (Tatm*(1.0 - np.exp(-tau2[ind2])))

            # determine system temperatures
            Tsys1[ind1] = self.T_R + Tb1[ind1]
            Tsys2[ind2] = self.T_R + Tb2[ind2]

            # determine SEFDs
            SEFD1[ind1] = (2.0*const.k*Tsys1[ind1])/((np.pi/4.0)*self.eta_dict[site]*(self.D_dict[site])**2)
            SEFD2[ind2] = (2.0*const.k*Tsys2[ind2])/((np.pi/4.0)*self.eta_dict[site]*(self.D_dict[site])**2)

            # generate gains
            if addgains:
                for t in tuniq:
                    ind1here = ((times == t) & (t1 == site))
                    ind2here = ((times == t) & (t2 == site))
                    gainamphere = 10.0**(gainamp*np.random.normal(0.0,1.0))
                    gainphasehere = np.random.uniform(-np.pi,np.pi)
                    gainamp1R[ind1here] = gainamphere
                    gainamp2R[ind2here] = gainamphere
                    gainphase1R[ind1here] = gainphasehere
                    gainphase2R[ind2here] = gainphasehere
                    gainamp1L[ind1here] = gainamphere
                    gainamp2L[ind2here] = gainamphere
                    gainphase1L[ind1here] = gainphasehere
                    gainphase2L[ind2here] = gainphasehere

        # store opacities as part of the observation
        obs.data['tau1'] = tau1
        obs.data['tau2'] = tau2

        # store and apply gains
        if addgains:
            g1R = gainamp1R*np.exp((1.0j)*gainphase1R)
            g2R = gainamp2R*np.exp((1.0j)*gainphase2R)
            g1L = gainamp1R*np.exp((1.0j)*gainphase1L)
            g2L = gainamp2R*np.exp((1.0j)*gainphase2L)
            self.station_gains1R = g1R
            self.station_gains2R = g2R
            self.station_gains1L = g1L
            self.station_gains2L = g2L
            obs.data['rrvis'] *= g1R*np.conj(g2R)
            obs.data['llvis'] *= g1L*np.conj(g2L)
            obs.data['rlvis'] *= g1R*np.conj(g2L)
            obs.data['lrvis'] *= g1L*np.conj(g2R)
            obs.data['rrsigma'] *= np.abs(g1R*g2R)
            obs.data['llsigma'] *= np.abs(g1L*g2L)
            obs.data['rlsigma'] *= np.abs(g1R*g2L)
            obs.data['lrsigma'] *= np.abs(g1L*g2R)

        # store things differently depending on whether opacity is assumed to be calibrated or not
        if opacitycal:

            # determine baseline thermal noise levels
            tint = obs.data['tint']
            sigma = np.sqrt((SEFD1*SEFD2*np.exp(tau1)*np.exp(tau2))/(2.0*obs.bw*tint)) / const.quant_eff
            obs.data['rrsigma'] = sigma
            obs.data['llsigma'] = sigma
            obs.data['rlsigma'] = sigma
            obs.data['lrsigma'] = sigma

        else:

            # apply opacity attenuation
            obs.data['rrvis'] *= np.exp(-tau1)*np.exp(-tau2)
            obs.data['llvis'] *= np.exp(-tau1)*np.exp(-tau2)
            obs.data['rlvis'] *= np.exp(-tau1)*np.exp(-tau2)
            obs.data['lrvis'] *= np.exp(-tau1)*np.exp(-tau2)

            # determine baseline thermal noise levels
            tint = obs.data['tint']
            sigma = np.sqrt((SEFD1*SEFD2)/(2.0*obs.bw*tint)) / const.quant_eff
            obs.data['rrsigma'] = sigma
            obs.data['llsigma'] = sigma
            obs.data['rlsigma'] = sigma
            obs.data['lrsigma'] = sigma

        # add thermal noise to observations
        obs.data['rrvis'] += sigma*np.random.normal(0.0,1.0,len(obs.data['rrsigma']))
        obs.data['llvis'] += sigma*np.random.normal(0.0,1.0,len(obs.data['llsigma']))
        obs.data['rlvis'] += sigma*np.random.normal(0.0,1.0,len(obs.data['rlsigma']))
        obs.data['lrvis'] += sigma*np.random.normal(0.0,1.0,len(obs.data['lrsigma']))

        # apply pointing errors, if desired
        if apply_pointing_errors:

            t1 = obs.data['t1']
            t2 = obs.data['t2']
            gain1 = np.ones(t1.shape)
            gain2 = np.ones(t2.shape)

            for si in obs.tarr['site']:
                if si in const.D_existing_dict.keys():
                    gain1[t1 == si] *= np.exp(-8*np.log(2)**2*((np.random.normal(scale=self.FWHM_beam_dict[si]/const.existing_pt_accuracy_factor,size=gain1[t1 == si].shape)/self.FWHM_beam_dict[si])**2))
                    gain2[t2 == si] *= np.exp(-8*np.log(2)**2*((np.random.normal(scale=self.FWHM_beam_dict[si]/const.existing_pt_accuracy_factor,size=gain2[t2 == si].shape)/self.FWHM_beam_dict[si])**2))
                else:
                    gain1[t1 == si] *= np.exp(-8*np.log(2)**2*((np.random.normal(scale=self.settings['RMS_point_err'],size=gain1[t1 == si].shape)/self.FWHM_beam_dict[si])**2))
                    gain2[t2 == si] *= np.exp(-8*np.log(2)**2*((np.random.normal(scale=self.settings['RMS_point_err'],size=gain2[t2 == si].shape)/self.FWHM_beam_dict[si])**2))

            # reduce visibility amplitudes by pointing-offset-induced gain correction
            obs.data['rrvis'] *= gain1*gain2
            obs.data['llvis'] *= gain1*gain2
            obs.data['rlvis'] *= gain1*gain2
            obs.data['lrvis'] *= gain1*gain2

        # restore Stokes polrep
        obs = obs.switch_polrep(polrep_out='stokes')

        return obs

    # generate observation
    def make_obs(self,addgains=True,gainamp=0.04,opacitycal=True,fft_pad_factor=2,verbose=False,apply_pointing_errors=False):

        # determine SNR thresholding scheme and values
        snr_algo, snr_args = self.settings['SNR_cutoff']

        # initialization
        im = self.im
        im.rf = self.freq
        obs = list()
        self.initialize_dicts()
        self.set_TR()
        if verbose:
            print("************** T_R set to : {0}".format(self.T_R))
        self.get_obs_times()
        # print('************** Scan start times: {0}'.format(self.t_seg_times))
        
        # loop through the bands
        nbands = self.settings['nbands']
        freq_offsets = (np.arange(float(nbands)) - np.mean(np.arange(float(nbands)))) * float(self.settings['rf_offset']) * (1.0e9)
        for i_band in range(nbands):

            # make a copy of the original image
            im_tmp = im.copy()

            # adjust the observing frequency to account for the band offset from central frequency
            im_tmp.rf = self.freq + freq_offsets[i_band]

            # generate observation
            obs_seg = self.observe(im_tmp,addgains=addgains,gainamp=gainamp,opacitycal=opacitycal,fft_pad_factor=fft_pad_factor,apply_pointing_errors=apply_pointing_errors)

            # apply naive SNR thresholding
            if (snr_algo == 'naive'):
                obs_seg = obs_seg.flag_low_snr(snr_cut=snr_args,output='kept')
            
            # apply an ad hoc phasing proxy for SNR thresholding
            elif (snr_algo == 'adhoc'):

                # parse SNR_cutoff arguments
                snr_ref = snr_args[0]
                tint_ref = snr_args[1]
                snr_noref = snr_args[2]
                snr_backup = snr_args[3]

                # check if the reference station is in the array
                ref = self.settings['ref_station']
                stations_here = np.unique(np.concatenate((obs_seg.data['t1'],obs_seg.data['t2'])))
                
                # if the reference station is not in the array, resort to the backup SNR threshold
                if ref not in stations_here:
                    obs_seg = obs_seg.flag_low_snr(snr_cut=snr_backup,output='kept')

                # if the reference station is in the array, assume ad hoc phasing will be used
                else:

                    # scale the snr to use the reference station integration time
                    snr_seg = np.abs(obs_seg.data['vis'])/obs_seg.data['sigma']
                    scale_factor = np.sqrt(tint_ref / obs_seg.data['tint'])
                    snr_precheck = snr_seg*scale_factor

                    # get the timestamps
                    time = obs_seg.data['time']
                    timestamps = np.unique(obs_seg.data['time'])

                    # get the stations
                    ant1 = obs_seg.data['t1']
                    ant2 = obs_seg.data['t2']

                    # create a running index list of sites to flag
                    master_index =  np.zeros(len(obs_seg.data),dtype='bool')

                    # check all timestamps
                    for itime, timestamp in enumerate(timestamps):
                        
                        ind_t = (time == timestamp)
                        stations_timestamp = np.unique(np.concatenate((ant1[ind_t],ant2[ind_t])))

                        # if the reference station is not in the segment, resort to the backup SNR threshold
                        if ref not in stations_timestamp:
                            ind_t &= (snr_precheck >= snr_backup)
                            master_index += ind_t

                        # if the reference station is in the segment, use ad hoc phasing
                        else:

                            # loop through stations
                            for istat, station_here in enumerate(stations_timestamp):

                                # check baselines between this station and ref
                                index_precheck = ((ant1 == station_here) & (ant2 == ref)) | ((ant2 == station_here) & (ant1 == ref))
                                index_precheck &= ind_t
                                if index_precheck.sum() > 0:

                                    # if the baseline to ref has too low SNR on the ref integration time, flag it
                                    if snr_precheck[index_precheck] < snr_ref:
                                        if verbose:
                                            print('SNR on '+ref+'-'+station_here+' baseline is '+str(snr_precheck[index_precheck][0])+', which is less than the specified threshold of '+str(snr_ref)+'; flagging this baseline.')

                                    # otherwise, check if the non-ref baselines have sufficient SNR
                                    else:

                                        if verbose:
                                            index_seg = ((ant1 == station_here) & (ant2 != ref)) | ((ant2 == station_here) & (ant1 != ref))
                                            index_seg &= (snr_precheck < snr_noref)
                                            index_seg &= ind_t

                                            # announce which baselines fail the SNR criterion
                                            if index_seg.sum() > 0:
                                                ant1list_here = ant1[index_seg]
                                                ant2list_here = ant2[index_seg]
                                                snr_seg_here = snr_precheck[index_seg]
                                                for iant in range(len(ant1list_here)):
                                                    print('SNR on '+ant1list_here[iant]+'-'+ant2list_here[iant]+' baseline is '+str(snr_seg_here[iant])+', which is less than the specified threshold of '+str(snr_noref)+'; flagging this baseline.')
                                            
                                        # retain the baselines that satisfy the SNR criterion
                                        index_seg = ((ant1 == station_here) & (ant2 != ref)) | ((ant2 == station_here) & (ant1 != ref))
                                        index_seg &= (snr_precheck >= snr_noref)
                                        index_seg &= ind_t
                                        master_index += index_seg

                                # if this station has no baselines to ref, then flag per the backup SNR threshold
                                else:

                                    if verbose:
                                        index_seg = ((ant1 == station_here) & (ant2 != ref)) | ((ant2 == station_here) & (ant1 != ref))
                                        index_seg &= (snr_precheck < snr_backup)
                                        index_seg &= ind_t

                                        # announce which baselines fail the SNR criterion
                                        if index_seg.sum() > 0:
                                            ant1list_here = ant1[index_seg]
                                            ant2list_here = ant2[index_seg]
                                            snr_seg_here = snr_precheck[index_seg]
                                            for iant in range(len(ant1list_here)):
                                                print('SNR on '+ant1list_here[iant]+'-'+ant2list_here[iant]+' baseline is '+str(snr_seg_here[iant])+', which is less than the specified threshold of '+str(snr_backup)+'; flagging this baseline.')

                                    # retain the baselines that satisfy the SNR criterion
                                    index_seg = ((ant1 == station_here) & (ant2 != ref)) | ((ant2 == station_here) & (ant1 != ref))
                                    index_seg &= (snr_precheck >= snr_backup)
                                    index_seg &= ind_t
                                    master_index += index_seg

                    # apply the flagging
                    data_copy = obs_seg.data.copy()
                    obs_seg.data = data_copy[master_index]

            # unrecognized SNR thresholding scheme
            else:
                raise ValueError('unknown algorithm for SNR_cutoff')

            # append this segment to the obs list
            if obs == []:
                obs = obs_seg.copy()
            else:
                obs.data = np.concatenate([obs.data,obs_seg.data])

        # Retrieve technincal readiness
        sites_in_obs = obs.tarr['site']
        sites_to_drop = self.get_unready_sites(sites_in_observ=sites_in_obs)
        if len(sites_to_drop) > 0:
            print("Dropping {0} due to technical (un)readiness.".format(sites_to_drop))
            obs = obs.flag_sites(sites_to_drop)

        return obs
