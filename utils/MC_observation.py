###################################################
# imports

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ehtim as eh
import scipy.stats as stats
from scipy.special import erf, erfinv
import yaml
import glob
import sys
import itertools
import traceback
from collections import defaultdict
from .metrics import fill_fracs_v4 as ff
from .metrics import fill_fracs_v5 as bff
from .metrics import lcg_metric as lcg
from astropy.time import Time

###################################################
# constants

k = 1381.0              # Boltzmann constant, in Jy m^2 / K
c = 3.0e8               # speed of light, in m/s
D_Earth = 12742000.0    # Earth diameter, in m

###################################################
# class definition

class MC_observation(object):

    # initialize class instantiation
    def __init__(self,settings_file, updates=None):

        # load settings file
        self.settings = {}
        self.settings_file = settings_file
        self.load_settings(updates=updates)
        print("========= Loaded settings from {0}".format(settings_file))

        # extract commonly-used settings
        self.new_sites = self.settings['new_sites']
        self.base_sites = self.settings['base_sites']
        self.model_files = self.settings['model_files']
        self.N_MC = self.settings['N_MC']
        self.site_depth = self.settings['site_depth']
        self.determine_mjd()

        # load input image
        self.load_image()
        self.determine_freqs()
        print("========= Frequencies in images: {0}".format(self.freqs))

        # other
        self.python_version = sys.version_info.major

    # load and store settings
    def load_settings(self, updates={}):
        loader = yaml.SafeLoader
        with open(self.settings_file, 'r') as fi:
            self.settings.update(yaml.load(fi, Loader=loader))
        if updates:
            self.settings.update(updates)

    # load and store image
    def load_image(self):
        self.ims = list()
        for model_file in self.model_files:
            im_tmp = eh.image.load_image(model_file)
            print('========= Source is '+im_tmp.source)
            im_tmp.rf = np.float(np.round(im_tmp.rf))

            # set M87 total flux
            if im_tmp.source == 'M87':
                if (im_tmp.rf/1e9) == 230.:
                    im_tmp.imvec *= self.settings['total_flux_230_M87']/im_tmp.total_flux()
                elif (im_tmp.rf/1e9) == 345.:
                    im_tmp.imvec *= self.settings['total_flux_345_M87']/im_tmp.total_flux()

            # set Sgr A* total flux
            elif im_tmp.source == 'SgrA':
                if (im_tmp.rf/1e9) == 230.:
                    im_tmp.imvec *= self.settings['total_flux_230_SgrA']/im_tmp.total_flux()
                elif (im_tmp.rf/1e9) == 345.:
                    im_tmp.imvec *= self.settings['total_flux_345_SgrA']/im_tmp.total_flux()

            self.ims = np.append(self.ims,im_tmp)

    # store the frequencies of the images
    def determine_freqs(self):
        self.freqs = np.array(list())
        for im in self.ims:
            self.freqs = np.append(self.freqs,np.round(im.rf/1e9))

    # store array object containing the requested base and new sites
    def make_array(self,new_sites):
        self.all_arr = eh.array.load_txt(self.settings['array_file'])
        if new_sites is None:
            self.arr = self.all_arr.make_subarray(self.base_sites)
        else:
            self.arr = self.all_arr.make_subarray(self.base_sites+list(new_sites))

    # compute aperture efficiency
    def eta_dish(self,sigma,offset,freq):
        # sigma : surface RMS, in meters
        # offset : focus offset, in meters
        # freq : observing frequency, in GHz
        eta_dish = np.exp(-((4*np.pi*np.sqrt((sigma)**2+(offset)**2))/(3.e8/(freq*1e9)))**2)
        return eta_dish

    # store receiver temperature
    def set_TR(self,freq):
        # freq : observing frequency, in GHz
        if freq == 230.:
            self.T_R = self.settings['T_R_230']
        elif freq == 345.:
            self.T_R = self.settings['T_R_345']

    # determine the nearest elevation values to an input elevation value
    def nearest_elevations(self,el,elev_array):
        # el : current elevation value
        # elev_array : array of elevations from which to determine the nearest
        indx1 = np.abs(np.subtract.outer(elev_array, el)).argmin(0)
        indx2 = np.abs(np.subtract.outer(np.delete(elev_array,indx1), el)).argmin(0)
        return([elev_array[indx1],np.delete(elev_array,indx1)[indx2]])

    # interpolate tau and Tb across elevation
    def interp_elevation(self,tau,Tb,site,elev,nn_elev,month):
        # tau : dictionary of tau values to use in interpolation
        # Tb : dictionary of Tb values to use in interpolation
        # site : name of site
        # elev : elevation value to use in determining tau and Tb
        # nn_elev : nearest neighbor elevation values
        # month : name of the month

        # interpolate tau quartiles
        tau25 = np.interp(elev,nn_elev,[float(tau[site][25][nn_elev[0]][month]),float(tau[site][25][nn_elev[1]][month])])
        tau50 = np.interp(elev,nn_elev,[float(tau[site][50][nn_elev[0]][month]),float(tau[site][50][nn_elev[1]][month])])
        tau75 = np.interp(elev,nn_elev,[float(tau[site][75][nn_elev[0]][month]),float(tau[site][75][nn_elev[1]][month])])
        tau_quarts = [tau25,tau50,tau75]

        # interpolate Tb quartiles
        Tb25 = np.interp(elev,nn_elev,[float(Tb[site][25][nn_elev[0]][month]),float(Tb[site][25][nn_elev[1]][month])])
        Tb50 = np.interp(elev,nn_elev,[float(Tb[site][50][nn_elev[0]][month]),float(Tb[site][50][nn_elev[1]][month])])
        Tb75 = np.interp(elev,nn_elev,[float(Tb[site][75][nn_elev[0]][month]),float(Tb[site][75][nn_elev[1]][month])])
        Tb_quarts = [Tb25,Tb50,Tb75]

        return(tau_quarts,Tb_quarts)

    # sample and store the weather percentile for each station
    def gen_percentile_for_obs(self):

        # sample percentile from uniform distribution
        roll_dict = defaultdict(dict)
        
        for site in self.arr.tarr['site']:
            tmp_val = np.random.uniform(low=0., high=1.0)
            roll_dict[site] = tmp_val

        # match the weather at co-located sites
        if (('ALMA' in self.arr.tarr['site']) & ('APEX' in self.arr.tarr['site'])):
            roll_dict['APEX'] = roll_dict['ALMA']
        if (('SMA' in self.arr.tarr['site']) & ('JCMT' in self.arr.tarr['site'])):
            roll_dict['JCMT'] = roll_dict['SMA']
        if (('SMT' in self.arr.tarr['site']) & ('KP' in self.arr.tarr['site'])):
            roll_dict['KP'] = roll_dict['SMT']
        
        self.roll_dict = roll_dict


    # generate the random tau values for an observation
    def get_RVs_for_obs(self,t_start,image):
        # t_start : start time, in hours
        # image : input image

        t_int = self.settings['t_int']
        t_rest = self.settings['t_rest']
        tau_RV_dict = defaultdict(dict)
        Tb_RV_dict = defaultdict(dict)

        # create dummy observation for determining elevations
        obs_elev = image.observe(self.arr,t_int,t_rest,t_start,t_start+(t_int/3600.),float(self.settings['bw_fringefind']),ttype=self.settings['ttype'])

        # determine elevations
        if self.python_version == 3:    ## Python 3
            elev_dict = {**dict(zip(obs_elev.unpack('t1')['t1'],obs_elev.unpack('el1')['el1'])),**dict(zip(obs_elev.unpack('t2')['t2'],obs_elev.unpack('el2')['el2']))}
        elif self.python_version == 2:  ## Python 2
            elev_dict = dict(zip(obs_elev.unpack('t1')['t1'].astype('S8'),obs_elev.unpack('el1')['el1'])+zip(obs_elev.unpack('t2')['t2'].astype('S8'),obs_elev.unpack('el2')['el2']))
        
        # loop over stations
        for site in elev_dict.keys():

            # obtain tau and Tb distribution quartiles at this elevation
            [tau_quartiles,Tb_quartiles] = self.interp_elevation(self.atm_dict_tau,self.atm_dict_Tb,site,elev_dict[site],self.nearest_elevations(elev_dict[site],self.elevation_array),self.settings['month'])
            
            # tau distribution mean and standard deviation
            tau_mean = tau_quartiles[1]
            tau_stdev = (tau_quartiles[2]-tau_quartiles[0])/1.349
            if tau_stdev < 0.001:
                tau_stdev = 0.001   # For sites like DomeA

            # generate the tau value
            if (self.settings['weather_randomness'] == 'random'):
                roll_tau = stats.truncnorm.ppf(self.roll_dict[site],(0.-tau_mean)/tau_stdev,(100.-tau_mean)/tau_stdev,loc=tau_mean,scale=tau_stdev)
            elif (self.settings['weather_randomness'] == 'mean'):
                roll_tau = tau_mean
            tau_RV_dict[site] = float(roll_tau)

            # Tb distribution mean and standard deviation
            Tb_mean = Tb_quartiles[1]
            Tb_stdev = (Tb_quartiles[2]-Tb_quartiles[0])/1.35

            # generate the Tb value
            if (self.settings['weather_randomness'] == 'random'):
                roll_Tb = stats.truncnorm.ppf(self.roll_dict[site],(0.-Tb_mean)/Tb_stdev,(100.-Tb_mean)/Tb_stdev,loc=Tb_mean,scale=Tb_stdev)
            elif (self.settings['weather_randomness'] == 'mean'):
                roll_Tb = Tb_mean
            Tb_RV_dict[site] = float(roll_Tb)

            # compute and store station SEFDs
            Tsys = (self.T_R+Tb_RV_dict[site])*np.exp(tau_RV_dict[site])
            SEFD = (2.0*k*Tsys)/((np.pi/4.0)*self.eta_dict[site]*(self.D_dict[site])**2)
            self.arr.tarr[self.arr.tkey[site]][4] = SEFD
            self.arr.tarr[self.arr.tkey[site]][5] = SEFD
            print("=========================================")
            print("SEFD of {0}: {1} Jy".format(site,SEFD))
            print("tau: {0}".format(roll_tau))
            print("Tb: {0}".format(roll_Tb))
            roll_percentile = 0.5*(1.+erf((roll_tau-tau_mean)/(np.sqrt(2)*tau_stdev)))
            print("percentile: {0}".format(roll_percentile))

        return(dict(tau_RV_dict), elev_dict)

    # generate dictionaries of atmospheric properties
    def create_atm_dictionary(self,sites,frequency):
        # sites : stations in the array
        # frequency : observing freqnuency, in GHz

        month_name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

        atm_dict_tau = defaultdict(dict)
        atm_dict_Tb = defaultdict(dict)

        for site in sites:
            atm_dict_tau[site] = {}
            atm_dict_tau[site][25] = {}
            atm_dict_tau[site][50] = {}
            atm_dict_tau[site][75] = {}
            atm_dict_Tb[site] = {}
            atm_dict_Tb[site][25] = {}
            atm_dict_Tb[site][50] = {}
            atm_dict_Tb[site][75] = {}

        for quartile in [25,50,75]:

            # optical depths
            files_list_tau = glob.glob(self.settings['path_to_weather']+'tau*quart{0}*frq'.format(quartile)+str(int(frequency))+'.txt')
            files_list_tau.sort()
            for file in files_list_tau:
                elev = 90.-float(file.split('ZA')[2].split('_')[0])
                sites_file = np.loadtxt(file,usecols=(0),unpack=True,dtype='str')
                vals_file = np.loadtxt(file,usecols=(1,2,3,4,5,6,7,8,9,10,11,12),unpack=True)
                for isite, site_here in enumerate(sites_file):
                    if site_here in sites:
                        atm_dict_tau[site_here][quartile][elev] = dict(zip(month_name,vals_file[:,isite]))
                        if ((site_here == 'APEX') & ('ALMA' in sites)):
                            atm_dict_tau['ALMA'][quartile][elev] = dict(zip(month_name,vals_file[:,isite]))
                        if ((site_here == 'SMA') & ('JCMT' in sites)):
                            atm_dict_tau['JCMT'][quartile][elev] = dict(zip(month_name,vals_file[:,isite]))
                

            # brightness temperatures
            files_list_Tb = glob.glob(self.settings['path_to_weather']+'Tb*quart{0}*frq'.format(quartile)+str(int(frequency))+'.txt')
            files_list_Tb.sort()
            elevation_list = list()
            for file in files_list_Tb:
                elev = 90.-float(file.split('ZA')[2].split('_')[0])
                elevation_list.append(elev)
                sites_file = np.loadtxt(file,usecols=(0),unpack=True,dtype='str')
                vals_file = np.loadtxt(file,usecols=(1,2,3,4,5,6,7,8,9,10,11,12),unpack=True)
                for isite, site_here in enumerate(sites_file):
                    if site_here in sites:
                        atm_dict_Tb[site_here][quartile][elev] = dict(zip(month_name,vals_file[:,isite]))
                        if ((site_here == 'APEX') & ('ALMA' in sites)):
                            atm_dict_Tb['ALMA'][quartile][elev] = dict(zip(month_name,vals_file[:,isite]))
                        if ((site_here == 'SMA') & ('JCMT' in sites)):
                            atm_dict_Tb['JCMT'][quartile][elev] = dict(zip(month_name,vals_file[:,isite]))
                

        elevation_array = np.unique(np.array(elevation_list))
        self.atm_dict_tau = atm_dict_tau
        self.atm_dict_Tb = atm_dict_Tb
        self.elevation_array = elevation_array

    # generate dictionaries of telescope properties
    def initialize_dicts(self,sites,freq):
        # sites : stations in the array
        # freq : observing freqnuency, in GHz

        print("========= freq: {0}".format(freq))
        
        focus_offset = float(self.settings['focus_offset'])
        eta_new = self.eta_dish(float(self.settings['sigma_surface']),focus_offset,freq)
        
        # diameters of existing telescopes, in meters
        D_existing_dict = self.settings['D_existing_dict']

        # aperture efficiencies of existing telescopes
        eta_existing_dict = {'ALMA':self.eta_dish(65e-6,0.,freq),'APEX':self.eta_dish(73e-6,0.,freq),
        'GLT':self.eta_dish(68e-6,0.,freq),'PV':self.eta_dish(90e-6,0.,freq),'KP':self.eta_dish(75e-6,0.,freq),
        'LMT':self.eta_dish(117e-6,0.,freq),'NOEMA':self.eta_dish(86e-6,0.,freq),'SMA':self.eta_dish(62e-6,0.,freq),
        'JCMT':self.eta_dish(84e-6,0.,freq),'SMT':self.eta_dish(74e-6,0.,freq),'SPT':self.eta_dish(74e-6,0.,freq),
        'HAY':self.eta_dish(100e-6,focus_offset,freq)*0.5,'NOB':self.eta_dish(100e-6,focus_offset,freq),
        'SUF':self.eta_dish(150e-6,focus_offset,freq),'KVNYS':self.eta_dish(124e-6,0.,freq),
        'GAM':self.eta_dish(65e-6,focus_offset,freq),'LAS':self.eta_dish(65e-6,focus_offset,freq),
        'OVRO':self.eta_dish(50e-6,focus_offset,freq)}

        # SEFD error budget for existing telescopes
        SEFD_error_budget_existing_dict = {'ALMA':0.10,'APEX':0.11,'SMT':0.07,'LMT':0.22,
            'PV':0.10,'SMA':0.15,'JCMT':0.14,'SPT':0.07,'GLT':0.10}

        D_dict = {}
        FWHM_beam = {}
        eta_dict = {}
        SEFD_error_budget_dict = {}
        for site in sites:
            if site in D_existing_dict.keys():
                D_dict[site] = D_existing_dict[site]
            else:
                D_dict[site] = self.settings['D_new']
            FWHM_beam[site] = (c/(freq*1e9))/D_dict[site]*(60.*60.*180./np.pi)    #arc-seconds

            if site in eta_existing_dict.keys():
                eta_dict[site] = eta_existing_dict[site]
            else:
                eta_dict[site] = eta_new

            if site in SEFD_error_budget_existing_dict.keys():
                SEFD_error_budget_dict[site] = SEFD_error_budget_existing_dict[site]
            else:
                SEFD_error_budget_dict[site] = self.settings['SEFD_error_budget_new']

        self.D_dict = D_dict
        self.FWHM_beam_dict = FWHM_beam
        # self.offset_rand_dict = dict(zip(sites,np.random.normal(scale=self.settings['RMS_point_err'],size=sites.shape)))
        self.eta_dict = eta_dict
        self.SEFD_error_budget_dict = SEFD_error_budget_dict
        self.create_atm_dictionary(sites,freq)

    # segment the observation into timestamps
    def get_obs_times(self):
        t_first = self.settings['t_start']
        N_obs = int(np.ceil(self.settings['dt']/(self.settings['t_rest']/3600.)))
        t_last = t_first+float(N_obs-1)*(self.settings['t_rest']/3600.)
        self.t_seg_times = np.linspace(t_first,t_last,N_obs)
        print("========= N_obs {0}".format(N_obs))
        print("========= t_first {0}".format(t_first))
        print("========= t_last {0}".format(t_last))

    # generate observation
    def make_obs(self,band_count,new_sites=None):
        
        # initialize the array object self.arr
        self.make_array(new_sites)

        # generate weather percentiles
        self.gen_percentile_for_obs()
        print("************** {0}".format(self.roll_dict))

        # determine SNR thresholding scheme and values
        snr_algo, snr_args = self.settings['SNR_cutoff']

        # initiate observation list
        obs = list()

        # loop through the images
        for im in self.ims:

            # initialization
            self.initialize_dicts(self.arr.tarr['site'],np.round(im.rf/1e9))
            self.set_TR(np.round(im.rf/1e9))
            print("************** T_R set to : {0}".format(self.T_R))
            self.get_obs_times()
            print('************** Scan start times: {0}'.format(self.t_seg_times))

            freq_sky = im.rf

            # loop through the bands
            for i_band in range(band_count):

                # make a copy of the original image
                im_tmp = im.copy()

                # adjust the observing frequency to account for band offset from nominal frequency
                band_multiplier = [-0.5,0.5,-1.5,1.5]
                if band_multiplier[i_band]<0.:
                    im_tmp.rf = freq_sky+float(float(self.settings['bw_fringefind'])*band_multiplier[i_band]-float(self.settings['rf_offset']))
                else:
                    im_tmp.rf = freq_sky+float(float(self.settings['bw_fringefind'])*band_multiplier[i_band]+float(self.settings['rf_offset']))

                # loop through the start times
                for t_seg_time in self.t_seg_times:

                    try:

                        # get tau values, passing im (rather than im_tmp) so the frequency isn't offset
                        tau_RV_dict, elev_dict = self.get_RVs_for_obs(t_seg_time,im)

                        # convert to zenith opacities
                        tau_RV_dict_zenith = tau_RV_dict.copy()
                        for site in tau_RV_dict.keys():
                            tau_RV_dict_zenith[site] = tau_RV_dict[site]*np.sin((np.pi/180.)*elev_dict[site])
                        
                        # generate observation
                        print('--------- observing at {0} UTC----------'.format(t_seg_time))
                        obs_seg = im.observe(self.arr,self.settings['t_int'],self.settings['t_rest'],t_seg_time,t_seg_time+(self.settings['t_int'])/3600.,float(self.settings['bw_fringefind']),add_th_noise=False,jones=True,inv_jones=True,opacitycal=False,phasecal=False,ampcal=False,tau=tau_RV_dict_zenith,taup=0,elevmin=self.settings['el_min'],elevmax=90.0,ttype=self.settings['ttype'])
                        
                        # apply naive SNR thresholding
                        if (snr_algo == 'naive'):
                            obs_seg = obs_seg.flag_low_snr(snr_cut=snr_args,output='kept')

                        # apply an ad hoc phasing proxy for SNR thresholding
                        elif (snr_algo == 'adhoc'):

                            # parse SNR_cutoff arguments
                            snr_ALMA = snr_args[0]
                            tint_ALMA = snr_args[1]
                            snr_noALMA = snr_args[2]
                            snr_backup = snr_args[3]

                            # check if ALMA is in the array
                            stations_here = np.unique(np.concatenate((obs_seg.data['t1'],obs_seg.data['t2'])))
                            
                            # if ALMA is not in the array, resort to the backup SNR threshold
                            if 'ALMA' not in stations_here:
                                obs_seg = obs_seg.flag_low_snr(snr_cut=snr_backup,output='kept')

                            # if ALMA is in the array, assume ad hoc phasing will be used
                            else:

                                # generate an observation with the ALMA integration time
                                obs_precheck = im.observe(self.arr,tint_ALMA,self.settings['t_rest'],t_seg_time,t_seg_time+(self.settings['t_int'])/3600.,float(self.settings['bw_fringefind']),add_th_noise=False,jones=True,inv_jones=True,opacitycal=False,phasecal=False,ampcal=False,tau=tau_RV_dict_zenith,taup=0,elevmin=self.settings['el_min'],elevmax=90.0,ttype=self.settings['ttype'])
                                
                                # parse antenna info
                                ant1_precheck = obs_precheck.data['t1']
                                ant2_precheck = obs_precheck.data['t2']
                                ant1_seg = obs_seg.data['t1']
                                ant2_seg = obs_seg.data['t2']

                                # compute SNR values on both integration times
                                snr_precheck = np.abs(obs_precheck.data['vis'])/obs_precheck.data['sigma']
                                snr_seg = np.abs(obs_seg.data['vis'])/obs_seg.data['sigma']

                                # loop through stations
                                for istat, station_here in enumerate(stations_here):

                                    # check baselines between this station and ALMA
                                    index_precheck = ((ant1_precheck == station_here) & (ant2_precheck == 'ALMA')) | ((ant2_precheck == station_here) & (ant1_precheck == 'ALMA'))
                                    if index_precheck.sum() > 0:

                                        # if the baseline to ALMA has too low SNR on the ALMA integration time, flag it
                                        if snr_precheck[index_precheck] < snr_ALMA:
                                            print('SNR on ALMA-'+station_here+' baseline is '+str(snr_precheck[index_precheck][0])+', which is less than the specified threshold of '+str(snr_ALMA)+'; flagging this baseline.')
                                            obs_seg = obs_seg.flag_sites([station_here],output='kept')

                                        # otherwise, check if the non-ALMA baselines have sufficient SNR on the full segment integration time
                                        else:
                                            index_seg = ((ant1_seg == station_here) & (ant2_seg != 'ALMA')) | ((ant2_seg == station_here) & (ant1_seg != 'ALMA'))
                                            index_seg &= (snr_seg < snr_noALMA)
                                            if index_seg.sum() > 0:
                                                ant1list_here = ant1_seg[index_seg]
                                                ant2list_here = ant2_seg[index_seg]
                                                snr_seg_here = snr_seg[index_seg]
                                                for iant in range(len(ant1list_here)):
                                                    print('SNR on '+ant1list_here[iant]+'-'+ant2list_here[iant]+' baseline is '+str(snr_seg_here[iant])+', which is less than the specified threshold of '+str(snr_noALMA)+'; flagging this baseline.')
                                                    obs_seg = obs_seg.flag_bl([ant1list_here[iant],ant2list_here[iant]])
                                    
                                    # if this station has no baselines to ALMA, then flag per the backup SNR threshold
                                    else:
                                        index_seg = ((ant1_seg == station_here) & (ant2_seg != 'ALMA')) | ((ant2_seg == station_here) & (ant1_seg != 'ALMA'))
                                        index_seg &= (snr_seg < snr_backup)
                                        if index_seg.sum() > 0:
                                            ant1list_here = ant1_seg[index_seg]
                                            ant2list_here = ant2_seg[index_seg]
                                            snr_seg_here = snr_seg[index_seg]
                                            for iant in range(len(ant1list_here)):
                                                print('SNR on '+ant1list_here[iant]+'-'+ant2list_here[iant]+' baseline is '+str(snr_seg_here[iant])+', which is less than the specified threshold of '+str(snr_backup)+'; flagging this baseline.')
                                                obs_seg = obs_seg.flag_bl([ant1list_here[iant],ant2list_here[iant]])

                        # unrecognized SNR thresholding scheme
                        else:
                            raise ValueError('unknown algorithm for SNR_cutoff')

                        # append this segment to the obs list
                        if obs == []:
                            obs = obs_seg.copy()
                        else:
                            obs.data['time'] += 0.00001
                            obs.data = np.concatenate([obs.data,obs_seg.data])

                    except Exception as e:
                        print('--------- No sites up --------', file=sys.stderr)
                        print('actual exception is:', str(e), file=sys.stderr)
                        traceback.print_exc()

        return(obs)

    # defines a specific MJD associated with an observing month
    def determine_mjd(self):
        if (self.settings['month'] == 'Jan'):
            t = Time('2020-01-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Feb'):
            t = Time('2020-02-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Mar'):
            t = Time('2020-03-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Apr'):
            t = Time('2020-04-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'May'):
            t = Time('2020-05-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Jun'):
            t = Time('2020-06-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Jul'):
            t = Time('2020-07-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Aug'):
            t = Time('2020-08-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Sep'):
            t = Time('2020-09-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Oct'):
            t = Time('2020-10-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Nov'):
            t = Time('2020-11-15T00:00:00',format='isot',scale='utc')
        elif (self.settings['month'] == 'Dec'):
            t = Time('2020-12-15T00:00:00',format='isot',scale='utc')
        else:
            print('This month abbreviation is not recognized; should be one of: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec')

        self.mjd = t.mjd
        
    # extract the opacity quartiles from weather tables
    def tau_quartiles(self,freq):
        # note: freq here is a string

        # determine which column of the tables to read
        month_arr = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        col = 1 + np.where(month_arr == self.settings['month'])[0][0]

        # weather table filenames
        file_25 = self.settings['path_to_weather']+'tau_ZA0_quart25_frq'+freq+'.txt'
        file_50 = self.settings['path_to_weather']+'tau_ZA0_quart50_frq'+freq+'.txt'
        file_75 = self.settings['path_to_weather']+'tau_ZA0_quart75_frq'+freq+'.txt'

        # get the zenith tau values at every site for the requested month
        sites = np.loadtxt(file_25,usecols=(0),unpack=True,dtype='str')
        taus_25 = np.loadtxt(file_25,usecols=(col),unpack=True)
        taus_50 = np.loadtxt(file_50,usecols=(col),unpack=True)
        taus_75 = np.loadtxt(file_75,usecols=(col),unpack=True)

        return sites, taus_25, taus_50, taus_75

    # extract the brightness temperature quartiles from weather tables
    def Tb_quartiles(self,freq):
        # note: freq here is a string

        # determine which column of the tables to read
        month_arr = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        col = 1 + np.where(month_arr == self.settings['month'])[0][0]

        # weather table filenames
        file_25 = self.settings['path_to_weather']+'Tb_ZA0_quart25_frq'+freq+'.txt'
        file_50 = self.settings['path_to_weather']+'Tb_ZA0_quart50_frq'+freq+'.txt'
        file_75 = self.settings['path_to_weather']+'Tb_ZA0_quart75_frq'+freq+'.txt'

        # get the zenith tau values at every site for the requested month
        sites = np.loadtxt(file_25,usecols=(0),unpack=True,dtype='str')
        Tb_25 = np.loadtxt(file_25,usecols=(col),unpack=True)
        Tb_50 = np.loadtxt(file_50,usecols=(col),unpack=True)
        Tb_75 = np.loadtxt(file_75,usecols=(col),unpack=True)

        return sites, Tb_25, Tb_50, Tb_75

    # observation generating function
    def observe(self,im,freq,opacitycal=True,fft_pad_factor=2,apply_pointing_errors=True):
        # note: freq here is a string

        # generate empty observation
        obs_temp = self.arr.obsdata(im.ra,
                                    im.dec,
                                    im.rf,
                                    float(self.settings['bw_fringefind']),
                                    self.settings['t_int'],
                                    self.settings['t_rest'],
                                    self.settings['t_start'],
                                    self.settings['t_start'] + self.settings['dt'],
                                    mjd = self.mjd,
                                    polrep = im.polrep,
                                    tau = 0.0,
                                    timetype = 'UTC',
                                    elevmin = self.settings['el_min'],
                                    elevmax = 90.0,
                                    fix_theta_GMST = False)

        # observe the source
        obs = im.observe_same_nonoise(obs_temp,ttype=self.settings['ttype'],fft_pad_factor=fft_pad_factor)
        obs.mjd = obs_temp.mjd

        # make sure we're in a circular basis
        obs = obs.switch_polrep(polrep_out='circ')

        # extract elevation information
        t1 = obs.data['t1']
        t2 = obs.data['t2']
        sites_obs = np.unique(np.concatenate((t1,t2)))
        els = obs.unpack(['el1','el2'],ang_unit='rad')
        el1 = els['el1']
        el2 = els['el2']

        # get opacity and brightness temperature distributions
        tau_sites, taus_25, taus_50, taus_75 = self.tau_quartiles(freq)
        Tb_sites, Tbs_25, Tbs_50, Tbs_75 = self.Tb_quartiles(freq)
        
        # initialize dictionaries
        tau_RV_dict = defaultdict(dict)
        Tb_RV_dict = defaultdict(dict)

        # initialize various arrays
        tau1 = np.zeros_like(el1)
        tau2 = np.zeros_like(el2)
        Tb1 = np.zeros_like(el1)
        Tb2 = np.zeros_like(el2)
        Tsys1 = np.zeros_like(el1)
        Tsys2 = np.zeros_like(el2)
        SEFD1 = np.zeros_like(el1)
        SEFD2 = np.zeros_like(el2)
        
        for isite in range(len(sites_obs)):

            site = sites_obs[isite]
            site_label = site

            # handle co-located sites
            if (site_label == 'ALMA'):
                site = 'APEX'
            if (site_label == 'JCMT'):
                site = 'SMA'
            if (site_label == 'BAR'):
                site = 'OVRO'

            # indices for this site
            ind_obs1 = (t1 == site_label)
            ind_obs2 = (t2 == site_label)
            ind_tau = (tau_sites == site)
            ind_Tb = (Tb_sites == site)

            # zenith tau distribution mean and standard deviation
            tau_mean = taus_50[ind_tau]
            tau_stdev = (taus_75[ind_tau]-taus_25[ind_tau])/1.349
            if (tau_stdev < 0.001):
                tau_stdev = 0.001   # For sites like DomeA

            # zenith brightness temperature distribution mean and standard deviation
            Tb_mean = Tbs_50[ind_Tb]
            Tb_stdev = (Tbs_75[ind_Tb]-Tbs_25[ind_Tb])/1.349

            # generate the zenith tau and Tb values
            roll_tau = stats.truncnorm.ppf(self.roll_dict[site_label],(0.-tau_mean)/tau_stdev,(100.-tau_mean)/tau_stdev,loc=tau_mean,scale=tau_stdev)[0]
            roll_Tb = stats.truncnorm.ppf(self.roll_dict[site_label],(0.-Tb_mean)/Tb_stdev,(100.-Tb_mean)/Tb_stdev,loc=Tb_mean,scale=Tb_stdev)[0]

            # divide out the opacity term to get the actual atmospheric Tb
            roll_Tb /= (1.0 - np.exp(-roll_tau))

            # store the zenith tau value
            tau_RV_dict[site_label] = roll_tau
            Tb_RV_dict[site_label] = roll_Tb

            # get actual opacities at each timestamp
            tau1[ind_obs1] = roll_tau / np.cos((np.pi/2.0) - el1[ind_obs1])
            tau2[ind_obs2] = roll_tau / np.cos((np.pi/2.0) - el2[ind_obs2])

            # get actual Tb contributions at each timestamp
            Tb1[ind_obs1] = roll_Tb * (1.0 - np.exp(-tau1[ind_obs1]))
            Tb2[ind_obs2] = roll_Tb * (1.0 - np.exp(-tau2[ind_obs2]))

            # determine system temperatures
            Tsys1[ind_obs1] = self.T_R + Tb1[ind_obs1]
            Tsys2[ind_obs2] = self.T_R + Tb2[ind_obs2]

            # determine SEFDs
            SEFD1[ind_obs1] = (2.0*k*Tsys1[ind_obs1])/((np.pi/4.0)*self.eta_dict[site_label]*(self.D_dict[site_label])**2)
            SEFD2[ind_obs2] = (2.0*k*Tsys2[ind_obs2])/((np.pi/4.0)*self.eta_dict[site_label]*(self.D_dict[site_label])**2)

        # store opacities as part of the observation
        obs.data['tau1'] = tau1
        obs.data['tau2'] = tau2

        # store things differently depending on whether opacity is assumed to be calibrated or not
        if opacitycal:

            # determine baseline thermal noise levels
            tint = obs.data['tint']
            sigma = np.sqrt((SEFD1*SEFD2*np.exp(tau1)*np.exp(tau2))/(2.0*obs.bw*tint)) / 0.88
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
            sigma = np.sqrt((SEFD1*SEFD2)/(2.0*obs.bw*tint)) / 0.88
            obs.data['rrsigma'] = sigma
            obs.data['llsigma'] = sigma
            obs.data['rlsigma'] = sigma
            obs.data['lrsigma'] = sigma

        # Apply pointing errors
        if apply_pointing_errors:
            t1 = obs.data['t1']
            t2 = obs.data['t2']
            gain1 = np.ones(t1.shape)
            gain2 = np.ones(t2.shape)

            for si in obs.tarr['site']:
                if si in self.settings['D_existing_dict'].keys():
                    gain1[t1 == si] *= np.exp(-8*np.log(2)**2*((np.random.normal(scale=self.FWHM_beam_dict[si]/self.settings['existing_pt_accuracy_factor'],size=gain1[t1 == si].shape)/self.FWHM_beam_dict[si])**2))
                    gain2[t2 == si] *= np.exp(-8*np.log(2)**2*((np.random.normal(scale=self.FWHM_beam_dict[si]/self.settings['existing_pt_accuracy_factor'],size=gain2[t2 == si].shape)/self.FWHM_beam_dict[si])**2))
                else:
                    gain1[t1 == si] *= np.exp(-8*np.log(2)**2*((np.random.normal(scale=self.settings['RMS_point_err'],size=gain1[t1 == si].shape)/self.FWHM_beam_dict[si])**2))
                    gain2[t2 == si] *= np.exp(-8*np.log(2)**2*((np.random.normal(scale=self.settings['RMS_point_err'],size=gain2[t2 == si].shape)/self.FWHM_beam_dict[si])**2))

            print("Visibility gains range from {0} to {1} with median {2}".format(np.min(gain1*gain2),np.max(gain1*gain2),np.median(gain1*gain2)))
            obs.data['rrvis'] *= gain1*gain2
            obs.data['llvis'] *= gain1*gain2
            obs.data['rlvis'] *= gain1*gain2
            obs.data['lrvis'] *= gain1*gain2

        # restore Stokes polrep
        obs = obs.switch_polrep(polrep_out='stokes')

        return obs

    # generate observation
    def make_obs2(self,band_count,freq='230',new_sites=None,opacitycal=True,fft_pad_factor=2,verbose=False,apply_pointing_errors=True):
        
        # make sure the frequency is one we have
        if freq not in ['86', '230', '345']:
            print('WARNING: parameter "freq" should be one of ',freq,' and should be a string.')

        # initialize the array object self.arr
        self.make_array(new_sites)

        # generate weather percentiles
        self.gen_percentile_for_obs()

        # determine SNR thresholding scheme and values
        snr_algo, snr_args = self.settings['SNR_cutoff']

        # initiate observation list
        obs = list()

        # loop through the images
        for im in self.ims:
            
            # initialization
            self.initialize_dicts(self.arr.tarr['site'],float(freq))
            self.set_TR(float(freq))
            if verbose:
                print("************** T_R set to : {0}".format(self.T_R))
            self.get_obs_times()
            # print('************** Scan start times: {0}'.format(self.t_seg_times))
            
            freq_sky = float(freq)*(1.0e9)
            im.rf = freq_sky
            
            # loop through the bands
            for i_band in range(band_count):

                # make a copy of the original image
                im_tmp = im.copy()

                # adjust the observing frequency to account for band offset from nominal frequency
                band_multiplier = [-0.5,0.5,-1.5,1.5]
                if band_multiplier[i_band]<0.:
                    im_tmp.rf = freq_sky+float(float(self.settings['bw_fringefind'])*band_multiplier[i_band]-float(self.settings['rf_offset']))
                else:
                    im_tmp.rf = freq_sky+float(float(self.settings['bw_fringefind'])*band_multiplier[i_band]+float(self.settings['rf_offset']))

                # generate observation
                obs_seg = self.observe(im_tmp,freq,opacitycal=opacitycal,fft_pad_factor=fft_pad_factor,apply_pointing_errors=apply_pointing_errors)

                # apply naive SNR thresholding
                if (snr_algo == 'naive'):
                    obs_seg = obs_seg.flag_low_snr(snr_cut=snr_args,output='kept')
                
                # apply an ad hoc phasing proxy for SNR thresholding
                elif (snr_algo == 'adhoc'):

                    # parse SNR_cutoff arguments
                    snr_ALMA = snr_args[0]
                    tint_ALMA = snr_args[1]
                    snr_noALMA = snr_args[2]
                    snr_backup = snr_args[3]

                    # check if ALMA is in the array
                    stations_here = np.unique(np.concatenate((obs_seg.data['t1'],obs_seg.data['t2'])))
                    
                    # if ALMA is not in the array, resort to the backup SNR threshold
                    if 'ALMA' not in stations_here:
                        obs_seg = obs_seg.flag_low_snr(snr_cut=snr_backup,output='kept')

                    # if ALMA is in the array, assume ad hoc phasing will be used
                    else:

                        # scale the snr to use the ALMA integration time
                        snr_seg = np.abs(obs_seg.data['vis'])/obs_seg.data['sigma']
                        scale_factor = np.sqrt(tint_ALMA / obs_seg.data['tint'])
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

                            # if ALMA is not in the segment, resort to the backup SNR threshold
                            if 'ALMA' not in stations_timestamp:
                                ind_t &= (snr_precheck >= snr_backup)
                                master_index += ind_t

                            # if ALMA is in the segment, use ad hoc phasing
                            else:

                                # loop through stations
                                for istat, station_here in enumerate(stations_timestamp):

                                    # check baselines between this station and ALMA
                                    index_precheck = ((ant1 == station_here) & (ant2 == 'ALMA')) | ((ant2 == station_here) & (ant1 == 'ALMA'))
                                    index_precheck &= ind_t
                                    if index_precheck.sum() > 0:

                                        # if the baseline to ALMA has too low SNR on the ALMA integration time, flag it
                                        if snr_precheck[index_precheck] < snr_ALMA:
                                            if verbose:
                                                print('SNR on ALMA-'+station_here+' baseline is '+str(snr_precheck[index_precheck][0])+', which is less than the specified threshold of '+str(snr_ALMA)+'; flagging this baseline.')

                                        # otherwise, check if the non-ALMA baselines have sufficient SNR
                                        else:

                                            if verbose:
                                                index_seg = ((ant1 == station_here) & (ant2 != 'ALMA')) | ((ant2 == station_here) & (ant1 != 'ALMA'))
                                                index_seg &= (snr_precheck < snr_noALMA)
                                                index_seg &= ind_t

                                                # announce which baselines fail the SNR criterion
                                                if index_seg.sum() > 0:
                                                    ant1list_here = ant1[index_seg]
                                                    ant2list_here = ant2[index_seg]
                                                    snr_seg_here = snr_precheck[index_seg]
                                                    for iant in range(len(ant1list_here)):
                                                        print('SNR on '+ant1list_here[iant]+'-'+ant2list_here[iant]+' baseline is '+str(snr_seg_here[iant])+', which is less than the specified threshold of '+str(snr_noALMA)+'; flagging this baseline.')
                                                
                                            # retain the baselines that satisfy the SNR criterion
                                            index_seg = ((ant1 == station_here) & (ant2 != 'ALMA')) | ((ant2 == station_here) & (ant1 != 'ALMA'))
                                            index_seg &= (snr_precheck >= snr_noALMA)
                                            index_seg &= ind_t
                                            master_index += index_seg

                                    # if this station has no baselines to ALMA, then flag per the backup SNR threshold
                                    else:

                                        if verbose:
                                            index_seg = ((ant1 == station_here) & (ant2 != 'ALMA')) | ((ant2 == station_here) & (ant1 != 'ALMA'))
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
                                        index_seg = ((ant1 == station_here) & (ant2 != 'ALMA')) | ((ant2 == station_here) & (ant1 != 'ALMA'))
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
        sites_to_drop = list()
        for dy in range(self.settings['window_to_choose']):
            sites_to_drop_tmp = self.get_unready_sites(sites_in_observ=sites_in_obs)
            print("Tech readiness trial {0} dropped {1}".format(dy,sites_to_drop_tmp))
            if (sites_to_drop == list()) or (len(sites_to_drop_tmp)<len(sites_to_drop)):
                sites_to_drop = sites_to_drop_tmp
        print("Dropping {0} due to technincal unreadinness.".format(sites_to_drop))
        if len(sites_to_drop) == len(sites_in_obs):
            sites_to_drop = sites_to_drop[2:]
        if len(sites_to_drop) > 0:
            obs = obs.flag_sites(sites_to_drop)
        return obs

    def get_unready_sites(self,sites_in_observ):
        if type(self.settings['tech_readiness']) is dict:
            print("ERROR. Dictionary of technical readiness values not implemented")
        elif type(self.settings['tech_readiness']) is float:
            sites_to_drop = sites_in_observ[np.random.choice([0, 1], size=(len(sites_in_observ)), p=[self.settings['tech_readiness'], 1-self.settings['tech_readiness']]).astype(bool)]
        else:
            print("ERROR. Technical readiness must be dict or float")
        return(sites_to_drop)

    # compute the (u,v) filling fraction on snapshots
    def calc_fill_frac_continuous(self,obs,longest_BL,start_time=0.0,end_time=24.0,snapshot_interval=600.0):
        # expensive if snapshot_interval (given in seconds) is small
        
        # observation info
        times_obs = obs.data['time']
        datatable = obs.data.copy()
        
        # make a small blank copy to use for snapshots
        obs_blank = obs.copy()
        obs_blank.data = None
        
        # compute fill fractions
        times_temp = np.copy(self.t_seg_times)
        times = np.concatenate((times_temp,[2.0*times_temp[-1]-times_temp[-2]]))
        fills = np.zeros(len(times)-1)
        for itime in range(len(times)-1):
            UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
            if (UT_mask.sum() > 0):
                obs_snapshot = obs_blank.copy()
                obs_snapshot.data = datatable[UT_mask]
                fill_snapshot = ff.obs_fill(obs_snapshot,fov=self.settings['fillfov'],longest=longest_BL,N=self.settings['fillpix'])
                fills[itime] = fill_snapshot
        fill_mean = np.mean(fills)
        fill_median = np.median(fills)
        fill_max = np.max(fills)
        fill = {'fill_mean': fill_mean, 'fill_median': fill_median, 'fill_max': fill_max}
        
        return fill

    # parse and execute the (u,v) filling fraction requests
    def calc_fill_frac(self,obs):

        # longest baseline
        longest_BL = D_Earth/(c/(np.max(self.freqs)*1e9))

        # if computing only on a full observation
        if (self.settings['ff_type'] == 'fullobs'):
            fill = ff.obs_fill(obs,fov=self.settings['fillfov'],longest=longest_BL,N=self.settings['fillpix'])
        
        # if computing on individual snapshots
        elif (self.settings['ff_type'] == 'snapshot'):
            fill_fullobs = ff.obs_fill(obs,fov=self.settings['fillfov'],longest=longest_BL,N=self.settings['fillpix'])
            fill_snapshot = self.calc_fill_frac_continuous(obs, longest_BL,snapshot_interval=self.settings['t_rest'])
            fill = {'fill_fullobs': fill_fullobs, 'fill_mean': fill_snapshot['fill_mean'], 'fill_median': fill_snapshot['fill_median'], 'fill_max': fill_snapshot['fill_max']}
            
        return fill

    # compute the extended (u,v) filling fraction on snapshots
    def calc_bff_continuous(self,obs,longest_BL,start_time=0.0,end_time=24.0,snapshot_interval=600.0,logmid=1.5,logwid=0.525,stokes='I'):
        # expensive if snapshot_interval (given in seconds) is small
        
        # observation info
        times_obs = obs.data['time']
        datatable = obs.data.copy()
        
        # make a small blank copy to use for snapshots
        obs_blank = obs.copy()
        obs_blank.data = None
        
        # compute fill fractions
        times_temp = np.copy(self.t_seg_times)
        times = np.concatenate((times_temp,[2.0*times_temp[-1]-times_temp[-2]]))
        bffs = np.zeros(len(times)-1)
        for itime in range(len(times)-1):
            UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
            if (UT_mask.sum() > 0):
                obs_snapshot = obs_blank.copy()
                obs_snapshot.data = datatable[UT_mask]
                bff_snapshot = bff.obs_fill(obs_snapshot,fov=self.settings['fillfov'],longest=longest_BL,N=self.settings['fillpix'],logmid=logmid,logwid=logwid,stokes=stokes)
                bffs[itime] = bff_snapshot
        bff_mean = np.mean(bffs)
        bff_median = np.median(bffs)
        bff_max = np.max(bffs)
        bff_out = {'bff_mean': bff_mean, 'bff_median': bff_median, 'bff_max': bff_max}
        
        return bff_out

    # parse and execute the extended (u,v) filling fraction requests
    def calc_bff(self,obs,logmid=1.5,logwid=0.525,stokes='I'):

        # longest baseline
        longest_BL = D_Earth/(c/(np.max(self.freqs)*1e9))

        # if computing only on a full observation
        if (self.settings['ff_type'] == 'fullobs'):
            bff_out = bff.obs_fill(obs,fov=self.settings['fillfov'],longest=longest_BL,N=self.settings['fillpix'],logmid=logmid,logwid=logwid,stokes=stokes)
        
        # if computing on individual snapshots
        elif (self.settings['ff_type'] == 'snapshot'):
            bff_fullobs = bff.obs_fill(obs,fov=self.settings['fillfov'],longest=longest_BL,N=self.settings['fillpix'],logmid=logmid,logwid=logwid,stokes=stokes)
            bff_snapshot = self.calc_bff_continuous(obs, longest_BL,snapshot_interval=self.settings['t_rest'],logmid=logmid,logwid=logwid,stokes=stokes)
            bff_out = {'bff_fullobs': bff_fullobs, 'bff_mean': bff_snapshot['bff_mean'], 'bff_median': bff_snapshot['bff_median'], 'bff_max': bff_snapshot['bff_max']}
            
        return bff_out

    # compute the largest circular gap on snapshots
    def calc_lcg_continuous(self,obs,start_time=0.0,end_time=24.0,snapshot_interval=600.0,method='analytic',tavg=None,scan_avg=False,dummy_circ=True,dummy_circ_res=None,plot_solution=False,niter=1000,specify_x0=None):
        # expensive if snapshot_interval (given in seconds) is small
        
        # observation info
        times_obs = obs.data['time']
        datatable = obs.data.copy()
        
        # make a small blank copy to use for snapshots
        obs_blank = obs.copy()
        obs_blank.data = None
        
        # compute LCG in snapshots
        times_temp = np.copy(self.t_seg_times)
        times = np.concatenate((times_temp,[2.0*times_temp[-1]-times_temp[-2]]))
        lcgs = np.zeros(len(times)-1)
        for itime in range(len(times)-1):
            UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
            if (UT_mask.sum() > 0):
                obs_snapshot = obs_blank.copy()
                obs_snapshot.data = datatable[UT_mask]
                lcg_snapshot = lcg.LCG_metric(obs_snapshot,method=method,tavg=tavg,scan_avg=scan_avg,dummy_circ=dummy_circ,dummy_circ_res=dummy_circ_res,plot_solution=plot_solution,niter=niter,specify_x0=specify_x0)
                lcgs[itime] = lcg_snapshot
        lcg_mean = np.mean(lcgs)
        lcg_median = np.median(lcgs)
        lcg_max = np.max(lcgs)
        lcg_out = {'lcg_mean': lcg_mean, 'lcg_median': lcg_median, 'lcg_max': lcg_max}
        
        return lcg_out

    # parse and execute the largest circular gap requests
    def calc_lcg(self,obs,method='analytic',tavg=None,scan_avg=False,dummy_circ=True,dummy_circ_res=None,plot_solution=False,niter=1000,specify_x0=None):
        
        # if computing only on a full observation
        if (self.settings['ff_type'] == 'fullobs'):
            lcg_out = lcg.LCG_metric(obs,method=method,tavg=tavg,scan_avg=scan_avg,dummy_circ=dummy_circ,dummy_circ_res=dummy_circ_res,plot_solution=plot_solution,niter=niter,specify_x0=specify_x0)
        
        # if computing on individual snapshots
        elif (self.settings['ff_type'] == 'snapshot'):
            lcg_fullobs = lcg.LCG_metric(obs,method=method,tavg=tavg,scan_avg=scan_avg,dummy_circ=dummy_circ,dummy_circ_res=dummy_circ_res,plot_solution=plot_solution,niter=niter,specify_x0=specify_x0)
            lcg_snapshot = self.calc_lcg_continuous(obs,snapshot_interval=self.settings['t_rest'],method=method,tavg=tavg,scan_avg=scan_avg,dummy_circ=dummy_circ,dummy_circ_res=dummy_circ_res,plot_solution=plot_solution,niter=niter,specify_x0=specify_x0)
            lcg_out = {'lcg_fullobs': lcg_fullobs, 'lcg_mean': lcg_snapshot['lcg_mean'],'lcg_median': lcg_snapshot['lcg_median'], 'lcg_max': lcg_snapshot['lcg_max']}
        
        return lcg_out

    # compute the point source sensitivity on snapshots
    def calc_pss_continuous(self,obs):
        
        # observation info
        times_obs = obs.data['time']
        datatable = obs.data.copy()
        
        # make a small blank copy to use for snapshots
        obs_blank = obs.copy()
        obs_blank.data = None
        
        # compute PSS in snapshots
        times_temp = np.copy(self.t_seg_times)
        times = np.concatenate((times_temp,[2.0*times_temp[-1]-times_temp[-2]]))
        psss = np.zeros(len(times)-1)
        for itime in range(len(times)-1):
            UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
            if (UT_mask.sum() > 0):
                obs_snapshot = obs_blank.copy()
                obs_snapshot.data = datatable[UT_mask]
                pss_snapshot = 1.0/np.sqrt(np.sum(1.0/obs_snapshot.data['sigma']**2.0))
                psss[itime] = pss_snapshot
        pss_mean = np.mean(psss)
        pss_median = np.median(psss)
        pss_max = np.max(psss)
        pss_out = {'pss_mean': pss_mean, 'pss_median': pss_median, 'pss_max': pss_max}
        
        return pss_out

    # parse and execute the point source sensitivity requests
    def calc_pss(self,obs):
        
        # if computing only on a full observation
        if (self.settings['ff_type'] == 'fullobs'):
            pss_out = 1.0/np.sqrt(np.sum(1.0/obs.data['sigma']**2.0))
        
        # if computing on individual snapshots
        elif (self.settings['ff_type'] == 'snapshot'):
            pss_fullobs = 1.0/np.sqrt(np.sum(1.0/obs.data['sigma']**2.0))
            pss_snapshot = self.calc_pss_continuous(obs)
            pss_out = {'pss_fullobs': pss_fullobs, 'pss_mean': pss_snapshot['pss_mean'],'pss_median': pss_snapshot['pss_median'], 'pss_max': pss_snapshot['pss_max']}
        
        return pss_out

    def angular_resolution(self,obs,weighting='natural',robust=0.0):
        # output minor and major axes are in uas
        # output PA is measured from the major axis, in degrees East of North
        
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
        u2 = np.average(u**2.0,weights=weights)
        v2 = np.average(v**2.0,weights=weights)
        uv = np.average(u*v,weights=weights)

        # compute (u,v) eigenvalues
        minor_uv = 0.5*(u2 + v2 - np.sqrt((u2**2.0) - (2.0*u2*v2) + (4.0*(uv**2.0)) + (v2**2.0)))
        major_uv = 0.5*(u2 + v2 + np.sqrt((u2**2.0) - (2.0*u2*v2) + (4.0*(uv**2.0)) + (v2**2.0)))

        # compute (u,v) eigenvectors
        vec_major_uv = np.array([(-(1.0/(2.0*uv))*(v2 - u2 - np.sqrt((u2**2.0) - (2.0*u2*v2) + (4.0*(uv**2.0)) + (v2**2.0)))),1.0])
        vec_major_uv /= np.sqrt((vec_major_uv[0]**2.0) + (vec_major_uv[1]**2.0))

        # compute (u,v) position angle
        theta_uv = np.arctan2(vec_major_uv[0],vec_major_uv[1])

        # convert to image-domain
        minor = (0.5/np.sqrt(major_uv)) / eh.RADPERUAS
        major = (0.5/np.sqrt(minor_uv)) / eh.RADPERUAS
        theta = (180.0/np.pi)*(theta_uv + (np.pi/2.0))

        return minor, major, theta

    # compute the angular resolution on snapshots
    def calc_ar_continuous(self,obs,artype='mean',weighting='natural',robust=0.0):
        
        # observation info
        times_obs = obs.data['time']
        datatable = obs.data.copy()
        
        # make a small blank copy to use for snapshots
        obs_blank = obs.copy()
        obs_blank.data = None
        
        # compute angular resolution in snapshots
        times_temp = np.copy(self.t_seg_times)
        times = np.concatenate((times_temp,[2.0*times_temp[-1]-times_temp[-2]]))
        ars = np.zeros(len(times)-1)
        for itime in range(len(times)-1):
            UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
            if (UT_mask.sum() > 0):
                obs_snapshot = obs_blank.copy()
                obs_snapshot.data = datatable[UT_mask]
                ar_snapshot = self.angular_resolution(obs_snapshot,weighting=weighting,robust=robust)
                if (artype == 'minor'):
                    ars[itime] = ar_snapshot[0]
                if (artype == 'major'):
                    ars[itime] = ar_snapshot[1]
                if ((artype == 'PA') | (artype == 'angle')):
                    ars[itime] = ar_snapshot[2]
                if (artype == 'mean'):
                    ars[itime] = np.sqrt(ar_snapshot[0]*ar_snapshot[1])
        ar_mean = np.mean(ars)
        ar_median = np.median(ars)
        ar_max = np.max(ars)
        ar_out = {'ar_mean': ar_mean, 'ar_median': ar_median, 'ar_max': ar_max}
        
        return ar_out

    # parse and execute the angular resolution requests
    def calc_ar(self,obs,artype='mean',weighting='natural',robust=0.0):

        # if computing only on a full observation
        if (self.settings['ff_type'] == 'fullobs'):
            ar_list = self.angular_resolution(obs,weighting=weighting,robust=robust)
            if (artype == 'minor'):
                ar_out = ar_list[0]
            if (artype == 'major'):
                ar_out = ar_list[1]
            if ((artype == 'PA') | (artype == 'angle')):
                ar_out = ar_list[2]
            if (artype == 'mean'):
                ar_out = np.sqrt(ar_list[0]*ar_list[1])
        
        # if computing on individual snapshots
        elif (self.settings['ff_type'] == 'snapshot'):
            ar_fullobs_list = self.angular_resolution(obs,weighting=weighting,robust=robust)
            if (artype == 'minor'):
                ar_fullobs = ar_fullobs_list[0]
            if (artype == 'major'):
                ar_fullobs = ar_fullobs_list[1]
            if ((artype == 'PA') | (artype == 'angle')):
                ar_fullobs = ar_fullobs_list[2]
            if (artype == 'mean'):
                ar_fullobs = np.sqrt(ar_fullobs_list[0]*ar_fullobs_list[1])
            ar_snapshot = self.calc_ar_continuous(obs,artype=artype,weighting=weighting,robust=robust)
            ar_out = {'ar_fullobs': ar_fullobs, 'ar_mean': ar_snapshot['ar_mean'],'ar_median': ar_snapshot['ar_median'], 'ar_max': ar_snapshot['ar_max']}
        
        return ar_out

    def converge(self,imgr,res,major=3, blur_frac=1.0):
        for repeat in range(major):
            init = imgr.out_last().blur_circ(blur_frac*res)
            imgr.init_next = init
            imgr.make_image_I(show_updates=False)
        return imgr

    def run_imager(self,obs):
        if self.ims[0].source == 'M87':
            zbl = self.settings['zbl_flux_230_M87']
        elif self.ims[0].source == 'SgrA':
            zbl = self.settings['zbl_flux_230_SgrA']

        prior_fwhm = self.settings['prior_fwhm']*eh.RADPERUAS
        fov = self.settings['fov']*eh.RADPERUAS
        npix = self.settings['npix']

        sys_noise  = self.settings['sys_noise']               # fractional systematic noise
        reg_term  = {'simple': self.settings['reg_term_simple'],    # Maximum-Entropy
            'tv': self.settings['reg_term_tv'],    # Total Variation
            'tv2': self.settings['reg_term_tv2'],    # Total Squared Variation
            'l1': self.settings['reg_term_l1'],    # L1 sparsity prior
            'flux': self.settings['reg_term_flux']}    # compact flux constraint
        data_term = {'amp': self.settings['data_term_amp'],    # visibility amplitudes
            'cphase': self.settings['data_term_cphase'],    # closure phases
            'logcamp': self.settings['data_term_logcamp']}    # log closure amplitudes
        ttype = self.settings['ttype']              # Type of Fourier transform ('direct', 'nfft', or 'fast')
        maxit = self.settings['maxit']                 # Maximum number of convergence iterations for imaging
        stop = self.settings['stop']               # Imager stopping criterion
        gain_tol = self.settings['gain_tol']          # Asymmetric gain tolerance for self-cal; we expect larger values # for unaccounted sensitivity loss # than for unaccounted sensitivity improvement
        reverse_taper_uas = self.settings['reverse_taper_uas']         # Finest resolution of reconstructed features


        SEFD_error_budget = self.SEFD_error_budget_dict
        # Start with the SEFD noise (but need sqrt)
        # then rescale to ensure that final results respect the stated error budget
        systematic_noise = SEFD_error_budget.copy()
        for key in systematic_noise.keys():
            systematic_noise[key] = ((1.0+systematic_noise[key])**0.5 - 1.0) * 0.25
        # Extra noise added for the LMT, which has much more variability than the a-priori error budget
        if 'LMT' in systematic_noise.keys():
            systematic_noise['LMT'] += 0.15
        obs_sc = obs.copy() # From here on out, don't change obs. Use obs_sc to track gain changes
        res = obs_sc.res()  # The nominal array resolution: 1/(longest baseline)
        gaussprior = eh.image.make_square(obs_sc, npix, fov)
        gaussprior = gaussprior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
        gaussprior = gaussprior.add_gauss(zbl*1e-3, (prior_fwhm, prior_fwhm, 0, prior_fwhm, prior_fwhm))
        if reverse_taper_uas > 0:
            obs_sc = obs_sc.reverse_taper(reverse_taper_uas*eh.RADPERUAS)
        obs_sc = obs_sc.add_fractional_noise(sys_noise)
        imgr = eh.imager.Imager(obs_sc, gaussprior, prior_im=gaussprior,
            flux=zbl, data_term=data_term, maxit=maxit,
            norm_reg=True, systematic_noise=systematic_noise,
            reg_term=reg_term, ttype=ttype, stop=stop)
        imgr.make_image_I(show_updates=False)
        imgr = self.converge(imgr,res)

        if self.settings['SC_bool']:
            obs_sc = eh.selfcal(obs_sc, imgr.out_last(),method='amp',ttype=ttype,gain_tol=gain_tol,processes=1)
            obs_sc = eh.selfcal(obs_sc, imgr.out_last(),method='phase',ttype=ttype,gain_tol=gain_tol,processes=1)
            obs_sc = eh.selfcal(obs_sc, imgr.out_last(),method='both',ttype=ttype,gain_tol=gain_tol,processes=1)
        else:
            pass

        # Second  Round of Imaging
        #-------------------------
        # Blur the previous reconstruction to the intrinsic resolution of ~25 uas
        init = imgr.out_last().blur_circ(res)
        data_term_intermediate = {'vis':imgr.dat_terms_last()['amp']*10,
            'cphase':imgr.dat_terms_last()['cphase']*10,
            'logcamp':imgr.dat_terms_last()['logcamp']*10} # Increase the weights on the data terms and reinitialize imaging

        imgr = eh.imager.Imager(obs_sc, init, prior_im=gaussprior, flux=zbl,
            data_term=data_term_intermediate, maxit=maxit, norm_reg=True,
            systematic_noise=systematic_noise, reg_term = reg_term, ttype=ttype,
            stop=stop)
        imgr.make_image_I(show_updates=False)

        if self.settings['SC_bool']:
            obs_sc = eh.selfcal(obs_sc, imgr.out_last(),method='amp',ttype=ttype,gain_tol=gain_tol,processes=1)
            obs_sc = eh.selfcal(obs_sc, imgr.out_last(),method='phase',ttype=ttype,gain_tol=gain_tol,processes=1)
            obs_sc = eh.selfcal(obs_sc, imgr.out_last(),method='both',ttype=ttype,gain_tol=gain_tol,processes=1)
        else:
            pass

        # Third and Fourth Rounds of Imaging
        #-----------------------------------
        data_term_final = {'vis':imgr.dat_terms_last()['vis']*5,
            'cphase':imgr.dat_terms_last()['cphase']*2,
            'logcamp':imgr.dat_terms_last()['logcamp']*2} # Increase the data weights before imaging again

        # Repeat imaging twice
        for repeat_selfcal in range(2):
            init = imgr.out_last().blur_circ(res) # Blur the previous reconstruction to the intrinsic resolution of ~25 uas
            imgr = eh.imager.Imager(obs_sc, init, prior_im=gaussprior, flux=zbl,
                data_term=data_term_final, maxit=maxit, norm_reg=True,
                systematic_noise=0.01, reg_term=reg_term, ttype=ttype,
                stop=stop)          # Reinitialize imaging now using complex visibilities; common systematic noise
            imgr.make_image_I(show_updates=False)
            imgr = self.converge(imgr,res)

            # Self-calibrate
            if self.settings['SC_bool']:
                obs_sc = eh.selfcal(obs_sc, imgr.out_last(),method='amp',ttype=ttype,gain_tol=gain_tol,processes=1)
                obs_sc = eh.selfcal(obs_sc, imgr.out_last(),method='phase',ttype=ttype,gain_tol=gain_tol,processes=1)
                obs_sc = eh.selfcal(obs_sc, imgr.out_last(),method='both',ttype=ttype,gain_tol=gain_tol,processes=1)
            else:
                pass

        im_out = imgr.out_last().copy()
        return(obs_sc,im_out)

    # make a plot of (u,v)-coverage
    def plot_uv(self,obs,file_tag):

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
        ax.set_xlim(15,-15)
        ax.set_ylim(-15,15)
        plt.savefig(file_tag+'_uv.pdf', bbox_inches='tight')
        plt.close()

    # make a plot of visibility amplitude vs (u,v)-distance
    def plot_amp(self,obs,file_tag):
        
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
        ax.set_xlim(0,15)
        ax.set_ylim(0.01,10)
        plt.savefig(file_tag+'_amp.pdf', bbox_inches='tight')
        plt.close()

    # make a plot of visibility phase vs (u,v)-distance
    def plot_phase(self,obs,file_tag):

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
        ax.set_xlim(0,15)
        ax.set_ylim(-180,180)
        plt.savefig(file_tag+'_phase.pdf', bbox_inches='tight')
        plt.close()

    # make a plot of SNR vs (u,v)-distance
    def plot_snr(self,obs,file_tag):

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
        ax.set_xlim(0,15)
        plt.savefig(file_tag+'_snr.pdf', bbox_inches='tight')
        plt.close()

    def save_img_out(self,im_out,file_tag):
        # If an inverse taper was used, restore the final image
        # to be consistent with the original data
        reverse_taper_uas = self.settings['reverse_taper_uas']
        if reverse_taper_uas > 0.0:
            im_out = im_out.blur_circ(reverse_taper_uas*eh.RADPERUAS)

        # Save the final image
        im_out.save_fits(file_tag+ '.fits')
        im_out.display(cbar_unit=['Tb'],label_type='scale',export_pdf=file_tag + '.jpg')
        im_out.display(scale='log', cbar_unit=['Tb'],label_type='scale',export_pdf=file_tag + '_log.jpg')
        plt.close('all')

    def save_uvfits(self,obs,file_tag):
        obs.save_uvfits(file_tag + '.uvfits')

    def calc_fill_frac_snapshot(self,obs,longest_BL=None,start_time=0.0,end_time=24.0,snapshot_interval=600.0,fov=100.0,fillpix=10):
        """
        Calculate the (u,v)-filling fraction metric on snapshots
        
        obs : input ehtim obsdata object
        longest_BL : length of the bounding baseline, dimensionless
        start_time : starting time of first snapshot, in hours
        end_time : ending time of last snapshot, in hours
        snapshot_interval : length of a single snapshot, in seconds
        fov : FOV to consider when computing ff, in uas
        fillpix : number of resolution elements across a convolving kernel in ff
        
        returns : the segmentation times and filling fraction values for each snapshot

        """
        
        if (longest_BL == None):
            longest_BL = D_Earth/(c/obs.rf)

        # observation info
        times_obs = obs.data['time']
        datatable = obs.data.copy()

        # make a small blank copy to use for snapshots
        obs_blank = obs.copy()
        obs_blank.data = None
        
        # compute fill fractions
        times_temp = np.arange(start_time,end_time,snapshot_interval/3600.0)
        times = np.concatenate((times_temp,[2.0*times_temp[-1]-times_temp[-2]]))
        fills = np.zeros(len(times)-1)
        for itime in range(len(times)-1):
            UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
            if (UT_mask.sum() > 0):
                obs_snapshot = obs_blank.copy()
                obs_snapshot.data = datatable[UT_mask]
                fill_snapshot = ff.obs_fill(obs_snapshot,fov=fov,longest=longest_BL,N=fillpix)
                fills[itime] = fill_snapshot
        
        return times, fills

    def calc_lcg_snapshot(self,obs,start_time=0.0,end_time=24.0,snapshot_interval=600.0,dummy_circ_res=None):
        """
        Calculate the largest circular gap metric on snapshots
        
        obs : input ehtim obsdata object
        start_time : starting time of first snapshot, in hours
        end_time : ending time of last snapshot, in hours
        snapshot_interval : length of a single snapshot, in seconds
        dummy_circ_res : resolution of "dummy circle", in uas

        returns : the segmentation times and LCG values for each snapshot

        """
        
        if (dummy_circ_res == None):
            dummy_circ_res = ((c/obs.rf) / D_Earth) / eh.RADPERUAS
        
        # observation info
        times_obs = obs.data['time']
        datatable = obs.data.copy()
        
        # make a small blank copy to use for snapshots
        obs_blank = obs.copy()
        obs_blank.data = None
        
        # compute LCG in snapshots
        times_temp = np.arange(start_time,end_time,snapshot_interval/3600.0)
        times = np.concatenate((times_temp,[2.0*times_temp[-1]-times_temp[-2]]))
        lcgs = np.zeros(len(times)-1)
        for itime in range(len(times)-1):
            UT_mask = ((times_obs >= times[itime]) & (times_obs <= times[itime+1]))
            if (UT_mask.sum() > 0):
                obs_snapshot = obs_blank.copy()
                obs_snapshot.data = datatable[UT_mask]
                lcg_snapshot = lcg.LCG_metric(obs_snapshot,method='analytic',tavg=None,scan_avg=False,dummy_circ=True,dummy_circ_res=dummy_circ_res,plot_solution=False,niter=1000,specify_x0=None)
                lcgs[itime] = lcg_snapshot

        return times, lcgs

    def plot_snapshot(self,obs,metric,file_tag,start_time=0.0,end_time=24.0,snapshot_interval=600.0,fov=1000.0,fillpix=10,timetype='GMST'):
        """
        Plot the chosen metric on snapshots
        
        obs : input ehtim obsdata object
        metric : selected metric to plot, either 'ff' or 'lcg'
        start_time : starting time of first snapshot, in hours
        end_time : ending time of last snapshot, in hours
        snapshot_interval : length of a single snapshot, in seconds
        fov : FOV to consider when computing ff, in uas
        fillpix : number of resolution elements across a convolving kernel in ff
        
        returns : a figure containin the plot

        """

        # ensure the observation has the correct timetype
        obs = obs.switch_timetype(timetype_out=timetype)

        # compute metric
        if (metric == 'ff'):
            longest_BL = D_Earth/(c/obs.rf)
            times, y = self.calc_fill_frac_snapshot(obs,longest_BL,start_time=start_time,end_time=end_time,snapshot_interval=snapshot_interval,fov=fov,fillpix=fillpix)
        if (metric == 'lcg'):
            dummy_circ_res = ((c/obs.rf) / D_Earth) / eh.RADPERUAS
            times, y = self.calc_lcg_snapshot(obs,start_time=start_time,end_time=end_time,snapshot_interval=snapshot_interval,dummy_circ_res=dummy_circ_res)

        # get snapshot times
        t = 0.5*(times[1:] + times[0:-1])

        # initialize figure
        fig = plt.figure(figsize=(4,12))
        ax1 = fig.add_axes([0.1,0.8,0.8,0.35])
        ax2 = fig.add_axes([0.1,0.1,0.8,0.7])

        # plot metric
        ax1.plot(t,y,'b-')

        # plot stations
        t1 = obs.data['t1']
        t2 = obs.data['t2']
        stations = np.sort(np.unique(np.concatenate((t1,t2))))
        time = obs.data['time']
        yticks2 = []
        for istat, station in enumerate(stations):
            
            # plot indicator line
            ax2.plot([start_time,end_time],[istat,istat],'k--',alpha=0.2,linewidth=0.5)

            # get the timestamps for this station
            index = ((t1 == station) | (t2 == station))
            timehere = np.sort(time[index])

            # check if there's a big gap (> 30 minutes)
            if (np.max(np.diff(timehere)) > 0.5):
                tmax = timehere[np.argmax(np.diff(timehere))]
                tmin = timehere[np.argmax(np.diff(timehere))+1]
                ax2.plot([start_time,tmax],[istat,istat],'k-',linewidth=2)
                ax2.plot([tmin,end_time],[istat,istat],'k-',linewidth=2)
            else:
                tmin = np.min(timehere)
                tmax = np.max(timehere)
                ax2.plot([tmin,tmax],[istat,istat],'k-',linewidth=2)

            yticks2.append(istat)

        # clean up plot
        ax1.set_xlim(start_time,end_time)
        ax2.set_xlim(start_time,end_time)
        ax1.set_ylim(0.0,1.1*np.max(y))
        ax2.set_ylim(np.min(yticks2)-1,np.max(yticks2)+1.5)

        ax1.tick_params(axis='x',which='both',direction='inout',bottom=True)
        ax2.tick_params(axis='x',which='both',direction='inout',bottom=True,top=True)

        if (metric == 'ff'):
            ax1.set_ylabel(r'$(u,v)$ filling fraction')
        if (metric == 'lcg'):
            ax1.set_ylabel('largest circular gap')
        
        if (timetype == 'UTC'):
            ax2.set_xlabel('UT (hr)')
        if (timetype == 'GMST'):
            ax2.set_xlabel('GMST (hr)')
        
        ax1.set_xticklabels([])
        ax2.set_yticks(yticks2)
        ax2.set_yticklabels(stations,fontsize=10)

        # Save the final image
        plt.savefig(file_tag + '_snapshot.png',dpi=300,bbox_inches='tight')
        plt.close('all')


    def insert_rests_and_scan_avg(self,obs_in):
        times_in_input = np.unique(obs_in.data['time'])
        if len(times_in_input) > 0:
            gap_starts = np.linspace(times_in_input[0],times_in_input[-1],int((times_in_input[-1]-times_in_input[0])/(self.settings['del_gap']/3600.)))
            gap_ends = gap_starts+(self.settings['dt_gap']/3600.)

            for aa,gap_start in enumerate(gap_starts):
                obs_in = obs_in.flag_UT_range(gap_start,gap_ends[aa],output='kept')
            obs_in.add_scans()
            obs_out = obs_in.avg_incoherent(self.settings['del_gap']-self.settings['dt_gap'],scan_avg=False).copy()
        else:
            obs_out = obs_in
        return(obs_out)
