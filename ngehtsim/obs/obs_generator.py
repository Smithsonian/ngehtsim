###################################################
# imports

import numpy as np
import ehtim as eh
import ngehtutil as ng
import scipy.stats as stats
from scipy.special import erf, erfinv
from collections import defaultdict
from astropy.time import Time
import ngEHTforecast.fisher as fp
import yaml
import glob
import time
import sys
import os

import ngehtsim.const_def as const
import ngehtsim.weather.weather as nw

###################################################
# class definition

class obs_generator(object):
    """
    Class that organizes information for generating synthetic observations.  Typically initialized by
    passing a settings file.

    Attributes:
      settings (dict): Dictionary of information about the observation generation setup
      settings_file (str): Path to the input settings file; if set to None, will use default settings.
                           Note that any settings specified by the settings keyword argument will override
                           the corresponding settings from the settings file.
      verbose (float): Set to >0 for more verbose output
      D_override_dict (dict): A dictionary of station names and diameters to override the internal defaults
      array_name (str): Name to get assigned to the ngehtutil array object
    """

    # initialize class instantiation
    def __init__(self, settings={}, settings_file=None, verbose=0, D_override_dict={}, array_name=None):

        self.settings = {}
        self.settings_file = settings_file
        self.verbosity = verbose
        self.D_override_dict = D_override_dict
        self.array_name = array_name

        # start with some default settings
        self.settings = const.default_settings

        # check if user wants to load settings from a passed file
        if settings_file is not None:
            self.load_yaml_settings()
            if self.verbosity > 0:
                print('========= Loaded settings from {0}'.format(settings_file))
        else:
            if self.verbosity > 0:
                print('========= Loaded default settings')

        # update the settings with any additional passed information
        self.settings.update(settings)

        # set absolute path to weather
        self.path_to_weather = os.path.abspath(const.path_to_weather)

        # check for issues, fix some easy ones, complain about the others
        if (self.settings['weather_freq'] is not None) & (self.settings['weather_freq'] not in ['86', '230', '345', '690']):
            raise ValueError('Input weather frequency needs to be one of 86, 230, 345, or 690.')
        if (self.settings['nbands'] < 1):
            self.settings['nbands'] = 1
            raise Warning('Input nbands must be at least 1; setting to 1.')
        if ((self.settings['nbands'] > 1) & (self.settings['rf_offset'] < self.settings['bandwidth'])):
            raise Exception('Input rf_offset must be greater than or equal to input bandwidth when nbands > 1.')
        if (self.path_to_weather[-1] != '/'):
            self.path_to_weather += '/'
        if self.array_name is None:
            if self.settings['array'] is not None:
                self.array_name = self.settings['array']

        # extract commonly-used settings
        self.model_file = self.settings['model_file']
        self.freq = float(self.settings['frequency'])*(1.0e9)
        self.nbands = self.settings['nbands']
        self.freq_offsets = (np.arange(float(self.nbands)) - np.mean(np.arange(float(self.nbands)))) * float(self.settings['rf_offset']) * (1.0e9)
        self.weather = self.settings['weather']

        # run initialization functions
        self.set_seed()
        self.get_sites()
        self.translate_sites()
        self.set_weather_freq()
        self.set_TR()
        self.set_coords()
        self.mjd = determine_mjd(self.settings['day'],self.settings['month'],self.settings['year'])
        self.array, self.arr = make_array(self.sites,self.settings['D_new'],D_override_dict=self.D_override_dict,array_name=self.array_name,freq=self.freq/(1.0e9))
        self.im = load_image(self.model_file,freq=self.freq,verbose=self.verbosity)
        self.tabulate_weather()
        self.telescope_properties()
        self.get_obs_times()

        # other settings
        self.obs_empty = None

    # load and store settings from file
    def load_yaml_settings(self):
        loader = yaml.SafeLoader
        with open(self.settings_file, 'r') as fi:
            self.settings.update(yaml.load(fi, Loader=loader))

    # set random number seed
    def set_seed(self):
        if self.settings['random_seed'] is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(self.settings['random_seed'])

    # generate the site list
    def get_sites(self):

        # initialize site list
        self.sites = list()

        # if a known array is specified, pull its sites and overrides
        if self.settings['array'] in const.known_arrays.keys():
            self.sites = const.known_arrays[self.settings['array']]
            override_dict_here = const.known_array_overrides[self.settings['array']]
            override_dict_here.update(self.D_override_dict)
            self.D_override_dict = override_dict_here

        # add in any additional sites
        else:
            if self.settings['sites'] is not None:
                self.sites += self.settings['sites']
            else:
                raise Exception('No known array or sites have been specified!')
        if self.settings['sites'] is not None:
            self.sites += self.settings['sites']

        # remove duplicates
        temp_sites = np.unique(np.array(self.sites))
        self.sites = list(temp_sites)

    # make sure all sites are known
    def translate_sites(self):
        for isite, site in enumerate(self.sites):
            if site in list(const.translation_dict.keys()):
                self.sites[isite] = const.translation_dict[site]
            else:
                if site not in ng.Station.get_list():
                    raise Exception(site+' is not a known station.')

    # determine the weather frequency to use
    def set_weather_freq(self):
        freq_options = np.array([86.0,230.0,345.0,690.0])
        weath_options = np.array(['86','230','345','690'])
        if self.settings['weather_freq'] is None:
            freqhere = self.freq / (1.0e9)
            self.weather_freq = weath_options[np.argmin(np.abs(freqhere - freq_options))]
        if self.verbosity > 0:
            print("************** Weather frequency set to " + str(self.weather_freq) + ' GHz.')

    # store receiver temperature
    def set_TR(self):
        self.T_R = const.T_R_dict[self.weather_freq]
        if self.verbosity > 0:
            print("************** Receiver temperature set to " + str(self.T_R) + ' K.')
    
    # set source coordinates
    def set_coords(self):

        # retrieve coordinates from source, if known
        if self.settings['source'] in const.known_sources.keys():
            self.RA = const.known_sources[self.settings['source']]['RA']
            self.DEC = const.known_sources[self.settings['source']]['DEC']
        else:
            if ((self.settings['RA'] is None) & (self.settings['DEC'] is None)):
                raise Exception('A known source and/or a set of (RA,DEC) coordinates must be specified.')

        # if coordinates are specified, use those instead
        if self.settings['RA'] is not None:
            self.RA = self.settings['RA']
        if self.settings['DEC'] is not None:
            self.DEC = self.settings['DEC']

    # extract the opacity and Tb information from weather tables
    def tabulate_weather(self):

        # initialize dictionaries
        tau_dict = defaultdict(dict)
        Tatm_dict = defaultdict(dict)
        Tb_dict = defaultdict(dict)

        # extract month number
        monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
        monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        monthnum = monthnums[monthnams == self.settings['month']][0]

        # get a day and year for the weather parameters
        if (self.weather == 'random'):
            # pick a random past date from which to pull the weather
            self.randyear = np.random.randint(const.year_min,const.year_max+1)
            if (self.settings['month'] == 'Feb'):
                self.randday = np.random.randint(1,29)
            elif (self.settings['month'] in ['Apr','Jun','Sep','Nov']):
                self.randday = np.random.randint(1,31)
            else:
                self.randday = np.random.randint(1,32)
        elif (self.weather == 'exact'):
            # use the specified date
            self.randyear = int(self.settings['year'])
            self.randday = int(self.settings['day'])

        # read in the weather info and store it
        for isite, site in enumerate(self.sites):

            # determine which table to read
            pathhere = self.path_to_weather
            pathhere += site + '/'
            pathhere += monthnum + self.settings['month'] + '/'
            pathhere += 'mean_SEFD_info_' + self.weather_freq + '.csv'
            
            # read in the table
            year, monthdum, day, tau, Tb = np.loadtxt(pathhere,skiprows=7,unpack=True,delimiter=',')

            if ((self.weather == 'random') | (self.weather == 'exact')):
                # pull out the info for the selected date
                index = ((year == self.randyear) & (day == self.randday))
                if (np.array(index).sum() == 0):
                    raise Exception('No weather on file for the selected date!')
                tau_here = tau[index][0]
                Tb_here = Tb[index][0]
            elif (self.weather == 'typical'):
                tau_here = np.median(tau)
                Tb_here = np.median(Tb)
            elif (self.weather == 'good'):
                tau_here = np.percentile(tau,15.87)
                Tb_here = np.percentile(Tb,15.87)
            elif (self.weather == 'poor'):
                tau_here = np.percentile(tau,84.13)
                Tb_here = np.percentile(Tb,84.13)

            # divide out the opacity term to get the actual atmospheric temperature
            Tatm = (Tb_here - (const.T_CMB_AM*np.exp(-tau_here))) / (1.0 - np.exp(-tau_here))

            # store the info in the dictionaries
            tau_dict[site] = tau_here
            Tatm_dict[site] = Tatm
            Tb_dict[site] = Tb_here

        # store the dictionaries
        self.tau_dict = tau_dict
        self.Tatm_dict = Tatm_dict
        self.Tb_dict = Tb_dict

    # generate dictionaries of telescope properties
    def telescope_properties(self):

        # aperture efficiency of new dishes
        eta_new = eta_dish(self.freq,const.sigma_surface,const.focus_offset)

        D_dict = {}
        eta_dict = {}
        for station in self.array.stations():
            D_dict[station.name] = station.diameter()
            eta_dict[station.name] = eta_new

        self.D_dict = D_dict
        self.eta_dict = eta_dict

    # segment the observation into timestamps
    def get_obs_times(self):
        t_first = self.settings['t_start']
        N_obs = int(np.ceil(self.settings['dt']/(self.settings['t_rest']/3600.)))
        t_last = t_first+float(N_obs-1)*(self.settings['t_rest']/3600.)
        self.t_seg_times = np.linspace(t_first,t_last,N_obs)
        if self.verbosity > 0:
            print("========= Number of timestamps: {0}".format(N_obs))
            print("========= Beginning of first integration: {0}".format(t_first))
            print("========= Beginning of last integration: {0}".format(t_last))
            print('************** Scan start times: {0}'.format(self.t_seg_times))

    # generate a raw observation
    def observe(self,input_model,obsfreq,addnoise=True,addgains=True,gainamp=0.04,opacitycal=True,p=None):
        """
        Generate a raw single-band observation that folds in weather-based opacity and sensitivity effects.

        Args:
          input_model (ehtim.image.Image): input source model; can be ehtim.image.Image, ehtim.movie.Movie,
                                           ehtim.model.Model, or ngEHTforecast.fisher.fisher_forecast.FisherForecast
          obsfreq (float): observing frequency, in Hz
          addnoise (bool): flag for whether or not to add thermal noise to the visibilities
          addgains (bool): flag for whether or not to add station gain corruptions
          gainamp (float): standard deviation of amplitude log-gains
          opacitycal (bool): flag for whether or not to assume that atmospheric opacity is assumed to be calibrated out
          p (numpy.ndarray): list of parameters for an input ngEHTforecast.fisher.fisher_forecast.FisherForecast object
        
        Returns:
          (ehtim.obsdata.Obsdata): eht-imaging Obsdata object containing the generated observation
        """

        # generate an empty obsdata object
        if self.obs_empty is None:
            self.obs_empty = self.arr.obsdata(self.RA,
                                              self.DEC,
                                              obsfreq,
                                              (1.0e9)*float(self.settings['bandwidth']),
                                              self.settings['t_int'],
                                              self.settings['t_rest'],
                                              self.settings['t_start'],
                                              self.settings['t_start'] + self.settings['dt'],
                                              mjd = self.mjd,
                                              polrep = 'stokes',
                                              tau = 0.0,
                                              timetype = 'UTC',
                                              elevmin = const.el_min,
                                              elevmax = const.el_max,
                                              fix_theta_GMST = False)

        # observe the source
        if isinstance(input_model, eh.image.Image):
            input_model.ra = self.RA
            input_model.dec = self.DEC
            input_model.mjd = self.mjd
            input_model.source = self.settings['source']
            input_model.rf = obsfreq
            obs = input_model.observe_same_nonoise(self.obs_empty,ttype=self.settings['ttype'],fft_pad_factor=self.settings['fft_pad_factor'])
        elif isinstance(input_model, eh.movie.Movie):
            input_model.ra = self.RA
            input_model.dec = self.DEC
            input_model.mjd = self.mjd
            input_model.source = self.settings['source']
            input_model.rf = obsfreq
            obs = input_model.observe_same_nonoise(self.obs_empty,ttype=self.settings['ttype'],fft_pad_factor=self.settings['fft_pad_factor'],repeat=True)
        elif isinstance(input_model, eh.model.Model):
            input_model.ra = self.RA
            input_model.dec = self.DEC
            input_model.mjd = self.mjd
            input_model.source = self.settings['source']
            input_model.rf = obsfreq
            obs = input_model.observe_same_nonoise(self.obs_empty)
        elif isinstance(input_model, fp.FisherForecast):
            if p is None:
                raise Exception('When observing an ngEHTforecast model, the parameter vector keyword argument p must be specified!')
            obs = self.obs_empty.copy()
            obs.source = self.settings['source']
            if (input_model.stokes == 'I'):
                Ivis = input_model.visibilities(obs,p,verbosity=self.verbosity)
                obs.data['vis'] = Ivis
            else:
                obs.switch_polrep(polrep_out='circ')
                RRvis, LLvis, RLvis, LRvis = input_model.visibilities(obs,p,verbosity=self.verbosity)
                obs.data['rrvis'] = RRvis
                obs.data['llvis'] = LLvis
                obs.data['rlvis'] = RLvis
                obs.data['lrvis'] = LRvis

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
        if addnoise:
            obs.data['rrvis'] += sigma*(np.random.normal(0.0,1.0,len(obs.data['rrsigma'])) + ((1.0j)*np.random.normal(0.0,1.0,len(obs.data['rrsigma']))))
            obs.data['llvis'] += sigma*(np.random.normal(0.0,1.0,len(obs.data['llsigma'])) + ((1.0j)*np.random.normal(0.0,1.0,len(obs.data['llsigma']))))
            obs.data['rlvis'] += sigma*(np.random.normal(0.0,1.0,len(obs.data['rlsigma'])) + ((1.0j)*np.random.normal(0.0,1.0,len(obs.data['rlsigma']))))
            obs.data['lrvis'] += sigma*(np.random.normal(0.0,1.0,len(obs.data['lrsigma'])) + ((1.0j)*np.random.normal(0.0,1.0,len(obs.data['lrsigma']))))

        # restore Stokes polrep
        obs = obs.switch_polrep(polrep_out='stokes')

        return obs

    # generate observation
    def make_obs(self,input_model=None,addnoise=True,addgains=True,gainamp=0.04,opacitycal=True,p=None):
        """
        Generate an observation (possibly multi-band) that folds in weather-based opacity effects
        and applies a specified SNR thresholding scheme to mimic fringe-finding.
        
        Args:
          input_model (ehtim.image.Image): input source model; can be ehtim.image.Image, ehtim.movie.Movie,
                                           ehtim.model.Model, or ngEHTforecast.fisher.fisher_forecast.FisherForecast
          addnoise (bool): flag for whether or not to add thermal noise to the visibilities
          addgains (bool): flag for whether or not to add station gain corruptions
          gainamp (float): standard deviation of amplitude log-gains
          opacitycal (bool): flag for whether or not to assume that atmospheric opacity is assumed to be calibrated out
          p (numpy.ndarray): list of parameters for an input ngEHTforecast.fisher.fisher_forecast.FisherForecast object

        Returns:
          (ehtim.obsdata.Obsdata): eht-imaging Obsdata object containing the generated observation
        """

        # determine SNR thresholding scheme and values
        snr_algo, snr_args = self.settings['SNR_cutoff']

        # retrieve stored input_model if it has been set to None
        if input_model is None:
            input_model = self.im
            if self.im is None:
                raise Exception('If there is no input model specified in the settings, then make_obs must specify one!')
            else:
                if self.verbosity > 0:
                    print('No input model passed to make_obs; using the model provided in the settings.')

        # loop through the bands
        for i_band in range(self.nbands):

            # adjust the observing frequency to account for the band offset
            adjusted_frequency = self.freq + self.freq_offsets[i_band]

            # generate raw observation for this band
            obs_seg = self.observe(input_model,adjusted_frequency,addnoise=addnoise,addgains=addgains,gainamp=gainamp,opacitycal=opacitycal,p=p)

            # apply naive SNR thresholding
            if (snr_algo == 'naive'):
                obs_seg = obs_seg.flag_low_snr(snr_cut=snr_args,output='kept')

            # apply a proxy for fringe-finding in HOPS
            elif (snr_algo == 'fringegroups'):

                # parse SNR_cutoff arguments
                snr_ref = snr_args[0]
                tint_ref = snr_args[1]

                obs_seg = fringegroups(obs_seg,snr_ref,tint_ref)

            # apply an FPT proxy for SNR thresholding
            elif (snr_algo == 'fpt'):

                # parse SNR_cutoff arguments
                snr_ref = snr_args[0]
                tint_ref = snr_args[1]
                freq_ref = snr_args[2]
                model_path_ref = snr_args[3]

                obs_seg = FPT(self,obs_seg,snr_ref,tint_ref,freq_ref,model_path_ref,obsfreq=adjusted_frequency,addnoise=addnoise,addgains=addgains,gainamp=gainamp,opacitycal=opacitycal,p=p)

            # unrecognized SNR thresholding scheme
            else:
                raise ValueError('unknown algorithm for SNR_cutoff')

            # append this segment to the obs list
            if i_band == 0:
                obs = obs_seg.copy()
            else:
                obs_seg.data['time'] += i_band*0.00001
                obs.data = np.concatenate([obs.data,obs_seg.data])

        # drop any sites randomly deemed to be technically unready
        sites_in_obs = obs.tarr['site']
        sites_to_drop = get_unready_sites(sites_in_obs, self.settings['tech_readiness'])
        if len(sites_to_drop) > 0:
            obs = obs.flag_sites(sites_to_drop)
            if self.verbosity > 0:
                print("Dropping {0} due to technical (un)readiness.".format(sites_to_drop))

        return obs

    def export_SYMBA(self,output_filename='obsgen.antennas',t_coh=10.0,RMS_point=1.0,PB_model='gaussian'):
        """
        Export a SYMBA-compatible .antennas file from the obs_generator object.

        Args:
          output_filename (str): name of .antennas file to save
          t_coh (float): default coherence time, in seconds
          RMS_point (float): default RMS pointing uncertainty, in arcseconds
          PB_model (str): primary beam model to use; only option right now is 'gaussian'

        Returns:
          SYMBA-compatible .antennas file containing the observation information
        """

        # make a list of station names in the correct order for the ngehtutils array object
        stationnames = list()
        for stat in self.array.stations():
            stationnames.append(stat.name)
        stationnames = np.array(stationnames)

        with open(output_filename,'w') as outfile:

            # add file header 
            header = 'station'.ljust(9)
            header += 's_rx[Jy]'.ljust(11)
            header += 'pwv[mm]'.ljust(9)
            header += 'gpress[mb]'.ljust(12)
            header += 'gtemp[K]'.ljust(10)
            header += 'c_time[sec]'.ljust(13)
            header += 'ptg_rms[arcsec]'.ljust(17)
            header += 'PB_FWHM230[arcsec]'.ljust(20)
            header += 'PB_model'.ljust(12)
            header += 'ap_eff'.ljust(9)
            header += 'gainR_mean'.ljust(11)
            header += 'gainR_std'.ljust(12)
            header += 'gainL_mean'.ljust(11)
            header += 'gainL_std'.ljust(12)
            header += 'leakR_mean'.ljust(12)
            header += 'leakR_std'.ljust(12)
            header += 'leakL_mean'.ljust(12)
            header += 'leakL_std'.ljust(12)
            header += 'feed_angle[degree]'.ljust(20)
            header += 'mount'.ljust(18)
            header += 'dish_diameter'.ljust(17)
            header += 'xzy_position_m' + '\n'
            outfile.write(header)

            for site in self.sites:

                # initialize empty string
                strhere = ''

                # add station name as a two-letter code
                strhere += const.two_letter_station_codes[site].ljust(9)

                # add receiver SEFD, in Jy
                SEFD_R = (2.0*const.k*self.T_R)/((np.pi/4.0)*self.eta_dict[site]*(self.D_dict[site])**2)
                strhere += str(np.round(SEFD_R,2)).ljust(11)

                # add PWV, in mm
                if ((self.weather == 'exact') | (self.weather == 'random')):
                    PWV = nw.PWV(site, form='exact', month=self.settings['month'], day=self.randday, year=self.randyear)
                elif (self.weather == 'typical'):
                    PWV = nw.PWV(site, form='median', month=self.settings['month'])
                elif (self.weather == 'good'):
                    PWV = nw.PWV(site, form='good', month=self.settings['month'])
                elif (self.weather == 'bad'):
                    PWV = nw.PWV(site, form='bad', month=self.settings['month'])
                strhere += str(np.round(PWV,4)).ljust(9)

                # add surface pressure, in mbar
                if ((self.weather == 'exact') | (self.weather == 'random')):
                    pres = nw.pressure(site, form='exact', month=self.settings['month'], day=self.randday, year=self.randyear)
                elif (self.weather == 'typical'):
                    pres = nw.pressure(site, form='median', month=self.settings['month'])
                elif (self.weather == 'good'):
                    pres = nw.pressure(site, form='good', month=self.settings['month'])
                elif (self.weather == 'bad'):
                    pres = nw.pressure(site, form='bad', month=self.settings['month'])
                strhere += str(np.round(pres,2)).ljust(12)

                # add surface temperature, in K
                if ((self.weather == 'exact') | (self.weather == 'random')):
                    temp = nw.temperature(site, form='exact', month=self.settings['month'], day=self.randday, year=self.randyear)
                elif (self.weather == 'typical'):
                    temp = nw.temperature(site, form='median', month=self.settings['month'])
                elif (self.weather == 'good'):
                    temp = nw.temperature(site, form='good', month=self.settings['month'])
                elif (self.weather == 'bad'):
                    temp = nw.temperature(site, form='bad', month=self.settings['month'])
                strhere += str(np.round(temp,2)).ljust(10)

                # add coherence time, in seconds
                strhere += str(np.round(t_coh,2)).ljust(13)

                # add RMS pointing uncertainty, in seconds
                strhere += str(np.round(RMS_point,2)).ljust(17)

                # add 230GHz FWHM primary beam size
                ind = (stationnames == site)
                stat = np.array(self.array.stations())[ind][0]
                diam = stat.dishes[0].diameter
                pb = ((180.0/np.pi)*3600.0)*((const.c / (230.0e9)) / diam)
                strhere += str(np.round(pb,2)).ljust(20)

                # add the primary beam model
                strhere += PB_model.ljust(12)

                # add the aperture efficiency
                strhere += str(np.round(self.eta_dict[site],4)).ljust(9)

                # add gain means and stds
                strhere += str(1.0).ljust(11)
                strhere += str(0.1).ljust(12)
                strhere += str(1.0).ljust(11)
                strhere += str(0.1).ljust(12)

                # add leakage means and stds                
                strhere += str(0.05+0.05j)[1:-1].ljust(12)
                strhere += str(0.0).ljust(12)
                strhere += str(0.05+0.05j)[1:-1].ljust(12)
                strhere += str(0.0).ljust(12)

                # add feed angle
                if site in const.known_feed_angles.keys():
                    strhere += str(const.known_feed_angles[site]).ljust(20)
                else:
                    strhere += str(const.feed_angle).ljust(20)

                # add mount type
                if site in const.known_mount_types.keys():
                    strhere += const.known_mount_types[site].ljust(18)
                else:
                    strhere += const.mount_type.ljust(18)

                # add dish diameter
                diam = stat.diameter()
                strhere += str(np.round(diam,2)).ljust(17)

                # add xyz coordinates
                coords = stat.xyz()
                strhere += str(np.round(coords[0],8)) + ',' 
                strhere += str(np.round(coords[1],8)) + ',' 
                strhere += str(np.round(coords[2],8))

                # write line
                strhere += '\n'
                outfile.write(strhere)

###################################################
# other functions


def get_station_list():
    """
    Return a list of known stations; "get_station_list" and "get_site_list" are equivalent
    
    Returns:
      (list): a list of station names
    """

    return ng.Station.get_list()


# alias for get_station_list
get_site_list = get_station_list


def determine_mjd(day,month,year):
    """
    Determine the MJD from a given day, month, and year.
    
    Args:
      day (str): Numerical cay of the month; e.g. '15' or '22'
      month (str): Three-letter abbreviation for month of the year; e.g., 'Feb' or 'Sep'
      year (str): Calendar year; e.g., '2025'
    
    Returns:
      (float): MJD corresponding to the input date
    """

    if (month == 'Jan'):
        if int(day) > 31:
            raise Exception('January has fewer than ' + day + 'days!')
        t = Time(year+'-01-'+day+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Feb'):
        if int(day) > 28:
            raise Exception('February has fewer than ' + day + 'days!')
        t = Time(year+'-02-'+day+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Mar'):
        if int(day) > 31:
            raise Exception('March has fewer than ' + day + 'days!')
        t = Time(year+'-03-'+day+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Apr'):
        if int(day) > 30:
            raise Exception('April has fewer than ' + day + 'days!')
        t = Time(year+'-04-'+day+'T00:00:00', format='isot', scale='utc')
    elif (month == 'May'):
        if int(day) > 31:
            raise Exception('May has fewer than ' + day + 'days!')
        t = Time(year+'-05-'+day+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Jun'):
        if int(day) > 30:
            raise Exception('June has fewer than ' + day + 'days!')
        t = Time(year+'-06-'+day+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Jul'):
        if int(day) > 31:
            raise Exception('July has fewer than ' + day + 'days!')
        t = Time(year+'-07-'+day+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Aug'):
        if int(day) > 31:
            raise Exception('August has fewer than ' + day + 'days!')
        t = Time(year+'-08-'+day+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Sep'):
        if int(day) > 30:
            raise Exception('September has fewer than ' + day + 'days!')
        t = Time(year+'-09-'+day+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Oct'):
        if int(day) > 31:
            raise Exception('October has fewer than ' + day + 'days!')
        t = Time(year+'-10-'+day+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Nov'):
        if int(day) > 30:
            raise Exception('November has fewer than ' + day + 'days!')
        t = Time(year+'-11-'+day+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Dec'):
        if int(day) > 31:
            raise Exception('December has fewer than ' + day + 'days!')
        t = Time(year+'-12-'+day+'T00:00:00', format='isot', scale='utc')
    else:
        raise Exception('This month abbreviation is not recognized; should be one of: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec')

    return t.mjd


def make_array(sitelist,D_new,D_override_dict={},array_name=None,freq=230.0):
    """
    Create ngehtutil and ehtim array objects from a list of sites.
    
    Args:
      sitelist (list): A list of site names
      D_new (float): New dish diameter, in meters
      D_override_dict (dict): A dictionary of station names and diameters to override the defaults
      array_name (str): Name to get assigned to the ngehtutil array object
      freq (float): Observing frequency, in GHz
    
    Returns:
      (ngehtutil.array, ehtim.array.Array): An ngehtutil array object and an ehtim array object
    """

    stations = list()
    for site in sitelist:
        stationhere = ng.Station.from_name(site)
        if stationhere.name in list(D_override_dict.keys()):
            stationhere.dishes = [ng.station.Dish(diameter=D_override_dict[stationhere.name])]
        else:
            if (stationhere.existing_dish == False):
                stationhere.dishes = [ng.station.Dish(diameter=D_new)]
        stations.append(stationhere)

    if array_name is not None:
        array = ng.Array(array_name,stations)
    else:
        array = ng.Array('nameless array',stations)
    arr = array.to_ehtim_array(freq)

    return array, arr


def load_image(infile,freq=230.0e9,verbose=0):
    """
    Load an ehtim image or movie object.
    
    Args:
      infile (str): The input path and filename
      freq (float): Observing frequency, in Hz
      verbose (float): Set to >0 for more verbose output
    
    Returns:
      (ehtim.image.Image): An ehtim image (or possibly movie) object; returns None if infile is None
    """

    if infile is None:
        return None

    else:
        try:
            im = eh.image.load_image(infile)
            im.rf = float(np.round(im.rf))
        except:
            if verbose > 0:
                print('Source file does not appear to be an image; assuming that it is a movie file instead.')
            extension = infile.split('.')[-1]
            if extension.lower() in ['hdf5','h5']:
                im = eh.movie.load_hdf5(infile)
            elif extension.lower() == ['fits']:
                im = eh.movie.load_fits(infile)
            elif extension.lower() == ['txt']:
                im = eh.movie.load_txt(infile)
            else:
                raise Exception('Source file does not have a recognized file extension.')
            im.rf = freq
        return im


def eta_dish(freq,sigma,offset):
    """
    Function for computing aperture efficiency.
    
    Args:
      freq (float): observing frequency, in Hz
      sigma (float): surface RMS, in meters
      offset (float): focus offset, in meters
    
    Returns:
      (float): aperture efficiency
    """

    etahere = np.exp(-((4*np.pi*np.sqrt((sigma)**2+(offset)**2))/(const.c/freq))**2)
    return etahere


def get_unready_sites(sites_in_observ,tech_readiness):
    """
    Function to determine which sites will randomly fail technical readiness.
    
    Args:
      sites_in_observ (list): list of sites to use in the observation
      tech_readiness (float): probability of any individual site being technically ready to observe;
                              takes on a value between 0 and 1
            
    Returns:
      (list): sites to drop
    """

    if (tech_readiness > 1.0) | (tech_readiness < 0.0):
        raise Exception('The tech_readiness keyword must take on a value between 0 and 1!')

    p = tech_readiness
    index = np.random.choice([0, 1], size=(len(sites_in_observ)), p=[p,1-p]).astype(bool)
    sites_to_drop = sites_in_observ[index]
    return sites_to_drop


def fringegroups(obs,snr_ref,tint_ref):
    """
    Function to apply the "fringegroups" SNR thresholding scheme to an observation.
    This scheme attempts to mimic the fringe-fitting carried out in the HOPS calibration pipeline.
    
    Args:
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object containing the input observation
      snr_ref (float): strong baseline SNR threshold
      tint_ref (float): strong baseline coherence time, in seconds

    Returns:
      (ehtim.obsdata.Obsdata): eht-imaging Obsdata object containing the thresholded observation
    """

    # get the timestamps
    time = obs.data['time']
    timestamps = np.unique(obs.data['time'])

    # create a running index list of baselines to flag
    master_index = np.zeros(len(obs.data),dtype='bool')
    count = 0

    # create blank dummy obsdata objects
    obs_here = obs.copy()
    obs_here.data = None
    obs_search = obs_here.copy()

    # check all timestamps
    for itime, timestamp in enumerate(timestamps):

        ind_t = (time == timestamp)
        obs_here.data = obs.data[ind_t]

        # scale effective SNR to the actual integration time
        snr_scaled = snr_ref*np.sqrt(obs_here.data['tint'] / tint_ref)

        # determine which baselines are "strong"
        index = (np.abs(obs_here.data['vis'])/obs_here.data['sigma']) >= snr_scaled

        # limit the searched baselines to those that are strong
        obs_search.data = obs_here.data[index]

        # group stations that are connected by strong baselines
        groups = list()
        for datum in obs_search.data:
            bl = [datum['t1'],datum['t2']]
            (merged, remaining) = (set(bl), [])
            for g in groups:
                if bl[0] in g or bl[1] in g:
                    merged |= g
                else:
                    remaining.append(g)
            groups = remaining + [merged]
            
        # assign stations to groups
        site_dict = {}
        for ig, group in enumerate(groups):
            for station in group:
                site_dict[station] = ig

        # check whether both stations on each baseline are in the same group
        for datum in obs_here.data:
            if ((datum['t1'] in site_dict.keys()) & (datum['t2'] in site_dict.keys())):
                if (site_dict[datum['t1']] == site_dict[datum['t2']]):
                    master_index[count] = True
            count += 1

    # apply the flagging
    data_copy = obs.data.copy()
    obs.data = data_copy[master_index]

    return obs


def FPT(obsgen,obs,snr_ref,tint_ref,freq_ref,model_ref=None,**kwargs):
    """
    Function to apply the frequency phase transfer ("FPT") SNR thresholding scheme to an observation.
    This scheme attempts to mimic the fringe-fitting carried out in the HOPS calibration pipeline.
    
    Args:
      obsgen (ngehtsim.obs.obs_generator.obs_generator): ngehtsim obs_generator object containing information about the observation
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object containing the input observation
      snr_ref (float): strong baseline SNR threshold
      tint_ref (float): strong baseline coherence time, in seconds
      freq_ref(float): FPT reference frequency, in GHz
      model_ref (str): path to FPT reference model, or the reference model itself

    Returns:
      (ehtim.obsdata.Obsdata): eht-imaging Obsdata object containing the thresholded observation
    """

    # create dummy obsgen object
    new_settings = obsgen.settings
    new_settings['frequency'] = freq_ref
    new_settings['bandwidth'] = obsgen.settings['bandwidth'] * (freq_ref/float(obsgen.settings['frequency']))
    if ((model_ref is None) | isinstance(model_ref,str)):
        new_settings['model_file'] = model_ref
    obsgen_ref = obs_generator(new_settings)
    if ((model_ref is not None) & (not isinstance(model_ref,str))):
        obsgen_ref.im = model_ref

    # generate observation at reference frequency
    obs_ref = obsgen_ref.observe(obsgen_ref.im,**kwargs)

    # get the timestamps
    time = obs_ref.data['time']
    timestamps = np.unique(obs_ref.data['time'])

    # create a running index list of baselines to flag
    master_index = np.zeros(len(obs_ref.data),dtype='bool')
    count = 0

    # create blank dummy obsdata objects
    obs_here = obs_ref.copy()
    obs_here.data = None
    obs_search = obs_here.copy()

    # check all timestamps
    for itime, timestamp in enumerate(timestamps):

        ind_t = (time == timestamp)
        obs_here.data = obs_ref.data[ind_t]

        # scale effective SNR to the actual integration time
        snr_scaled = snr_ref*np.sqrt(obs_here.data['tint'] / tint_ref)

        # determine which baselines are "strong"
        index = (np.abs(obs_here.data['vis'])/obs_here.data['sigma']) >= snr_scaled

        # limit the searched baselines to those that are strong
        obs_search.data = obs_here.data[index]

        # group stations that are connected by strong baselines
        groups = list()
        for datum in obs_search.data:
            bl = [datum['t1'],datum['t2']]
            (merged, remaining) = (set(bl), [])
            for g in groups:
                if bl[0] in g or bl[1] in g:
                    merged |= g
                else:
                    remaining.append(g)
            groups = remaining + [merged]
            
        # assign stations to groups
        site_dict = {}
        for ig, group in enumerate(groups):
            for station in group:
                site_dict[station] = ig

        # check whether both stations on each baseline are in the same group
        for datum in obs_here.data:
            if ((datum['t1'] in site_dict.keys()) & (datum['t2'] in site_dict.keys())):
                if (site_dict[datum['t1']] == site_dict[datum['t2']]):
                    master_index[count] = True
            count += 1

    # apply the flagging
    data_copy = obs.data.copy()
    obs.data = data_copy[master_index]

    return obs
