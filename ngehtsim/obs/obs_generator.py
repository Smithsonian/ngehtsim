###################################################
# imports

import numpy as np
import ehtim as eh
from collections import defaultdict
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun
import yaml
import time
import os
import copy

try:
    import ngEHTforecast.fisher as fp
except ImportError:
    print('Warning: ngEHTforecast not installed! Cannot use FisherForecast functionality.')

import ngehtsim.const_def as const
import ngehtsim.weather.weather as nw

###################################################
# class definition


class obs_generator(object):
    """
    Class that organizes information for generating synthetic observations.

    Attributes:
      settings (dict): Dictionary of information about the observation generation setup
      settings_file (str): Path to the input settings file; if set to None, will use default settings.
                           Note that any settings specified by the settings keyword argument will override
                           the corresponding settings from the settings file.
      verbosity (float): Set to >0 for more verbose output
      weight (float): Set to >0 to store more information in the obs_generator object
      D_overrides (dict): A dictionary of station names and diameters to override defaults
      surf_rms_overrides (dict): A dictionary of station names and surface RMS values (in microns) to override defaults
      receiver_configuration_overrides (dict): A dictionary of station names and available receivers to override defaults
      bandwidth_overrides (dict): A dictionary of station names and bandwidth values to override defaults
      T_R_overrides (dict): A dictionary of station names and receiver temperature values to override defaults
      sideband_ratio_overrides (dict): A dictionary of station names and sideband ratio values to override defaults
      lo_freq_overrides (dict): A dictionary of station names and receiver lowest frequency values to override defaults
      hi_freq_overrides (dict): A dictionary of station names and receiver lowest frequency values to override defaults
      ap_eff_overrides (dict): A dictionary of station names and aperture efficiency values to override defaults
      custom_receivers (dict): A dictionary of custom receiver names and properties
      station_uptimes (dict): A dictionary of station names and associated uptime ranges, in UT
      array (str): Provide the name of a known array to load the corresponding sites and configuration
      ephem (str): path to the ephemeris for a space station
    """

    # initialize class instantiation
    def __init__(self, settings={}, settings_file=None, verbosity=0, weight=0, D_overrides={},
                 surf_rms_overrides={}, receiver_configuration_overrides={}, bandwidth_overrides={},
                 T_R_overrides={}, sideband_ratio_overrides={}, lo_freq_overrides={}, hi_freq_overrides={},
                 ap_eff_overrides={}, custom_receivers={}, station_uptimes={}, array=None, ephem='ephemeris/space'):

        #############################
        # parse inputs

        self.settings_file = settings_file
        self.verbosity = verbosity
        self.weight = weight
        self.D_overrides = copy.deepcopy(D_overrides)
        self.surf_rms_overrides = copy.deepcopy(surf_rms_overrides)
        self.receiver_configuration_overrides = copy.deepcopy(receiver_configuration_overrides)
        self.bandwidth_overrides = copy.deepcopy(bandwidth_overrides)
        self.T_R_overrides = copy.deepcopy(T_R_overrides)
        self.sideband_ratio_overrides = copy.deepcopy(sideband_ratio_overrides)
        self.lo_freq_overrides = copy.deepcopy(lo_freq_overrides)
        self.hi_freq_overrides = copy.deepcopy(hi_freq_overrides)
        self.ap_eff_overrides = copy.deepcopy(ap_eff_overrides)
        self.custom_receivers = copy.deepcopy(custom_receivers)
        self.station_uptimes = copy.deepcopy(station_uptimes)
        self.array = array
        self.ephem = ephem

        #############################
        # load settings

        # start with some default settings
        self.settings = copy.deepcopy(const.default_settings)

        # check if user wants to load settings from a passed file
        if settings_file is not None:
            self.load_yaml_settings()
            if self.verbosity > 0:
                print('Loading settings from {0}'.format(settings_file))

        # update the settings with any additional passed information
        self.settings.update(settings)

        #############################
        # check/fix some easy issues

        # make sure the passed settings are all valid
        for key in self.settings.keys():
            if key not in const.default_settings.keys():
                raise Exception(key+' is not a recognized setting!')

        # set array name if it is provided
        if self.array is None:
            if self.settings['array'] is not None:
                self.array = self.settings['array']

        # if sites are specified, ignore the array
        if (self.settings['sites'] is not None):
            if len(self.settings['sites']) > 0:
                self.array = None

        # check that any custom receivers have all of the necessary settings
        if len(self.custom_receivers.keys()) > 0:
            for rec in list(self.custom_receivers.keys()):
                if ('lo' not in self.custom_receivers[rec].keys()):
                    raise Exception('Custom receivers must contain a "lo" key specifying the lowest frequency.')
                if ('hi' not in self.custom_receivers[rec].keys()):
                    raise Exception('Custom receivers must contain a "hi" key specifying the highest frequency.')
                if ('T_R' not in self.custom_receivers[rec].keys()):
                    raise Exception('Custom receivers must contain a "T_R" key specifying the receiver temperature (in K).')
                if ('SSR' not in self.custom_receivers[rec].keys()):
                    raise Exception('Custom receivers must contain a "SSR" key specifying sideband separation ratio.')

        # check that all station uptimes specify two times
        if len(self.station_uptimes.keys()) > 0:
            for site in list(self.station_uptimes.keys()):
                if len(self.station_uptimes[site]) != 2:
                    raise Exception('Station uptime dictionary must provide an earliest and latest time for each specified station.')

        #############################
        # extract commonly-used settings

        self.model_file = self.settings['model_file']
        self.freq = float(self.settings['frequency'])*(1.0e9)
        self.weather = self.settings['weather']
        self.weather_year = self.settings['weather_year']
        self.weather_day = self.settings['weather_day']

        #############################
        # run initialization functions

        self.set_seed()
        self.get_sites()
        self.translate_sites()
        self.set_coords()
        self.mjd = determine_mjd(self.settings['day'], self.settings['month'], self.settings['year'])
        self.arr = make_array(self.sites, ephem=self.ephem, verbosity=self.verbosity)
        self.set_receivers()
        self.set_bands()
        self.set_bandwidths()
        self.set_ap_effs()
        self.im = load_image(self.model_file, freq=self.freq, verbosity=self.verbosity)
        self.tabulate_weather()
        self.set_telescope_properties()
        self.get_obs_times()

        #############################
        # other settings

        self.obs_empty = None

    ###################################################
    # initialization functions

    # load and store settings from file
    def load_yaml_settings(self):
        loader = yaml.SafeLoader
        with open(self.settings_file, 'r') as fi:
            self.settings.update(yaml.load(fi, Loader=loader))

    # set random number seed and generator
    def set_seed(self):
        if self.settings['random_seed'] is None:
            self.seed = int((time.time() % 100000) * 1000)
        else:
            self.seed = self.settings['random_seed']
        self.rng = np.random.default_rng(seed=self.seed)

    # generate the site list
    def get_sites(self):

        # initialize site list
        self.sites = list()

        # if a known array is specified, pull its sites and overrides
        if self.array in list(const.known_arrays.keys()):
            self.sites = copy.deepcopy(const.known_arrays[self.array])

            D_overrides_here = copy.deepcopy(const.known_array_D_overrides[self.array])
            D_overrides_here.update(self.D_overrides)
            self.D_overrides = D_overrides_here

            surf_rms_overrides_here = copy.deepcopy(const.known_array_surf_rms_overrides[self.array])
            surf_rms_overrides_here.update(self.surf_rms_overrides)
            self.surf_rms_overrides = surf_rms_overrides_here

            receiver_configuration_overrides_here = copy.deepcopy(const.known_array_receiver_configuration_overrides[self.array])
            receiver_configuration_overrides_here.update(self.receiver_configuration_overrides)
            self.receiver_configuration_overrides = receiver_configuration_overrides_here

            bandwidth_overrides_here = copy.deepcopy(const.known_array_bandwidth_overrides[self.array])
            bandwidth_overrides_here.update(self.bandwidth_overrides)
            self.bandwidth_overrides = bandwidth_overrides_here

            T_R_overrides_here = copy.deepcopy(const.known_array_T_R_overrides[self.array])
            T_R_overrides_here.update(self.T_R_overrides)
            self.T_R_overrides = T_R_overrides_here

            sideband_ratio_overrides_here = copy.deepcopy(const.known_array_sideband_ratio_overrides[self.array])
            sideband_ratio_overrides_here.update(self.sideband_ratio_overrides)
            self.sideband_ratio_overrides = sideband_ratio_overrides_here

            lo_freq_overrides_here = copy.deepcopy(const.known_array_lo_freq_overrides[self.array])
            lo_freq_overrides_here.update(self.lo_freq_overrides)
            self.lo_freq_overrides = lo_freq_overrides_here

            hi_freq_overrides_here = copy.deepcopy(const.known_array_hi_freq_overrides[self.array])
            hi_freq_overrides_here.update(self.hi_freq_overrides)
            self.hi_freq_overrides = hi_freq_overrides_here

            ap_eff_overrides_here = copy.deepcopy(const.known_array_ap_eff_overrides[self.array])
            ap_eff_overrides_here.update(self.ap_eff_overrides)
            self.ap_eff_overrides = ap_eff_overrides_here

        # but if sites are provided, then override the array
        if self.settings['sites'] is not None:
            self.sites = self.settings['sites']

            # still consider the overrides if the array is named
            if self.array in list(const.known_arrays.keys()):

                D_overrides_here = copy.deepcopy(const.known_array_D_overrides[self.array])
                D_overrides_here.update(self.D_overrides)
                self.D_overrides = D_overrides_here

                surf_rms_overrides_here = copy.deepcopy(const.known_array_surf_rms_overrides[self.array])
                surf_rms_overrides_here.update(self.surf_rms_overrides)
                self.surf_rms_overrides = surf_rms_overrides_here

                receiver_configuration_overrides_here = copy.deepcopy(const.known_array_receiver_configuration_overrides[self.array])
                receiver_configuration_overrides_here.update(self.receiver_configuration_overrides)
                self.receiver_configuration_overrides = receiver_configuration_overrides_here

                bandwidth_overrides_here = copy.deepcopy(const.known_array_bandwidth_overrides[self.array])
                bandwidth_overrides_here.update(self.bandwidth_overrides)
                self.bandwidth_overrides = bandwidth_overrides_here

                T_R_overrides_here = copy.deepcopy(const.known_array_T_R_overrides[self.array])
                T_R_overrides_here.update(self.T_R_overrides)
                self.T_R_overrides = T_R_overrides_here

                sideband_ratio_overrides_here = copy.deepcopy(const.known_array_sideband_ratio_overrides[self.array])
                sideband_ratio_overrides_here.update(self.sideband_ratio_overrides)
                self.sideband_ratio_overrides = sideband_ratio_overrides_here

                lo_freq_overrides_here = copy.deepcopy(const.known_array_lo_freq_overrides[self.array])
                lo_freq_overrides_here.update(self.lo_freq_overrides)
                self.lo_freq_overrides = lo_freq_overrides_here

                hi_freq_overrides_here = copy.deepcopy(const.known_array_hi_freq_overrides[self.array])
                hi_freq_overrides_here.update(self.hi_freq_overrides)
                self.hi_freq_overrides = hi_freq_overrides_here

                ap_eff_overrides_here = copy.deepcopy(const.known_array_ap_eff_overrides[self.array])
                ap_eff_overrides_here.update(self.ap_eff_overrides)
                self.ap_eff_overrides = ap_eff_overrides_here

        # otherwise it's unclear what the user wants
        if (self.array not in list(const.known_arrays.keys())) and (self.settings['sites'] is None):
            raise Exception('No known array or sites have been specified!')

        # remove duplicates
        temp_sites = np.unique(np.array(self.sites))
        self.sites = list(temp_sites)

    # use common site names and make sure all sites are known
    def translate_sites(self):
        for isite, site in enumerate(self.sites):
            if site in list(const.translation_dict.keys()):
                self.sites[isite] = copy.deepcopy(const.translation_dict[site])
            else:
                if site not in const.known_stations:
                    if site != 'space':
                        raise Exception(site+' is not a known station.')

    # set source coordinates
    def set_coords(self):

        # retrieve coordinates from source, if known
        if self.settings['source'] in list(const.known_sources.keys()):
            self.RA = copy.deepcopy(const.known_sources[self.settings['source']]['RA'])
            self.DEC = copy.deepcopy(const.known_sources[self.settings['source']]['DEC'])
        else:
            if ((self.settings['RA'] is None) & (self.settings['DEC'] is None)):
                raise Exception('A known source and/or a set of (RA,DEC) coordinates must be specified.')

        # if coordinates are specified, use those instead
        if self.settings['RA'] is not None:
            self.RA = self.settings['RA']
        if self.settings['DEC'] is not None:
            self.DEC = self.settings['DEC']

    # create a receiver suite dictionary
    def set_receivers(self):

        receiver_setup = {}

        for site in self.sites:

            receiver_setup[site] = {}

            if site in list(self.receiver_configuration_overrides.keys()):
                for rec in self.receiver_configuration_overrides[site]:
                    if rec in list(const.receivers.keys()):
                        receiver_setup[site][rec] = copy.deepcopy(const.receivers[rec])
                    elif rec in list(self.custom_receivers.keys()):
                        receiver_setup[site][rec] = copy.deepcopy(self.custom_receivers[rec])
                    else:
                        raise Exception('Receiver '+rec+' not recognized.')
            else:
                receiver_setup[site] = copy.deepcopy(const.receivers)

            if site in list(self.T_R_overrides.keys()):
                for rec in self.T_R_overrides[site]:
                    if rec in list(receiver_setup[site].keys()):
                        receiver_setup[site][rec]['T_R'] = self.T_R_overrides[site][rec]

            if site in list(self.sideband_ratio_overrides.keys()):
                for rec in self.sideband_ratio_overrides[site]:
                    if rec in list(receiver_setup[site].keys()):
                        receiver_setup[site][rec]['SSR'] = self.sideband_ratio_overrides[site][rec]

            if site in list(self.lo_freq_overrides.keys()):
                for rec in self.lo_freq_overrides[site]:
                    if rec in list(receiver_setup[site].keys()):
                        receiver_setup[site][rec]['lo'] = self.lo_freq_overrides[site][rec]

            if site in list(self.hi_freq_overrides.keys()):
                for rec in self.hi_freq_overrides[site]:
                    if rec in list(receiver_setup[site].keys()):
                        receiver_setup[site][rec]['hi'] = self.hi_freq_overrides[site][rec]

        self.receivers = receiver_setup

    # set the receiver bands that will be used for each site
    def set_bands(self):
        self.bands = {}
        freq = self.freq / (1.0e9)
        for site in self.sites:
            self.bands[site] = None
            for band in list(self.receivers[site].keys()):
                if ((self.receivers[site][band]['lo'] <= freq) & (self.receivers[site][band]['hi'] >= freq)):
                    self.bands[site] = band

    # sort out the bandwidth info for each site and for the whole array
    def set_bandwidths(self):

        # set up the bandwidth dictionary
        bandwidth_setup = {}
        for site in self.sites:
            if site in list(self.bandwidth_overrides.keys()):
                bandwidth_setup[site] = self.bandwidth_overrides[site]
            else:
                bandwidth_setup[site] = {}
                for key in list(self.receivers[site].keys()):
                    bandwidth_setup[site][key] = self.settings['bandwidth']

        # determine the unique bandwidths
        unique_bandwidths = list()
        for key in list(bandwidth_setup.keys()):
            for key2 in list(bandwidth_setup[key].keys()):
                if bandwidth_setup[key][key2] not in unique_bandwidths:
                    unique_bandwidths.append(bandwidth_setup[key][key2])

        self.bandwidth_setup = bandwidth_setup
        self.unique_bandwidths = unique_bandwidths

    # create an aperture efficiency dictionary
    def set_ap_effs(self):

        # initialize the aperture efficiency dictionary
        ap_eff_setup = {}
        for site in self.sites:
            ap_eff_setup[site] = {}
            for key in list(self.receivers[site].keys()):
                ap_eff_setup[site][key] = const.ap_eff

        # update according to overrides
        for site in self.sites:
            if site in list(self.ap_eff_overrides.keys()):
                ap_eff_setup[site].update(self.ap_eff_overrides[site])

        self.ap_eff_setup = ap_eff_setup

    # extract and store the relevant weather information
    def tabulate_weather(self):

        # initialize dictionaries
        tau_dict = defaultdict(dict)
        Tatm_dict = defaultdict(dict)
        Tgnd_dict = defaultdict(dict)
        Tb_dict = defaultdict(dict)
        windspeed_dict = defaultdict(dict)

        # get a day and year for the weather parameters
        if (self.weather == 'random'):
            # pick a random past date from which to pull the weather
            self.weather_year = self.rng.integers(const.year_min, const.year_max, endpoint=True)
            if (self.settings['month'] == 'Feb'):
                self.weather_day = self.rng.integers(1, 28, endpoint=True)
            elif (self.settings['month'] in ['Apr', 'Jun', 'Sep', 'Nov']):
                self.weather_day = self.rng.integers(1, 30, endpoint=True)
            else:
                self.weather_day = self.rng.integers(1, 31, endpoint=True)
        else:
            # use the specified date
            if self.weather_year is None:
                self.weather_year = int(self.settings['year'])
            if self.weather_day is None:
                self.weather_day = int(self.settings['day'])

        # read in the weather info and store it
        for isite, site in enumerate(self.sites):

            if site != 'space':

                if ((self.weather == 'random') | (self.weather == 'exact')):
                    form = 'exact'
                elif ((self.weather == 'mean') | (self.weather == 'average')):
                    form = 'mean'
                elif ((self.weather == 'typical') | (self.weather == 'median')):
                    form = 'median'
                elif (self.weather == 'good'):
                    form = 'good'
                elif ((self.weather == 'bad') | (self.weather == 'poor')):
                    form = 'bad'

                tau_here = nw.opacity(site, form=form, month=self.settings['month'], day=self.weather_day, year=self.weather_year, freq=self.freq/(1.0e9))
                Tb_here = nw.brightness_temperature(site, form=form, month=self.settings['month'], day=self.weather_day, year=self.weather_year, freq=self.freq/(1.0e9))
                ws_here = nw.windspeed(site, form=form, month=self.settings['month'], day=self.weather_day, year=self.weather_year)
                Tgnd_here = nw.temperature(site, form=form, month=self.settings['month'], day=self.weather_day, year=self.weather_year)

                # divide out the opacity term to get the effective atmospheric temperature
                Tatm_here = (Tb_here - (const.T_CMB*np.exp(-tau_here))) / (1.0 - np.exp(-tau_here))

                # store the info in the dictionaries
                tau_dict[site] = tau_here
                Tatm_dict[site] = Tatm_here
                Tb_dict[site] = Tb_here
                windspeed_dict[site] = ws_here
                Tgnd_dict[site] = Tgnd_here

            else:

                if self.verbosity > 1:
                    print('For space dish, assuming perfect weather.')

                tau_dict[site] = 0.0
                Tatm_dict[site] = 0.0
                Tb_dict[site] = const.T_CMB
                windspeed_dict[site] = 0.0
                Tgnd_dict[site] = const.T_CMB

        # store the dictionaries
        self.tau_dict = tau_dict
        self.Tatm_dict = Tatm_dict
        self.Tb_dict = Tb_dict
        self.windspeed_dict = windspeed_dict
        self.Tgnd_dict = Tgnd_dict

    # generate dictionaries of telescope properties
    def set_telescope_properties(self):

        D_dict = {}
        eta_dict = {}
        for site in self.sites:

            # start with the values for a new site
            D_dict[site] = self.settings['D_new']
            rms_here = const.surf_rms
            ap_eff_here = const.ap_eff

            # if the site is known, replace those values with the known ones
            if site in list(const.known_diameters.keys()):
                D_dict[site] = const.known_diameters[site]
            if site in list(const.known_surf_rms.keys()):
                rms_here = const.known_surf_rms[site]

            # if the user has provided overrides, use those instead
            if site in list(self.D_overrides.keys()):
                D_dict[site] = self.D_overrides[site]
            if site in list(self.surf_rms_overrides.keys()):
                rms_here = self.surf_rms_overrides[site]
            if site in list(self.ap_eff_overrides.keys()):
                if self.bands[site] in list(self.ap_eff_overrides[site].keys()):
                    ap_eff_here = self.ap_eff_overrides[site][self.bands[site]]

            eta_dict[site] = eta_dish(self.freq, rms_here, const.focus_offset, ap_eff_here)

        self.D_dict = D_dict
        self.eta_dict = eta_dict

    # segment the observation into timestamps
    def get_obs_times(self):
        t_first = self.settings['t_start']
        N_obs = int(np.ceil(self.settings['dt']/(self.settings['t_rest']/3600.)))
        t_last = t_first+float(N_obs-1)*(self.settings['t_rest']/3600.)
        self.t_seg_times = np.linspace(t_first, t_last, N_obs)
        if self.verbosity > 0:
            print("Number of timestamps: {0}".format(N_obs))
            print("Beginning of first integration: {0}".format(t_first))
            print("Beginning of last integration: {0}".format(t_last))
            print('Scan start times: {0}'.format(self.t_seg_times))

    ###################################################
    # functions for generating observations

    # generate a raw observation
    def observe(self, input_model, addnoise=True, addgains=True, gainamp=0.04, opacitycal=True,
                flagwind=True, flagday=False, addFR=False, allow_mixed_basis=False,
                el_min=const.el_min, el_max=const.el_max, p=None):
        """
        Generate a raw single-band observation that folds in weather-based opacity and sensitivity effects.

        Args:
          input_model (ehtim.image.Image, ehtim.movie.Movie, ehtim.model.Model, ngEHTforecast.fisher.fisher_forecast.FisherForecast): input source model
          addnoise (bool): flag for whether or not to add thermal noise to the visibilities
          addgains (bool): flag for whether or not to add station gain corruptions
          gainamp (float): standard deviation of amplitude log-gains
          opacitycal (bool): flag for whether or not to assume that atmospheric opacity is assumed to be calibrated out
          flagwind (bool): flag for whether to derate sites with high wind
          flagday (bool): flag for whether to flag sites during the local daytime
          addFR (bool): flag for whether or not to add feed rotations
          allow_mixed_basis (bool): flag for whether to apply polarization basis conversions
          el_min (float): minimum elevation that a site can observe at, in degrees
          el_max (float): maximum elevation that a site can observe at, in degrees
          p (numpy.ndarray): list of parameters for an input ngEHTforecast.fisher.fisher_forecast.FisherForecast object

        Returns:
          (ehtim.obsdata.Obsdata): eht-imaging Obsdata object containing the generated observation
        """

        # print some warnings
        if addFR:
            print('WARNING: adding feed rotations is currently known to break with multi-frequency data generation, and it is suspect at all times.')
        if allow_mixed_basis:
            print('WARNING: data generated in a non-circular polarization basis does not have properly-stored metadata info.')

        # generate an empty obsdata object
        if ((self.obs_empty is None) or (self.obs_empty.rf != self.freq)):
            self.obs_empty = self.arr.obsdata(self.RA,
                                              self.DEC,
                                              self.freq,
                                              (1.0e9)*float(self.settings['bandwidth']),
                                              self.settings['t_int'],
                                              self.settings['t_rest'],
                                              self.settings['t_start'],
                                              self.settings['t_start'] + self.settings['dt'],
                                              mjd=self.mjd,
                                              polrep='stokes',
                                              tau=0.0,
                                              timetype='UTC',
                                              elevmin=-90,
                                              elevmax=90,
                                              fix_theta_GMST=False)

        # apply elevation cuts to ground stations
        els = self.obs_empty.unpack(['el1', 'el2'])
        mask = (self.obs_empty.data['t1'] == 'space') | ((els['el1'] > el_min) & (els['el1'] < el_max))
        mask &= (self.obs_empty.data['t2'] == 'space') | ((els['el2'] > el_min) & (els['el2'] < el_max))
        self.obs_empty.data = self.obs_empty.data[mask]

        # observe the source
        if isinstance(input_model, eh.image.Image):
            input_model.ra = self.RA
            input_model.dec = self.DEC
            input_model.mjd = self.mjd
            input_model.source = self.settings['source']
            input_model.rf = self.freq
            if self.verbosity <= 0:
                with eh.parloop.HiddenPrints():
                    obs = input_model.observe_same_nonoise(self.obs_empty, ttype=self.settings['ttype'], fft_pad_factor=self.settings['fft_pad_factor'])
            else:
                obs = input_model.observe_same_nonoise(self.obs_empty, ttype=self.settings['ttype'], fft_pad_factor=self.settings['fft_pad_factor'])
            F0 = np.abs(input_model.sample_uv([[0.0, 0.0]], ttype=self.settings['ttype'])[0][0])
        elif isinstance(input_model, eh.movie.Movie):
            input_model.ra = self.RA
            input_model.dec = self.DEC
            input_model.mjd = self.mjd
            input_model.source = self.settings['source']
            input_model.rf = self.freq
            if self.verbosity <= 0:
                with eh.parloop.HiddenPrints():
                    obs = input_model.observe_same_nonoise(self.obs_empty, ttype=self.settings['ttype'], fft_pad_factor=self.settings['fft_pad_factor'], repeat=True)
            else:
                obs = input_model.observe_same_nonoise(self.obs_empty, ttype=self.settings['ttype'], fft_pad_factor=self.settings['fft_pad_factor'], repeat=True)
            F0 = np.mean(input_model.lightcurve)
        elif isinstance(input_model, eh.model.Model):
            input_model.ra = self.RA
            input_model.dec = self.DEC
            input_model.mjd = self.mjd
            input_model.source = self.settings['source']
            input_model.rf = self.freq
            if self.verbosity <= 0:
                with eh.parloop.HiddenPrints():
                    obs = input_model.observe_same_nonoise(self.obs_empty)
            else:
                obs = input_model.observe_same_nonoise(self.obs_empty)
            F0 = np.abs(input_model.sample_uv(0.0, 0.0))
        elif isinstance(input_model, fp.FisherForecast):
            if p is None:
                raise Exception('When observing an ngEHTforecast model, the parameter vector keyword argument p must be specified!')
            obs = self.obs_empty.copy()
            obs.source = self.settings['source']
            if (input_model.stokes == 'I'):
                Ivis = input_model.visibilities(obs, p, verbosity=self.verbosity)
                obs.data['vis'] = Ivis
            else:
                obs.switch_polrep(polrep_out='circ')
                RRvis, LLvis, RLvis, LRvis = input_model.visibilities(obs, p, verbosity=self.verbosity)
                obs.data['rrvis'] = RRvis
                obs.data['llvis'] = LLvis
                obs.data['rlvis'] = RLvis
                obs.data['lrvis'] = LRvis
            dumobs = self.obs_empty.copy()
            dumdatatable = dumobs.data[0]
            dumdatatable['u'] = 0.0
            dumdatatable['v'] = 0.0
            dumobs.data = dumdatatable
            F0 = np.abs(input_model.visibilities(dumobs, p))

        # make sure we're in a circular basis
        obs = obs.switch_polrep(polrep_out='circ')

        # extract relevant information
        t1 = obs.data['t1']
        t2 = obs.data['t2']
        sites_obs = np.unique(np.concatenate((t1, t2)))
        els = obs.unpack(['el1', 'el2'], ang_unit='rad')
        pars = obs.unpack(['par_ang1', 'par_ang2'], ang_unit='rad')
        el1 = els['el1']
        el2 = els['el2']
        par1 = pars['par_ang1']
        par2 = pars['par_ang2']
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

        # loop through the sites in the array
        flagsites = list()
        uptime_mask = np.ones(len(obs.data),dtype=bool)
        for isite, site in enumerate(sites_obs):

            # zenith opacity, atmospheric temperature, ground temperature, and windspeed
            tau_z = self.tau_dict[site]
            Tatm = self.Tatm_dict[site]
            Tgnd = self.Tgnd_dict[site]
            ws = self.windspeed_dict[site]

            # determine effective collecting area
            Aeff = (np.pi/4.0)*self.eta_dict[site]*((self.D_dict[site])**2)

            # if the windspeed exceeds the shutdown threshold, mark the site as to be flagged
            if (ws > const.windspeed_shutdown):
                if flagwind:
                    flagsites.append(site)
                    if self.verbosity > 0:
                        print(site + ' cannot observe because of high wind.')

            # flag the daytime observations, if desired
            if flagday:
                if site != 'space':
                    
                    # get location of this site
                    lon = const.known_longitudes[site]
                    lat = const.known_latitudes[site]
                    elev = const.known_elevations[site]
                    location = EarthLocation.from_geodetic(lon,lat,height=elev)

                    # get the altitude of the Sun over time
                    jd = obs.mjd + 2400000.5 + (times/24.0)
                    timehere = Time(jd, format='jd')
                    altazframe = AltAz(obstime=timehere, location=location)
                    sun_altaz = get_sun(timehere).transform_to(altazframe)
                    alt = sun_altaz.alt.value

                    # mark as to-be-flagged all times for which the Sun is above the horizon
                    ind_daytime = (((t1 == site) | (t2 == site)) & (sun_altaz.alt.value > 0.0))
                    uptime_mask[ind_daytime] = False

            # flag the times that fall outside of the specified station uptime window
            if site in list(self.station_uptimes.keys()):
                ind_too_early = (((t1 == site) | (t2 == site)) & (times < self.station_uptimes[site][0]))
                ind_too_late = (((t1 == site) | (t2 == site)) & (times > self.station_uptimes[site][1]))
                uptime_mask[ind_too_early] = False
                uptime_mask[ind_too_late] = False

            # indices for this site
            ind1 = (t1 == site)
            ind2 = (t2 == site)

            # transform polarization basis if need be
            if allow_mixed_basis:
                if site in list(const.known_polbases.keys()):
                    if (const.known_polbases[site] == 'linear'):

                        # populate vectors of transform matrices
                        tform_mat1 = np.zeros((ind1.sum(), 2, 2), dtype=complex)
                        tform_mat2 = np.zeros((ind2.sum(), 2, 2), dtype=complex)
                        tform_mat1[:] = const.circ_to_lin
                        tform_mat2[:] = np.conj(const.circ_to_lin).T

                        # populate vectors of coherency matrices
                        coh_mat1 = np.zeros((ind1.sum(), 2, 2), dtype=complex)
                        coh_mat1[:,0,0] = obs.data['rrvis'][ind1]
                        coh_mat1[:,0,1] = obs.data['rlvis'][ind1]
                        coh_mat1[:,1,0] = obs.data['lrvis'][ind1]
                        coh_mat1[:,1,1] = obs.data['llvis'][ind1]
                        coh_mat2 = np.zeros((ind2.sum(), 2, 2), dtype=complex)
                        coh_mat2[:,0,0] = obs.data['rrvis'][ind2]
                        coh_mat2[:,0,1] = obs.data['rlvis'][ind2]
                        coh_mat2[:,1,0] = obs.data['lrvis'][ind2]
                        coh_mat2[:,1,1] = obs.data['llvis'][ind2]

                        # transform the basis
                        coh_mat_tformed1 = np.matmul(tform_mat1,coh_mat1)
                        coh_mat_tformed2 = np.matmul(coh_mat2,tform_mat2)

                        # re-populate the data vector
                        obs.data['rrvis'][ind1] = coh_mat_tformed1[:,0,0]
                        obs.data['rlvis'][ind1] = coh_mat_tformed1[:,0,1]
                        obs.data['lrvis'][ind1] = coh_mat_tformed1[:,1,0]
                        obs.data['llvis'][ind1] = coh_mat_tformed1[:,1,1]
                        obs.data['rrvis'][ind2] = coh_mat_tformed2[:,0,0]
                        obs.data['rlvis'][ind2] = coh_mat_tformed2[:,0,1]
                        obs.data['lrvis'][ind2] = coh_mat_tformed2[:,1,0]
                        obs.data['llvis'][ind2] = coh_mat_tformed2[:,1,1]

            # get opacities at each timestamp
            if site != 'space':
                tau1[ind1] = tau_z / np.cos((np.pi/2.0) - el1[ind1])
                tau2[ind2] = tau_z / np.cos((np.pi/2.0) - el2[ind2])
            else:
                tau1[ind1] = 0.0
                tau2[ind2] = 0.0

            # get Tb contributions at each timestamp
            Tsource = (F0*Aeff)/(2.0*const.k)

            if site != 'space':
                Tb1[ind1] = ((const.T_CMB + Tsource)*np.exp(-tau1[ind1])) + (Tatm*(1.0 - np.exp(-tau1[ind1])))
                Tb2[ind2] = ((const.T_CMB + Tsource)*np.exp(-tau2[ind2])) + (Tatm*(1.0 - np.exp(-tau2[ind2])))
            else:
                Tb1[ind1] = const.T_CMB + Tsource
                Tb2[ind2] = const.T_CMB + Tsource

            # if this site does not have an appropriate receiver, temporarily assign it some values
            band = self.bands[site]
            if band is None:
                T_R = 0.0
                sideband_ratio = 0.0

            # otherwise, retrieve the receiver temperature and sideband separation ratio
            else:
                T_R = self.receivers[site][band]['T_R']
                sideband_ratio = self.receivers[site][band]['SSR']

            # determine system temperatures
            Tsys1[ind1] = (T_R + (const.eta_ff*Tb1[ind1]) + ((1.0 - const.eta_ff)*Tgnd))*(1.0 + sideband_ratio)
            Tsys2[ind2] = (T_R + (const.eta_ff*Tb2[ind2]) + ((1.0 - const.eta_ff)*Tgnd))*(1.0 + sideband_ratio)

            # determine SEFDs
            SEFD1[ind1] = (2.0*const.k*Tsys1[ind1])/Aeff
            SEFD2[ind2] = (2.0*const.k*Tsys2[ind2])/Aeff

            # modify SEFDs to account for wind
            if flagwind:
                SEFD_factor = windspeed_SEFD_modification(ws)
                SEFD1[ind1] *= SEFD_factor
                SEFD2[ind2] *= SEFD_factor

            # determine bandwidth
            if band in list(self.bandwidth_setup[site].keys()):
                valhere = self.bandwidth_setup[site][band]*(1.0e9)
            else:
                valhere = obs.bw
            bw1[ind1] = valhere
            bw2[ind2] = valhere

            # determine various angles relevant for feed rotations
            if site in list(const.known_mount_types.keys()):
                mttyp = const.known_mount_types[site]
            else:
                mttyp = const.mount_type
            if site in list(const.known_feed_angles.keys()):
                fdang = const.known_feed_angles[site]
            else:
                fdang = const.feed_angle
            f_el1[ind1] = const.mount_type_dict[mttyp]['f_el']
            f_el2[ind2] = const.mount_type_dict[mttyp]['f_el']
            f_par1[ind1] = const.mount_type_dict[mttyp]['f_par']
            f_par2[ind2] = const.mount_type_dict[mttyp]['f_par']
            phi_off1[ind1] = fdang
            phi_off2[ind2] = fdang

            # generate gains
            if addgains:
                for t in tuniq:
                    ind1here = ((times == t) & (t1 == site))
                    ind2here = ((times == t) & (t2 == site))
                    gainamphere = 10.0**(gainamp*self.rng.normal(0.0, 1.0))
                    gainphasehere = self.rng.uniform(-np.pi, np.pi)
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

        # use the smaller bandwidth on each baseline
        bw = np.zeros_like(bw1)
        ind1 = (bw1 <= bw2)
        ind2 = (bw2 <= bw1)
        bw[ind1] = bw1[ind1]
        bw[ind2] = bw2[ind2]

        # store and apply feed rotations
        if addFR:
            fa_1 = (f_par1*par1) + (f_el1*el1) + ((np.pi/180.0)*phi_off1)
            fa_2 = (f_par2*par2) + (f_el2*el2) + ((np.pi/180.0)*phi_off2)
            if self.weight > 0:
                self.fa_1 = fa_1
                self.fa_2 = fa_2
            obs.data['rrvis'] *= np.exp(-(1.0j)*fa_1)*np.exp((1.0j)*fa_2)
            obs.data['rlvis'] *= np.exp(-(1.0j)*fa_1)*np.exp(-(1.0j)*fa_2)
            obs.data['lrvis'] *= np.exp((1.0j)*fa_1)*np.exp((1.0j)*fa_2)
            obs.data['llvis'] *= np.exp((1.0j)*fa_1)*np.exp(-(1.0j)*fa_2)

        # store and apply gains
        if addgains:
            g1R = gainamp1R*np.exp((1.0j)*gainphase1R)
            g2R = gainamp2R*np.exp((1.0j)*gainphase2R)
            g1L = gainamp1L*np.exp((1.0j)*gainphase1L)
            g2L = gainamp2L*np.exp((1.0j)*gainphase2L)
            if self.weight > 0:
                self.station_gains1R = g1R
                self.station_gains2R = g2R
                self.station_gains1L = g1L
                self.station_gains2L = g2L
            obs.data['rrvis'] *= g1R*np.conj(g2R)
            obs.data['llvis'] *= g1L*np.conj(g2L)
            obs.data['rlvis'] *= g1R*np.conj(g2L)
            obs.data['lrvis'] *= g1L*np.conj(g2R)

        # store things differently depending on whether opacity is assumed to be calibrated or not
        if opacitycal:

            # determine baseline thermal noise levels
            tint = obs.data['tint']
            sigma = np.sqrt((SEFD1*SEFD2*np.exp(tau1)*np.exp(tau2))/(2.0*bw*tint)) / const.quant_eff

        else:

            # apply opacity attenuation
            obs.data['rrvis'] *= np.sqrt(np.exp(-tau1)*np.exp(-tau2))
            obs.data['llvis'] *= np.sqrt(np.exp(-tau1)*np.exp(-tau2))
            obs.data['rlvis'] *= np.sqrt(np.exp(-tau1)*np.exp(-tau2))
            obs.data['lrvis'] *= np.sqrt(np.exp(-tau1)*np.exp(-tau2))

            # determine baseline thermal noise levels
            tint = obs.data['tint']
            sigma = np.sqrt((SEFD1*SEFD2)/(2.0*bw*tint)) / const.quant_eff

        # specify baseline thermal noise levels
        obs.data['rrsigma'] = sigma
        obs.data['llsigma'] = sigma
        obs.data['rlsigma'] = sigma
        obs.data['lrsigma'] = sigma

        # apply gains
        if addgains:
            obs.data['rrsigma'] *= np.abs(g1R*np.conj(g2R))
            obs.data['llsigma'] *= np.abs(g1L*np.conj(g2L))
            obs.data['rlsigma'] *= np.abs(g1R*np.conj(g2L))
            obs.data['lrsigma'] *= np.abs(g1L*np.conj(g2R))

        # add thermal noise to observations
        if addnoise:
            obs.data['rrvis'] += sigma*(self.rng.normal(0.0, 1.0, len(obs.data['rrsigma'])) + ((1.0j)*self.rng.normal(0.0, 1.0, len(obs.data['rrsigma']))))
            obs.data['llvis'] += sigma*(self.rng.normal(0.0, 1.0, len(obs.data['llsigma'])) + ((1.0j)*self.rng.normal(0.0, 1.0, len(obs.data['llsigma']))))
            obs.data['rlvis'] += sigma*(self.rng.normal(0.0, 1.0, len(obs.data['rlsigma'])) + ((1.0j)*self.rng.normal(0.0, 1.0, len(obs.data['rlsigma']))))
            obs.data['lrvis'] += sigma*(self.rng.normal(0.0, 1.0, len(obs.data['lrsigma'])) + ((1.0j)*self.rng.normal(0.0, 1.0, len(obs.data['lrsigma']))))

        # create mask and populate it with the sites that should not be flagged
        t1_list = obs.unpack('t1')['t1']
        t2_list = obs.unpack('t2')['t2']
        mask = np.array([t1_list[j] not in flagsites and t2_list[j] not in flagsites for j in range(len(t1_list))])

        # add the daytime flags
        mask &= uptime_mask

        # apply the flags to the observation
        data_copy = obs.data.copy()
        obs.data = data_copy[mask]
        if self.verbosity > 0:
            print('Flagged '+str(len(mask) - mask.sum())+' of '+str(len(mask))+' data points because of wind.')

        # restore Stokes polrep
        obs = obs.switch_polrep(polrep_out='stokes')

        # store additional info if requested
        if self.weight > 0:
            self.timestamps = times[mask]
            self.ant1 = t1_list[mask]
            self.ant2 = t2_list[mask]
            self.bandwidths = bw[mask]
            self.Tsys1 = Tsys1[mask]
            self.Tsys2 = Tsys2[mask]
            self.tau1 = tau1[mask]
            self.tau2 = tau2[mask]
            self.Tb1 = Tb1[mask]
            self.Tb2 = Tb2[mask]
            if opacitycal:
                self.SEFD1 = SEFD1[mask]*np.exp(tau1[mask])
                self.SEFD2 = SEFD2[mask]*np.exp(tau2[mask])
            else:
                self.SEFD1 = SEFD1[mask]
                self.SEFD2 = SEFD2[mask]
            if addgains:
                self.station_gains1R = self.station_gains1R[mask]
                self.station_gains2R = self.station_gains2R[mask]
                self.station_gains1L = self.station_gains1L[mask]
                self.station_gains2L = self.station_gains2L[mask]

        # return observation object
        return obs

    # generate observation
    def make_obs(self, input_model=None, addnoise=True, addgains=True, gainamp=0.04, opacitycal=True,
                 addFR=False, flagwind=True, flagday=False, allow_mixed_basis=False,
                 el_min=const.el_min, el_max=const.el_max, p=None):
        """
        Generate an observation that folds in weather-based opacity effects
        and applies a specified SNR thresholding scheme to mimic fringe-finding.

        Args:
          input_model (ehtim.image.Image, ehtim.movie.Movie, ehtim.model.Model, ngEHTforecast.fisher.fisher_forecast.FisherForecast): input source model
          addnoise (bool): flag for whether or not to add thermal noise to the visibilities
          addgains (bool): flag for whether or not to add station gain corruptions
          gainamp (float): standard deviation of amplitude log-gains
          opacitycal (bool): flag for whether or not to assume that atmospheric opacity is assumed to be calibrated out
          flagwind (bool): flag for whether to derate sites with high wind
          flagday (bool): flag for whether to flag sites during the local daytime
          addFR (bool): flag for whether or not to add feed rotations
          allow_mixed_basis (bool): flag for whether to apply polarization basis conversions
          el_min (float): minimum elevation that a site can observe at, in degrees
          el_max (float): maximum elevation that a site can observe at, in degrees
          p (numpy.ndarray): list of parameters for an input ngEHTforecast.fisher.fisher_forecast.FisherForecast object

        Returns:
          (ehtim.obsdata.Obsdata): eht-imaging Obsdata object containing the generated observation
        """

        # determine SNR thresholding scheme and values
        snr_algo, snr_args = self.settings['fringe_finder']

        # retrieve stored input_model if it has been set to None
        if input_model is None:
            input_model = self.im
            if self.im is None:
                raise Exception('If there is no input model specified in the settings, then make_obs must specify one!')
            else:
                if self.verbosity > 0:
                    print('No input model passed to make_obs; using the model provided in the settings.')

        # generate raw observation
        obs = self.observe(input_model,
                           addnoise=addnoise,
                           addgains=addgains,
                           gainamp=gainamp,
                           opacitycal=opacitycal,
                           flagwind=flagwind,
                           flagday=flagday,
                           addFR=addFR,
                           allow_mixed_basis=allow_mixed_basis,
                           el_min=el_min,
                           el_max=el_max,
                           p=p)

        # apply naive SNR thresholding
        if (snr_algo.lower() == 'naive'):
            mask = obs.unpack('snr')['snr'] > snr_args

        # apply a proxy for the "fringegroups" procedure from HOPS
        elif (snr_algo.lower() == 'fringegroups'):

            # parse fringe_finder arguments
            snr_ref = snr_args[0]
            tint_ref = snr_args[1]

            mask = fringegroups(self, obs, snr_ref, tint_ref)

        # apply an FPT proxy for SNR thresholding
        elif (snr_algo.lower() == 'fpt'):

            # parse fringe_finder arguments
            snr_ref = snr_args[0]
            tint_ref = snr_args[1]
            freq_ref = snr_args[2]
            model_path_ref = snr_args[3]

            mask = FPT(self, obs, snr_ref, tint_ref, freq_ref, model_path_ref, ephem=self.ephem, addnoise=addnoise, addgains=addgains, gainamp=gainamp, opacitycal=opacitycal, flagwind=flagwind, flagday=flagday, addFR=addFR, el_min=el_min, el_max=el_max, p=p)

        # unrecognized SNR thresholding scheme
        else:
            raise ValueError('Unknown algorithm for fringe_finder.')

        # apply the data flags to the observation
        data_copy = obs.data.copy()
        obs.data = data_copy[mask]
        if self.verbosity > 0:
            print('Flagged '+str(len(mask) - mask.sum())+' of '+str(len(mask))+' data points during fringe-finding emulation.')

        # flag the additional stored quantities as well
        if self.weight > 0:
            self.timestamps = self.timestamps[mask]
            self.ant1 = self.ant1[mask]
            self.ant2 = self.ant2[mask]
            self.bandwidths = self.bandwidths[mask]
            self.Tsys1 = self.Tsys1[mask]
            self.Tsys2 = self.Tsys2[mask]
            self.tau1 = self.tau1[mask]
            self.tau2 = self.tau2[mask]
            self.Tb1 = self.Tb1[mask]
            self.Tb2 = self.Tb2[mask]
            self.SEFD1 = self.SEFD1[mask]
            self.SEFD2 = self.SEFD2[mask]
            if addgains:
                self.station_gains1R = self.station_gains1R[mask]
                self.station_gains2R = self.station_gains2R[mask]
                self.station_gains1L = self.station_gains1L[mask]
                self.station_gains2L = self.station_gains2L[mask]

        # remove sites that can't observe at the requested frequency
        sites_to_remove = list()
        for site in obs.tarr['site']:
            if self.bands[site] is None:
                sites_to_remove.append(site)
                if self.verbosity > 0:
                    print(site + ' cannot observe at '+str(self.freq/(1.0e9))+' GHz.')
        if len(sites_to_remove) > 0:
            if len(obs.data) > 0:
                t1_list = obs.unpack('t1')['t1']
                t2_list = obs.unpack('t2')['t2']
                mask = np.array([t1_list[j] not in sites_to_remove and t2_list[j] not in sites_to_remove for j in range(len(t1_list))])

                # apply the data flags to the observation
                data_copy = obs.data.copy()
                obs.data = data_copy[mask]
                if self.verbosity > 0:
                    print('Flagged '+str(len(mask) - mask.sum())+' of '+str(len(mask))+' data points because of no appropriate receiver at the observing frequency.')

                # flag the additional stored quantities as well
                if self.weight > 0:
                    self.timestamps = self.timestamps[mask]
                    self.ant1 = self.ant1[mask]
                    self.ant2 = self.ant2[mask]
                    self.bandwidths = self.bandwidths[mask]
                    self.Tsys1 = self.Tsys1[mask]
                    self.Tsys2 = self.Tsys2[mask]
                    self.tau1 = self.tau1[mask]
                    self.tau2 = self.tau2[mask]
                    self.Tb1 = self.Tb1[mask]
                    self.Tb2 = self.Tb2[mask]
                    self.SEFD1 = self.SEFD1[mask]
                    self.SEFD2 = self.SEFD2[mask]
                    if addgains:
                        self.station_gains1R = self.station_gains1R[mask]
                        self.station_gains2R = self.station_gains2R[mask]
                        self.station_gains1L = self.station_gains1L[mask]
                        self.station_gains2L = self.station_gains2L[mask]

        # drop any sites that are randomly deemed to be technically unready
        sites_to_remove = get_unready_sites(obs.tarr['site'], self.settings['tech_readiness'], rng=self.rng)
        if len(sites_to_remove) > 0:
            if self.verbosity > 0:
                print("Dropping {0} due to technical (un)readiness.".format(sites_to_remove))
            if len(obs.data) > 0:
                t1_list = obs.unpack('t1')['t1']
                t2_list = obs.unpack('t2')['t2']
                mask = np.array([t1_list[j] not in sites_to_remove and t2_list[j] not in sites_to_remove for j in range(len(t1_list))])

                # apply the data flags to the observation
                data_copy = obs.data.copy()
                obs.data = data_copy[mask]
                if self.verbosity > 0:
                    print('Flagged '+str(len(mask) - mask.sum())+' of '+str(len(mask))+' data points because of techincal unreadiness.')

                # flag the additional stored quantities as well
                if self.weight > 0:
                    self.timestamps = self.timestamps[mask]
                    self.ant1 = self.ant1[mask]
                    self.ant2 = self.ant2[mask]
                    self.bandwidths = self.bandwidths[mask]
                    self.Tsys1 = self.Tsys1[mask]
                    self.Tsys2 = self.Tsys2[mask]
                    self.tau1 = self.tau1[mask]
                    self.tau2 = self.tau2[mask]
                    self.Tb1 = self.Tb1[mask]
                    self.Tb2 = self.Tb2[mask]
                    self.SEFD1 = self.SEFD1[mask]
                    self.SEFD2 = self.SEFD2[mask]
                    if addgains:
                        self.station_gains1R = self.station_gains1R[mask]
                        self.station_gains2R = self.station_gains2R[mask]
                        self.station_gains1L = self.station_gains1L[mask]
                        self.station_gains2L = self.station_gains2L[mask]

        # return observation object
        return obs

    # generate multifrequency observation, assuming that FPT will be used wherever possible
    def make_obs_mf(self, freqs, input_models, addnoise=True, addgains=True, gainamp=0.04, opacitycal=True,
                    addFR=False, el_min=const.el_min, el_max=const.el_max, flagwind=True, flagday=False, p=None):
        """
        Generate a multi-frequency observation

        Args:
          freqs (list): list of frequencies at which to carry out the observation, in GHz
          input_models (list): list of input source models; one for each frequency
          addnoise (bool): flag for whether or not to add thermal noise to the visibilities
          addgains (bool): flag for whether or not to add station gain corruptions
          gainamp (float): standard deviation of amplitude log-gains
          opacitycal (bool): flag for whether or not to assume that atmospheric opacity is assumed to be calibrated out
          addFR (bool): flag for whether or not to add feed rotations
          el_min (float): minimum elevation that a site can observe at, in degrees
          el_max (float): maximum elevation that a site can observe at, in degrees
          flagwind (bool): flag for whether to derate sites with high wind
          flagday (bool): flag for whether to flag sites during the local daytime
          p (list): list of lists of parameters for input ngEHTforecast.fisher.fisher_forecast.FisherForecast objects; one for each frequency

        Returns:
          (list): list of ehtim.obsdata.Obsdata objects containing the generated observations; one for each frequency
        """

        #################################
        # initial checks and fixes

        if len(freqs) < 2:
            raise Exception('Please provide at least 2 frequencies for a multi-frequency observation.')

        if p is None:
            p = [list()]*len(freqs)

        if (len(input_models) != len(freqs)):
            raise Exception('The number of input models must match the number of frequencies.')
        if (len(p) != len(freqs)):
            raise Exception('The number of lists of FisherForecast parameters must match the number of frequencies; if some input models are not FisherForecast objects, then the corresponding elements of the list may be empty.')

        #################################
        # estimate coherence times

        tcoh_230 = 10.0
        tcohs = list()
        for freq in freqs:
            tcoh = tcoh_230/(freq/230.0)
            tcohs.append(tcoh)

        #################################
        # loop through all frequency pairs
        # index i denotes the "currently observed" frequency
        # index j denotes the frequency being tried for FPT

        obslist = list()

        for ifreq, freq_target in enumerate(freqs):

            # retrieve the model for the target frequency
            model_target = input_models[ifreq]
            p_target = p[ifreq]

            # keep track of whether it's the first reference or not
            count = 0

            for jfreq, freq_ref in enumerate(freqs):
                if (jfreq == ifreq):
                    continue

                # retrieve the model for the reference frequency
                model_ref = input_models[jfreq]
                p_ref = p[jfreq]

                # determine the coherence time to use
                tcoh_here = np.min([tcohs[ifreq], tcohs[jfreq]])

                # determine the SNR to use
                SNR_here = np.max([5.0, 5.0*(freq_target/freq_ref)])

                # initialize the settings for a dummy obsgen object
                settings = copy.deepcopy(self.settings)
                settings['frequency'] = freq_target
                settings['fringe_finder'] = ['fpt', [SNR_here, tcoh_here, freq_ref, model_ref]]
                settings['random_seed'] = self.seed
                if ((model_target is None) | isinstance(model_target, str)):
                    settings['model_file'] = model_target
                if ((self.weather == 'random') | (self.weather == 'exact')):
                    settings['weather'] = 'exact'
                    settings['weather_year'] = str(self.weather_year)
                    settings['weather_day'] = str(self.weather_day)

                # create dummy obsgen object
                obsgen_here = obs_generator(settings=copy.deepcopy(settings),
                                            verbosity=self.verbosity,
                                            weight=self.weight,
                                            D_overrides=copy.deepcopy(self.D_overrides),
                                            surf_rms_overrides=copy.deepcopy(self.surf_rms_overrides),
                                            receiver_configuration_overrides=copy.deepcopy(self.receiver_configuration_overrides),
                                            bandwidth_overrides=copy.deepcopy(self.bandwidth_overrides),
                                            T_R_overrides=copy.deepcopy(self.T_R_overrides),
                                            sideband_ratio_overrides=copy.deepcopy(self.sideband_ratio_overrides),
                                            lo_freq_overrides=copy.deepcopy(self.lo_freq_overrides),
                                            hi_freq_overrides=copy.deepcopy(self.hi_freq_overrides),
                                            ap_eff_overrides=copy.deepcopy(self.ap_eff_overrides),
                                            custom_receivers=copy.deepcopy(self.custom_receivers),
                                            station_uptimes=copy.deepcopy(self.station_uptimes),
                                            array=self.array,
                                            ephem=self.ephem)

                if ((model_target is not None) & (not isinstance(model_target, str))):
                    obsgen_here.im = model_target

                # generate observation at target frequency
                obs_here = obsgen_here.make_obs(input_model=obsgen_here.im, addnoise=addnoise, addgains=addgains, gainamp=gainamp, opacitycal=opacitycal, addFR=addFR, el_min=el_min, el_max=el_max, flagwind=flagwind, flagday=flagday, p=p_target)

                # add any new detections to the running datatable
                if count == 0:
                    datatable_init = obs_here.data.copy()
                    t1 = obs_here.data['time']
                    t11 = obs_here.data['t1']
                    t21 = obs_here.data['t2']
                else:
                    t2 = obs_here.data['time']
                    t12 = obs_here.data['t1']
                    t22 = obs_here.data['t2']

                    for ii in range(len(t2)):
                        ind = ((t1 == t2[ii]) & (t11 == t12[ii]) & (t21 == t22[ii]))
                        if ind.sum() == 0:
                            datatable_init = np.append(datatable_init, obs_here.data[ii])

                # update the obsdata object
                obs = obs_here.copy()
                obs.datatable = datatable_init
                obs.data = datatable_init

                count += 1

            # add to the list
            obslist.append(obs)

        return obslist

    ###################################################
    # other functions

    def export_SYMBA(self, symba_workdir='./data',
                     output_filenames=['obsgen.antennas', 'master_input.txt'],
                     t_coh=10.0, RMS_point=0.0, PB_model='gaussian', use_two_letter=True,
                     gain_mean=1.0, leak_mean=0.0j, master_input_args={}, master_input_comments={}):
        """
        Export SYMBA-compatible directory structure and input files from the obs_generator object.

        Args:
          symba_workdir (str): name of SYMBA working directory to use or create
          output_filenames (list): names of .antennas and master_input.txt files to save
          t_coh (float): default coherence time, in seconds
          RMS_point (float): default RMS pointing uncertainty, in arcseconds
          PB_model (str): primary beam model to use; only option right now is 'gaussian'
          use_two_letter (bool): convert all station names to two-letter codes
          gain_mean (float, complex, dict): Value of the mean gain offset for each station.
                                           If float or complex, will apply to all stations;
                                           if a dict, should be indexed by station name
          leak_mean (float, complex, dict): Value of the mean leakage offset for each station.
                                            If float or complex, will apply to all stations;
                                            if a dict, should be indexed by station name
          master_input_args (dict): dictionary of master input arguments
          master_input_comments (dict): dictionary of comments associated with master input arguments

        Returns:
          SYMBA-compatible .antennas and master_input.txt files
        """

        # create SYMBA working directory
        os.makedirs(symba_workdir, exist_ok=True)

        # create input and output folders within the working directory
        inpdir = symba_workdir + '/symba_input'
        outdir = symba_workdir + '/symba_output'
        os.makedirs(inpdir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)

        # modify filenames appropriately
        for i in range(len(output_filenames)):
            output_filenames[i] = inpdir + '/' + output_filenames[i]

        # export .antennas file
        export_SYMBA_antennas(self,
                              output_filename=output_filenames[0],
                              t_coh=t_coh,
                              RMS_point=RMS_point,
                              PB_model=PB_model,
                              use_two_letter=use_two_letter,
                              gain_mean=gain_mean,
                              leak_mean=leak_mean)

        # export master_input.txt file
        if 'outdirname' not in list(master_input_args.keys()):
            master_input_args.update({'outdirname': outdir})
        if 'ms_antenna_table' not in list(master_input_args.keys()):
            master_input_args.update({'ms_antenna_table': output_filenames[0]})
        if 'input_fitsimage' not in list(master_input_args.keys()):
            master_input_args.update({'input_fitsimage': inpdir + '/*.fits'})
        export_SYMBA_master_input(self,
                                  input_args=master_input_args,
                                  input_comments=master_input_comments,
                                  output_filename=output_filenames[1],
                                  use_two_letter=use_two_letter)


###################################################
# other functions


def get_station_list():
    """
    Return a list of known stations; "get_station_list" and "get_site_list" are equivalent

    Returns:
      (list): a list of station names
    """

    return list(const.known_stations)


# alias for get_station_list
get_site_list = get_station_list


def determine_mjd(day, month, year):
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
            raise Exception('January has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-01-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Feb'):
        if int(day) > 28:
            try:
                t = Time(str(year)+'-02-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
            except:
                raise Exception('February has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-02-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Mar'):
        if int(day) > 31:
            raise Exception('March has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-03-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Apr'):
        if int(day) > 30:
            raise Exception('April has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-04-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    elif (month == 'May'):
        if int(day) > 31:
            raise Exception('May has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-05-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Jun'):
        if int(day) > 30:
            raise Exception('June has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-06-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Jul'):
        if int(day) > 31:
            raise Exception('July has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-07-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Aug'):
        if int(day) > 31:
            raise Exception('August has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-08-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Sep'):
        if int(day) > 30:
            raise Exception('September has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-09-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Oct'):
        if int(day) > 31:
            raise Exception('October has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-10-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Nov'):
        if int(day) > 30:
            raise Exception('November has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-11-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    elif (month == 'Dec'):
        if int(day) > 31:
            raise Exception('December has fewer than ' + str(day).zfill(2) + ' days!')
        t = Time(str(year)+'-12-'+str(day).zfill(2)+'T00:00:00', format='isot', scale='utc')
    else:
        raise Exception('This month abbreviation is not recognized; should be one of: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec')

    return t.mjd


def make_array(sitelist, ephem='ephemeris/space', verbosity=0):
    """
    Create an ehtim array object from a list of sites.

    Args:
      sitelist (list): A list of site names
      ephem (str): path to the ephemeris for a space station
      verbosity (float): Set to >0 for more verbose output

    Returns:
      (ehtim.array.Array): An ehtim array object
    """

    if 'space' not in sitelist:

        tarr = np.recarray(len(sitelist), dtype=eh.const_def.DTARR)

        for isite, site in enumerate(sitelist):

            lon = const.known_longitudes[site]
            lat = const.known_latitudes[site]
            elev = const.known_elevations[site]

            earthloc = EarthLocation.from_geodetic(lon, lat, elev)
            x = earthloc.x.value
            y = earthloc.y.value
            z = earthloc.z.value

            fr_par = 1.0
            fr_elev = 0.0
            if site in list(const.known_mount_types.keys()):
                mthere = const.known_mount_types[site]
                if 'NASMYTH-R' in mthere:
                    fr_elev = 1.0
                elif 'NASMYTH-L' in mthere:
                    fr_elev = -1.0
            fr_off = 0.0
            if site in list(const.known_feed_angles.keys()):
                fr_off = const.known_feed_angles[site]

            tarr[isite]['site'] = site
            tarr[isite]['x'] = x
            tarr[isite]['y'] = y
            tarr[isite]['z'] = z
            tarr[isite]['sefdr'] = 10000.0
            tarr[isite]['sefdl'] = 10000.0
            tarr[isite]['dr'] = 0.0 + 0.0j
            tarr[isite]['dl'] = 0.0 + 0.0j
            tarr[isite]['fr_par'] = fr_par
            tarr[isite]['fr_elev'] = fr_elev
            tarr[isite]['fr_off'] = fr_off

        arr = eh.array.Array(tarr)

    else:

        sitelist2 = sitelist.copy()
        sitelist2.remove('space')
        tarr = np.recarray(len(sitelist2), dtype=eh.const_def.DTARR)

        # first add the non-space dishes
        for isite, site in enumerate(sitelist2):

            lon = const.known_longitudes[site]
            lat = const.known_latitudes[site]
            elev = const.known_elevations[site]

            earthloc = EarthLocation.from_geodetic(lon, lat, elev)
            x = earthloc.x.value
            y = earthloc.y.value
            z = earthloc.z.value

            fr_par = 1.0
            fr_elev = 0.0
            if site in list(const.known_mount_types.keys()):
                mthere = const.known_mount_types[site]
                if 'NASMYTH-R' in mthere:
                    fr_elev = 1.0
                elif 'NASMYTH-L' in mthere:
                    fr_elev = -1.0
            fr_off = 0.0
            if site in list(const.known_feed_angles.keys()):
                fr_off = const.known_feed_angles[site]

            tarr[isite]['site'] = site
            tarr[isite]['x'] = x
            tarr[isite]['y'] = y
            tarr[isite]['z'] = z
            tarr[isite]['sefdr'] = 10000.0
            tarr[isite]['sefdl'] = 10000.0
            tarr[isite]['dr'] = 0.0 + 0.0j
            tarr[isite]['dl'] = 0.0 + 0.0j
            tarr[isite]['fr_par'] = fr_par
            tarr[isite]['fr_elev'] = fr_elev
            tarr[isite]['fr_off'] = fr_off

        arr = eh.array.Array(tarr)

        # then add the space dish
        space_entry = ('space', 0., 0., 0., 10000., 10000., 0.+0.j, 0.+0.j, 1., 0., 0.)
        arr_templist = list()
        for i in range(len(arr.tarr)):
            arr_templist.append(arr.tarr[i])
        arr_templist.append(space_entry)
        arr.tarr = np.array(arr_templist, dtype=eh.const_def.DTARR)
        arr.tkey['space'] = len(sitelist)-1

        # load the ephemeris
        edata = {}
        sitename = 'space'
        try:
            edata[sitename] = np.loadtxt(ephem, dtype=bytes,
                                         comments='#', delimiter='/').astype(str)
            if (verbosity > 0):
                print('Loaded spacecraft ephemeris %s' % ephem)
        except IOError:
            raise Exception('No ephemeris file %s !' % ephem)

        # add the ephemeris to the array object
        arr.ephem = edata

    return arr


def load_image(infile, freq=230.0e9, verbosity=0):
    """
    Load an ehtim image or movie object.

    Args:
      infile (str): The input path and filename
      freq (float): Observing frequency, in Hz
      verbosity (float): Set to >0 for more verbose output

    Returns:
      (ehtim.image.Image, ehtim.movie.Movie): An ehtim image or movie object; returns None if infile is None
    """

    if infile is None:
        return None

    else:
        if verbosity <= 0:
            with eh.parloop.HiddenPrints():
                try:
                    im = eh.image.load_image(infile)
                    im.rf = float(np.round(im.rf))
                except:
                    if verbosity > 0:
                        print('Source file does not appear to be an image; assuming that it is a movie file instead.')
                    extension = infile.split('.')[-1]
                    if extension.lower() in ['hdf5', 'h5']:
                        im = eh.movie.load_hdf5(infile)
                    elif extension.lower() == ['fits']:
                        im = eh.movie.load_fits(infile)
                    elif extension.lower() == ['txt']:
                        im = eh.movie.load_txt(infile)
                    else:
                        raise Exception('Source file does not have a recognized file extension.')
                    im.rf = freq
                return im
        elif verbosity > 0:
            try:
                im = eh.image.load_image(infile)
                im.rf = float(np.round(im.rf))
            except:
                if verbosity > 0:
                    print('Source file does not appear to be an image; assuming that it is a movie file instead.')
                extension = infile.split('.')[-1]
                if extension.lower() in ['hdf5', 'h5']:
                    im = eh.movie.load_hdf5(infile)
                elif extension.lower() == ['fits']:
                    im = eh.movie.load_fits(infile)
                elif extension.lower() == ['txt']:
                    im = eh.movie.load_txt(infile)
                else:
                    raise Exception('Source file does not have a recognized file extension.')
                im.rf = freq
            return im


def eta_dish(freq, sigma, offset, ap_eff):
    """
    Function for computing overall antenna aperture efficiency.

    Args:
      freq (float): observing frequency, in Hz
      sigma (float): surface RMS, in microns
      offset (float): focus offset, in equivalent microns of surface RMS
      ap_eff (float): nominal aperture efficiency

    Returns:
      (float): overall aperture efficiency
    """

    # Ruze's law for surface + focus
    etahere = np.exp(-((4*np.pi*np.sqrt((sigma)**2+(offset)**2))/((const.c*(1.0e6))/freq))**2)

    # additional aperture inefficiency
    etahere *= ap_eff

    return etahere


def get_unready_sites(sites, tech_readiness, rng=np.random.default_rng()):
    """
    Function to determine which sites will randomly fail technical readiness.

    Args:
      sites (list): list of sites participating in the observation
      tech_readiness (float): probability of any individual site being technically ready to observe;
                              takes on a value between 0 and 1
      rng (numpy.random.Generator): a numpy random number generator

    Returns:
      (list): sites to drop
    """

    if (tech_readiness > 1.0) | (tech_readiness < 0.0):
        raise Exception('The tech_readiness keyword must take on a value between 0 and 1!')

    p = tech_readiness
    index = rng.choice([0, 1], size=(len(sites)), p=[p, 1-p]).astype(bool)
    sites_to_drop = sites[index]
    return sites_to_drop


def windspeed_SEFD_modification(windspeed, windspeed_degradation=const.windspeed_degradation,
                                windspeed_shutdown=const.windspeed_shutdown):
    """
    Function to convert a windspeed to an effective SEFD scaling factor.

    Args:
      windspeed (float): windspeed value, in m/s
      windspeed_degradation (float): windspeed value at which to start substantially degrading performance
      windspeed_shutdown (float): windspeed value at which a site must be shut down

    Returns:
      (float): factor by which to scale the SEFD
    """

    centerpoint = 0.5*(windspeed_degradation + windspeed_shutdown)
    rate = (windspeed_shutdown - windspeed_degradation) / 10.0
    scale_factor = 1.0 - (1.0 / (1.0 + np.exp(-5.0*((windspeed / (2.0*(windspeed_shutdown - windspeed_degradation))) - 1.0))))
    
    return 1.0/scale_factor


def fringegroups(obsgen, obs, snr_ref, tint_ref):
    """
    Function to apply the "fringegroups" SNR thresholding scheme to an observation.
    This scheme attempts to mimic the fringe-fitting carried out in the HOPS calibration pipeline.

    Args:
      obsgen (ngehtsim.obs.obs_generator.obs_generator): ngehtsim obs_generator object containing information about the observation
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object containing the input observation
      snr_ref (float): strong baseline SNR threshold
      tint_ref (float): strong baseline coherence time, in seconds

    Returns:
      (numpy.ndarray): An array of kept data indices
    """

    # get the timestamps
    time = obs.data['time']
    timestamps = np.unique(time)

    # get the stations that are able to observe at this frequency
    available_sites = list()
    for site in obsgen.sites:
        if obsgen.bands[site] is not None:
            available_sites.append(site)

    # create a running index list of baselines to flag
    master_index = np.zeros(len(obs.data), dtype='bool')
    count = 0

    # create blank dummy obsdata objects
    obs = obs.switch_polrep(polrep_out='circ')
    obs_here = copy.deepcopy(obs)
    obs_here.data = None
    obs_search = obs_here.copy()

    # check all timestamps
    for itime, timestamp in enumerate(timestamps):

        ind_t = (time == timestamp)
        obs_here.data = obs.data[ind_t]

        # scale effective SNR to the actual integration time
        snr_scaled = snr_ref*np.sqrt(obs_here.data['tint'] / tint_ref)

        # determine which baselines are "strong"
        pseudo_I_amp = 0.5*(np.abs(obs_here.data['rrvis']) + np.abs(obs_here.data['llvis']))
        pseudo_I_sig = obs_here.data['rrsigma'] / np.sqrt(2.0)
        index = (pseudo_I_amp/pseudo_I_sig) >= snr_scaled

        # determine which sites are available to observe at this frequency
        t1_available = np.isin(obs_here.data['t1'], available_sites)
        t2_available = np.isin(obs_here.data['t2'], available_sites)
        index &= t1_available
        index &= t2_available

        # limit the searched baselines to those that are strong and available
        obs_search.data = obs_here.data[index]

        # group stations that are connected by strong baselines
        groups = list()
        for datum in obs_search.data:
            bl = [datum['t1'], datum['t2']]
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
            if ((datum['t1'] in list(site_dict.keys())) & (datum['t2'] in list(site_dict.keys()))):
                if (site_dict[datum['t1']] == site_dict[datum['t2']]):
                    master_index[count] = True
            count += 1

    return master_index


def FPT(obsgen, obs, snr_ref, tint_ref, freq_ref, model_ref=None, ephem='ephemeris/space', **kwargs):
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
      (numpy.ndarray): An array of kept data indices
    """

    # determine settings for dummy obsgen object
    new_settings = copy.deepcopy(obsgen.settings)
    new_settings['frequency'] = freq_ref
    new_settings['bandwidth'] = obsgen.settings['bandwidth']
    new_settings['fringe_finder'] = ['fringegroups', [snr_ref, tint_ref]]
    new_settings['random_seed'] = obsgen.seed
    if ((model_ref is None) | isinstance(model_ref, str)):
        new_settings['model_file'] = model_ref
    if ((obsgen.weather == 'random') | (obsgen.weather == 'exact')):
        new_settings['weather'] = 'exact'
        new_settings['weather_year'] = str(obsgen.weather_year)
        new_settings['weather_day'] = str(obsgen.weather_day)
    new_D_overrides = copy.deepcopy(obsgen.D_overrides)
    new_surf_rms_overrides = copy.deepcopy(obsgen.surf_rms_overrides)
    new_receiver_configuration_overrides = copy.deepcopy(obsgen.receiver_configuration_overrides)
    new_bandwidth_overrides = copy.deepcopy(obsgen.bandwidth_overrides)
    new_T_R_overrides = copy.deepcopy(obsgen.T_R_overrides)
    new_sideband_ratio_overrides = copy.deepcopy(obsgen.sideband_ratio_overrides)
    new_lo_freq_overrides = copy.deepcopy(obsgen.lo_freq_overrides)
    new_hi_freq_overrides = copy.deepcopy(obsgen.hi_freq_overrides)
    new_ap_eff_overrides = copy.deepcopy(obsgen.ap_eff_overrides)
    new_custom_receivers = copy.deepcopy(obsgen.custom_receivers)
    new_station_uptimes = copy.deepcopy(obsgen.station_uptimes)

    # create dummy obsgen object
    obsgen_ref = obs_generator(new_settings,
                               D_overrides=new_D_overrides,
                               receiver_configuration_overrides=new_receiver_configuration_overrides,
                               surf_rms_overrides=new_surf_rms_overrides,
                               bandwidth_overrides=new_bandwidth_overrides,
                               T_R_overrides=new_T_R_overrides,
                               sideband_ratio_overrides=new_sideband_ratio_overrides,
                               lo_freq_overrides=new_lo_freq_overrides,
                               hi_freq_overrides=new_hi_freq_overrides,
                               ap_eff_overrides=new_ap_eff_overrides,
                               custom_receivers=new_custom_receivers,
                               station_uptimes=new_station_uptimes,
                               ephem=ephem)
    if ((model_ref is not None) & (not isinstance(model_ref, str))):
        obsgen_ref.im = model_ref

    # generate observation at reference frequency
    obs_ref = obsgen_ref.observe(obsgen_ref.im, **kwargs)

    # create a running index list of baselines to flag
    master_index = np.zeros(len(obs_ref.data), dtype='bool')

    # get detections from the reference frequency
    fringegroups_index = fringegroups(obsgen_ref, obs_ref, snr_ref, tint_ref)
    master_index |= fringegroups_index

    # get any additional detections from normal fringe-fitting
    snr_fringegroups = snr_ref * (freq_ref/(obsgen.freq/(1.0e9)))
    fringegroups_index = fringegroups(obsgen, obs, snr_fringegroups, tint_ref)
    master_index |= fringegroups_index

    return master_index


def export_SYMBA_antennas(obsgen, output_filename='obsgen.antennas', t_coh=10.0, RMS_point=1.0,
                          PB_model='gaussian', use_two_letter=True, gain_mean=1.0, leak_mean=0.0j):
    """
    Export a SYMBA-compatible .antennas file from the obs_generator object.

    Args:
      obsgen (ngehtsim.obs.obs_generator.obs_generator): ngehtsim obs_generator object containing information about the observation
      output_filename (str): name of .antennas file to save
      t_coh (float): default coherence time, in seconds
      RMS_point (float): default RMS pointing uncertainty, in arcseconds
      PB_model (str): primary beam model to use; only option right now is 'gaussian'
      use_two_letter (bool): convert all station names to two-letter codes
      gain_mean (float, complex, dict): Value of the mean gain offset for each station.
                                       If float or complex, will apply to all stations;
                                       if a dict, should be indexed by station name
      leak_mean (float, complex, dict): Value of the mean leakage offset for each station.
                                        If float or complex, will apply to all stations;
                                        if a dict, should be indexed by station name

    Returns:
      SYMBA-compatible .antennas file containing the observation information
    """

    with open(output_filename, 'w') as outfile:

        # add file header
        header = 'station'.ljust(9)
        header += 'T_rx[K]'.ljust(11)
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

        # determine form of weather return
        if ((obsgen.weather == 'random') | (obsgen.weather == 'exact')):
            form = 'exact'
        elif ((obsgen.weather == 'mean') | (obsgen.weather == 'average')):
            form = 'mean'
        elif ((obsgen.weather == 'typical') | (obsgen.weather == 'median')):
            form = 'median'
        elif (obsgen.weather == 'good'):
            form = 'good'
        elif ((obsgen.weather == 'bad') | (obsgen.weather == 'poor')):
            form = 'bad'

        for site in obsgen.sites:

            band = obsgen.bands[site]

            if band is not None:

                # initialize empty string
                strhere = ''

                # add station name as a two-letter code
                if use_two_letter:
                    strhere += const.two_letter_station_codes[site].ljust(9)
                else:
                    strhere += site.ljust(9)

                # add receiver temperature, in K
                strhere += str(np.round(obsgen.receivers[site][band]['T_R'], 2)).ljust(11)

                # add PWV, in mm
                PWV = nw.PWV(site, form=form, month=obsgen.settings['month'], day=obsgen.weather_day, year=obsgen.weather_year)
                strhere += str(np.round(PWV, 4)).ljust(9)

                # add surface pressure, in mbar
                pres = nw.pressure(site, form=form, month=obsgen.settings['month'], day=obsgen.weather_day, year=obsgen.weather_year)
                strhere += str(np.round(pres, 2)).ljust(12)

                # add surface temperature, in K
                temp = nw.temperature(site, form=form, month=obsgen.settings['month'], day=obsgen.weather_day, year=obsgen.weather_year)
                strhere += str(np.round(temp, 2)).ljust(10)

                # add coherence time, in seconds
                strhere += str(np.round(t_coh, 2)).ljust(13)

                # add RMS pointing uncertainty, in seconds
                strhere += str(np.round(RMS_point, 2)).ljust(17)

                # add 230GHz FWHM primary beam size
                diam = obsgen.D_dict[site]
                pb = ((180.0/np.pi)*3600.0)*((const.c / (230.0e9)) / diam)
                strhere += str(np.round(pb, 2)).ljust(20)

                # add the primary beam model
                strhere += PB_model.ljust(12)

                # add the aperture efficiency
                strhere += str(np.round(obsgen.eta_dict[site], 4)).ljust(9)

                # add gain means and stds
                if isinstance(gain_mean, float) or isinstance(gain_mean, complex):
                    gain_here = gain_mean
                elif isinstance(gain_mean, dict):
                    gain_here = gain_mean[site]
                if isinstance(gain_here, complex):
                    gain_str = str(gain_here)[1:-1]
                else:
                    gain_str = str(gain_here)
                strhere += gain_str.ljust(11)
                strhere += str(0.0).ljust(12)
                strhere += gain_str.ljust(11)
                strhere += str(0.0).ljust(12)

                # add leakage means and stds
                if isinstance(leak_mean, float) or isinstance(leak_mean, complex):
                    leak_here = leak_mean
                elif isinstance(leak_mean, dict):
                    leak_here = leak_mean[site]
                if isinstance(leak_here, complex):
                    if (np.sign(np.imag(leak_here)) == 0.0) | (np.sign(np.imag(leak_here)) == 1.0):
                        signhere = '+'
                    else:
                        signhere = '-'
                    leak_str = str(np.real(leak_here)) + signhere + str(np.imag(leak_here)) + 'j'
                else:
                    leak_str = str(leak_here)
                strhere += leak_str.ljust(12)
                strhere += str(0.0).ljust(12)
                strhere += leak_str.ljust(12)
                strhere += str(0.0).ljust(12)

                # add feed angle
                if site in list(const.known_feed_angles.keys()):
                    strhere += str(const.known_feed_angles[site]).ljust(20)
                else:
                    strhere += str(const.feed_angle).ljust(20)

                # add mount type
                if site in list(const.known_mount_types.keys()):
                    strhere += const.known_mount_types[site].ljust(18)
                else:
                    strhere += const.mount_type.ljust(18)

                # add dish diameter
                diam = obsgen.D_dict[site]
                strhere += str(np.round(diam, 2)).ljust(17)

                # add xyz coordinates
                lon = const.known_longitudes[site]
                lat = const.known_latitudes[site]
                elev = const.known_elevations[site]
                earthloc = EarthLocation.from_geodetic(lon, lat, elev)
                x = earthloc.x.value
                y = earthloc.y.value
                z = earthloc.z.value
                strhere += str(np.round(x, 8)) + ','
                strhere += str(np.round(y, 8)) + ','
                strhere += str(np.round(z, 8))

                # write line
                strhere += '\n'
                outfile.write(strhere)


def export_SYMBA_master_input(obsgen, input_args={}, input_comments={}, output_filename='master_input.txt', use_two_letter=True):
    """
    Export a SYMBA-compatible master_input.txt file from the obs_generator object.

    Args:
      obsgen (ngehtsim.obs.obs_generator.obs_generator): ngehtsim obs_generator object containing information about the observation
      input_args (dict): dictionary of input arguments
      input_comments (dict): dictionary of comments associated with input arguments
      output_filename (str): name of master_input.txt file to save
      use_two_letter (bool): convert all station names to two-letter codes

    Returns:
      SYMBA-compatible master_input.txt file containing the observation information
    """

    # load up the default input arguments and comments
    args = copy.deepcopy(const.SYMBA_master_input_arguments)
    comms = copy.deepcopy(const.SYMBA_master_input_comments)

    #########################################################
    # overwrite various defaults using the obsgen information

    # determine the top 5 most sensitive sites in the array
    indices = np.argsort(list(obsgen.D_dict.values()))
    sitenames = np.array(list(obsgen.D_dict.keys()))[indices][::-1]
    strsites = ''
    count = 0
    for site in sitenames:
        if obsgen.bands[site] is not None:
            count += 1
            if use_two_letter:
                strsites += const.two_letter_station_codes[site]
            else:
                strsites += site
            if count < 5:
                strsites += ', '
            else:
                break
    args['rpicard_refants'] = strsites

    # source name
    args['vex_source'] = obsgen.settings['source']

    # integration time
    args['time_avg'] = str(obsgen.settings['t_int'])+'s'

    # bandwidth
    args['ms_dnu'] = str(obsgen.settings['bandwidth'])

    # frequency
    args['skyfreq'] = str(obsgen.freq/(1.0e9))

    # RA and DEC
    args['ms_RA'] = str(obsgen.RA*15.0)
    args['ms_DEC'] = str(obsgen.DEC)

    # observation start time
    t_start = obsgen.settings['t_start']
    t = Time(obsgen.mjd, format='mjd')
    dumt = t.fits
    dumt2 = '/'.join(dumt.split('-'))
    dumt3 = dumt2.split('T')
    dumt4 = dumt3[1].split(':')
    dumt5 = str(int(float(dumt4[0])+np.floor(t_start))).zfill(2)
    dumt6 = str(int(np.floor((t_start - np.floor(t_start))*60.0))).zfill(2)
    dumt7 = '{:05.2f}'.format((((t_start - float(dumt5))*60.0) - float(dumt6))*60.0)
    dumt8 = ':'.join([dumt5,dumt6,dumt7])
    dumt9 = '/'.join([dumt3[0],dumt8])
    args['ms_StartTime'] = 'UTC,' + dumt9

    # other observation time parameters
    args['ms_obslength'] = str(len(obsgen.t_seg_times)*obsgen.settings['t_int'] / 3600.0)
    args['ms_nscan'] = str(len(obsgen.t_seg_times))
    args['ms_scan_lag'] = str((obsgen.settings['t_rest'] - obsgen.settings['t_int']) / 3600.0)

    #########################################################

    # update with any passed overrides
    args.update(input_args)
    comms.update(input_comments)

    with open(output_filename, 'w') as outfile:

        # loop through the arguments
        for key in list(args.keys()):

            # initialize empty string
            strhere = ''

            # add comment
            strhere += comms[key] + '\n'

            # add argument
            strhere += key + ' = ' + args[key] + '\n'

            # write line
            strhere += '\n'
            outfile.write(strhere)
