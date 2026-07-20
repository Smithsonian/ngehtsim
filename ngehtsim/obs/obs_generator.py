###################################################
# imports

import numpy as np
import ehtim as eh
from dataclasses import replace
from collections import defaultdict
from astropy.time import Time
from astropy import units as astrounits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.utils.exceptions import AstropyWarning
from astropy.utils.iers import IERS_Auto, conf as iers_conf
import yaml
import time
import os
import copy
import warnings

import ngehtsim.const_def as const
import ngehtsim.weather.weather as nw
from ngehtsim.weather.zarr_store import ZarrWeatherStore
import ngehtsim.obs.instrumental_corruptions as instrumental_corruptions
import ngehtsim.obs.fringe_selection as fringe_selection
import ngehtsim.obs.source_models as source_models
import ngehtsim.obs.observation_geometry as observation_geometry
import ngehtsim.obs.station_observation as station_observation
from ngehtsim.obs.simulation_result import SimulationResult

###################################################
# helpers

def _ensure_iers_cached():
    """Pre-load IERS once for Astropy sidereal-time calculations used by ehtim."""
    if IERS_Auto.iers_table is not None:
        return

    auto_download = iers_conf.auto_download
    try:
        iers_conf.auto_download = False
        IERS_Auto.iers_table = IERS_Auto.open()
    finally:
        iers_conf.auto_download = auto_download


def _weather_form(weather):
    forms = {
        'random': 'exact',
        'exact': 'exact',
        'mean': 'mean',
        'average': 'mean',
        'typical': 'median',
        'median': 'median',
        'good': 'good',
        'bad': 'bad',
        'poor': 'bad',
    }
    try:
        return forms[weather]
    except KeyError as exc:
        raise ValueError('Unknown weather form: {0}'.format(weather)) from exc


def _interpolate_spectra(frequency_ghz, spectra, target_frequency_ghz):
    return np.asarray([
        np.interp(target_frequency_ghz, frequency_ghz, spectrum)
        for spectrum in spectra
    ])

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
      wind_loading_overrides (dict): A dictionary of station names and wind-loading v0, w values (in m/s) to override defaults
      custom_receivers (dict): A dictionary of custom receiver names and properties
      station_uptimes (dict): A dictionary of station names and associated uptime ranges, in UT
      array (str): Provide the name of a known array to load the corresponding sites and configuration
      ephem (str): path to the ephemeris for a space station
      weather_store (ngehtsim.weather.zarr_store.ZarrWeatherStore): Optional local Zarr weather dataset.
                                                                    If None, use the packaged binary weather data.
      weather_cadence (str): ``"daily"`` for the established scalar weather behavior or ``"native"``
                             for linearly interpolated three-hour Zarr weather during observation generation.
    """

    # initialize class instantiation
    def __init__(self, settings=None, settings_file=None, verbosity=0, weight=0, D_overrides=None,
                 surf_rms_overrides=None, receiver_configuration_overrides=None, bandwidth_overrides=None,
                 T_R_overrides=None, sideband_ratio_overrides=None, lo_freq_overrides=None, hi_freq_overrides=None,
                 ap_eff_overrides=None, wind_loading_overrides=None, custom_receivers=None, station_uptimes=None,
                 array=None, ephem='ephemeris/space', weather_store=None, weather_cadence='daily'):

        #############################
        # astropy cache

        _ensure_iers_cached()

        #############################
        # initialize inputs

        settings = {} if settings is None else settings
        D_overrides = {} if D_overrides is None else D_overrides
        surf_rms_overrides = {} if surf_rms_overrides is None else surf_rms_overrides
        receiver_configuration_overrides = {} if receiver_configuration_overrides is None else receiver_configuration_overrides
        bandwidth_overrides = {} if bandwidth_overrides is None else bandwidth_overrides
        T_R_overrides = {} if T_R_overrides is None else T_R_overrides
        sideband_ratio_overrides = {} if sideband_ratio_overrides is None else sideband_ratio_overrides
        lo_freq_overrides = {} if lo_freq_overrides is None else lo_freq_overrides
        hi_freq_overrides = {} if hi_freq_overrides is None else hi_freq_overrides
        ap_eff_overrides = {} if ap_eff_overrides is None else ap_eff_overrides
        wind_loading_overrides = {} if wind_loading_overrides is None else wind_loading_overrides
        custom_receivers = {} if custom_receivers is None else custom_receivers
        station_uptimes = {} if station_uptimes is None else station_uptimes
        if weather_store is not None and not isinstance(weather_store, ZarrWeatherStore):
            raise TypeError("weather_store must be a ZarrWeatherStore instance or None.")
        if weather_cadence not in ('daily', 'native'):
            raise ValueError("weather_cadence must be either 'daily' or 'native'.")
        if weather_cadence == 'native' and weather_store is None:
            raise ValueError("Native weather cadence requires a Zarr weather_store.")

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
        self.wind_loading_overrides = copy.deepcopy(wind_loading_overrides)
        self.custom_receivers = copy.deepcopy(custom_receivers)
        self.station_uptimes = copy.deepcopy(station_uptimes)
        self.array = array
        self.ephem = ephem
        self.weather_store = weather_store
        self.weather_cadence = weather_cadence
        self._weather_tables_ready = False
        self._tau_dict = None
        self._Tatm_dict = None
        self._Tb_dict = None
        self._windspeed_dict = None
        self._Tgnd_dict = None

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
        self.mjd = determine_mjd(self.settings['day'], self.settings['month'], self.settings['year'])
        self.arr = make_array(self.sites, ephem=self.ephem, verbosity=self.verbosity)
        self.set_coords()
        self.set_receivers()
        self.set_bands()
        self.set_bandwidths()
        self.set_ap_effs()
        self.im = load_image(self.model_file, freq=self.freq, verbosity=self.verbosity)
        if self.weather_cadence == 'native':
            self._select_weather_date()
        else:
            self.tabulate_weather()
        self.set_telescope_properties()
        self.get_obs_times()

        #############################
        # other settings

        self.obs_empty = None
        self.obs_empty_key = None
        self.obs_template_cache = {}
        self.native_visibility_template = None
        self.native_visibility_template_key = None
        self.native_visibility_template_cache = {}
        self.station_term_cache = {}

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

        # determine solar angle
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            source_location = SkyCoord(ra=self.RA*15.0*astrounits.degree, dec=self.DEC*astrounits.degree, frame='gcrs')
            jd = self.mjd + 2400000.5
            sun_location = get_sun(Time(jd, format='jd'))
            self.solar_angle = sun_location.separation(source_location).value

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

    # select the date used for weather lookups
    def _select_weather_date(self):
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

    # extract and store the relevant weather information
    def tabulate_weather(self):

        self._select_weather_date()
        self._populate_static_weather_tables()

    def _populate_static_weather_tables(self):

        # initialize dictionaries
        tau_dict = defaultdict(dict)
        Tatm_dict = defaultdict(dict)
        Tgnd_dict = defaultdict(dict)
        Tb_dict = defaultdict(dict)
        windspeed_dict = defaultdict(dict)

        form = _weather_form(self.weather)

        # read in the weather info and store it
        for isite, site in enumerate(self.sites):

            if site != 'space':

                tau_here = nw.opacity(site, form=form, month=self.settings['month'], day=self.weather_day,
                                       year=self.weather_year, freq=self.freq/(1.0e9),
                                       weather_store=self.weather_store)
                Tb_here = nw.brightness_temperature(site, form=form, month=self.settings['month'],
                                                     day=self.weather_day, year=self.weather_year,
                                                     freq=self.freq/(1.0e9), weather_store=self.weather_store)
                ws_here = nw.windspeed(site, form=form, month=self.settings['month'], day=self.weather_day,
                                        year=self.weather_year, weather_store=self.weather_store)
                Tgnd_here = nw.temperature(site, form=form, month=self.settings['month'], day=self.weather_day,
                                           year=self.weather_year, weather_store=self.weather_store)

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
        self._tau_dict = tau_dict
        self._Tatm_dict = Tatm_dict
        self._Tb_dict = Tb_dict
        self._windspeed_dict = windspeed_dict
        self._Tgnd_dict = Tgnd_dict
        self._weather_tables_ready = True

    def _ensure_static_weather_tables(self):
        if not self._weather_tables_ready:
            self._populate_static_weather_tables()

    @property
    def tau_dict(self):
        self._ensure_static_weather_tables()
        return self._tau_dict

    @property
    def Tatm_dict(self):
        self._ensure_static_weather_tables()
        return self._Tatm_dict

    @property
    def Tb_dict(self):
        self._ensure_static_weather_tables()
        return self._Tb_dict

    @property
    def windspeed_dict(self):
        self._ensure_static_weather_tables()
        return self._windspeed_dict

    @property
    def Tgnd_dict(self):
        self._ensure_static_weather_tables()
        return self._Tgnd_dict

    def _native_weather_terms(self, times):
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError('Native weather sampling requires finite one-dimensional observation times.')

        unique_times, inverse = np.unique(times, return_inverse=True)
        tau = {}
        Tatm = {}
        Tgnd = {}
        windspeed = {}
        form = _weather_form(self.weather)

        for site in self.sites:
            if site == 'space':
                tau[site] = np.zeros(len(times))
                Tatm[site] = np.zeros(len(times))
                Tgnd[site] = np.full(len(times), const.T_CMB)
                windspeed[site] = np.zeros(len(times))
                continue

            samples = self.weather_store.sample_native(
                site,
                year=int(self.weather_year),
                month=self.settings['month'],
                day=int(self.weather_day),
                utc_hours=unique_times,
                form=form,
            )
            tau_unique = _interpolate_spectra(
                self.weather_store.frequency_ghz,
                samples.opacity,
                self.freq/(1.0e9),
            )
            Tb_unique = _interpolate_spectra(
                self.weather_store.frequency_ghz,
                samples.brightness_temperature,
                self.freq/(1.0e9),
            )
            Tatm_unique = (
                Tb_unique - (const.T_CMB*np.exp(-tau_unique))
            ) / (1.0 - np.exp(-tau_unique))

            tau[site] = tau_unique[inverse]
            Tatm[site] = Tatm_unique[inverse]
            Tgnd[site] = samples.surface_temperature_k[inverse]
            windspeed[site] = samples.wind_speed_m_s[inverse]

        return {
            'tau': tau,
            'Tatm': Tatm,
            'Tgnd': Tgnd,
            'windspeed': windspeed,
        }

    # generate dictionaries of telescope properties
    def set_telescope_properties(self):

        D_dict = {}
        eta_dict = {}
        wind_loading_dict = {}
        solar_avoidance_dict = {}
        for site in self.sites:

            # start with the default values for a new site
            D_dict[site] = self.settings['D_new']
            rms_here = const.surf_rms
            ap_eff_here = const.ap_eff
            solar_avoidance_dict[site] = const.sol_avoid

            # if the site is known, replace those values with the known ones or start with defaults
            if site in list(const.known_diameters.keys()):
                D_dict[site] = const.known_diameters[site]
            if site in list(const.known_surf_rms.keys()):
                rms_here = const.known_surf_rms[site]
            wind_loading_dict[site] = {'v0': const.windspeed_v0,
                                       'w': const.windspeed_w,
                                       'shutdown': const.windspeed_shutdown}
            if site in list(const.known_solar_avoidance_angles.keys()):
                solar_avoidance_dict[site] = const.known_solar_avoidance_angles[site]

            # if the user has provided overrides, use those instead
            if site in list(self.D_overrides.keys()):
                D_dict[site] = self.D_overrides[site]
            if site in list(self.surf_rms_overrides.keys()):
                rms_here = self.surf_rms_overrides[site]
            if site in list(self.ap_eff_overrides.keys()):
                if self.bands[site] in list(self.ap_eff_overrides[site].keys()):
                    ap_eff_here = self.ap_eff_overrides[site][self.bands[site]]
            if site in list(self.wind_loading_overrides.keys()):
                wind_loading_dict[site] = {'v0': self.wind_loading_overrides[site]['v0'],
                                           'w': self.wind_loading_overrides[site]['w'],
                                           'shutdown': self.wind_loading_overrides[site]['shutdown']}

            eta_dict[site] = eta_dish(self.freq, rms_here, const.focus_offset, ap_eff_here)

        self.D_dict = D_dict
        self.eta_dict = eta_dict
        self.wind_loading_dict = wind_loading_dict
        self.solar_avoidance_dict = solar_avoidance_dict

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

    # build context for empty observation construction
    def geometry_context(self):
        return {
            "sites": tuple(self.sites),
            "ra": self.RA,
            "dec": self.DEC,
            "rf": self.freq,
            "bandwidth_hz": (1.0e9)*float(self.settings["bandwidth"]),
            "t_int": self.settings["t_int"],
            "t_rest": self.settings["t_rest"],
            "t_start": self.settings["t_start"],
            "t_stop": self.settings["t_start"] + self.settings["dt"],
            "mjd": self.mjd,
        }

    # build context for source-model adapters
    def source_context(self):
        return {
            "ra": self.RA,
            "dec": self.DEC,
            "mjd": self.mjd,
            "source": self.settings["source"],
            "rf": self.freq,
            "ttype": self.settings["ttype"],
            "fft_pad_factor": self.settings["fft_pad_factor"],
            "verbosity": self.verbosity,
        }

    ###################################################
    # functions for generating observations

    # build context for station/weather/noise calculations
    def station_context(self, times=None):
        receiver_temperature = {}
        sideband_ratio = {}
        bandwidth_hz = {}
        effective_area = {}
        mount_type = {}
        feed_angle = {}
        polarization_basis = {}

        for site in self.sites:
            band = self.bands[site]

            if band is None:
                receiver_temperature[site] = 0.0
                sideband_ratio[site] = 0.0
                bandwidth_hz[site] = None
            else:
                receiver_temperature[site] = self.receivers[site][band]["T_R"]
                sideband_ratio[site] = self.receivers[site][band]["SSR"]
                if band in self.bandwidth_setup[site]:
                    bandwidth_hz[site] = self.bandwidth_setup[site][band]*(1.0e9)
                else:
                    bandwidth_hz[site] = None

            effective_area[site] = (np.pi/4.0)*self.eta_dict[site]*((self.D_dict[site])**2)
            mount_type[site] = const.known_mount_types.get(site, const.mount_type)
            feed_angle[site] = const.known_feed_angles.get(site, const.feed_angle)
            polarization_basis[site] = const.known_polbases.get(site, const.pol_basis)

        if self.weather_cadence == 'native':
            if times is None:
                raise ValueError('Native weather station contexts require observation times.')
            weather_terms = self._native_weather_terms(times)
        else:
            weather_terms = {
                'tau': self.tau_dict,
                'Tatm': self.Tatm_dict,
                'Tgnd': self.Tgnd_dict,
                'windspeed': self.windspeed_dict,
            }

        return {
            "sites": tuple(self.sites),
            "tau": weather_terms['tau'],
            "Tatm": weather_terms['Tatm'],
            "Tgnd": weather_terms['Tgnd'],
            "windspeed": weather_terms['windspeed'],
            "bands": self.bands,
            "wind_loading": self.wind_loading_dict,
            "solar_avoidance": self.solar_avoidance_dict,
            "station_uptimes": self.station_uptimes,
            "receiver_temperature": receiver_temperature,
            "sideband_ratio": sideband_ratio,
            "bandwidth_hz": bandwidth_hz,
            "effective_area": effective_area,
            "mount_type": mount_type,
            "feed_angle": feed_angle,
            "polarization_basis": polarization_basis,
        }

    def _resolve_input_model(self, input_model, caller):
        if input_model is not None:
            return input_model
        if self.im is None:
            raise ValueError(
                "No input model is configured; {0} requires input_model.".format(caller)
            )
        if self.verbosity > 0:
            print("No input model passed to {0}; using the configured model.".format(caller))
        return self.im

    def simulate(self, input_model=None, addnoise=True, addgains=True, gainamp=0.04,
                 leakamp=0.1, opacitycal=True, addFR=True, addleakage=False,
                 flagwind=True, flagday=False, flagsun=True,
                 allow_mixed_basis=False, el_min=const.el_min, el_max=const.el_max,
                 p=None):
        """Simulate a ground-array observation into a native result object.

        This is the primary v2 simulation API. It retains all rows, including
        rows rejected by station-based flagging, in ``result.dataset.flags``.
        ``ehtim`` source classes are accepted as input adapters, but no
        ``ehtim.Obsdata`` is constructed during simulation.
        """

        del p  # Native Fisher-forecast sampling is intentionally not supported.
        input_model = self._resolve_input_model(input_model, "simulate")
        if allow_mixed_basis:
            raise NotImplementedError(
                "Mixed-polarization simulation will be added to the native RIME path."
            )
        if "space" in self.sites:
            raise NotImplementedError(
                "Native spacecraft geometry is not implemented; use observe_legacy() "
                "or make_obs_legacy() explicitly."
            )
        adapter = source_models.adapter_for(input_model)
        if not hasattr(adapter, "observe_dataset"):
            raise TypeError(
                "Native simulation currently supports ehtim Image, Movie, and Model inputs only."
            )

        (self.native_visibility_template,
         self.native_visibility_template_key,
         self.native_visibility_template_cache,
         template) = observation_geometry.native_visibility_template(
            self.native_visibility_template,
            self.native_visibility_template_key,
            self.native_visibility_template_cache,
            self.arr,
            self.geometry_context(),
            el_min=el_min,
            el_max=el_max,
        )
        if template.row_count == 0:
            return SimulationResult(template, {})

        sampled, F0 = source_models.observe_source_dataset(
            input_model,
            template,
            self.source_context(),
        )
        station_terms, stations = station_observation.station_terms_for_dataset(
            sampled,
            F0,
            self.station_context((sampled.time_mjd - self.mjd) * 24.0),
            self.rng,
            gainamp=gainamp,
            leakamp=leakamp,
            addgains=addgains,
            addleakage=addleakage,
            flagwind=flagwind,
            flagday=flagday,
            flagsun=flagsun,
            solar_angle=self.solar_angle,
            verbosity=self.verbosity,
            windspeed_sefd_modifier=windspeed_SEFD_modification,
            reference_mjd=self.mjd,
            cache=self.station_term_cache,
        )
        corrupted = instrumental_corruptions.apply_circular_corruptions(
            sampled,
            station_terms,
            stations,
            self.rng,
            addnoise=addnoise,
            addgains=addgains,
            opacitycal=opacitycal,
            addFR=addFR,
            addleakage=addleakage,
        )
        return SimulationResult(corrupted, station_terms)

    def observe_legacy(self, input_model, addnoise=True, addgains=True, gainamp=0.04, leakamp=0.1,
                opacitycal=True, addFR=True, addleakage=False,
                flagwind=True, flagday=False, flagsun=True,
                allow_mixed_basis=False, el_min=const.el_min, el_max=const.el_max, p=None):
        """
        Generate a raw single-band observation that folds in weather-based opacity and sensitivity effects.

        Args:
          input_model (ehtim.image.Image, ehtim.movie.Movie, ehtim.model.Model, ngEHTforecast.fisher.fisher_forecast.FisherForecast): input source model
          addnoise (bool): flag for whether or not to add thermal noise to the visibilities
          addgains (bool): flag for whether or not to add station gain corruptions
          gainamp (float): standard deviation of amplitude log-gains
          leakamp (float): standard deviation of leakage real and imaginary parts
          opacitycal (bool): flag for whether or not to assume that atmospheric opacity is assumed to be calibrated out
          addFR (bool): flag for whether or not to add feed rotations
          addleakage (bool): flag for whether or not to add polarization leakage corruptions
          flagwind (bool): flag for whether to derate sites with high wind
          flagday (bool): flag for whether to flag sites during the local daytime
          flagsun (bool): flag for whether to impose a minimum solar avoidance angle
          allow_mixed_basis (bool): currently unsupported; must be False
          el_min (float): minimum elevation that a site can observe at, in degrees
          el_max (float): maximum elevation that a site can observe at, in degrees
          p (numpy.ndarray): list of parameters for an input ngEHTforecast.fisher.fisher_forecast.FisherForecast object

        Returns:
          (ehtim.obsdata.Obsdata): eht-imaging Obsdata object containing the generated observation
        """

        if allow_mixed_basis:
            raise NotImplementedError(
                "Mixed-polarization output requires VisibilityDataset support and "
                "cannot be represented safely as ehtim Obsdata."
            )

        # generate and elevation-limit an empty observation template
        self.obs_empty, self.obs_empty_key, self.obs_template_cache, obs_empty = observation_geometry.observation_template(
            self.obs_empty,
            self.obs_empty_key,
            self.obs_template_cache,
            self.arr,
            self.geometry_context(),
            el_min=el_min,
            el_max=el_max,
        )

        # observe the source
        obs, F0 = source_models.observe_source(input_model, obs_empty, self.source_context(), p=p)

        # calculate station/weather/noise terms
        station_terms = station_observation.station_terms(
            obs,
            F0,
            self.station_context(obs.data['time']),
            self.arr,
            self.rng,
            gainamp=gainamp,
            leakamp=leakamp,
            addgains=addgains,
            addleakage=addleakage,
            flagwind=flagwind,
            flagday=flagday,
            flagsun=flagsun,
            allow_mixed_basis=allow_mixed_basis,
            solar_angle=self.solar_angle,
            verbosity=self.verbosity,
            windspeed_sefd_modifier=windspeed_SEFD_modification,
            cache=self.station_term_cache,
        )

        t1 = station_terms["t1"]
        t2 = station_terms["t2"]
        el1 = station_terms["el1"]
        el2 = station_terms["el2"]
        par1 = station_terms["par1"]
        par2 = station_terms["par2"]
        times = station_terms["times"]
        tau1 = station_terms["tau1"]
        tau2 = station_terms["tau2"]
        Tb1 = station_terms["Tb1"]
        Tb2 = station_terms["Tb2"]
        Tsys1 = station_terms["Tsys1"]
        Tsys2 = station_terms["Tsys2"]
        SEFD1 = station_terms["SEFD1"]
        SEFD2 = station_terms["SEFD2"]
        bw1 = station_terms["bw1"]
        bw2 = station_terms["bw2"]
        f_el1 = station_terms["f_el1"]
        f_el2 = station_terms["f_el2"]
        f_par1 = station_terms["f_par1"]
        f_par2 = station_terms["f_par2"]
        phi_off1 = station_terms["phi_off1"]
        phi_off2 = station_terms["phi_off2"]
        flagsites = station_terms["flagsites"]
        uptime_mask = station_terms["uptime_mask"]

        if addgains:
            gainamp1R = station_terms["gainamp1R"]
            gainamp2R = station_terms["gainamp2R"]
            gainphase1R = station_terms["gainphase1R"]
            gainphase2R = station_terms["gainphase2R"]
            gainamp1L = station_terms["gainamp1L"]
            gainamp2L = station_terms["gainamp2L"]
            gainphase1L = station_terms["gainphase1L"]
            gainphase2L = station_terms["gainphase2L"]

        if addleakage:
            leak1R = station_terms["leak1R"]
            leak2R = station_terms["leak2R"]
            leak1L = station_terms["leak1L"]
            leak2L = station_terms["leak2L"]

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
            fa_1[(t1 == 'space')] = 0.0
            fa_2[(t2 == 'space')] = 0.0
            if self.weight > 0:
                self.fa_1 = fa_1
                self.fa_2 = fa_2
            obs.data['rrvis'] *= np.exp(-(1.0j)*fa_1)*np.exp((1.0j)*fa_2)
            obs.data['rlvis'] *= np.exp(-(1.0j)*fa_1)*np.exp(-(1.0j)*fa_2)
            obs.data['lrvis'] *= np.exp((1.0j)*fa_1)*np.exp((1.0j)*fa_2)
            obs.data['llvis'] *= np.exp((1.0j)*fa_1)*np.exp(-(1.0j)*fa_2)

        # store and apply leakages
        if addleakage:
            if self.weight > 0:
                self.station_leakage1R = leak1R
                self.station_leakage2R = leak2R
                self.station_leakage1L = leak1L
                self.station_leakage2L = leak2L
            visibilities = np.column_stack((
                obs.data['rrvis'],
                obs.data['llvis'],
                obs.data['rlvis'],
                obs.data['lrvis'],
            ))
            visibilities = instrumental_corruptions.apply_circular_leakage(
                visibilities,
                leak1R,
                leak1L,
                leak2R,
                leak2L,
            )
            obs.data['rrvis'] = visibilities[:, 0]
            obs.data['llvis'] = visibilities[:, 1]
            obs.data['rlvis'] = visibilities[:, 2]
            obs.data['lrvis'] = visibilities[:, 3]

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
            obs.data['rrvis'] += obs.data['rrsigma']*(self.rng.normal(0.0, 1.0, len(obs.data['rrsigma'])) + ((1.0j)*self.rng.normal(0.0, 1.0, len(obs.data['rrsigma']))))
            obs.data['llvis'] += obs.data['llsigma']*(self.rng.normal(0.0, 1.0, len(obs.data['llsigma'])) + ((1.0j)*self.rng.normal(0.0, 1.0, len(obs.data['llsigma']))))
            obs.data['rlvis'] += obs.data['rlsigma']*(self.rng.normal(0.0, 1.0, len(obs.data['rlsigma'])) + ((1.0j)*self.rng.normal(0.0, 1.0, len(obs.data['rlsigma']))))
            obs.data['lrvis'] += obs.data['lrsigma']*(self.rng.normal(0.0, 1.0, len(obs.data['lrsigma'])) + ((1.0j)*self.rng.normal(0.0, 1.0, len(obs.data['lrsigma']))))

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
            if addFR:
                self.fa_1 = self.fa_1[mask]
                self.fa_2 = self.fa_2[mask]
            if addleakage:
                self.station_leakage1R = self.station_leakage1R[mask]
                self.station_leakage2R = self.station_leakage2R[mask]
                self.station_leakage1L = self.station_leakage1L[mask]
                self.station_leakage2L = self.station_leakage2L[mask]

        obs.ampcal = not addgains
        obs.phasecal = not addgains
        obs.opacitycal = opacitycal
        obs.dcal = not addleakage
        obs.frcal = not addFR

        # return observation object
        return obs

    # Legacy Obsdata-only observation path. It is retained for capabilities
    # that have not yet acquired a native implementation, notably FPT and
    # spacecraft geometry.
    def make_obs_legacy(self, input_model=None, addnoise=True, addgains=True, gainamp=0.04, leakamp=0.1,
                 opacitycal=True, addFR=True, addleakage=False,
                 flagwind=True, flagday=False, flagsun=True,
                 allow_mixed_basis=False, el_min=const.el_min, el_max=const.el_max, p=None):
        """
        Generate an observation that folds in weather-based opacity effects
        and applies a specified SNR thresholding scheme to mimic fringe-finding.

        Args:
          input_model (ehtim.image.Image, ehtim.movie.Movie, ehtim.model.Model, ngEHTforecast.fisher.fisher_forecast.FisherForecast): input source model
          addnoise (bool): flag for whether or not to add thermal noise to the visibilities
          addgains (bool): flag for whether or not to add station gain corruptions
          gainamp (float): standard deviation of amplitude log-gains
          leakamp (float): standard deviation of leakage real and imaginary parts
          opacitycal (bool): flag for whether or not to assume that atmospheric opacity is assumed to be calibrated out
          addFR (bool): flag for whether or not to add feed rotations
          addleakage (bool): flag for whether or not to add polarization leakage corruptions
          flagwind (bool): flag for whether to derate sites with high wind
          flagday (bool): flag for whether to flag sites during the local daytime
          flagsun (bool): flag for whether to impose a minimum solar avoidance angle
          allow_mixed_basis (bool): currently unsupported; must be False
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
        obs = self.observe_legacy(input_model,
                           addnoise=addnoise,
                           addgains=addgains,
                           gainamp=gainamp,
                           leakamp=leakamp,
                           opacitycal=opacitycal,
                           flagwind=flagwind,
                           flagday=flagday,
                           flagsun=flagsun,
                           addFR=addFR,
                           addleakage=addleakage,
                           allow_mixed_basis=allow_mixed_basis,
                           el_min=el_min,
                           el_max=el_max,
                           p=p)

        # create a running index list of baselines to keep
        master_index = np.ones(len(obs.data), dtype='bool')

        # identify sites that can't observe at the requested frequency
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
                master_index &= np.array([t1_list[j] not in sites_to_remove and t2_list[j] not in sites_to_remove for j in range(len(t1_list))])

        # identify sites that are randomly deemed to be technically unready
        unready_sites = get_unready_sites(obs.tarr['site'], self.settings['tech_readiness'], rng=self.rng)
        if len(unready_sites) > 0:
            if self.verbosity > 0:
                print("Dropping {0} due to technical (un)readiness.".format(unready_sites))
            if len(obs.data) > 0:
                t1_list = obs.unpack('t1')['t1']
                t2_list = obs.unpack('t2')['t2']
                master_index &= np.array([t1_list[j] not in unready_sites and t2_list[j] not in unready_sites for j in range(len(t1_list))])

        # apply naive SNR thresholding
        if (snr_algo.lower() == 'naive'):
            master_index &= obs.unpack('snr')['snr'] > snr_args

        # apply a proxy for the "fringegroups" procedure from HOPS
        elif (snr_algo.lower() == 'fringegroups'):

            # parse fringe_finder arguments
            snr_ref = snr_args[0]
            tint_ref = snr_args[1]

            # run fringegroups
            obs_pass = obs.copy()
            obs_pass.data = obs.data.copy()[master_index]
            master_index[np.where(master_index)] &= fringegroups(self, obs_pass, snr_ref, tint_ref)

        # apply an FPT proxy for SNR thresholding
        elif (snr_algo.lower() == 'fpt'):

            # parse fringe_finder arguments
            snr_ref = snr_args[0]
            tint_ref = snr_args[1]
            freq_ref = snr_args[2]
            model_path_ref = snr_args[3]

            # run FPT
            master_index &= FPT(self, obs, snr_ref, tint_ref, freq_ref, model_path_ref, ephem=self.ephem, addnoise=addnoise, addgains=addgains, gainamp=gainamp, leakamp=leakamp, opacitycal=opacitycal, flagwind=flagwind, flagday=flagday, flagsun=flagsun, addFR=addFR, addleakage=addleakage, el_min=el_min, el_max=el_max, p=p, unready_sites=unready_sites, target_model=input_model)

        # unrecognized SNR thresholding scheme
        else:
            raise ValueError('Unknown algorithm for fringe_finder.')

        # apply the data flags to the observation
        data_copy = obs.data.copy()
        obs.data = data_copy[master_index]
        if self.verbosity > 0:
            print('Flagged '+str(len(master_index) - master_index.sum())+' of '+str(len(master_index))+' data points during fringe-finding emulation.')

        # flag the additional stored quantities as well
        if self.weight > 0:
            self.timestamps = self.timestamps[master_index]
            self.ant1 = self.ant1[master_index]
            self.ant2 = self.ant2[master_index]
            self.bandwidths = self.bandwidths[master_index]
            self.Tsys1 = self.Tsys1[master_index]
            self.Tsys2 = self.Tsys2[master_index]
            self.tau1 = self.tau1[master_index]
            self.tau2 = self.tau2[master_index]
            self.Tb1 = self.Tb1[master_index]
            self.Tb2 = self.Tb2[master_index]
            self.SEFD1 = self.SEFD1[master_index]
            self.SEFD2 = self.SEFD2[master_index]
            if addgains:
                self.station_gains1R = self.station_gains1R[master_index]
                self.station_gains2R = self.station_gains2R[master_index]
                self.station_gains1L = self.station_gains1L[master_index]
                self.station_gains2L = self.station_gains2L[master_index]
            if addFR:
                self.fa_1 = self.fa_1[master_index]
                self.fa_2 = self.fa_2[master_index]
            if addleakage:
                self.station_leakage1R = self.station_leakage1R[master_index]
                self.station_leakage2R = self.station_leakage2R[master_index]
                self.station_leakage1L = self.station_leakage1L[master_index]
                self.station_leakage2L = self.station_leakage2L[master_index]

        # return observation object
        return obs

    def _native_selection_mask(self, dataset, input_model=None, simulation_kwargs=None):
        """Return the native row-selection mask for availability and fringe finding."""

        mask = ~np.any(dataset.flags, axis=(1, 2))
        names = np.asarray(dataset.stations.names)
        t1 = names[dataset.antenna1]
        t2 = names[dataset.antenna2]

        unavailable = tuple(site for site in dataset.stations.names if self.bands[site] is None)
        if unavailable:
            if self.verbosity > 0:
                for site in unavailable:
                    print(site + " cannot observe at " + str(self.freq / 1.0e9) + " GHz.")
            mask &= ~np.isin(t1, unavailable) & ~np.isin(t2, unavailable)

        unready = get_unready_sites(
            np.asarray(dataset.stations.names),
            self.settings["tech_readiness"],
            rng=self.rng,
        )
        if len(unready):
            if self.verbosity > 0:
                print("Dropping {0} due to technical (un)readiness.".format(unready))
            mask &= ~np.isin(t1, unready) & ~np.isin(t2, unready)

        available_sites = [
            site for site in dataset.stations.names
            if self.bands[site] is not None and site not in unready
        ]

        snr_algorithm, snr_args = self.settings["fringe_finder"]
        snr_algorithm = snr_algorithm.lower()
        if snr_algorithm == "naive":
            pseudo_i_amplitude = 0.5 * (
                np.abs(dataset.visibilities[:, 0, 0])
                + np.abs(dataset.visibilities[:, 0, 1])
            )
            pseudo_i_sigma = 1.0 / np.sqrt(2.0 * dataset.weights[:, 0, 0])
            mask &= (pseudo_i_amplitude / pseudo_i_sigma) > snr_args
        elif snr_algorithm == "fringegroups":
            selected_indices = np.flatnonzero(mask)
            selected = dataset.take_rows(selected_indices)
            mask[selected_indices] &= fringegroups_dataset(
                self,
                selected,
                snr_args[0],
                snr_args[1],
            )
        elif snr_algorithm == "fpt":
            if input_model is None or simulation_kwargs is None:
                raise ValueError(
                    "Native FPT selection requires the target model and simulation settings."
                )
            snr_ref, tint_ref, freq_ref, model_ref = snr_args
            mask &= self._native_fpt_selection_mask(
                dataset,
                input_model,
                snr_ref,
                tint_ref,
                freq_ref,
                model_ref,
                simulation_kwargs,
                target_available_sites=available_sites,
                target_row_available=mask,
                unready_sites=unready,
            )
        else:
            raise ValueError("Unknown algorithm for fringe_finder.")
        return mask

    def _native_fpt_selection_mask(self, target_dataset, target_model, snr_ref,
                                   tint_ref, freq_ref, model_ref, simulation_kwargs,
                                   target_available_sites, target_row_available,
                                   unready_sites):
        """Run the FPT reference simulation without constructing ``ehtim.Obsdata``."""

        reference_generator, reference_model = _fpt_reference_generator(
            self,
            snr_ref,
            tint_ref,
            freq_ref,
            model_ref,
            target_model,
            ephem=self.ephem,
        )
        reference_result = reference_generator.simulate(
            input_model=reference_model,
            **simulation_kwargs
        )
        reference_dataset = reference_result.dataset
        reference_available_sites = [
            site for site in reference_dataset.stations.names
            if reference_generator.bands[site] is not None
            and site not in unready_sites
        ]
        return fringe_selection.fpt_fringe_group_mask(
            _fringe_rows_from_dataset(target_dataset),
            _fringe_rows_from_dataset(reference_dataset),
            snr_ref,
            tint_ref,
            freq_ref / (self.freq / 1.0e9),
            target_available_sites=target_available_sites,
            reference_available_sites=reference_available_sites,
            target_row_available=target_row_available,
            reference_row_available=~np.any(reference_dataset.flags, axis=(1, 2)),
        )

    def make_dataset(self, input_model=None, addnoise=True, addgains=True, gainamp=0.04,
                     leakamp=0.1, opacitycal=True, addFR=True, addleakage=False,
                     flagwind=True, flagday=False, flagsun=True,
                     allow_mixed_basis=False, el_min=const.el_min,
                     el_max=const.el_max, p=None):
        """Generate a fully selected native :class:`SimulationResult`.

        Station, availability, technical-readiness, and fringe-selection
        failures are represented in ``result.dataset.flags`` rather than by
        deleting rows from the internal data model.
        """

        input_model = self._resolve_input_model(input_model, "make_dataset")
        simulation_kwargs = {
            "addnoise": addnoise,
            "addgains": addgains,
            "gainamp": gainamp,
            "leakamp": leakamp,
            "opacitycal": opacitycal,
            "addFR": addFR,
            "addleakage": addleakage,
            "flagwind": flagwind,
            "flagday": flagday,
            "flagsun": flagsun,
            "allow_mixed_basis": allow_mixed_basis,
            "el_min": el_min,
            "el_max": el_max,
            "p": p,
        }
        result = self.simulate(
            input_model=input_model,
            **simulation_kwargs
        )
        if not result.dataset.row_count:
            return result

        mask = self._native_selection_mask(
            result.dataset,
            input_model=input_model,
            simulation_kwargs=simulation_kwargs,
        )
        flags = np.array(result.dataset.flags, copy=True)
        flags[~mask] = True
        if self.verbosity > 0:
            print(
                "Flagged {0} of {1} data points during fringe-finding emulation.".format(
                    len(mask) - np.count_nonzero(mask),
                    len(mask),
                )
            )
        return SimulationResult(replace(result.dataset, flags=flags), result.station_terms)

    def observe(self, input_model=None, addnoise=True, addgains=True, gainamp=0.04,
                leakamp=0.1, opacitycal=True, addFR=True, addleakage=False,
                flagwind=True, flagday=False, flagsun=True,
                allow_mixed_basis=False, el_min=const.el_min,
                el_max=const.el_max, p=None, backend="native"):
        """Generate a raw ``ehtim.Obsdata`` export from the selected backend.

        ``backend="native"`` is the default and constructs no ``Obsdata``
        until export. ``backend="legacy"`` explicitly selects the retained
        legacy implementation for capabilities not yet available natively.
        """

        if backend == "legacy":
            input_model = self._resolve_input_model(input_model, "observe_legacy")
            return self.observe_legacy(
                input_model,
                addnoise=addnoise,
                addgains=addgains,
                gainamp=gainamp,
                leakamp=leakamp,
                opacitycal=opacitycal,
                addFR=addFR,
                addleakage=addleakage,
                flagwind=flagwind,
                flagday=flagday,
                flagsun=flagsun,
                allow_mixed_basis=allow_mixed_basis,
                el_min=el_min,
                el_max=el_max,
                p=p,
            )
        if backend != "native":
            raise ValueError("backend must be either 'native' or 'legacy'.")
        return self.simulate(
            input_model=input_model,
            addnoise=addnoise,
            addgains=addgains,
            gainamp=gainamp,
            leakamp=leakamp,
            opacitycal=opacitycal,
            addFR=addFR,
            addleakage=addleakage,
            flagwind=flagwind,
            flagday=flagday,
            flagsun=flagsun,
            allow_mixed_basis=allow_mixed_basis,
            el_min=el_min,
            el_max=el_max,
            p=p,
        ).to_ehtim_obsdata()

    def make_obs(self, input_model=None, addnoise=True, addgains=True, gainamp=0.04,
                 leakamp=0.1, opacitycal=True, addFR=True, addleakage=False,
                 flagwind=True, flagday=False, flagsun=True,
                 allow_mixed_basis=False, el_min=const.el_min,
                 el_max=const.el_max, p=None, backend="native"):
        """Generate an ``ehtim.Obsdata`` export of a selected observation."""

        if backend == "legacy":
            return self.make_obs_legacy(
                input_model=input_model,
                addnoise=addnoise,
                addgains=addgains,
                gainamp=gainamp,
                leakamp=leakamp,
                opacitycal=opacitycal,
                addFR=addFR,
                addleakage=addleakage,
                flagwind=flagwind,
                flagday=flagday,
                flagsun=flagsun,
                allow_mixed_basis=allow_mixed_basis,
                el_min=el_min,
                el_max=el_max,
                p=p,
            )
        if backend != "native":
            raise ValueError("backend must be either 'native' or 'legacy'.")
        return self.make_dataset(
            input_model=input_model,
            addnoise=addnoise,
            addgains=addgains,
            gainamp=gainamp,
            leakamp=leakamp,
            opacitycal=opacitycal,
            addFR=addFR,
            addleakage=addleakage,
            flagwind=flagwind,
            flagday=flagday,
            flagsun=flagsun,
            allow_mixed_basis=allow_mixed_basis,
            el_min=el_min,
            el_max=el_max,
            p=p,
        ).to_ehtim_obsdata()

    # generate multifrequency observation, assuming that FPT will be used wherever possible
    def make_obs_mf(self, freqs, input_models, addnoise=True, addgains=True, gainamp=0.04, leakamp=0.1,
                    opacitycal=True, addFR=True, addleakage=False,
                    flagwind=True, flagday=False, flagsun=True,
                    el_min=const.el_min, el_max=const.el_max, p=None):
        """
        Generate a multi-frequency observation

        Args:
          freqs (list): list of frequencies at which to carry out the observation, in GHz
          input_models (list): list of input source models; one for each frequency
          addnoise (bool): flag for whether or not to add thermal noise to the visibilities
          addgains (bool): flag for whether or not to add station gain corruptions
          gainamp (float): standard deviation of amplitude log-gains
          leakamp (float): standard deviation of leakage real and imaginary parts
          opacitycal (bool): flag for whether or not to assume that atmospheric opacity is assumed to be calibrated out
          addFR (bool): flag for whether or not to add feed rotations
          addleakage (bool): flag for whether or not to add polarization leakage corruptions
          flagwind (bool): flag for whether to derate sites with high wind
          flagday (bool): flag for whether to flag sites during the local daytime
          flagsun (bool): flag for whether to impose a minimum solar avoidance angle
          el_min (float): minimum elevation that a site can observe at, in degrees
          el_max (float): maximum elevation that a site can observe at, in degrees
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
                                            wind_loading_overrides=copy.deepcopy(self.wind_loading_overrides),
                                            custom_receivers=copy.deepcopy(self.custom_receivers),
                                            station_uptimes=copy.deepcopy(self.station_uptimes),
                                            array=self.array,
                                            ephem=self.ephem,
                                            weather_store=self.weather_store,
                                            weather_cadence=self.weather_cadence)

                if ((model_target is not None) & (not isinstance(model_target, str))):
                    obsgen_here.im = model_target

                # generate observation at target frequency
                obs_here = obsgen_here.make_obs(input_model=obsgen_here.im, addnoise=addnoise, addgains=addgains, gainamp=gainamp, leakamp=leakamp, opacitycal=opacitycal, addFR=addFR, addleakage=addleakage, el_min=el_min, el_max=el_max, flagwind=flagwind, flagday=flagday, flagsun=flagsun, p=p_target, backend='legacy')

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

        if self.weather_cadence == 'native':
            raise ValueError('SYMBA antenna exports do not support native time-varying weather.')

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
      day (str): Numerical day of the month; e.g. '15' or '22'
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


def windspeed_SEFD_modification(windspeed, windspeed_v0=const.windspeed_v0,
                                windspeed_w=const.windspeed_w):
    """
    Function to convert a windspeed to an effective SEFD scaling factor.

    Args:
      windspeed (float): windspeed value, in m/s
      windspeed_v0 (float): central point of the logistic function
      windspeed_w (float): parameter describing the width of the logistic function; larger is more permissive
      
    Returns:
      (float): factor by which to scale the SEFD
    """

    scale_factor = 1.0 - (1.0 / (1.0 + np.exp(-(3.0/windspeed_w)*(windspeed - windspeed_v0))))
        
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

    rows = _fringe_rows_from_obsdata(obs)
    available_sites = [site for site in obsgen.sites if obsgen.bands[site] is not None]
    return fringe_selection.fringe_group_mask(
        rows,
        snr_ref,
        tint_ref,
        available_sites=available_sites,
    )


def _fringe_rows_from_dataset(dataset):
    """Extract circular parallel-hand fringe-selection inputs from a native dataset."""

    if dataset.channel_count != 1:
        raise ValueError("Native fringe selection requires exactly one spectral channel.")
    names = np.asarray(dataset.stations.names)
    return fringe_selection.FringeRows(
        time=dataset.time_mjd,
        station1=names[dataset.antenna1],
        station2=names[dataset.antenna2],
        integration_time_s=dataset.integration_time_s,
        rr=dataset.visibilities[:, 0, 0],
        ll=dataset.visibilities[:, 0, 1],
        rr_sigma=1.0 / np.sqrt(dataset.weights[:, 0, 0]),
        ll_sigma=1.0 / np.sqrt(dataset.weights[:, 0, 1]),
    )


def fringegroups_dataset(obsgen, dataset, snr_ref, tint_ref):
    """Apply the fringe-group proxy directly to a native circular dataset."""

    if dataset.row_count == 0:
        return np.zeros(0, dtype=bool)

    rows = _fringe_rows_from_dataset(dataset)
    available_sites = [site for site in obsgen.sites if obsgen.bands[site] is not None]
    return fringe_selection.fringe_group_mask(
        rows,
        snr_ref,
        tint_ref,
        available_sites=available_sites,
    )


def _fpt_reference_generator(obsgen, snr_ref, tint_ref, freq_ref, model_ref,
                             target_model=None, ephem='ephemeris/space'):
    """Create the isolated native/legacy reference generator used by FPT."""

    if target_model is None:
        target_model = obsgen.im

    new_settings = copy.copy(obsgen.settings)
    new_settings['frequency'] = freq_ref
    new_settings['bandwidth'] = obsgen.settings['bandwidth']
    new_settings['fringe_finder'] = ['fringegroups', [snr_ref, tint_ref]]
    new_settings['random_seed'] = obsgen.seed
    if isinstance(model_ref, str):
        new_settings['model_file'] = model_ref
    else:
        new_settings['model_file'] = None
    if ((obsgen.weather == 'random') | (obsgen.weather == 'exact')):
        new_settings['weather'] = 'exact'
        new_settings['weather_year'] = str(obsgen.weather_year)
        new_settings['weather_day'] = str(obsgen.weather_day)

    obsgen_ref = obs_generator(
        new_settings,
        D_overrides=copy.deepcopy(obsgen.D_overrides),
        receiver_configuration_overrides=copy.deepcopy(obsgen.receiver_configuration_overrides),
        surf_rms_overrides=copy.deepcopy(obsgen.surf_rms_overrides),
        bandwidth_overrides=copy.deepcopy(obsgen.bandwidth_overrides),
        T_R_overrides=copy.deepcopy(obsgen.T_R_overrides),
        sideband_ratio_overrides=copy.deepcopy(obsgen.sideband_ratio_overrides),
        lo_freq_overrides=copy.deepcopy(obsgen.lo_freq_overrides),
        hi_freq_overrides=copy.deepcopy(obsgen.hi_freq_overrides),
        ap_eff_overrides=copy.deepcopy(obsgen.ap_eff_overrides),
        wind_loading_overrides=copy.deepcopy(obsgen.wind_loading_overrides),
        custom_receivers=copy.deepcopy(obsgen.custom_receivers),
        station_uptimes=copy.deepcopy(obsgen.station_uptimes),
        array=getattr(obsgen, 'array', None),
        ephem=ephem,
        weather_store=obsgen.weather_store,
        weather_cadence=obsgen.weather_cadence,
    )
    if model_ref is None:
        reference_model = target_model
    elif isinstance(model_ref, str):
        reference_model = obsgen_ref.im
    else:
        reference_model = model_ref
    obsgen_ref.im = reference_model
    return obsgen_ref, reference_model


def FPT(obsgen, obs, snr_ref, tint_ref, freq_ref, model_ref=None, ephem='ephemeris/space',
        unready_sites=(), target_model=None, **kwargs):
    """
    Function to apply the frequency phase transfer ("FPT") SNR thresholding scheme to an observation.
    This scheme attempts to mimic the fringe-finding consequences of phase
    transfer in the HOPS calibration pipeline. It only selects detectable
    target rows; it does not apply phase-transfer corrections to visibilities.

    Args:
      obsgen (ngehtsim.obs.obs_generator.obs_generator): ngehtsim obs_generator object containing information about the observation
      obs (ehtim.obsdata.Obsdata): eht-imaging Obsdata object containing the input observation
      snr_ref (float): strong baseline SNR threshold
      tint_ref (float): strong baseline coherence time, in seconds
      freq_ref(float): FPT reference frequency, in GHz
      model_ref (str): path to FPT reference model, the reference model itself,
                       or None to reuse the target model

    Returns:
      (numpy.ndarray): An array of kept data indices
    """

    freq_rat = freq_ref / (obsgen.freq / 1.0e9)
    obsgen_ref, reference_model = _fpt_reference_generator(
        obsgen,
        snr_ref,
        tint_ref,
        freq_ref,
        model_ref,
        target_model,
        ephem=ephem,
    )

    # generate observation at reference frequency
    obs_ref = obsgen_ref.observe_legacy(reference_model, **kwargs)

    target_rows = _fringe_rows_from_obsdata(obs)
    reference_rows = _fringe_rows_from_obsdata(obs_ref)
    unready_sites = set(unready_sites)
    target_available = [
        site for site in obsgen.sites
        if obsgen.bands[site] is not None and site not in unready_sites
    ]
    reference_available = [
        site for site in obsgen_ref.sites
        if obsgen_ref.bands[site] is not None and site not in unready_sites
    ]
    return fringe_selection.fpt_fringe_group_mask(
        target_rows,
        reference_rows,
        snr_ref,
        tint_ref,
        freq_rat,
        target_available_sites=target_available,
        reference_available_sites=reference_available,
    )


def _fringe_rows_from_obsdata(obs):
    """Extract circular parallel-hand fringe-selection inputs from Obsdata."""
    circular = obs.switch_polrep(polrep_out='circ')
    return fringe_selection.FringeRows(
        time=circular.data['time'],
        station1=circular.data['t1'],
        station2=circular.data['t2'],
        integration_time_s=circular.data['tint'],
        rr=circular.data['rrvis'],
        ll=circular.data['llvis'],
        rr_sigma=circular.data['rrsigma'],
        ll_sigma=circular.data['llsigma'],
    )


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

    if obsgen.weather_cadence == 'native':
        raise ValueError('SYMBA antenna exports do not support native time-varying weather.')

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

        form = _weather_form(obsgen.weather)

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
                PWV = nw.PWV(site, form=form, month=obsgen.settings['month'], day=obsgen.weather_day,
                             year=obsgen.weather_year, weather_store=obsgen.weather_store)
                strhere += str(np.round(PWV, 4)).ljust(9)

                # add surface pressure, in mbar
                pres = nw.pressure(site, form=form, month=obsgen.settings['month'], day=obsgen.weather_day,
                                   year=obsgen.weather_year, weather_store=obsgen.weather_store)
                strhere += str(np.round(pres, 2)).ljust(12)

                # add surface temperature, in K
                temp = nw.temperature(site, form=form, month=obsgen.settings['month'], day=obsgen.weather_day,
                                      year=obsgen.weather_year, weather_store=obsgen.weather_store)
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
