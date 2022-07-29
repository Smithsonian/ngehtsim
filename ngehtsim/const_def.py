import ngehtsim as ng
import os

###################################################
# some defaults

# relative path to the weather information
path_to_weather = os.path.dirname(ng.__file__) + '/weather_data/'

default_settings = {'model_file': None,
                    'source': 'M87',
                    'frequency': 230.0,
                    'RA': 12.51373,
                    'DEC': 12.39112,
                    'bandwidth': 2.0,
                    'nbands': 1,
                    'rf_offset': 0.0,
                    'month': 'Jan',
                    'year': '2025',
                    'day': '15',
                    't_start': 0.0,
                    'dt': 24.0,
                    't_int': 600.0,
                    't_rest': 1200.0,
                    'SNR_cutoff': ['fringegroups', [5.0, 10.0]],
                    'array': 'EHT2022',
                    'sites': None,
                    'D_new': 10.0,
                    'tech_readiness': 1.0,
                    'weather': 'random',
                    'weather_freq': None,
                    'ttype': 'fast',
                    'fft_pad_factor': 2,
                    'random_seed': None}

# minimum and maximum elevations at which a site may observe a source
el_min = 10.0
el_max = 80.0

# minimum and maximum years from which to query weather data
year_min = 2009
year_max = 2018

# fiducial receiver temperatures, in K
T_R_dict = {'86': 40.0,
            '230': 50.0,
            '345': 75.0,
            '410': 150.0,
            '690': 100.0}

# quantization efficiency
quant_eff = 0.88

# fiducial focus offset, in effective surface accuracy units
focus_offset = 50.0e-6

# fiducial surface RMS, in meters
sigma_surface = 40.0e-6

# fiducial pointing accuracy factor for existing telescopes, such that pointing RMS = primary_beamsize / accuracy_factor
existing_pt_accuracy_factor = 10.0

###################################################
# physical constants

# Boltzmann constant, in Jy m^2 / K
k = 1381.0

# speed of light, in m/s
c = 3.0e8

# Earth diameter, in m
D_Earth = 12742000.0

# CMB temperature, in K
T_CMB = 2.72548

# CMB temperature assumed by the AM code, in K
T_CMB_AM = 2.7

###################################################
# known arrays

known_arrays = {'EHT2017': ['ALMA',
                            'APEX',
                            'IRAM',
                            'JCMT',
                            'LMT',
                            'SMA',
                            'SMT',
                            'SPT'],
                'EHT2018': ['ALMA',
                            'APEX',
                            'GLT',
                            'IRAM',
                            'JCMT',
                            'LMT',
                            'SMA',
                            'SMT',
                            'SPT'],
                'EHT2021': ['ALMA',
                            'APEX',
                            'GLT',
                            'IRAM',
                            'JCMT',
                            'KP',
                            'LMT',
                            'NOEMA',
                            'SMA',
                            'SMT',
                            'SPT'],
                'EHT2022': ['ALMA',
                            'APEX',
                            'GLT',
                            'IRAM',
                            'JCMT',
                            'KP',
                            'LMT',
                            'NOEMA',
                            'SMA',
                            'SMT',
                            'SPT'],
                'ngEHTphase1': ['ALMA',
                                'APEX',
                                'BAJA',
                                'CNI',
                                'GLT',
                                'HAY',
                                'IRAM',
                                'JCMT',
                                'KP',
                                'LAS',
                                'LMT',
                                'OVRO',
                                'NOEMA',
                                'SMA',
                                'SMT',
                                'SPT'],
                'ngEHTphase2': ['ALMA',
                                'APEX',
                                'BAJA',
                                'BOL',
                                'BRZ',
                                'CAT',
                                'CNI',
                                'GAM',
                                'GARS',
                                'GLT',
                                'HAY',
                                'IRAM',
                                'JCMT',
                                'KP',
                                'LAS',
                                'LMT',
                                'OVRO',
                                'NOEMA',
                                'PIKE',
                                'SMA',
                                'SMT',
                                'SPT']}

known_array_overrides = {'EHT2017': {},
                         'EHT2018': {},
                         'EHT2021': {},
                         'EHT2022': {},
                         'ngEHTphase1': {'BAJA': 6.1,
                                         'CNI': 6.1,
                                         'LAS': 6.1,
                                         'OVRO': 10.4},
                         'ngEHTphase2': {'BAJA': 6.1,
                                         'CNI': 6.1,
                                         'LAS': 6.1,
                                         'OVRO': 10.4}}

###################################################
# other items

# translation dictionary between alternative names for some stations
translation_dict = {'AMT': 'GAM',
                    'GLT-S': 'GLTS',
                    'HOP': 'FLWO',
                    'IRAM-30m': 'IRAM',
                    'PDB': 'NOEMA',
                    'PIKES': 'PIKE',
                    'PV': 'IRAM',
                    'SOC': 'VLA'}
