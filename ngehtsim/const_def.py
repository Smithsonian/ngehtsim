import ngehtsim as ng
import os
from collections import OrderedDict
import numpy as np

###################################################
# some defaults

# absolute path to the weather information
path_to_weather = os.path.abspath(os.path.dirname(ng.__file__) + '/weather_data/')

# absolute path to the eigenspectra
path_to_eigenspectra = os.path.abspath(os.path.dirname(ng.__file__) + '/files/eigenspectra')

# absolute path to the telescope site matrix (TSM)
path_to_tsm = os.path.abspath(os.path.dirname(ng.__file__) + '/files/Telescope_Site_Matrix.csv')

# absolute path to the receiver file (RF)
path_to_rf = os.path.abspath(os.path.dirname(ng.__file__) + '/files/Receivers.csv')

default_settings = {'model_file': None,
                    'source': 'M87',
                    'frequency': 230.0,
                    'RA': None,
                    'DEC': None,
                    'bandwidth': 2.0,
                    'month': 'Jan',
                    'year': '2025',
                    'day': '15',
                    't_start': 0.0,
                    'dt': 24.0,
                    't_int': 600.0,
                    't_rest': 1200.0,
                    'SNR_cutoff': ['fringegroups', [5.0, 10.0]],
                    'array': None,
                    'sites': None,
                    'D_new': 10.0,
                    'tech_readiness': 1.0,
                    'weather': 'random',
                    'weather_year': 2015,
                    'weather_day': 15,
                    'ttype': 'fast',
                    'fft_pad_factor': 2,
                    'random_seed': None}

# minimum and maximum elevations at which a site may observe a source
el_min = 10.0
el_max = 80.0

# windspeed limitations
windspeed_degradation = 15.0
windspeed_shutdown = 25.0

# minimum and maximum years from which to query weather data
year_min = 2012
year_max = 2022

# default aperture efficiency (beyond what is accounted for by surface and focus RMS)
ap_eff = 0.7

# quantization efficiency
quant_eff = 0.88

# default surface RMS, in microns
surf_rms = 40.0

# default focus offset, in microns
focus_offset = 10.0

# default mount type, if not otherwise known
mount_type = 'ALT-AZ'

# default feed angle, if not otherwise known
feed_angle = 0.0

###################################################
# physical constants

# Boltzmann constant, in Jy m^2 / K
k = 1381.0

# speed of light, in m/s
c = 299792458.0

# Earth diameter, in m
D_Earth = 12742000.0

# CMB temperature, in K
T_CMB = 2.725

###################################################
# recognized receiver bands and properties

rec_name = np.loadtxt(path_to_rf,
                      delimiter=',',
                      skiprows=1,
                      usecols=(0),
                      dtype=str)

rec_lo, rec_hi, rec_T, rec_SSR = np.loadtxt(path_to_rf,
                                            delimiter=',',
                                            skiprows=1,
                                            usecols=(1,2,3,4),
                                            unpack=True)

receivers = {}
for i in range(len(rec_name)):
    receivers[rec_name[i]] = {'lo': rec_lo[i], 'hi': rec_hi[i], 'T_R': rec_T[i], 'SSR': rec_SSR[i]}

###################################################
# known arrays

known_arrays = {}

# known_arrays = {'EHT2017': ['ALMA',
#                             'APEX',
#                             'IRAM',
#                             'JCMT',
#                             'LMT',
#                             'SMA',
#                             'SMT',
#                             'SPT'],
#                 'EHT2018': ['ALMA',
#                             'APEX',
#                             'GLT',
#                             'IRAM',
#                             'JCMT',
#                             'LMT',
#                             'SMA',
#                             'SMT',
#                             'SPT'],
#                 'EHT2021': ['ALMA',
#                             'APEX',
#                             'GLT',
#                             'IRAM',
#                             'JCMT',
#                             'KP',
#                             'LMT',
#                             'NOEMA',
#                             'SMA',
#                             'SMT',
#                             'SPT'],
#                 'EHT2022': ['ALMA',
#                             'APEX',
#                             'GLT',
#                             'IRAM',
#                             'JCMT',
#                             'KP',
#                             'LMT',
#                             'NOEMA',
#                             'SMA',
#                             'SMT',
#                             'SPT'],
#                 'ngEHTphase1': ['ALMA',
#                                 'APEX',
#                                 'BAJA',
#                                 'CNI',
#                                 'GLT',
#                                 'HAY',
#                                 'IRAM',
#                                 'JCMT',
#                                 'KP',
#                                 'LAS',
#                                 'LMT',
#                                 'OVRO',
#                                 'NOEMA',
#                                 'SMA',
#                                 'SMT',
#                                 'SPT'],
#                 'ngEHTphase2': ['ALMA',
#                                 'APEX',
#                                 'BAJA',
#                                 'BOL',
#                                 'BRZ',
#                                 'CAT',
#                                 'CNI',
#                                 'GAM',
#                                 'GARS',
#                                 'GLT',
#                                 'HAY',
#                                 'IRAM',
#                                 'JCMT',
#                                 'KP',
#                                 'LAS',
#                                 'LMT',
#                                 'OVRO',
#                                 'NOEMA',
#                                 'PIKE',
#                                 'SMA',
#                                 'SMT',
#                                 'SPT']}

# known_array_D_overrides = {'EHT2017': {},
#                            'EHT2018': {},
#                            'EHT2021': {},
#                            'EHT2022': {},
#                            'ngEHTphase1': {'BAJA': 6.1,
#                                            'CNI': 6.1,
#                                            'LAS': 6.1,
#                                            'OVRO': 10.4},
#                            'ngEHTphase2': {'BAJA': 6.1,
#                                            'CNI': 6.1,
#                                            'LAS': 6.1,
#                                            'OVRO': 10.4}}

# known_array_receiver_overrides = {'EHT2017': {'ALMA': ['230'],
#                                               'APEX': ['230'],
#                                               'IRAM': ['230'],
#                                               'JCMT': ['230'],
#                                               'LMT': ['230'],
#                                               'SMA': ['230'],
#                                               'SMT': ['230'],
#                                               'SPT': ['230']},
#                                   'EHT2018': {'ALMA': ['230'],
#                                               'APEX': ['230'],
#                                               'GLT': ['230'],
#                                               'IRAM': ['230'],
#                                               'JCMT': ['230'],
#                                               'LMT': ['230'],
#                                               'SMA': ['230'],
#                                               'SMT': ['230'],
#                                               'SPT': ['230']},
#                                   'EHT2021': {'ALMA': ['230'],
#                                               'APEX': ['230'],
#                                               'GLT': ['230'],
#                                               'IRAM': ['230'],
#                                               'JCMT': ['230'],
#                                               'KP': ['230'],
#                                               'LMT': ['230'],
#                                               'NOEMA': ['230'],
#                                               'SMA': ['230'],
#                                               'SMT': ['230'],
#                                               'SPT': ['230']},
#                                   'EHT2022': {'ALMA': ['230'],
#                                               'APEX': ['230'],
#                                               'GLT': ['230'],
#                                               'IRAM': ['230'],
#                                               'JCMT': ['230'],
#                                               'KP': ['230'],
#                                               'LMT': ['230'],
#                                               'NOEMA': ['230'],
#                                               'SMA': ['230'],
#                                               'SMT': ['230'],
#                                               'SPT': ['230']},
#                                   'ngEHTphase1': {'ALMA': ['86', '230', '345'],
#                                                   'APEX': ['86', '230', '345'],
#                                                   'BAJA': ['86', '230', '345'],
#                                                   'CNI': ['86', '230', '345'],
#                                                   'GLT': ['86', '230', '345'],
#                                                   'HAY': ['86', '230', '345'],
#                                                   'IRAM': ['86', '230', '345'],
#                                                   'JCMT': ['86', '230', '345'],
#                                                   'KP': ['86', '230', '345'],
#                                                   'LAS': ['86', '230', '345'],
#                                                   'LMT': ['86', '230', '345'],
#                                                   'OVRO': ['86', '230', '345'],
#                                                   'NOEMA': ['86', '230', '345'],
#                                                   'SMA': ['86', '230', '345'],
#                                                   'SMT': ['86', '230', '345'],
#                                                   'SPT': ['86', '230', '345']},
#                                   'ngEHTphase2': {'ALMA': ['86', '230', '345'],
#                                                   'APEX': ['86', '230', '345'],
#                                                   'BAJA': ['86', '230', '345'],
#                                                   'BOL': ['86', '230', '345'],
#                                                   'BRZ': ['86', '230', '345'],
#                                                   'CAT': ['86', '230', '345'],
#                                                   'CNI': ['86', '230', '345'],
#                                                   'GAM': ['86', '230', '345'],
#                                                   'GARS': ['86', '230', '345'],
#                                                   'GLT': ['86', '230', '345'],
#                                                   'HAY': ['86', '230', '345'],
#                                                   'IRAM': ['86', '230', '345'],
#                                                   'JCMT': ['86', '230', '345'],
#                                                   'KP': ['86', '230', '345'],
#                                                   'LAS': ['86', '230', '345'],
#                                                   'LMT': ['86', '230', '345'],
#                                                   'OVRO': ['86', '230', '345'],
#                                                   'NOEMA': ['86', '230', '345'],
#                                                   'PIKE': ['86', '230', '345'],
#                                                   'SMA': ['86', '230', '345'],
#                                                   'SMT': ['86', '230', '345'],
#                                                   'SPT': ['86', '230', '345']}}

###################################################
# known sources

known_sources = {'M87': {'RA': 12.51373,
                         'DEC': 12.39112},
                 'M 87': {'RA': 12.51373,
                          'DEC': 12.39112},
                 'SgrA': {'RA': 17.76112,
                          'DEC': -29.007797},
                 'Sgr A': {'RA': 17.76112,
                           'DEC': -29.007797},
                 'SgrA*': {'RA': 17.76112,
                           'DEC': -29.007797},
                 'Sgr A*': {'RA': 17.76112,
                            'DEC': -29.007797},
                 '3C279': {'RA': 12.93642,
                           'DEC': -5.78944},
                 '3C 279': {'RA': 12.93642,
                            'DEC': -5.78944},}

###################################################
# pull antenna properties from table

known_stations, tlcs, diam_arr, surf_arr, mnts_arr, fa_arr, altnames = np.loadtxt(path_to_tsm,
                                                                                  delimiter=',',
                                                                                  skiprows=1,
                                                                                  usecols=(0,1,7,8,9,10,11),
                                                                                  dtype=str,
                                                                                  unpack=True)
lat_arr, lon_arr, elev_arr = np.loadtxt(path_to_tsm,
                                        delimiter=',',
                                        skiprows=1,
                                        usecols=(3,4,5),
                                        unpack=True)

known_diameters = {}
for i in range(len(known_stations)):
    if (diam_arr[i] != ''):
        known_diameters[known_stations[i]] = float(diam_arr[i])

known_surf_rms = {}
for i in range(len(known_stations)):
    if (surf_arr[i] != ''):
        known_surf_rms[known_stations[i]] = float(surf_arr[i])

known_latitudes = {}
for i in range(len(known_stations)):
    if (lat_arr[i] != ''):
        known_latitudes[known_stations[i]] = lat_arr[i]

known_longitudes = {}
for i in range(len(known_stations)):
    if (lon_arr[i] != ''):
        known_longitudes[known_stations[i]] = lon_arr[i]

known_elevations = {}
for i in range(len(known_stations)):
    if (elev_arr[i] != ''):
        known_elevations[known_stations[i]] = elev_arr[i]

known_mount_types = {}
for i in range(len(known_stations)):
    if (mnts_arr[i] != ''):
        known_mount_types[known_stations[i]] = mnts_arr[i]

known_feed_angles = {}
for i in range(len(known_stations)):
    if (fa_arr[i] != ''):
        known_feed_angles[known_stations[i]] = float(fa_arr[i])

# translation between full and two-letter station codes
two_letter_station_codes = {}
for i in range(len(known_stations)):
    if (tlcs[i] != ''):
        two_letter_station_codes[known_stations[i]] = tlcs[i]

# translation dictionary between alternative names for some stations
translation_dict = {}
for i in range(len(known_stations)):
    if (altnames[i] != ''):
        altname_list = altnames[i].split(';')
        for altname in altname_list:
            translation_dict[altname.strip()] = known_stations[i]
    if (tlcs[i] != ''):
        translation_dict[tlcs[i]] = known_stations[i]

###################################################
# other items

# PCA spectral decomposition quantities
number_of_components = 40
length_of_spectrum = 2001
spectrum_frequency = np.linspace(0.0,2000.0,length_of_spectrum)

SYMBA_master_input_arguments = OrderedDict({'rpicard_path': '/usr/local/src/picard/input_template',
                                            'meqsilhouette_path': '/usr/local/src/MeqSilhouette/meqsilhouette/data',
                                            'outdirname': '/data',
                                            'ms_antenna_table': '/usr/local/src/symba/symba_input/VLBIarrays/eht.antennas',
                                            'rpicard_refants': 'AA, LM, SM, PV',
                                            'fringe_solint': '0.0001;2;3;4;5;6;7;8;9;10;15',
                                            'obs_vex': 'False',
                                            'vexfile': '/usr/local/src/symba/symba_input/vex_examples/EHT2017/e17e11.vex',
                                            'vex_source': 'M87',
                                            'input_fitsimage': '/usr/local/src/symba/symba_input/fits_examples/*.fits',
                                            'input_fitspol': 'False',
                                            'input_changroups': '1',
                                            'model_rotangle': '0.0',
                                            'model_scale': '1.0',
                                            'mod_minflux': '0.0',
                                            'frameduration': '99999999',
                                            'loop_movie': 'True',
                                            'do_netcal': 'True',
                                            'time_avg': '10s',
                                            'match_uv': 'False',
                                            'matchuv_rawdata': 'False',
                                            'realdata_uvfits': '/usr/local/src/symba/symba_input/uvfits/simobs_e17e11cov.uvfits',
                                            'add_scattering': 'False',
                                            'scattering_vx': '0.0',
                                            'scattering_vy': '0.0',
                                            'quantization_efficiency': str(quant_eff),
                                            'ms_dnu': '2',
                                            'ms_nchan': '64',
                                            'flag_instructions': 'None',
                                            'do_gausscal': 'False',
                                            'gain_tol': '0.2',
                                            'bandpass_enabled': 'False',
                                            'bandpass_txt': '/usr/local/src/symba/symba_input/example.bandpass',
                                            'skyfreq': '230',
                                            'ms_RA': '187.7059308',
                                            'ms_DEC': '12.39112331',
                                            'ms_StartTime': 'UTC,2017/04/01/00:00:00.00',
                                            'ms_obslength': '2.2',
                                            'ms_nscan': '22',
                                            'ms_scan_lag': '0.26',
                                            'ms_tint': '0.5',
                                            'elevation_limit': str(el_min),
                                            'ms_correctCASAoffset': 'True',
                                            'add_thermal_noise': 'False',
                                            'trop_enabled': 'True',
                                            'trop_attenuate': 'True',
                                            'trop_turbulence': 'True',
                                            'trop_mean_delay': 'True',
                                            'trop_percentage_calibration_error': '100',
                                            'variable_weather': 'False',
                                            'variable_weatherfile': '/usr/local/src/symba/symba_input/VLBIarrays/variable_weather/variableweather_example.txt',
                                            'pointing_enabled': 'False',
                                            'pointing_rms2offset': '1.5',
                                            'Nscan_repoint': '5',
                                            'Nscan_pointing_grow': '0.03',
                                            'uvjones_g_on': 'False',
                                            'uvjones_d_on': 'False',
                                            'parang_corrected': 'True',
                                            'predict_seed': '-1',
                                            'cpDATAtoMODEL': 'False',
                                            'rpicard_fullpipeline': 'True',
                                            'rpicard_steps': '-rq c,h,i,l',
                                            'rpicard_fringecuts': '3.0',
                                            'make_image': 'False',
                                            'fov_image': '200',
                                            'keep_rawdata': 'False',
                                            'N_cores': '0'})

SYMBA_master_input_comments = {'rpicard_path': '#Path to rPICARD input.',
                               'meqsilhouette_path': '#Path to MeqSilhouette input.',
                               'outdirname': '#Working directory where the output files will be created.',
                               'ms_antenna_table': '#Info about antennas for the synthetic observation.',
                               'rpicard_refants': '#Ordered list of preferred fringe-fit reference antennas from the ms_antenna_table.',
                               'fringe_solint': '#Solint search range for fringe-fit (see rPICARD).',
                               'obs_vex': '#Mimic a real VLBI observation based on a vex file.',
                               'vexfile': '',
                               'vex_source': '#Name of the observed source. Must be present in vexfile if a vexfile is given.',
                               'input_fitsimage': '#Observed sourcemodel. Must be a square image fits file, square image .txt file (as read by\n#eht-imaging, see examples), or MeqSilhouette .txt sky model.\n#When observing a time variable source: Set path/to/folder/*.fits or path/to/folder/*.txt here\n#and put individual files in that folder according to frames of a movie with an ordering naming\n#convention of xxxxxx.fits or xxxxxx.txt with xxxxxx from 000000 to 999999.\n#For polarized fits models, use xxxxxx-{I,Q,U,V}-model.fits. For txt models, the polarization\n#information is given in the files themselves.\n#For frequency-resolved models, use xxxxxx-yyyy-{I,Q,U,V}-model.fits with yyyy the frequency\n#index from 0000 to 9999',
                               'input_fitspol': '#Flag for polarized input models.',
                               'input_changroups': '#Number of frequencies in the input source model.\n#This should not be larger than the number of channels set below.',
                               'model_rotangle': '#Rotation angle for input model in radians',
                               'model_scale': '#Field of view scaling factor for input model\n#Done in the uv-plane by default. Add a ! to the number (e.g., 2.0!), for a simple pixel size scaling.\n#The pixel size scaling does not work for MeqSilhouette .txt sky models.',
                               'mod_minflux': '#Rescale model fluxes such that their total flux never drops below the specified value in Jy.\n#Does not work for MeqSilhouette .txt sky models.\n#Can give a number with an exclamation mark, to always enforce the specified total flux (can also\n#downscale the total flux).',
                               'frameduration': '#Duration [in seconds] of a single .fits frame for a movie of a time variable source.\n#Set it to a large number like 99999999 for a static source with a single fits file.',
                               'loop_movie': '#When observing a time variable source:\n#Restart movie in a loop if the duration of the simulation is shorter than the observation.',
                               'do_netcal': '#Perform network calibration. Stations eligible are determined automatically based on uv-distances.\n#The zero-baseline flux is taken directly from the input source model file.',
                               'time_avg': "#Time-averaging of final data products after calibration.\n#Set to False for no averaging, set to '5s' for example to average in 5second bins.\n#This time will also be used as the netcal solution interval. If set to False, 10s will be used\n#for the network calibration.",
                               'match_uv': '#Match uv points based on comparison with real data:\n#The uv-coverage will be matched to the file specified as realdata_uvfits below.',
                               'matchuv_rawdata': '#Match uv points based on comparison with real data before calibration:\n#The uv-coverage will be matched to the file specified as realdata_uvfits below.',
                               'realdata_uvfits': '#UVFITS file of real data to compare uv coverage.',
                               'add_scattering': '#Can specify a file with scattering parameters to apply a scattering screen to the input images.\n#Does not work for MeqSilhouette .txt sky models.',
                               'scattering_vx': '#Velocities of the scattering screen in the x and y directions.\n#If both are zero, a constant scattering screen is used.\n#A time-variable scattering screen with non-zero velocities only works for\n#h5 and unpolarized fits input files.',
                               'scattering_vy': '',
                               'quantization_efficiency': '#Quantization efficiency factor, see Sections 8.3 and 8.4 in Thompson, Moran, Swenson 2017.\n#The default value of 0.88 is appropriate for 2-bit quantization.',
                               'ms_dnu': '#Frequency setup of synthetic data:\n#Bandwidth [GHz] and number of channels.\n#For a frequency-resolved source, the bandwidth should be set to the source frequency range.',
                               'ms_nchan': '',
                               'flag_instructions': '#Can give a filename here, which contains flagging instructions in the CASA flagdata format.\n#These flags will be applied after the data has been generated with MeqSilhouette and before it is\n#passed to rPICARD for calibration.',
                               'do_gausscal': '#Perform LZgauss_flux calibration in eht-imager.',
                               'gain_tol': '#Gain tolerance for network calibration and imaging.',
                               'bandpass_enabled': '',
                               'bandpass_txt': '',
                               'skyfreq': '#Observing frequency in GHz when a .txt source model is specified as input_fitsimage.\n#This parameter is ignored for a fits input_fitsimage, where the frequency is read from the fits\n#header.',
                               'ms_RA': '#Sky coordinates [in degrees] of the observed source, when no vex file is given.\n#With a vex file, this parameter is ignored and the coordinates are taken from the file directly.\n#Relative position offsets between multiple geometric models are unaffected.',
                               'ms_DEC': '',
                               'ms_StartTime': '#Observation schedule parameters used when no vex file is given:\n#Start time, length [h], number of scans, gaps between scans [h].',
                               'ms_obslength': '',
                               'ms_nscan': '',
                               'ms_scan_lag': '',
                               'ms_tint': '#Accumulation period (data integration time or correlator dump time) in seconds.',
                               'elevation_limit': '#Elevation limit in degrees. Data below this elevation will be flagged.',
                               'ms_correctCASAoffset': '#Correct for ~30 s start time offset introduced by CASA',
                               'add_thermal_noise': '',
                               'trop_enabled': '#Tropospheric corruption effects.',
                               'trop_attenuate': '',
                               'trop_turbulence': '',
                               'trop_mean_delay': '',
                               'trop_percentage_calibration_error': '',
                               'variable_weather': '#Option to observe with variable weather specified in a txt file.',
                               'variable_weatherfile': '',
                               'pointing_enabled': '#If enabled, one random pointing offset error is used for every N_scan_repoint(see below) scans.\n#Based on the ptg_rms from ms_antenna_table multiplied by a .\n#Note that the PB_FWHM230 beam sizes in the ms_antenna_table are the 230GHz beam sizes. The beam\n#sizes used by the code will be a factor skyfreq_in_GHz/230 smaller.',
                               'pointing_rms2offset': '#The ptg_rms values in the ms_antenna_table introduce variability in the amplitudes on short\n#timescales given by the coherence time. The idea is that this is due to the atmospheric seeing\n#and wind shaking the telescope. The ptg_rms values are also used to compute pointing offsets\n#for every N_scan_repoint scans if pointing is enabled, after multiplying the rms values with the\n#factor set here.',
                               'Nscan_repoint': '#If pointing is enabled, set the typical cadence, in number of VLBI scans, used to adopt new\n#pointing solutions. The actual re-pointing times are determined randomly.',
                               'Nscan_pointing_grow': '#Grow the pointing offsets by a set fraction for every scan until re-pointing.',
                               'uvjones_g_on': '#Enable gain errors.',
                               'uvjones_d_on': '#Enable leakage corruptions.',
                               'parang_corrected': '#If dterms are included: correct for parallactic angle rotation in final dataset.',
                               'predict_seed': '#Seed value allowing to reproduce realizations of thermal noise, antenna pointing, and\n#atmospheric turbulence. Set to -1 to run without seed.',
                               'cpDATAtoMODEL': '#Can specify a MS here, for which the DATA column will be copied to the MODEL_DATA of the SYMBA MS\n#before it is passed to rPicard. The DATA will therefore serve as a MODEL for the calibration. The\n#rows of the two MS must match exactly for this to work properly.',
                               'rpicard_fullpipeline': '#Run the full calibration pipeline (steps -rq 0,1,2,h,i,l see rPICARD).',
                               'rpicard_steps': '#If the rpicard fullpipeline parameter above is set to False,\n#the calibration steps to be executed can be set here.',
                               'rpicard_fringecuts': '#If >0, change the default rpicard fringe-fit SNR cuts to the number specified.',
                               'make_image': '#Make image with fiducial eht-imaging script.',
                               'fov_image': '#Field of view of the reconstructed image [in micro arcseconds].',
                               'keep_rawdata': '#Keep auxiliary simulation data created by MeqSilhouette.',
                               'N_cores': '#Number of CPU cores to be used for rPICARD (set to 0 to use all available).'}
