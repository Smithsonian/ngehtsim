import ngehtsim as ng
import os

###################################################
# useful constants

# relative path to the weather information
path_to_weather = os.path.dirname(ng.__file__) + '/weather_data/'

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

# minimum and maximum elevations at which a site may observe a source
el_min = 10.0
el_max = 80.0

# minimum and maximum years from which to query weather data
year_min = 2009
year_max = 2018

# fiducial receiver temperatures, in K
T_R_86 = 40.0
T_R_230 = 50.0
T_R_345 = 75.0

# quantization efficiency
quant_eff = 0.88

# fiducial focus offset, in effective surface accuracy units
focus_offset = 50.0e-6

# fiducial surface RMS, in meters
sigma_surface = 40.0e-6

# fiducial pointing accuracy factor for existing telescopes, such that pointing RMS = primary_beamsize / accuracy_factor
existing_pt_accuracy_factor = 10.0

# diameters of existing telescopes, in meters
D_existing_dict = {'ALMA':73.0,
                   'APEX':12.0,
                   'GAM':15.0,
                   'GLT':12.0,
                   'HAY':37.0,
                   'JCMT':15.0,
                   'KP':12.0,
                   'KVNYS':21.0,
                   'LAS':15.0,
                   'LMT':50.0,
                   'NOB':45.0,
                   'NOEMA':52.0,
                   'OVRO':10.0,
                   'PV':30.0,
                   'SMA':14.7,
                   'SMT':10.0,
                   'SPT':6.0,
                   'SUF':70.0}

# surface RMS of existing telescopes, in meters
sigma_existing_dict = {'ALMA': 65.0e-6,
                       'APEX': 73.0e-6,
                       'GAM':  65.0e-6,
                       'GLT':  68.0e-6,
                       'HAY':  100.0e-6,
                       'JCMT': 84.0e-6,
                       'KP':   75.0e-6,
                       'KVNYS':124.0e-6,
                       'LAS':  65.0e-6,
                       'LMT':  117.0e-6,
                       'NOB':  100.0e-6,
                       'NOEMA':86.0e-6,
                       'OVRO': 50.0e-6,
                       'PV':   90.0e-6,
                       'SMA':  62.0e-6,
                       'SMT':  74.0e-6,
                       'SPT':  74.0e-6,
                       'SUF':  150.0e-6}
