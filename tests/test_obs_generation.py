#######################################################
# imports

import numpy as np
import ehtim as eh
import yaml
import os
import ngehtsim.obs.obs_generator as og
import ngehtsim.const_def as const

#######################################################
# make an observation generator

# input settings file
yamlfile = './tests/settings.yaml'

# initialize the observation generator
obsgen = og.obs_generator(settings_file=yamlfile)

# load settings file
loader = yaml.SafeLoader
with open(yamlfile, 'r') as fi:
    settings = yaml.load(fi, Loader=loader)

# load input model
infile = settings['model_file']
input_model = eh.image.load_image(infile)

# generate the observation
obs = obsgen.make_obs()

#######################################################
# test that observation generation works with and
# without explicitly passing an image


def with_vs_without(obsgen):
    obs1 = obsgen.make_obs(addnoise=False, addgains=False)
    obs2 = obsgen.make_obs(input_model, addnoise=False, addgains=False)
    return obs1, obs2


def test_with_vs_without():
    obs1, obs2 = with_vs_without(obsgen)
    len1 = (obs1.data['vis'] == obs2.data['vis']).sum()
    len2 = len(obs1.data)
    assert len1 == len2

#######################################################
# test MJD conversion function


def test_MJD():
    trueval = 60714.0
    assert og.determine_mjd('8', 'Feb', '2025') == trueval

#######################################################
# test eta_dish


def test_eta_dish():
    freq = 230.0e9
    sigma = 60.0e-6
    offset = 60.0e-6
    ap_eff = 1.0
    trueval = ap_eff*np.exp(-((4*np.pi*np.sqrt((sigma)**2+(offset)**2))/(const.c/freq))**2)
    assert og.eta_dish(freq, sigma, offset, ap_eff) == trueval

# #######################################################
# # test weather data availability


# def test_weather_data_existence():
#     path_to_weather = os.path.abspath(const.path_to_weather)
#     sitelist = og.get_site_list()
#     monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
#     monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
#     yearlist = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]
#     weather_freqs = ['86','230','345','410','690']
#     failed_calls = list()
#     for site in sitelist:
#         for imonth, month in enumerate(monthnams):
#             for weather_freq in weather_freqs:
#                 pathhere = path_to_weather + '/'
#                 pathhere += site + '/'
#                 pathhere += monthnums[imonth] + month + '/'
#                 pathhere += 'mean_SEFD_info_' + weather_freq + '.csv'

#                 # read in the table
#                 try:
#                     year, monthdum, day, tau, Tb = np.loadtxt(pathhere,skiprows=7,unpack=True,delimiter=',')
#                 except:
#                     print(pathhere + ' does not exist!')
#                     failed_calls.append(pathhere)

#     # check that all files are present    
#     assert len(failed_calls) == 0


# def test_weather_data_completeness():
#     path_to_weather = os.path.abspath(const.path_to_weather)
#     sitelist = og.get_site_list()
#     monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
#     monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
#     yearlist = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]
#     weather_freqs = ['86','230','345','410','690']
#     failed_calls = list()
#     for site in sitelist:
#         for imonth, month in enumerate(monthnams):
#             for weather_freq in weather_freqs:
#                 pathhere = path_to_weather + '/'
#                 pathhere += site + '/'
#                 pathhere += monthnums[imonth] + month + '/'
#                 pathhere += 'mean_SEFD_info_' + weather_freq + '.csv'

#                 # read in the table
#                 try:
#                     year, monthdum, day, tau, Tb = np.loadtxt(pathhere,skiprows=7,unpack=True,delimiter=',')
#                     for yearhere in yearlist:
#                         if yearhere not in year:
#                             failed_calls.append([yearhere,pathhere])
#                             print(pathhere + ' seems to be missing ' + str(yearhere) + ' data!')
#                 except:
#                     print(pathhere + ' does not even exist!')

#     # check that all data are present    
#     assert len(failed_calls) == 0
