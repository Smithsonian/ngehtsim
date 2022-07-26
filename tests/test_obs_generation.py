#######################################################
# imports

import numpy as np
import ehtim as eh
import yaml
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
# test array dish override


def test_diameter_override():
    sitelist = ['ALMA', 'APEX', 'JCMT', 'LMT', 'SMT']
    D_new = 10.0
    D_override_dict = {'APEX': 100.0}
    array, arr = og.make_array(sitelist, D_new, D_override_dict=D_override_dict)
    assert array.stations()[1].diameter() == D_override_dict['APEX']

#######################################################
# test eta_dish


def test_eta_dish():
    freq = 230.0e9
    sigma = 60.0e-6
    offset = 60.0e-6
    trueval = np.exp(-((4*np.pi*np.sqrt((sigma)**2+(offset)**2))/(const.c/freq))**2)
    assert og.eta_dish(freq, sigma, offset) == trueval
