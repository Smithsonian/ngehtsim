#######################################################
# imports

import numpy as np
import ehtim as eh
import yaml
import ngehtsim.obs.obs_generator as og

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
# make a trivial initial test, just to get CI working


def settings_test(obsgen):
    return obsgen.settings['month']


def test_answer():
    assert settings_test(obsgen) == 'Jan'


#######################################################
# test that observation generation works with and
# without explicitly passing an image


def with_vs_without(obsgen):
    obs1 = obsgen.make_obs(addnoise=False,addgains=False)
    obs2 = obsgen.make_obs(input_model,addnoise=False,addgains=False)
    return obs1, obs2

def test_with_vs_without():
    obs1, obs2 = with_vs_without(obsgen)
    len1 = (obs1.data['vis'] == obs2.data['vis']).sum()
    len2 = len(obs1.data)
    assert len1 == len2

