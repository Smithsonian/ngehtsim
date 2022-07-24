#######################################################
# imports

import numpy as np
import ngehtsim.obs.obs_generator as og

#######################################################
# make an observation generator

# input settings file
yamlfile = './tests/settings.yaml'

# initialize the observation generator
obsgen = og.obs_generator(yamlfile)

#######################################################
# make a trivial initial test, just to get CI working

def settings_test(obsgen):
    return obsgen.settings['month']

def test_answer():
    assert settings_test(obsgen) == 'Jan'

