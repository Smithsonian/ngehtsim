#######################################################
# imports

import ehtim as eh
import ngehtsim.obs.obs_generator as og

#######################################################
# generate an observation

# input settings file
yamlfile = 'example_script/settings.yaml'

# initialize the observation generator
obsgen = og.obs_generator(yamlfile)

# generate the observation
obs = obsgen.make_obs()

# save it as a uvfits file
obs.save_uvfits('example_script/example_datafile.uvfits')

