#######################################################
# imports

import numpy as np
import ehtim as eh
import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.obs_plotter as op

#######################################################
# generate an observation

sites = ['ALMA','APEX','GLT','JCMT','KP','LMT','NOEMA',
         'PV','SMA','SMT','SPT','space']
settings = {'sites': sites,
            'D_new': 100.0}

# input settings file
yamlfile = './settings.yaml'

# initialize the observation generator
obsgen = og.obs_generator(settings,settings_file=yamlfile)

# generate the observation
obs = obsgen.make_obs()

# save it as a uvfits file
obs.save_uvfits('./example_datafile.uvfits')

#######################################################
# make some plots of the data

op.plot_uv(obs,filename='./example_plot_uv.png',umax=60)
