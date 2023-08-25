#######################################################
# imports

import numpy as np
import ehtim as eh
import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.obs_plotter as op
import ngehtsim.metrics as cm

#######################################################
# generate an observation

# input settings file
yamlfile = './settings.yaml'

# some sites only have access to certain frequencies
receiver_configuration_overrides = {'ALMA': ['Band7'],
                                    'APEX': ['Band7'],
                                    'BAJA': ['Band3', 'Band6', 'Band7'],
                                    'CNI': ['Band3', 'Band6', 'Band7'],
                                    'GLT': ['Band3', 'Band6', 'Band7'],
                                    'HAY': ['Band3', 'Band6'],
                                    'IRAM': ['Band3', 'Band6'],
                                    'JCMT': ['Band3', 'Band6', 'Band7'],
                                    'JELM': ['Band3', 'Band6', 'Band7'],
                                    'KP': ['Band3', 'Band6'],
                                    'LAS': ['Band3', 'Band6', 'Band7'],
                                    'LMT': ['Band6'],
                                    'NOEMA': ['Band3', 'Band6'],
                                    'OVRO': ['Band3', 'Band6'],
                                    'SMA': ['Band7'],
                                    'SMT': ['Band3', 'Band6', 'Band7'],
                                    'SPT': ['Band3', 'Band6', 'Band7']}

# initialize the observation generator
obsgen = og.obs_generator(settings_file=yamlfile,verbosity=2,
                          receiver_configuration_overrides=receiver_configuration_overrides)

# specify the frequencies at which to observe
freqs = [86.0, 230.0, 345.0]

# specify the corresponding model files
input_models = ['../data_files/M87_86GHz_Chael.fits',
                '../data_files/M87_230GHz_Chael.fits',
                '../data_files/M87_345GHz_Chael.fits']

# generate the multi-frequency observation
obslist = obsgen.make_obs_mf(freqs,input_models)

# save the observations as uvfits files
for iobs, obs in enumerate(obslist):
    obs.save_uvfits('./datafile_'+str(int(freqs[iobs]))+'GHz.uvfits')

#######################################################
# make some plots of the data

for iobs, obs in enumerate(obslist):
    op.plot_uv(obs,filename='./plot_uv_'+str(int(freqs[iobs]))+'GHz.png',umax=15)
    op.plot_amp(obs,filename='./plot_amp_'+str(int(freqs[iobs]))+'GHz.png',xlim=(0,15),ylim=(0.001,10.0))
    op.plot_phase(obs,filename='./plot_phase_'+str(int(freqs[iobs]))+'GHz.png',xlim=(0,15))
    op.plot_snr(obs,filename='./plot_snr_'+str(int(freqs[iobs]))+'GHz.png',xlim=(0,15),ylim=(0.1,1.0e4))
