#######################################################
# imports

import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.obs_plotter as op

#######################################################
# generate an observation

# input settings file
yamlfile = './settings.yaml'

# override some of the settings in the settings file
sites = ['ALMA', 'APEX', 'GLT', 'JCMT', 'KP', 'LMT', 'NOEMA', 'PV', 'SMA',
         'SMT', 'SPT', 'space']
settings = {'sites': sites,
            'D_new': 3.5,
            'frequency': 230}

# initialize the observation generator
obsgen = og.obs_generator(settings, settings_file=yamlfile, ephem='ephemeris/space')

# generate the observation
obs = obsgen.make_obs()

# save it as a uvfits file
obs.save_uvfits('./example_datafile.uvfits')

#######################################################
# make some plots of the data

op.plot_uv(obs, filename='./example_plot_uv.png', umax=40)
op.plot_amp(obs, filename='./example_plot_amp.png', xlim=(0, 40), ylim=(1.0e-3, 1.0))
op.plot_phase(obs, filename='./example_plot_phase.png', xlim=(0, 40))
op.plot_snr(obs, filename='./example_plot_snr.png', xlim=(0, 40))
