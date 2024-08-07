#######################################################
# imports

import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.obs_plotter as op
import ngehtsim.metrics as cm

#######################################################
# generate an observation

# input settings file
yamlfile = './tests/data_generation/settings.yaml'

# initialize the observation generator
obsgen = og.obs_generator(settings_file=yamlfile)

# generate the observation
obs = obsgen.make_obs()

# save it as a uvfits file
obs.save_uvfits('./tests/data_generation/example_datafile.uvfits')

#######################################################
# make some plots of the data

op.plot_uv(obs, filename='./tests/data_generation/example_plot_uv.png')
op.plot_amp(obs, filename='./tests/data_generation/example_plot_amp.png')
op.plot_phase(obs, filename='./tests/data_generation/example_plot_phase.png')
op.plot_snr(obs, filename='./tests/data_generation/example_plot_snr.png')

#######################################################
# compute various metrics

# compute FF metric
ff = cm.calc_ff(obs, fov=200.0)
print('FF metric value is: ', ff)

# compute BFF metric for each Stokes parameter
for stokes in ['I', 'Q', 'U', 'V']:
    bff = cm.calc_bff(obs, fov=200.0, stokes=stokes)
    print('Stokes ' + stokes + ' BFF metric value is: ', bff)

# compute LCG metric
lcg = cm.calc_lcg(obs)
print('LCG metric value is: ', lcg)

# compute PSS metric
pss = cm.calc_pss(obs)
print('PSS metric value (in Jy) is: ', pss)

# compute angular resolution metric with different weightings
for weighting in ['natural', 'uniform', 'robust']:
    ar = cm.calc_ar(obs, artype='mean', weighting=weighting)
    print('Average beam size (in uas) with ' + weighting + ' weighting is: ', ar)

#######################################################
# plot a metric versus time for the observation

op.plot_snapshot(obs, obsgen, 'FF', fov=200.0, filename='./tests/data_generation/example_plot_FF.png')
op.plot_snapshot(obs, obsgen, 'LCG', filename='./tests/data_generation/example_plot_LCG.png')
op.plot_snapshot(obs, obsgen, 'BFF', filename='./tests/data_generation/example_plot_BFF.png')
op.plot_snapshot(obs, obsgen, 'PSS', filename='./tests/data_generation/example_plot_PSS.png')
op.plot_snapshot(obs, obsgen, 'AR', ylim=(0,100), filename='./tests/data_generation/example_plot_AR.png')
