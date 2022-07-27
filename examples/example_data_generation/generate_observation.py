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

# initialize the observation generator
obsgen = og.obs_generator(settings_file=yamlfile)

# generate the observation
obs = obsgen.make_obs()

# save it as a uvfits file
obs.save_uvfits('./example_datafile.uvfits')

print(obsgen.tau_dict)
print(obsgen.Tb_dict)

#######################################################
# make some plots of the data

op.plot_uv(obs,filename='./example_plot_uv.png')
op.plot_amp(obs,filename='./example_plot_amp.png')
op.plot_phase(obs,filename='./example_plot_phase.png')
op.plot_snr(obs,filename='./example_plot_snr.png')

#######################################################
# compute various metrics

# compute FF metric
ff = cm.calc_ff(obs,fov=200.0)
print('FF metric value is: ',ff)

# compute BFF metric for each Stokes parameter
for stokes in ['I','Q','U','V']:
    bff = cm.calc_bff(obs,fov=200.0,stokes=stokes)
    print('Stokes ' + stokes + ' BFF metric value is: ',bff)

# compute LCG metric
lcg = cm.calc_lcg(obs)
print('LCG metric value is: ',lcg)

# compute PSS metric
pss = cm.calc_pss(obs)
print('PSS metric value (in Jy) is: ',pss)

# compute angular resolution metric with different weightings
for weighting in ['natural','uniform','robust']:
    ar = cm.calc_ar(obs,artype='mean',weighting=weighting)
    print('Average beam size (in uas) with ' + weighting + ' weighting is: ',ar)

# compute array cost metric
total_cost, operating_cost = cm.calc_cost(obs)
print('Total cost to build this array is: $%.2fM' % (total_cost/(1.0e6)))
print('Annual operating cost of this array is: $%.2fM' % (operating_cost/(1.0e6)))

#######################################################
# plot a metric versus time for the observation

op.plot_snapshot(obs,obsgen,'FF',fov=200.0,filename='./example_plot_FF.png')
