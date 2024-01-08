#######################################################
# imports

import ngehtsim.obs.obs_generator as og
import ngEHTforecast.fisher as fp
import ngehtsim.obs.obs_plotter as op
import ngehtsim.metrics as cm

#######################################################
# generate a FisherForecast object

ff = fp.FF_symmetric_gaussian()

#######################################################
# generate an observation

# initialize the observation generator
settings = {'source': 'M87',
            'array': 'EHT2017',
            'frequency': 230.0,
            'weather': 'typical'}
obsgen = og.obs_generator(settings)

# generate the observation by passing the FisherForecast object and parameters
p = [0.2, 20.0]
obs = obsgen.make_obs(ff, p=p, addnoise=False, addgains=False)

# save it as a uvfits file
obs.save_uvfits('./example_datafile.uvfits')

#######################################################
# make some plots of the data

op.plot_uv(obs, filename='./example_plot_uv.png')
op.plot_amp(obs, filename='./example_plot_amp.png')
op.plot_phase(obs, filename='./example_plot_phase.png')
op.plot_snr(obs, filename='./example_plot_snr.png')

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

op.plot_snapshot(obs, obsgen, 'FF', fov=200.0, filename='./example_plot_FF.png')
