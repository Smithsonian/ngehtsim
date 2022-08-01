#######################################################
# imports

import ngehtsim.obs.obs_generator as og
import ngEHTforecast.fisher as fp

#######################################################

# generate a FisherForecast object
ff = fp.FF_symmetric_gaussian()

# initialize the observation generator
settings = {'weather': 'typical'}
obsgen = og.obs_generator(settings)

# generate the observation by passing the FisherForecast object and parameters
p = [0.2,20.0]
obs = obsgen.make_obs(ff,p=p,addnoise=False,addgains=False)

# export a SYMBA-compatible input file
obsgen.export_SYMBA()
