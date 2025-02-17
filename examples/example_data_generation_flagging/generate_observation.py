#######################################################
# imports

import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.obs_plotter as op
import ngehtsim.metrics as cm
import matplotlib.pyplot as plt

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

#######################################################
# generate an observation with flagging

# initialize the observation generator
obsgen_flagged = og.obs_generator(settings_file=yamlfile)

# generate the observation
obs_flagged = obsgen_flagged.make_obs(flagday=True)

# save it as a uvfits file
obs_flagged.save_uvfits('./example_datafile_flagged.uvfits')

#######################################################
# generate an observation with flagging

station_uptimes = {'ALMA': [0.0,24.0],
                   'APEX': [0.0,24.0],
                   'GLT': [4.0,5.0],
                   'IRAM': [0.0,24.0],
                   'JCMT': [0.0,24.0],
                   'KP': [0.0,24.0],
                   'LMT': [0.0,24.0],
                   'NOEMA': [0.0,24.0],
                   'SMA': [0.0,24.0],
                   'SMT': [0.0,24.0],
                   'SPT': [0.0,24.0]}

# some sites have different wind tolerances
wind_loading_overrides = {'LMT': {'v0': 5.0, 'w': 1.0, 'shutdown': 10.0},
                          'IRAM': {'v0': 10.0, 'w': 2.0, 'shutdown': 15.0}}

# initialize the observation generator
obsgen_flagged2 = og.obs_generator(settings_file=yamlfile,
                                   station_uptimes=station_uptimes,
                                   wind_loading_overrides=wind_loading_overrides)

# generate the observation
obs_flagged2 = obsgen_flagged2.make_obs(flagday=True,flagwind=True)

# save it as a uvfits file
obs_flagged2.save_uvfits('./example_datafile_flagged2.uvfits')

#######################################################
# plot

u1 = obs.data['u'] / (1.0e9)
v1 = obs.data['v'] / (1.0e9)
u2 = obs_flagged.data['u'] / (1.0e9)
v2 = obs_flagged.data['v'] / (1.0e9)
u3 = obs_flagged2.data['u'] / (1.0e9)
v3 = obs_flagged2.data['v'] / (1.0e9)

fig = plt.figure(figsize=(4,4))
ax = fig.add_axes([0.1,0.1,0.8,0.8])

ax.plot(u1,v1,'k.',markersize=4)
ax.plot(-u1,-v1,'k.',markersize=4)

ax.plot(u2,v2,'r.',markersize=4)
ax.plot(-u2,-v2,'r.',markersize=4)

ax.plot(u3,v3,'g.',markersize=2)
ax.plot(-u3,-v3,'g.',markersize=2)

ax.set_xlim(10,-10)
ax.set_ylim(-10,10)

ax.set_xlabel(r'u (G$\lambda$)')
ax.set_ylabel(r'v (G$\lambda$)')

plt.savefig('uv-coverage.png',dpi=300,bbox_inches='tight')
plt.close()
