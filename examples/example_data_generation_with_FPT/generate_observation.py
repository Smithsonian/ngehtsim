#######################################################
# imports

import numpy as np
import matplotlib.pyplot as plt
import ngehtsim.obs.obs_generator as og

#######################################################
# generate an observation with FPT

# input settings file
yamlfile = './settings_fpt.yaml'

# specify a custom W-band receiver that some sites will use
custom_receivers = {'Wband': {'lo': 67.0, 'hi': 116.0, 'T_R': 30.0, 'SSR': 0.1}}

# some sites have modified diameters compared to their defaults
D_overrides = {'LMT': 32.5,
               'SMA': 14.7}

# some sites only have access to certain frequencies
receiver_configuration_overrides = {'ALMA': ['Band7'],
                                    'APEX': ['Band7'],
                                    'BAJA': ['Band3', 'Band6', 'Band7'],
                                    'CNI': ['Band3', 'Band6', 'Band7'],
                                    'GLT': ['Wband', 'Band6', 'Band7'],
                                    'IRAM': ['Band7'],
                                    'JCMT': ['Band3', 'Band6', 'Band7'],
                                    'JELM': ['Band3', 'Band6', 'Band7'],
                                    'KP': ['Wband', 'Band6'],
                                    'LAS': ['Band3', 'Band6', 'Band7'],
                                    'LMT': ['Band6'],
                                    'NOEMA': ['Band7'],
                                    'SMA': ['Band7'],
                                    'SMT': ['Wband', 'Band6', 'Band7'],
                                    'SPT': ['Band3', 'Band6', 'Band7']}

# some sites can record different bandwidths
bandwidth_overrides = {'BAJA': {'Band3': 8.0, 'Band6': 16.0, 'Band7': 16.0},
                       'CNI': {'Band3': 8.0, 'Band6': 16.0, 'Band7': 16.0},
                       'JELM': {'Band3': 8.0, 'Band6': 16.0, 'Band7': 16.0},
                       'LAS': {'Band3': 8.0, 'Band6': 16.0, 'Band7': 16.0}}

# some sites have different receiver temperatures
T_R_overrides = {'APEX': {'Band6': 90.0},
                 'IRAM': {'Band6': 80.0},
                 'LMT': {'Band6': 70.0},
                 'SMT': {'Wband': 40.0, 'Band6': 66.0},
                 'SMA': {'Band6': 66.0}}

# some sites have different sideband ratios (SSB = 0, DSB = 1)
sideband_ratio_overrides = {'JCMT': {'Band3': 1.0, 'Band6': 1.0, 'Band7': 0.03},
                            'LMT': {'Band3': 1.0, 'Band6': 1.0, 'Band7': 0.03},
                            'SMA': {'Band3': 1.0, 'Band6': 1.0, 'Band7': 0.03}}

# some sites have receiver lowest frequencies
lo_freq_overrides = {'BAJA': {'Band3': 67.0},
                     'CNI': {'Band3': 67.0},
                     'JELM': {'Band3': 67.0},
                     'LAS': {'Band3': 67.0}}

# initialize the observation generator
obsgen_fpt = og.obs_generator(settings_file=yamlfile,
                              custom_receivers=custom_receivers,
                              D_overrides=D_overrides,
                              receiver_configuration_overrides=receiver_configuration_overrides,
                              bandwidth_overrides=bandwidth_overrides,
                              T_R_overrides=T_R_overrides,
                              sideband_ratio_overrides=sideband_ratio_overrides,
                              lo_freq_overrides=lo_freq_overrides,
                              verbosity=0)

# generate the observation
obs_fpt = obsgen_fpt.make_obs()

# save it as a uvfits file
obs_fpt.save_uvfits('./example_datafile_with_fpt.uvfits')

#######################################################
# generate an observation without FPT

# input settings file
yamlfile = './settings_no_fpt.yaml'

# initialize the observation generator
obsgen = og.obs_generator(settings_file=yamlfile,
                          custom_receivers=custom_receivers,
                          D_overrides=D_overrides,
                          receiver_configuration_overrides=receiver_configuration_overrides,
                          bandwidth_overrides=bandwidth_overrides,
                          T_R_overrides=T_R_overrides,
                          sideband_ratio_overrides=sideband_ratio_overrides,
                          lo_freq_overrides=lo_freq_overrides,
                          verbosity=0)

# generate the observation
obs = obsgen.make_obs()

# save it as a uvfits file
obs.save_uvfits('./example_datafile_no_fpt.uvfits')

#######################################################
# make a comparison (u,v) coverage plot

fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

u1 = obs_fpt.data['u']
v1 = obs_fpt.data['v']
u2 = obs.data['u']
v2 = obs.data['v']

ax.plot(u1/(1.0e9), v1/(1.0e9), 'bo', markersize=5, markeredgewidth=0, label='with FPT')
ax.plot(-u1/(1.0e9), -v1/(1.0e9), 'bo', markersize=5, markeredgewidth=0)
ax.plot(u2/(1.0e9), v2/(1.0e9), 'ro', markersize=2, markeredgewidth=0, label='without FPT')
ax.plot(-u2/(1.0e9), -v2/(1.0e9), 'ro', markersize=2, markeredgewidth=0)

ax.set_xlabel(r'$u$ (G$\lambda$)')
ax.set_ylabel(r'$v$ (G$\lambda$)')
ax.set_xlim(15, -15)
ax.set_ylim(-15, 15)

ax.legend(fontsize=10)

plt.savefig('./uv-coverage.png', dpi=300, bbox_inches='tight')
plt.close()

#######################################################
# make a comparison radplot

fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

rho1 = np.sqrt(u1**2.0 + v1**2.0)
rho2 = np.sqrt(u2**2.0 + v2**2.0)
amp1 = np.abs(obs_fpt.data['rrvis'])
amp2 = np.abs(obs.data['rrvis'])

ax.plot(rho1/(1.0e9), amp1, 'bo', markersize=5, markeredgewidth=0, label='with FPT')
ax.plot(rho2/(1.0e9), amp2, 'ro', markersize=2, markeredgewidth=0, label='without FPT')

ax.semilogy()

ax.set_xlabel(r'$|u|$ (G$\lambda$)')
ax.set_ylabel(r'Visibility amplitude (Jy)')

ax.legend(fontsize=10)

plt.savefig('./radplot.png', dpi=300, bbox_inches='tight')
plt.close()

#######################################################
# make a comparison data histogram

fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

maxr = np.max([np.ceil(np.max(rho1)/(1.0e9)), np.ceil(np.max(rho2)/(1.0e9))])
nbins = int(4.0*maxr)

ax.hist(rho1/(1.0e9), bins=nbins, range=(0.0, maxr), alpha=0.4, label='with FPT')
ax.hist(rho2/(1.0e9), bins=nbins, range=(0.0, maxr), alpha=0.4, label='without FPT')

ax.set_xlabel(r'$|u|$ (G$\lambda$)')
ax.set_ylabel(r'Number of data points')

ax.legend(fontsize=10)

plt.savefig('./data_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
