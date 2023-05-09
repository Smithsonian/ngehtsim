#######################################################
# imports

import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh
import ngehtsim.obs.obs_generator as og

#######################################################
# generate an observation with FPT

# input settings file
yamlfile = './settings_fpt.yaml'

# some sites only have access to certain frequencies
receiver_override_dict = {'ALMA': ['345'],
                          'HAY': ['86', '230'],
                          'KP': ['86', '230'],
                          'OVRO': ['86', '230'],
                          'SPT': ['230', '345']}

# some sites can record higher bandwidths
bandwidth_override_dict = {'HAY': {'86': 8.0, '230': 16.0},
                           'OVRO': {'86': 8.0, '230': 16.0},
                           'BAJA': {'86': 8.0, '230': 16.0, '345': 16.0},
                           'CNI': {'86': 8.0, '230': 16.0, '345': 16.0},
                           'LAS': {'86': 8.0, '230': 16.0, '345': 16.0}}

# some sites have different receiver temperatures
T_R_override_dict = {'HAY': {'86': 30.0, '230': 40.0},
                     'BAJA': {'86': 30.0, '230': 40.0, '345': 60.0},
                     'CNI': {'86': 30.0, '230': 40.0, '345': 60.0},
                     'LAS': {'86': 30.0, '230': 40.0, '345': 60.0}}

# initialize the observation generator
obsgen_fpt = og.obs_generator(settings_file=yamlfile,receiver_override_dict=receiver_override_dict,bandwidth_override_dict=bandwidth_override_dict,T_R_override_dict=T_R_override_dict)

# generate the observation
obs_fpt = obsgen_fpt.make_obs()

# save it as a uvfits file
obs_fpt.save_uvfits('./example_datafile_with_fpt.uvfits')

#######################################################
# generate an observation without FPT

# input settings file
yamlfile = './settings_no_fpt.yaml'

# initialize the observation generator
obsgen = og.obs_generator(settings_file=yamlfile,receiver_override_dict=receiver_override_dict,bandwidth_override_dict=bandwidth_override_dict,T_R_override_dict=T_R_override_dict)

# generate the observation
obs = obsgen.make_obs()

# save it as a uvfits file
obs.save_uvfits('./example_datafile_no_fpt.uvfits')

#######################################################
# make a comparison (u,v) coverage plot

fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([0.1,0.1,0.8,0.8])

u1 = obs_fpt.data['u']
v1 = obs_fpt.data['v']
u2 = obs.data['u']
v2 = obs.data['v']

ax.plot(u1/(1.0e9),v1/(1.0e9),'bo',markersize=5,markeredgewidth=0,label='with FPT')
ax.plot(-u1/(1.0e9),-v1/(1.0e9),'bo',markersize=5,markeredgewidth=0)
ax.plot(u2/(1.0e9),v2/(1.0e9),'ro',markersize=2,markeredgewidth=0,label='without FPT')
ax.plot(-u2/(1.0e9),-v2/(1.0e9),'ro',markersize=2,markeredgewidth=0)

ax.set_xlabel(r'$u$ (G$\lambda$)')
ax.set_ylabel(r'$v$ (G$\lambda$)')

ax.legend(fontsize=10)

plt.savefig('./uv-coverage.png',dpi=300,bbox_inches='tight')
plt.close()

#######################################################
# make a comparison radplot

fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([0.1,0.1,0.8,0.8])

rho1 = np.sqrt(u1**2.0 + v1**2.0)
rho2 = np.sqrt(u2**2.0 + v2**2.0)
amp1 = np.abs(obs_fpt.data['vis'])
amp2 = np.abs(obs.data['vis'])

ax.plot(rho1/(1.0e9),amp1,'bo',markersize=5,markeredgewidth=0,label='with FPT')
ax.plot(rho2/(1.0e9),amp2,'ro',markersize=2,markeredgewidth=0,label='without FPT')

ax.semilogy()

ax.set_xlabel(r'$|u|$ (G$\lambda$)')
ax.set_ylabel(r'Visibility amplitude (Jy)')

ax.legend(fontsize=10)

plt.savefig('./radplot.png',dpi=300,bbox_inches='tight')
plt.close()

#######################################################
# make a comparison data histogram

fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([0.1,0.1,0.8,0.8])

maxr = np.max([np.ceil(np.max(rho1)/(1.0e9)),np.ceil(np.max(rho2)/(1.0e9))])
nbins = int(4.0*maxr)

ax.hist(rho1/(1.0e9),bins=nbins,range=(0.0,maxr),alpha=0.4,label='with FPT')
ax.hist(rho2/(1.0e9),bins=nbins,range=(0.0,maxr),alpha=0.4,label='without FPT')

ax.set_xlabel(r'$|u|$ (G$\lambda$)')
ax.set_ylabel(r'Number of data points')

ax.legend(fontsize=10)

plt.savefig('./data_histogram.png',dpi=300,bbox_inches='tight')
plt.close()
