#######################################################
# imports

import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh

import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.obs_plotter as op
import ngehtsim.metrics as cm

#######################################################
# set up obsgen object

# input settings file
yamlfile = './tests/data_generation_instrumental_corruptions/settings.yaml'

# initialize the observation generator
obsgen = og.obs_generator(settings_file=yamlfile)

#######################################################
# generate observations with varying corruptions

# observe a polarized double Gaussian model
mod = eh.model.Model()
mod = mod.add_circ_gauss(0.5, 20.*eh.RADPERUAS, x0=20.0*eh.RADPERUAS, y0=0.0, pol_frac=0.20, pol_evpa=67.5*eh.DEGREE, cpol_frac=0.10)
mod = mod.add_circ_gauss(0.5, 30.*eh.RADPERUAS, x0=-20.0*eh.RADPERUAS, y0=20.0*eh.RADPERUAS, pol_frac=0.10, pol_evpa=30.0*eh.DEGREE, cpol_frac=-0.05)

# generate an observation with no noise
obs_nonoise = obsgen.make_obs(input_model=mod,
                              addnoise=False,addgains=False,addFR=False,addleakage=False)

# generate an observation with only thermal noise
obs_thnoise = obsgen.make_obs(input_model=mod,
                              addnoise=True,addgains=False,addFR=False,addleakage=False)

# generate an observation with thermal noise and station gains
obs_thgains = obsgen.make_obs(input_model=mod,
                              addnoise=True,addgains=True,addFR=False,addleakage=False)

# generate an observation with thermal noise and leakage
obs_thleak = obsgen.make_obs(input_model=mod,
                              addnoise=True,addgains=False,addFR=True,addleakage=True)

# generate an observation with thermal noise, station gains, and leakage
obs_full = obsgen.make_obs(input_model=mod,
                           addnoise=True,addgains=True,addFR=True,addleakage=True)

# save uvfits files
obs_nonoise.save_uvfits('./tests/data_generation_instrumental_corruptions/datafile_nonoise.uvfits')
obs_thnoise.save_uvfits('./tests/data_generation_instrumental_corruptions/datafile_thermal_noise_only.uvfits')
obs_thgains.save_uvfits('./tests/data_generation_instrumental_corruptions/datafile_thermal_noise_plus_gains.uvfits')
obs_thleak.save_uvfits('./tests/data_generation_instrumental_corruptions/datafile_thermal_noise_plus_leakage.uvfits')
obs_full.save_uvfits('./tests/data_generation_instrumental_corruptions/datafile_thermal_noise_plus_gains_plus_leakage.uvfits')

#######################################################
# make plots of the noise-free data

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_axes([0.1,0.5,0.8,0.4])
ax2 = fig.add_axes([0.1,0.1,0.8,0.4])

k = np.sqrt((obs_nonoise.data['u']**2.0) + (obs_nonoise.data['v']**2.0)) / (1.0e9)
RR = obs_nonoise.data['rrvis']
LL = obs_nonoise.data['llvis']
RL = obs_nonoise.data['rlvis']
LR = obs_nonoise.data['lrvis']

ax1.plot(k,np.abs(RR),'r.',markersize=2,label='RR',zorder=5)
ax1.plot(k,np.abs(LL),'b.',markersize=2,label='LL',zorder=5)
ax1.plot(k,np.abs(RL),'m.',markersize=2,label='RL',zorder=3)
ax1.plot(k,np.abs(LR),'c.',markersize=2,label='LR',zorder=3)
ax1.set_xlim(0,10)
ax1.set_ylim(8e-3,2)
ax1.semilogy()
ax1.set_xticklabels([])
ax1.set_ylabel('Visibility amplitude (Jy)')
ax1.legend()
ax1.grid(linestyle='--',linewidth=0.5,color='black',alpha=0.1)

ax2.plot(k,np.angle(RR),'r.',markersize=2,zorder=5)
ax2.plot(k,np.angle(LL),'b.',markersize=2,zorder=5)
ax2.plot(k,np.angle(RL),'m.',markersize=2,zorder=3)
ax2.plot(k,np.angle(LR),'c.',markersize=2,zorder=3)
ax2.set_xlim(0,10)
ax2.set_ylim(-np.pi,np.pi)
ax2.set_ylabel('Visibility phase (radians)')
ax2.grid(linestyle='--',linewidth=0.5,color='black',alpha=0.1)

plt.savefig('./tests/data_generation_instrumental_corruptions/radplot_nonoise.png',dpi=300,bbox_inches='tight')
plt.close()

#######################################################
# make plots of the thermal-noise-only data

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_axes([0.1,0.5,0.8,0.4])
ax2 = fig.add_axes([0.1,0.1,0.8,0.4])

k = np.sqrt((obs_thnoise.data['u']**2.0) + (obs_thnoise.data['v']**2.0)) / (1.0e9)
RR = obs_thnoise.data['rrvis']
LL = obs_thnoise.data['llvis']
RL = obs_thnoise.data['rlvis']
LR = obs_thnoise.data['lrvis']

ax1.plot(k,np.abs(RR),'r.',markersize=2,label='RR',zorder=5)
ax1.plot(k,np.abs(LL),'b.',markersize=2,label='LL',zorder=5)
ax1.plot(k,np.abs(RL),'m.',markersize=2,label='RL',zorder=3)
ax1.plot(k,np.abs(LR),'c.',markersize=2,label='LR',zorder=3)
ax1.set_xlim(0,10)
ax1.set_ylim(8e-3,2)
ax1.semilogy()
ax1.set_xticklabels([])
ax1.set_ylabel('Visibility amplitude (Jy)')
ax1.legend()
ax1.grid(linestyle='--',linewidth=0.5,color='black',alpha=0.1)

ax2.plot(k,np.angle(RR),'r.',markersize=2,zorder=5)
ax2.plot(k,np.angle(LL),'b.',markersize=2,zorder=5)
ax2.plot(k,np.angle(RL),'m.',markersize=2,zorder=3)
ax2.plot(k,np.angle(LR),'c.',markersize=2,zorder=3)
ax2.set_xlim(0,10)
ax2.set_ylim(-np.pi,np.pi)
ax2.set_ylabel('Visibility phase (radians)')
ax2.grid(linestyle='--',linewidth=0.5,color='black',alpha=0.1)

plt.savefig('./tests/data_generation_instrumental_corruptions/radplot_thermal_noise_only.png',dpi=300,bbox_inches='tight')
plt.close()

#######################################################
# make plots of the data with thermal noise + gains

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_axes([0.1,0.5,0.8,0.4])
ax2 = fig.add_axes([0.1,0.1,0.8,0.4])

k = np.sqrt((obs_thgains.data['u']**2.0) + (obs_thgains.data['v']**2.0)) / (1.0e9)
RR = obs_thgains.data['rrvis']
LL = obs_thgains.data['llvis']
RL = obs_thgains.data['rlvis']
LR = obs_thgains.data['lrvis']

ax1.plot(k,np.abs(RR),'r.',markersize=2,label='RR',zorder=5)
ax1.plot(k,np.abs(LL),'b.',markersize=2,label='LL',zorder=5)
ax1.plot(k,np.abs(RL),'m.',markersize=2,label='RL',zorder=3)
ax1.plot(k,np.abs(LR),'c.',markersize=2,label='LR',zorder=3)
ax1.set_xlim(0,10)
ax1.set_ylim(8e-3,2)
ax1.semilogy()
ax1.set_xticklabels([])
ax1.set_ylabel('Visibility amplitude (Jy)')
ax1.legend()
ax1.grid(linestyle='--',linewidth=0.5,color='black',alpha=0.1)

ax2.plot(k,np.angle(RR),'r.',markersize=2,zorder=5)
ax2.plot(k,np.angle(LL),'b.',markersize=2,zorder=5)
ax2.plot(k,np.angle(RL),'m.',markersize=2,zorder=3)
ax2.plot(k,np.angle(LR),'c.',markersize=2,zorder=3)
ax2.set_xlim(0,10)
ax2.set_ylim(-np.pi,np.pi)
ax2.set_ylabel('Visibility phase (radians)')
ax2.grid(linestyle='--',linewidth=0.5,color='black',alpha=0.1)

plt.savefig('./tests/data_generation_instrumental_corruptions/radplot_thermal_noise_plus_gains.png',dpi=300,bbox_inches='tight')
plt.close()

#######################################################
# make plots of the data with thermal noise + leakage

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_axes([0.1,0.5,0.8,0.4])
ax2 = fig.add_axes([0.1,0.1,0.8,0.4])

k = np.sqrt((obs_thleak.data['u']**2.0) + (obs_thleak.data['v']**2.0)) / (1.0e9)
RR = obs_thleak.data['rrvis']
LL = obs_thleak.data['llvis']
RL = obs_thleak.data['rlvis']
LR = obs_thleak.data['lrvis']

ax1.plot(k,np.abs(RR),'r.',markersize=2,label='RR',zorder=5)
ax1.plot(k,np.abs(LL),'b.',markersize=2,label='LL',zorder=5)
ax1.plot(k,np.abs(RL),'m.',markersize=2,label='RL',zorder=3)
ax1.plot(k,np.abs(LR),'c.',markersize=2,label='LR',zorder=3)
ax1.set_xlim(0,10)
ax1.set_ylim(8e-3,2)
ax1.semilogy()
ax1.set_xticklabels([])
ax1.set_ylabel('Visibility amplitude (Jy)')
ax1.legend()
ax1.grid(linestyle='--',linewidth=0.5,color='black',alpha=0.1)

ax2.plot(k,np.angle(RR),'r.',markersize=2,zorder=5)
ax2.plot(k,np.angle(LL),'b.',markersize=2,zorder=5)
ax2.plot(k,np.angle(RL),'m.',markersize=2,zorder=3)
ax2.plot(k,np.angle(LR),'c.',markersize=2,zorder=3)
ax2.set_xlim(0,10)
ax2.set_ylim(-np.pi,np.pi)
ax2.set_ylabel('Visibility phase (radians)')
ax2.grid(linestyle='--',linewidth=0.5,color='black',alpha=0.1)

plt.savefig('./tests/data_generation_instrumental_corruptions/radplot_thermal_noise_plus_leakage.png',dpi=300,bbox_inches='tight')
plt.close()

#######################################################
# make plots of the data with thermal noise + gains + leakage

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_axes([0.1,0.5,0.8,0.4])
ax2 = fig.add_axes([0.1,0.1,0.8,0.4])

k = np.sqrt((obs_full.data['u']**2.0) + (obs_full.data['v']**2.0)) / (1.0e9)
RR = obs_full.data['rrvis']
LL = obs_full.data['llvis']
RL = obs_full.data['rlvis']
LR = obs_full.data['lrvis']

ax1.plot(k,np.abs(RR),'r.',markersize=2,label='RR',zorder=5)
ax1.plot(k,np.abs(LL),'b.',markersize=2,label='LL',zorder=5)
ax1.plot(k,np.abs(RL),'m.',markersize=2,label='RL',zorder=3)
ax1.plot(k,np.abs(LR),'c.',markersize=2,label='LR',zorder=3)
ax1.set_xlim(0,10)
ax1.set_ylim(8e-3,2)
ax1.semilogy()
ax1.set_xticklabels([])
ax1.set_ylabel('Visibility amplitude (Jy)')
ax1.legend()
ax1.grid(linestyle='--',linewidth=0.5,color='black',alpha=0.1)

ax2.plot(k,np.angle(RR),'r.',markersize=2,zorder=5)
ax2.plot(k,np.angle(LL),'b.',markersize=2,zorder=5)
ax2.plot(k,np.angle(RL),'m.',markersize=2,zorder=3)
ax2.plot(k,np.angle(LR),'c.',markersize=2,zorder=3)
ax2.set_xlim(0,10)
ax2.set_ylim(-np.pi,np.pi)
ax2.set_ylabel('Visibility phase (radians)')
ax2.grid(linestyle='--',linewidth=0.5,color='black',alpha=0.1)

plt.savefig('./tests/data_generation_instrumental_corruptions/radplot_thermal_noise_plus_gains_plus_leakage.png',dpi=300,bbox_inches='tight')
plt.close()
