############################################
# imports

import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh
import ngehtsim.calibration.calibration as nc

############################################
# inputs

# alist file
infile = './tests/data_flux_calibration/M87_observation.alist'

# source name
sourcename = 'M87'

# observing bandwidth, in GHz
bandwidth = 2.0

# name of output dlist file
outfilename = './tests/data_flux_calibration/'+sourcename+'_apriorical.dlist'

###################################################
# specify some properties of individual stations

T_R_overrides = {'ALMA': {'Band6': 40.0},
                 'APEX': {'Band6': 85.0},
                 'IRAM': {'Band6': 60.0},
                 'JCMT': {'Band6': 60.0},
                 'LMT': {'Band6': 70.0},
                 'SMA': {'Band6': 70.0},
                 'SMT': {'Band6': 80.0},
                 'SPT': {'Band6': 40.0}}

sideband_ratio_overrides = {'ALMA': {'Band6': 0.01},
                            'APEX': {'Band6': 0.03},
                            'IRAM': {'Band6': 0.03},
                            'JCMT': {'Band6': 0.03},
                            'LMT': {'Band6': 0.03},
                            'SMA': {'Band6': 1.0},
                            'SMT': {'Band6': 0.03},
                            'SPT': {'Band6': 0.03}}

###################################################
# carry out a prior flux density calibration

df = nc.apriorical(infile, sourcename, bandwidth,
                   T_R_overrides=T_R_overrides,
                   sideband_ratio_overrides=sideband_ratio_overrides)

###################################################
# export a dlist file

nc.write_dlist(infile, sourcename, bandwidth, outfilename,
               T_R_overrides=T_R_overrides,
               sideband_ratio_overrides=sideband_ratio_overrides)

############################################
# make a combined SNR plot + radplot

# station code conversion dictionary
station_dict = {'A': 'ALMA',
                'J': 'JCMT',
                'L': 'LMT',
                'P': 'IRAM',
                'S': 'SMA',
                'X': 'APEX',
                'Y': 'SPT',
                'Z': 'SMT'}

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_axes([0.1,0.5,0.8,0.4])
ax2 = fig.add_axes([0.1,0.1,0.8,0.4])

rho = np.sqrt((df.u**2.0) + (df.v**2.0)) / (1.0e9)
bl_unique = np.unique(df.bl)
for i, bli in enumerate(bl_unique):
    indhere = (df.bl == bli)
    label = station_dict[bli[0]] + '-' + station_dict[bli[1]]
    colhere = 'C' + str(i % 10)

    # only plot RR or LL
    rhohere = rho[indhere]
    snrhere = df.snr[indhere]
    ind2here = ((df.pl[indhere] == 'RR') | (df.pl[indhere] == 'LL'))

    if ind2here.sum() > 0:
        ax1.plot(rhohere[ind2here],snrhere[ind2here],linewidth=0,marker='o',markersize=3,color=colhere,label=label)

for i, bli in enumerate(bl_unique):
    indhere = (df.bl == bli)
    label = station_dict[bli[0]] + '-' + station_dict[bli[1]]
    colhere = 'C' + str(i % 10)

    # only plot RR or LL
    rhohere = rho[indhere]
    amphere = np.abs(df.vis)[indhere]
    ind2here = ((df.pl[indhere] == 'RR') | (df.pl[indhere] == 'LL'))

    if ind2here.sum() > 0:
        ax2.plot(rhohere[ind2here],amphere[ind2here],linewidth=0,marker='o',markersize=3,color=colhere,label=label)

xtext = 0.0
ytext = 9000
ax1.text(xtext,ytext,sourcename,ha='left',va='top')

ax1.set_xlim(-0.2,10)
ax2.set_xlim(-0.2,10)

ax1.set_xticklabels([])

ax1.set_ylim(0.2,10000)
ax2.set_ylim(1.0e-3,2.0e0)

ax1.set_ylabel('SNR')
ax2.set_xlabel(r'$|u|$ (G$\lambda$)')
ax2.set_ylabel('Flux density (Jy)')

ax1.semilogy()
ax2.semilogy()

ax1.grid(linewidth=0.5,linestyle='--',color='black',alpha=0.1)
ax2.grid(linewidth=0.5,linestyle='--',color='black',alpha=0.1)

ax2.legend(loc=(1.02,0.0),fontsize=10,ncol=2)

plt.savefig('./tests/data_flux_calibration/radplot_'+sourcename+'.png',dpi=300,bbox_inches='tight')
plt.close()
