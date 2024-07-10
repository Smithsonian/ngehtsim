import numpy as np
import matplotlib.pyplot as plt
import ngehtsim.weather.weather as nw
import ngehtsim.obs.obs_generator as og
import os

#######################################################
# extract weather info for a siteon some date 

site = 'ALMA'
day = 11
month = 'Apr'
year = 2017
freq = 230.0

P = nw.pressure(site, form='exact', month=month, day=day, year=year)
T = nw.temperature(site, form='exact', month=month, day=day, year=year)
PWV = nw.PWV(site, form='exact', month=month, day=day, year=year)
WS = nw.windspeed(site, form='exact', month=month, day=day, year=year)
tau = nw.opacity(site, freq=230.0, form='exact', month=month, day=day, year=year)
Tb = nw.brightness_temperature(site, freq=230.0, form='exact', month=month, day=day, year=year)

print('='*100)
print('Atmospheric conditions for '+site+' on '+month+' '+str(day)+', '+str(year)+':\n')
print('Surface pressure (mbar):', P)
print('Surface temperature (K):', T)
print('Precipitable water vapor (mm):', PWV)
print('Windspeed (m/s):', WS)
print('Atmospheric opacity at '+str(freq)+' GHz:', tau)
print('Atmospheric brightness temperature at '+str(freq)+' GHz (K):', Tb)
print('='*100)

#######################################################
# make plots of opacity versus month of year

print('Plotting opacities...')

# specify frequency
freq = 230.0

# make directory for plots
plotdir = './tests/weather_plots/opacity'
os.makedirs(plotdir, exist_ok=True)

# get lists of sites and months
sites = og.get_site_list()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for site in sites:

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # compute opacity median and 1-sigma intervals
    tau16 = np.zeros(14)
    tau50 = np.zeros(14)
    tau84 = np.zeros(14)
    for imonth, month in enumerate(months):
        tau = nw.opacity(site, freq=freq, form='all', month=month)
        if (imonth == 0):
            tau16[-1] = tau16[imonth+1] = np.nanpercentile(tau, 16)
            tau50[-1] = tau50[imonth+1] = np.nanpercentile(tau, 50)
            tau84[-1] = tau84[imonth+1] = np.nanpercentile(tau, 84)
        elif ((imonth > 0) & (imonth < 11)):
            tau16[imonth+1] = np.nanpercentile(tau, 16)
            tau50[imonth+1] = np.nanpercentile(tau, 50)
            tau84[imonth+1] = np.nanpercentile(tau, 84)
        else:
            tau16[0] = tau16[imonth+1] = np.nanpercentile(tau, 16)
            tau50[0] = tau50[imonth+1] = np.nanpercentile(tau, 50)
            tau84[0] = tau84[imonth+1] = np.nanpercentile(tau, 84)

    # plot opacity
    xdum = np.linspace(0.0, 13.0, 14)
    ax.fill_between(xdum, tau16, tau84, color='blue', alpha=0.2, linewidth=0)
    ax.plot(xdum, tau50, 'b-')

    # extra plot items
    ax.set_title(site)
    ax.set_ylabel('Opacity')
    ax.set_xticks(xdum)
    ax.set_xticklabels(['']+months+[''], rotation=60)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(bottom=0.0)

    # save the figure
    plt.savefig(plotdir+'/'+site+'.png', dpi=300, bbox_inches='tight')
    plt.close()

#######################################################
# make plots of PWV versus month of year

print('Plotting PWVs...')

# make directory for plots
plotdir = './tests/weather_plots/PWV'
os.makedirs(plotdir, exist_ok=True)

# get lists of sites and months
sites = og.get_site_list()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for site in sites:

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # compute opacity median and 1-sigma intervals
    PWV16 = np.zeros(14)
    PWV50 = np.zeros(14)
    PWV84 = np.zeros(14)
    for imonth, month in enumerate(months):
        PWV = nw.PWV(site, form='all', month=month)
        if (imonth == 0):
            PWV16[-1] = PWV16[imonth+1] = np.nanpercentile(PWV, 16)
            PWV50[-1] = PWV50[imonth+1] = np.nanpercentile(PWV, 50)
            PWV84[-1] = PWV84[imonth+1] = np.nanpercentile(PWV, 84)
        elif ((imonth > 0) & (imonth < 11)):
            PWV16[imonth+1] = np.nanpercentile(PWV, 16)
            PWV50[imonth+1] = np.nanpercentile(PWV, 50)
            PWV84[imonth+1] = np.nanpercentile(PWV, 84)
        else:
            PWV16[0] = PWV16[imonth+1] = np.nanpercentile(PWV, 16)
            PWV50[0] = PWV50[imonth+1] = np.nanpercentile(PWV, 50)
            PWV84[0] = PWV84[imonth+1] = np.nanpercentile(PWV, 84)

    # plot opacity
    xdum = np.linspace(0.0, 13.0, 14)
    ax.fill_between(xdum, PWV16, PWV84, color='blue', alpha=0.2, linewidth=0)
    ax.plot(xdum, PWV50, 'b-')

    # extra plot items
    ax.set_title(site)
    ax.set_ylabel('PWV (mm)')
    ax.set_xticks(xdum)
    ax.set_xticklabels(['']+months+[''], rotation=60)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(bottom=0.0)

    # save the figure
    plt.savefig(plotdir+'/'+site+'.png', dpi=300, bbox_inches='tight')
    plt.close()

#######################################################
# make plots of pressure versus month of year

print('Plotting pressures...')

# make directory for plots
plotdir = './tests/weather_plots/pressure'
os.makedirs(plotdir, exist_ok=True)

# get lists of sites and months
sites = og.get_site_list()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for site in sites:

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # compute opacity median and 1-sigma intervals
    pressure16 = np.zeros(14)
    pressure50 = np.zeros(14)
    pressure84 = np.zeros(14)
    for imonth, month in enumerate(months):
        pressure = nw.pressure(site, form='all', month=month)
        if (imonth == 0):
            pressure16[-1] = pressure16[imonth+1] = np.nanpercentile(pressure, 16)
            pressure50[-1] = pressure50[imonth+1] = np.nanpercentile(pressure, 50)
            pressure84[-1] = pressure84[imonth+1] = np.nanpercentile(pressure, 84)
        elif ((imonth > 0) & (imonth < 11)):
            pressure16[imonth+1] = np.nanpercentile(pressure, 16)
            pressure50[imonth+1] = np.nanpercentile(pressure, 50)
            pressure84[imonth+1] = np.nanpercentile(pressure, 84)
        else:
            pressure16[0] = pressure16[imonth+1] = np.nanpercentile(pressure, 16)
            pressure50[0] = pressure50[imonth+1] = np.nanpercentile(pressure, 50)
            pressure84[0] = pressure84[imonth+1] = np.nanpercentile(pressure, 84)

    # plot opacity
    xdum = np.linspace(0.0, 13.0, 14)
    ax.fill_between(xdum, pressure16, pressure84, color='blue', alpha=0.2, linewidth=0)
    ax.plot(xdum, pressure50, 'b-')

    # extra plot items
    ax.set_title(site)
    ax.set_ylabel('Pressure (mbar)')
    ax.set_xticks(xdum)
    ax.set_xticklabels(['']+months+[''], rotation=60)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0.98*np.min(pressure16),1.02*np.max(pressure84))

    # save the figure
    plt.savefig(plotdir+'/'+site+'.png', dpi=300, bbox_inches='tight')
    plt.close()

#######################################################
# make plots of temperature versus month of year

print('Plotting temperatures...')

# make directory for plots
plotdir = './tests/weather_plots/temperature'
os.makedirs(plotdir, exist_ok=True)

# get lists of sites and months
sites = og.get_site_list()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for site in sites:

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # compute opacity median and 1-sigma intervals
    temperature16 = np.zeros(14)
    temperature50 = np.zeros(14)
    temperature84 = np.zeros(14)
    for imonth, month in enumerate(months):
        temperature = nw.temperature(site, form='all', month=month)
        if (imonth == 0):
            temperature16[-1] = temperature16[imonth+1] = np.nanpercentile(temperature, 16)
            temperature50[-1] = temperature50[imonth+1] = np.nanpercentile(temperature, 50)
            temperature84[-1] = temperature84[imonth+1] = np.nanpercentile(temperature, 84)
        elif ((imonth > 0) & (imonth < 11)):
            temperature16[imonth+1] = np.nanpercentile(temperature, 16)
            temperature50[imonth+1] = np.nanpercentile(temperature, 50)
            temperature84[imonth+1] = np.nanpercentile(temperature, 84)
        else:
            temperature16[0] = temperature16[imonth+1] = np.nanpercentile(temperature, 16)
            temperature50[0] = temperature50[imonth+1] = np.nanpercentile(temperature, 50)
            temperature84[0] = temperature84[imonth+1] = np.nanpercentile(temperature, 84)

    # plot opacity
    xdum = np.linspace(0.0, 13.0, 14)
    ax.fill_between(xdum, temperature16, temperature84, color='blue', alpha=0.2, linewidth=0)
    ax.plot(xdum, temperature50, 'b-')

    # extra plot items
    ax.set_title(site)
    ax.set_ylabel('Temperature (K)')
    ax.set_xticks(xdum)
    ax.set_xticklabels(['']+months+[''], rotation=60)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(200,320)

    # save the figure
    plt.savefig(plotdir+'/'+site+'.png', dpi=300, bbox_inches='tight')
    plt.close()

#######################################################
# make plots of windspeed versus month of year

print('Plotting windspeeds...')

# make directory for plots
plotdir = './tests/weather_plots/windspeed'
os.makedirs(plotdir, exist_ok=True)

# get lists of sites and months
sites = og.get_site_list()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for site in sites:

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # compute opacity median and 1-sigma intervals
    windspeed16 = np.zeros(14)
    windspeed50 = np.zeros(14)
    windspeed84 = np.zeros(14)
    for imonth, month in enumerate(months):
        windspeed = nw.windspeed(site, form='all', month=month)
        if (imonth == 0):
            windspeed16[-1] = windspeed16[imonth+1] = np.nanpercentile(windspeed, 16)
            windspeed50[-1] = windspeed50[imonth+1] = np.nanpercentile(windspeed, 50)
            windspeed84[-1] = windspeed84[imonth+1] = np.nanpercentile(windspeed, 84)
        elif ((imonth > 0) & (imonth < 11)):
            windspeed16[imonth+1] = np.nanpercentile(windspeed, 16)
            windspeed50[imonth+1] = np.nanpercentile(windspeed, 50)
            windspeed84[imonth+1] = np.nanpercentile(windspeed, 84)
        else:
            windspeed16[0] = windspeed16[imonth+1] = np.nanpercentile(windspeed, 16)
            windspeed50[0] = windspeed50[imonth+1] = np.nanpercentile(windspeed, 50)
            windspeed84[0] = windspeed84[imonth+1] = np.nanpercentile(windspeed, 84)

    # plot opacity
    xdum = np.linspace(0.0, 13.0, 14)
    ax.fill_between(xdum, windspeed16, windspeed84, color='blue', alpha=0.2, linewidth=0)
    ax.plot(xdum, windspeed50, 'b-')

    # extra plot items
    ax.set_title(site)
    ax.set_ylabel('Windspeed (m/s)')
    ax.set_xticks(xdum)
    ax.set_xticklabels(['']+months+[''], rotation=60)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(bottom=0.0)

    # save the figure
    plt.savefig(plotdir+'/'+site+'.png', dpi=300, bbox_inches='tight')
    plt.close()
