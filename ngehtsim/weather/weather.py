###################################################
# imports

import numpy as np
import ngehtsim.const_def as const
import os
import struct

###################################################
# load eigenspectra

meanspec_tau = np.loadtxt(const.path_to_eigenspectra+'/spectrum_mean.txt',unpack=True)
meanspec_Tb = np.loadtxt(const.path_to_eigenspectra+'_Tb/spectrum_mean.txt',unpack=True)
tau_spectra = list()
Tb_spectra = list()
for i in range(const.number_of_components):
    spechere = np.loadtxt(const.path_to_eigenspectra+'/spectrum_'+str(i).zfill(4)+'.txt',unpack=True)
    spechere_Tb = np.loadtxt(const.path_to_eigenspectra+'_Tb/spectrum_'+str(i).zfill(4)+'.txt',unpack=True)
    tau_spectra.append(spechere)
    Tb_spectra.append(spechere_Tb)

###################################################
# function definitions

def read_binary_atm(filename,Ncomps=const.number_of_components):
    """
    Read a stored weather data file containing either opacity or brightness temperature info.

    Args:
      filename (str): The name of the weather data file
      Ncomps (int): The number of PCA component coefficients that have been stored

    Returns:
      (numpy.ndarray): Several arrays containing the dates/times and PCA component coefficients
    """

    with open(filename, 'rb') as binary_file:
        contents = bytearray(binary_file.read())

    linelength = int(contents[0:2][0])
    prelength = linelength - (2*Ncomps)
    Nlines = int((len(contents) - 2) / linelength)

    years = np.zeros(Nlines)
    months = np.zeros(Nlines)
    days = np.zeros(Nlines)
    if prelength > 4:
        times = np.zeros(Nlines)
    coeffs = np.zeros((Nlines,Ncomps))
    for i in range(Nlines):

        istart = 2 + (linelength*i)
        iend = istart + linelength
        linehere = contents[istart:iend]

        years[i] = int(struct.unpack('<h', linehere[0:2])[0])
        months[i] = int(struct.unpack('b', linehere[2:3])[0])
        days[i] = int(struct.unpack('b', linehere[3:4])[0])
        if prelength > 4:
            times[i] = int(struct.unpack('b', linehere[4:5])[0])
        coeffs[i,:] = np.array(struct.unpack('<'+'e'*Ncomps, linehere[prelength:])).astype(float)

    if prelength > 4:
        return years, months, days, times, coeffs
    else:
        return years, months, days, coeffs

def read_binary_weather(filename):
    """
    Read a stored weather data file containing pressure, temperature, wind, or PWV info.

    Args:
      filename (str): The name of the weather data file

    Returns:
      (numpy.ndarray): Several arrays containing the dates/times and weather values read from the file
    """

    with open(filename, 'rb') as binary_file:
        contents = bytearray(binary_file.read())

    linelength = int(contents[0:2][0])
    prelength = linelength - 8
    Nlines = int((len(contents) - 2) / linelength)

    years = np.zeros(Nlines)
    months = np.zeros(Nlines)
    days = np.zeros(Nlines)
    if prelength > 4:
        times = np.zeros(Nlines)
    vals = np.zeros(Nlines)
    for i in range(Nlines):
        istart = 2 + (linelength*i)
        iend = istart + linelength
        linehere = contents[istart:iend]

        years[i] = int(struct.unpack('<h', linehere[0:2])[0])
        months[i] = int(struct.unpack('b', linehere[2:3])[0])
        days[i] = int(struct.unpack('b', linehere[3:4])[0])
        if prelength > 4:
            times[i] = int(struct.unpack('b', linehere[4:5])[0])
        vals[i] = float(struct.unpack('<d', linehere[prelength:])[0])

    if prelength > 4:
        return years, months, days, times, vals
    else:
        return years, months, days, vals

def reconstruct_spectrum_tau(coeffs):
    """
    Reconstruct an opacity spectrum from PCA component coefficients.

    Args:
      coeffs (numpy.ndarray): Array containing the PCA component coefficients

    Returns:
      (numpy.ndarray): Array containing the opacity spectrum
    """

    reconstructed_spectrum = np.zeros(const.length_of_spectrum)
    for ispec, eigenspec in enumerate(tau_spectra):
        reconstructed_spectrum += coeffs[ispec]*eigenspec
    reconstructed_spectrum += meanspec_tau
    reconstructed_spectrum = 10.0**reconstructed_spectrum
    return reconstructed_spectrum

def reconstruct_spectrum_Tb(coeffs):
    """
    Reconstruct a brightness temperature spectrum from PCA component coefficients.

    Args:
      coeffs (numpy.ndarray): Array containing the PCA component coefficients

    Returns:
      (numpy.ndarray): Array containing the brightness temperature spectrum
    """

    reconstructed_spectrum = np.zeros(const.length_of_spectrum)
    for ispec, eigenspec in enumerate(Tb_spectra):
        reconstructed_spectrum += coeffs[ispec]*eigenspec
    reconstructed_spectrum += meanspec_Tb
    return reconstructed_spectrum

def opacity_spectrum(site, form='exact', month='Apr', day=15, year=2015):
    """
    Retrieve the zenith opacity information for a specified site as a function of frequency.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str or int): The month for which to report weather
      day (str or int): The day of the month for which to report weather; only used for form = 'exact'
      year (str or int): The year for which to report weather; only used for form = 'exact'

    Returns:
      (numpy.ndarray): The requested opacity values; if form = 'all', then returns a 2D array
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif str(month).zfill(2) in monthnums:
        monthnum = str(month).zfill(2)
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # determine which table to read
    pathhere = const.path_to_weather + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'tau.txt'

    # read in the table
    years, months, days, coeffs = read_binary_atm(pathhere)

    # retrieve the requested opacity values
    if (form == 'exact'):
        ind = ((years == int(year)) & (days == int(day)))
        if (np.array(ind).sum() == 0):
            raise Exception('No weather on file for the selected date.')
        tauspec = reconstruct_spectrum_tau(coeffs[ind][0])

    else:
        tauspec_arr = np.zeros((len(coeffs),const.length_of_spectrum))
        for i in range(len(coeffs)):
            tauspec_arr[i,:] = reconstruct_spectrum_tau(coeffs[i])
        if (form == 'mean'):
            tauspec = np.mean(tauspec_arr,axis=0)
        elif (form == 'median'):
            tauspec = np.median(tauspec_arr,axis=0)
        elif (form == 'good'):
            tauspec = np.percentile(tauspec_arr,15.87,axis=0)
        elif (form == 'bad'):
            tauspec = np.percentile(tauspec_arr,84.13,axis=0)
        elif (form == 'all'):
            tauspec = tauspec_arr
    
    return tauspec

def opacity(site, form='exact', month='Apr', day=15, year=2015, freq=230.0):
    """
    Retrieve the zenith opacity information for a specified site at a specified frequency.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str or int): The month for which to report weather
      day (str or int): The day of the month for which to report weather; only used for form = 'exact'
      year (str or int): The year for which to report weather; only used for form = 'exact'
      freq (float): The observing frequency, in GHz

    Returns:
      (float): The requested opacity value(s); if form = 'all', then returns a numpy.ndarray
    """

    # check frequency
    if ((freq < 0.0) | (freq > 2000.0)):
        raise Exception('Specified frequency is outside of the acceptable range (0, 2000) GHz.')

    # get full spectrum
    tauspec = opacity_spectrum(site, form=form, month=month, day=day, year=year)

    # isolate the specified frequency
    if (form != 'all'):
        tau = np.interp(freq,const.spectrum_frequency,tauspec)
    else:
        tau = np.zeros(len(tauspec))
        for i in range(len(tauspec)):
            tau[i] = np.interp(freq,const.spectrum_frequency,tauspec[i])

    return tau

def brightness_temperature_spectrum(site, form='exact', month='Apr', day=15, year=2015):
    """
    Retrieve the zenith brightness temperature information for a specified site as a function of frequency.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str or int): The month for which to report weather
      day (str or int): The day of the month for which to report weather; only used for form = 'exact'
      year (str or int): The year for which to report weather; only used for form = 'exact'

    Returns:
      (numpy.ndarray): The requested brightness temperature values; if form = 'all', then returns a 2D array
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif str(month).zfill(2) in monthnums:
        monthnum = str(month).zfill(2)
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # determine which table to read
    pathhere = const.path_to_weather + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'Tb.txt'

    # read in the table
    years, months, days, coeffs = read_binary_atm(pathhere)

    # retrieve the requested brightness temperature values
    if (form == 'exact'):
        ind = ((years == int(year)) & (days == int(day)))
        if (np.array(ind).sum() == 0):
            raise Exception('No weather on file for the selected date.')
        Tbspec = reconstruct_spectrum_Tb(coeffs[ind][0])

    else:
        Tbspec_arr = np.zeros((len(coeffs),const.length_of_spectrum))
        for i in range(len(coeffs)):
            Tbspec_arr[i,:] = reconstruct_spectrum_Tb(coeffs[i])
        if (form == 'mean'):
            Tbspec = np.mean(Tbspec_arr,axis=0)
        elif (form == 'median'):
            Tbspec = np.median(Tbspec_arr,axis=0)
        elif (form == 'good'):
            Tbspec = np.percentile(Tbspec_arr,15.87,axis=0)
        elif (form == 'bad'):
            Tbspec = np.percentile(Tbspec_arr,84.13,axis=0)
        elif (form == 'all'):
            Tbspec = Tbspec_arr
    
    return Tbspec

def brightness_temperature(site, form='exact', month='Apr', day=15, year=2015, freq=230.0):
    """
    Retrieve the zenith brightness temperature information for a specified site at a specified frequency.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str or int): The month for which to report weather
      day (str or int): The day of the month for which to report weather; only used for form = 'exact'
      year (str or int): The year for which to report weather; only used for form = 'exact'
      freq (float): The observing frequency, in GHz

    Returns:
      (float): The requested brightness temperature value(s), in K; if form = 'all', then returns a numpy.ndarray
    """

    # check frequency
    if ((freq < 0.0) | (freq > 2000.0)):
        raise Exception('Specified frequency is outside of the acceptable range (0, 2000) GHz.')

    # get full spectrum
    Tbspec = brightness_temperature_spectrum(site, form=form, month=month, day=day, year=year)

    # isolate the specified frequency
    if (form != 'all'):
        Tb = np.interp(freq,const.spectrum_frequency,Tbspec)
    else:
        Tb = np.zeros(len(Tbspec))
        for i in range(len(Tbspec)):
            Tb[i] = np.interp(freq,const.spectrum_frequency,Tbspec[i])

    return Tb

def pressure(site, form='exact', month='Apr', day=15, year=2015):
    """
    Retrieve the surface pressure information for a specified site.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str or int): The month for which to report weather
      day (str or int): The day of the month for which to report weather; only used for form = 'exact'
      year (str or int): The year for which to report weather; only used for form = 'exact'

    Returns:
      (float): The requested pressure value(s), in mbar; if form = 'all', then returns a numpy.ndarray
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif str(month).zfill(2) in monthnums:
        monthnum = str(month).zfill(2)
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # determine which table to read
    pathhere = const.path_to_weather + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'Pbase.txt'

    # read in the table
    years, months, days, vals = read_binary_weather(pathhere)

    # retrieve the requested pressure value(s)
    if (form == 'exact'):
        ind = ((years == int(year)) & (days == int(day)))
        if (np.array(ind).sum() == 0):
            raise Exception('No weather on file for the selected date.')
        P = vals[ind][0]

    else:
        if (form == 'mean'):
            P = np.mean(vals)
        elif (form == 'median'):
            P = np.median(vals)
        elif (form == 'good'):
            P = np.percentile(vals,15.87,axis=0)
        elif (form == 'bad'):
            P = np.percentile(vals,84.13,axis=0)
        elif (form == 'all'):
            P = vals

    return P

def temperature(site, form='exact', month='Apr', day=15, year=2015):
    """
    Retrieve the surface temperature information for a specified site.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str or int): The month for which to report weather
      day (str or int): The day of the month for which to report weather; only used for form = 'exact'
      year (str or int): The year for which to report weather; only used for form = 'exact'

    Returns:
      (float): The requested temperature value(s), in K; if form = 'all', then returns a numpy.ndarray
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif str(month).zfill(2) in monthnums:
        monthnum = str(month).zfill(2)
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # determine which table to read
    pathhere = const.path_to_weather + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'Tbase.txt'

    # read in the table
    years, months, days, vals = read_binary_weather(pathhere)

    # retrieve the requested temperature value(s)
    if (form == 'exact'):
        ind = ((years == int(year)) & (days == int(day)))
        if (np.array(ind).sum() == 0):
            raise Exception('No weather on file for the selected date.')
        T = vals[ind][0]

    else:
        if (form == 'mean'):
            T = np.mean(vals)
        elif (form == 'median'):
            T = np.median(vals)
        elif (form == 'good'):
            T = np.percentile(vals,15.87,axis=0)
        elif (form == 'bad'):
            T = np.percentile(vals,84.13,axis=0)
        elif (form == 'all'):
            T = vals

    return T

def PWV(site, form='exact', month='Apr', day=15, year=2015):
    """
    Retrieve the precipitable water vapor (PWV) information for a specified site.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str or int): The month for which to report weather
      day (str or int): The day of the month for which to report weather; only used for form = 'exact'
      year (str or int): The year for which to report weather; only used for form = 'exact'

    Returns:
      (float): The requested PWV value(s), in mm; if form = 'all', then returns a numpy.ndarray
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif str(month).zfill(2) in monthnums:
        monthnum = str(month).zfill(2)
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # determine which table to read
    pathhere = const.path_to_weather + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'PWV.txt'

    # read in the table
    years, months, days, vals = read_binary_weather(pathhere)

    # retrieve the requested temperature value(s)
    if (form == 'exact'):
        ind = ((years == int(year)) & (days == int(day)))
        if (np.array(ind).sum() == 0):
            raise Exception('No weather on file for the selected date.')
        pwv = vals[ind][0]

    else:
        if (form == 'mean'):
            pwv = np.mean(vals)
        elif (form == 'median'):
            pwv = np.median(vals)
        elif (form == 'good'):
            pwv = np.percentile(vals,15.87,axis=0)
        elif (form == 'bad'):
            pwv = np.percentile(vals,84.13,axis=0)
        elif (form == 'all'):
            pwv = vals

    return pwv

def windspeed(site, form='exact', month='Apr', day=15, year=2015):
    """
    Retrieve the windspeed information for a specified site.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str or int): The month for which to report weather
      day (str or int): The day of the month for which to report weather; only used for form = 'exact'
      year (str or int): The year for which to report weather; only used for form = 'exact'

    Returns:
      (float): The requested windspeed value(s), in m/s; if form = 'all', then returns a numpy.ndarray
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif str(month).zfill(2) in monthnums:
        monthnum = str(month).zfill(2)
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # determine which table to read
    pathhere = const.path_to_weather + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'windspeed.txt'

    # read in the table
    years, months, days, vals = read_binary_weather(pathhere)

    # retrieve the requested temperature value(s)
    if (form == 'exact'):
        ind = ((years == int(year)) & (days == int(day)))
        if (np.array(ind).sum() == 0):
            raise Exception('No weather on file for the selected date.')
        ws = vals[ind][0]

    else:
        if (form == 'mean'):
            ws = np.mean(vals)
        elif (form == 'median'):
            ws = np.median(vals)
        elif (form == 'good'):
            ws = np.percentile(vals,15.87,axis=0)
        elif (form == 'bad'):
            ws = np.percentile(vals,84.13,axis=0)
        elif (form == 'all'):
            ws = vals

    return ws
