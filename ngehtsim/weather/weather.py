###################################################
# imports

import numpy as np
import ngehtutil as ng
import ngehtsim.const_def as const
import os

###################################################
# function definitions

def opacity(site, form='mean', month='Apr', day=15, year=2015, freq='230'):
    """
    Retrieve the zenith opacity information for a specified site.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str): The month for which to report weather
      day (int): The day of the month for which to report weather; only used for form = 'exact'
      year (int): The year for which to report weather; only used for form = 'exact'
      freq (str): The observing frequency; can be '86', '230', '345', '410', or '690'

    Returns:
      (float): The requested opacity value; if form = 'all', then returns a numpy.ndarray
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif month in monthnums:
        monthnum = month
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # check frequency
    if freq not in ['86','230','345','410','690']:
        raise Exception('Specified frequency not recognized; please pick one of 86, 230, 345, 410, or 690.')

    # determine which table to read
    pathhere = os.path.abspath(const.path_to_weather) + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'mean_SEFD_info_' + freq + '.csv'

    # read in the table
    yeardum, monthdum, daydum, taubase, Tb = np.loadtxt(pathhere,skiprows=7,unpack=True,delimiter=',')

    # retrieve the requested opacity value
    if (form == 'mean'):
        tau = np.mean(taubase)
    elif (form == 'median'):
        tau = np.median(taubase)
    elif (form == 'good'):
        tau = np.percentile(taubase,15.87)
    elif (form == 'bad'):
        tau = np.percentile(taubase,84.13)
    elif (form == 'exact'):
        index = ((yeardum == int(year)) & (daydum == int(day)))
        if (np.array(index).sum() == 0):
            raise Exception('No weather on file for the selected date!')
        tau = taubase[index][0]
    elif (form == 'all'):
        tau = taubase

    return tau


def brightness_temperature(site, form='mean', month='Apr', day=15, year=2015, freq='230'):
    """
    Retrieve the zenith brightness temperature information for a specified site.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str): The month for which to report weather
      day (int): The day of the month for which to report weather; only used for form = 'exact'
      year (int): The year for which to report weather; only used for form = 'exact'
      freq (str): The observing frequency; can be '86', '230', '345', '410', or '690'

    Returns:
      (float): The requested brightness temperature value, in K; if form = 'all', then returns a numpy.ndarray
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif month in monthnums:
        monthnum = month
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # check frequency
    if freq not in ['86','230','345','410','690']:
        raise Exception('Specified frequency not recognized; please pick one of 86, 230, 345, 410, or 690.')

    # determine which table to read
    pathhere = os.path.abspath(const.path_to_weather) + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'mean_SEFD_info_' + freq + '.csv'

    # read in the table
    yeardum, monthdum, daydum, tau, Tbbase = np.loadtxt(pathhere,skiprows=7,unpack=True,delimiter=',')

    # retrieve the requested brightness temperature value
    if (form == 'mean'):
        Tb = np.mean(Tbbase)
    elif (form == 'median'):
        Tb = np.median(Tbbase)
    elif (form == 'good'):
        Tb = np.percentile(Tbbase,15.87)
    elif (form == 'bad'):
        Tb = np.percentile(Tbbase,84.13)
    elif (form == 'exact'):
        index = ((yeardum == int(year)) & (daydum == int(day)))
        if (np.array(index).sum() == 0):
            raise Exception('No weather on file for the selected date!')
        Tb = Tbbase[index][0]
    elif (form == 'all'):
        Tb = Tbbase

    return Tb


def pressure(site, form='mean', month='Apr', day=15, year=2015):
    """
    Retrieve the surface pressure information for a specified site.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str): The month for which to report weather
      day (int): The day of the month for which to report weather; only used for form = 'exact'
      year (int): The year for which to report weather; only used for form = 'exact'

    Returns:
      (float): The requested pressure value, in mbar; if form = 'all', then returns a numpy.ndarray
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif month in monthnums:
        monthnum = month
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # determine which table to read
    pathhere = os.path.abspath(const.path_to_weather) + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'mean_Pbase.csv'

    # read in the table
    yeardum, monthdum, daydum, Pbase = np.loadtxt(pathhere,skiprows=6,unpack=True,delimiter=',')

    # retrieve the requested pressure value
    if (form == 'mean'):
        P = np.mean(Pbase)
    elif (form == 'median'):
        P = np.median(Pbase)
    elif (form == 'good'):
        P = np.percentile(Pbase,15.87)
    elif (form == 'bad'):
        P = np.percentile(Pbase,84.13)
    elif (form == 'exact'):
        index = ((yeardum == int(year)) & (daydum == int(day)))
        if (np.array(index).sum() == 0):
            raise Exception('No weather on file for the selected date!')
        P = Pbase[index][0]
    elif (form == 'all'):
        P = Pbase

    return P


def temperature(site, form='mean', month='Apr', day=15, year=2015):
    """
    Retrieve the surface temperature information for a specified site.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str): The month for which to report weather
      day (int): The day of the month for which to report weather; only used for form = 'exact'
      year (int): The year for which to report weather; only used for form = 'exact'

    Returns:
      (float): The requested temperature value, in K; if form = 'all', then returns a numpy.ndarray
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif month in monthnums:
        monthnum = month
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # determine which table to read
    pathhere = os.path.abspath(const.path_to_weather) + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'mean_Tbase.csv'

    # read in the table
    yeardum, monthdum, daydum, Tbase = np.loadtxt(pathhere,skiprows=6,unpack=True,delimiter=',')

    # retrieve the requested temperature value
    if (form == 'mean'):
        T = np.mean(Tbase)
    elif (form == 'median'):
        T = np.median(Tbase)
    elif (form == 'good'):
        T = np.percentile(Tbase,15.87)
    elif (form == 'bad'):
        T = np.percentile(Tbase,84.13)
    elif (form == 'exact'):
        index = ((yeardum == int(year)) & (daydum == int(day)))
        if (np.array(index).sum() == 0):
            raise Exception('No weather on file for the selected date!')
        T = Tbase[index][0]
    elif (form == 'all'):
        T = Tbase

    return T


def PWV(site, form='mean', month='Apr', day=15, year=2015):
    """
    Retrieve the precipitable water vapor (PWV) information for a specified site.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str): The month for which to report weather
      day (int): The day of the month for which to report weather; only used for form = 'exact'
      year (int): The year for which to report weather; only used for form = 'exact'

    Returns:
      (float): The requested PWV value, in mm; if form = 'all', then returns a numpy.ndarray
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif month in monthnums:
        monthnum = month
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # determine which table to read
    pathhere = os.path.abspath(const.path_to_weather) + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'mean_PWV.csv'

    # read in the table
    yeardum, monthdum, daydum, PWVhere = np.loadtxt(pathhere,skiprows=6,unpack=True,delimiter=',')

    # retrieve the requested temperature value
    if (form == 'mean'):
        pwv = np.mean(PWVhere)
    elif (form == 'median'):
        pwv = np.median(PWVhere)
    elif (form == 'good'):
        pwv = np.percentile(PWVhere,15.87)
    elif (form == 'bad'):
        pwv = np.percentile(PWVhere,84.13)
    elif (form == 'exact'):
        index = ((yeardum == int(year)) & (daydum == int(day)))
        if (np.array(index).sum() == 0):
            raise Exception('No weather on file for the selected date!')
        pwv = PWVhere[index][0]
    elif (form == 'all'):
        pwv = PWVhere

    return pwv


def windspeed(site, form='mean', month='Apr', day=15, year=2015):
    """
    Retrieve the surface wind speed information for a specified site.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str): The month for which to report weather
      day (int): The day of the month for which to report weather; only used for form = 'exact'
      year (int): The year for which to report weather; only used for form = 'exact'

    Returns:
      (float): The requested wind speed value, in m/s; if form = 'all', then returns a numpy.ndarray
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif month in monthnums:
        monthnum = month
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # determine which table to read
    pathhere = os.path.abspath(const.path_to_weather) + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'mean_wind_speed.csv'

    # read in the table
    yeardum, monthdum, daydum, WSbase = np.loadtxt(pathhere,skiprows=6,unpack=True,delimiter=',')

    # retrieve the requested temperature value
    if (form == 'mean'):
        WS = np.mean(WSbase)
    elif (form == 'median'):
        WS = np.median(WSbase)
    elif (form == 'good'):
        WS = np.percentile(WSbase,15.87)
    elif (form == 'bad'):
        WS = np.percentile(WSbase,84.13)
    elif (form == 'exact'):
        index = ((yeardum == int(year)) & (daydum == int(day)))
        if (np.array(index).sum() == 0):
            raise Exception('No weather on file for the selected date!')
        WS = WSbase[index][0]
    elif (form == 'all'):
        WS = WSbase

    return WS


def relative_humidity(site, form='mean', month='Apr', day=15, year=2015):
    """
    Retrieve the relative humidity information for a specified site.

    Args:
      site (str): The name of the site
      form (str): The form of value to report; can be 'mean', 'median', 'good', 'bad', exact', or 'all'
      month (str): The month for which to report weather
      day (int): The day of the month for which to report weather; only used for form = 'exact'
      year (int): The year for which to report weather; only used for form = 'exact'

    Returns:
      (float): The requested relative humidity value; if form = 'all', then returns a numpy.ndarray
    """

    # extract month number
    monthnums = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
    monthnams = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if month in monthnams:
        monthnum = monthnums[monthnams == month][0]
        monthnam = month
    elif month in monthnums:
        monthnum = month
        monthnam = monthnams[monthnums == month][0]
    else:
        raise Exception('Specified month not recognized; please use either a three-letter abbreviation (e.g., Jan, Apr) or else a two-digit number (e.g., 03, 10).')

    # determine which table to read
    pathhere = os.path.abspath(const.path_to_weather) + '/'
    pathhere += site + '/'
    pathhere += monthnum + monthnam + '/'
    pathhere += 'mean_RH.csv'

    # read in the table
    yeardum, monthdum, daydum, RH = np.loadtxt(pathhere,skiprows=6,unpack=True,delimiter=',')

    # retrieve the requested temperature value
    if (form == 'mean'):
        rh = np.mean(RH)
    elif (form == 'median'):
        rh = np.median(RH)
    elif (form == 'good'):
        rh = np.percentile(RH,15.87)
    elif (form == 'bad'):
        rh = np.percentile(RH,84.13)
    elif (form == 'exact'):
        index = ((yeardum == int(year)) & (daydum == int(day)))
        if (np.array(index).sum() == 0):
            raise Exception('No weather on file for the selected date!')
        rh = RH[index][0]
    elif (form == 'all'):
        rh = RH

    return rh
