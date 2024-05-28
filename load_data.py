import numpy as np
import pandas as pd
import pytplot
import pyspedas
import os

def load_mag(days_of_interest):
    """
    Loads MAG data from MAVEN using pyspedas.

    Parameters:
    day_of_interest (list): List containing dates of interest, in the form
        "YYYY-MM-DD"

    Returns:
    mag: A dictionary containing the following keys:
        - times: An array of size n with the time (np.datetime64) of each point
                in the time series at the precision of MAG.
        - B: An (n,3) array containing the vector components (Bx, By, Bz) of the 
                magnetic field B at each point in the time series.
        - posn: An (n,3) array containing the vector components (x, y, z) of 
                MAVEN's position (in MSO coordinates) at each point in the time
                series.
    """
    for day in days_of_interest:
        pyspedas.maven.mag(trange=[day, day])
    mag = pytplot.data_quants['OB_B']
    times = []
    for nanos in mag.time.values:   # if in UNIX nanosecond format
        ts = pd.Timestamp(nanos, tz = 'US/Eastern')
        times.append(pd.Timestamp.to_datetime64(ts))
    times = np.array(times)
    Bvals = mag.values[:, 0:3]
    posn = pytplot.data_quants['POSN'].values
    posn = posn/3389.5 # normalized to Mars radii
    return {'times': times, 'B': Bvals, 'posn': posn}

def load_mag_sts(directory, filenames):
    """
    Loads MAG data from MAVEN with pre-existing .sts files.

    Parameters:
    directory (str): Path to directory containing .sts files.
    filenames (list): List of .sts filenames to load.

    Returns:
    mag: A dictionary containing the following keys:
        - times: An array of size n with the time (np.datetime64) of each point
                in the time series at the precision of MAG.
        - B: An (n,3) array containing the vector components (Bx, By, Bz) of the 
                magnetic field B at each point in the time series.
        - posn: An (n,3) array containing the vector components (x, y, z) of 
                MAVEN's position (in MSO coordinates) at each point in the time
                series.
    """
    for file in filenames:
        f = os.path.join(directory, file)
        pytplot.sts_to_tplot(f, prefix='',suffix='', merge=True)
    mag = pytplot.data_quants['OB_B']
    times = []
    for nanos in mag.time.values:   # if in UNIX nanosecond format
        ts = pd.Timestamp(nanos, tz = 'US/Eastern')
        times.append(pd.Timestamp.to_datetime64(ts))
    times = np.array(times)
    Bvals = mag.values[:, 0:3]
    posn = pytplot.data_quants['POSN'].values
    posn = posn/3389.5 # normalized to Mars radii
    return {'times': times, 'B': Bvals, 'posn': posn}

def load_swea(days_of_interest):
    """
    Loads SWEA flux data from MAVEN using pyspedas.

    Parameters:
    day_of_interest (list): List containing dates of interest, in the form
        "YYYY-MM-DD"

    Returns:
    swea: A dictionary containing the following keys:
        - times: An array of size n with the time (np.datetime64) of each point
                in the time series at the precision of SWEA.
        - flux: An (n,64) array containing the eflux (eV/(eV cm^2 sr s)) of 
                electrons measured by SWEA at each of its 64 energy bands, at
                each point in the time series.
        - v: An array of size 64 containing the energy values (eV) of each of
                SWEA's measured energy bands.
    """
    for day in days_of_interest:
        pyspedas.maven.swea(trange=[day, day])
    swea = pytplot.data_quants['diff_en_fluxes_svyspec']
    swea_times = swea.time.values
    swea_flux = swea.values
    swea_v = swea.v.values
    return {'times': swea_times, 'flux': swea_flux, 'v': swea_v}

def load_swia_mom(days_of_interest):
    """
    Loads SWIA ion moments from MAVEN using pyspedas.

    Parameters:
    day_of_interest (list): List containing dates of interest, in the form
        "YYYY-MM-DD"

    Returns:
    swia_mom: A dictionary containing the following keys:
        - times: An array of size n with the time (np.datetime64) of each point
                in the time series at the precision of SWIA.
        - temp: An (n,3) array containing the vector components (Tx, Ty, Tz) of
                the ion temperature at each point in the time series.
        - vel: An (n,3) array containing the vector components (v_x, v_y, v_z) of
                the ion velocity at each point in the time series.
        - density: An array of size n containing the ion density measured at 
                each point in the time series.
    """
    for day in days_of_interest:
        pyspedas.maven.swia(trange=[day, day], datatype = 'onboardsvymom')
    temperature = pytplot.data_quants['temperature_mso_onboardsvymom']
    swia_time = temperature.time.values
    temp = temperature.values
    velocity = pytplot.data_quants['velocity_mso_onboardsvymom'].values
    density = pytplot.data_quants['density_onboardsvymom'].values
    return {'times': swia_time, 'temp': temp, 'vel': velocity, 'density': density}

def load_swia_flux(days_of_interest):
    """
    Loads SWIA flux data from MAVEN using pyspedas.

    Parameters:
    day_of_interest (list): List containing dates of interest, in the form
        "YYYY-MM-DD"

    Returns:
    swia_flux: A dictionary containing the following keys:
        - times: An array of size n with the time (np.datetime64) of each point
                in the time series at the precision of SWIA.
        - flux: An (n,48) array containing the eflux (eV/(eV cm^2 sr s)) of 
                ions (assumed to be H+ protons) measured by SWIA at each of its
                64 energy bands, at each point in the time series.
        - v: An array of size 48 containing the energy values (eV) of each of
                SWIA's measured energy bands.
    """
    for day in days_of_interest:
        pyspedas.maven.swia(trange=[day, day], datatype = 'onboardsvyspec')
    swia = pytplot.data_quants['spectra_diff_en_fluxes_onboardsvyspec']
    swia_times = swia.time.values
    swia_flux = swia.values
    swia_v = swia.v.values
    return {'times': swia_times, 'flux': swia_flux, 'v': swia_v}

def load_static(days_of_interest):
    """
    Loads STATIC flux data from MAVEN using pyspedas.

    Parameters:
    day_of_interest (list): List containing dates of interest, in the form
        "YYYY-MM-DD"

    Returns:
    static: A dictionary containing the following keys:
        - times: An array of size n with the time (np.datetime64) of each point
                in the time series at the precision of STATIC.
        - flux: An (n,48) array containing the eflux (eV/(eV cm^2 sr s)) of 
                ions (assumed to be H+ protons) measured by SWIA at each of its
                64 energy bands, at each point in the time series.
        - v: An array of size 48 containing the energy values (eV) of each of
                SWIA's measured energy bands.
    """
    for day in days_of_interest:
        pyspedas.maven.sta(trange=[day, day], level='l2',datatype='c6', 
                           get_support_data = True)
    eflux = pytplot.data_quants['eflux_c6-32e64m']
    static_times = eflux.time.values
    swp_ind = pytplot.data_quants['swp_ind_c6-32e64m']
    energy = pytplot.data_quants['energy_c6-32e64m']['data']
    mass = pytplot.data_quants['mass_arr_c6-32e64m']['data']

    mas = []
    for i in range(len(swp_ind.values)):
        sweep = swp_ind.values[i]
        mas.append(mass[:,:,sweep])
    mas = np.asarray(mas)

    return {'times': static_times, 'flux': eflux.values,
            'sweep_index': swp_ind.values, 'energy': energy, 'mass': mas}