import numpy as np
import matplotlib.pyplot as plt
from process_data import *
from matplotlib.colors import LogNorm

def plot_mag_helper(ax, i, mag, start, end):
    # ax is the axis object and i is the index of the axis window to 
    # start plotting the data in
    start_time = np.datetime64(start)
    end_time = np.datetime64(end)
    mag_start = find_nearest_time(mag['times'], start_time)
    mag_end = find_nearest_time(mag['times'], end_time)

    mag_x = mag['B'][:,0]
    mag_y = mag['B'][:,1]
    mag_z = mag['B'][:,2]
    btot = np.sqrt(np.square(mag['B'][:,0]) + np.square(mag['B'][:,1]) + 
                   np.square(mag['B'][:,2]))

    # using standard colours for x, y, z vector components, but you can change
    # the colours here to whatever your preference
    ax[i].plot(mag['times'][mag_start:mag_end], mag_x[mag_start:mag_end], 
               label = r"$B_x$", color='b')
    ax[i].plot(mag['times'][mag_start:mag_end], mag_y[mag_start:mag_end], 
               label = r"$B_y$", color='g')
    ax[i].plot(mag['times'][mag_start:mag_end], mag_z[mag_start:mag_end],
               label = r"$B_z$", color='r')
    ax[i+1].plot(mag['times'][mag_start:mag_end], btot[mag_start:mag_end],
                 color='orchid')

    ax[i].set_ylabel(r'$B$ (nT)')
    ax[i].legend(framealpha = 0.95)
    ax[i+1].set_ylabel(r'$B_{tot}$ (nT)')

def plot_mag(mag, start, end):
    """
    Plots the magnetic field components (Bx, By, Bz) as well as the magnitude of
    the magnetic field |B| over a specified time range.

    Parameters:
    mag (dict): MAG dictionary containing magnetic field data.
    start (str): Start time of the plot in the format "YYYY-MM-DDTHH:MM:SS".
    end (str): End time of the plot in the format "YYYY-MM-DDTHH:MM:SS".

    Returns:
    None
    """
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
    plot_mag_helper(ax, 0, mag, start, end)
    ax[1].set_xlabel('Date/Time')
    plt.show()

def plot_moments_helper(ax, i, mag, swia_mom, start, end):
    start_time = np.datetime64(start)
    end_time = np.datetime64(end)

    mag_start = find_nearest_time(mag['times'], start_time)
    mag_end = find_nearest_time(mag['times'], end_time) 
    swia_start = find_nearest_time(swia_mom['times'], start_time)
    swia_end = find_nearest_time(swia_mom['times'], end_time) 
    pressure = compute_pressure(mag, swia_mom)
    prs_start = find_nearest_time(pressure['times'], start_time)
    prs_end = find_nearest_time(pressure['times'], end_time)

    v_x = swia_mom['vel'][:,0]
    v_y = swia_mom['vel'][:,1]
    v_z = swia_mom['vel'][:,2]    

    ax[i].plot(swia_mom['times'][swia_start:swia_end], swia_mom['density'][swia_start:swia_end])
    ax[i].set_ylabel(r'Density ($cm^{-3}$)')
    ax[i].set_yscale('log')

    ax[i+1].plot(swia_mom['times'][swia_start:swia_end], v_x[swia_start:swia_end],
                 label = r"$v_x$", color='b')
    ax[i+1].plot(swia_mom['times'][swia_start:swia_end], v_y[swia_start:swia_end],
                 label = r"$v_y$", color='g')
    ax[i+1].plot(swia_mom['times'][swia_start:swia_end], v_z[swia_start:swia_end],
                 label = r"$v_z$", color='r')
    ax[i+1].legend(framealpha = 0.95)
    ax[i+1].set_ylabel('Velocity (km/s)')

    ax[i+2].plot(swia_mom['times'][swia_start:swia_end], pressure['plasma_pressure'][swia_start:swia_end],
               label = "Plasma pressure")
    ax[i+2].plot(mag['times'][mag_start:mag_end], pressure['magnetic_pressure'][mag_start:mag_end],
               label = "Magnetic pressure")
    ax[i+2].plot(pressure['times'][prs_start:prs_end], pressure['total_pressure'][prs_start:prs_end],
               label = "Total pressure")
    ax[i+2].legend(framealpha = 0.95)
    ax[i+2].set_ylabel('Pressure (nPa)')
    ax[i+2].set_yscale('log')

def plot_moments(mag, swia_mom, start, end):
    """
    Plots the moments of the SWIA instrument data over a specified time range.

    Parameters:
    mag (dict): MAG dictionary containing magnetic field data.
    swia (dict): SWIA dictionary containing ion moment data.
    start (str): Start time of the plot in the format "YYYY-MM-DDTHH:MM:SS".
    end (str): End time of the plot in the format "YYYY-MM-DDTHH:MM:SS".

    Returns:
    None
    """
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8,6)) 
    plot_moments_helper(ax, 0, mag, swia_mom, start, end)
    ax[2].set_xlabel('Date/Time')
    plt.show()

def plot_swea_helper(fig, ax, i, swea, start, end):
    start_time = np.datetime64(start)
    end_time = np.datetime64(end)

    swea_start = find_nearest_time(swea['times'], start_time)
    swea_end = find_nearest_time(swea['times'], end_time)
    flux_transp = np.transpose(swea['flux'][swea_start:swea_end])

    if i == 0:
        p = ax.pcolormesh(swea['times'][swea_start:swea_end], swea['v'],
                                    flux_transp, norm = LogNorm(), shading = 'auto')
        cb = fig.colorbar(p, ax = ax, pad = 0.01)
        ax.set_ylabel('SWEA \n Energy (eV)')
        ax.set_yscale('log')
        cb.set_label(r'EFlux (eV/(eV $cm^{2}$ sr s))')
    else:
        p = ax[i].pcolormesh(swea['times'][swea_start:swea_end], swea['v'],
                                    flux_transp, norm = LogNorm(), shading = 'auto')
        cb = fig.colorbar(p, ax = ax[i], pad = 0.01)
        ax[i].set_ylabel('SWEA \n Energy (eV)')
        ax[i].set_yscale('log')

def plot_swea(swea, start, end):
    """
    Plots the SWEA energy spectrogram over a specified time range.

    Parameters:
    swea (dict): SWEA dictionary containing energy flux data.
    start (str): Start time of the plot in the format "YYYY-MM-DDTHH:MM:SS".
    end (str): End time of the plot in the format "YYYY-MM-DDTHH:MM:SS".

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(8,2))
    plot_swea_helper(fig, ax, 0, swea, start, end)
    ax.set_xlabel('Date/Time')
    plt.show()

def plot_swia_flux_helper(fig, ax, i, swia_flux, start, end):
    start_time = np.datetime64(start)
    end_time = np.datetime64(end)

    swia_start = find_nearest_time(swia_flux['times'], start_time)
    swia_end = find_nearest_time(swia_flux['times'], end_time)
    flux_transp = np.transpose(swia_flux['flux'][swia_start:swia_end])

    if i == 0:
        p = ax.pcolormesh(swia_flux['times'][swia_start:swia_end], swia_flux['v'],
                                flux_transp, norm = LogNorm(), shading = 'auto')
        cb = fig.colorbar(p, ax = ax, pad = 0.01)
        ax.set_yscale('log')
        ax.set_ylabel('SWIA \n Energy (eV)')
        cb.set_label(r'EFlux (eV/(eV $cm^{2}$ sr s))')
    else:
        p = ax[i].pcolormesh(swia_flux['times'][swia_start:swia_end], swia_flux['v'],
                                flux_transp, norm = LogNorm(), shading = 'auto')
        cb = fig.colorbar(p, ax = ax[i], pad = 0.01)
        ax[i].set_yscale('log')
        ax[i].set_ylabel('SWIA \n Energy (eV)')

def plot_swia_flux(swia_flux, start, end):
    """
    Plots the SWIA energy spectrogram over a specified time range.

    Parameters:
    - swia_flux (dict): SWIA dictionary containing energy flux data.
    - start (str): Start time of the plot in the format "YYYY-MM-DDTHH:MM:SS".
    - end (str): End time of the plot in the format "YYYY-MM-DDTHH:MM:SS".

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=(8,2))
    plot_swia_flux_helper(fig, ax, 0, swia_flux, start, end)
    ax.set_xlabel('Date/Time')
    plt.show()

def plot_static_helper(fig, ax, i, static, start, end):
    start_time = np.datetime64(start)
    end_time = np.datetime64(end)

    static_start = find_nearest_time(static['times'], start_time)
    static_end = find_nearest_time(static['times'], end_time)
    hydrogen, oxygen, o2, energy_grid = static_spectrogram(static, static_start, static_end)
    
    swp_ind = static['sweep_index']
    unique_sweeps = list(np.unique(swp_ind[static_start:static_end]))
    swp_index = unique_sweeps.index(swp_ind[static_start])
    current_swp = swp_ind[static_start]
    start_index = 0
    static_time_slice = static['times'][static_start:static_end]

    for k in range(static_end - static_start):
        if swp_ind[static_start + k] != current_swp:
            h_flux = np.transpose(hydrogen[start_index:k,swp_index*32:(swp_index+1)*32])
            p_h = ax[i].pcolormesh(static_time_slice[start_index:k], energy_grid[
                swp_index*32:(swp_index+1)*32], h_flux, norm = LogNorm(), 
                shading = 'auto', snap = True) 
            # #make sure all colorbars in same subplot have same range
            # p_h.set_clim(vmax = np.nanmax(hydrogen), vmin = (np.nanmin(hydrogen) if np.nanmin(hydrogen) > 0 else 1e-4))
            p_h.set_clim(vmax = np.nanmax(hydrogen))
            
            o_flux = np.transpose(oxygen[start_index:k,swp_index*32:(swp_index+1)*32])
            p_o = ax[i+1].pcolormesh(static_time_slice[start_index:k], energy_grid[
                swp_index*32:(swp_index+1)*32], o_flux, norm = LogNorm(), 
                shading = 'auto', snap = True)
            # p_o.set_clim(vmax = np.nanmax(oxygen), vmin = (np.nanmin(oxygen) if np.nanmin(oxygen) > 0 else 1e-4))
            p_o.set_clim(vmax = np.nanmax(oxygen))
            
            o2_flux = np.transpose(o2[start_index:k,swp_index*32:(swp_index+1)*32])
            p_o2 = ax[i+2].pcolormesh(static_time_slice[start_index:k], energy_grid[
                swp_index*32:(swp_index+1)*32], o2_flux, norm = LogNorm(),
                shading = 'auto', snap = True)
            # p_o2.set_clim(vmax = np.nanmax(o2), vmin = (np.nanmin(o2) if np.nanmin(o2) > 0 else 1e-4))
            p_o2.set_clim(vmax = np.nanmax(o2))

            current_swp = swp_ind[static_start + k]
            swp_index = unique_sweeps.index(current_swp)
            start_index = k
            
        elif k == (static_end - static_start - 1):
            h_flux = np.transpose(hydrogen[start_index:k,swp_index*32:(swp_index+1)*32])
            p_h = ax[i].pcolormesh(static_time_slice[start_index:k], energy_grid[
                swp_index*32:(swp_index+1)*32], h_flux, norm = LogNorm(), 
                shading = 'auto', snap = True)
            # p_h.set_clim(vmax = np.nanmax(hydrogen), vmin = (np.nanmin(hydrogen) if np.nanmin(hydrogen) > 0 else 1e-4))
            p_h.set_clim(vmax = np.nanmax(hydrogen))
            
            o_flux = np.transpose(oxygen[start_index:k,swp_index*32:(swp_index+1)*32])
            p_o = ax[i+1].pcolormesh(static_time_slice[start_index:k], energy_grid[
                swp_index*32:(swp_index+1)*32], o_flux, norm = LogNorm(),
                shading = 'auto', snap = True)
            # p_o.set_clim(vmax = np.nanmax(oxygen), vmin = (np.nanmin(oxygen) if np.nanmin(oxygen) > 0 else 1e-4))
            p_o.set_clim(vmax = np.nanmax(oxygen))
            
            o2_flux = np.transpose(o2[start_index:k,swp_index*32:(swp_index+1)*32])
            p_o2 = ax[i+2].pcolormesh(static_time_slice[start_index:k], energy_grid[
                swp_index*32:(swp_index+1)*32], o2_flux, norm = LogNorm(),
                shading = 'auto', snap = True)
            # p_o2.set_clim(vmax = np.nanmax(o2), vmin = (np.nanmin(o2) if np.nanmin(o2) > 0 else 1e-4))
            p_o2.set_clim(vmax = np.nanmax(o2))

    # hydrogen
    cb = fig.colorbar(p_h, ax = ax[i], pad = 0.01)
    if i != 0:
        cb.set_label(r'EFlux (eV/(eV $cm^{2}$ sr s))', fontsize = 'large')
    ax[i].set_yscale('log')
    ax[i].set_ylabel(r'STATIC $H^+$' '\n Energy (eV)')

    # oxygen
    cb = fig.colorbar(p_o, ax =  ax[i+1], pad = 0.01)
    if i == 0:
        cb.set_label(r'EFlux (eV/(eV $cm^{2}$ sr s))', fontsize = 'large')
    ax[i+1].set_yscale('log')
    ax[i+1].set_ylabel(r'STATIC $O^+$' '\n Energy (eV)')

    # o2
    cb = fig.colorbar(p_o2, ax = ax[i+2], pad = 0.01)
    ax[i+2].set_yscale('log')
    ax[i+2].set_ylabel(r'STATIC $O_{2}^{+}$' '\n Energy (eV)')

def plot_static(static, start, end):
    """
    Plot STATIC energy spectrograms for H+, O+ and O2+, over a given time range.

    Parameters:
    static (dict): STATIC dictionary containing spectrogram data.
    start (str): Start time of the plot in the format "YYYY-MM-DDTHH:MM:SS".
    end (str): End time of the plot in the format "YYYY-MM-DDTHH:MM:SS".

    Returns:
    None
    """
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8,6))
    plot_static_helper(fig, ax, 0, static, start, end)
    ax[2].set_xlabel('Date/Time')
    plt.show()

def plot_all(mag, swia_mom, swea, swia_flux, static, start, end):
    """
    Plots all the relevant data for the MAVEN mission over a specified time range.

    Parameters:
    mag (dict): MAG dictionary containing magnetic field data.
    swia_mom (dict): SWIA dictionary containing ion moment data.
    swea (dict): SWEA dictionary containing energy flux data.
    swia_flux (dict): SWIA dictionary containing energy flux data.
    static (dict): STATIC dictionary containing spectrogram data.
    start (str): Start time of the plot in the format "YYYY-MM-DDTHH:MM:SS".
    end (str): End time of the plot in the format "YYYY-MM-DDTHH:MM:SS".

    Returns:
    None
    """
    fig, ax = plt.subplots(10, 1, sharex=True, figsize=(8, 15), layout='constrained')
    plot_mag_helper(ax, 0, mag, start, end)
    plot_moments_helper(ax, 2, mag, swia_mom, start, end)
    plot_swea_helper(fig, ax, 5, swea, start, end)
    plot_swia_flux_helper(fig, ax, 6, swia_flux, start, end)
    plot_static_helper(fig, ax, 7, static, start, end)
    ax[9].set_xlabel('Date/Time')
    plt.margins(x=0.02)
    plt.show()