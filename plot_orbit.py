import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import process_data as proc

def get_cylindrical_rho(mag):
    """
    Calculate the cylindrical rho coordinate from position data (using MAG). The
    cylindrical coordinate is defined as sqrt(y^2 + z^2), where y and z are the
    y and z coordinates of MAVEN's position in MSO coordinates.

    Parameters:
    mag (dict): Dictionary containing magnetic field data, following style of
        :func:`./load_data.load_mag`.

    Returns:
    out (np.ndarray): Array containing the calculated cylindrical rho coordinate.
    """
    rho = np.sqrt(np.square(mag['posn'][:,1]) + np.square(mag['posn'][:,2]))
    return rho

def get_IMB_BS():
    """
    Calculate the x and r coordinates for induced magnetospheric boundary (IMB)
    and bow shock (BS) points on an orbit.

    Returns:
    xBS (np.ndarray): List of x coordinates for BS points.
    rBS (np.ndarray): List of r coordinates for BS points.
    xIMB (np.ndarray): List of x coordinates for IMB points.
    rIMB (np.ndarray): List of r coordinates for IMB points.
    """
    theta = np.arange(0, 180, 1)
    xBS = []
    rBS = []
    xIMB = []
    rIMB = []
    for j in range(len(theta)):
        angle = math.pi * theta[j]/180
        dummy = 1.93/(1 + 1.02 * math.cos(angle))
        xBS.append(0.72 + dummy*math.cos(angle))
        rBS.append(dummy * math.sin(angle))
        dummy2 = 0.96/(1 + 0.9* math.cos(angle))
        xIMB.append(0.78 + dummy2 * math.cos(angle))
        rIMB.append(dummy2 * math.sin(angle))
    return np.array(xBS), np.array(rBS), np.array(xIMB), np.array(rIMB)

def plot_orbit_MSO(mag, interest, savefile = None):
    """
    Plot the orbit of MAVEN in MSO coordinates, with the bow shock and induced
    magnetospheric boundary marked. The orbit is plotted in four different planes:
    x-y, x-z, y-z, and x-rho.

    Parameters:
    mag (dict): Dictionary containing magnetic field data, following style of
        :func:`./load_data.load_mag`.
    interest (list): List of strings in the format 'YYYY-MM-DD' indicating the
        dates of interest for orbit plot.
    savefile (str): The name of the file to save the plot to; the default value
        is None. If None, the plot will be displayed but not saved.

    Returns:
    None
    """
    # Get position coordinates
    x_pos = mag['posn'][:,0]
    y_pos = mag['posn'][:,1]
    z_pos = mag['posn'][:,2]
    rho = get_cylindrical_rho(mag)
    xBS, rBS, xIMB, rIMB = get_IMB_BS()

    # Create figure and subplots
    fig = plt.figure(constrained_layout = True, figsize=(6, 6))
    axsPos = fig.subplots(2,2)
    colormap = plt.get_cmap("Set2")
    fsize = 'large'
        
    # Plot x-y plane
    axsPos[0,0].set_ylabel(r'$y_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[0,0].set_xlabel(r'$x_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[0,0].plot(xBS[:145], rBS[:145], color = 'black', linewidth=1)
    axsPos[0,0].plot(xBS[:145], np.negative(np.array(rBS[:145])), color='black', linewidth=1)
    axsPos[0,0].plot(xIMB, rIMB, color = 'gray', linewidth=1) 
    axsPos[0,0].plot(xIMB, np.negative(np.array(rIMB)), color = 'gray', linewidth=1)
    day_hemisphere = patches.Wedge((0, 0), 1, 270, 450, facecolor='#ffcccc', zorder =5)
    night_hemisphere = patches.Wedge((0, 0), 1, 90, 270, facecolor='#ad8989', zorder =5)
    axsPos[0,0].add_patch(day_hemisphere)
    axsPos[0,0].add_patch(night_hemisphere)
        
    # Plot x-z plane
    axsPos[0,1].set_ylabel(r'$z_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[0,1].set_xlabel(r'$x_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[0,1].plot(xBS[:145], rBS[:145], color = 'black', linewidth=1)
    axsPos[0,1].plot(xBS[:145], np.negative(np.array(rBS[:145])), color = 'black', linewidth=1)
    axsPos[0,1].plot(xIMB, rIMB, color = 'gray', linewidth=1)
    axsPos[0,1].plot(xIMB, np.negative(np.array(rIMB)), color = 'gray', linewidth=1)
    day_hemisphere = patches.Wedge((0, 0), 1, 270, 450, facecolor='#ffcccc', zorder =5)
    night_hemisphere = patches.Wedge((0, 0), 1, 90, 270, facecolor='#ad8989', zorder =5)
    axsPos[0,1].add_patch(day_hemisphere)
    axsPos[0,1].add_patch(night_hemisphere)

    # Plot y-z plane
    axsPos[1,0].set_ylabel(r'$z_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[1,0].set_xlabel(r'$y_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    circ_BS = plt.Circle((0, 0), radius=2.63, edgecolor='black', facecolor=None, fill=False, linewidth=1)
    circ_IMB = plt.Circle((0, 0), radius=1.44, edgecolor='gray', facecolor=None, fill=False, linewidth=1)
    circ = plt.Circle((0, 0), radius=1, edgecolor=None, facecolor='#ffcccc', zorder = 5)
    axsPos[1,0].add_patch(circ)
    axsPos[1,0].add_patch(circ_IMB)
    axsPos[1,0].add_patch(circ_BS)

    for k in range(3):
        div = k//2
        rem = k%2
        axsPos[div,rem].set_xlim(-3, 3)
        axsPos[div,rem].set_ylim(-3, 3)
        axsPos[div,rem].set_aspect('equal')

    # Plot x-rho plane
    axsPos[1,1].set_ylabel(r'$R_{cyl,MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[1,1].set_xlabel(r'$x_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[1,1].plot(xBS[:145], rBS[:145], color = 'black', label = 'Bow Shock', linewidth=1)
    axsPos[1,1].plot(xIMB, rIMB, color = 'gray', label = 'IMB', linewidth=1)        
    day_hemisphere = patches.Wedge((0, 0), 1, 270, 450, facecolor='#ffcccc', zorder =5)
    night_hemisphere = patches.Wedge((0, 0), 1, 90, 270, facecolor='#ad8989', zorder =5)
    axsPos[1,1].add_patch(day_hemisphere)
    axsPos[1,1].add_patch(night_hemisphere)
    axsPos[1,1].set_xlim(-2, 2)
    axsPos[1,1].set_ylim(0, 4.0)
    axsPos[1,1].set_aspect('equal')

    for i in range(len(interest)):        
        d1 = np.datetime64(interest[i] + 'T00:00:02')
        d2 = np.datetime64(interest[i] + 'T23:59:58')
        
        day_start = proc.find_nearest_time(mag['times'], d1)
        day_end = proc.find_nearest_time(mag['times'], d2)
        
        xpos_slice = x_pos[day_start:day_end]
        ypos_slice = y_pos[day_start:day_end]
        zpos_slice = z_pos[day_start:day_end]
    
        # Hide portion of orbit behind the planet
        hide_xy = zpos_slice < 0
        hide_xz = ypos_slice < 0
        hide_yz = xpos_slice < 0
    
        x_behind_xy = np.copy(xpos_slice)
        x_front_xy = np.copy(xpos_slice)
        x_front_xy[hide_xy] = np.nan
        x_behind_xy[np.invert(hide_xy)] = np.nan
        y_behind_xy = np.copy(ypos_slice)
        y_front_xy = np.copy(ypos_slice)
        y_front_xy[hide_xy] = np.nan
        y_behind_xy[np.invert(hide_xy)] = np.nan

        x_behind_xz = np.copy(xpos_slice)
        x_front_xz = np.copy(xpos_slice)
        x_front_xz[hide_xz] = np.nan
        x_behind_xz[np.invert(hide_xz)] = np.nan
        z_behind_xz = np.copy(zpos_slice)
        z_front_xz = np.copy(zpos_slice)
        z_front_xz[hide_xz] = np.nan
        z_behind_xz[np.invert(hide_xz)] = np.nan

        y_behind_yz = np.copy(ypos_slice)
        y_front_yz = np.copy(ypos_slice)
        y_front_yz[hide_yz] = np.nan
        y_behind_yz[np.invert(hide_yz)] = np.nan
        z_behind_yz = np.copy(zpos_slice)
        z_front_yz = np.copy(zpos_slice)
        z_front_yz[hide_yz] = np.nan
        z_behind_yz[np.invert(hide_yz)] = np.nan

        # Plot the orbit in the x-y plane        
        axsPos[0,0].plot(x_behind_xy, y_behind_xy, linestyle = 'dotted', 
                         color = colormap(i), zorder = 1) 
        axsPos[0,0].plot(x_front_xy, y_front_xy, linestyle = 'dotted', 
                         color = colormap(i), zorder = 10)

        # Plot the orbit in the x-z plane
        axsPos[0,1].plot(x_behind_xz, z_behind_xz, linestyle = 'dotted', 
                         color = colormap(i), zorder = 1) 
        axsPos[0,1].plot(x_front_xz, z_front_xz, linestyle = 'dotted', 
                         color = colormap(i), zorder = 10)

        # Plot the orbit in the y-z plane
        axsPos[1,0].plot(y_behind_yz, z_behind_yz, linestyle = 'dotted', 
                         color = colormap(i), zorder = 1) 
        axsPos[1,0].plot(y_front_yz, z_front_yz, linestyle = 'dotted', 
                         color = colormap(i), zorder = 10)

        # Plot the orbit in the x-rho plane
        axsPos[1,1].plot(x_pos[day_start:day_end], rho[day_start:day_end], 
                         linestyle = 'dotted', color = colormap(i), label=interest[i])
        
    lgd = fig.legend(bbox_to_anchor=(0.5, -0.05), ncols=4, loc='center', fontsize='large') 
    if savefile != None:
        fig.savefig(savefile, format='pdf', dpi=1200, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
    plt.show()

def plot_event_MSO(mag, interest, savefile = None):
    """
    Plot an event at the given time of interest in MSO coordinates, with the bow
    shock and induced magnetospheric boundary marked. The event is plotted in four
    different planes: x-y, x-z, y-z, and x-rho. The corresponding orbit of MAVEN
    on that date is also plotted.

    Parameters:
    mag (dict): Dictionary containing magnetic field data, following style of
        :func:`./load_data.load_mag`.
    interest: list of strings in the format 'YYYY-MM-DDTHH:MM:SS' indicating the
        times of interest for scatter plot.
    savefile (str): The name of the file to save the plot to; the default value
        is None. If None, the plot will be displayed but not saved.

    Returns: 
    None
    """
    x_pos = mag['posn'][:,0]
    y_pos = mag['posn'][:,1]
    z_pos = mag['posn'][:,2]
    rho = get_cylindrical_rho(mag)
    xBS, rBS, xIMB, rIMB = get_IMB_BS()

    fig = plt.figure(constrained_layout = True, figsize=(6, 6))
    axsPos = fig.subplots(2,2)
    colormap = plt.get_cmap("Set2")
    markers = ['v', 'o', 's', '^', 'D', 'P']
    fsize = 'large'
    bg_trans = 0.6
        
    axsPos[0,0].set_ylabel(r'$y_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[0,0].set_xlabel(r'$x_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[0,0].plot(xBS[:145], rBS[:145], color = 'black', linewidth=1) 
    axsPos[0,0].plot(xBS[:145], np.negative(np.array(rBS[:145])), color = 'black', linewidth=1)
    axsPos[0,0].plot(xIMB, rIMB, color = 'gray', linewidth=1) 
    axsPos[0,0].plot(xIMB, np.negative(np.array(rIMB)), color = 'gray', linewidth=1)
    day_hemisphere = patches.Wedge((0, 0), 1, 270, 450, facecolor='#ffcccc', zorder =5)
    night_hemisphere = patches.Wedge((0, 0), 1, 90, 270, facecolor='#ad8989', zorder =5)
    axsPos[0,0].add_patch(day_hemisphere)
    axsPos[0,0].add_patch(night_hemisphere)
        
    axsPos[0,1].set_ylabel(r'$z_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[0,1].set_xlabel(r'$x_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[0,1].plot(xBS[:145], rBS[:145], color = 'black', linewidth=1)
    axsPos[0,1].plot(xBS[:145], np.negative(np.array(rBS[:145])), color = 'black', linewidth=1)
    axsPos[0,1].plot(xIMB, rIMB, color = 'gray', linewidth=1)
    axsPos[0,1].plot(xIMB, np.negative(np.array(rIMB)), color = 'gray', linewidth=1)
    day_hemisphere = patches.Wedge((0, 0), 1, 270, 450, facecolor='#ffcccc', zorder =5)
    night_hemisphere = patches.Wedge((0, 0), 1, 90, 270, facecolor='#ad8989', zorder =5)
    axsPos[0,1].add_patch(day_hemisphere)
    axsPos[0,1].add_patch(night_hemisphere)

    axsPos[1,0].set_ylabel(r'$z_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[1,0].set_xlabel(r'$y_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    circ_BS = plt.Circle((0, 0), radius=2.63, edgecolor='black', facecolor=None, fill=False, linewidth=1)
    circ_IMB = plt.Circle((0, 0), radius=1.44, edgecolor='gray', facecolor=None, fill=False, linewidth=1)
    circ = plt.Circle((0, 0), radius=1, edgecolor=None, facecolor='#ffcccc', zorder = 5)
    axsPos[1,0].add_patch(circ)
    axsPos[1,0].add_patch(circ_IMB)
    axsPos[1,0].add_patch(circ_BS)

    for k in range(3):
        div = k//2
        rem = k%2
        axsPos[div,rem].set_xlim(-3, 3)
        axsPos[div,rem].set_ylim(-3, 3)
        axsPos[div,rem].set_aspect('equal')

    axsPos[1,1].plot(xBS[:145], rBS[:145], color = 'black', label = 'Bow Shock', linewidth=1)
    axsPos[1,1].plot(xIMB, rIMB, color = 'gray', label = 'IMB', linewidth=1)        
    axsPos[1,1].set_ylabel(r'$R_{cyl,MSO}$ $(R_{Mars})$', fontsize = fsize)
    axsPos[1,1].set_xlabel(r'$x_{MSO}$ $(R_{Mars})$', fontsize = fsize)
    day_hemisphere = patches.Wedge((0, 0), 1, 270, 450, facecolor='#ffcccc', zorder =5)
    night_hemisphere = patches.Wedge((0, 0), 1, 90, 270, facecolor='#ad8989', zorder =5)
    axsPos[1,1].add_patch(day_hemisphere)
    axsPos[1,1].add_patch(night_hemisphere)
    axsPos[1,1].set_xlim(-2, 2)
    axsPos[1,1].set_ylim(0, 4.0)
    axsPos[1,1].set_aspect('equal')

    for i in range(len(interest)):        
        vortex_time = np.datetime64(interest[i])
        d1 = np.datetime64(interest[i][:10] + 'T00:00:02')
        d2 = np.datetime64(interest[i][:10] + 'T23:59:59')
        
        vt_time = proc.find_nearest_time(mag['times'], vortex_time)
        day_start = proc.find_nearest_time(mag['times'], d1)
        day_end = proc.find_nearest_time(mag['times'], d2)
        
        xpos_slice = x_pos[day_start:day_end]
        ypos_slice = y_pos[day_start:day_end]
        zpos_slice = z_pos[day_start:day_end]
    
        hide_xy = zpos_slice < 0
        hide_xz = ypos_slice < 0
        hide_yz = xpos_slice < 0
    
        x_behind_xy = np.copy(xpos_slice)
        x_front_xy = np.copy(xpos_slice)
        x_front_xy[hide_xy] = np.nan
        x_behind_xy[np.invert(hide_xy)] = np.nan
        y_behind_xy = np.copy(ypos_slice)
        y_front_xy = np.copy(ypos_slice)
        y_front_xy[hide_xy] = np.nan
        y_behind_xy[np.invert(hide_xy)] = np.nan

        x_behind_xz = np.copy(xpos_slice)
        x_front_xz = np.copy(xpos_slice)
        x_front_xz[hide_xz] = np.nan
        x_behind_xz[np.invert(hide_xz)] = np.nan
        z_behind_xz = np.copy(zpos_slice)
        z_front_xz = np.copy(zpos_slice)
        z_front_xz[hide_xz] = np.nan
        z_behind_xz[np.invert(hide_xz)] = np.nan

        y_behind_yz = np.copy(ypos_slice)
        y_front_yz = np.copy(ypos_slice)
        y_front_yz[hide_yz] = np.nan
        y_behind_yz[np.invert(hide_yz)] = np.nan
        z_behind_yz = np.copy(zpos_slice)
        z_front_yz = np.copy(zpos_slice)
        z_front_yz[hide_yz] = np.nan
        z_behind_yz[np.invert(hide_yz)] = np.nan
        
        axsPos[0,0].plot(x_behind_xy, y_behind_xy, linestyle = 'dotted', 
                         color = colormap(i), alpha = bg_trans, zorder = 1) 
        axsPos[0,0].plot(x_front_xy, y_front_xy, linestyle = 'dotted', 
                         color = colormap(i), alpha = bg_trans, zorder = 10)
        axsPos[0,0].scatter(x_pos[vt_time], y_pos[vt_time], color=colormap(i), 
                            marker=markers[i], edgecolors='black', s=100, zorder = 12)

        axsPos[0,1].plot(x_behind_xz, z_behind_xz, linestyle = 'dotted', 
                         color = colormap(i), alpha = bg_trans, zorder = 1) 
        axsPos[0,1].plot(x_front_xz, z_front_xz, linestyle = 'dotted', 
                         color = colormap(i), alpha = bg_trans, zorder = 10)
        axsPos[0,1].scatter(x_pos[vt_time], z_pos[vt_time], color=colormap(i), 
                            marker=markers[i], edgecolors='black', s=100, zorder = 12)

        axsPos[1,0].plot(y_behind_yz, z_behind_yz, linestyle = 'dotted', 
                         color = colormap(i), alpha = bg_trans, zorder = 1) 
        axsPos[1,0].plot(y_front_yz, z_front_yz, linestyle = 'dotted', 
                         color = colormap(i), alpha = bg_trans, zorder = 10)
        axsPos[1,0].scatter(y_pos[vt_time], z_pos[vt_time], color=colormap(i), 
                            marker=markers[i], edgecolors='black', s=100, zorder = 12)

        axsPos[1,1].plot(x_pos[day_start:day_end], rho[day_start:day_end], 
                         linestyle = 'dotted', color = colormap(i), alpha = bg_trans)
        axsPos[1,1].scatter(x_pos[vt_time], rho[vt_time], color=colormap(i), 
                            marker=markers[i], edgecolors='black', s=100, zorder = 12, label = interest[i][:10])
        
    lgd = fig.legend(bbox_to_anchor=(0.5, -0.05), ncols=4, loc='center', fontsize='large') 
    if savefile != None:
        fig.savefig(savefile, format='pdf', dpi=1200, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
    plt.show()