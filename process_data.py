import numpy as np

def find_nearest_time(arr, time):
    """
    Finds the index of the element in the given array that is closest to the
    given time.

    Parameters:
    arr (np.ndarray): An array of np.datetime64 objects.
    time (np.datetime64): The target time.

    Returns:
    int: The index of the element in the array that is closest to the target time.
    """
    i = 0
    while arr[i] < time:
        i += 1
    if i > 0:
        if (arr[i] - time) > (arr[i-1] - time):
            return i-1
    return i

def compare_time_arrays(arr1, arr2):
    """
    Compare two numpy datetime64 arrays and find instances in arr2 closest to arr1.

    Parameters:
    arr1 (np.ndarray): The first datetime64 array.
    arr2 (np.ndarray): The second datetime64 array.

    Returns:
    tuple: A tuple containing three lists:
        - result: Instances in arr2 that are closest to arr1.
        - indices_arr1: Indices of the closest instances in arr1.
        - indices_arr2: Indices of the closest instances in arr2.
    """
    i = 0
    result = []
    indices_arr1 = []
    indices_arr2 = []
    
    for j in range(len(arr2)):
        element = arr2[j]
        
        if i >= len(arr1):
            return result, indices_arr1, indices_arr2
                    
        while arr1[i] < element:
            i += 1
            if i >= len(arr1):
                return result, indices_arr1, indices_arr2
            
        if arr1[i] <= element:
            larger = element
            smaller = arr1[i]
        else:
            larger = arr1[i]
            smaller = element
        if (larger - smaller) > np.timedelta64(2, 's'):
            continue

        result.append(arr2[j])
        indices_arr1.append(i)
        indices_arr2.append(j)
        
    return result, indices_arr1, indices_arr2

def compute_pressure(mag, swia):
    """
    Compute the total pressure using magnetic field and plasma data, where
    total pressure = magnetic pressure + plasma pressure.

    Parameters:
    mag (dict): MAG dictionary containing magnetic field data.
    swia (dict): SWIA dictionary containing plasma data.

    Returns:
    dict: A dictionary containing the following keys:
        - times: An array of size n with the time (np.datetime64) of each point
                in the time series at the matched precision between MAG and SWIA.
        - plasma_pressure: An array of size n with the plasma pressure (in nPa)
                            at the precision of SWIA.
        - magnetic_pressure: An array of size n with the magnetic pressure (in nPa)
                            at the precision of MAG.
        - total_pressure: An array of size n with the total pressure (in nPa) at
                            the matched precision between MAG and SWIA.
    """
    mag_time = mag['times']
    btot = np.sqrt(np.square(mag['B'][:,0]) + np.square(mag['B'][:,1]) + 
                   np.square(mag['B'][:,2]))
    swia_time = swia['times']
    temp = swia['temp']
    density = swia['density']

    temp = temp * 1.160451812 * (10**4) #eV to K
    temp_tot = np.sqrt(np.square(temp[:,0]) + np.square(temp[:,1]) + 
                       np.square(temp[:,2]))

    density_m = density * (10**6)
    k = 1.380649 * (10 ** (-23)) 
    plasma_pressure = density_m * k * temp_tot * (10**9) # to turn into nPa

    btot_squared = np.square(btot * (10**(-9)))
    magnetic_pressure = btot_squared/(2 * 1.25663706 * (10**(-6)))
    magnetic_pressure = magnetic_pressure * (10**9)

    # Match resolution of MAG and SWIA dataset
    pressure_time, indices_mag, indices_swia = compare_time_arrays(
        mag_time, swia_time)
    magp = magnetic_pressure[indices_mag]
    plasmap = plasma_pressure[indices_swia]
    total_pressure = magp + plasmap

    return {'times': pressure_time, 'plasma_pressure': plasma_pressure, 
            'magnetic_pressure': magnetic_pressure, 'total_pressure': total_pressure}

def static_spectrogram(static, ind1, ind2):
    """
    Separates eflux values for the STATIC energy spectrogram into 3 different ion
    species: H+, O+, and O2+ ions.

    Args:
        static (dict): STATIC dictionary containing plasma data.
        ind1 (int): The starting index for slicing the data.
        ind2 (int): The ending index for slicing the data.

    Returns:
        tuple: A tuple containing the spectrogram data for hydrogen, oxygen, and
            O2, along with the energy grid for relevant energy bands.
    """
    energy = static['energy']
    sweep_index = static['sweep_index'][ind1:ind2]
    eflux = static['flux'][ind1:ind2]
    mas = static['mass'][ind1:ind2]

    energy_grid = []
    unique_sweeps = list(np.unique(sweep_index))
    for i in unique_sweeps:
        energy_grid.append(energy[0,:,i]) #0th sweep table
    energy_grid = np.array(energy_grid).flatten()
    grid_no = np.size(energy_grid)

    hydrogen = np.zeros((len(eflux), grid_no)) # len(eflux) is time slices
    oxygen = np.zeros((len(eflux), grid_no))
    o2 = np.zeros((len(eflux), grid_no))

    idx_h = (mas>=0)*(mas<=1.55)
    hydrogen_index = np.argwhere(idx_h)
    for x in hydrogen_index:
        k = x[0] #time
        i = x[1] #mass
        j = x[2] #energy
        #position in the unique sweep index array
        idx_sweep = unique_sweeps.index(sweep_index[k]) 
        index = (idx_sweep * 32) + j
        hydrogen[k][index] += eflux[k, i, j]    
    
    idx_o = (mas>=14)*(mas<=20)
    oxygen_index = np.argwhere(idx_o)
    for x in oxygen_index:
        k = x[0] #time
        i = x[1] #mass
        j = x[2] #energy
        idx_sweep = unique_sweeps.index(sweep_index[k]) 
        index = (idx_sweep * 32) + j
        oxygen[k][index] += eflux[k, i, j]    

    idx_o2 = (mas>=24)*(mas<=40)
    o2_index = np.argwhere(idx_o2)
    for x in o2_index:
        k = x[0] #time
        i = x[1] #mass
        j = x[2] #energy
        idx_sweep = unique_sweeps.index(sweep_index[k]) 
        index = (idx_sweep * 32) + j
        o2[k][index] += eflux[np.array(k), np.array(i), np.array(j)]  
    
    for k in range(len(eflux)):
        for j in range(grid_no):
            idx_sweep = unique_sweeps.index(sweep_index[k])
            if j not in range(idx_sweep * 32, (idx_sweep+1)*32):
                hydrogen[k][j] = np.nan
                oxygen[k][j] = np.nan
                o2[k][j] = np.nan
    
    return hydrogen, oxygen, o2, energy_grid