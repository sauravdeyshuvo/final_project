# Basics 
# Spherical harmonics problem 

# Task 1: What do those spherical harmonics look like?

import numpy as np
import netCDF4 as nc
import pyshtools as pysh
import matplotlib.pyplot as plt

# Part b: Load the 5D geopotential height array
file_path = '/fs/ess/PAS2856/SPEEDY_ensemble_data/reference_ens/201101010000.nc'
try:
    dataset = nc.Dataset(file_path)
    print("File loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}. Please check the file path.")
    raise

geopotential_height = dataset.variables['phi'][:]  # Assuming 'phi' is the variable name for geopotential height

# Part c: Subset the geopotential array to member 0, model level 0, and time 0
subset_array = geopotential_height[0, 0, 0, :, :]
print("Subset array shape (should be 48x96):", subset_array.shape)

# Part d: Remove every other longitude
subset_array = subset_array[:, ::2]
print("Array shape after removing every other longitude (should be 48x48):", subset_array.shape)

# Ensure the array is contiguous and of type float64 (double precision)
subset_array = np.ascontiguousarray(subset_array, dtype=np.float64)

# Part e: Decompose the 2D geopotential into spherical harmonic components
geopot_coeffs = pysh.expand.SHExpandDH(subset_array)
print("Spherical harmonic decomposition completed.")

# Part f: Generate filtered plots
def plot_filtered_spherical_harmonics(coeffs, lmin, lmax, title, ax):
    coeffs_filtered = coeffs.copy()
    coeffs_filtered[:, lmax:, :] = 0
    if lmin > 0:
        coeffs_filtered[:, :lmin, :] = 0
    grid_filtered = pysh.expand.MakeGridDH(coeffs_filtered, sampling=2)
    ax.imshow(grid_filtered, extent=(0, 360, -90, 90))
    ax.set(xlabel='Longitude', ylabel='Latitude', title=title,
           yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45))
    print(f"Plot created: {title}")

fig, (row1, row2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the sum of the 8 largest scales
plot_filtered_spherical_harmonics(geopot_coeffs, 0, 8, 'l = 0 - 7', row1)

# Plot the sum of the 9th to 19th largest scales
plot_filtered_spherical_harmonics(geopot_coeffs, 8, 20, 'l = 8 - 19', row2)

fig.tight_layout()
plt.savefig('sample_figure1.png', bbox_inches = 'tight', dpi = 300)

# Part g: Adapt the code to make the following three subplots
fig, (row1, row2, row3) = plt.subplots(3, 1, figsize=(10, 12))

# (i) The sum of the 16 largest scales
plot_filtered_spherical_harmonics(geopot_coeffs, 0, 16, 'l = 0 - 15', row1)

# (ii) The sum of the 16th to 32nd largest scales
plot_filtered_spherical_harmonics(geopot_coeffs, 16, 33, 'l = 16 - 32', row2)

# (iii) The sum of the remaining scales
plot_filtered_spherical_harmonics(geopot_coeffs, 33, geopot_coeffs.shape[1], 'l > 32', row3)

fig.tight_layout()
plt.savefig('sample_figure2.png', bbox_inches = 'tight', dpi = 300)

# Part h: Test the code from part g
# Add the three arrays produced by part f and plot out the result
coeffs_large = geopot_coeffs.copy()
coeffs_large[:, 16:, :] = 0  # l = 0 - 15

coeffs_medium = geopot_coeffs.copy()
coeffs_medium[:, :16, :] = 0  # l >= 16
coeffs_medium[:, 33:, :] = 0  # l = 16 - 32

coeffs_small = geopot_coeffs.copy()
coeffs_small[:, :33, :] = 0  # l > 32

geopot_large = pysh.expand.MakeGridDH(coeffs_large, sampling=2)
geopot_medium = pysh.expand.MakeGridDH(coeffs_medium, sampling=2)
geopot_small = pysh.expand.MakeGridDH(coeffs_small, sampling=2)

# Sum the three filtered arrays
geopot_sum = geopot_large + geopot_medium + geopot_small

# Plot the sum and the original (48x48) geopotential array from part c
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.imshow(geopot_sum, extent=(0, 360, -90, 90))
ax1.set(xlabel='Longitude', ylabel='Latitude', title='Sum of filtered arrays',
        yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45))

ax2.imshow(subset_array, extent=(0, 360, -90, 90))
ax2.set(xlabel='Longitude', ylabel='Latitude', title='Original array',
        yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45))

fig.tight_layout()
plt.savefig('sample_figure3.png', bbox_inches = 'tight', dpi = 300)
print("Plots for Part h completed.")

# Task 2: Ensemble Variances at the Three Scale Bands 
import numpy as np
import netCDF4 as nc
from scipy.ndimage import gaussian_filter
import pyshtools as pysh
import matplotlib.pyplot as plt

# Part a: Function to decompose a 2D array into three spatial scale bands
def three_scale_decomposition(data2d):
    """
    Decompose a 2D array into three spatial scale bands.
    
    Parameters:
    data2d (numpy.ndarray): Input 2D array.
    
    Returns:
    numpy.ndarray: Large scale band.
    numpy.ndarray: Medium scale band.
    numpy.ndarray: Small scale band.
    """
    # Apply Gaussian filters to decompose the data into different scales
    data_large_band_2d = gaussian_filter(data2d, sigma=10)
    data_medium_band2d = gaussian_filter(data2d, sigma=5) - data_large_band_2d
    data_small_band2d = data2d - gaussian_filter(data2d, sigma=5)
    
    return data_large_band_2d, data_medium_band2d, data_small_band2d

# Part b: Test the three_scale_decomposition function
# Assuming geopotential_array is already loaded with shape (48, 48)
geopotential_array = np.random.rand(48, 48)  # Placeholder for the actual data

# Apply the decomposition function
large_band, medium_band, small_band = three_scale_decomposition(geopotential_array)

# Check if the output matches the expected results from Task 1g
# This would involve comparing with precomputed arrays from Task 1g

# Part c: Load the geopotential arrays from the reference ensemble on 1st March 2011
# Placeholder array for demonstration (the actual array should be loaded from a file)
reference_ensemble = np.random.rand(1000, 8, 48, 48)  # Placeholder for the actual data

# Remove the time dimension and every other longitude location
reference_ensemble = reference_ensemble[:, :, :, ::2]

# Part d and e: Decompose and compute ensemble variances
def compute_ensemble_3scale_variance(ensemble_data):
    """
    Compute ensemble variances for three scale bands.
    
    Parameters:
    ensemble_data (numpy.ndarray): 4D array of ensemble data (shape: (1000, 8, 48, 48)).
    
    Returns:
    numpy.ndarray: Large scale band variance.
    numpy.ndarray: Medium scale band variance.
    numpy.ndarray: Small scale band variance.
    """
    large_band_ensemble = np.empty_like(ensemble_data)
    medium_band_ensemble = np.empty_like(ensemble_data)
    small_band_ensemble = np.empty_like(ensemble_data)
    
    for i in range(ensemble_data.shape[0]):
        for j in range(ensemble_data.shape[1]):
            large_band_ensemble[i, j], medium_band_ensemble[i, j], small_band_ensemble[i, j] = three_scale_decomposition(ensemble_data[i, j])
    
    # Compute variances for each scale band
    large_scale_variance = np.var(large_band_ensemble, axis=0)
    medium_scale_variance = np.var(medium_band_ensemble, axis=0)
    small_scale_variance = np.var(small_band_ensemble, axis=0)
    
    return large_scale_variance, medium_scale_variance, small_scale_variance

# Perform the variance computation
large_scale_variance, medium_scale_variance, small_scale_variance = compute_ensemble_3scale_variance(reference_ensemble)

# Part g: Documenting functions in pysh_ens_variance.py
# Copy all functions into the file pysh_ens_variance.py and add comments
with open('pysh_ens_variance.py', 'w') as f:
    f.write("""
import numpy as np
from scipy.ndimage import gaussian_filter
import pyshtools as pysh

def three_scale_decomposition(data2d):
    \"""
    Decompose a 2D array into three spatial scale bands.
    
    Parameters:
    data2d (numpy.ndarray): Input 2D array.
    
    Returns:
    numpy.ndarray: Large scale band.
    numpy.ndarray: Medium scale band.
    numpy.ndarray: Small scale band.
    \"""
    data_large_band_2d = gaussian_filter(data2d, sigma=10)
    data_medium_band2d = gaussian_filter(data2d, sigma=5) - data_large_band_2d
    data_small_band2d = data2d - gaussian_filter(data2d, sigma=5)
    
    return data_large_band_2d, data_medium_band2d, data_small_band2d

def compute_ensemble_3scale_variance(ensemble_data):
    \"""
    Compute ensemble variances for three scale bands.
    
    Parameters:
    ensemble_data (numpy.ndarray): 4D array of ensemble data (shape: (1000, 8, 48, 48)).
    
    Returns:
    numpy.ndarray: Large scale band variance.
    numpy.ndarray: Medium scale band variance.
    numpy.ndarray: Small scale band variance.
    \"""
    large_band_ensemble = np.empty_like(ensemble_data)
    medium_band_ensemble = np.empty_like(ensemble_data)
    small_band_ensemble = np.empty_like(ensemble_data)
    
    for i in range(ensemble_data.shape[0]):
        for j in range(ensemble_data.shape[1]):
            large_band_ensemble[i, j], medium_band_ensemble[i, j], small_band_ensemble[i, j] = three_scale_decomposition(ensemble_data[i, j])
    
    large_scale_variance = np.var(large_band_ensemble, axis=0)
    medium_scale_variance = np.var(medium_band_ensemble, axis=0)
    small_scale_variance = np.var(small_band_ensemble, axis=0)
    
    return large_scale_variance, medium_scale_variance, small_scale_variance
    """)
# Task 3: Flexible Python script for scale decomposition of SPEEDY ensemble 
import sys
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from pysh_ens_variance import compute_ensemble_3scale_variance

def get_date_from_days(days_since_20110101):
    """
    Convert days since 2011-01-01 to a formatted date string.
    
    Parameters:
    days_since_20110101 (int): Number of days since 2011-01-01.
    
    Returns:
    str: Formatted date string (YYYYMMDDHH).
    """
    start_date = datetime(2011, 1, 1)
    target_date = start_date + timedelta(days=days_since_20110101)
    return target_date.strftime('%Y%m%d%H')

def compute_theoretical_pressure(sigma):
    """
    Compute theoretical pressure from sigma values.
    
    Parameters:
    sigma (numpy.ndarray): Array of sigma values.
    
    Returns:
    numpy.ndarray: Array of theoretical pressure values.
    """
    return sigma * 1000

def main():
    if len(sys.argv) != 5:
        print("Usage: python compute_ens_variance.py <days_since_20110101> <ensemble_name> <variable_name> <output_directory>")
        sys.exit(1)

    days_since_20110101 = int(sys.argv[1])
    ensemble_name = sys.argv[2]
    variable_name = sys.argv[3]
    output_directory = sys.argv[4]

    date_str = get_date_from_days(days_since_20110101)
    file_name = f"{variable_name}_{ensemble_name}_{date_str}_variance.pkl"
    output_path = os.path.join(output_directory, file_name)

    # Assuming the data is loaded from some source, here we use dummy data
    data = np.random.rand(8, 48, 48)  # Replace with actual data loading

    # Compute variances for the three scale bands
    large_scale_variance, medium_scale_variance, small_scale_variance = compute_ensemble_3scale_variance(data)

    # Placeholder sigma values (replace with actual values)
    sigma = np.linspace(0.1, 1.0, 8)  # Replace with actual sigma values
    theoretical_pressure = compute_theoretical_pressure(sigma)

    result = {
        "date": date_str,
        "vname": variable_name,
        "large scale average variance": large_scale_variance,
        "medium scale average variance": medium_scale_variance,
        "small scale average variance": small_scale_variance,
        "theoretical pressure": theoretical_pressure
    }

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"Variance data saved to {output_path}")

if __name__ == "__main__":
    main()

# Task 4: Visualizing Patterns in Ensemble Variances 
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_pickle_files(start_date, end_date, interval, variable_name, ensemble_type, directory):
    """
    Load pickle files between the specified dates for the given variable and ensemble type.
    
    Parameters:
    start_date (str): Start date in YYYYMMDD format.
    end_date (str): End date in YYYYMMDD format.
    interval (int): Interval in days between pickle files.
    variable_name (str): Name of the variable.
    ensemble_type (str): Type of the ensemble (reference or perturbed).
    directory (str): Path to the directory containing the pickle files.
    
    Returns:
    list: List of loaded pickle data.
    """
    data = []
    current_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")
    
    while current_date <= end_date:
        file_name = f"{variable_name}_{ensemble_type}_{current_date.strftime('%Y%m%d')}_variance.pkl"
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data.append(pickle.load(f))
        current_date += timedelta(days=interval)
    
    return data

def plot_variance_diagrams(data, variable_name, ensemble_type):
    """
    Generate level-time and/or latitude-time diagrams to show how ensemble variances grow over time.
    
    Parameters:
    data (list): List of loaded pickle data.
    variable_name (str): Name of the variable.
    ensemble_type (str): Type of the ensemble (reference or perturbed).
    """
    times = [entry['date'] for entry in data]
    large_variances = [entry['large scale average variance'] for entry in data]
    medium_variances = [entry['medium scale average variance'] for entry in data]
    small_variances = [entry['small scale average variance'] for entry in data]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    for ax, variances, scale in zip(axes, [large_variances, medium_variances, small_variances], ['Large', 'Medium', 'Small']):
        var_matrix = np.array(variances)
        im = ax.imshow(var_matrix.T, aspect='auto', origin='lower', extent=[0, len(times)-1, 0, var_matrix.shape[1]])
        ax.set_xticks(np.arange(len(times)))
        ax.set_xticklabels(times, rotation=45)
        ax.set_title(f'{scale} Scale Variance over Time for {variable_name} ({ensemble_type})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Model Level or Latitude')
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('sample_figure4.png', bbox_inches = 'tight', dpi = 300)

def plot_normalized_variance(data_perturbed, data_reference, variable_name):
    """
    Generate plots of normalized variances.
    
    Parameters:
    data_perturbed (list): List of loaded pickle data for the perturbed ensemble.
    data_reference (list): List of loaded pickle data for the reference ensemble.
    variable_name (str): Name of the variable.
    """
    times = [entry['date'] for entry in data_perturbed]
    norm_variances_large = []
    norm_variances_medium = []
    norm_variances_small = []

    for perturbed, reference in zip(data_perturbed, data_reference):
        norm_variances_large.append(perturbed['large scale average variance'] / reference['large scale average variance'])
        norm_variances_medium.append(perturbed['medium scale average variance'] / reference['medium scale average variance'])
        norm_variances_small.append(perturbed['small scale average variance'] / reference['small scale average variance'])
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    for ax, variances, scale in zip(axes, [norm_variances_large, norm_variances_medium, norm_variances_small], ['Large', 'Medium', 'Small']):
        var_matrix = np.array(variances)
        im = ax.imshow(var_matrix.T, aspect='auto', origin='lower', extent=[0, len(times)-1, 0, var_matrix.shape[1]])
        ax.set_xticks(np.arange(len(times)))
        ax.set_xticklabels(times, rotation=45)
        ax.set_title(f'Normalized {scale} Scale Variance over Time for {variable_name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Model Level or Latitude')
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('sample_figure5.png', bbox_inches = 'tight', dpi = 300)

def main():
    if len(sys.argv) != 7:
        print("Usage: python examine_variance_behavior.py <start_date> <end_date> <interval_days> <variable_name> <ensemble_type> <directory>")
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]
    interval_days = int(sys.argv[3])
    variable_name = sys.argv[4]
    ensemble_type = sys.argv[5]
    directory = sys.argv[6]

    data = load_pickle_files(start_date, end_date, interval_days, variable_name, ensemble_type, directory)
    
    if not data:
        print("No data loaded. Please check the input parameters and the existence of pickle files.")
        sys.exit(1)
    
    plot_variance_diagrams(data, variable_name, ensemble_type)
    
    if ensemble_type == 'perturbed':
        reference_data = load_pickle_files(start_date, end_date, interval_days, variable_name, 'reference', directory)
        if not reference_data:
            print("No reference data loaded. Please check the existence of reference pickle files.")
            sys.exit(1)
        
        plot_normalized_variance(data, reference_data, variable_name)

if __name__ == "__main__":
    main()
