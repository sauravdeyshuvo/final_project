
import numpy as np
from scipy.ndimage import gaussian_filter
import pyshtools as pysh

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
    data_large_band_2d = gaussian_filter(data2d, sigma=10)
    data_medium_band2d = gaussian_filter(data2d, sigma=5) - data_large_band_2d
    data_small_band2d = data2d - gaussian_filter(data2d, sigma=5)
    
    return data_large_band_2d, data_medium_band2d, data_small_band2d

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
    
    large_scale_variance = np.var(large_band_ensemble, axis=0)
    medium_scale_variance = np.var(medium_band_ensemble, axis=0)
    small_scale_variance = np.var(small_band_ensemble, axis=0)
    
    return large_scale_variance, medium_scale_variance, small_scale_variance
    