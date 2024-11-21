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