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
    plt.savefig('sample_figure4.png', bbox_inches='tight', dpi=300)

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
    plt.savefig('sample_figure5.png', bbox_inches='tight', dpi=300)

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

    # Updated the directory path 
    directory = '/fs/scratch/PAS2856/AS4194_Project/PatelShuvo'
    
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
