import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

# File path 
input_file = "/fs/ess/PAS2856/SPEEDY_ensemble_data/reference_ens/201101010000.nc"

# Open the netCDF file
ds = xr.open_dataset(input_file)

# Check for latitude and longitude variables
lat_var = None
lon_var = None
for var in ds.variables:
    if "lat" in var.lower():
        lat_var = var
    if "lon" in var.lower():
        lon_var = var

if not lat_var or not lon_var:
    raise ValueError("Latitude and/or Longitude variables could not be identified in the dataset.")

latitude = ds[lat_var]
longitude = ds[lon_var]

# Variables to decompose ( u, v, w, ps)
variables = ["u", "v", "w", "ps"]
data_vars = {var: ds[var] for var in variables if var in ds}

if not data_vars:
    raise ValueError("No matching variables (u, v, w, ps) found in the dataset.")

# Convert to float function
def convert_to_float(array):
    try:
        return np.array(array, dtype=np.float64)
    except Exception as e:
        print(f"Error converting array to float: {e}")
        return np.zeros_like(array, dtype=np.float64)

# Function to decompose data into scales using variance
def decompose_scales(data_vars):
    large_scale_vars = {}
    medium_scale_vars = {}
    small_scale_vars = {}

    for var_name, var_data in data_vars.items():
        print(f"Processing variable: {var_name}")

        # Convert to NumPy array for processing
        data_array = convert_to_float(var_data.values)

        # Decompose data into large, medium, and small scales
        large_scale = np.mean(data_array, axis=(2, 3))  # Large scale computation (mean over specified dimensions)
        medium_scale_intermediate = data_array - large_scale  # Intermediate step
        medium_scale = np.var(medium_scale_intermediate, axis=(2, 3))  # Medium scale computation (variance over specified dimensions)
        small_scale = data_array - (large_scale + medium_scale)  # Small scale computation (residual)

        # Convert results back to xarray.DataArray
        dims = var_data.dims
        coords = var_data.coords
        large_scale_vars[var_name] = xr.DataArray(large_scale, dims=dims, coords=coords)
        medium_scale_vars[var_name] = xr.DataArray(medium_scale, dims=dims, coords=coords)
        small_scale_vars[var_name] = xr.DataArray(small_scale, dims=dims, coords=coords)

    return large_scale_vars, medium_scale_vars, small_scale_vars

# Perform decomposition
large_scale_vars, medium_scale_vars, small_scale_vars = decompose_scales(data_vars)

# Save decomposed data to separate netCDF files
output_dir = "/fs/ess/PAS2856/SPEEDY_ensemble_data/decomposed_scales/"
large_ds = xr.Dataset(large_scale_vars, coords=ds.coords)
medium_ds = xr.Dataset(medium_scale_vars, coords=ds.coords)
small_ds = xr.Dataset(small_scale_vars, coords=ds.coords)

large_ds.to_netcdf(f"{output_dir}large_scale_output.nc")
medium_ds.to_netcdf(f"{output_dir}medium_scale_output.nc")
small_ds.to_netcdf(f"{output_dir}small_scale_output.nc")

print("Decomposition complete. Files saved to output directory.")
