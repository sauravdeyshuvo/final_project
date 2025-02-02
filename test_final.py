import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

# File path (based on your provided location)
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

# Select variables to decompose (e.g., u, v, w, ps)
variables = ["u", "v", "w", "ps"]
data_vars = {var: ds[var] for var in variables if var in ds}

if not data_vars:
    raise ValueError("No matching variables (u, v, w, ps) found in the dataset.")

# Function to decompose data into scales
def decompose_scales(data, lat_dim, lon_dim):
    """
    Decompose data into large, medium, and small scales using Gaussian filters.
    """
    # Large-scale (smooth with high sigma)
    large_scale = gaussian_filter(data, sigma=10, mode="nearest")

    # Medium-scale (subtract large-scale, then smooth with lower sigma)
    medium_scale_intermediate = data - large_scale
    medium_scale = gaussian_filter(medium_scale_intermediate, sigma=5, mode="nearest")

    # Small-scale (subtract large and medium from the original)
    small_scale = data - (large_scale + medium_scale)

    return large_scale, medium_scale, small_scale

# Perform decomposition for each variable and save to separate files
large_scale_vars = {}
medium_scale_vars = {}
small_scale_vars = {}

for var_name, var_data in data_vars.items():
    print(f"Processing variable: {var_name}")

    # Convert to NumPy array for processing
    data = var_data.values
    dims = var_data.dims

    # Decompose scales
    large, medium, small = decompose_scales(data, lat_dim=lat_var, lon_dim=lon_var)

    # Convert results back to xarray.DataArray
    large_scale_vars[var_name] = xr.DataArray(large, dims=dims, coords=var_data.coords)
    medium_scale_vars[var_name] = xr.DataArray(medium, dims=dims, coords=var_data.coords)
    small_scale_vars[var_name] = xr.DataArray(small, dims=dims, coords=var_data.coords)

# Save decomposed data to separate netCDF files
large_ds = xr.Dataset(large_scale_vars, coords=ds.coords)
medium_ds = xr.Dataset(medium_scale_vars, coords=ds.coords)
small_ds = xr.Dataset(small_scale_vars, coords=ds.coords)

output_dir = "/fs/ess/PAS2856/SPEEDY_ensemble_data/decomposed_scales/"
large_ds.to_netcdf(f"{output_dir}large_scale_output.nc")
medium_ds.to_netcdf(f"{output_dir}medium_scale_output.nc")
small_ds.to_netcdf(f"{output_dir}small_scale_output.nc")

print("Decomposition complete. Files saved to output directory.")
