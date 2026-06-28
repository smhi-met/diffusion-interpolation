import xarray as xr
 
 
source_zarr_path = "/ec/res4/hpcperm/smcd/data/aifs-meps-2.5km-2020-2023-1h-v2_SMHI_subdomain_reduced.zarr"
target_zarr_path = "/ec/res4/hpcperm/smcd/data/aifs-meps-2.5km-2020-2023-1h-v2_SMHI_subdomain_reduced_time_7_chunked.zarr"

ds = xr.open_zarr(source_zarr_path, zarr_format=2)

# Ensure we have a clean multiple of 7
count = len(ds.time)
max_time = (count // 7) * 7
subset = ds.isel(time=slice(0, max_time))

# Optimal Chunks for 2D Conv training
# We want 7 steps in ONE chunk so one disk read = one window.
new_chunks = {
    'time': 7,
    'variable': -1,  # Keep variables separate for flexible loading
    'ensemble': 1,
    'cell': 65536
}

ds_rechunked = subset.chunk(new_chunks)
ds_rechunked.data.encoding = {}
# Use 'w' to overwrite or 'w-' to fail if it exists (safer)
print("Starting rechunking...")
ds_rechunked.to_zarr(
    target_zarr_path, 
    consolidated=True, 
    mode='w', 
    zarr_format=2,
    align_chunks=True 
)
print("Rechunking complete!")