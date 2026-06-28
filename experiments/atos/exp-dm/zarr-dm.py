print("Testing Zarr read performance with xarray and DataLoader")

import xarray as xr 
import torch
from data.meps.splits import BlockWindowSplitter
from data.meps import ZarrWindowDataModule
import time 

print("Imports successful")

zarr_path = "/ec/res4/hpcperm/smcd/data/aifs-meps-2.5km-2020-2023-1h-v2_SMHI_subdomain_reduced.zarr" # Atos 
#zarr_path = "/mnt/tier1/project/p200177/u101329/datasets/aifs-meps-2.5km-2020-2023-1h-v2_SMHI_subdomain_reduced.zarr" # Meluxina hpc 
zarr_path = "/ec/res4/hpcperm/smcd/data/aifs-meps-2.5km-2020-2023-1h-v2_SMHI_subdomain_reduced_time_7_chunked.zarr" # Atos 

# Usage:
wsplit = BlockWindowSplitter(train=1, val=1, test=1, skip=1, window_size=7)
print("Splitter initialized")
dm = ZarrWindowDataModule(dataset_path = zarr_path, variables=['10u','10v','2t','10si'], num_workers=1, splitter = wsplit, 
                          pin_memory=True, prefetch_factor=2, batch_size=8, persistent_workers = True)
print("module initialized, setting up datasets...")
dm.setup("fit")     # build train/val/test datasets
print("Datasets set up, accessing train_ds...")
train_ds = dm.train_ds
print(f"Train dataset length: {len(train_ds)}")
# for i in range(0,10):
#     print(train_ds[i].shape)

print("Iteration with DataLoader:")

# train_loader = dm.train_dataloader()
# batch = next(iter(train_loader))


print("simple read" )

ds = xr.open_zarr(zarr_path, consolidated=True, zarr_format=2)

# Get the specific DataArray you want to measure
da = ds["data"]  # Access by variable name

for i in range(10):
    
    start_tick = time.time()
    val = train_ds[i] # Access the i-th time step as
    print(val.shape)
    end_tick = time.time()
    print(f"Index {i} | Shape: {val.shape} | Time: {(end_tick - start_tick) * 1000:.2f} ms")

 
print("Testing DataLoader performance with num_workers=1"   )
 
loader = dm.train_dataloader()
counter = 0
# This is the most efficient way to loop manually
for batch_idx, batch in enumerate(loader):
    # The loader is already working on batch_idx + 1 in the background
    # because of num_workers and prefetch_factor.
    
    # Process your batch
    print(f"Batch {batch_idx} shape: {batch.shape}")
    
    if batch_idx > 5: break
    counter += 1
    if counter % 10 == 0:
        exit()
print("finished")