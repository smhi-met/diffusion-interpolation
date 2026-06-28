from __future__ import annotations
import numpy as np
import logging
from typing import Sequence, Literal

import torch
import zarr
from torch.utils.data import DataLoader, Dataset

from dssml.data.modules import DataModuleBase

from dssml.data.helpers.manifests import MepsZarrManifest

from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

 
MepsZarrSample = dict[Literal["x", "timestamps"], torch.Tensor]  # keys: 'x', 'timestamps'
 

class MepsZarrWindowDataset(Dataset):
    """
    Returns windows from root['data'] with shape:
      - (T, V, E, H, W)  
     """

    def __init__(
        self,
        dataset_path: str,
        var_indices: np.ndarray,
        starts: dict[str, np.ndarray] | None = None,
        ends: dict[str, np.ndarray] | None = None,
        reshape_to: list[int] | None = None,
    ):
        self.dataset_path =  dataset_path
        self._ends = ends
        self._starts = starts
        self.var_indices = var_indices
        self.reshape_to = reshape_to

        self._data = None  # Opened per worker process

    def __len__(self) -> int:
        return len(self._starts)

    def _ensure_open(self) -> None:
        if self._data is not None:
            return

        root = zarr.open(self.dataset_path, mode="r")
        data = root["data"]
        self._dates = root["dates"]   

        dims = data.attrs.get("_ARRAY_DIMENSIONS")
        if dims is not None and tuple(dims) != ("time", "variable", "ensemble", "cell"):
            raise ValueError(f"Unexpected dim order {dims}; expected ('time','variable','ensemble','cell').")

        self._data = data

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._ensure_open()
        start = int(self._starts[idx])
        end = int(self._ends[idx])
        timestamps = self._dates[start:end]
        # Initial slice: (time, variable, ensemble, cell)
        x = self._data[start:end, self.var_indices, :, :]

        x = torch.from_numpy(x)
         # 2. Spatial Reshape: (T, V, C) -> (T, V, H, W)
        if self.reshape_to is not None:
                lead = x.shape[:-1]
                x = x.view(*lead, *self.reshape_to)
        return {"x": x, "timestamps": timestamps}


class MEPSZarrWindowDataModule(DataModuleBase):
    def __init__(
        self,
        manifest: MepsZarrManifest,
        dl_kwargs: dict | None = None,
 
    ):
        super().__init__() 

        if not isinstance(manifest, MepsZarrManifest):
            raise TypeError(f"Expected MepsZarrManifest, got {type(manifest)}")
        
        # Extract necessary info from manifest to configure datasets and dataloaders
        self.dataset_path = manifest.dataset_path
        self.var_indices = manifest.variables_indices
        self.splits = manifest.splits
        self.reshape_to = manifest.reshape_to
        self._manifest = manifest
        self.dl_kwargs = dl_kwargs or {}
 
 
 
    @property
    def val_dataset(self) -> Dataset:
        return self._val_ds
    
    @property
    def train_dataset(self) -> Dataset:
        return self._train_ds
    
    @property
    def test_dataset(self) -> Dataset:
        return self._test_ds 


    def setup(self, stage: str | None = None) -> None:

        if stage in ("fit", None):
            self._train_ds = self._make_dataset("train")
            self._val_ds = self._make_dataset("val")
            print(f"Training samples count: {len(self._train_ds)}")
            print(f"Validation samples count: {len(self._val_ds)}")
 

        if stage in ("test", "predict", None):
            self._test_ds = self._make_dataset("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_ds, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_ds, shuffle=False, **self.dl_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_ds, shuffle=False, **self.dl_kwargs)

    def on_exception(self, exception: Exception):
        LOGGER.error(f"DataModule exception occurred: {exception}")
        raise exception
    
    #### inheriter methods ####
    @property
    def name(self) -> str:
        return "zarr_window_dm"
    
    @property
    def manifest(self) -> str:
      return self._manifest
   


    #### Helper Methods ####
    def _make_dataset(self, split_key: str) -> MepsZarrWindowDataset:
        """Helper to instantiate datasets consistently."""
        return MepsZarrWindowDataset(
            dataset_path=self.dataset_path,
            var_indices=self.var_indices,
            starts=self.splits[split_key]["starts"],
            ends=self.splits[split_key]["ends"],
            reshape_to=self.reshape_to
        )
