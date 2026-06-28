import torch
import xarray as xr
import numpy as np
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

class LatentSpaceSampler(Callback):
    def __init__(
        self, 
        output_path: str, 
        num_samples: int = 10, 
        log_interval: int = 5,
        variables: list[str] = None
    ):
        super().__init__()
        self.output_path = Path(output_path)
        self.num_samples = num_samples
        self.log_interval = log_interval
        self.variables = variables  # List of names like ['2t', '10u']
        
        self.var_indices = None
        self.samples = None
        self.timestamps = None
        self.ready = False

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        # 1. Resolve Variable Indices via Manifest
        
        manifest = trainer.datamodule.manifest
        if self.variables:
            self.var_indices = manifest.find_variables_indices(self.variables)
        else:
            raise ValueError("variables is not set")

        # 2. Capture Deterministic Samples from Validation Dataset
        val_ds = trainer.datamodule.val_dataset
        collected_x = []
        collected_ts = []
        
        indices = torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(42))[:self.num_samples]

        for i in indices:
            item = val_ds[i]
            collected_x.append(item["x"].unsqueeze(0))
            collected_ts.append(item["timestamps"])
        
        # Shape: (B, T, V, E, H, W)
        self.samples = torch.cat(collected_x, dim=0).to(pl_module.device)
        self.timestamps = np.array(collected_ts)
        self.ready = True
        
        print(f"Zarr Logger init with vars: {self.variables} (indices: {self.var_indices})")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.ready or (trainer.current_epoch % self.log_interval != 0):
            return

        pl_module.eval()
        with torch.no_grad():
            # Pass 6D tensor: (B, T, V, E, H, W)
            reconstructions = pl_module(self.samples)
        pl_module.train()

        # 3. Subset variables using resolved indices (Dimension 2 is Variable)
        orig_np = self.samples[:, :, self.var_indices].cpu().numpy()
        recon_np = reconstructions[:, :, self.var_indices].cpu().numpy()

        # 4. Construct Xarray Dataset
        # We use self.variables (the names) as the coordinate for the variable dimension
        ds = xr.Dataset(
            data_vars={
                "original": (["epoch", "sample", "time", "variable", "ensemble", "y", "x"], orig_np[None, ...]),
                "reconstruction": (["epoch", "sample", "time", "variable", "ensemble", "y", "x"], recon_np[None, ...]),
            },
            coords={
                "epoch": [trainer.current_epoch],
                "sample": np.arange(self.num_samples),
                "time_step": np.arange(orig_np.shape[1]),
                "variable": self.variables,  # Use names as coordinates
                "ensemble": np.arange(orig_np.shape[3]),
                "timestamp": (["sample", "time_step"], self.timestamps)
            }
        )

        # 5. Persistent Storage
        if not self.output_path.exists():
            ds.to_zarr(self.output_path, mode="w")
        else:
            ds.to_zarr(self.output_path, append_dim="epoch")