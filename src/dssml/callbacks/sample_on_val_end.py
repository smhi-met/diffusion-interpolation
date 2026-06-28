from __future__ import annotations
import os
import warnings
from typing import Any, Optional, Tuple, Dict

import torch
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only

class SampleOnValEndCallback(L.Callback):
    def __init__(
        self,
        every_n_epochs: int = 1,
        save_npz: bool = False,
        out_dir: Optional[str] = None,
        max_plot_channels: int = 3,
    ):
        super().__init__()
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.save_npz = bool(save_npz)
        self.out_dir = out_dir
        self.max_plot_channels = max_plot_channels
        self._cache: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]] = None

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Cache only the first batch of the epoch
        if self._cache is not None:
            return
            
        # Robust batch unpacking
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 2:
                x, y = batch[:2]
                meta = batch[2] if len(batch) > 2 else None
            else:
                return
        elif isinstance(batch, dict):
            # Fallback for dict batches
            x = batch.get('input') or batch.get('x')
            y = batch.get('target') or batch.get('y')
            meta = batch.get('meta')
            if x is None or y is None:
                return
        else:
            return

        self._cache = (x.detach().cpu(), y.detach().cpu(), meta)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            self._cache = None
            return
            
        # Only rank 0 saves the images
        if not trainer.is_global_zero:
            self._cache = None
            return
            
        if self._cache is None:
            return

        x, y, meta = self._cache
        
        # Move to device for sampling
        x = x.to(pl_module.device)
        y = y.to(pl_module.device)

        # Prepare condition and target
        # If model has specific logic for packing inputs, use it
        if hasattr(pl_module, "_pack_xy"):
            cond, target = pl_module._pack_xy(x, y) 
        else:
            cond, target = x, y

        # Run Sampling
        with torch.no_grad():
            # Uses the model's internal sampler (self.sampler from base.py)
            if hasattr(pl_module, "sampler") and pl_module.sampler is not None:
                 y_pred = pl_module.sampler.sample(pl_module, cond=cond, target_shape=target.shape, device=pl_module.device)
            elif hasattr(pl_module, "sample"):
                # Fallback to model method if sampler attribute isn't set
                y_pred = pl_module.sample(cond=cond, target_shape=target.shape, device=pl_module.device)
            else:
                print("Warning: Model has no .sample() method or .sampler attribute.")
                self._cache = None
                return

        # Save/Plot
        self._save_results(trainer, cond, y_pred, target, meta)
        
        # Clear cache
        self._cache = None

    @rank_zero_only
    def _save_results(self, trainer, cond, pred, gt, meta):
        root = self.out_dir or os.path.join(trainer.default_root_dir, "val_samples")
        os.makedirs(root, exist_ok=True)
        
        stem = f"epoch{trainer.current_epoch:04d}"
        
        if self.save_npz:
            try:
                import numpy as np
                np.savez_compressed(
                    os.path.join(root, f"{stem}__sample.npz"),
                    cond=cond[0].detach().cpu().numpy(),
                    pred=pred[0].detach().cpu().numpy(),
                    gt=gt[0].detach().cpu().numpy(),
                    meta=meta if isinstance(meta, dict) else {},
                )
            except ImportError:
                warnings.warn("Numpy not installed, skipping npz save")

        # Visualization
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib not installed, skipping visualization")
            return

        def _save_fig(tensor, name):
            fig = self._create_grid(tensor, name)
            if fig:
                fig.savefig(os.path.join(root, f"{stem}__{name}.png"), dpi=100, bbox_inches="tight")
                plt.close(fig)

        _save_fig(cond, "cond")
        _save_fig(pred, "pred")
        _save_fig(gt, "gt")

    def _create_grid(self, t: torch.Tensor, title: str):
        t = t.detach().float().cpu()
        b, c, h, w = t.shape
        ch = min(c, self.max_plot_channels)
        
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, ch, figsize=(ch * 3, 3))
            if ch == 1:
                axes = [axes]
            
            for i in range(ch):
                ax = axes[i]
                ax.imshow(t[0, i].numpy(), cmap="viridis")
                ax.axis("off")
                ax.set_title(f"{title} ch{i}")
            
            fig.tight_layout()
            return fig
        except Exception:
            return None