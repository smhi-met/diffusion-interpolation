from __future__ import annotations
import os
import warnings
from typing import Any, Optional, Tuple, Dict

import torch
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only

class SimpleAEOutput(L.Callback):
    def __init__(
        self,
        every_n_epochs: int = 1,
        save_npz: bool = False,
        out_dir: Optional[str] = None,
    ):
        super().__init__()
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.save_npz = bool(save_npz)
        self.out_dir = out_dir
        self._cache: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]] = None

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0 ,
    ) -> None:


        pass
        # TODO: add option to save metadata (e.g. time indices) along with reconstructions
 