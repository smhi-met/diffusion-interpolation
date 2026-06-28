from __future__ import annotations
import os
import torch
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info

class GPUsAssignedCallback(L.Callback):
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        rank = trainer.global_rank
        local_rank = getattr(trainer, "local_rank", None)
        dev_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        cur_dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
        
        msg = (
            f"[Start] rank={rank} local_rank={local_rank} "
            f"cuda_count={dev_count} current_device={cur_dev} "
            f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
        print(msg, flush=True)

class GPUMemoryMonitorCallback(L.Callback):
    def __init__(self, every_n_steps: int = 50) -> None:
        self.every_n_steps = int(every_n_steps)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (trainer.global_step + 1) % self.every_n_steps != 0:
            return
        
        if not torch.cuda.is_available():
            return
            
        # Monitor current device for this process
        idx = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(idx) / (1024**2)
        reserv = torch.cuda.memory_reserved(idx) / (1024**2)
        
        print(
            f"GPUMemoryMonitor: [Rank {trainer.global_rank}] "
            f"GPU {idx}: allocated={alloc:.1f} MB, reserved={reserv:.1f} MB",
            flush=True,
        )