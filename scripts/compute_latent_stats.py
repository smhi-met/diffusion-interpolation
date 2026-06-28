"""
Compute per-channel latent mean and std for whitening the EDM latent space.

Usage
-----
    python scripts/compute_latent_stats.py \\
        --config experiments/mlx/14-latent-edm-intrp/14-latent-edm-intrp-z32.yaml \\
        --ckpt   _assets/09-winds-focal/.../last.ckpt \\
        --n-batches 200

Output: prints YAML-ready `latent_mean` and `latent_std` lists to stdout.
Add them to the model section of the config, set sigma_data: 1.0,
and p_mean: -1.2 (Karras default for unit-std latents).

Why this matters
----------------
Without whitening, per-channel stds range widely (e.g. 0.11–0.46 for z64).
A single scalar sigma_data cannot capture this imbalance.  After whitening
every channel has std≈1, sigma_data=1.0 is exact, and training is more
efficient (equal EDM loss weight across channels).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True, help="Hydra config YAML path")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-batches", type=int, default=200,
                   help="Number of training batches to average over")
    p.add_argument("--split",     default="train",
                   help="Dataset split to compute stats on: train | val")
    return p.parse_args()


def main():
    args = parse_args()

    # Activate environment if needed (called via run.sh / salloc)
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    # Load config
    cfg = OmegaConf.load(args.config)

    # Build manifest → normalizer → datamodule (mirrors DefaultTrainer wiring)
    print("Building manifest and normalizer...", flush=True)
    manifest   = instantiate(cfg.manifest, _recursive_=False)
    normalizer = instantiate(
        cfg.normalizer,
        stats=manifest.stats,
        var_dim=manifest.var_dim,
        variables_conf=manifest.variables_conf,
    ) if cfg.get("normalizer") else None
    datamodule = instantiate(cfg.datamodule, manifest=manifest, _recursive_=False)
    datamodule.setup("fit")

    loader = (
        datamodule.train_dataloader()
        if args.split == "train"
        else datamodule.val_dataloader()
    )

    # Load first stage AE (no diffusion UNet needed)
    print("Loading first stage AE...", flush=True)
    first_stage_conf = dict(cfg.model.first_stage_conf)
    ckpt_path = first_stage_conf.pop("ckpt_path")
    target    = first_stage_conf.pop("_target_")

    from hydra.utils import get_class
    model_cls  = get_class(target)
    first_stage = model_cls.load_from_checkpoint(
        ckpt_path, strict=False, **first_stage_conf
    ).to(args.device).eval()
    ae = first_stage.auto_encoder.eval()

    C_z = None
    z_sum    = None
    z_sum_sq = None
    z_count  = 0

    print(f"Accumulating stats over up to {args.n_batches} batches...", flush=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.n_batches:
                break

            x = batch["x"].to(args.device)      # (B, T, V, E, H, W)
            B, T, V, E, H, W = x.shape
            x_norm = normalizer(x).view(B, T, V * E, H, W)

            # Encode all interior frames (skip boundaries for pure interior stats)
            for t in range(1, T - 1):
                z = ae.encode(x_norm[:, t])      # (B, C_z, H_z, W_z)
                if C_z is None:
                    C_z   = z.shape[1]
                    z_sum    = torch.zeros(C_z, device=args.device, dtype=torch.float64)
                    z_sum_sq = torch.zeros(C_z, device=args.device, dtype=torch.float64)

                # Flatten spatial dims: (B*H_z*W_z, C_z)
                flat = z.permute(0, 2, 3, 1).reshape(-1, C_z).double()
                z_sum    += flat.sum(0)
                z_sum_sq += flat.pow(2).sum(0)
                z_count  += flat.shape[0]

            if (batch_idx + 1) % 20 == 0:
                print(f"  processed {batch_idx + 1} batches...", flush=True)

    if C_z is None:
        print("ERROR: no batches processed", file=sys.stderr)
        sys.exit(1)

    z_mean = (z_sum / z_count).float().cpu()
    z_var  = (z_sum_sq / z_count - z_mean.double().to(args.device) ** 2).float().cpu()
    z_std  = z_var.clamp(min=0).sqrt().clamp(min=1e-4)

    print(f"\nComputed over {z_count} samples per channel")
    print(f"Per-channel mean: min={z_mean.min():.4f}  max={z_mean.max():.4f}  |max|={z_mean.abs().max():.4f}")
    print(f"Per-channel std:  min={z_std.min():.4f}   max={z_std.max():.4f}")
    print()
    print("─" * 60)
    print("Add the following to your config under model:")
    print("─" * 60)
    mean_list = "[" + ", ".join(f"{v:.6f}" for v in z_mean.tolist()) + "]"
    std_list  = "[" + ", ".join(f"{v:.6f}" for v in z_std.tolist())  + "]"
    print(f"  latent_mean: {mean_list}")
    print(f"  latent_std:  {std_list}")
    print()
    print("Also update:")
    print("  sigma_data: 1.0   # correct after whitening")
    print("  p_mean: -1.2      # Karras default for unit-std latents")
    print("─" * 60)


if __name__ == "__main__":
    main()
