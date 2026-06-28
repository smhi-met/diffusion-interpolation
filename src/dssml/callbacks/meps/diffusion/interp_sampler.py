"""
Validation callback for the latent interpolation diffusion model.

At the end of each validation epoch (every `log_interval` epochs) it:

  1. Picks `num_samples` deterministic validation windows.
  2. Runs the diffusion sampler conditioned on the two boundary latents.
  3. Decodes sampled + ground-truth latents back to physical-space variables.
  4. Writes results to a zarr store on rank-zero only.

Zarr layout
-----------
  dims: (epoch, sample, step, variable, ensemble, y, x)
  step = 0              → boundary frame t=0
  step = 1 … n_interp   → interpolated frames (generated / ground-truth)
  step = n_interp+1     → boundary frame t=6
  variables             → subset given by `variables` list

The store is opened in append mode on the epoch dimension so you can monitor
progress while training is still running.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class LatentInterpSampler(Callback):
    """
    Parameters
    ----------
    output_path : str
        Path to the output zarr store (created on first write).
    variables : list[str]
        Variable names to save (subset of manifest variables).
    num_samples : int
        Number of held-out validation windows to sample at each log step.
    log_interval : int
        Log every this many epochs (1 = every epoch).
    seed : int
        RNG seed for deterministic sample selection across runs.
    boundary_mode : {"gt", "reconstruct"}
        How boundary frames (t=0 and t=6) appear in the saved output.
        "gt"          — use the raw ground-truth values directly (default).
                        Boundaries are identical in generated and gt sequences,
                        making the interpolated steps easy to compare.
        "reconstruct" — encode then decode through the VAE.  Useful to audit
                        reconstruction quality but adds AE error to boundaries.
    """

    def __init__(
        self,
        output_path: str,
        variables: list[str],
        num_samples: int = 8,
        log_interval: int = 5,
        seed: int = 42,
        boundary_mode: str = "gt",
    ):
        super().__init__()
        if boundary_mode not in ("gt", "reconstruct"):
            raise ValueError(f"boundary_mode must be 'gt' or 'reconstruct', got {boundary_mode!r}")
        self.output_path   = Path(output_path)
        self.variables     = variables
        self.num_samples   = num_samples
        self.log_interval  = log_interval
        self.seed          = seed
        self.boundary_mode = boundary_mode

        # Set in on_train_start
        self._samples: torch.Tensor | None = None
        self._var_indices: list[int] | None = None
        self._V: int = 0
        self._E: int = 0
        self._ready = False

    # ── Setup on train start ─────────────────────────────────────────────────

    @rank_zero_only
    def on_train_start(self, trainer, pl_module) -> None:
        manifest = trainer.datamodule.manifest

        # Resolve variable indices
        self._var_indices = manifest.find_variables_indices(self.variables)

        # Grab a fixed subset of validation samples
        val_ds = trainer.datamodule.val_dataset
        rng    = torch.Generator().manual_seed(self.seed)
        idx    = torch.randperm(len(val_ds), generator=rng)[: self.num_samples]

        collected = []
        for i in idx.tolist():
            item = val_ds[i]
            collected.append(item["x"].unsqueeze(0))

        # Shape: (N, T, V, E, H, W) — kept on CPU, moved to device each epoch
        self._samples = torch.cat(collected, dim=0)
        _, _, self._V, self._E, _, _ = self._samples.shape
        self._ready = True
        print(
            f"[LatentInterpSampler] Initialised with {self.num_samples} samples, "
            f"vars={self.variables} (indices={self._var_indices})"
        )

    # ── Validation epoch end ─────────────────────────────────────────────────

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._ready:
            return
        if trainer.current_epoch % self.log_interval != 0:
            return

        device = pl_module.device
        x      = self._samples.to(device)           # (N, T, V, E, H, W)

        pl_module.eval()
        with torch.no_grad():
            # Generated interpolations + ground-truth latents
            z_gen, z_gt = pl_module.sample(x)       # both (N, n*C_z, H_z, W_z)

            # Latent statistics — helps calibrate sigma_data and detect mode collapse.
            # z_gt.std() ≈ sigma_data;  z_gen.std() should be close to z_gt.std().
            print(
                f"[LatentInterpSampler] Epoch {trainer.current_epoch}: "
                f"z_gt.std={z_gt.std():.4f}  z_gen.std={z_gen.std():.4f}  "
                f"(sigma_data={getattr(pl_module, 'sigma_data', '?')})"
            )
            # Per-channel std of ground-truth latents.  Channels with std≈0 are dead
            # (AE bottleneck not being used); large spread across channels means the
            # latent space is unbalanced and sigma_data is per-channel, not global.
            C_z = pl_module.latent_channels
            # z_gt: (N, n_interp*C_z, H_z, W_z) — reshape to (N, n_interp, C_z, H_z, W_z)
            N_s = z_gt.shape[0]
            z_gt_ch = z_gt.view(N_s, -1, C_z, *z_gt.shape[-2:])  # (N, n, C, H, W)
            ch_mean = z_gt_ch.mean(dim=(0, 1, 3, 4))              # (C_z,)
            ch_std  = z_gt_ch.std(dim=(0, 1, 3, 4))               # (C_z,)
            ch_std_min  = ch_std.min().item()
            ch_std_max  = ch_std.max().item()
            ch_mean_abs_max = ch_mean.abs().max().item()
            ch_dead = (ch_std < 0.05).sum().item()
            print(
                f"[LatentInterpSampler] latent channel std:  "
                f"min={ch_std_min:.4f}  max={ch_std_max:.4f}  "
                f"dead(<0.05)={ch_dead}/{C_z}"
            )
            whitening_on = getattr(pl_module, "_whitening_enabled", False)
            mean_hint = (
                "residual after whitening"
                if whitening_on
                else ">0 → latent space is not zero-centered; whitening needed"
            )
            print(
                f"[LatentInterpSampler] latent channel mean: "
                f"|mean|_max={ch_mean_abs_max:.4f}  ({mean_hint})"
            )

            V, E = self._V, self._E

            if self.boundary_mode == "reconstruct":
                # Encode then decode: useful to audit AE reconstruction quality.
                # encode_boundary_latents returns whitened latents; decode_to_physical
                # handles unwhitening + AE decode + denormalization in one call.
                z_cond = pl_module.encode_boundary_latents(x)  # (N, 2*C_z, …)
                b0_phys = pl_module.decode_to_physical(z_cond[:, :C_z], V, E).squeeze(1)
                b6_phys = pl_module.decode_to_physical(z_cond[:, C_z:], V, E).squeeze(1)

                # Per-variable reconstruction RMSE — reveals which variables the AE
                # reconstructs poorly; small alpha (tp, q_850) or large alpha (cbh, sp)
                # tend to be hardest.
                manifest = trainer.datamodule.manifest
                var_names = manifest.variables
                for frame, name in [(x[:, 0], "t=0"), (x[:, -1], "t=6")]:
                    recon = b0_phys if name == "t=0" else b6_phys
                    rmse_per_var = ((frame - recon) ** 2).mean(dim=(0, 2, 3, 4)).sqrt()
                    worst_idx = rmse_per_var.argmax().item()
                    print(
                        f"[LatentInterpSampler] AE recon RMSE {name}: "
                        f"mean={rmse_per_var.mean():.4f}  "
                        f"worst={var_names[worst_idx]}({rmse_per_var[worst_idx]:.4f})"
                    )
            else:
                # Use exact ground-truth boundaries — no encode/decode round-trip.
                # Boundaries are given data and should appear identical in both
                # generated and gt output sequences.
                b0_phys = x[:, 0]   # (N, V, E, H, W)
                b6_phys = x[:, -1]

            # Decode stacked interp latents to (N, n_interp, V, E, H, W)
            gen_phys = pl_module.decode_to_physical(z_gen, V, E)
            gt_phys  = pl_module.decode_to_physical(z_gt,  V, E)

        # Build full sequence tensors: (N, T, V, E, H, W)  T = n_interp + 2
        def _full_seq(boundary_0, intermed, boundary_6):
            return torch.cat(
                [boundary_0.unsqueeze(1), intermed, boundary_6.unsqueeze(1)],
                dim=1,
            ).cpu().numpy()

        gen_seq = _full_seq(b0_phys, gen_phys, b6_phys)  # (N, T, V, E, H, W)
        gt_seq  = _full_seq(b0_phys, gt_phys,  b6_phys)

        # Subset to requested variables
        vi = self._var_indices
        gen_sub = gen_seq[:, :, vi, :, :, :]   # (N, T, n_vars, E, H, W)
        gt_sub  = gt_seq[:,  :, vi, :, :, :]

        T     = gen_sub.shape[1]
        H, W  = gen_sub.shape[-2], gen_sub.shape[-1]
        epoch = trainer.current_epoch

        ds = xr.Dataset(
            data_vars={
                "generated": (
                    ["epoch", "sample", "step", "variable", "ensemble", "y", "x"],
                    gen_sub[None],      # add epoch dim
                ),
                "ground_truth": (
                    ["epoch", "sample", "step", "variable", "ensemble", "y", "x"],
                    gt_sub[None],
                ),
            },
            coords={
                "epoch":    [epoch],
                "sample":   np.arange(self.num_samples),
                "step":     np.arange(T),
                "variable": self.variables,
                "ensemble": np.arange(self._E),
                "y":        np.arange(H),
                "x":        np.arange(W),
            },
            attrs={
                "step_0":   "boundary t=0",
                f"step_{T - 1}": "boundary t=6",
                "steps_1_to": f"generated/gt interpolation (t=1 … {T - 2})",
            },
        )

        if not self.output_path.exists():
            ds.to_zarr(self.output_path, mode="w")
        else:
            ds.to_zarr(self.output_path, append_dim="epoch")

        print(
            f"[LatentInterpSampler] Epoch {epoch}: saved samples to "
            f"{self.output_path}"
        )
