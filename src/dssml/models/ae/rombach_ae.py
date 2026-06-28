"""
Lightning model for the Rombach-style first-stage autoencoder.

Wraps RombachAutoEncoder and handles:
  - KL mode: MSE + KL-regularization (with linear warmup) + optional spectral/gradient losses
  - VQ mode: MSE + codebook loss (commitment + codebook update)
  - Per-variable loss weighting via normalizer.loss_weights
  - Batch shape: (B, T, V, E, H, W) → (B, T*V*E, H, W) before forwarding to network
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from hydra.utils import instantiate

from dssml.models.base import ModelBase
from dssml.networks.ae import IAutoEncoder
from dssml.networks.ae.ldm_vae import DiagonalGaussian


class RombachAEModel(ModelBase):
    """
    Lightning model wrapping RombachAutoEncoder.

    Supports both regularization modes:
      regulation='kl'  →  loss_type can include 'mse', 'kl', 'spectral', 'gradient'
      regulation='vq'  →  loss_type can include 'mse', 'vq', 'spectral', 'gradient'
                          ('kl' is silently ignored in VQ mode)

    Loss type string (combine with '+'):
      - 'mse'       : per-variable weighted MSE reconstruction loss
      - 'kl'        : KL divergence (KL mode only); linearly warmed up from 0
      - 'vq'        : VQ codebook + commitment loss (VQ mode only)
      - 'spectral'  : frequency-domain penalty on amplitude spectrum
      - 'gradient'  : finite-difference spatial gradient penalty

    Example configs:
      regulation='kl',  loss_type='mse+kl'
      regulation='kl',  loss_type='mse+kl+spectral+gradient'
      regulation='vq',  loss_type='mse+vq'
      regulation='vq',  loss_type='mse+vq+spectral+gradient'
    """

    def __init__(
        self,
        optimizers_conf: list[dict],
        ae_conf: dict,
        loss_type: str = "mse+kl",
        kl_weight: float = 1e-6,
        kl_warmup_steps: int = 10000,
        vq_loss_weight: float = 1.0,
        spectral_weight: float = 0.1,
        gradient_weight: float = 0.1,
        normalizer: nn.Module | None = None,
    ):
        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        self.save_hyperparameters(ignore=["normalizer"])

        self.loss_type = loss_type
        self.kl_weight = kl_weight
        self.kl_warmup_steps = kl_warmup_steps
        self.vq_loss_weight = vq_loss_weight
        self.spectral_weight = spectral_weight
        self.gradient_weight = gradient_weight

        self.ae: IAutoEncoder = instantiate(ae_conf, _recursive_=False)
        if not isinstance(self.ae, IAutoEncoder):
            raise ValueError(
                f"ae_conf must instantiate an IAutoEncoder, got {type(self.ae)}"
            )

        # Cache regulation mode from the instantiated network
        self.regulation = getattr(self.ae, "regulation", "kl")

    @property
    def name(self):
        return "RombachAEModel"

    # ── Physical losses ────────────────────────────────────────────────────────

    def _compute_spectral_loss(self, x_hat: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        fft_hat = torch.fft.rfft2(x_hat, norm="ortho")
        fft_tgt = torch.fft.rfft2(x_target, norm="ortho")
        return F.mse_loss(fft_hat.abs(), fft_tgt.abs())

    def _compute_gradient_loss(self, x_hat: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        dx_hat = x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1]
        dx_tgt = x_target[:, :, :, 1:] - x_target[:, :, :, :-1]
        dy_hat = x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :]
        dy_tgt = x_target[:, :, 1:, :] - x_target[:, :, :-1, :]
        return F.mse_loss(dx_hat, dx_tgt) + F.mse_loss(dy_hat, dy_tgt)

    def _get_kl_weight(self) -> float:
        return self.kl_weight * min(1.0, self.global_step / max(1, self.kl_warmup_steps))

    # ── Combined loss ──────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        x_hat: torch.Tensor,
        x_target: torch.Tensor,
        reg_term,    # DiagonalGaussian (KL) or scalar Tensor (VQ)
        prefix: str,
    ) -> torch.Tensor:

        total_loss = torch.zeros(1, device=x_hat.device, dtype=x_hat.dtype).squeeze()

        # 1. MSE reconstruction with per-variable weighting
        if "mse" in self.loss_type:
            per_var_mse = (x_hat - x_target).pow(2)  # (B, C, H, W)
            if self.normalizer is not None and hasattr(self.normalizer, "loss_weights"):
                w = self.normalizer.loss_weights.view(1, -1, 1, 1)
                rec_loss = (per_var_mse * w).mean()
            else:
                rec_loss = per_var_mse.mean()
            self.log(f"{prefix}_rec_loss", rec_loss, prog_bar=True, on_epoch=True, sync_dist=True)
            total_loss = total_loss + rec_loss

        # 2a. KL divergence (KL mode only)
        if "kl" in self.loss_type and self.regulation == "kl":
            kl_loss = reg_term.kl().mean()
            kl_w = self._get_kl_weight()
            self.log(f"{prefix}_kl_loss", kl_loss, on_epoch=True, sync_dist=True)
            self.log(f"{prefix}_kl_weight", kl_w, on_epoch=True, sync_dist=True)
            total_loss = total_loss + kl_w * kl_loss

        # 2b. VQ codebook + commitment loss (VQ mode only)
        if "vq" in self.loss_type and self.regulation == "vq":
            self.log(f"{prefix}_vq_loss", reg_term, on_epoch=True, sync_dist=True)
            total_loss = total_loss + self.vq_loss_weight * reg_term

        # 3. Spectral loss
        if "spectral" in self.loss_type:
            spectral_loss = self._compute_spectral_loss(x_hat, x_target)
            self.log(f"{prefix}_spectral_loss", spectral_loss, on_epoch=True, sync_dist=True)
            total_loss = total_loss + self.spectral_weight * spectral_loss

        # 4. Gradient loss
        if "gradient" in self.loss_type:
            gradient_loss = self._compute_gradient_loss(x_hat, x_target)
            self.log(f"{prefix}_gradient_loss", gradient_loss, on_epoch=True, sync_dist=True)
            total_loss = total_loss + self.gradient_weight * gradient_loss

        self.log(f"{prefix}_loss", total_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return total_loss

    # ── Lightning steps ────────────────────────────────────────────────────────

    def _shared_step(self, batch, batch_idx, prefix="val"):
        x = batch["x"]
        B, T, V, E, H, W = x.shape

        if self.normalizer is not None:
            _x = self.normalizer(x).view(B, T * V * E, H, W)
        else:
            _x = x.view(B, T * V * E, H, W)

        # sample_posterior=True during training for stochastic KL,
        # False during validation for deterministic (posterior mean)
        x_hat, reg_term = self.ae(_x, sample_posterior=self.training)
        return self._compute_loss(x_hat, _x, reg_term, prefix)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="test")

    def predict_step(self, batch, batch_idx):
        x = batch["x"]
        B, T, V, E, H, W = x.shape
        if self.normalizer is not None:
            _x = self.normalizer(x).view(B, T * V * E, H, W)
        else:
            _x = x.view(B, T * V * E, H, W)

        z = self.ae.get_latent(_x, deterministic=True)
        x_hat = self.ae.decode(z).view(B, T, V, E, H, W)

        if self.normalizer is not None:
            x_hat = self.normalizer.denormalize(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V, E, H, W = x.shape
        if self.normalizer is not None:
            _x = self.normalizer(x).view(B, T * V * E, H, W)
        else:
            _x = x.view(B, T * V * E, H, W)

        z = self.ae.get_latent(_x, deterministic=True)
        x_hat = self.ae.decode(z).view(B, T, V, E, H, W)

        if self.normalizer is not None:
            x_hat = self.normalizer.denormalize(x_hat)
        return x_hat
