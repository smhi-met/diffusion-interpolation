"""
Lightning model wrapper for DCAutoEncoder.

Supports configurable L1 / L2 reconstruction losses with per-variable
weighting from the normalizer.  No perceptual loss, no KL divergence.
Latent regularization is entirely provided by LatentSaturation inside the
network.  Optional Gaussian noise on z during training smooths the latent
manifold for downstream diffusion.

loss_type string examples:
    "l1"     – weighted MAE only
    "l2"     – weighted MSE only
    "l1+l2"  – both terms (l1_weight and l2_weight control the mix)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor

from dssml.models.base import ModelBase
from dssml.networks.ae import IAutoEncoder
from dssml.networks.ae.dcae import DCAutoEncoder


class DCAEModel(ModelBase):
    """
    Lightning wrapper for DCAutoEncoder.

    The reconstruction loss is applied in the normalized input space
    (after MultiNormalizer) and weighted per variable using the
    loss_weights tensor that MultiNormalizer exposes.

    Arguments:
        optimizers_conf:  list of optimizer/scheduler dicts (same format as other models)
        ae_conf:          Hydra config for DCAutoEncoder (_target_ = dssml.networks.ae.DCAutoEncoder)
        loss_type:        "l1", "l2", or "l1+l2"
        l1_weight:        coefficient for the L1 term (used when "l1" in loss_type)
        l2_weight:        coefficient for the L2 term (used when "l2" in loss_type)
        noise_std:        std-dev of Gaussian noise added to z during training only
                          (0 = disabled; 0.05–0.2 recommended when enabling)
        normalizer:       injected by DefaultTrainer; exposes .loss_weights
    """

    def __init__(
        self,
        optimizers_conf: list[dict],
        ae_conf: dict,
        loss_type: str = "l1+l2",
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
        noise_std: float = 0.0,
        normalizer: nn.Module | None = None,
    ):
        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        self.save_hyperparameters(ignore=["normalizer"])

        self.loss_type = loss_type
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.noise_std = noise_std

        self.auto_encoder: DCAutoEncoder = instantiate(ae_conf, _recursive_=False)
        if not isinstance(self.auto_encoder, IAutoEncoder):
            raise ValueError(
                f"ae_conf must instantiate an IAutoEncoder, got {type(self.auto_encoder)}"
            )

    @property
    def name(self) -> str:
        return "DCAEModel"

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _per_var_weights(self, x: Tensor) -> Tensor | None:
        """Return (1, C, 1, 1) loss-weight tensor if the normalizer has one."""
        if self.normalizer is not None and hasattr(self.normalizer, "loss_weights"):
            return self.normalizer.loss_weights.view(1, -1, 1, 1)
        return None

    def _compute_loss(self, x_hat: Tensor, x_target: Tensor, prefix: str) -> Tensor:
        total = x_hat.new_zeros(1).squeeze()
        w = self._per_var_weights(x_hat)

        if "l1" in self.loss_type:
            err = (x_hat - x_target).abs()
            l1 = (err * w).mean() if w is not None else err.mean()
            self.log(f"{prefix}_l1", l1, on_epoch=True, sync_dist=True)
            total = total + self.l1_weight * l1

        if "l2" in self.loss_type:
            err = (x_hat - x_target).pow(2)
            l2 = (err * w).mean() if w is not None else err.mean()
            self.log(f"{prefix}_l2", l2, on_epoch=True, sync_dist=True)
            total = total + self.l2_weight * l2

        self.log(f"{prefix}_loss", total, prog_bar=True, on_epoch=True, sync_dist=True)
        return total

    # ── Shared step ───────────────────────────────────────────────────────────

    def _shared_step(self, batch: dict, batch_idx: int, prefix: str) -> Tensor:
        x = batch["x"]
        B, T, V, E, H, W = x.shape

        _x = (
            self.normalizer(x).view(B, T * V * E, H, W)
            if self.normalizer is not None
            else x.view(B, T * V * E, H, W)
        )

        z = self.auto_encoder.encode(_x)

        # Latent noise injection: training-only regularization
        if self.training and self.noise_std > 0.0:
            z = z + self.noise_std * torch.randn_like(z)

        x_hat = self.auto_encoder.decode(z)
        return self._compute_loss(x_hat, _x, prefix)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_step(self, batch, batch_idx):
        x = batch["x"]
        B, T, V, E, H, W = x.shape
        _x = (
            self.normalizer(x).view(B, T * V * E, H, W)
            if self.normalizer is not None
            else x.view(B, T * V * E, H, W)
        )
        z = self.auto_encoder.encode(_x)
        x_hat = self.auto_encoder.decode(z).view(B, T, V, E, H, W)
        if self.normalizer is not None:
            x_hat = self.normalizer.denormalize(x_hat)
        return x_hat

    def forward(self, x: Tensor) -> Tensor:
        """
        Full encode → decode pass in physical space.
        Input/output: (B, T, V, E, H, W) in original (un-normalized) units.
        Used by LatentSpaceSampler callback and downstream inference scripts.
        """
        B, T, V, E, H, W = x.shape
        _x = (
            self.normalizer(x).view(B, T * V * E, H, W)
            if self.normalizer is not None
            else x.view(B, T * V * E, H, W)
        )
        z = self.auto_encoder.encode(_x)
        x_hat = self.auto_encoder.decode(z).view(B, T, V, E, H, W)
        if self.normalizer is not None:
            x_hat = self.normalizer.denormalize(x_hat)
        return x_hat
