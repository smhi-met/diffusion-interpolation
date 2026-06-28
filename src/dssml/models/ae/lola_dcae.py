"""
Lightning model wrapper for LoLAAutoEncoder.

Supports configurable LoLA-style reconstruction losses with per-variable
physics weighting from the normalizer.  No perceptual loss, no KL divergence.
Latent regularisation is provided entirely by LoLASaturation inside the network.

loss_type strings
-----------------
    "l1"         – per-variable weighted MAE
    "l2"         – per-variable weighted MSE
    "l1+l2"      – MAE + MSE
    "vmse"       – variance-normalised MSE  (LoLA paper primary loss)
    "mae+vmse"   – MAE + vmse
    "vrmse"      – variance-normalised RMSE (√vmse)
    "vmse+vrmse" – both variance-normalised losses

Combinations are formed by joining terms with "+".  Each term gets its own
weight coefficient (l1_weight, vmse_weight, etc.).

Per-variable physics weights (from normalizer.loss_weights) are applied on
top of every loss term; variables with weight = 0 (e.g. lsm) are excluded.
The weight tensor comes from MultiNormalizer, which reads the `loss_weight`
field from each variable's config entry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor

from dssml.models.base import ModelBase
from dssml.networks.ae import IAutoEncoder
from dssml.networks.ae.lola_ae import LoLAAutoEncoder


# LoLA paper uses ε = 10^-2 in the variance denominator
_VMSE_EPS = 1e-2


# ── Loss helpers ──────────────────────────────────────────────────────────────

def _weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
    """Weighted mean over channel dim (dim=0); falls back to plain mean."""
    if weights is None or weights.sum() < 1e-8:
        return values.mean()
    return (values * weights).sum() / weights.sum()


def _vmse_per_channel(x: Tensor, y: Tensor) -> Tensor:
    """
    Variance-normalised MSE per channel, averaged over batch.

    VMSE(x, y) = ⟨(x − y)²⟩_spatial / (Var(x)_spatial + ε)

    x, y : (B, C, H, W)
    returns : (C,)
    """
    B, C = x.shape[:2]
    xf = x.reshape(B, C, -1)           # (B, C, S)
    yf = y.reshape(B, C, -1)
    num = (xf - yf).square().mean(dim=-1)   # (B, C)
    den = xf.var(dim=-1) + _VMSE_EPS        # (B, C)
    return (num / den).mean(dim=0)           # (C,)


# ── Model ─────────────────────────────────────────────────────────────────────

class LoLADCAEModel(ModelBase):
    """
    Lightning wrapper for LoLAAutoEncoder.

    The reconstruction loss is applied in the normalized input space
    (after MultiNormalizer) and weighted per variable using the
    loss_weights tensor that MultiNormalizer exposes.

    Arguments
    ---------
    optimizers_conf:
        List of optimizer/scheduler dicts (same format as other models).
    ae_conf:
        Hydra config for LoLAAutoEncoder (_target_ = dssml.networks.ae.LoLAAutoEncoder).
    loss_type:
        Reconstruction loss string; see module docstring for valid values.
    l1_weight:
        Coefficient for the MAE / L1 term.
    l2_weight:
        Coefficient for the MSE / L2 term.
    vmse_weight:
        Coefficient for the variance-normalised MSE term.
    vrmse_weight:
        Coefficient for the variance-normalised RMSE term.
    noise_std:
        Std-dev of Gaussian noise added to z during *training* only.
        Smooths the latent manifold for downstream diffusion training.
        Start at 0; consider 0.05–0.1 if diffusion shows latent artefacts.
    normalizer:
        Injected by DefaultTrainer; must expose a .loss_weights buffer of
        shape (V,) where V is the number of variables.
    """

    def __init__(
        self,
        optimizers_conf: list[dict],
        ae_conf: dict,
        loss_type: str = "vmse",
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
        vmse_weight: float = 1.0,
        vrmse_weight: float = 1.0,
        noise_std: float = 0.0,
        normalizer: nn.Module | None = None,
    ):
        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        self.save_hyperparameters(ignore=["normalizer"])

        self.loss_type = loss_type
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.vmse_weight = vmse_weight
        self.vrmse_weight = vrmse_weight
        self.noise_std = noise_std

        # Parse loss terms once at construction time for fast forward passes
        terms = set(loss_type.replace("+", " ").split())
        self._has_l1    = bool(terms & {"l1", "mae"})
        self._has_l2    = bool(terms & {"l2", "mse"})
        self._has_vmse  = "vmse"  in terms
        self._has_vrmse = "vrmse" in terms

        if not (self._has_l1 or self._has_l2 or self._has_vmse or self._has_vrmse):
            raise ValueError(
                f"loss_type='{loss_type}' contains no recognised terms. "
                "Valid: l1, mae, l2, mse, vmse, vrmse"
            )

        self.auto_encoder: LoLAAutoEncoder = instantiate(ae_conf, _recursive_=False)
        if not isinstance(self.auto_encoder, IAutoEncoder):
            raise ValueError(
                f"ae_conf must instantiate an IAutoEncoder, got {type(self.auto_encoder)}"
            )

    @property
    def name(self) -> str:
        return "LoLADCAEModel"

    # ── Weight helpers ────────────────────────────────────────────────────────

    def _expand_weights(self, V: int, T: int, E: int) -> Tensor | None:
        """
        Return per-channel weights of shape (T*V*E,) matching the flattened
        channel dimension of _x = x.view(B, T*V*E, H, W).

        Variable weights are broadcast: each (t, v, e) slot gets weight w[v].
        Variables with loss_weight=null in the config have weight 0 and are
        excluded from the weighted mean automatically.
        """
        if self.normalizer is None or not hasattr(self.normalizer, "loss_weights"):
            return None
        w = self.normalizer.loss_weights  # (V,)
        if T == 1 and E == 1:
            return w
        # expand: for flat index t*V*E + v*E + e  →  weight = w[v]
        w_ve = w.unsqueeze(1).expand(V, E).reshape(V * E)  # (V*E,)
        return w_ve.repeat(T)                                # (T*V*E,)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        x_hat: Tensor,
        x_target: Tensor,
        w: Tensor | None,
        prefix: str,
    ) -> Tensor:
        """
        x_hat, x_target : (B, C, H, W) — normalised space, C = T*V*E.
        w               : (C,) physics weights, or None.
        """
        total = x_hat.new_zeros(1).squeeze()

        if self._has_l1:
            mae_c = (x_hat - x_target).abs().mean(dim=[0, 2, 3])   # (C,)
            l1 = _weighted_mean(mae_c, w)
            self.log(f"{prefix}_l1", l1, on_epoch=True, sync_dist=True)
            total = total + self.l1_weight * l1

        if self._has_l2:
            mse_c = (x_hat - x_target).square().mean(dim=[0, 2, 3])  # (C,)
            l2 = _weighted_mean(mse_c, w)
            self.log(f"{prefix}_l2", l2, on_epoch=True, sync_dist=True)
            total = total + self.l2_weight * l2

        # Compute vmse_c once; reuse for both vmse and vrmse terms
        if self._has_vmse or self._has_vrmse:
            vmse_c = _vmse_per_channel(x_target, x_hat)   # (C,)

            if self._has_vmse:
                vmse = _weighted_mean(vmse_c, w)
                self.log(f"{prefix}_vmse", vmse, on_epoch=True, sync_dist=True)
                total = total + self.vmse_weight * vmse

            if self._has_vrmse:
                vrmse = _weighted_mean(vmse_c.sqrt(), w)
                self.log(f"{prefix}_vrmse", vrmse, on_epoch=True, sync_dist=True)
                total = total + self.vrmse_weight * vrmse

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

        if self.training and self.noise_std > 0.0:
            z = z + self.noise_std * torch.randn_like(z)

        x_hat = self.auto_encoder.decode(z)

        w = self._expand_weights(V, T, E)
        return self._compute_loss(x_hat, _x, w, prefix)

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
        Full encode → decode in physical space.
        Input/output shape: (B, T, V, E, H, W) in original (un-normalised) units.
        Used by LatentSpaceSampler callback and downstream latent diffusion.
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
