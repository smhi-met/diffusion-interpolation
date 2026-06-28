"""
Lightning model wrapper for LoLAVariationalAutoEncoder.

Variational extension of LoLADCAEModel: the encoder outputs a posterior
DiagonalGaussian; the ELBO combines LoLA reconstruction losses with KL
divergence and/or RBF-kernel MMD regularization.

────────────────────────────────────────────────────────────────────────────────
Reconstruction losses (identical to LoLADCAEModel)
────────────────────────────────────────────────────────────────────────────────
  "l1"     — per-variable weighted MAE
  "l2"     — per-variable weighted MSE
  "vmse"   — variance-normalised MSE  (LoLA paper primary loss)
  "vrmse"  — variance-normalised RMSE (√vmse)
  Combine with "+": e.g. "vmse" (recommended default)

────────────────────────────────────────────────────────────────────────────────
Regularization
────────────────────────────────────────────────────────────────────────────────
  "kl"     — KL divergence KL(q(z|x) ‖ N(0,I))  (standard β-VAE)
             Enforces zero-mean, unit-variance latents per sample.
  "mmd"    — Max-Mean Discrepancy with RBF kernel (InfoVAE / WAE-MMD)
             Matches the aggregate posterior to N(0,I) at distribution level.
  "kl+mmd" — Both terms combined

────────────────────────────────────────────────────────────────────────────────
Training vs inference
────────────────────────────────────────────────────────────────────────────────
  Training:    z ~ q(z|x)  (reparameterization sample)
  Val / Test:  z = μ        (saturated posterior mean — deterministic)

Reference:
    Lost in Latent Space (arXiv:2507.02608) — LoLA losses
    InfoVAE (arXiv:1706.02262) — MMD regularization
"""

from __future__ import annotations

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor

from dssml.models.base import ModelBase
from dssml.networks.ae import IAutoEncoder
from dssml.networks.ae.lola_ae import LoLAVariationalAutoEncoder

from .lola_dcae import _vmse_per_channel, _weighted_mean
from .qrl_vae import _kl_loss, _mmd_rbf


class LoLAVAEModel(ModelBase):
    """
    Lightning wrapper for LoLAVariationalAutoEncoder.

    Trains encoder and decoder with:
      • LoLA reconstruction losses (vmse / l1 / l2 / vrmse) — same as LoLADCAEModel.
      • Variational regularization: KL divergence and/or RBF-kernel MMD.

    At training z is sampled from the posterior (reparameterization trick).
    At validation/inference z is the saturated posterior mean (deterministic).

    Arguments
    ---------
    optimizers_conf : list[dict]
        Optimizer / scheduler configs (same format as all other models).
    ae_conf : dict
        Hydra config for LoLAVariationalAutoEncoder
        (_target_: dssml.networks.ae.LoLAVariationalAutoEncoder).
    loss_type : str
        Reconstruction terms joined by "+": "l1", "mae", "l2", "mse",
        "vmse", "vrmse".  Default: "vmse".
    reg_type : str
        Regularization term(s): "kl", "mmd", or "kl+mmd".  Default: "kl".
    kl_weight : float
        Weight β on the KL term.  Small values (1e-4 – 1e-3) enforce zero
        mean without causing posterior collapse.
    mmd_weight : float
        Weight on the MMD term.
    mmd_sigma : float
        RBF kernel bandwidth scale.  σ_eff = mmd_sigma × √D.
    l1_weight, l2_weight, vmse_weight, vrmse_weight : float
        Per-term reconstruction weights (same semantics as LoLADCAEModel).
    normalizer : nn.Module | None
        Injected by DefaultTrainer; must expose a .loss_weights buffer (V,).
    """

    def __init__(
        self,
        optimizers_conf: list[dict],
        ae_conf: dict,
        loss_type: str = "vmse",
        reg_type: str = "kl",
        kl_weight: float = 1e-4,
        mmd_weight: float = 1.0,
        mmd_sigma: float = 1.0,
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
        vmse_weight: float = 1.0,
        vrmse_weight: float = 1.0,
        normalizer: nn.Module | None = None,
    ):
        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        self.save_hyperparameters(ignore=["normalizer"])

        self.loss_type = loss_type
        self.reg_type = reg_type
        self.kl_weight = kl_weight
        self.mmd_weight = mmd_weight
        self.mmd_sigma = mmd_sigma
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.vmse_weight = vmse_weight
        self.vrmse_weight = vrmse_weight

        # Parse reconstruction loss flags once at init
        rec_terms = set(loss_type.replace("+", " ").split())
        self._has_l1    = bool(rec_terms & {"l1", "mae"})
        self._has_l2    = bool(rec_terms & {"l2", "mse"})
        self._has_vmse  = "vmse"  in rec_terms
        self._has_vrmse = "vrmse" in rec_terms

        if not (self._has_l1 or self._has_l2 or self._has_vmse or self._has_vrmse):
            raise ValueError(
                f"loss_type='{loss_type}' contains no recognised terms. "
                "Valid: l1, mae, l2, mse, vmse, vrmse"
            )

        # Parse regularization flags once at init
        reg_terms = set(reg_type.lower().replace("+", " ").split())
        self._has_kl  = "kl"  in reg_terms
        self._has_mmd = "mmd" in reg_terms

        if not (self._has_kl or self._has_mmd):
            raise ValueError(
                f"reg_type='{reg_type}' contains no recognised terms. Valid: kl, mmd"
            )

        self.auto_encoder: LoLAVariationalAutoEncoder = instantiate(ae_conf, _recursive_=False)
        if not isinstance(self.auto_encoder, IAutoEncoder):
            raise ValueError(
                f"ae_conf must instantiate an IAutoEncoder, got {type(self.auto_encoder)}"
            )

    @property
    def name(self) -> str:
        return "LoLAVAEModel"

    # ── Weight helpers ────────────────────────────────────────────────────────

    def _expand_weights(self, V: int, T: int, E: int) -> Tensor | None:
        """Per-channel physics weights (T*V*E,) for the flattened channel dim."""
        if self.normalizer is None or not hasattr(self.normalizer, "loss_weights"):
            return None
        w = self.normalizer.loss_weights   # (V,)
        if T == 1 and E == 1:
            return w
        w_ve = w.unsqueeze(1).expand(V, E).reshape(V * E)
        return w_ve.repeat(T)   # (T*V*E,)

    # ── Reconstruction loss ───────────────────────────────────────────────────

    def _compute_rec_loss(
        self,
        x_hat: Tensor,
        x_target: Tensor,
        w: Tensor | None,
        prefix: str,
    ) -> Tensor:
        """x_hat, x_target: (B, C, H, W) normalised.  w: (C,) or None."""
        total = x_hat.new_zeros(1).squeeze()

        if self._has_l1:
            mae_c = (x_hat - x_target).abs().mean(dim=[0, 2, 3])
            l1 = _weighted_mean(mae_c, w)
            self.log(f"{prefix}_l1", l1, on_epoch=True, sync_dist=True)
            total = total + self.l1_weight * l1

        if self._has_l2:
            mse_c = (x_hat - x_target).square().mean(dim=[0, 2, 3])
            l2 = _weighted_mean(mse_c, w)
            self.log(f"{prefix}_l2", l2, on_epoch=True, sync_dist=True)
            total = total + self.l2_weight * l2

        if self._has_vmse or self._has_vrmse:
            vmse_c = _vmse_per_channel(x_target, x_hat)

            if self._has_vmse:
                vmse = _weighted_mean(vmse_c, w)
                self.log(f"{prefix}_vmse", vmse, on_epoch=True, sync_dist=True)
                total = total + self.vmse_weight * vmse

            if self._has_vrmse:
                vrmse = _weighted_mean(vmse_c.sqrt(), w)
                self.log(f"{prefix}_vrmse", vrmse, on_epoch=True, sync_dist=True)
                total = total + self.vrmse_weight * vrmse

        self.log(f"{prefix}_rec_loss", total, on_epoch=True, sync_dist=True)
        return total

    # ── Regularization loss ───────────────────────────────────────────────────

    def _compute_reg_loss(self, posterior, z_sample: Tensor, prefix: str) -> Tensor:
        """
        posterior : DiagonalGaussianDistribution
        z_sample  : (B, C, H, W) — a sample from the posterior
        """
        total = z_sample.new_zeros(1).squeeze()

        if self._has_kl:
            kl = _kl_loss(posterior)
            self.log(f"{prefix}_kl", kl, on_epoch=True, sync_dist=True)
            total = total + self.kl_weight * kl

        if self._has_mmd:
            mmd = _mmd_rbf(z_sample, self.mmd_sigma)
            self.log(f"{prefix}_mmd", mmd, on_epoch=True, sync_dist=True)
            total = total + self.mmd_weight * mmd

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

        posterior = self.auto_encoder.encode_posterior(_x)

        if self.training:
            z = posterior.sample()
            z_for_reg = z
        else:
            # Deterministic reconstruction for clean val metrics
            z = posterior.mode()
            # MMD needs samples to estimate distributional match; draw separately
            z_for_reg = posterior.sample() if self._has_mmd else z

        x_hat = self.auto_encoder.decode(z)
        w = self._expand_weights(V, T, E)

        rec_loss = self._compute_rec_loss(x_hat, _x, w, prefix)
        reg_loss = self._compute_reg_loss(posterior, z_for_reg, prefix)

        total = rec_loss + reg_loss
        self.log(f"{prefix}_loss", total, prog_bar=True, on_epoch=True, sync_dist=True)
        return total

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
        z = self.auto_encoder.encode(_x)   # returns μ (deterministic)
        x_hat = self.auto_encoder.decode(z).view(B, T, V, E, H, W)
        if self.normalizer is not None:
            x_hat = self.normalizer.denormalize(x_hat)
        return x_hat

    def forward(self, x: Tensor) -> Tensor:
        """
        Full encode → decode in physical space (deterministic mean).
        Input/output: (B, T, V, E, H, W) in original (un-normalised) units.
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
