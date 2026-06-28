"""
Lightning model wrapper for QRLVariationalAutoEncoder.

Variational extension of QRLDCAEModel: the encoder outputs a posterior
DiagonalGaussian; the ELBO combines distribution-aware reconstruction losses
(from the QRL paper) with KL divergence and/or RBF-kernel MMD regularization.

────────────────────────────────────────────────────────────────────────────────
Reconstruction losses (identical to QRLDCAEModel)
────────────────────────────────────────────────────────────────────────────────
  "mse"     — plain weighted MSE (baseline)
  "qrl"     — Quantized Reconstruction Loss (uniform bin average)
  "wqrl"    — Weighted QRL (rarity-weighted bins)
  "fl_mse"  — Sigmoid-based Focal Loss
  "fl_hist" — Histogram-based Focal Loss
  Combine with "+": e.g. "mse+wqrl+fl_hist" (recommended default)

────────────────────────────────────────────────────────────────────────────────
Regularization (new in VAE)
────────────────────────────────────────────────────────────────────────────────
  "kl"     — KL divergence KL(q(z|x) ‖ N(0,I))  (standard β-VAE)
  "mmd"    — Max-Mean Discrepancy with RBF kernel (InfoVAE / WAE-MMD)
             This is what the paper uses.
             σ_eff = mmd_sigma × √D,  D = C×H_z×W_z, so mmd_sigma=1.0 gives
             E[k(z,z′)] ≈ exp(−½) ≈ 0.6 regardless of latent dimension.
  "kl+mmd" — Both terms combined (for ablation / comparison experiments)

────────────────────────────────────────────────────────────────────────────────
Training vs inference
────────────────────────────────────────────────────────────────────────────────
  Training:   z ~ q(z|x)  (reparameterization sample)
  Val / Test: z = μ        (saturated posterior mean — deterministic)
  MMD at val draws a separate posterior sample to estimate the distributional
  regularization signal without affecting the reconstruction metric.

Reference:
    "Quantizing reconstruction losses for improving weather data synthesis"
    Nature Scientific Reports, 2024 (doi:10.1038/s41598-024-52773-2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor

from dssml.models.base import ModelBase
from dssml.networks.ae import IAutoEncoder
from dssml.networks.ae.qrl_vae import QRLVariationalAutoEncoder

from .qrl_ae import (
    _fl_hist_per_channel,
    _fl_mse_per_channel,
    _mse_per_channel,
    _qrl_per_channel,
    _weighted_mean,
    _wqrl_per_channel,
)


# ── Regularization helpers ────────────────────────────────────────────────────

def _kl_loss(posterior) -> Tensor:
    """Mean KL(q(z|x) ‖ N(0,I)) over the batch.  Shape: scalar."""
    return posterior.kl().mean()


def _mmd_rbf(z_q: Tensor, sigma: float) -> Tensor:
    """
    Unbiased RBF-kernel MMD between posterior samples z_q and N(0, I) prior.

    Bandwidth σ_eff = sigma × √D (D = flattened latent dim) ensures the kernel
    is informative for any latent size: with sigma=1.0, E[k(z,z′)] ≈ exp(−½) ≈
    0.6 for two independent N(0,I) draws regardless of D.

    z_q : (B, C, H, W) — samples already drawn from the posterior
    returns : scalar MMD estimate (can be negative for small B)
    """
    B = z_q.shape[0]
    z_q_flat = z_q.reshape(B, -1).float()            # (B, D)
    D = z_q_flat.shape[1]
    sigma_eff_sq = 2.0 * (sigma * D ** 0.5) ** 2    # 2σ² in exponent denominator

    z_p_flat = torch.randn_like(z_q_flat)             # (B, D) ~ N(0, I)

    def _gram(a: Tensor, b: Tensor) -> Tensor:
        """RBF gram matrix: (n, D) × (m, D) → (n, m)."""
        a_sq = a.pow(2).sum(1, keepdim=True)          # (n, 1)
        b_sq = b.pow(2).sum(1, keepdim=True)          # (m, 1)
        sq_dist = (a_sq + b_sq.T - 2.0 * (a @ b.T)).clamp(min=0.0)
        return torch.exp(-sq_dist / sigma_eff_sq)

    k_qq = _gram(z_q_flat, z_q_flat)                 # (B, B)
    k_pp = _gram(z_p_flat, z_p_flat)                 # (B, B)
    k_qp = _gram(z_q_flat, z_p_flat)                 # (B, B)

    # Unbiased estimate: exclude diagonal for qq and pp terms
    mask = ~torch.eye(B, dtype=torch.bool, device=z_q.device)
    denom = float(B * (B - 1))

    return (k_qq[mask].sum() + k_pp[mask].sum()) / denom - 2.0 * k_qp.mean()


# ── Model ─────────────────────────────────────────────────────────────────────

class QRLVAEModel(ModelBase):
    """
    Lightning wrapper for QRLVariationalAutoEncoder.

    Trains encoder and decoder with:
      • Distribution-aware reconstruction losses (QRL / focal) — same as QRLDCAEModel.
      • Variational regularization: KL divergence and/or RBF-kernel MMD.

    At training z is sampled from the posterior (reparameterization trick).
    At validation/inference z is the saturated posterior mean (deterministic).

    Arguments
    ---------
    optimizers_conf : list[dict]
        Optimizer / scheduler configs (same format as all other models).
    ae_conf : dict
        Hydra config for QRLVariationalAutoEncoder
        (_target_: dssml.networks.ae.QRLVariationalAutoEncoder).
    loss_type : str
        Reconstruction terms joined by "+": "mse", "qrl", "wqrl", "fl_mse",
        "fl_hist".  Default: "mse+wqrl+fl_hist".
    reg_type : str
        Regularization term(s): "kl", "mmd", or "kl+mmd".
        Default: "mmd"  (InfoVAE approach from the paper).
    kl_weight : float
        Weight β on the KL term.  Small values (1e-4 – 1e-3) work well to
        avoid posterior collapse while keeping the latent space well-shaped.
    mmd_weight : float
        Weight on the MMD term.
    mmd_sigma : float
        RBF kernel bandwidth scale.  σ_eff = mmd_sigma × √D.
        Default 1.0 is informative for any latent dimension.
    mse_weight, qrl_weight, wqrl_weight, fl_mse_weight, fl_hist_weight : float
        Per-term reconstruction weights (same semantics as QRLDCAEModel).
    n_bins : int
        Histogram bins for QRL / WQRL / FL_hist (paper: 100).
    fl_beta : float
        β in FL_MSE sigmoid modulation (paper: 0.2).
    fl_gamma : float
        γ exponent for both focal losses (paper: 1.0).
    focal_scale : float
        Global scale λ applied to focal loss terms (paper: 100.0).
    normalizer : nn.Module | None
        Injected by DefaultTrainer; must expose a .loss_weights buffer (V,).
    """

    def __init__(
        self,
        optimizers_conf: list[dict],
        ae_conf: dict,
        loss_type: str = "mse+wqrl+fl_hist",
        reg_type: str = "mmd",
        kl_weight: float = 1e-4,
        mmd_weight: float = 1.0,
        mmd_sigma: float = 1.0,
        mse_weight: float = 1.0,
        qrl_weight: float = 1.0,
        wqrl_weight: float = 1.0,
        fl_mse_weight: float = 1.0,
        fl_hist_weight: float = 1.0,
        n_bins: int = 100,
        fl_beta: float = 0.2,
        fl_gamma: float = 1.0,
        focal_scale: float = 100.0,
        normalizer: nn.Module | None = None,
    ):
        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        self.save_hyperparameters(ignore=["normalizer"])

        self.loss_type = loss_type
        self.reg_type = reg_type
        self.kl_weight = kl_weight
        self.mmd_weight = mmd_weight
        self.mmd_sigma = mmd_sigma
        self.mse_weight = mse_weight
        self.qrl_weight = qrl_weight
        self.wqrl_weight = wqrl_weight
        self.fl_mse_weight = fl_mse_weight
        self.fl_hist_weight = fl_hist_weight
        self.n_bins = n_bins
        self.fl_beta = fl_beta
        self.fl_gamma = fl_gamma
        self.focal_scale = focal_scale

        # Parse reconstruction loss flags once at init
        rec_terms = set(loss_type.lower().replace("+", " ").split())
        self._has_mse     = bool(rec_terms & {"mse", "l2"})
        self._has_qrl     = "qrl"     in rec_terms
        self._has_wqrl    = "wqrl"    in rec_terms
        self._has_fl_mse  = "fl_mse"  in rec_terms
        self._has_fl_hist = "fl_hist" in rec_terms

        if not (self._has_mse or self._has_qrl or self._has_wqrl or
                self._has_fl_mse or self._has_fl_hist):
            raise ValueError(
                f"loss_type='{loss_type}' contains no recognised terms. "
                "Valid: mse, l2, qrl, wqrl, fl_mse, fl_hist"
            )

        # Parse regularization flags once at init
        reg_terms = set(reg_type.lower().replace("+", " ").split())
        self._has_kl  = "kl"  in reg_terms
        self._has_mmd = "mmd" in reg_terms

        if not (self._has_kl or self._has_mmd):
            raise ValueError(
                f"reg_type='{reg_type}' contains no recognised terms. Valid: kl, mmd"
            )

        self.auto_encoder: QRLVariationalAutoEncoder = instantiate(ae_conf, _recursive_=False)
        if not isinstance(self.auto_encoder, IAutoEncoder):
            raise ValueError(
                f"ae_conf must instantiate an IAutoEncoder, got {type(self.auto_encoder)}"
            )

    @property
    def name(self) -> str:
        return "QRLVAEModel"

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

        if self._has_mse:
            mse = _weighted_mean(_mse_per_channel(x_target, x_hat), w)
            self.log(f"{prefix}_mse", mse, on_epoch=True, sync_dist=True)
            total = total + self.mse_weight * mse

        if self._has_qrl:
            qrl = _weighted_mean(_qrl_per_channel(x_target, x_hat, self.n_bins), w)
            self.log(f"{prefix}_qrl", qrl, on_epoch=True, sync_dist=True)
            total = total + self.qrl_weight * qrl

        if self._has_wqrl:
            wqrl = _weighted_mean(_wqrl_per_channel(x_target, x_hat, self.n_bins), w)
            self.log(f"{prefix}_wqrl", wqrl, on_epoch=True, sync_dist=True)
            total = total + self.wqrl_weight * wqrl

        if self._has_fl_mse:
            fl_mse = _weighted_mean(
                _fl_mse_per_channel(x_target, x_hat, self.fl_beta, self.fl_gamma), w
            )
            self.log(f"{prefix}_fl_mse", fl_mse, on_epoch=True, sync_dist=True)
            total = total + self.focal_scale * self.fl_mse_weight * fl_mse

        if self._has_fl_hist:
            fl_hist = _weighted_mean(
                _fl_hist_per_channel(x_target, x_hat, self.n_bins, self.fl_gamma), w
            )
            self.log(f"{prefix}_fl_hist", fl_hist, on_epoch=True, sync_dist=True)
            total = total + self.focal_scale * self.fl_hist_weight * fl_hist

        self.log(f"{prefix}_rec_loss", total, on_epoch=True, sync_dist=True)
        return total

    # ── Regularization loss ───────────────────────────────────────────────────

    def _compute_reg_loss(self, posterior, z_sample: Tensor, prefix: str) -> Tensor:
        """
        posterior : DiagonalGaussianDistribution
        z_sample  : (B, C, H, W) — a sample from the posterior
                    (used for MMD; KL is computed analytically from the posterior)
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
