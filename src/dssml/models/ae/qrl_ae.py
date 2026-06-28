"""
Lightning model wrapper for QRLAutoEncoder.

Implements Approach 1 (Quantized Reconstruction Losses) and Approach 2
(Reconstruction Focal Losses) from:

    "Quantizing reconstruction losses for improving weather data synthesis"
    Nature Scientific Reports, 2024 (doi:10.1038/s41598-024-52773-2)

────────────────────────────────────────────────────────────────────────────────
Approach 1 — Quantized Reconstruction Losses
────────────────────────────────────────────────────────────────────────────────
Standard MSE treats every pixel equally, which allows dominant value ranges
(e.g. zero-precipitation) to completely swamp the gradient signal from rare
but physically important extremes (heavy rain, extreme temperatures).

The fix: compute MSE *per histogram bin* of the target field and then average
uniformly across bins.  Each value range contributes equally regardless of
frequency.

  "qrl"   — Quantized Reconstruction Loss:
              L_qrl = (1/|B|) Σ_j  MSE_j
              where MSE_j = mean squared error over pixels in bin j.
              All bins contribute equally; common and rare values are balanced.

  "wqrl"  — Weighted Quantized Reconstruction Loss:
              L_wqrl = Σ_j ω_j · MSE_j  /  Σ_j ω_j
              ω_j = 1 − h_j,  h_j = normalised bin frequency ∈ [0, 1]
              Rare bins (h_j ≈ 0) get weight ≈ 1; common bins are down-weighted.

────────────────────────────────────────────────────────────────────────────────
Approach 2 — Reconstruction Focal Losses
────────────────────────────────────────────────────────────────────────────────
Inspired by focal loss from object detection: a per-pixel modulating factor
suppresses easy samples and focuses training on hard / rare ones.

  "fl_mse"  — Sigmoid-based focal loss:
              FL_MSE(x, x̂) = σ(β|x − x̂|)^γ · (x − x̂)²
              β = 0.2, γ = 1.0  (paper defaults)
              Pixels with large residuals receive a larger modulating factor.

  "fl_hist" — Histogram-based focal loss:
              FL_hist(x, x̂) = ω_j(x)^γ · (x − x̂)²
              ω_j = 1 − h_j  (same as WQRL weights)
              The modulating factor depends on how rare the *target value* is,
              not the error magnitude.

────────────────────────────────────────────────────────────────────────────────
Baseline
────────────────────────────────────────────────────────────────────────────────
  "mse"   — plain weighted MSE (baseline, same as DCAEModel with l2 loss)

────────────────────────────────────────────────────────────────────────────────
Combinations
────────────────────────────────────────────────────────────────────────────────
Any terms can be combined with "+": e.g. "mse+wqrl+fl_hist" (recommended
default — balances standard fidelity with distribution-aware correction).

Per-variable physics weights (from normalizer.loss_weights) multiply every
loss term; variables with weight=0 (e.g. lsm) are excluded.

────────────────────────────────────────────────────────────────────────────────
Paper hyperparameters
────────────────────────────────────────────────────────────────────────────────
  n_bins      = 100
  fl_beta     = 0.2    (β in FL_MSE sigmoid)
  fl_gamma    = 1.0    (γ exponent in both focal losses)
  focal_scale = 100.0  (λ; focal terms are ~100× smaller than MSE without it)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor

from dssml.models.base import ModelBase
from dssml.networks.ae import IAutoEncoder
from dssml.networks.ae.qrl_ae import QRLAutoEncoder


# ── Per-channel loss helpers ──────────────────────────────────────────────────

def _weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
    """Weighted mean over channel dim (C,); falls back to plain mean."""
    if weights is None or weights.sum() < 1e-8:
        return values.mean()
    return (values * weights).sum() / weights.sum()


def _mse_per_channel(x: Tensor, y: Tensor) -> Tensor:
    """Standard MSE per channel, averaged over batch and spatial dims.
    x, y : (B, C, H, W) — returns (C,)
    """
    return (x - y).square().mean(dim=[0, 2, 3])


def _qrl_per_channel(x: Tensor, y: Tensor, n_bins: int) -> Tensor:
    """
    Quantized Reconstruction Loss (Approach 1, variant 1).

    For each channel independently, bin the *target* field into n_bins equal-
    width bins, compute MSE within each bin, then average uniformly across
    non-empty bins.  Histogram boundaries are recomputed per batch so the
    full dynamic range of each mini-batch is always covered.

    x, y : (B, C, H, W) — normalised target and reconstruction
    returns : (C,) — mean QRL loss per channel
    """
    B, C, H, W = x.shape
    S = H * W
    BC = B * C

    xf = x.contiguous().view(BC, S)   # (BC, S)
    yf = y.contiguous().view(BC, S)

    # Per-(batch,channel) dynamic bin boundaries
    x_min = xf.amin(dim=1, keepdim=True)    # (BC, 1)
    x_max = xf.amax(dim=1, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-8)

    bin_idx = ((xf - x_min) / scale * n_bins).long().clamp(0, n_bins - 1)  # (BC, S)

    sq_err = (xf - yf).square()
    ones = torch.ones_like(sq_err)

    # Aggregate per bin via scatter_add — fully vectorised, no Python loops
    bin_sum = torch.zeros(BC, n_bins, device=x.device, dtype=x.dtype).scatter_add(1, bin_idx, sq_err)
    bin_cnt = torch.zeros(BC, n_bins, device=x.device, dtype=x.dtype).scatter_add(1, bin_idx, ones)

    non_empty = bin_cnt > 0
    bin_mse = torch.where(non_empty, bin_sum / bin_cnt.clamp(min=1.0), torch.zeros_like(bin_sum))
    n_non_empty = non_empty.float().sum(dim=1).clamp(min=1.0)   # (BC,)

    loss_per_bc = bin_mse.sum(dim=1) / n_non_empty   # (BC,) — uniform average over bins

    return loss_per_bc.view(B, C).mean(dim=0)   # (C,) — average over batch


def _wqrl_per_channel(x: Tensor, y: Tensor, n_bins: int) -> Tensor:
    """
    Weighted Quantized Reconstruction Loss (Approach 1, variant 2).

    Same binning as QRL, but each bin is weighted by ω_j = 1 − h_j, where
    h_j is the normalised bin frequency.  Rare value ranges (small h_j) get
    weight ≈ 1; common ranges (h_j ≈ 1) are down-weighted toward zero.

    x, y : (B, C, H, W) — returns (C,)
    """
    B, C, H, W = x.shape
    S = H * W
    BC = B * C

    xf = x.contiguous().view(BC, S)
    yf = y.contiguous().view(BC, S)

    x_min = xf.amin(dim=1, keepdim=True)
    x_max = xf.amax(dim=1, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-8)
    bin_idx = ((xf - x_min) / scale * n_bins).long().clamp(0, n_bins - 1)

    sq_err = (xf - yf).square()
    ones = torch.ones_like(sq_err)

    bin_sum = torch.zeros(BC, n_bins, device=x.device, dtype=x.dtype).scatter_add(1, bin_idx, sq_err)
    bin_cnt = torch.zeros(BC, n_bins, device=x.device, dtype=x.dtype).scatter_add(1, bin_idx, ones)

    # h_j = normalised frequency; ω_j = 1 − h_j
    total = bin_cnt.sum(dim=1, keepdim=True).clamp(min=1.0)
    h_norm = bin_cnt / total       # (BC, n_bins) ∈ [0, 1]
    omega = 1.0 - h_norm           # (BC, n_bins); rare bins → 1, common bins → 0

    non_empty = bin_cnt > 0
    bin_mse = torch.where(non_empty, bin_sum / bin_cnt.clamp(min=1.0), torch.zeros_like(bin_sum))

    # Weighted average: Σ(ω_j · MSE_j) / Σ(ω_j) over non-empty bins
    weighted_sum = torch.where(non_empty, bin_mse * omega, torch.zeros_like(bin_mse))
    omega_valid = torch.where(non_empty, omega, torch.zeros_like(omega)).sum(dim=1).clamp(min=1e-8)
    loss_per_bc = weighted_sum.sum(dim=1) / omega_valid

    return loss_per_bc.view(B, C).mean(dim=0)   # (C,)


def _fl_mse_per_channel(x: Tensor, y: Tensor, beta: float, gamma: float) -> Tensor:
    """
    Reconstruction Focal Loss with sigmoid modulation (Approach 2, FL_MSE).

    FL_MSE(x, x̂) = σ(β |x − x̂|)^γ · (x − x̂)²

    High-error pixels receive a larger modulating factor (σ → 1 as error
    grows), focusing training effort on hard-to-reconstruct samples.

    β = 0.2 controls how quickly the sigmoid saturates with error magnitude.
    γ = 1.0 is the focusing exponent.

    x, y : (B, C, H, W) — returns (C,)
    """
    diff = x - y
    modulate = torch.sigmoid(beta * diff.abs()).pow(gamma)  # ∈ (0, 1)
    return (modulate * diff.square()).mean(dim=[0, 2, 3])   # (C,)


def _fl_hist_per_channel(x: Tensor, y: Tensor, n_bins: int, gamma: float) -> Tensor:
    """
    Histogram-based Reconstruction Focal Loss (Approach 2, FL_hist).

    FL_hist(x, x̂) = ω_j(x)^γ · (x − x̂)²

    where ω_j = 1 − h_j is the inverse normalised frequency of the bin
    containing the *target* pixel value x.  Pixels whose target values are
    rare in the training batch receive a stronger gradient signal.

    x, y : (B, C, H, W) — returns (C,)
    """
    B, C, H, W = x.shape
    S = H * W
    BC = B * C

    xf = x.contiguous().view(BC, S)
    yf = y.contiguous().view(BC, S)

    x_min = xf.amin(dim=1, keepdim=True)
    x_max = xf.amax(dim=1, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-8)
    bin_idx = ((xf - x_min) / scale * n_bins).long().clamp(0, n_bins - 1)

    # Build histogram
    ones = torch.ones_like(xf)
    bin_cnt = torch.zeros(BC, n_bins, device=x.device, dtype=x.dtype).scatter_add(1, bin_idx, ones)
    total = bin_cnt.sum(dim=1, keepdim=True).clamp(min=1.0)
    h_norm = bin_cnt / total           # (BC, n_bins) — normalised frequency
    omega_per_bin = 1.0 - h_norm       # (BC, n_bins) — rare bins get ≈ 1

    # Look up per-pixel weight via the bin index
    omega_per_pixel = omega_per_bin.gather(1, bin_idx)     # (BC, S)

    sq_err = (xf - yf).square()
    focal = (omega_per_pixel.pow(gamma) * sq_err).mean(dim=1)   # (BC,)

    return focal.view(B, C).mean(dim=0)   # (C,)


# ── Model ─────────────────────────────────────────────────────────────────────

class QRLDCAEModel(ModelBase):
    """
    Lightning wrapper for QRLAutoEncoder.

    Trains the encoder and decoder using distribution-aware reconstruction
    losses: Quantized Reconstruction Losses (Approach 1) and Reconstruction
    Focal Losses (Approach 2) from the paper cited in the module docstring.

    All reconstruction losses operate in the normalised input space (after
    MultiNormalizer) and are additionally weighted per variable using
    MultiNormalizer.loss_weights.

    Arguments
    ---------
    optimizers_conf : list[dict]
        List of optimizer / scheduler dicts (same format as all other models).
    ae_conf : dict
        Hydra config for QRLAutoEncoder
        (_target_: dssml.networks.ae.QRLAutoEncoder).
    loss_type : str
        One or more loss terms joined by "+".  Valid terms:
        "mse", "qrl", "wqrl", "fl_mse", "fl_hist".
        Default: "mse+wqrl+fl_hist".
    mse_weight : float
        Coefficient for the plain MSE term.
    qrl_weight : float
        Coefficient for the QRL term (Approach 1, uniform bins).
    wqrl_weight : float
        Coefficient for the WQRL term (Approach 1, rarity-weighted).
    fl_mse_weight : float
        Coefficient for FL_MSE (before focal_scale).
    fl_hist_weight : float
        Coefficient for FL_hist (before focal_scale).
    n_bins : int
        Number of histogram bins per channel per batch (paper: 100).
    fl_beta : float
        β in FL_MSE sigmoid modulation (paper: 0.2).
    fl_gamma : float
        γ exponent for both focal losses (paper: 1.0).
    focal_scale : float
        Global scale λ applied to focal loss terms.
        Without scaling, focal losses are ~100× smaller than MSE (paper: 100.0).
    noise_std : float
        Std-dev of Gaussian noise added to z during *training only* to smooth
        the latent manifold for downstream diffusion training (0 = disabled).
    normalizer : nn.Module | None
        Injected by DefaultTrainer; must expose a .loss_weights buffer (V,).
    """

    def __init__(
        self,
        optimizers_conf: list[dict],
        ae_conf: dict,
        loss_type: str = "mse+wqrl+fl_hist",
        mse_weight: float = 1.0,
        qrl_weight: float = 1.0,
        wqrl_weight: float = 1.0,
        fl_mse_weight: float = 1.0,
        fl_hist_weight: float = 1.0,
        n_bins: int = 100,
        fl_beta: float = 0.2,
        fl_gamma: float = 1.0,
        focal_scale: float = 100.0,
        noise_std: float = 0.0,
        normalizer: nn.Module | None = None,
    ):
        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        self.save_hyperparameters(ignore=["normalizer"])

        self.loss_type = loss_type
        self.mse_weight = mse_weight
        self.qrl_weight = qrl_weight
        self.wqrl_weight = wqrl_weight
        self.fl_mse_weight = fl_mse_weight
        self.fl_hist_weight = fl_hist_weight
        self.n_bins = n_bins
        self.fl_beta = fl_beta
        self.fl_gamma = fl_gamma
        self.focal_scale = focal_scale
        self.noise_std = noise_std

        # Parse once at init for fast forward passes
        terms = set(loss_type.lower().replace("+", " ").split())
        self._has_mse     = bool(terms & {"mse", "l2"})
        self._has_qrl     = "qrl"     in terms
        self._has_wqrl    = "wqrl"    in terms
        self._has_fl_mse  = "fl_mse"  in terms
        self._has_fl_hist = "fl_hist" in terms

        if not (self._has_mse or self._has_qrl or self._has_wqrl or
                self._has_fl_mse or self._has_fl_hist):
            raise ValueError(
                f"loss_type='{loss_type}' contains no recognised terms. "
                "Valid: mse, l2, qrl, wqrl, fl_mse, fl_hist"
            )

        self.auto_encoder: QRLAutoEncoder = instantiate(ae_conf, _recursive_=False)
        if not isinstance(self.auto_encoder, IAutoEncoder):
            raise ValueError(
                f"ae_conf must instantiate an IAutoEncoder, got {type(self.auto_encoder)}"
            )

    @property
    def name(self) -> str:
        return "QRLDCAEModel"

    # ── Weight helpers ────────────────────────────────────────────────────────

    def _expand_weights(self, V: int, T: int, E: int) -> Tensor | None:
        """
        Per-channel physics weights of shape (T*V*E,) for the flattened
        channel dim _x = x.view(B, T*V*E, H, W).

        Variable weights broadcast: each (t, v, e) slot gets weight w[v].
        Variables with loss_weight=null have weight 0 and contribute nothing
        to the weighted mean.
        """
        if self.normalizer is None or not hasattr(self.normalizer, "loss_weights"):
            return None
        w = self.normalizer.loss_weights   # (V,)
        if T == 1 and E == 1:
            return w
        w_ve = w.unsqueeze(1).expand(V, E).reshape(V * E)
        return w_ve.repeat(T)   # (T*V*E,)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        x_hat: Tensor,
        x_target: Tensor,
        w: Tensor | None,
        prefix: str,
    ) -> Tensor:
        """
        x_hat, x_target : (B, C, H, W) — normalised space, C = T*V*E
        w               : (C,) physics weights, or None
        """
        total = x_hat.new_zeros(1).squeeze()

        if self._has_mse:
            mse_c = _mse_per_channel(x_target, x_hat)
            mse = _weighted_mean(mse_c, w)
            self.log(f"{prefix}_mse", mse, on_epoch=True, sync_dist=True)
            total = total + self.mse_weight * mse

        if self._has_qrl:
            qrl_c = _qrl_per_channel(x_target, x_hat, self.n_bins)
            qrl = _weighted_mean(qrl_c, w)
            self.log(f"{prefix}_qrl", qrl, on_epoch=True, sync_dist=True)
            total = total + self.qrl_weight * qrl

        if self._has_wqrl:
            wqrl_c = _wqrl_per_channel(x_target, x_hat, self.n_bins)
            wqrl = _weighted_mean(wqrl_c, w)
            self.log(f"{prefix}_wqrl", wqrl, on_epoch=True, sync_dist=True)
            total = total + self.wqrl_weight * wqrl

        if self._has_fl_mse:
            fl_mse_c = _fl_mse_per_channel(x_target, x_hat, self.fl_beta, self.fl_gamma)
            fl_mse = _weighted_mean(fl_mse_c, w)
            self.log(f"{prefix}_fl_mse", fl_mse, on_epoch=True, sync_dist=True)
            total = total + self.focal_scale * self.fl_mse_weight * fl_mse

        if self._has_fl_hist:
            fl_hist_c = _fl_hist_per_channel(x_target, x_hat, self.n_bins, self.fl_gamma)
            fl_hist = _weighted_mean(fl_hist_c, w)
            self.log(f"{prefix}_fl_hist", fl_hist, on_epoch=True, sync_dist=True)
            total = total + self.focal_scale * self.fl_hist_weight * fl_hist

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
