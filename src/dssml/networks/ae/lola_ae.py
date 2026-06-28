"""
LoLA-style deterministic autoencoder for weather fields.

Reference:
    Lost in Latent Space (arXiv:2507.02608)

Key design choices from the paper:
  - Latent saturation z → z / √(1 + z²/B²) replaces KL divergence.
  - Down/up-sampling convolutions initialised near identity.
  - Residual autoencoding shortcuts (DC-AE §3.1) for stable deep training.
  - No perceptual loss — variance-normalised reconstruction losses only.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .base import IAutoEncoder
from .dcae import DCDecoder, DCEncoder
from .distributions import DiagonalGaussianDistribution


class LoLASaturation(nn.Module):
    """
    Configurable latent-space saturation.

    Prevents unbounded growth of latent codes without the reconstruction-
    regularisation trade-off of KL divergence.

    Modes
    -----
    softclip2 (default, LoLA paper):
        z → z / √(1 + z²/B²)
        Asymptotically linear near zero (gradient ≈ 1), smoothly bounds
        the output to (−B, +B).  B=5 mimics a standard-Gaussian range.

    softclip:
        z → z / (1 + |z|/B)

    tanh:
        z → B·tanh(z/B)

    arcsinh:
        z → arcsinh(z)   — unbounded but sub-linear for large |z|

    rmsnorm:
        z → z / √(mean(z², dim=C) + ε)
        Normalises the whole latent map; no explicit bound.
    """

    def __init__(self, bound: float = 5.0, mode: str = "softclip2"):
        super().__init__()
        self.bound = bound
        self.mode = mode

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, bound={self.bound}"

    def forward(self, z: Tensor) -> Tensor:
        B = self.bound
        if self.mode == "softclip2":
            return z * torch.rsqrt(1.0 + torch.square(z / B))
        elif self.mode == "softclip":
            return z / (1.0 + z.abs() / B)
        elif self.mode == "tanh":
            return torch.tanh(z / B) * B
        elif self.mode == "arcsinh":
            return torch.arcsinh(z)
        elif self.mode == "rmsnorm":
            return z * torch.rsqrt(torch.mean(torch.square(z), dim=1, keepdim=True) + 1e-5)
        else:
            raise ValueError(
                f"Unknown saturation mode '{self.mode}'. "
                "Valid: softclip2, softclip, tanh, arcsinh, rmsnorm"
            )


class LoLAAutoEncoder(nn.Module, IAutoEncoder):
    """
    LoLA-style deterministic autoencoder.

    Architecture:  DCEncoder → LoLASaturation → DCDecoder

    Designed as the frozen first stage of a latent diffusion pipeline.
    Regularisation is provided entirely by LoLASaturation — no KL divergence.

    Default settings achieve 16× spatial compression:
        (B, V, 256, 256)  →  (B, latent_channels, 16, 16)
    via patch_size=2 (initial patchify) and 3 × stride=2 pixel-unshuffle downs.

    Latent noise (training-time regularisation to smooth the latent manifold)
    is intentionally kept in the model wrapper (LoLADCAEModel), not here,
    so this network can be used cleanly in downstream inference.

    Arguments
    ---------
    in_channels:
        Number of input/output variable channels V.
    latent_channels:
        Depth of the latent representation (e.g. 32).
    hid_channels:
        Feature-map widths at each encoder/decoder level.
    hid_blocks:
        Number of DCResBlocks per level.
    patch_size:
        Initial pixel-unshuffle factor (folds spatial pixels into channels).
    stride:
        Per-level pixel-unshuffle downsampling factor.
    saturation_bound:
        B for latent saturation; output values lie in (−B, +B).
    saturation_mode:
        Saturation function; see LoLASaturation for valid values.
    attn_levels:
        Level indices at which to insert a spatial self-attention block.
    dropout:
        Dropout probability inside DCResBlocks (0 = disabled).
    identity_init:
        Initialise down/up-sampling convolutions near identity.
        Enables stable training of deeper networks.
    residual_autoencoding:
        Use DC-AE §3.1 residual shortcuts around every spatial re-sample.
        Recommended for new experiments.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        hid_channels: Sequence[int] = (96, 192, 384, 384),
        hid_blocks: Sequence[int] = (2, 2, 2, 2),
        patch_size: int = 2,
        stride: int = 2,
        saturation_bound: float = 5.0,
        saturation_mode: str = "softclip2",
        attn_levels: Sequence[int] = (2,),
        dropout: float = 0.0,
        identity_init: bool = True,
        residual_autoencoding: bool = True,
    ):
        super().__init__()

        shared = dict(
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            patch_size=patch_size,
            stride=stride,
            attn_levels=attn_levels,
            dropout=dropout,
            identity_init=identity_init,
            residual_autoencoding=residual_autoencoding,
        )

        self.encoder = DCEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            **shared,
        )
        self.saturation = LoLASaturation(bound=saturation_bound, mode=saturation_mode)
        self.decoder = DCDecoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            **shared,
        )

    # ── IAutoEncoder interface ────────────────────────────────────────────────

    def encode(self, x: Tensor) -> Tensor:
        """Encode x → saturated latent z.  Shape: (B, latent_channels, H/r, W/r)."""
        return self.saturation(self.encoder(x))

    def decode(self, z: Tensor) -> Tensor:
        """Decode z → reconstruction x̂.  No output activation."""
        return self.decoder(z)

    def forward(self, x: Tensor):
        """Full pass; returns (x̂, z)."""
        z = self.encode(x)
        return self.decoder(z), z


class LoLAVariationalAutoEncoder(nn.Module, IAutoEncoder):
    """
    Variational extension of LoLAAutoEncoder.

    Architecture: DCEncoder (2×lat_ch) → LoLASaturation(μ) → reparameterize → DCDecoder

    The encoder outputs 2*latent_channels (μ ‖ logvar). LoLASaturation bounds μ only;
    logvar is left free. Reparameterization sampling enables KL/MMD training.

    At inference:  encode(x)            → saturated mean μ  (deterministic)
    At training:   encode_posterior(x)  → DiagonalGaussianDistribution

    Arguments — identical to LoLAAutoEncoder.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        hid_channels: Sequence[int] = (96, 192, 384, 384),
        hid_blocks: Sequence[int] = (2, 2, 2, 2),
        patch_size: int = 2,
        stride: int = 2,
        saturation_bound: float = 5.0,
        saturation_mode: str = "softclip2",
        attn_levels: Sequence[int] = (2,),
        dropout: float = 0.0,
        identity_init: bool = True,
        residual_autoencoding: bool = True,
    ):
        super().__init__()

        self.latent_channels = latent_channels

        shared = dict(
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            patch_size=patch_size,
            stride=stride,
            attn_levels=attn_levels,
            dropout=dropout,
            identity_init=identity_init,
            residual_autoencoding=residual_autoencoding,
        )

        # Encoder outputs (μ ‖ logvar): 2× latent_channels
        self.encoder = DCEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels * 2,
            **shared,
        )
        # Saturation applied to μ only — bounds the posterior mean
        self.saturation = LoLASaturation(bound=saturation_bound, mode=saturation_mode)
        self.decoder = DCDecoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            **shared,
        )

    # ── Posterior ─────────────────────────────────────────────────────────────

    def encode_posterior(self, x: Tensor) -> DiagonalGaussianDistribution:
        """Encode x → posterior q(z|x). Saturates μ; returns distribution."""
        h = self.encoder(x)                           # (B, 2*lat_ch, H/r, W/r)
        mu, logvar = torch.chunk(h, 2, dim=1)         # (B, lat_ch, ...) each
        mu = self.saturation(mu)                      # bound μ to (−B, +B)
        return DiagonalGaussianDistribution(torch.cat([mu, logvar], dim=1))

    # ── IAutoEncoder interface ────────────────────────────────────────────────

    def encode(self, x: Tensor) -> Tensor:
        """Encode x → saturated mean μ. Deterministic; use for inference."""
        return self.encode_posterior(x).mode()

    def decode(self, z: Tensor) -> Tensor:
        """Decode z → reconstruction x̂. No output activation."""
        return self.decoder(z)

    def forward(self, x: Tensor):
        """Full pass; returns (x̂, posterior)."""
        posterior = self.encode_posterior(x)
        return self.decoder(posterior.sample()), posterior
