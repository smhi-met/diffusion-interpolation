"""
QRL Variational AutoEncoder network.

Architecture: DCEncoder (2×latent_ch) → split μ/logvar → LoLASaturation(μ) → DCDecoder

Variational extension of QRLAutoEncoder. The encoder outputs 2*latent_channels
(μ and logvar concatenated); LoLASaturation bounds μ only; the decoder is unchanged.

  encode(x)            → μ  (deterministic mean — use for downstream inference)
  encode_posterior(x)  → DiagonalGaussianDistribution  (use during training)
  forward(x)           → (x̂, posterior)

Reference:
    "Quantizing reconstruction losses for improving weather data synthesis"
    Nature Scientific Reports, 2024 (doi:10.1038/s41598-024-52773-2)
    VAE with MMD regularization (InfoVAE) + distribution-aware reconstruction losses.
    Architecture adapted from DC-AE (arXiv:2410.10733) and LoLA (arXiv:2507.02608).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .base import IAutoEncoder
from .dcae import DCDecoder, DCEncoder
from .distributions import DiagonalGaussianDistribution
from .lola_ae import LoLASaturation


class QRLVariationalAutoEncoder(nn.Module, IAutoEncoder):
    """
    Variational autoencoder for Quantized Reconstruction Loss experiments.

    Architecture: DCEncoder (2×lat_ch) → LoLASaturation(μ) → reparameterize → DCDecoder

    Extends QRLAutoEncoder by replacing the deterministic saturation bottleneck with a
    variational latent: the encoder outputs 2*latent_channels (μ ‖ logvar),
    LoLASaturation bounds μ, and reparameterization sampling enables KL/MMD training.

    At inference:  encode(x)            → saturated mean μ (deterministic)
    At training:   encode_posterior(x)  → DiagonalGaussianDistribution

    Arguments
    ---------
    in_channels : int
        Number of input/output variable channels (V).
    latent_channels : int
        Depth of the latent representation.  The encoder internally outputs
        2*latent_channels to produce (μ ‖ logvar) before splitting.
    hid_channels : Sequence[int]
        Feature-map widths at each encoder/decoder level.
    hid_blocks : Sequence[int]
        Number of DCResBlocks per level.
    patch_size : int
        Initial pixel-unshuffle factor.
    stride : int
        Per-level pixel-unshuffle downsampling factor.
    saturation_bound : float
        B for LoLASaturation on μ; output values lie in (−B, +B).
    saturation_mode : str
        Saturation function applied to μ; see LoLASaturation for valid values.
    attn_levels : Sequence[int]
        Level indices at which to insert a spatial self-attention block.
    dropout : float
        Dropout probability inside DCResBlocks (0 = disabled).
    identity_init : bool
        Near-identity init for down/up-sampling convolutions.
    residual_autoencoding : bool
        DC-AE §3.1 residual shortcuts around every spatial re-sample.
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

        # Encoder outputs (μ ‖ logvar): 2× latent_channels channels
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
        """Encode x → posterior q(z|x).  Saturates μ; returns distribution."""
        h = self.encoder(x)                           # (B, 2*lat_ch, H/r, W/r)
        mu, logvar = torch.chunk(h, 2, dim=1)         # (B, lat_ch, ...) each
        mu = self.saturation(mu)                      # bound μ to (−B, +B)
        return DiagonalGaussianDistribution(torch.cat([mu, logvar], dim=1))

    # ── IAutoEncoder interface ────────────────────────────────────────────────

    def encode(self, x: Tensor) -> Tensor:
        """Encode x → saturated mean μ.  Deterministic; use for inference."""
        return self.encode_posterior(x).mode()

    def decode(self, z: Tensor) -> Tensor:
        """Decode z → reconstruction x̂.  No output activation."""
        return self.decoder(z)

    def forward(self, x: Tensor):
        """Full pass; returns (x̂, posterior)."""
        posterior = self.encode_posterior(x)
        return self.decoder(posterior.sample()), posterior
