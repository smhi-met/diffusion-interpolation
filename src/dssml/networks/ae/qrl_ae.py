"""
QRL-experiment autoencoder network.

Architecture: DCEncoder → LoLASaturation → DCDecoder

Designed for use with QRLDCAEModel (quantized reconstruction + focal losses).
Provides a distinct class from LoLAAutoEncoder so the QRL-experiment network
can evolve independently (different saturation, architecture tweaks, etc.)
while sharing the underlying DC-AE building blocks.

Reference:
    "Quantizing reconstruction losses for improving weather data synthesis"
    Nature Scientific Reports, 2024 (doi:10.1038/s41598-024-52773-2)
    Architecture adapted from DC-AE (arXiv:2410.10733) and LoLA (arXiv:2507.02608).
"""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from torch import Tensor

from .base import IAutoEncoder
from .dcae import DCDecoder, DCEncoder
from .lola_ae import LoLASaturation


class QRLAutoEncoder(nn.Module, IAutoEncoder):
    """
    Deterministic autoencoder for Quantized Reconstruction Loss experiments.

    Architecture: DCEncoder → LoLASaturation → DCDecoder

    Achieves 16× spatial compression by default:
        (B, V, 256, 256)  →  (B, latent_channels, 16, 16)
    via patch_size=2 (initial patchify) and 3 × stride=2 pixel-unshuffle downs.

    Latent regularisation is provided entirely by LoLASaturation (no KL
    divergence).  The reconstruction loss strategy is handled by QRLDCAEModel,
    which is the intended training wrapper for this network.

    Latent noise (training-time smoothing of the latent manifold for downstream
    diffusion) is kept in QRLDCAEModel, not here, so this network stays clean
    for downstream inference.

    Arguments
    ---------
    in_channels : int
        Number of input/output variable channels (V).
    latent_channels : int
        Depth of the latent representation (e.g. 32).
    hid_channels : Sequence[int]
        Feature-map widths at each encoder/decoder level.
    hid_blocks : Sequence[int]
        Number of DCResBlocks per level.
    patch_size : int
        Initial pixel-unshuffle factor (folds spatial pixels into channels).
    stride : int
        Per-level pixel-unshuffle downsampling factor.
    saturation_bound : float
        B for latent saturation; output values lie in (−B, +B).
    saturation_mode : str
        Saturation function; see LoLASaturation for valid values.
        Default "softclip2": z → z / √(1 + z²/B²).
    attn_levels : Sequence[int]
        Level indices at which to insert a spatial self-attention block.
    dropout : float
        Dropout probability inside DCResBlocks (0 = disabled).
    identity_init : bool
        Initialise down/up-sampling convolutions near identity for stable
        training of deeper stacks.
    residual_autoencoding : bool
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
