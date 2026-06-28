"""
Rombach-style first-stage autoencoder (LDM paper, CVPR 2022).

Reference:
    High-Resolution Image Synthesis with Latent Diffusion Models
    Rombach et al., CVPR 2022  https://arxiv.org/abs/2112.10752

Supports both regularization modes with a single class:
  - regulation='kl': KL-regularized VAE (continuous latent, DiagonalGaussian posterior)
  - regulation='vq': VQ-regularized (discrete codebook, straight-through estimator)

Encoder/Decoder are the same UNet-style blocks used in LDMVariationalAutoEncoder
(GroupNorm ResBlocks + spatial self-attention). Bottleneck 1×1 conv layers
(quant_conv / post_quant_conv) decouple the encoder's channel count from the
latent dimension, as in the original paper.

Recommended first-stage configuration (paper Table 1, best trade-off):
  - f=4  →  KL z_channels=4, R-FID 0.27
  - f=8  →  KL z_channels=4, R-FID 0.90  ← default (good for weather at 256px)
  - VQ   →  n_embed=16384, embed_dim=4   (for either f)

The decoder uses no output activation (tanh_out=False) because weather inputs
are z-score normalised and span values well outside (−1, +1).

encode() return types (depends on regulation):
  KL  →  DiagonalGaussian   (use .sample() / .mode() / .kl())
  VQ  →  (z_q, codebook_loss, indices)   (z_q is the straight-through quantised z)

get_latent(x, deterministic) provides a unified Tensor interface for both modes,
intended for the downstream diffusion model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import IAutoEncoder
from .ldm_vae import LDMEncoder, LDMDecoder, DiagonalGaussian


# ─── Vector Quantizer ─────────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """
    Straight-through vector quantizer (VQ-VAE style).

    Codebook entries are learned via backprop (not EMA).  Two loss terms:

      codebook_loss = MSE(sg[z_e], z_q)   ← moves codebook towards encoder output
      commitment_loss = beta * MSE(z_e, sg[z_q])  ← moves encoder towards codebook

    where sg = stop-gradient (.detach()).

    Straight-through estimator: gradients flow through the quantisation step
    as if it were the identity (z_q = z_e + (z_q - z_e).detach()).

    Args:
        n_embed:   number of codebook entries (vocabulary size)
        embed_dim: dimension of each codebook entry (= latent channel depth)
        beta:      commitment loss weight (paper default 0.25)
    """

    def __init__(self, n_embed: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta

        self.embedding = nn.Embedding(n_embed, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / n_embed, 1.0 / n_embed)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            z: (B, embed_dim, H, W)

        Returns:
            z_q:           (B, embed_dim, H, W) — quantised, straight-through
            codebook_loss: scalar — commitment + codebook update loss
            indices:       (B, H, W) — nearest codebook index per spatial position
        """
        B, C, H, W = z.shape
        # (B, H, W, C) for distance computation
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (BHW, C)

        # L2 distances to all codebook entries: ||z - e||^2
        # = ||z||^2 + ||e||^2 - 2 z·e^T
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1)
            - 2.0 * z_flat @ self.embedding.weight.T
        )  # (BHW, n_embed)

        indices_flat = dist.argmin(1)                         # (BHW,)
        z_q = self.embedding(indices_flat)                    # (BHW, C)
        z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2)      # (B, C, H, W)
        indices = indices_flat.view(B, H, W)                  # (B, H, W)

        # VQ loss: codebook update + commitment
        codebook_loss = (
            F.mse_loss(z_q, z.detach())
            + self.beta * F.mse_loss(z_q.detach(), z)
        )

        # Straight-through estimator: preserve z gradients
        z_q = z + (z_q - z).detach()

        return z_q, codebook_loss, indices

    def get_codebook_entry(self, indices: Tensor, shape: tuple | None = None) -> Tensor:
        """Decode flat indices back to quantised embeddings."""
        z_q = self.embedding(indices)
        if shape is not None:
            B, H, W, C = shape
            z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2)
        return z_q


# ─── Rombach Autoencoder ──────────────────────────────────────────────────────

class RombachAutoEncoder(nn.Module, IAutoEncoder):
    """
    LDM first-stage autoencoder supporting KL and VQ regularization.

    Architecture (default f=8, ch_mult=(1,2,4,4)):
        Input  (B, in_channels, 256, 256)
        Encode → (B, 512, 32, 32) bottleneck
        quant_conv → latent moments / pre-quantisation features
        KL: DiagonalGaussian → sample/mode → (B, z_channels, 32, 32)
        VQ: VectorQuantizer  → z_q         → (B, embed_dim, 32, 32)
        post_quant_conv → (B, z_channels, 32, 32)
        Decode → (B, in_channels, 256, 256)

    Args:
        in_channels:       number of input/output channels (19 for MEPS variables)
        z_channels:        latent depth (4 recommended for f=8, per paper Table 1)
        embed_dim:         codebook/bottleneck dimension; defaults to z_channels
        ch:                base channel width (128 default)
        ch_mult:           channel multipliers per resolution level
                           (1,2,4,4) gives f=8; (1,2,4,4,8) gives f=16
        num_res_blocks:    ResnetBlocks per resolution level
        attn_resolutions:  spatial resolutions at which attention blocks are added
                           (32,) = attention at 32×32 bottleneck for f=8
        resolution:        input spatial resolution (H = W)
        dropout:           dropout rate
        resamp_with_conv:  use learned conv for up/downsampling (True = paper default)
        regulation:        'kl' (continuous VAE) or 'vq' (discrete codebook)
        n_embed:           VQ codebook size (16384 per paper Table 1 best VQ f=8)
        vq_beta:           VQ commitment loss coefficient (0.25 = paper default)

    encode() return types:
        regulation='kl' → DiagonalGaussian  (has .sample(), .mode(), .kl())
        regulation='vq' → (z_q, codebook_loss, indices)

    Use get_latent(x, deterministic) for a unified Tensor interface.
    """

    def __init__(
        self,
        in_channels: int,
        z_channels: int = 4,
        embed_dim: int | None = None,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (32,),
        resolution: int = 256,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        regulation: str = "kl",
        n_embed: int = 16384,
        vq_beta: float = 0.25,
    ):
        super().__init__()

        if regulation not in ("kl", "vq"):
            raise ValueError(f"regulation must be 'kl' or 'vq', got '{regulation}'")

        self.regulation = regulation
        self.z_channels = z_channels
        self.embed_dim = embed_dim if embed_dim is not None else z_channels

        # ── Encoder ──────────────────────────────────────────────────────────
        # KL mode: encoder outputs 2*z_channels (mean + logvar)
        # VQ mode: encoder outputs z_channels
        self.encoder = LDMEncoder(
            in_channels=in_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            resolution=resolution,
            dropout=dropout,
            double_z=(regulation == "kl"),
            resamp_with_conv=resamp_with_conv,
        )

        # ── Bottleneck projection ─────────────────────────────────────────────
        if regulation == "kl":
            # 2*z_channels → 2*embed_dim  (maps encoder moments to latent moments)
            self.quant_conv = nn.Conv2d(2 * z_channels, 2 * self.embed_dim, 1)
            # embed_dim → z_channels  (maps sampled z back to decoder channel count)
            self.post_quant_conv = nn.Conv2d(self.embed_dim, z_channels, 1)

        else:  # vq
            # z_channels → embed_dim  (map pre-quant features to codebook space)
            self.quant_conv = nn.Conv2d(z_channels, self.embed_dim, 1)
            self.quantizer = VectorQuantizer(n_embed, self.embed_dim, beta=vq_beta)
            # embed_dim → z_channels
            self.post_quant_conv = nn.Conv2d(self.embed_dim, z_channels, 1)

        # ── Decoder ──────────────────────────────────────────────────────────
        # tanh_out=False: weather data is z-score normalized, not [-1,1]
        self.decoder = LDMDecoder(
            out_channels=in_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            resolution=resolution,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            tanh_out=False,
        )

    # ── Encode ────────────────────────────────────────────────────────────────

    def encode(self, x: Tensor):
        """
        Encode input to latent representation.

        Returns:
            KL:  DiagonalGaussian  — call .sample() for stochastic z, .mode() for mean
            VQ:  (z_q, codebook_loss, indices)  — z_q is straight-through quantised
        """
        h = self.encoder(x)
        h = self.quant_conv(h)

        if self.regulation == "kl":
            return DiagonalGaussian(h)

        # VQ: quantise and return codebook loss
        z_q, codebook_loss, indices = self.quantizer(h)
        return z_q, codebook_loss, indices

    # ── Decode ────────────────────────────────────────────────────────────────

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent tensor z → reconstruction."""
        z = self.post_quant_conv(z)
        return self.decoder(z)

    # ── Unified latent interface (for downstream diffusion model) ─────────────

    def get_latent(self, x: Tensor, deterministic: bool = True) -> Tensor:
        """
        Encode x and return a plain latent Tensor — unified for KL and VQ modes.

        Args:
            x:             input tensor (B, C, H, W)
            deterministic: KL only — True = posterior mean (mode()), False = sample

        Returns:
            z: (B, embed_dim, H, W) latent tensor
        """
        if self.regulation == "kl":
            posterior = self.encode(x)
            return posterior.mode() if deterministic else posterior.sample()
        else:
            z_q, _, _ = self.encode(x)
            return z_q

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor, sample_posterior: bool = True):
        """
        Full encode → decode pass.

        Returns:
            KL:  (x_hat, DiagonalGaussian posterior)
            VQ:  (x_hat, codebook_loss)
        """
        if self.regulation == "kl":
            posterior = self.encode(x)
            z = posterior.sample() if sample_posterior else posterior.mode()
            return self.decode(z), posterior
        else:
            z_q, codebook_loss, _ = self.encode(x)
            return self.decode(z_q), codebook_loss
