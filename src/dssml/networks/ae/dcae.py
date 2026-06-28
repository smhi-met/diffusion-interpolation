"""
Deep Compression AutoEncoder (DCAE) for weather fields.

Architecture follows:
    Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
    Chen et al., 2024  https://arxiv.org/abs/2410.10733
    Code: lola/efficientvit/efficientvit/models/efficientvit/dc_ae.py

Two modes are selected by the ``residual_autoencoding`` flag:

  residual_autoencoding=False  (default — backward-compatible):
    Original implementation.  Pixel_unshuffle then Conv at the start of each
    encoder level, Conv then pixel_shuffle at the end of each decoder level.
    No residual shortcuts around the spatial re-sampling operations.

  residual_autoencoding=True  (paper-correct, recommended for new runs):
    Implements DC-AE §3.1 "Residual Autoencoding":
      Encoder: project_in → [ResBlocks → [Attn] → ResidualDownsample]ₙ₋₁ → project_out
      Decoder: project_in → [ResidualUpsample → ResBlocks → [Attn]]ₙ₋₁ → project_out
    Key changes vs the old path:
      1. Conv(in → out//factor²) FIRST, then pixel_unshuffle (paper §3.1 Fig 3)
      2. Residual shortcut around every down/upsample: main + channel-avg/dup bypass
      3. ResBlocks run at the current spatial scale BEFORE downsampling (encoder),
         or AFTER upsampling (decoder) — matching the paper's stage ordering
      4. Channel-averaging shortcut at encoder output (project_out)
      5. Channel-duplicating shortcut at decoder input (project_in)

Note: latent saturation z → z/√(1+z²/B²) is kept in both modes.  It is not
from the DC-AE paper (which trains on natural images with perceptual loss) but
is well-motivated for weather fields where no perceptual loss is used.

Default settings achieve 16× spatial compression:
    (B, V, 256, 256)  →  (B, 32, 16, 16)  via patch_size=2, stride=2, 4 levels
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import IAutoEncoder


# ─── Normalization ─────────────────────────────────────────────────────────────

class SpatialLayerNorm(nn.Module):
    """
    Affine-free layer norm over the channel dimension at each spatial position.

    For x with shape (B, C, H, W) this standardizes over the C axis at every
    (b, h, w) coordinate independently — equivalent to LOLA's LayerNorm(dim=1).
    No learnable affine parameters: the subsequent conv learns any scaling.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        var, mean = torch.var_mean(x, dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps)


# ─── Latent saturation ─────────────────────────────────────────────────────────

class LatentSaturation(nn.Module):
    """
    Soft-clips latent values to the open interval (−B, +B).

    Formula:  z  →  z / √(1 + z²/B²)

    Unlike tanh it is asymptotically linear near zero (gradient ≈ 1) and
    saturates smoothly for |z| ≫ B, avoiding vanishing gradients while
    bounding the latent range.  B=5 follows the LOLA default.
    """

    def __init__(self, bound: float = 5.0):
        super().__init__()
        self.bound = bound

    def forward(self, z: Tensor) -> Tensor:
        return z * torch.rsqrt(1.0 + torch.square(z / self.bound))


# ─── Conv helper with optional identity initialization ─────────────────────────

def _conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding: int = 1,
    *,
    identity_init: bool = False,
) -> nn.Conv2d:
    """
    Standard 2-D convolution with optional near-identity weight initialization.

    When identity_init=True: random weights scaled to 1e-2, then a
    pseudo-identity is added (output channel i connects to input channel
    i % in_channels at the kernel center).
    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    if identity_init:
        center = kernel_size // 2
        eye = torch.zeros_like(conv.weight.data)
        for i in range(out_channels):
            eye[i, i % in_channels, center, center] = 1.0
        conv.weight.data.mul_(1e-2)
        conv.weight.data.add_(eye)
    return conv


# ─── Residual block ────────────────────────────────────────────────────────────

class DCResBlock(nn.Module):
    """
    Pre-norm residual block: SpatialLayerNorm → Conv → SiLU → [Dropout] → Conv.

    The second convolution is initialized near zero so the block starts as the
    identity map and gradually learns corrections.
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm = SpatialLayerNorm()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2.weight.data.mul_(1e-2)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.act(self.conv1(h))
        h = self.drop(h)
        h = self.conv2(h)
        return x + h


# ─── Spatial self-attention block ─────────────────────────────────────────────

class SelfAttention2d(nn.Module):
    """
    Spatial self-attention for (B, C, H, W) tensors.

    Flattens spatial dims into a token sequence, applies multi-head attention,
    then reshapes back.  Pre-norm pattern, residual connection.
    """

    def __init__(self, channels: int, heads: int = 1):
        super().__init__()
        self.norm = SpatialLayerNorm()
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


# ─── Paper-style pixel-shuffle layers (DC-AE §3.1) ────────────────────────────

class ConvPixelUnshuffle(nn.Module):
    """
    Conv(in → out//factor²) → pixel_unshuffle(factor).  No residual shortcut.

    The conv REDUCES channels before the spatial packing, so its weight tensor
    is factor² times smaller than the naive (unshuffle-first) alternative.
    Used at the encoder input (project_in) where the shortcut is not feasible
    because in_channels (number of variables) need not divide out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        *,
        identity_init: bool = False,
    ):
        super().__init__()
        out_ratio = factor * factor
        assert out_channels % out_ratio == 0, (
            f"out_channels ({out_channels}) must be divisible by factor² ({out_ratio})"
        )
        self.factor = factor
        self.conv = _conv2d(in_channels, out_channels // out_ratio, 3, 1, identity_init=identity_init)

    def forward(self, x: Tensor) -> Tensor:
        return F.pixel_unshuffle(self.conv(x), self.factor)


class ConvPixelShuffle(nn.Module):
    """
    Conv(in → out*factor²) → pixel_shuffle(factor).  No residual shortcut.

    Used at the decoder output (project_out) where the shortcut is not feasible
    because out_channels (number of variables) need not divide in_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        *,
        identity_init: bool = False,
    ):
        super().__init__()
        self.factor = factor
        self.conv = _conv2d(in_channels, out_channels * factor * factor, 3, 1, identity_init=identity_init)

    def forward(self, x: Tensor) -> Tensor:
        return F.pixel_shuffle(self.conv(x), self.factor)


class ResidualConvPixelUnshuffle(nn.Module):
    """
    DC-AE §3.1 Residual Autoencoding — downsampling block.

    main:     Conv(in → out//factor²) → pixel_unshuffle(factor)
    shortcut: pixel_unshuffle(factor) → channel-average groups → out_channels
    output:   main(x) + shortcut(x)

    Constraint: in_channels * factor² must be divisible by out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2):
        super().__init__()
        out_ratio = factor * factor
        assert out_channels % out_ratio == 0, (
            f"out_channels ({out_channels}) must be divisible by factor² ({out_ratio})"
        )
        assert in_channels * out_ratio % out_channels == 0, (
            f"in_channels*factor² ({in_channels * out_ratio}) must be divisible by "
            f"out_channels ({out_channels}) for the channel-averaging shortcut"
        )
        self.factor = factor
        self.conv = nn.Conv2d(in_channels, out_channels // out_ratio, 3, padding=1, bias=True)
        self.shortcut_group_size = in_channels * out_ratio // out_channels

    def forward(self, x: Tensor) -> Tensor:
        main = F.pixel_unshuffle(self.conv(x), self.factor)
        sc = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = sc.shape
        sc = sc.view(B, -1, self.shortcut_group_size, H, W).mean(dim=2)
        return main + sc


class ResidualConvPixelShuffle(nn.Module):
    """
    DC-AE §3.1 Residual Autoencoding — upsampling block.

    main:     Conv(in → out*factor²) → pixel_shuffle(factor)
    shortcut: channel-duplicate in_channels → out*factor², then pixel_shuffle(factor)
    output:   main(x) + shortcut(x)

    Constraint: out_channels * factor² must be divisible by in_channels.
    """

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2):
        super().__init__()
        out_ratio = factor * factor
        assert out_channels * out_ratio % in_channels == 0, (
            f"out_channels*factor² ({out_channels * out_ratio}) must be divisible by "
            f"in_channels ({in_channels}) for the channel-duplicating shortcut"
        )
        self.factor = factor
        self.conv = nn.Conv2d(in_channels, out_channels * out_ratio, 3, padding=1, bias=True)
        self.shortcut_repeats = out_channels * out_ratio // in_channels

    def forward(self, x: Tensor) -> Tensor:
        main = F.pixel_shuffle(self.conv(x), self.factor)
        sc = x.repeat_interleave(self.shortcut_repeats, dim=1)
        sc = F.pixel_shuffle(sc, self.factor)
        return main + sc


class ChannelAveragingResidualConv(nn.Module):
    """
    Conv(in → out) + channel-averaging residual shortcut (factor=1 spatial).

    Used as the encoder project_out when residual_autoencoding=True.
    Constraint: in_channels must be divisible by out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        identity_init: bool = False,
    ):
        super().__init__()
        assert in_channels % out_channels == 0, (
            f"in_channels ({in_channels}) must be divisible by out_channels ({out_channels})"
        )
        self.conv = _conv2d(in_channels, out_channels, 3, 1, identity_init=identity_init)
        self.group_size = in_channels // out_channels

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        sc = x.view(B, -1, self.group_size, H, W).mean(dim=2)
        return self.conv(x) + sc


class ChannelDuplicatingResidualConv(nn.Module):
    """
    Conv(in → out) + channel-duplicating residual shortcut (factor=1 spatial).

    Used as the decoder project_in when residual_autoencoding=True.
    Constraint: out_channels must be divisible by in_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        identity_init: bool = False,
    ):
        super().__init__()
        assert out_channels % in_channels == 0, (
            f"out_channels ({out_channels}) must be divisible by in_channels ({in_channels})"
        )
        self.conv = _conv2d(in_channels, out_channels, 3, 1, identity_init=identity_init)
        self.repeats = out_channels // in_channels

    def forward(self, x: Tensor) -> Tensor:
        sc = x.repeat_interleave(self.repeats, dim=1)
        return self.conv(x) + sc


# ─── Encoder ──────────────────────────────────────────────────────────────────

class DCEncoder(nn.Module):
    """
    Hierarchical encoder using pixel-unshuffle (space-to-channel) downsampling.

    Spatial flow with default settings (patch_size=2, stride=2, 4 levels):
        (B, 19, 256, 256)  →  (B, 32, 16, 16)

    residual_autoencoding=False (default):
        pixel_unshuffle → Conv → ResBlocks  per level (old behaviour)

    residual_autoencoding=True (paper DC-AE §3.1):
        project_in (ConvPixelUnshuffle, no shortcut)
        → [ResBlocks → [Attn] → ResidualConvPixelUnshuffle]  for levels 0..N-2
        → [ResBlocks]  for level N-1 (deepest)
        → project_out (ChannelAveragingResidualConv)
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        hid_channels: Sequence[int] = (96, 192, 384, 384),
        hid_blocks: Sequence[int] = (2, 2, 2, 2),
        patch_size: int = 2,
        stride: int = 2,
        attn_levels: Sequence[int] = (2,),
        dropout: float = 0.0,
        identity_init: bool = True,
        residual_autoencoding: bool = False,
    ):
        super().__init__()
        assert len(hid_blocks) == len(hid_channels)
        n_levels = len(hid_blocks)
        self.patch_size = patch_size
        self.stride = stride
        self.residual_autoencoding = residual_autoencoding
        attn_set = set(attn_levels)

        if not residual_autoencoding:
            # ── Old behaviour (backward-compatible) ───────────────────────────
            self.levels = nn.ModuleList()
            for i in range(n_levels):
                blocks: list[nn.Module] = []
                if i == 0:
                    stem_in = in_channels * patch_size * patch_size
                    blocks.append(_conv2d(stem_in, hid_channels[0], 3, 1))
                else:
                    down_in = hid_channels[i - 1] * stride * stride
                    blocks.append(
                        _conv2d(down_in, hid_channels[i], 3, 1, identity_init=identity_init)
                    )
                for _ in range(hid_blocks[i]):
                    blocks.append(DCResBlock(hid_channels[i], dropout=dropout))
                if i in attn_set:
                    blocks.append(SelfAttention2d(hid_channels[i]))
                if i == n_levels - 1:
                    blocks.append(
                        _conv2d(hid_channels[i], latent_channels, 3, 1, identity_init=identity_init)
                    )
                self.levels.append(nn.ModuleList(blocks))

        else:
            # ── Paper-style: Residual Autoencoding (DC-AE §3.1) ──────────────
            # project_in: no residual shortcut — in_channels need not divide hid[0]
            self.project_in = ConvPixelUnshuffle(
                in_channels, hid_channels[0], factor=patch_size, identity_init=False
            )
            # levels: ResBlocks → [Attn] → [ResidualDownsample]
            self.levels = nn.ModuleList()
            for i in range(n_levels):
                blocks: list[nn.Module] = []
                for _ in range(hid_blocks[i]):
                    blocks.append(DCResBlock(hid_channels[i], dropout=dropout))
                if i in attn_set:
                    blocks.append(SelfAttention2d(hid_channels[i]))
                if i < n_levels - 1:
                    blocks.append(
                        ResidualConvPixelUnshuffle(hid_channels[i], hid_channels[i + 1], factor=stride)
                    )
                self.levels.append(nn.ModuleList(blocks))
            # project_out: channel-averaging residual shortcut (hid[-1] → latent)
            self.project_out = ChannelAveragingResidualConv(
                hid_channels[-1], latent_channels, identity_init=identity_init
            )

    def forward(self, x: Tensor) -> Tensor:
        if not self.residual_autoencoding:
            if self.patch_size > 1:
                x = F.pixel_unshuffle(x, self.patch_size)
            for i, blocks in enumerate(self.levels):
                if i > 0:
                    x = F.pixel_unshuffle(x, self.stride)
                for block in blocks:
                    x = block(x)
        else:
            x = self.project_in(x)
            for blocks in self.levels:
                for block in blocks:
                    x = block(x)
            x = self.project_out(x)
        return x


# ─── Decoder ──────────────────────────────────────────────────────────────────

class DCDecoder(nn.Module):
    """
    Hierarchical decoder using pixel-shuffle (channel-to-space) upsampling.

    Mirrors DCEncoder exactly.

    residual_autoencoding=False (default):
        ResBlocks → Conv → pixel_shuffle  per level (old behaviour)

    residual_autoencoding=True (paper DC-AE §3.1):
        project_in (ChannelDuplicatingResidualConv)
        → [ResBlocks]  for level N-1 (deepest, no upsample yet)
        → [ResidualConvPixelShuffle → ResBlocks → [Attn]]  for levels N-2..0
        → project_out (ConvPixelShuffle, no shortcut)
    """

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        hid_channels: Sequence[int] = (96, 192, 384, 384),
        hid_blocks: Sequence[int] = (2, 2, 2, 2),
        patch_size: int = 2,
        stride: int = 2,
        attn_levels: Sequence[int] = (2,),
        dropout: float = 0.0,
        identity_init: bool = True,
        residual_autoencoding: bool = False,
    ):
        super().__init__()
        assert len(hid_blocks) == len(hid_channels)
        n_levels = len(hid_blocks)
        self.patch_size = patch_size
        self.stride = stride
        self._n_levels = n_levels
        self.residual_autoencoding = residual_autoencoding
        attn_set = set(attn_levels)

        if not residual_autoencoding:
            # ── Old behaviour (backward-compatible) ───────────────────────────
            self.levels = nn.ModuleList()
            for i in reversed(range(n_levels)):
                blocks: list[nn.Module] = []
                if i == n_levels - 1:
                    blocks.append(
                        _conv2d(latent_channels, hid_channels[i], 3, 1, identity_init=identity_init)
                    )
                for _ in range(hid_blocks[i]):
                    blocks.append(DCResBlock(hid_channels[i], dropout=dropout))
                if i in attn_set:
                    blocks.append(SelfAttention2d(hid_channels[i]))
                if i > 0:
                    up_out = hid_channels[i - 1] * stride * stride
                    blocks.append(
                        _conv2d(hid_channels[i], up_out, 3, 1, identity_init=identity_init)
                    )
                else:
                    final_out = out_channels * patch_size * patch_size
                    blocks.append(_conv2d(hid_channels[0], final_out, 3, 1))
                self.levels.append(nn.ModuleList(blocks))

        else:
            # ── Paper-style: Residual Autoencoding (DC-AE §3.1) ──────────────
            # project_in: channel-duplicating residual shortcut (latent → hid[-1])
            self.project_in = ChannelDuplicatingResidualConv(
                latent_channels, hid_channels[-1], identity_init=identity_init
            )
            # levels built deepest-first (reversed order):
            #   deepest level (i=N-1): no upsample at start, just ResBlocks
            #   all others (i=N-2..0): ResidualUpsample first, then ResBlocks
            self.levels = nn.ModuleList()
            for i in reversed(range(n_levels)):
                blocks: list[nn.Module] = []
                if i < n_levels - 1:
                    # upsample at the START: from hid[i+1] (deeper) → hid[i] (shallower)
                    blocks.append(
                        ResidualConvPixelShuffle(hid_channels[i + 1], hid_channels[i], factor=stride)
                    )
                for _ in range(hid_blocks[i]):
                    blocks.append(DCResBlock(hid_channels[i], dropout=dropout))
                if i in attn_set:
                    blocks.append(SelfAttention2d(hid_channels[i]))
                self.levels.append(nn.ModuleList(blocks))
            # project_out: no shortcut — out_channels need not divide hid[0]
            self.project_out = ConvPixelShuffle(
                hid_channels[0], out_channels, factor=patch_size, identity_init=False
            )

    def forward(self, z: Tensor) -> Tensor:
        if not self.residual_autoencoding:
            x = z
            for idx, blocks in enumerate(self.levels):
                level_i = self._n_levels - 1 - idx
                for block in blocks:
                    x = block(x)
                if level_i > 0:
                    x = F.pixel_shuffle(x, self.stride)
                else:
                    x = F.pixel_shuffle(x, self.patch_size)
        else:
            x = self.project_in(z)
            for blocks in self.levels:
                for block in blocks:
                    x = block(x)
            x = self.project_out(x)
        return x


# ─── Full autoencoder ──────────────────────────────────────────────────────────

class DCAutoEncoder(nn.Module, IAutoEncoder):
    """
    Deterministic DCAE with soft latent saturation.

    Designed as the frozen first stage of a latent diffusion pipeline.
    Regularization is provided entirely by LatentSaturation (no KL divergence).

    Set ``residual_autoencoding=True`` to use the paper-correct DC-AE §3.1
    architecture.  The default (False) preserves the original behaviour.

    Interface (IAutoEncoder):
        encode(x) → z          deterministic, saturated
        decode(z) → x_hat      no activation on output
        forward(x) → (x_hat, z)
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
        attn_levels: Sequence[int] = (2,),
        dropout: float = 0.0,
        identity_init: bool = True,
        residual_autoencoding: bool = False,
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
        self.saturation = LatentSaturation(bound=saturation_bound)
        self.decoder = DCDecoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            **shared,
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.saturation(self.encoder(x))

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor):
        z = self.encode(x)
        return self.decoder(z), z
