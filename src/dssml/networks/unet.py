"""
2D UNet backbone for latent-space diffusion models.

Designed for small spatial resolutions (16×16 – 32×32) typical of DC-AE or
LDM latent codes.  Time/noise conditioning uses AdaGN (adaptive group norm):
each GroupNorm scale/shift is predicted from the noise embedding.

Public API
----------
LatentUNet(in_ch, out_ch, base_channels, channel_mult, num_res_blocks,
           attn_levels, dropout, n_groups, n_heads)

`emb_dim` is exposed so the caller can build a matching Fourier MLP.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_groups(channels: int, n_groups: int = 32) -> int:
    """Largest power-of-two ≤ n_groups that divides channels."""
    g = n_groups
    while g > 1 and channels % g != 0:
        g //= 2
    return max(g, 1)


# ---------------------------------------------------------------------------
# Fourier feature embedding (noise level → dense vector)
# ---------------------------------------------------------------------------

def fourier_embed(x: Tensor, dim: int = 64) -> Tensor:
    """
    Deterministic Fourier features for scalar noise levels.

    x       : (B,)
    returns : (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), half,
                       device=x.device, dtype=x.dtype)
    )
    angles = x.view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# ---------------------------------------------------------------------------
# AdaGN: GroupNorm whose scale/shift are predicted from a time embedding
# ---------------------------------------------------------------------------

class AdaGroupNorm(nn.Module):
    """
    Adaptive Group Norm (Dhariwal & Nichol 2021).

    emb → Linear(2*channels) → split scale, shift
    output = norm(x) * (1 + scale) + shift
    """

    def __init__(self, channels: int, emb_dim: int, n_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(_valid_groups(channels, n_groups), channels,
                                 affine=False)
        self.proj = nn.Linear(emb_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        scale, shift = self.proj(emb).chunk(2, dim=1)       # (B, C) each
        return (self.norm(x)
                * (1.0 + scale[:, :, None, None])
                + shift[:, :, None, None])


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Pre-norm residual block with AdaGN time conditioning.

    AdaGN + SiLU → Conv 3×3
    AdaGN + SiLU → Dropout → Conv 3×3 (zero-init)
    residual (1×1 conv if in_ch ≠ out_ch)
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int,
                 dropout: float = 0.0, n_groups: int = 32):
        super().__init__()
        self.norm1 = AdaGroupNorm(in_ch,  emb_dim, n_groups)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2 = AdaGroupNorm(out_ch, emb_dim, n_groups)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x, emb)))
        h = self.conv2(self.drop(F.silu(self.norm2(h, emb))))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Self-attention block
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """
    Pre-norm spatial multi-head self-attention.  Input: (B, C, H, W).

    Residual + zero-init output projection.
    """

    def __init__(self, channels: int, n_heads: int = 4, n_groups: int = 32):
        super().__init__()
        assert channels % n_heads == 0, (
            f"AttentionBlock: channels ({channels}) must be divisible by "
            f"n_heads ({n_heads})"
        )
        self.norm     = nn.GroupNorm(_valid_groups(channels, n_groups), channels)
        self.n_heads  = n_heads
        self.head_dim = channels // n_heads
        self.qkv  = nn.Conv1d(channels, 3 * channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W)         # (B, C, S)
        qkv = self.qkv(h)                              # (B, 3C, S)
        q, k, v = qkv.chunk(3, dim=1)

        def _heads(t):
            return t.reshape(B, self.n_heads, self.head_dim, H * W)

        q, k, v = _heads(q), _heads(k), _heads(v)
        scale = self.head_dim ** -0.5
        # (B, n_heads, S, S) via einsum
        attn = torch.einsum("bhds,bhdS->bhsS", q * scale, k).softmax(dim=-1)
        out  = torch.einsum("bhsS,bhdS->bhds", attn, v)   # (B, nh, hd, S)
        out  = self.proj(out.reshape(B, C, H * W)).reshape(B, C, H, W)
        return x + out


# ---------------------------------------------------------------------------
# Down / Upsample
# ---------------------------------------------------------------------------

class Downsample2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ---------------------------------------------------------------------------
# LatentUNet
# ---------------------------------------------------------------------------

class LatentUNet(nn.Module):
    """
    2D UNet for latent-space temporal interpolation.

    Parameters
    ----------
    in_ch : int
        Input channels = cond_channels + noisy_target_channels.
        For boundary-conditioned interpolation: (2 + n_interp) × C_z.
    out_ch : int
        Output channels = target_channels = n_interp × C_z.
    base_channels : int
        Feature width at the finest resolution (before channel multipliers).
    channel_mult : Sequence[int]
        Channel multiplier per encoder level (coarse = last entry).
        E.g. (1, 2, 4) → widths [base, 2*base, 4*base].
    num_res_blocks : int
        ResBlocks per encoder / decoder level.  Each produces one skip tensor.
    attn_levels : Sequence[int] | None
        Level indices (0 = finest) at which to add self-attention.
        None → attention at every level (recommended for small latents).
    dropout : float
        Dropout probability in ResBlocks.
    n_groups : int
        GroupNorm target group count.
    n_heads : int
        Attention heads.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        base_channels: int = 128,
        channel_mult: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_levels: Sequence[int] | None = None,
        dropout: float = 0.0,
        n_groups: int = 32,
        n_heads: int = 4,
    ):
        super().__init__()

        n_levels = len(channel_mult)
        ch_list  = [base_channels * m for m in channel_mult]
        emb_dim  = base_channels * 4

        if attn_levels is None:
            attn_set = set(range(n_levels))
        else:
            attn_set = set(attn_levels)

        self.emb_dim = emb_dim   # exposed for caller's MLP

        # ── Input projection ─────────────────────────────────────────────
        self.input_proj = nn.Conv2d(in_ch, ch_list[0], 3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────
        # For each level i: num_res_blocks ResBlocks → optional Attention → Downsample
        # Each ResBlock saves one skip tensor.

        self.enc_blocks  = nn.ModuleList()   # enc_blocks[i] = ModuleList of blocks
        self.enc_downs   = nn.ModuleList()   # enc_downs[i]  = Downsample or Identity

        enc_skip_chs: list[int] = []         # ch_out per enc ResBlock, in order
        ch_prev = ch_list[0]

        for level_i, ch_out in enumerate(ch_list):
            level_mods = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_mods.append(ResBlock(ch_prev, ch_out, emb_dim, dropout, n_groups))
                enc_skip_chs.append(ch_out)
                ch_prev = ch_out
            if level_i in attn_set:
                level_mods.append(AttentionBlock(ch_out, n_heads, n_groups))
            self.enc_blocks.append(level_mods)
            self.enc_downs.append(
                Downsample2d(ch_out) if level_i < n_levels - 1 else nn.Identity()
            )

        # ── Bottleneck ────────────────────────────────────────────────────
        ch_bot = ch_list[-1]
        self.mid = nn.ModuleList([
            ResBlock(ch_bot, ch_bot, emb_dim, dropout, n_groups),
            AttentionBlock(ch_bot, n_heads, n_groups),
            ResBlock(ch_bot, ch_bot, emb_dim, dropout, n_groups),
        ])

        # ── Decoder ──────────────────────────────────────────────────────
        # Mirror of encoder: for each level i (0 = coarsest, n-1 = finest):
        #   • Upsample BEFORE blocks (Identity at coarsest = i=0)
        #   • num_res_blocks ResBlocks, each concatenates one encoder skip
        #   • optional Attention
        #
        # Skip consumption order: reversed enc_skip_chs (deepest first).

        dec_skip_chs = list(reversed(enc_skip_chs))
        skip_ptr  = 0
        ch_prev   = ch_bot      # output of bottleneck

        self.dec_ups    = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for level_i, ch_out in enumerate(reversed(ch_list)):
            self.dec_ups.append(
                Upsample2d(ch_prev) if level_i > 0 else nn.Identity()
            )
            level_mods = nn.ModuleList()
            for rb_j in range(num_res_blocks):
                skip_ch   = dec_skip_chs[skip_ptr]
                skip_ptr += 1
                ch_in_rb  = (ch_prev if rb_j == 0 else ch_out) + skip_ch
                level_mods.append(ResBlock(ch_in_rb, ch_out, emb_dim, dropout, n_groups))
            if (n_levels - 1 - level_i) in attn_set:
                level_mods.append(AttentionBlock(ch_out, n_heads, n_groups))
            self.dec_blocks.append(level_mods)
            ch_prev = ch_out

        # ── Output ───────────────────────────────────────────────────────
        ch_final = ch_list[0]
        self.out_norm = nn.GroupNorm(_valid_groups(ch_final, n_groups), ch_final)
        self.out_conv = nn.Conv2d(ch_final, out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        x   : (B, in_ch, H_z, W_z)   — conditioning cat noisy-target
        emb : (B, emb_dim)            — projected noise/time embedding
        """
        h = self.input_proj(x)

        # Encoder — collect skip tensors
        skips: list[Tensor] = []
        for blocks, down in zip(self.enc_blocks, self.enc_downs):
            for layer in blocks:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                    skips.append(h)
                else:
                    h = layer(h)      # AttentionBlock
            h = down(h)

        # Bottleneck
        for layer in self.mid:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)

        # Decoder — consume skips (LIFO)
        for up, blocks in zip(self.dec_ups, self.dec_blocks):
            h = up(h)
            for layer in blocks:
                if isinstance(layer, ResBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = layer(h, emb)
                else:
                    h = layer(h)

        return self.out_conv(F.silu(self.out_norm(h)))
