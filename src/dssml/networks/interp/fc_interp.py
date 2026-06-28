"""
Fully-connected (per-pixel) latent interpolation network.

Operates entirely in the latent space of a frozen autoencoder.  Each spatial
position (h, w) is processed independently by the same shared MLP — equivalent
to a stack of 1×1 Conv2d layers.  No spatial mixing occurs; the network learns
channel-wise non-linear mappings from the two boundary latent frames to all
n_interp intermediate frames at once.

Input : (B, 2*C_z, H_z, W_z)   — cat(z_0, z_T)  boundary latents
Output: (B, n_interp*C_z, H_z, W_z) — cat(z_1, ..., z_{n_interp}) predicted

Architecture
------------
  Conv2d(2*C_z, hidden_ch, 1) → norm → act
  [Conv2d(hidden_ch, hidden_ch, 1) → norm → act] × (n_layers - 2)
  Conv2d(hidden_ch, n_interp*C_z, 1)

Each Conv2d(*, *, kernel_size=1) is mathematically identical to applying a
Linear layer to the channel dimension at every spatial location simultaneously.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class _Block(nn.Module):
    """Conv1×1 → GroupNorm → GELU with optional residual."""

    def __init__(self, in_ch: int, out_ch: int, n_groups: int = 16):
        super().__init__()
        n_groups = min(n_groups, out_ch)
        # ensure n_groups divides out_ch
        while out_ch % n_groups != 0 and n_groups > 1:
            n_groups -= 1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=(n_groups == 1))
        self.norm = nn.GroupNorm(n_groups, out_ch) if n_groups > 1 else nn.Identity()
        self.act  = nn.GELU()
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x))) + self.skip(x)


class LatentFCInterpolator(nn.Module):
    """
    Per-pixel MLP for latent temporal interpolation.

    Parameters
    ----------
    in_channels : int
        Number of input channels = 2 * C_z  (boundary pair concatenated).
    out_channels : int
        Number of output channels = n_interp * C_z.
    hidden_channels : int
        Width of the hidden layers (default 512).
    n_layers : int
        Total depth including input and output projections (≥ 2).
    n_groups : int
        Number of groups for GroupNorm inside hidden blocks (default 16).
    dropout : float
        Dropout probability applied after each hidden block activation (0 = off).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 512,
        n_layers: int = 6,
        n_groups: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"n_layers must be >= 2, got {n_layers}")

        layers: list[nn.Module] = []

        # Input projection
        layers.append(_Block(in_channels, hidden_channels, n_groups=n_groups))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(_Block(hidden_channels, hidden_channels, n_groups=n_groups))
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))

        # Output projection (no norm, no activation)
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, z_boundaries: Tensor) -> Tensor:
        """
        z_boundaries : (B, 2*C_z, H_z, W_z) — cat(z_0, z_T)
        returns      : (B, n_interp*C_z, H_z, W_z)
        """
        return self.net(z_boundaries)
