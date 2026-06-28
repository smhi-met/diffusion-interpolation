import torch
import torch.nn as nn
import numpy as np
from .base import IAutoEncoder


def nonlinearity(x):
    return x * torch.sigmoid(x)  # SiLU/Swish


def GNormalize(in_channels, num_groups=32):
    # Clamp groups to avoid GroupNorm errors with small channel counts
    num_groups = min(num_groups, in_channels)
    while in_channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            # Asymmetric padding as in the paper
            x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = GNormalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = GNormalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = GNormalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, h_dim, w_dim = q.shape
        # Reshape for attention
        q = q.reshape(b, c, h_dim * w_dim).permute(0, 2, 1)   # (B, HW, C)
        k = k.reshape(b, c, h_dim * w_dim)                     # (B, C, HW)
        w = torch.bmm(q, k) * (c ** -0.5)
        w = torch.nn.functional.softmax(w, dim=2)
        v = v.reshape(b, c, h_dim * w_dim)
        w = w.permute(0, 2, 1)
        h_ = torch.bmm(v, w).reshape(b, c, h_dim, w_dim)
        return x + self.proj_out(h_)


class LDMEncoder(nn.Module):
    """
    Encoder from LDM paper.
    Outputs 2*z_channels (mean + logvar) when double_z=True (VAE mode).
    """
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        ch: int = 128,
        ch_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (16,),
        resolution: int = 256,
        dropout: float = 0.0,
        double_z: bool = True,
        resamp_with_conv: bool = True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # Stem
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # Downsampling
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()

            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attns.append(AttnBlock(block_in))

            level = nn.Module()
            level.block = blocks
            level.attn = attns
            if i_level != self.num_resolutions - 1:
                level.downsample = Downsample(block_in, resamp_with_conv)
                curr_res //= 2
            self.down.append(level)

        # Middle (bottleneck)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)

        # Output
        self.norm_out = GNormalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], 'downsample'):
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LDMDecoder(nn.Module):
    """
    Decoder from LDM paper. Mirrors the encoder architecture.
    """
    def __init__(
        self,
        out_channels: int,
        z_channels: int,
        ch: int = 128,
        ch_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (16,),
        resolution: int = 256,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        tanh_out: bool = True,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        # z → block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()

            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attns.append(AttnBlock(block_in))

            level = nn.Module()
            level.block = blocks
            level.attn = attns
            if i_level != 0:
                level.upsample = Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, level)  # prepend for consistent ordering

        # Output
        self.norm_out = GNormalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)

        # Middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Upsampling (reversed order)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], 'upsample'):
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class DiagonalGaussian:
    """
    Reparameterization trick for VAE.
    Wraps the encoder output (mean + logvar) into a distribution.
    """
    def __init__(self, parameters: torch.Tensor):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        """Returns mean — used at inference (no sampling noise)."""
        return self.mean

    def kl(self) -> torch.Tensor:
        """KL divergence from N(0,1): -0.5 * (1 + logvar - mean^2 - var)"""
        return 0.5 * torch.sum(
            self.mean ** 2 + self.var - 1.0 - self.logvar,
            dim=[1, 2, 3]
        )


class LDMVariationalAutoEncoder(nn.Module, IAutoEncoder):
    """
    Full VAE following the LDM paper architecture.
    
    At training: encode → sample from posterior → decode
    At inference: encode → take mean (mode) → decode
    
    Args:
        input_channels:    number of input/output channels (e.g. 22 for your 22 variables)
        z_channels:        latent channel depth
        ch:                base channel count
        ch_mult:           channel multipliers per resolution level
        num_res_blocks:    ResnetBlocks per level
        attn_resolutions:  spatial resolutions where attention is applied
        resolution:        input spatial resolution (H=W)
        dropout:           dropout rate
        kl_weight:         weight on KL term in the ELBO loss
        resamp_with_conv:  use learned conv for up/downsampling
    """
    def __init__(
        self,
        input_channels: int,
        z_channels: int,
        ch: int = 128,
        ch_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (16,),
        resolution: int = 256,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
    ):
        super().__init__()

        self.z_channels = z_channels

        self.encoder = LDMEncoder(
            in_channels=input_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            resolution=resolution,
            dropout=dropout,
            double_z=True,          # outputs mean + logvar
            resamp_with_conv=resamp_with_conv,
        )
        self.decoder = LDMDecoder(
            out_channels=input_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            resolution=resolution,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            tanh_out=True,
        )

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        """Returns a DiagonalGaussian distribution, not a raw tensor."""
        return DiagonalGaussian(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior