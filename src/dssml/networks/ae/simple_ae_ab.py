import torch
import torch.nn as nn
from .base import IAutoEncoder


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            # Stage 1: H -> H (no downsample on first stage)
            nn.Conv2d(input_channels, 32, 3, stride=1, padding=1),
            nn.SiLU(),

            # Stage 2: H -> H/2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),

            # Stage 3: H/2 -> H/4
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),

            # Stage 4: H/4 -> H/8
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.SiLU(),

            # Bottleneck: H/8 -> H/16
            nn.Conv2d(256, latent_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            # Expand from latent
            nn.Conv2d(latent_dim, 256, kernel_size=1),
            nn.SiLU(),

            # Stage 4: H/16 -> H/8
            Upsample(256, 128),
            nn.SiLU(),

            # Stage 3: H/8 -> H/4
            Upsample(128, 64),
            nn.SiLU(),

            # Stage 2: H/4 -> H/2
            Upsample(64, 32),
            nn.SiLU(),

            # Stage 1: H/2 -> H
            Upsample(32, 32),
            nn.SiLU(),

            # Final refinement
            nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


class AutoEncoderAB(nn.Module, IAutoEncoder):
    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(input_channels, latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)