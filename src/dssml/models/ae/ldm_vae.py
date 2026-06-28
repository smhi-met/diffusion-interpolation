import torch
import torch.nn.functional as F
from torch import nn
from hydra.utils import instantiate

from dssml.models.base import ModelBase
from dssml.networks.ae import IAutoEncoder
from dssml.networks.ae.ldm_vae import LDMVariationalAutoEncoder, DiagonalGaussian


class LDMVAEModel(ModelBase):
    """
    Lightning model wrapping LDMVariationalAutoEncoder.

    Supported loss_type values (combine with '+'):
      - 'mse'       : pixel-wise MSE reconstruction loss
      - 'kl'        : KL divergence — regularizes latent space to N(0,1)
      - 'spectral'  : penalizes differences in spatial frequency content
      - 'gradient'  : penalizes differences in spatial gradients (fronts, edges)

    Examples:
      loss_type: "mse"
      loss_type: "mse+kl"
      loss_type: "mse+kl+spectral"
      loss_type: "mse+kl+gradient"
      loss_type: "mse+kl+spectral+gradient"   ← recommended for weather
    """

    def __init__(
        self,
        optimizers_conf: list[dict],
        auto_encoder_conf: dict,
        loss_type: str = "mse+kl",
        spectral_weight: float = 0.1,
        gradient_weight: float = 0.1,
        kl_weight: float = 1e-6,
        kl_warmup_steps: int = 10000,

        normalizer: nn.Module | None = None,
    ):
        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        self.save_hyperparameters(ignore=["normalizer"])

        self.loss_type = loss_type
        self.kl_weight = kl_weight
        self.kl_warmup_steps = kl_warmup_steps
        self.spectral_weight = spectral_weight
        self.gradient_weight = gradient_weight

        # Instantiate the VAE network
        self.auto_encoder: LDMVariationalAutoEncoder = instantiate(
            auto_encoder_conf, _recursive_=False
        )
        if not isinstance(self.auto_encoder, IAutoEncoder):
            raise ValueError(
                f"auto_encoder must implement IAutoEncoder, got {type(self.auto_encoder)}"
            )

    @property
    def name(self):
        return "LDMVAEModel"

    # ------------------------------------------------------------------
    # Physical losses
    # ------------------------------------------------------------------

    def _compute_spectral_loss(
        self, x_hat: torch.Tensor, x_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalizes differences in spatial frequency power spectra.
        Weather fields have characteristic scale-dependent energy distributions
        (synoptic scales dominate). This ensures the AE preserves that structure
        rather than producing spatially blurry reconstructions.

        Operates on the amplitude spectrum (ignores phase differences).
        Shape: (B, C, H, W) → rfft2 → compare amplitude spectra.
        """
        # rfft2 returns complex tensor of shape (B, C, H, W//2+1)
        fft_hat = torch.fft.rfft2(x_hat, norm="ortho")
        fft_tgt = torch.fft.rfft2(x_target, norm="ortho")
        return F.mse_loss(fft_hat.abs(), fft_tgt.abs())

    def _compute_gradient_loss(
        self, x_hat: torch.Tensor, x_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalizes differences in spatial gradients (finite differences).
        This directly targets the preservation of:
          - Atmospheric fronts (sharp temperature/pressure gradients)
          - Jet stream boundaries
          - Precipitation edges

        Pure MSE tends to smooth these sharp features; gradient loss
        explicitly penalizes their loss.
        """
        # x-direction gradient (longitude)
        dx_hat = x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1]
        dx_tgt = x_target[:, :, :, 1:] - x_target[:, :, :, :-1]

        # y-direction gradient (latitude)
        dy_hat = x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :]
        dy_tgt = x_target[:, :, 1:, :] - x_target[:, :, :-1, :]

        return F.mse_loss(dx_hat, dx_tgt) + F.mse_loss(dy_hat, dy_tgt)

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def _get_kl_weight(self, warmup_steps):
        """Linearly ramp KL weight from 0 to kl_weight over warmup_steps."""
        return self.kl_weight * min(1.0, self.global_step/warmup_steps)

    def _compute_loss(
        self,
        x_hat: torch.Tensor,
        x_target: torch.Tensor,
        posterior: DiagonalGaussian,
        prefix: str,
    ) -> torch.Tensor:

        total_loss = torch.zeros(1, device=x_hat.device, dtype=x_hat.dtype).squeeze()

        # 1. MSE reconstruction loss
        if "mse" in self.loss_type:
            per_var_mse = (x_hat - x_target).pow(2)          # (B, C, H, W)
            if (
                self.normalizer is not None
                and hasattr(self.normalizer, "loss_weights")
            ):
                w = self.normalizer.loss_weights.view(1, -1, 1, 1)  # (1, C, 1, 1)
                rec_loss = (per_var_mse * w).mean()
            else:
                rec_loss = per_var_mse.mean()                 # SymRangeNormalizer fallback
            self.log(f"{prefix}_rec_loss", rec_loss,
                    prog_bar=True, on_epoch=True, sync_dist=True)
            total_loss = total_loss + rec_loss

        # 2. KL divergence — regularizes latent space
        if "kl" in self.loss_type:
            kl_loss = posterior.kl().mean()
            weighted_kl = self._get_kl_weight(self.kl_warmup_steps)

            self.log(f"{prefix}_kl_loss", kl_loss,
                    on_epoch=True, sync_dist=True)
            self.log(f"{prefix}_kl_weighted", weighted_kl,
                    on_epoch=True, sync_dist=True)
            total_loss = total_loss + weighted_kl

        # 3. Spectral loss — preserves spatial frequency content
        if "spectral" in self.loss_type:
            spectral_loss = self._compute_spectral_loss(x_hat, x_target)
            weighted_spectral = self.spectral_weight * spectral_loss
            self.log(f"{prefix}_spectral_loss", spectral_loss,
                    on_epoch=True, sync_dist=True)
            self.log(f"{prefix}_spectral_weighted", weighted_spectral,
                    on_epoch=True, sync_dist=True)
            total_loss = total_loss + weighted_spectral

        # 4. Gradient loss — preserves fronts and sharp features
        if "gradient" in self.loss_type:
            gradient_loss = self._compute_gradient_loss(x_hat, x_target)
            weighted_gradient = self.gradient_weight * gradient_loss
            self.log(f"{prefix}_gradient_loss", gradient_loss,
                    on_epoch=True, sync_dist=True)
            self.log(f"{prefix}_gradient_weighted", weighted_gradient,
                    on_epoch=True, sync_dist=True)
            total_loss = total_loss + weighted_gradient

        self.log(f"{prefix}_loss", total_loss,
                prog_bar=True, on_epoch=True, sync_dist=True)
        return total_loss

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def _shared_step(self, batch, batch_idx, prefix="val"):
        x = batch["x"]
        B, T, V, E, H, W = x.shape

        if self.normalizer is not None:
            _x = self.normalizer(x).view(B, T * V * E, H, W)
        else:
            _x = x.view(B, T * V * E, H, W)

        # sample_posterior=True during training (stochastic),
        # False during validation/test (deterministic, use mean)
        x_hat, posterior = self.auto_encoder(_x, sample_posterior=self.training)

        return self._compute_loss(x_hat, _x, posterior, prefix)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="test")

    def predict_step(self, batch, batch_idx):
        x = batch["x"]
        B, T, V, E, H, W = x.shape
        if self.normalizer is not None:
            _x = self.normalizer(x).view(B, T * V * E, H, W)
        else:
            _x = x.view(B, T * V * E, H, W)

        posterior = self.auto_encoder.encode(_x)
        z = posterior.mode()   # deterministic — no sampling noise at inference
        x_hat = self.auto_encoder.decode(z).view(B, T, V, E, H, W)

        if self.normalizer is not None:
            x_hat = self.normalizer.denormalize(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V, E, H, W = x.shape
        if self.normalizer is not None:
            _x = self.normalizer(x).view(B, T * V * E, H, W)
        else:
            _x = x.view(B, T * V * E, H, W)

        posterior = self.auto_encoder.encode(_x)
        z = posterior.mode()
        x_hat = self.auto_encoder.decode(z).view(B, T, V, E, H, W)

        if self.normalizer is not None:
            x_hat = self.normalizer.denormalize(x_hat)
        return x_hat