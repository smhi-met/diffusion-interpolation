"""
Latent Interpolation Model (LIM / LND) — no diffusion.

Given a 7-frame weather window at hourly resolution the model learns to
predict the 5 intermediate frames conditioned on the two boundary frames,
operating entirely in the latent space of a frozen autoencoder via a
per-pixel fully-connected (1×1 conv) MLP.

Training data flow
------------------
  batch["x"] : (B, 7, V, E, H, W)   raw physical frames from zarr
    ↓ normalizer (per-variable)
    ↓ view  (B, V*E, H, W) per frame
    ↓ frozen AE encoder
  z_boundary  : (B, 2*C_z, H_z, W_z)   cat(z_0, z_6)   — network input
  z_target    : (B, 5*C_z, H_z, W_z)   cat(z_1…z_5)   — ground truth

  loss = MSE( network(z_boundary), z_target )

Supports both deterministic AEs (LoLA / QRL) and variational AEs (QRL-VAE).
For variational first stages, ``encode_mode="sample"`` draws from the posterior
during training to give the network richer boundary coverage.

References
----------
  Companion to LatentEDMModel (src/dssml/models/diffusion/edm.py).
  Replaces the EDM denoising UNet with a lightweight per-pixel MLP.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import torch
import torch.nn as nn
from hydra.utils import instantiate, get_class
from torch import Tensor

from dssml.models.base import ModelBase
from dssml.networks.ae.base import IAutoEncoder
from dssml.networks.interp import LatentFCInterpolator


class LatentNDIModel(ModelBase):
    """
    Latent No-Diffusion Interpolation model (LIM).

    Parameters
    ----------
    optimizers_conf : list[dict]
        Standard optimizer / scheduler config blocks.
    first_stage_conf : dict
        Hydra config for the frozen first-stage AE Lightning module.
        Must include ``_target_`` and ``ckpt_path``.
    interp_conf : dict
        Hydra config for the interpolation network
        (_target_: dssml.networks.interp.LatentFCInterpolator).
        ``in_channels`` and ``out_channels`` are computed automatically.
    n_interp : int
        Number of interior frames to predict (default 5).
    encode_mode : "mean" | "sample"
        How to encode boundary/target frames from variational AEs.
        "mean"   → posterior mean μ (deterministic, faster).
        "sample" → reparameterization sample z ~ q(z|x) during training.
    normalizer : nn.Module | None
        Injected by DefaultTrainer.
    latent_mean, latent_std : list[float] | None
        Per-channel whitening statistics (C_z-length).  When provided every
        latent channel is mapped to mean≈0, std≈1 before the interpolation
        network and un-whitened before decoding.  Compute with
        scripts/compute_latent_stats.py.
    """

    def __init__(
        self,
        optimizers_conf: list[dict],
        first_stage_conf: dict,
        interp_conf: dict,
        n_interp: int = 5,
        encode_mode: Literal["mean", "sample"] = "mean",
        normalizer: nn.Module | None = None,
        latent_mean: list[float] | None = None,
        latent_std: list[float] | None = None,
    ):
        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        self.save_hyperparameters(ignore=["normalizer"])

        self.n_interp    = int(n_interp)
        self.encode_mode = encode_mode

        # ── Frozen first stage ───────────────────────────────────────────
        self.first_stage: nn.Module = self._load_first_stage(first_stage_conf)
        self._freeze_first_stage()

        ae_net: IAutoEncoder = self.first_stage.auto_encoder
        C_z = self._infer_latent_channels(ae_net)
        self._latent_channels = C_z

        # ── Per-channel latent whitening ─────────────────────────────────
        if latent_mean is not None:
            _lm = torch.tensor(latent_mean, dtype=torch.float32)
            _ls = torch.tensor(latent_std,  dtype=torch.float32).clamp(min=1e-4)
            if _lm.shape[0] != C_z or _ls.shape[0] != C_z:
                raise ValueError(
                    f"latent_mean/latent_std must have length {C_z}, "
                    f"got {_lm.shape[0]} / {_ls.shape[0]}"
                )
        else:
            _lm = torch.zeros(C_z)
            _ls = torch.ones(C_z)

        self.register_buffer("_z_mean", _lm)
        self.register_buffer("_z_std",  _ls)
        self._whitening_enabled = latent_mean is not None

        # ── Interpolation network ────────────────────────────────────────
        interp_conf = deepcopy(dict(interp_conf))
        interp_conf["in_channels"]  = 2 * C_z
        interp_conf["out_channels"] = self.n_interp * C_z

        self.network: LatentFCInterpolator = instantiate(interp_conf, _recursive_=False)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "LatentNDIModel"

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    # ── Per-channel whitening ─────────────────────────────────────────────────

    def _whiten(self, z: Tensor) -> Tensor:
        if not self._whitening_enabled:
            return z
        C = self._latent_channels
        B, nC, H, W = z.shape
        z = z.view(B, -1, C, H, W)
        z = (z - self._z_mean.to(z).view(1, 1, C, 1, 1)) / self._z_std.to(z).view(1, 1, C, 1, 1)
        return z.view(B, nC, H, W)

    def _unwhiten(self, z: Tensor) -> Tensor:
        if not self._whitening_enabled:
            return z
        C = self._latent_channels
        B, nC, H, W = z.shape
        z = z.view(B, -1, C, H, W)
        z = z * self._z_std.to(z).view(1, 1, C, 1, 1) + self._z_mean.to(z).view(1, 1, C, 1, 1)
        return z.view(B, nC, H, W)

    # ── First-stage loading ───────────────────────────────────────────────────

    @staticmethod
    def _load_first_stage(conf: dict) -> nn.Module:
        conf      = deepcopy(dict(conf))
        ckpt_path = conf.pop("ckpt_path", None)
        target    = conf.pop("_target_", None)

        if not target:
            raise ValueError("first_stage_conf must contain '_target_'")
        if not ckpt_path:
            raise ValueError("first_stage_conf must contain 'ckpt_path'")

        model_cls = get_class(target)
        print(f"[LatentNDIModel] Loading first stage ({target}) from {ckpt_path}")
        return model_cls.load_from_checkpoint(
            ckpt_path,
            strict=False,
            **conf,
        )

    def _freeze_first_stage(self) -> None:
        self.first_stage.eval()
        for p in self.first_stage.parameters():
            p.requires_grad_(False)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._freeze_first_stage()

    # ── Latent channel inference ──────────────────────────────────────────────

    @staticmethod
    def _infer_latent_channels(ae_net: IAutoEncoder) -> int:
        if hasattr(ae_net, "latent_channels"):
            return int(ae_net.latent_channels)
        if hasattr(ae_net, "decoder"):
            for module in ae_net.decoder.modules():
                if isinstance(module, nn.Conv2d):
                    return module.in_channels
        raise AttributeError(
            "Cannot infer latent_channels from the first-stage AE. "
            "Add a 'latent_channels' attribute to the network class."
        )

    # ── Encoding helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_frame(self, x_norm: Tensor, training: bool = False) -> Tensor:
        """
        x_norm : (B, V*E, H, W) — normalized single frame
        returns: (B, C_z, H_z, W_z)
        """
        ae = self.first_stage.auto_encoder
        ae.eval()

        use_sample = (
            training
            and self.encode_mode == "sample"
            and hasattr(ae, "encode_posterior")
        )
        if use_sample:
            return ae.encode_posterior(x_norm).sample()
        return ae.encode(x_norm)

    def _normalize_and_encode_window(
        self, x: Tensor, training: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        x : (B, T, V, E, H, W)  where T = 2 + n_interp

        Returns
        -------
        z_cond   : (B, 2*C_z, H_z, W_z)          — whitened cat(z_0, z_{T-1})
        z_target : (B, n_interp*C_z, H_z, W_z)   — whitened cat(z_1, …, z_{T-2})
        """
        B, T, V, E, H, W = x.shape
        assert T == self.n_interp + 2, (
            f"Expected T={self.n_interp + 2} frames, got T={T}"
        )

        x_norm = (
            self.normalizer(x).view(B, T, V * E, H, W)
            if self.normalizer is not None
            else x.view(B, T, V * E, H, W)
        )

        latents: list[Tensor] = []
        for t in range(T):
            latents.append(self._encode_frame(x_norm[:, t], training=training))

        z_cond   = self._whiten(torch.cat([latents[0], latents[-1]], dim=1))
        z_target = self._whiten(torch.cat(latents[1:-1],             dim=1))
        return z_cond, z_target

    # ── Loss ─────────────────────────────────────────────────────────────────

    def _compute_loss(self, z_cond: Tensor, z_target: Tensor) -> Tensor:
        z_pred = self.network(z_cond)
        return nn.functional.mse_loss(z_pred, z_target)

    # ── Lightning steps ───────────────────────────────────────────────────────

    def _shared_step(self, batch: dict, prefix: str) -> Tensor:
        x                = batch["x"]
        z_cond, z_target = self._normalize_and_encode_window(
            x, training=(prefix == "train")
        )
        loss = self._compute_loss(z_cond, z_target)
        self.log(
            f"{prefix}_loss", loss,
            prog_bar=True, on_step=(prefix == "train"),
            on_epoch=True, sync_dist=True, batch_size=x.size(0),
        )
        if prefix == "val":
            self.log(
                "val_z_std", z_target.std(),
                on_step=False, on_epoch=True, sync_dist=True, batch_size=x.size(0),
            )
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    def test_step(self, batch, _):
        return self._shared_step(batch, "test")

    # ── Inference API ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Generate interpolated intermediate frames.

        x : (B, T, V, E, H, W) — 7-frame window

        Returns
        -------
        z_pred   : (B, n_interp*C_z, H_z, W_z)  — predicted latents
        z_target : (B, n_interp*C_z, H_z, W_z)  — ground-truth latents
        """
        self.eval()
        z_cond, z_target = self._normalize_and_encode_window(x, training=False)
        z_pred = self.network(z_cond)
        return z_pred, z_target

    @torch.no_grad()
    def decode_latents(self, z: Tensor) -> Tensor:
        """
        z       : (B, n_interp*C_z, H_z, W_z)  — whitened latents
        returns : (B, n_interp, V*E, H, W)      — normalised pixel space
        """
        B, C_all, H_z, W_z = z.shape
        assert C_all % self._latent_channels == 0
        n  = C_all // self._latent_channels
        ae = self.first_stage.auto_encoder
        ae.eval()
        z = self._unwhiten(z)
        frames = []
        for i in range(n):
            z_i = z[:, i * self._latent_channels : (i + 1) * self._latent_channels]
            frames.append(ae.decode(z_i))
        return torch.stack(frames, dim=1)

    @torch.no_grad()
    def decode_to_physical(self, z: Tensor, V: int, E: int) -> Tensor:
        """
        z       : (B, n_interp*C_z, H_z, W_z)
        returns : (B, n_interp, V, E, H, W) in physical units
        """
        x_norm = self.decode_latents(z)
        B, n, VE, H, W = x_norm.shape
        if self.normalizer is not None:
            x_phys = self.normalizer.denormalize(x_norm.view(B, n, V, E, H, W))
        else:
            x_phys = x_norm.view(B, n, V, E, H, W)
        return x_phys

    @torch.no_grad()
    def encode_boundary_latents(self, x: Tensor) -> Tensor:
        """
        Encode only the two boundary frames (used by sampling callbacks).

        x : (B, T, V, E, H, W)
        returns : (B, 2*C_z, H_z, W_z)
        """
        B, T, V, E, H, W = x.shape
        x_norm = (
            self.normalizer(x).view(B, T, V * E, H, W)
            if self.normalizer is not None
            else x.view(B, T, V * E, H, W)
        )
        z0 = self._encode_frame(x_norm[:, 0],  training=False)
        z6 = self._encode_frame(x_norm[:, -1], training=False)
        return self._whiten(torch.cat([z0, z6], dim=1))
