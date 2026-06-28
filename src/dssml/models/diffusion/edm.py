"""
Latent EDM (Elucidating Diffusion Models) for temporal interpolation.

Given a 7-frame weather window at hourly resolution, the model learns to
generate the 5 intermediate frames conditioned on the two boundary frames,
operating entirely in the latent space of a frozen autoencoder.

Training data flow
------------------
  batch["x"] : (B, 7, V, E, H, W)   raw physical frames from zarr
    ↓ normalizer (per-variable)
    ↓ view  (B, V*E, H, W) per frame
    ↓ frozen AE encoder
  z_boundary  : (B, 2*C_z, H_z, W_z)   cat(z_0, z_6)   — conditioning
  z_target    : (B, 5*C_z, H_z, W_z)   cat(z_1…z_5)   — ground truth

  EDM loss  =  Σ_i  w(σ_i) · ‖D(z_cond, z_tgt + ε, σ) − z_tgt‖²

Supports both deterministic AEs (LoLA / QRL) and variational AEs (QRL-VAE).
For variational first stages, `encode_mode="sample"` draws from the posterior
during training (better latent coverage), "mean" uses μ (faster, good for small σ).

References
----------
  Karras et al. 2022 "Elucidating the Design Space of Diffusion-Based
  Generative Models"  (arXiv:2206.00364)
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Literal

import torch
import torch.nn as nn
from hydra.utils import instantiate, get_class
from torch import Tensor

from dssml.models.base import ModelBase
from dssml.networks.ae.base import IAutoEncoder  # avoid circular import via ae/__init__
from dssml.networks.unet import LatentUNet, fourier_embed


# ---------------------------------------------------------------------------
# EDM pre-conditioning  (Karras et al. eq. 7)
# ---------------------------------------------------------------------------

def _edm_precond(sigma: Tensor, sigma_data: float):
    """
    c_skip, c_out, c_in, c_noise — all broadcast-shaped (B, 1, 1, 1).
    """
    s2  = sigma * sigma
    sd2 = sigma_data ** 2
    c_skip  = sd2 / (s2 + sd2)
    c_out   = sigma * (sd2 ** 0.5) / (s2 + sd2).sqrt()
    c_in    = 1.0 / (s2 + sd2).sqrt()
    c_noise = 0.25 * torch.log(sigma.clamp(min=1e-12))
    view    = (-1, 1, 1, 1)
    return (c_skip.view(*view), c_out.view(*view),
            c_in.view(*view),  c_noise.squeeze())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LatentEDMModel(ModelBase):
    """
    EDM diffusion in latent space for temporal interpolation.

    Parameters
    ----------
    optimizers_conf : list[dict]
        Standard optimizer / scheduler config blocks (same format as AE models).
    first_stage_conf : dict
        Hydra config for the frozen first-stage AE Lightning module.
        Must include ``_target_`` (e.g. ``dssml.models.ae.LoLADCAEModel``) and
        ``ckpt_path`` pointing to a trained checkpoint.
    unet_conf : dict
        Hydra config for the denoising network (``_target_: dssml.networks.unet.LatentUNet``).
        ``in_ch`` and ``out_ch`` are computed automatically and should NOT be set here.
    sampler_conf : dict
        Hydra config for a sampler (``_target_: dssml.samplers.HeunEDMSampler``, etc.).
    sigma_data : float
        Expected RMS of latent codes. Set to the empirical std of the latent
        distribution from a validation run; 1.0 is a safe default when the AE
        uses LoLA saturation with bound=5.
    p_mean, p_std : float
        Log-normal σ distribution: log σ ~ N(p_mean, p_std²).
        Karras et al. default: p_mean=-1.2, p_std=1.2.
    n_interp : int
        Number of interior frames to predict (default 5, between two boundaries).
    encode_mode : "mean" | "sample"
        How to encode target frames from variational AEs.
        "mean"   → posterior mean μ (deterministic, faster).
        "sample" → single reparameterization sample z ~ q(z|x) during training.
    normalizer : nn.Module | None
        Injected by DefaultTrainer. Must match the stats used to train the
        first-stage AE (same variables and normalization types).
    """

    def __init__(
        self,
        optimizers_conf: list[dict],
        first_stage_conf: dict,
        unet_conf: dict,
        sampler_conf: dict,
        sigma_data: float = 1.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        n_interp: int = 5,
        n_sigma: int = 1,
        encode_mode: Literal["mean", "sample"] = "mean",
        normalizer: nn.Module | None = None,
        latent_mean: list[float] | None = None,
        latent_std: list[float] | None = None,
    ):
        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        self.save_hyperparameters(ignore=["normalizer"])

        self.sigma_data  = float(sigma_data)
        self.p_mean      = float(p_mean)
        self.p_std       = float(p_std)
        self.n_interp    = int(n_interp)
        self.n_sigma     = int(n_sigma)
        self.encode_mode = encode_mode

        # ── Frozen first stage ───────────────────────────────────────────
        self.first_stage: nn.Module = self._load_first_stage(first_stage_conf)
        self._freeze_first_stage()

        # Infer latent channel count from the AE network
        ae_net: IAutoEncoder = self.first_stage.auto_encoder
        C_z = self._infer_latent_channels(ae_net)
        self._latent_channels = C_z

        # ── Per-channel latent whitening (optional) ──────────────────────
        # Eliminates the uncentered / imbalanced latent channel issue that
        # inflates the global z_gt.std above sqrt(mean(per-channel variances)).
        # When enabled: sigma_data=1.0 is correct and Karras p_mean=-1.2 applies.
        #
        # Supply latent_mean / latent_std (C_z-length lists) in the config.
        # Compute them once with: scripts/compute_latent_stats.py
        # or read them from the callback log (|mean|_max and ch_std min/max).
        if latent_mean is not None:
            _lm = torch.tensor(latent_mean, dtype=torch.float32)
            _ls = torch.tensor(latent_std,  dtype=torch.float32).clamp(min=1e-4)
            if _lm.shape[0] != C_z or _ls.shape[0] != C_z:
                raise ValueError(
                    f"latent_mean/latent_std must have length {C_z} (= latent_channels), "
                    f"got {_lm.shape[0]} / {_ls.shape[0]}"
                )
        else:
            _lm = torch.zeros(C_z)
            _ls = torch.ones(C_z)

        self.register_buffer("_z_mean", _lm)   # (C_z,)
        self.register_buffer("_z_std",  _ls)   # (C_z,)
        self._whitening_enabled = latent_mean is not None

        # ── Denoising UNet ───────────────────────────────────────────────
        # Conditioning = 2 boundary frames concatenated → 2*C_z channels
        # Noisy target  = n_interp stacked frames       → n_interp*C_z channels
        cond_ch   = 2 * C_z
        target_ch = self.n_interp * C_z
        unet_conf = deepcopy(dict(unet_conf))
        unet_conf["in_ch"]  = cond_ch + target_ch
        unet_conf["out_ch"] = target_ch

        self.network: LatentUNet = instantiate(unet_conf, _recursive_=False)

        # ── Noise-level embedding MLP ─────────────────────────────────────
        self.emb_mlp = nn.Sequential(
            nn.Linear(64, self.network.emb_dim),
            nn.SiLU(),
            nn.Linear(self.network.emb_dim, self.network.emb_dim),
        )

        # ── Sampler (lazy) ───────────────────────────────────────────────
        self._sampler_conf = sampler_conf
        self._sampler       = None

        # Warn if the sampler's sigma_max is far outside the training distribution.
        # Training samples sigma ~ LogNormal(p_mean, p_std²).  The 99th-percentile
        # training sigma is exp(p_mean + 3*p_std).  If sigma_max >> this, the first
        # many denoising steps are at noise levels the network was never trained on.
        _sigma_99 = math.exp(self.p_mean + 3.0 * self.p_std)
        _sigma_max_cfg = sampler_conf.get("sigma_max", 80.0) if hasattr(sampler_conf, "get") else getattr(sampler_conf, "sigma_max", 80.0)
        if float(_sigma_max_cfg) > 4.0 * _sigma_99:
            import warnings
            warnings.warn(
                f"[LatentEDMModel] sigma_max={_sigma_max_cfg} is much larger than the "
                f"99th-percentile training sigma ({_sigma_99:.2f} = exp(p_mean + 3·p_std)). "
                f"Consider reducing sigma_max to ~{4*_sigma_99:.1f} or increasing p_mean/p_std "
                f"so training covers the full inference sigma range.",
                UserWarning, stacklevel=2,
            )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "LatentEDMModel"

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    @property
    def sampler(self):
        if self._sampler is None:
            self._sampler = instantiate(self._sampler_conf)
        return self._sampler

    # ── Per-channel whitening ─────────────────────────────────────────────────

    def _whiten(self, z: Tensor) -> Tensor:
        """(B, n*C_z, H, W) → per-channel zero-mean / unit-std."""
        if not self._whitening_enabled:
            return z
        C = self._latent_channels
        B, nC, H, W = z.shape
        z = z.view(B, -1, C, H, W)
        z = (z - self._z_mean.to(z).view(1, 1, C, 1, 1)) / self._z_std.to(z).view(1, 1, C, 1, 1)
        return z.view(B, nC, H, W)

    def _unwhiten(self, z: Tensor) -> Tensor:
        """Inverse of _whiten."""
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
        """
        Load a trained AE Lightning module as the frozen first stage.

        The AE model is loaded with ``strict=False`` so that the normalizer
        (excluded from hparams via ``ignore=["normalizer"]``) does not cause
        a missing-key error.  The diffusion model uses its own injected
        normalizer for encoding (same statistics, different object).
        """
        conf = deepcopy(dict(conf))
        ckpt_path = conf.pop("ckpt_path", None)
        target    = conf.pop("_target_", None)

        if not target:
            raise ValueError("first_stage_conf must contain '_target_'")
        if not ckpt_path:
            raise ValueError("first_stage_conf must contain 'ckpt_path'")

        model_cls = get_class(target)
        print(f"[LatentEDMModel] Loading first stage ({target}) from {ckpt_path}")
        model = model_cls.load_from_checkpoint(
            ckpt_path,
            strict=False,   # skip normalizer keys not in hparams
            **conf,
        )
        return model

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
        """
        Infer C_z from the first-stage AE network.

        Priority:
          1. self.latent_channels  (QRLVariationalAutoEncoder stores it explicitly)
          2. decoder.project_in first Conv2d in_channels  (DC-AE family)
          3. AttributeError with clear message
        """
        if hasattr(ae_net, "latent_channels"):
            return int(ae_net.latent_channels)

        # DC-AE decoders (LoLA, QRL): the decoder's first conv is project_in,
        # whose in_channels equals latent_channels.
        import torch.nn as nn
        if hasattr(ae_net, "decoder"):
            for module in ae_net.decoder.modules():
                if isinstance(module, nn.Conv2d):
                    return module.in_channels

        raise AttributeError(
            "Cannot infer latent_channels from the first-stage AE network. "
            "Add a 'latent_channels' attribute to the network class."
        )

    # ── Encoding helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_frame(self, x_norm: Tensor, training: bool = False) -> Tensor:
        """
        Encode a single (already-normalized) frame to latent.

        x_norm : (B, V*E, H, W) — normalized, one frame per sample
        returns: (B, C_z, H_z, W_z)

        For deterministic AEs: always returns the deterministic encode.
        For variational AEs:   returns a posterior sample during training
                               (if encode_mode="sample"), otherwise the mean.
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
        Normalize a 7-frame window and encode each frame independently.

        x : (B, T, V, E, H, W) where T = 2 + n_interp (typically 7)

        Returns
        -------
        z_cond   : (B, 2*C_z, H_z, W_z)     — cat(z_0, z_{T-1}) boundaries
        z_target : (B, n_interp*C_z, H_z, W_z) — cat(z_1, …, z_{T-2}) intermediates
        """
        B, T, V, E, H, W = x.shape
        assert T == self.n_interp + 2, (
            f"Expected T={self.n_interp + 2} frames (2 boundaries + {self.n_interp} "
            f"intermediates), got T={T}"
        )

        # Normalize entire window at once (MultiNormalizer broadcasts over T and E)
        x_norm = (
            self.normalizer(x).view(B, T, V * E, H, W)
            if self.normalizer is not None
            else x.view(B, T, V * E, H, W)
        )

        # Encode each frame: loop over T for memory efficiency
        latents: list[Tensor] = []
        for t in range(T):
            latents.append(self._encode_frame(x_norm[:, t], training=training))

        z_cond   = self._whiten(torch.cat([latents[0], latents[-1]], dim=1))
        z_target = self._whiten(torch.cat(latents[1:-1],             dim=1))
        return z_cond, z_target

    # ── EDM core ──────────────────────────────────────────────────────────────

    def _sample_sigma(self, B: int, device: torch.device) -> Tensor:
        """Log-normal σ sampling (Karras et al. eq. 5)."""
        return torch.exp(
            self.p_std * torch.randn(B, device=device) + self.p_mean
        )

    def _denoise_target(
        self,
        cond: Tensor,
        x_noisy: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        """
        EDM-preconditioned denoiser D(cond, x_noisy, σ) → x0_hat.

        cond    : (B, 2*C_z, H_z, W_z)       — clean boundary latents
        x_noisy : (B, n_interp*C_z, H_z, W_z) — noisy target
        sigma   : (B,)

        This method is called by HeunEDMSampler (and other compatible samplers).
        """
        c_skip, c_out, c_in, c_noise = _edm_precond(sigma, self.sigma_data)
        emb     = self.emb_mlp(fourier_embed(c_noise))
        net_in  = torch.cat([cond, c_in * x_noisy], dim=1)
        fx      = self.network(net_in, emb)
        return c_skip * x_noisy + c_out * fx

    def _compute_loss(self, z_cond: Tensor, z_target: Tensor) -> Tensor:
        """EDM training loss (Karras et al. eq. 5 + eq. 7 weighting).

        When n_sigma > 1, the UNet is called n_sigma times with independent
        (sigma, noise) draws while z_cond / z_target stay at their original
        (B, ...) shape, so latent memory does not grow.
        """
        B      = z_target.size(0)
        device = z_target.device
        sd     = self.sigma_data

        loss = z_target.new_zeros(())
        for _ in range(self.n_sigma):
            sigma   = self._sample_sigma(B, device)                   # (B,)
            noise   = torch.randn_like(z_target) * sigma.view(-1, 1, 1, 1)
            z_noisy = z_target + noise
            z_hat   = self._denoise_target(z_cond, z_noisy, sigma)
            w = (sigma ** 2 + sd ** 2) / ((sigma * sd) ** 2 + 1e-12)
            loss = loss + (w.view(-1, 1, 1, 1) * (z_hat - z_target) ** 2).mean()

        return loss / self.n_sigma

    # ── Lightning steps ───────────────────────────────────────────────────────

    def _shared_step(self, batch: dict, prefix: str) -> Tensor:
        x                   = batch["x"]                   # (B, T, V, E, H, W)
        z_cond, z_target    = self._normalize_and_encode_window(
            x, training=(prefix == "train")
        )
        loss = self._compute_loss(z_cond, z_target)
        self.log(
            f"{prefix}_loss", loss,
            prog_bar=True, on_step=(prefix == "train"),
            on_epoch=True, sync_dist=True, batch_size=x.size(0),
        )
        if prefix == "val":
            # Empirical latent std.
            # Without whitening: should match sigma_data in the config.
            # With whitening enabled: should be ≈ 1.0 (all channels normalised).
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

    # ── Sampling API ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Generate interpolated intermediate frames.

        x : (B, T, V, E, H, W) — 7-frame window (boundary frames at t=0 and t=6)

        Returns
        -------
        z_samples : (B, n_interp*C_z, H_z, W_z)   — generated latents
        z_target  : (B, n_interp*C_z, H_z, W_z)   — ground-truth latents (for metrics)
        """
        self.eval()
        z_cond, z_target = self._normalize_and_encode_window(x, training=False)
        B, C_t, H_z, W_z = z_target.shape
        z_samples = self.sampler.sample(
            module=self,
            cond=z_cond,
            target_shape=(B, C_t, H_z, W_z),
            device=x.device,
        )
        return z_samples, z_target

    @torch.no_grad()
    def decode_latents(self, z: Tensor) -> Tensor:
        """
        Decode stacked latent frames back to normalised physical space.

        z       : (B, n_interp*C_z, H_z, W_z)
        returns : (B, n_interp, V*E, H, W) in normalised space
        """
        B, C_all, H_z, W_z = z.shape
        assert C_all % self._latent_channels == 0
        n = C_all // self._latent_channels
        ae = self.first_stage.auto_encoder
        ae.eval()
        frames = []
        z = self._unwhiten(z)
        for i in range(n):
            z_i   = z[:, i * self._latent_channels : (i + 1) * self._latent_channels]
            x_hat = ae.decode(z_i)                          # (B, V*E, H, W) norm space
            frames.append(x_hat)
        return torch.stack(frames, dim=1)                   # (B, n, V*E, H, W)

    @torch.no_grad()
    def decode_to_physical(self, z: Tensor, V: int, E: int) -> Tensor:
        """
        Decode stacked latents and denormalise to physical space.

        z       : (B, n_interp*C_z, H_z, W_z)
        V, E    : variable count and ensemble size (from dataset)
        returns : (B, n_interp, V, E, H, W) in physical units
        """
        x_norm = self.decode_latents(z)                     # (B, n, V*E, H, W)
        B, n, VE, H, W = x_norm.shape
        if self.normalizer is not None:
            # Reshape to (B, n, V, E, H, W) for denormalization
            x_phys = self.normalizer.denormalize(
                x_norm.view(B, n, V, E, H, W)
            )
        else:
            x_phys = x_norm.view(B, n, V, E, H, W)
        return x_phys

    @torch.no_grad()
    def encode_boundary_latents(self, x: Tensor) -> Tensor:
        """
        Encode only the two boundary frames for use in the sampling callback.

        x : (B, T, V, E, H, W)
        returns : (B, 2*C_z, H_z, W_z)
        """
        B, T, V, E, H, W = x.shape
        x_norm = (
            self.normalizer(x).view(B, T, V * E, H, W)
            if self.normalizer is not None
            else x.view(B, T, V * E, H, W)
        )
        z0 = self._encode_frame(x_norm[:, 0], training=False)
        z6 = self._encode_frame(x_norm[:, -1], training=False)
        return self._whiten(torch.cat([z0, z6], dim=1))
