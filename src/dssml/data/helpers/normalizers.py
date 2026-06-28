"""
Per-variable normalizer — Hydra compatible.

All normalizations share one affine formula:

    normalize:   x_norm = (x - beta) / alpha
    denormalize: x_orig = x_norm * alpha + beta

(alpha, beta) resolved per NormType:

    NormType      beta                  alpha
    ────────────────────────────────────────────────────
    ZSCORE        mean                  std
    STD_ONLY      0                     std
    MINMAX        min                   max - min   → [0, 1]
    SYM_RANGE     (min + max) / 2       (max-min)/2 → [−1, 1]
    MAX           0                     max
    NONE          0                     1

Hydra usage (MultiNormalizer — preferred for new configs)
─────────────────────────────────────────────────────────
    normalizer:
      _target_: dssml.data.helpers.normalizers.MultiNormalizer
"""

from __future__ import annotations
from enum import Enum
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# NormType
# ---------------------------------------------------------------------------

class NormType(str, Enum):
    ZSCORE    = "zscore"
    STD_ONLY  = "std_only"
    MINMAX    = "minmax"
    SYM_RANGE = "sym_range"
    MAX       = "max"
    NONE      = "none"


_REQUIRED_STATS: dict[NormType, list[str]] = {
    NormType.ZSCORE:    ["mean", "std"],
    NormType.STD_ONLY:  ["std"],
    NormType.MINMAX:    ["minimum", "maximum"],
    NormType.SYM_RANGE: ["minimum", "maximum"],
    NormType.MAX:       ["maximum"],
    NormType.NONE:      [],
}


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class NormalizerBase(nn.Module):
    def __init__(
        self,
        required_stats: list[str],
        var_dim: int = -4,
        stats: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.var_dim = var_dim
        stats = stats or {}
        for key in required_stats:
            if key not in stats:
                raise ValueError(
                    f"Required stat '{key}' not found. Available: {list(stats.keys())}"
                )
            self.register_buffer(key, stats[key])

    def _broadcast(self, stat_name: str, x: torch.Tensor) -> torch.Tensor:
        tensor = getattr(self, stat_name)
        dim = self.var_dim % x.dim()
        shape = [1] * x.dim()
        shape[dim] = tensor.numel()
        return tensor.view(*shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Legacy normalizers — kept for backward compat with existing checkpoints.
# Use MultiNormalizer for new configs.
# ---------------------------------------------------------------------------

class SymRangeNormalizer(NormalizerBase):
    def __init__(
        self,
        stats: dict[str, torch.Tensor],
        var_dim: int = -4,
        norm_const: float = 1.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            required_stats=["minimum", "maximum"], var_dim=var_dim, stats=stats
        )
        self.eps = eps
        self.register_buffer(
            "norm_const_tensor", torch.tensor(norm_const, dtype=torch.float32)
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        minimum = self._broadcast("minimum", x)
        maximum = self._broadcast("maximum", x)
        denom = (maximum - minimum).clamp_min(self.eps)
        return (2.0 * (x - minimum) / denom - 1.0) * self.norm_const_tensor

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        minimum = self._broadcast("minimum", x)
        maximum = self._broadcast("maximum", x)
        denom = (maximum - minimum).clamp_min(self.eps)
        return (x / self.norm_const_tensor + 1.0) * 0.5 * denom + minimum


class NormalNormalizer(NormalizerBase):
    def __init__(
        self,
        stats: dict[str, torch.Tensor],
        var_dim: int = -4,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(required_stats=["mean", "std"], var_dim=var_dim, stats=stats)
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._broadcast("mean", x)
        std  = self._broadcast("std", x)
        return (x - mean) / std.clamp_min(self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._broadcast("mean", x)
        std  = self._broadcast("std", x)
        return x * std.clamp_min(self.eps) + mean


# ---------------------------------------------------------------------------
# Mapping from config normalization name strings → NormType
# ---------------------------------------------------------------------------

_CONF_NORM_MAP: dict[str, NormType] = {
    "z-score":   NormType.ZSCORE,
    "zscore":    NormType.ZSCORE,
    "z_score":   NormType.ZSCORE,
    "std":       NormType.STD_ONLY,
    "std_only":  NormType.STD_ONLY,
    "minmax":    NormType.MINMAX,
    "min_max":   NormType.MINMAX,
    "sym_range": NormType.SYM_RANGE,
    "sym-range": NormType.SYM_RANGE,
    "max":       NormType.MAX,
    "none":      NormType.NONE,
}


# ---------------------------------------------------------------------------
# MultiNormalizer
# ---------------------------------------------------------------------------

class MultiNormalizer(NormalizerBase):
    """
    Per-variable normalizer driven by the variables config from MepsZarrManifest.

    Normalization formula (same affine form as all other normalizers):
        normalize:   x_norm = (x - beta) / alpha
        denormalize: x_orig = (x_norm) * alpha + beta
    Hydra config:
        normalizer:
          _target_: dssml.data.helpers.normalizers.MultiNormalizer
    """

    def __init__(
        self,
        variables_conf: list[dict],
        stats: dict[str, torch.Tensor],
        var_dim: int = -4,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(required_stats=[], var_dim=var_dim, stats={})
        self.eps = eps

        n = len(variables_conf)
        alpha = torch.ones(n)
        beta  = torch.zeros(n)
        loss_weights  = torch.zeros(n)
        errors: list[str] = []

        for i, vc in enumerate(variables_conf):
            name      = vc.get("name")          if hasattr(vc, "get") else vc["name"]
            norm_str  = vc.get("normalization") if hasattr(vc, "get") else None
            loss_weight_val = vc.get("loss_weight") if hasattr(vc, "get") else None
            if loss_weight_val is not None:
                try:
                    loss_weights[i] = float(loss_weight_val)
                except ValueError:
                    errors.append(
                        f"  [{i}] '{name}': invalid loss_weight '{loss_weight_val}'. "
                        f"Must be a number."
                    )
            if norm_str is None:
                errors.append(f"  [{i}] '{name}': missing required 'normalization' field")
                continue

            norm_type = _CONF_NORM_MAP.get(str(norm_str).lower())
            if norm_type is None:
                errors.append(
                    f"  [{i}] '{name}': unknown normalization '{norm_str}'. "
                    f"Valid values: {sorted(_CONF_NORM_MAP)}"
                )
                continue


            missing_stat = next(
                (k for k in _REQUIRED_STATS[norm_type] if k not in stats or stats[k] is None),
                None,
            )
            if missing_stat:
                errors.append(
                    f"  [{i}] '{name}': normalization '{norm_str}' requires stat "
                    f"'{missing_stat}', which is not available in stats "
                    f"(available: {[k for k, v in stats.items() if v is not None]})"
                )
                continue

            if norm_type == NormType.ZSCORE:
                beta[i],  alpha[i] = stats["mean"][i],    stats["std"][i]
            elif norm_type == NormType.STD_ONLY:
                beta[i],  alpha[i] = 0.0,                 stats["std"][i]
            elif norm_type == NormType.MINMAX:
                beta[i],  alpha[i] = stats["minimum"][i], stats["maximum"][i] - stats["minimum"][i]
            elif norm_type == NormType.SYM_RANGE:
                mid       = (stats["minimum"][i] + stats["maximum"][i]) / 2.0
                half      = (stats["maximum"][i] - stats["minimum"][i]) / 2.0
                beta[i],  alpha[i] = mid, half
            elif norm_type == NormType.MAX:
                beta[i],  alpha[i] = 0.0,                 stats["maximum"][i]
            elif norm_type == NormType.NONE:
                beta[i],  alpha[i] = 0.0,                 1.0

        if errors:
            raise ValueError(
                f"MultiNormalizer: {len(errors)} variable(s) have configuration errors:\n"
                + "\n".join(errors)
            )

        self.register_buffer("alpha",         alpha.clamp_min(eps))
        self.register_buffer("beta",          beta)
        self.register_buffer("loss_weights", loss_weights)



    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self._broadcast("alpha", x)
        beta  = self._broadcast("beta",  x)
        return (x - beta) / alpha

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self._broadcast("alpha", x)
        beta  = self._broadcast("beta",  x)
        return x * alpha + beta