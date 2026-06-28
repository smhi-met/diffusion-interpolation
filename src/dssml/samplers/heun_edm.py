import math
import torch
from .base import SamplerBase

def _karras_sigma_schedule(steps: int, sigma_min: float, sigma_max: float, rho: float, device) -> torch.Tensor:
    i = torch.linspace(0, steps - 1, steps, device=device, dtype=torch.float64)
    inv = 1.0 / rho
    sigmas = (sigma_max**inv + (sigma_min**inv - sigma_max**inv) * i / max(steps - 1, 1)).pow(rho)
    return sigmas

class HeunEDMSampler(SamplerBase):
    def __init__(
        self,
        steps: int = 40,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
    ):
        self.steps = int(steps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.S_churn = float(S_churn)
        self.S_min = float(S_min)
        self.S_max = float(S_max)
        self.S_noise = float(S_noise)

    @property
    def name(self) -> str:
        return "heun_edm"

    @torch.no_grad()
    def sample(self, module, cond, target_shape, device, **kwargs):
        B, C_t, H, W = target_shape
        dtype = cond.dtype

        # Access underlying network if wrapped
        net = getattr(module, "network", module)
        sigma_min = max(self.sigma_min, float(getattr(net, "sigma_min", self.sigma_min)))
        sigma_max = min(self.sigma_max, float(getattr(net, "sigma_max", self.sigma_max)))
        round_sigma = getattr(net, "round_sigma", lambda s: s)

        sigmas = _karras_sigma_schedule(self.steps, sigma_min, sigma_max, self.rho, device=device)
        sigmas = round_sigma(sigmas).to(dtype)
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

        # Initialization
        x = torch.randn(B, C_t, H, W, device=device, dtype=dtype) * sigmas[0]

        for i in range(self.steps):
            s_i = sigmas[i]
            s_j = sigmas[i + 1]
            
            # Churn
            gamma = 0.0
            if self.S_min <= float(s_i) <= self.S_max and self.S_churn > 0:
                gamma = min(self.S_churn / self.steps, math.sqrt(2) - 1.0)

            t_hat = s_i
            if gamma > 0.0:
                t_hat = round_sigma(s_i * (1.0 + gamma)).to(dtype)
                noise = torch.randn_like(x)
                x = x + (t_hat.square() - s_i.square()).clamp(min=0).sqrt() * self.S_noise * noise
            
            # Euler Step
            sigma_in = t_hat.expand(B)
            x0_hat = module._denoise_target(cond, x, sigma_in)
            d_i = (x - x0_hat) / (t_hat + 1e-12)
            
            x_e = x + (s_j - t_hat) * d_i

            # Heun Correction
            if s_j > 0:
                sigma_next = s_j.expand(B)
                x0_hat_j = module._denoise_target(cond, x_e, sigma_next)
                d_j = (x_e - x0_hat_j) / (s_j + 1e-12)
                x = x + (s_j - t_hat) * 0.5 * (d_i + d_j)
            else:
                x = x_e

        return x