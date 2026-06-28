import torch
import math
from .base import SamplerBase

def _make_alpha_bar(T, s=0.008, device="cpu"):
    t = torch.arange(0, T+1, device=device, dtype=torch.float32)
    f = lambda u: torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2
    ab = f(t / T) / f(torch.tensor(0.0, device=device))
    return ab.clamp(1e-8, 1.0)

class DDIMSampler(SamplerBase):
    def __init__(self, T: int = 50, eta: float = 0.0, cosine_s: float = 0.008):
        self.T = int(T)
        self.eta = float(eta)
        self.cosine_s = float(cosine_s)

    @property
    def name(self) -> str:
        return "ddim"

    @torch.no_grad()
    def sample(self, module, cond, target_shape, device, **kwargs):
        B, C_t, H, W = target_shape
        ab = _make_alpha_bar(self.T, s=self.cosine_s, device=device)
        
        x_t = torch.randn(B, C_t, H, W, device=device)
        
        for t in range(self.T, 0, -1):
            alpha_bar_t = ab[t]
            alpha_bar_prev = ab[t-1]
            
            # Module must expose _denoise_target
            sigma_t_in = torch.sqrt((1 - alpha_bar_t) / alpha_bar_t)
            sigma_vec = torch.full((B,), sigma_t_in.item(), device=device)
            
            x0_hat = module._denoise_target(cond, x_t, sigma_vec)
            
            eps_hat = (x_t - alpha_bar_t.sqrt() * x0_hat) / (1 - alpha_bar_t).sqrt()
            x_mean = alpha_bar_prev.sqrt() * x0_hat + (1 - alpha_bar_prev).sqrt() * eps_hat
            
            if self.eta == 0.0:
                x_t = x_mean
            else:
                sigma_ddim = self.eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
                x_t = x_mean + sigma_ddim * torch.randn_like(x_t)
                
        return x_t.clamp(-1, 1)