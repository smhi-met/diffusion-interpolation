from __future__ import annotations
import torch

from dssml.models.base import ModelBase 

from .base import IAutoEncoder
 

class IdentityAutoEncoder(ModelBase, IAutoEncoder):
    
    @property
    def name(self) -> str:
        return "identity_ae"
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z