from abc import ABC, abstractmethod
import torch

class SamplerBase(ABC):
    """
    Abstract base for diffusion samplers.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def sample(self, module, **kwargs) -> torch.Tensor:
        """
        Generates samples using the provided module (model).
        """
        ...