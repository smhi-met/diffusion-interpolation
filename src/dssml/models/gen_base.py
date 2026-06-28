
from abc import ABC, abstractmethod
import lightning as L
from hydra.utils import instantiate, get_class
import torch
import torch.nn as nn



from .base import ModelBase  
from dssml.networks.ae import IAutoEncoder
class GenerativeModelBase(ModelBase):
    """
    Specialized LightningModule for Generative Models.
    Adds support for a sampling strategy.
    """
    def __init__(
        self, 
        optimizers_conf: list[dict[str, any]],
        sampler: dict[str, any], 
    ):
        # Pass optimizers up to the parent
        super().__init__(optimizers_conf=optimizers_conf)
        
        # Save sampler config specifically to hparams if not already captured by super
        self.save_hyperparameters(logger=False)

        # Instantiate Sampler immediately
        # Expects: _target_: dssml.samplers.X
        self.sampler = instantiate(sampler)

    @property
    @abstractmethod
    def get_accepted_samplers(self) -> str:
        """List of accepted samplers for this generative model."""
        pass

    def sample(self, **kwargs) -> torch.Tensor:
        """
        Public generation API.
        Delegates generation to the internal sampler strategy.
        """
        if not self.sampler:
            raise NotImplementedError("No sampler defined for this generative model.")
            
        # We pass 'self' (the model instance) so the sampler can access the model's internals
        # Note: We assume the sampler handles device placement via module.device
        return self.sampler.sample(module=self, **kwargs)
    


class LatentGenerativeModelBase(GenerativeModelBase, ABC):
    """
    Base class for Generative Models that operate in a compressed latent space (e.g., LDM).
    
    Responsibilities:
    1. Loads and freezes the First Stage Model (VAE/VQGAN) from a checkpoint.
    2. Manages the 'scale_factor' to normalize latent variance.
    3. Provides concrete implementations for encoding/decoding via the first stage.
    """
    def __init__(
        self, 
        first_stage: dict[str, any],
        scale_factor: float = 0.18215, # Default for KL-f8 (Stable Diffusion)
        **kwargs
    ):
 
        super().__init__(**kwargs)
        self.first_stage_conf = first_stage
        
        
        self.register_buffer("scale_factor", torch.tensor(scale_factor))

    def setup(self, stage):
        super().setup(stage)

        self.first_stage_model = self._build_first_stage(self.first_stage_conf)
        self._freeze_first_stage()

    def _build_first_stage(self, config: dict[str, any]) -> nn.Module:
            from copy import deepcopy
            
            # Work on a copy to avoid mutating the original config/hparams
            conf = deepcopy(config)
            
            ckpt_path = conf.pop("ckpt_path", None)
            target = conf.pop("_target_", None)

            if not target:
                raise ValueError("first_stage_config must contain '_target_'")
            if not ckpt_path:
                raise ValueError("first_stage_config must include 'ckpt_path'")

            model_cls = get_class(target)
            
            # Check inheritance
            if not issubclass(model_cls, IAutoEncoder):
                raise TypeError(f"{model_cls.__name__} must inherit from IAutoEncoder")
            return model_cls.load_from_checkpoint(ckpt_path, strict=False, **conf)
    
    def _freeze_first_stage(self):
        """
        Helper: Locks the first stage model (eval mode + no grads).
        """
        if hasattr(self, "first_stage_model"):
            self.first_stage_model.eval()
            self.first_stage_model.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Concrete implementation: Encodes x -> latent z using the frozen first stage.
        Handles both VQModel (discrete) and AutoencoderKL (probabilistic).
        """
        self.first_stage_model.eval()
        posterior = self.first_stage_model.encode(x)
        
        # Handle different VAE return types
        if hasattr(posterior, "sample"):
            z = posterior.sample() # KL-Autoencoder
        elif isinstance(posterior, (tuple, list)):
            z = posterior[0]       # VQGAN
        else:
            z = posterior          # Standard AE
            
        return z * self.scale_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Concrete implementation: Decodes latent z -> image x.
        """
        self.first_stage_model.eval()
        # Remove scaling before decoding
        z = (1.0 / self.scale_factor) * z
        return self.first_stage_model.decode(z)

    def on_train_epoch_start(self):
        """
        Lightning Hook: Final safety check to lock VAE in eval mode.
        """
        super().on_train_epoch_start()
        self._freeze_first_stage()
