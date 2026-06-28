import torch.nn.functional as F
from dssml.models.base import ModelBase 
from hydra.utils import instantiate
from dssml.networks.ae import IAutoEncoder
from dssml.data.helpers.normalizers import NormalizerBase
from torch import nn


class SimpleAutoEncoderModel(ModelBase):
    def __init__(self, 
        # ModelBase Parameters
        optimizers_conf: list[dict[str, any]],
        # Current class mandatory parameters
        auto_encoder_conf: dict[str, any],
        # ModelBase Optional Parameters
        normalizer: nn.Module | None = None,):
        # 1. Initialize Parent (saves hparams and sets up optimizers/sampler)
        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        
        self.save_hyperparameters(ignore=['normalizer'])
  
        # 2. Define Architecture
        self.auto_encoder = instantiate(auto_encoder_conf, _recursive_=False)
        # guardrail against misconfiguration where the instantiated auto_encoder doesn't implement the required interface
        if not isinstance(self.auto_encoder, IAutoEncoder):
            raise ValueError(f"{self.__class__.__name__} must implement {IAutoEncoder.__name__}. Got {type(self.auto_encoder)}")
     
    @property
    def name(self):
        return "SimpleConvAutoEncoderModel"
    
      
    def _shared_step(self, batch, batch_idx, prefix="val"):
        x = batch["x"]
        B, T, V, E, H, W = x.shape
   
        if self.normalizer is not None:
            _x = self.normalizer(x).view(B, T* V * E, H, W)   
        z = self.auto_encoder.encode(_x)
        x_hat = self.auto_encoder.decode(z) 
        loss = F.mse_loss(x_hat, _x)
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss


    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")
        
    def validation_step(self, batch, batch_idx):
        # called every 1 epoch during trainer.fit(), immediately after the training loop.
        return self._shared_step(batch, batch_idx, prefix="val")    
    
    def test_step(self, batch, batch_idx):
        # Called only when you explicitly run trainer.test()

        return self._shared_step(batch, batch_idx, prefix="test")    
    
    
    def predict_step(self, batch, batch_idx):
        # Called during trainer.predict()
        x = batch["x"]
        B, T, V, E, H, W = x.shape
        if self.normalizer is not None:
            _x = self.normalizer(x).view(B, T* V * E, H, W) 
        z = self.auto_encoder.encode(_x)
        x_hat = self.auto_encoder.decode(z).view(B, T, V, E, H, W)
        if self.normalizer is not None:
            x_hat = self.normalizer.denormalize(x_hat)
        return x_hat
    
    def forward(self, x):
        # Defines the "Inference API." It should be the simplest path from input to output.
        # Used manually by you (e.g., y_hat = model(x)).
        """Convenience method for encoding and decoding in one step."""

 
        B, T, V, E, H, W = x.shape
        if self.normalizer is not None:
            _x = self.normalizer(x).view(B, T* V * E, H, W) 
        z = self.auto_encoder.encode(_x)
        x_hat = self.auto_encoder.decode(z).view(B, T, V, E, H, W)
        if self.normalizer is not None:
            x_hat = self.normalizer.denormalize(x_hat)
        return x_hat
    

