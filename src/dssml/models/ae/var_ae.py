import torch
from torch import nn
import torch.nn.functional as F
from dssml.models.base import ModelBase 
from hydra.utils import instantiate
from dssml.networks.ae import IAutoEncoder
from dssml.data.helpers.normalizers import NormalizerBase

from dssml.networks.ae.model import Encoder, Decoder
from dssml.networks.ae.distributions import DiagonalGaussianDistribution

# From AutoencoderKL (https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py)
class VarAutoEncoderModel(ModelBase):
    def __init__(self, 
        # ModelBase Parameters
        optimizers_conf: list[dict[str, any]],
        # Current class mandatory parameters
        auto_encoder_conf: dict[str, any],
        embed_dim,
        # ModelBase Optional Parameters
        normalizer: nn.Module | None = None,):

        # 1. Initialize Parent (saves hparams and sets up optimizers/sampler)
        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        
        self.save_hyperparameters(ignore=['normalizer'])
  
        # 2. Define Architecture
        self.encoder = Encoder(auto_encoder_conf, _recursive_=False)
        self.decoder = Decoder(auto_encoder_conf, _recursive_=False)
        self.auto_encoder = instantiate(auto_encoder_conf, _recursive_=False)
        self.loss = instantiate(auto_encoder_conf, _recursive_=False)
        # guardrail against misconfiguration where the instantiated auto_encoder doesn't implement the required interface
        if not isinstance(self.auto_encoder, IAutoEncoder):
            raise ValueError(f"{self.__class__.__name__} must implement {IAutoEncoder.__name__}. Got {type(self.auto_encoder)}")

        assert auto_encoder_conf["double_z"]
        self.quant_conv = nn.Conv2d(2*auto_encoder_conf["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, auto_encoder_conf["z_channels"], 1)
        self.embed_dim = embed_dim
     
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

    
    #def predict_step(self, batch, batch_idx):
    #    # Called during trainer.predict()
    #    x = batch["x"]
    #    B, T, V, E, H, W = x.shape
    #    if self.normalizer is not None:
    #        _x = self.normalizer(x).view(B, T* V * E, H, W) 
    #   z = self.auto_encoder.encode(_x)
    #    x_hat = self.auto_encoder.decode(z).view(B, T, V, E, H, W)
    #    if self.normalizer is not None:
    #        x_hat = self.normalizer.denormalize(x_hat)
    #    return x_hat
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    
    #def get_input(self, batch, k):
    #    x = batch[k]
    #    if len(x.shape) == 3:
    #        x = x[..., None]
    #    x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
    #    return x

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def training_step(self, batch, batch_idx, optimizer_idx):
        #inputs = self.get_input(batch, self.image_key)
        x = batch["x"]
        B, T, V, E, H, W = x.shape
   
        if self.normalizer is not None:
            _x = self.normalizer(x).view(B, T* V * E, H, W)   
        reconstructions, posterior = self(x)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(x, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(x, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        #inputs = self.get_input(batch, self.image_key)
        x = batch["x"]
        B, T, V, E, H, W = x.shape
   
        if self.normalizer is not None:
            _x = self.normalizer(x).view(B, T* V * E, H, W)   
        reconstructions, posterior = self(x)
        aeloss, log_dict_ae = self.loss(x, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(x, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

