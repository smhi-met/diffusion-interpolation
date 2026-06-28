import torch
import torch.nn as nn
from .base import IAutoEncoder
from dssml.networks.ae.model import Encoder, Decoder

class VarAutoEncoder(nn.Module, IAutoEncoder):
    def __init__(self, auto_encoder_conf: dict[str, any]):
        super().__init__()
        self.encoder = Encoder(auto_encoder_conf, _recursive_=False)
        self.decoder = Decoder(auto_encoder_conf, _recursive_=False)
    
    def encode(self, x):
        return self.encoder(x) 

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
