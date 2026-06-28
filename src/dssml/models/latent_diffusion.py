from lightning.pytorch import LightningModule
from hydra.utils import instantiate, get_class
import torch

class LatentDiffusion(LightningModule):
    def __init__(
        self,
        first_stage_config: dict,
        first_stage_ckpt: str,
        unet_config: dict,
        scale_factor: float = 0.18215,
    ):
        super().__init__()
        
        # --- STAGE 1: LOAD THE VAE ---
        # We don't just instantiate(); we load weights directly from the ckpt.
        # 1. Get the class type (e.g., AutoencoderKL) from the config
        first_stage_cls = get_class(first_stage_config["_target_"])
        
        # 2. Load weights using Lightning's built-in method
        # strict=False is often useful if your VAE ckpt has extra keys (like loss/discriminator)
        # that aren't in the inference model.
        print(f"Loading first stage from {first_stage_ckpt}")
        self.first_stage_model = first_stage_cls.load_from_checkpoint(
            first_stage_ckpt, 
            **first_stage_config  # Pass structure args (channels, dim, etc.)
        )
        
        # --- STAGE 2: FREEZE IMMEDIATELY ---
        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)
        
        # --- STAGE 3: REST OF MODEL ---
        self.unet = instantiate(unet_config)
        self.scale_factor = scale_factor

    def on_train_start(self):
        # Double check: Ensure VAE stays in eval mode (locks BatchNorm, Dropout)
        self.first_stage_model.eval()