from abc import ABC, abstractmethod
from copy import deepcopy
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
import torch.nn as nn
from dssml.data.modules import DataModuleBase
import os
import logging

LOGGER = logging.getLogger(__name__)

class ModelBase(L.LightningModule, ABC):
    """
    Base LightningModule that strictly handles optimization and scheduling.
    Useful for classifiers, regressors, or base components that do not generate samples.
    """
    def __init__(
        self, 
        optimizers_conf: list[dict[str, any]],
        normalizer: nn.Module | None = None,
    ):
        super().__init__()
        self.optimizers_conf = optimizers_conf
        self.normalizer = normalizer

        self.save_hyperparameters(ignore=['normalizer'])
 
 
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the model architecture."""
        pass

            
 
    def _build_single_optimizer(self, config_block: dict[str, any]):
        """
        Parses a single block containing {'optimizer': ..., 'scheduler': ...}
        """
        # 1. Instantiate Optimizer
        opt_conf = config_block.get("optimizer")
        if not opt_conf or "_target_" not in opt_conf:
            raise ValueError(f"Missing 'optimizer' or '_target_' in config block: {config_block}")

        # Filter parameters that require gradients (safety for frozen layers)
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = instantiate(opt_conf, params=params)

        # 2. Instantiate Scheduler (Optional)
        sch_conf = config_block.get("scheduler")
        scheduler_dict = None
        
        if sch_conf and "_target_" in sch_conf:
            # Pass the just-created optimizer to the scheduler
            scheduler = instantiate(sch_conf, optimizer=optimizer)
            
            # Construct Lightning's required scheduler dictionary
            scheduler_dict = {
                "scheduler": scheduler,
                "interval": sch_conf.get("interval", "epoch"),
                "frequency": sch_conf.get("frequency", 1),
                "monitor": sch_conf.get("monitor", "val_loss"),
                "strict": sch_conf.get("strict", True),
                "name": sch_conf.get("name", None),
            }

        return optimizer, scheduler_dict

    def configure_optimizers(self):
        """
        Parses self.hparams.optimizers (list of dicts) and returns
        the format expected by PyTorch Lightning.
        """
        optimizers_conf = self.optimizers_conf
        
        if not optimizers_conf or not isinstance(optimizers_conf, (list, ListConfig)):
             raise ValueError(f"Model requires 'optimizers' to be a list. Got {type(optimizers_conf)}")
        final_optimizers = []
        final_schedulers = []

        for block in optimizers_conf:
            opt, sch = self._build_single_optimizer(block)
            final_optimizers.append(opt)
            if sch:
                final_schedulers.append(sch)
        
        # Case 1: Single Optimizer & Scheduler (Most common)
        if len(final_optimizers) == 1 and len(final_schedulers) == 1:
            return {
                "optimizer": final_optimizers[0],
                "lr_scheduler": final_schedulers[0]
            }
        
        # Case 2: Single Optimizer, No Scheduler
        if len(final_optimizers) == 1 and not final_schedulers:
            return final_optimizers[0]

        # Case 3: Multiple Optimizers (e.g. GANs)
        return final_optimizers, final_schedulers



class MultiStageBase(ModelBase, ABC):
    """
    This is a base class for models that have multiple stages (e.g., VAE + Diffusion).
    """
    def __init__(self,
        # 1. ModelBase Parameters
        optimizers_conf: list[dict[str, any]],
        
        # 2. Current class Specific Parameters

        stages_conf: dict[str, any],
        normalizer: nn.Module | None = None,):
    

        super().__init__(optimizers_conf=optimizers_conf, normalizer=normalizer)
        self.save_hyperparameters(ignore=['normalizer'])

        # 1. Use ModuleDict so Lightning tracks these sub-models automatically
        self.frozen_stages = nn.ModuleDict()

        # 2. Iterate and Build/Load each stage
        for stage_name, conf in stages_conf.items():
            self.frozen_stages[stage_name] = self._init_stage(stage_name, conf)
        
        # 3. Global Freeze
        self._freeze_all_stages()


    def _init_stage(self, name: str, conf: dict[str, any]) -> nn.Module:
        _conf = deepcopy(conf)  # Avoid mutating the original config
        ckpt_path = _conf.pop("ckpt_path", None)

        # Apply our "Safe Resume" logic to every stage
        if ckpt_path and os.path.exists(ckpt_path):
            model = instantiate(_conf, _recursive_=False)
            _conf.pop("_target_", None)   
            print(f"[Registry] Loading {name} from {ckpt_path}")
            return model.load_from_checkpoint(ckpt_path, **_conf)
        else:
            raise ValueError(f"Checkpoint path for stage '{name}' is invalid: {ckpt_path}")

    def _freeze_all_stages(self):
            """Iterates through the registry and locks everything."""
            self.frozen_stages.eval()
            for param in self.frozen_stages.parameters():
                param.requires_grad = False

    # Convenience method for child classes
    def get_stage(self, name: str) -> nn.Module:
        return self.frozen_stages[name]             
    
    # 
    def on_train_epoch_start(self):
        """Ensures all registered stages stay in eval mode during training."""
        super().on_train_epoch_start()
        self.frozen_stages.eval()
