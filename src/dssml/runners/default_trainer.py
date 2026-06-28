import logging
import lightning as L
from copy import deepcopy
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from .base import RunnerBase
from collections.abc import Mapping
import torch 

LOGGER = logging.getLogger(__name__)


class DefaultTrainer(RunnerBase):
    def __init__(
        self,
        resume = False,
        plfit=None,
        pltrainer=None,
        manifest=None,
        normalizer=None,
        model=None,
        datamodule=None,
        callbacks=None,
        logger=None,
        profiler=None,
        **kwargs
    ):

        self.resume = resume
        self.plfit_conf = plfit
        self.pltrainer_conf = pltrainer
        self.manifest_conf = manifest
        self.normalizer_conf = normalizer
        self.model_conf = model
        self.datamodule_conf = datamodule
        self.callbacks_conf = callbacks
        self.logger_conf = logger
        self.profiler_conf = profiler
 

    @property
    def name(self) -> str:
        return "default_trainer"

    def _instantiate_callbacks(self) -> list:
        if not self.callbacks_conf:
            return []

        # If it's a list (ListConfig), instantiate iterates through it automatically
        if isinstance(self.callbacks_conf, (list, ListConfig)):
            return [instantiate(c) for c in self.callbacks_conf]

        # Fallback if you ever switch to a dict-based config
        elif isinstance(self.callbacks_conf, (dict, DictConfig, Mapping)):
            return [instantiate(v) for v in self.callbacks_conf.values()]

        raise ValueError(
            f"Expected callbacks to be a list or dict, got {type(self.callbacks_conf)}"
        )
    
 


 





    def run(self):
 

        chpt_path = self.plfit_conf.get("ckpt_path", None) if self.plfit_conf else None
        if self.resume:
            if chpt_path is None:
                LOGGER.error(
                    "Resume is set to True but no checkpoint path provided in plfit.ckpt_path. Cannot resume training."
                )
                raise ValueError("Resume is set to True but no checkpoint path is provided.")
        else:
            if chpt_path is not None:
                LOGGER.error(
                    f"plfit.ckpt_path {chpt_path} is set but resume is False. To resume training from a checkpoint, set resume to True."
                )
                raise ValueError("plfit.ckpt_path is set but resume is False.")
            

        #    
        LOGGER.info("Instantiating DataProvider...")
        manifest = instantiate(self.manifest_conf, _recursive_=False)


        LOGGER.info("Instantiating and Configuring Normalizer...")
        normalizer = (
            instantiate(
                self.normalizer_conf,
                stats=manifest.stats,
                var_dim=manifest.var_dim,
                variables_conf=manifest.variables_conf,
            )
            if self.normalizer_conf
            else None
        )

        LOGGER.info("Instantiating Model...")
        model = instantiate(
            self.model_conf, normalizer=normalizer, _recursive_=False
        )

        LOGGER.info("Instantiating DataModule...")
        datamodule = instantiate(
            self.datamodule_conf,
            manifest=manifest,
            _recursive_=False,
        )


        LOGGER.info("Instantiating Components...")
        callbacks = self._instantiate_callbacks()
        logger = instantiate(self.logger_conf) if self.logger_conf else None
        pltrainer_args = dict(self.pltrainer_conf or {})

        trainer = L.Trainer(
            **pltrainer_args, callbacks=callbacks, logger=logger
        )
        
 
           
        LOGGER.info("Starting Training...")
        plfit_args = self.plfit_conf or {}
        torch.set_float32_matmul_precision("medium")
        trainer.fit(model, datamodule=datamodule, **plfit_args)
