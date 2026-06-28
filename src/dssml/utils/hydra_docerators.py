import os
import sys
import logging
import functools
from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose, core
# Import Lightning's rank utilities
from lightning.pytorch.utilities import rank_zero_warn

LOGGER = logging.getLogger(__name__)

def hydra_external_config():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config_file = os.environ.get("CONFIG_FILE")
            project_conf_dir = (Path(__file__).resolve().parents[1] / "conf").resolve()
            
            if config_file:
                config_file = Path(config_file)

            if not config_file or not config_file.is_file():
                raise ValueError(f"No valid config file provided: {config_file}")

            cli_overrides = [arg for arg in sys.argv[1:] if "=" in arg]

            config_file = config_file.resolve()
            external_conf_dir = config_file.parent   
            config_stem = config_file.stem
            
            search_path_override = f"hydra.searchpath=[{project_conf_dir.as_uri()}]"
            all_overrides = [search_path_override] + cli_overrides

            if core.global_hydra.GlobalHydra.instance().is_initialized():
                core.global_hydra.GlobalHydra.instance().clear()

            with initialize_config_dir(version_base="1.3", config_dir=str(external_conf_dir)):
                try:
                    cfg = compose(config_name=config_stem, overrides=all_overrides)
                except Exception as e:
                    LOGGER.error(f"Hydra composition failed: {e}")
                    raise e

            # --- RANK ZERO LOGIC START ---
            
            # Use .get() to avoid ConfigKeyError if the key is missing in struct mode
            work_dir = cfg.get("_work_dir")
            
            if work_dir:
                # Only Rank 0 creates the directory and saves the resolved config
                if os.environ.get("LOCAL_RANK", "0") == "0":
                    os.makedirs(work_dir, exist_ok=True)
                    save_path = Path(work_dir) / "config.yaml"
                    
                    with open(save_path, "w") as f:
                        f.write(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True))
            else:
                rank_zero_warn("Key '_work_dir' not found in config. Skipping config dump.")

            # --- RANK ZERO LOGIC END ---

            return func(cfg, *args, **kwargs)

        return wrapper
    return decorator