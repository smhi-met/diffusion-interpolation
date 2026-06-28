import logging
from dssml.utils.hydra_docerators import hydra_external_config
from hydra.utils import instantiate

LOGGER = logging.getLogger(__name__)

@hydra_external_config()
def main(cfg) -> None:
 
     
    runner = instantiate(cfg,  _recursive_=False, _partial_=False)
    runner.run()
 

if __name__ == "__main__":
    main()