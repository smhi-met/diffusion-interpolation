import logging
from dssml.utils.hydra_docerators import hydra_external_config

# Set up simple logging to match Hydra's style
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("config_viewer")

@hydra_external_config()
def main(cfg) -> None:
 
    print("\n--- FULL RESOLVED CONFIGURATION ---")
    print(cfg)
    print("-------------------------------------\n")
            
 
if __name__ == "__main__":
    main()