#!/bin/bash
#SBATCH --job-name=check-configs
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --time=1:00:00


set -e
 

module load conda 
conda activate de371-env 
export PYTHONPATH=".:$PYTHONPATH"
python3 -m dssml.commands.check_configs $@ # $@  hydra
