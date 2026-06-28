#!/bin/bash -l

#SBATCH --job-name=check-configs
#SBATCH --account=p200177
#SBATCH --time=00:15:00
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1



set -e
 

module --force purge 
module load env/release/2024.1 
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate


export PYTHONPATH=".:$PYTHONPATH"
python3 -m dssml.commands.check_configs $@ # $@  hydra
