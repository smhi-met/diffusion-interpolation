#!/bin/bash -l

#SBATCH --job-name=check-configs
#SBATCH --account=p200177
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=2


COMMAND=$1
shift

set -e


module --force purge 
module load env/release/2024.1 
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate


export PYTHONPATH=".:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
$COMMAND python -m dssml.commands.run $@  