#!/bin/bash -l
#SBATCH --job-name=run-salloc
#SBATCH --account=p200177
#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --qos=default
#SBATCH --error=run-salloc.out
#SBATCH --output=run-salloc.out

set -e
 

module --force purge 
module load env/release/2024.1 
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate


sleep 86400
