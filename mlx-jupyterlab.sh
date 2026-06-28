#!/bin/bash -l
#SBATCH --job-name=run-jupyterlab
#SBATCH --account=p200177
#SBATCH -p cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --qos=default
#SBATCH --error=run-jupyterlab.out
#SBATCH --output=run-jupyterlab.out

set -e
 

module --force purge 
module load env/release/2024.1 
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate


python -m jupyterlab --no-browser --ip "*" --notebook-dir "./notebooks"
