#!/bin/bash -l
#SBATCH --job-name=lnd-generate
#SBATCH --account=p200177
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
date

REPO=/mnt/tier2/project/p200177/u101329/DE371_bis/diffusion-interpolation

module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0
source "${REPO}/.venv/bin/activate"

export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

python3 "${REPO}/src/dssml/runners/_lnd_interpolator.py"

date
echo "Done."
