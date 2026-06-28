#!/bin/bash

set -e
 

module load conda 
conda activate de371-env

export PYTHONPATH=".:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
srun python -m dssml.commands.run $@  