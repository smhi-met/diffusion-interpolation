#!/bin/bash

set -e
 

module load conda 
conda activate de371-env

export PYTHONPATH=".:$PYTHONPATH"
python -m dssml.commands.check_configs $@ # $@  hydra overrides
