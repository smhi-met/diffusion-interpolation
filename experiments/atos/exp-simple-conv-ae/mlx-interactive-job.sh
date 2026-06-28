#!/bin/bash

set -e
 

module --force purge 
module load env/release/2024.1 
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

export PYTHONPATH=".:$PYTHONPATH"
python -m dssml.commands.run $@ # $@  hydra overrides
