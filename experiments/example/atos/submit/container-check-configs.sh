#!/bin/bash
#SBATCH --job-name=check-configs
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --time=1:00:00




module load apptainer/1.3.6

apptainer exec \
    --containall \
    --bind $PWD:/app \
    --bind $WORK_DIR:$WORK_DIR \
     $SIF_FILE \
    bash -c "
        set -e
        cd /app
        echo -----------------------
        ls -a1
        echo -----------------------
        echo ls -a1 $WORK_DIR
        export EXPERIMENT_NAME="${EXPERIMENT_NAME}"
        export WORK_DIR="${WORK_DIR}"
        export CONFIG_FILE="${CONFIG_FILE}"
        export HYDRA_FULL_ERROR=1
        export PYTHONPATH="/app/src:${PYTHONPATH}" 
        python3 -m dssml.commands.check_configs $@ 
"