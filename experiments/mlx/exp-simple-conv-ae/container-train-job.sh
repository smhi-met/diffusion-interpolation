#!/bin/bash -l

#SBATCH --job-name=check-configs-submit
#SBATCH --account=p200177
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1

set -x 
module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0
apptainer exec \
    --containall \
    --nv \
    --bind $PWD:/app \
    --bind $WORK_DIR:$WORK_DIR \
    --bind $DATASET_PATH:$DATASET_PATH \
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
        python3 -m dssml.commands.run $@ 
"