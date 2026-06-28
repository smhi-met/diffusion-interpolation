#!/bin/bash -l

#SBATCH --job-name=lola-vae-kl
#SBATCH --account=p200177
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1

echo "Running Slurm Job ID: $SLURM_JOB_ID"


module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0
module load git

OVERLAY=$WORK_DIR/_temp
mkdir -p $OVERLAY

apptainer exec \
    --containall \
    --nv \
    --bind $PWD:/app \
    --bind $WORK_DIR:$WORK_DIR \
    --bind $DATASET_PATH:$DATASET_PATH \
    --bind $OVERLAY:/tmp \
    --bind $ASSETS_DIR:$ASSETS_DIR \
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
        export ASSETS_DIR="${ASSETS_DIR}"
        export CONFIG_FILE="${CONFIG_FILE}"
        export HYDRA_FULL_ERROR=1
        export PYTHONPATH="/app/src:${PYTHONPATH}"

        export TMPDIR=/tmp
        export TMP=/tmp
        export TEMP=/tmp
        date
        python3 -m dssml.commands.run $@
        date
"
