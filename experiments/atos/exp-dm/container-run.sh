module load Apptainer/1.3.6-GCCcore-13.3.0

if [ -z "$SIF_FILE" ]; then
    echo "Error: SIF_FILE environment variable is not set."
    exit 1
fi

if [ -z "$WORK_DIR" ]; then
    echo "Error: WORK_DIR environment variable is not set."
    exit 1
fi

if [ -z "$DATA_DIR" ]; then
    echo "Error: DATA_DIR environment variable is not set."
    exit 1
fi

if [ -z "$ASSETS_DIR" ]; then
    echo "Error: ASSETS_DIR environment variable is not set."
    exit 1
fi

apptainer exec \
    --containall \
    --bind $PWD:/app \
    --bind $WORK_DIR:$WORK_DIR \
    --bind $DATA_DIR:$DATA_DIR \
    --bind $ASSETS_DIR:$ASSETS_DIR \
     $SIF_FILE \
    bash -c "
        set -e
        cd /app
        echo -----------------------
        ls -a1
        echo -----------------------
        echo ls -a1 $WORK_DIR
        echo ls -a1 $DATA_DIR
        export EXPERIMENT_NAME="${EXPERIMENT_NAME}"
        export WORK_DIR="${WORK_DIR}"
        export CONFIG_FILE="${CONFIG_FILE}"
        export HYDRA_FULL_ERROR=1
        export PYTHONPATH="/app/src:${PYTHONPATH}" 
        python3 experiments/exp-dm/zarr-dm.py $@ 
"