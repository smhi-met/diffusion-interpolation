#!/bin/bash

source scripts/hydrate.sh


COMMAND=$1
shift
SCRIPT_FILE=$1
shift


function print_usage {
  echo "Usage: CONFIG_FILE=<config_file> SIF_FILE=<sif_file> $0 [bash, sbatch]  <script_file> [hydra_args]"
  exit 1
}

if [ "x$COMMAND" = "x" ]
then
  print_usage
fi

if [ "x$SIF_FILE" = "x" ]
then
  print_usage
fi

if [ "x$SCRIPT_FILE" = "x" ]
then
  print_usage
fi

if [ "x$CONFIG_FILE" = "x" ]
then
  print_usage
fi


read WORK_DIR DATASET_PATH ASSETS_DIR< <(hydrate_config $CONFIG_FILE)


CONFIG_FILE="$WORK_DIR/config.yaml"
mkdir -p "$WORK_DIR"

if [ "$COMMAND" = "bash" ] || [ "$COMMAND" = "srun" ]; then
  SIF_FILE=$SIF_FILE CONFIG_FILE="$CONFIG_FILE" DATASET_PATH="$DATASET_PATH" WORK_DIR="$WORK_DIR" ASSETS_DIR="$ASSETS_DIR" $COMMAND $SCRIPT_FILE "$@"
elif [ "$COMMAND" = "sbatch" ]; then
  export WORK_DIR
  export SIF_FILE
  export CONFIG_FILE
  export DATASET_PATH
  export ASSETS_DIR
  sbatch \
    --output=$WORK_DIR/output.out \
    --error=$WORK_DIR/error.out \
    --export=WORK_DIR,SIF_FILE,CONFIG_FILE,DATASET_PATH,ASSETS_DIR \
    $SCRIPT_FILE "$@"
else
  echo "Error: Unknown command '$COMMAND'. Use 'bash' or 'sbatch'." >&2
  print_usage
fi
 