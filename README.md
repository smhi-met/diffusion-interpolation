# Difuusion Interpolation
## Table of files
  - ##### Python setup
    - [Python Setup on Atos](./docs/env-on-atos.md)
    - [Python Setup on MeluXina](./docs/env-on-mlx.md)
  - ##### Experiments files
    - [Experiments](./docs/exp).

## Setup
### Python 
See [here](./README.md#python-setup)

### Directories 
- Make a folder or link a directory called `_work`.
- Inside this folder all run output will be stored.
- Each run has an id and the output folder name will be the same as the run id.


### ML FLow
if activated one have to do this before the run
```bash
python3 -m mlflow server \
  --host localhost\
  --port 5000 \
  --backend-store-uri file:$(pwd)/_saved/mlflow \
  --default-artifact-root  file:$(pwd)/_saved/mlflow
```

### pre-commit hooks
To install the pre-commit hooks run the following command in the terminal from the project root directory
```bash
pre-commit install
```


## Experiments
Eech person has a file to list the experiments one's do. See [here](./README.md#experiments-files )
 
Each experiment need to have its own configurations overrides in a config file `some-configs.yml` and a job file with a name example `some-job.sh`

To submit a job we need to do it in this way from the home directory
```bash
CONFIG_FILE=/path/to/job-file/some-configs.yaml ./run.sh sbatch /path/to/job-file/some-job.sh <hydra_args>
```

To just run the job in the terminal just run:
```bash
CONFIG_FILE=/path/to/job-file/some-configs.yaml ./run.sh bash /path/to/job-file/some-job.sh <hydra_args>
```
`hydra_args` are optional.


in commands `bash` may  need tp be repaced by `srun` for interatice environments.
### MeluXina
```bash

CONFIG_FILE=experiments/example/meluxina/check-configs/some-simple-configs.yaml ./run.sh bash experiments/example/meluxina/check-configs/check-job.sh
CONFIG_FILE=experiments/example/meluxina/check-configs/some-configs.yaml ./run.sh bash experiments/example/meluxina/check-configs/check-job.sh

```
Here is an example to submit a job to the cluster
```bash
CONFIG_FILE=experiments/example/meluxina/submit/configs-submit.yaml ./run.sh sbatch experiments/example/meluxina/submit/check-configs.sh 
```

### Atos
```bash
CONFIG_FILE=experiments/example/atos/check-configs/some-simple-configs.yaml ./run.sh bash experiments/example/atos/check-configs/check-job.sh 
CONFIG_FILE=experiments/example/atos/check-configs/some-configs.yaml ./run.sh bash experiments/example/atos/check-configs/check-job.sh
```

Here is an example to submit a job to the cluster
```bash
CONFIG_FILE=experiments/example/atos/submit/configs-submit.yaml ./run.sh sbatch experiments/example/atos/submit/check-configs.sh 
```
### Shared
- Each config file **should have**  the following content:
```yaml
_experiment_name: "check-simple-configs"
_run_name: "test_${now:%Y%m%d_%H%M%S}"
_work_dir: ${hydra:runtime.cwd}/_work/${_experiment_name}/${_run_name}
_dataset_path: "path/to/dataset"
```
Otherwize it will fail. the values are not suposed to be the same as the example but the keys should be there.

`_dataset_path` is optional but needed when working with containrized environment.

- To control the logging we need to have this **optional** content in the config file or in the main config.yaml file

```yaml
hydra:
  job_logging:
    root:
      level: INFO
    loggers:
      # Specific control over libraries
      lightning.pytorch:
        level: INFO
      # Your project code (replace 'src' with your package name)
      src:
        level: INFO
```



## Using the contianer

## Build the container
Each time we change essential things in `pyproject.toml` we need to rebuild the container. To do that we need to run the following command in the terminal from the project root directory

### MeluXina
#### Build SIF file
```bash
module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0
mkdir -p ./_work/apptainer_tmp
export APPTAINER_TMPDIR=$(pwd)/_work/apptainer_tmp
export SINGULARITY_TMPDIR=$(pwd)/_work/apptainer_tmp
mkdir -p _saved/sif_files
apptainer build --fakeroot _saved/sif_files/container.sif  ./container/container.def
rm -rf ./_work/apptainer_tmp
```


#### Container Examples 
##### Interactive 
```bash
CONFIG_FILE=experiments/example/meluxina/check-configs/some-simple-configs.yaml SIF_FILE=_saved/sif_files/container.sif ./run-container.sh bash experiments/example/meluxina/check-configs/check-job-container.sh
```
```bash
CONFIG_FILE=experiments/example/meluxina/check-configs/some-configs.yaml SIF_FILE=_saved/sif_files/container.sif ./run-container.sh bash experiments/example/meluxina/check-configs/check-job-container.sh
```
##### Job submission
```bash
CONFIG_FILE=experiments/example/atos/submit/configs-submit.yaml SIF_FILE=_saved/sif_files/container.sif ./run-container.sh sbatch experiments/example/atos/submit/container-check-configs.sh 
```

### Atos
#### Build SIF file
```bash
module load apptainer/1.3.6
mkdir -p ./_work/apptainer_tmp
export APPTAINER_TMPDIR=$(pwd)/_work/apptainer_tmp
export SINGULARITY_TMPDIR=$(pwd)/_work/apptainer_tmp
mkdir -p _saved/sif_files
apptainer build --fakeroot _saved/sif_files/container.sif  ./container/container.def
rm -rf ./_work/apptainer_tmp
```
  
#### Container Examples 

##### Interactive 
```bash
CONFIG_FILE=experiments/example/atos/check-configs/some-simple-configs.yaml SIF_FILE=_saved/sif_files/container.sif ./run-container.sh bash experiments/example/atos/check-configs/check-job-container.sh
```
```bash
CONFIG_FILE=experiments/example/atos/check-configs/some-configs.yaml SIF_FILE=_saved/sif_files/container.sif ./run-container.sh bash experiments/example/atos/check-configs/check-job-container.sh
```
##### Job submission
```bash
CONFIG_FILE=experiments/example/atos/submit/configs-submit.yaml SIF_FILE=_saved/sif_files/container.sif ./run-container.sh sbatch experiments/example/atos/submit/container-check-configs.sh 
```

One may need an over lay for large temp files:
```bash
apptainer overlay create --size 2048 --create-dir /var/cache/sometemp /tmp/sometemp.img
```


# TODOs


###################### OLD ################################


## Environment on MiluXina
```bash
# start an interactive session
salloc -A p200177  -p gpu --qos default -N 1 -t 1:00:00

#Create the environment
module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0 
python -m venv .venv 
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# prepare mlflow directory 
# --backend-store-uri: The same value set to _mlflow_dir in the config files 
python3 -m mlflow server \
  --host localhost\
  --port 5000 \
  --backend-store-uri file:path/to/the/directory

# submit the job 
sbatch submit_meps_ddp.sh
```

## Prepare input data from MEPS npy 
```bash

python make_sequences.py \
  --labels labels.csv \
  --root /mnt/tier2/project/p200177/DE_371/datasets/datasets_SMHI/npy_intep/samples \
  --windows 0-6 \
  --members 0,1 \
  --start-date 2023-01-01T00:00:00Z \
  --end-date 2023-02-28T00:00:00Z \
  --merge-root merged_samples \
  --out sequences-reduced-test.csv  \
  --extra-out sequences-for-stats-reduced-test.csv \
  --no-verify-fs

python correct_merged.py  sequences-reduced-test.csv  --root  merged_samples --out  sequences-reduced-test-corrected.csv 

python make_stats.py --file-list sequences-for-stats-reduced-test.csv --root  merged_samples --out sequences-stats-reduced-test.npz


```
## Build apptainer the image
```bash
module load Apptainer/1.3.6-GCCcore-13.3.0
apptainer build --fakeroot ./_work/containers/container.sif  container.def
```

## Test Data module
 
```bash

 

module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0
module load git
source /mnt/tier2/project/p200177/u101329/diffusion-interp/.venv/bin/activate


ROOT_DIR=/mnt/tier2/project/p200177/u101329/diffusion-interp-phase2
DATA_BIND=/project/home/p200177/u101329/DE371_bis/MEPS_subdomain/
WORK_DIR=$ROOT_DIR/_work
MLFLOW_DIR=/mnt/tier2/project/p200177/u101329/_mlruns_apptainer
SIF_FILE=../containers/container.sif 
apptainer exec \
    --containall \
    --bind .:/code \
    --bind $DATA_BIND:$DATA_BIND \
    --bind $ROOT_DIR:$ROOT_DIR \
    --bind $WORK_DIR:$WORK_DIR \
    --bind $MLFLOW_DIR:$MLFLOW_DIR \
    $SIF_FILE \
    bash -c "
        set -e
        cd /code
        export ROOT_DIR=$ROOT_DIR
        export WORK_DIR=$WORK_DIR
        export MLFLOW_DIR=$MLFLOW_DIR
        export HYDRA_FULL_ERROR=1
        export PYTHONPATH=".:$PYTHONPATH" 
        python test_meps_zarr_datamodule.py -cn  ddp_zarr_config.yaml 
    "
```


## Train on interactive
 
```bash

 

module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0
module load git
source /mnt/tier2/project/p200177/u101329/diffusion-interp/.venv/bin/activate


ROOT_DIR=/mnt/tier2/project/p200177/u101329/diffusion-interp-phase2
DATA_BIND=/project/home/p200177/u101329/DE371_bis/MEPS_subdomain/
WORK_DIR=$ROOT_DIR/_work
MLFLOW_DIR=/mnt/tier2/project/p200177/u101329/_mlruns_apptainer
SIF_FILE=../containers/container.sif 
apptainer exec \
    --nv \
    --containall \
    --bind .:/code \
    --bind $DATA_BIND:$DATA_BIND \
    --bind $ROOT_DIR:$ROOT_DIR \
    --bind $WORK_DIR:$WORK_DIR \
    --bind $MLFLOW_DIR:$MLFLOW_DIR \
    $SIF_FILE \
    bash -c "
        set -e
        cd /code
        export ROOT_DIR=$ROOT_DIR
        export WORK_DIR=$WORK_DIR
        export MLFLOW_DIR=$MLFLOW_DIR
        export HYDRA_FULL_ERROR=1
        export PYTHONPATH=".:$PYTHONPATH" 
        python3 -u train.py -cn ddp_zarr_config.yaml
    "
```


## Sample 

- read the example  [here](generate/README.md)
