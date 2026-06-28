# Experiments

- Data Module 
- [Simple Convolutional Autoencoder](#simple-convolutional-autoencoder)


## Simple Convolutional Autoencoder 
### Miluxina



#### single-node-sbatch-container
```bash
export CMD=sbatch
CONFIG_FILE=experiments/mlx/single-node-sbatch-container/train-configs.yaml SIF_FILE=_saved/sif_files/container.sif  ./run-container.sh $CMD experiments/mlx/single-node-sbatch-container/container-train-job.sh
```

- location of experiment files: `experiments/mlx/exp-simple-conv-ae`
```bash
export CMD=bash
CONFIG_FILE=experiments/mlx/exp-simple-conv-ae/train-configs.yaml ./run.sh $CMD experiments/mlx/exp-simple-conv-ae/train-job.sh

## Container
export CMD=bash
CONFIG_FILE=experiments/mlx/exp-simple-conv-ae/train-configs.yaml SIF_FILE=_saved/sif_files/container.sif  ./run-container.sh $CMD experiments/mlx/exp-simple-conv-ae/container-train-job.sh



## Submit
export CMD=sbatch
CONFIG_FILE=experiments/mlx/exp-simple-conv-ae/train-configs.yaml SIF_FILE=_saved/sif_files/container.sif  ./run-container.sh $CMD experiments/mlx/exp-simple-conv-ae/container-train-job.sh

```

### Atos
- location of experiment files: `experiments/exp-simple-conv-ae`

```bash
CONFIG_FILE=experiments/exp-simple-conv-ae/atos-train-configs.yaml ./run.sh bash experiments/exp-simple-conv-ae/atos-train-job.sh
```

