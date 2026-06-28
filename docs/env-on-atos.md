# On Atos
## interactive session
```bash
ecinteractive  -g  -c 4 -t 1:00:00
# to kill the job
ecinteractive -g -k
# or by scancel and job id
scancel <job_id>
```


## Mamba
### Activation 
If already created the environment before, just activate it by running else follow the steps to create it and then activate it.
```bash
conda activate de371-env 
```
### Env setup
```bash 

conda config --remove-key pkgs_dirs || true
conda config --remove-key envs_dirs || true

mkdir $HPCPERM/$USER/conda/pkgs -p
mkdir $HPCPERM/$USER/conda/envs -p
conda config --add pkgs_dirs /perm/$USER/conda/pkgs
conda config --add envs_dirs /perm/$USER/conda/envs
# close and open a new terminal to apply the changes
conda create -n mamba-env -c conda-forge mamba
conda activate mamba-env
mamba create -n de371-env python=3.12  pip
mamba activate de371-env 
# install packages
pip install -e . 
```

##
