# Experiments

- Data Module 
- [Simple Convolutional Autoencoder](#simple-convolutional-autoencoder)


## Simple Convolutional Autoencoder 
### Miluxina



#### simple-ae
```bash
export CMD=sbatch
CONFIG_FILE=experiments/mlx/simple-ae/train-configs.yaml SIF_FILE=_saved/sif_files/container.sif  ./run-container.sh $CMD experiments/mlx/simple-ae/container-train-job.sh
```


#### simple-ae-aa
```bash
export CMD=sbatch
CONFIG_FILE=experiments/mlx/simple-ae/aa.yaml SIF_FILE=_saved/sif_files/container.sif  ./run-container.sh $CMD experiments/mlx/simple-ae/container-train-job.sh
```

 
#### ldm-vae
 ## Summary of Differences Between Configs






```bash
export CMD=sbatch
CONFIG_FILE=experiments/mlx/ldm-vae/mse_kl.yaml  SIF_FILE=_saved/sif_files/container.sif ./run-container.sh $CMD experiments/mlx/ldm-vae/container-train-job.sh
for CF in mse_kl_01 \
          mse_kl_02 \
          mse_kl_03 \
          mse_kl_03_2 \
          mse_kl_04 \
          mse_kl_04_2 \
          mse_kl_05 \
          mse_kl_06; do  CONFIG_FILE=experiments/mlx/ldm-vae/$CF.yaml  SIF_FILE=_saved/sif_files/container.sif ./run-container.sh $CMD experiments/mlx/ldm-vae/container-train-job.sh; done 

```



| Config | Loss | KL Weight | LR | Use Case |
|---|---|---|---|---|
| `mse_only` | MSE | 0 | `1e-4` | Fast baseline, no regularization |
| `mse_kl` | MSE + KL | `1e-6` | `4.5e-6` | Standard VAE, smooth latent space |
| `mse_kl_spectral_gradient.yaml` | MSE + KL + SG | `1e-6` | `4.5e-6` | Best quality, closest to paper |



The key design decisions worth noting: `spectral_weight` and `gradient_weight` are both `0.1` as a starting point — these will likely need tuning. If your reconstructions look spatially blurry (smooth fields where sharp features should exist), increase `gradient_weight`. If large-scale patterns are being lost while small-scale noise is preserved, increase `spectral_weight`. Monitor `val_spectral_loss` and `val_gradient_loss` separately in MLflow to understand which component is driving the loss.



z_channels: 4
ch: 64
ch_mult: [1, 2, 4]
num_res_blocks: 2
attn_resolutions: [32]



#### ldm-vae-anaswer

```bash
export CMD=sbatch
CF=mse_kl_04_0 
CONFIG_FILE=experiments/mlx/ldm-vae-answer/$CF.yaml  SIF_FILE=_saved/sif_files/container.sif ./run-container.sh $CMD experiments/mlx/ldm-vae-answer/container-train-job.sh 

```


#### 01-ldm-vae-norm

##### mse_kl_04_0_20260602_160754
- new scaling where the inverse of the square scales controbuted to the weighting.
- Failed to represent high values but very successful for the high values. mainly scaling problem. still has some blurriness
- git commnit: hash b06ac1125c2335bca42f1fd7ac8786b2d1353b27
```bash
export CMD=sbatch
CF=mse_kl_04_0 
CONFIG_FILE=experiments/mlx/01-ldm-vae-norm/$CF.yaml  SIF_FILE=_saved/sif_files/container.sif ./run-container.sh $CMD experiments/mlx/01-ldm-vae-norm/container-train-job.sh 
```


##### mse_kl_grad_04_0_20260602_202314 
- same as mse_kl_04_0_20260602_160754 but activatted the gradient loss 
- results are alomost the same
- git commnit: hash b06ac1125c2335bca42f1fd7ac8786b2d1353b27  same as mse_kl_04_0_20260602_160754 but change in the configurations

```bash
export CMD=sbatch
CF=mse_kl_grad_04_0
CONFIG_FILE=experiments/mlx/01-ldm-vae-norm/$CF.yaml  SIF_FILE=_saved/sif_files/container.sif ./run-container.sh $CMD experiments/mlx/01-ldm-vae-norm/container-train-job.sh 
```
 


##### mse_kl_04_0_20260603_111444
-  scaling the variables with scale factor was removed and instead only the loss_wwights acts to give importance to diferent variables. z (geopotential) is removed 
- git commnit: hash c4a1b52dd073d7195f2b048dde4a2a70b6aaa4a8
```bash
export CMD=sbatch
CF=mse_kl_04_0 
CONFIG_FILE=experiments/mlx/01-ldm-vae-norm/$CF.yaml  SIF_FILE=_saved/sif_files/container.sif ./run-container.sh $CMD experiments/mlx/01-ldm-vae-norm/container-train-job.sh 
```


 

##### 02-l2-no-norm/mse_kl_04_0_20260608_224928
from paper http://arxiv.org/abs/2410.10733
 - git commnit: hash 18b784dafa28a95c142d7440a5968c8bd59cac75
```bash
export CMD=bash
CF=02-l2-no-norm 
CONFIG_FILE=experiments/mlx/02-ldm-vae-norm/$CF.yaml  SIF_FILE=_saved/sif_files/container.sif ./run-container.sh $CMD experiments/mlx/02-ldm-vae-norm/container-train-job.sh 
```


#####  02-l2-no-norm/mse_kl_04_0_20260608_234422
from paper http://arxiv.org/abs/2410.10733
with residual_autoencoding=True
 - git commnit: hash c238888d86981c59419ccf8e00ee1aa23bfae750
```bash
export CMD=bash
CF=02-l2-no-norm 
CONFIG_FILE=experiments/mlx/02-ldm-vae-norm/$CF.yaml  SIF_FILE=_saved/sif_files/container.sif ./run-container.sh $CMD experiments/mlx/02-ldm-vae-norm/container-train-job.sh 
```


#####  03-lola-no-norm/lola_dcae_20260609_012853
from paper https://arxiv.org/abs/2507.02608 
 - git commnit: hash 38948c44b8fa0542e899568ce536aa71fc3399ab
```bash
export CMD=bash
CF=03-lola-no-norm
CONFIG_FILE=experiments/mlx/03-ldm-ae-norm/$CF.yaml  SIF_FILE=_saved/sif_files/container.sif ./run-container.sh $CMD experiments/mlx/03-ldm-ae-norm/container-train-job.sh 
```


##### 04-qrl-no-norm/qrl_dcae_20260609_012522
from paper https://arxiv.org/abs/2507.02608 
 - git commnit: hash 21774ed88b830b9a58929c1d39a69e344a0faf64
```bash
export CMD=bash
CF=04-qrl-no-norm
CONFIG_FILE=experiments/mlx/04-ldm-vae-no-norm/$CF.yaml  SIF_FILE=_saved/sif_files/container.sif ./run-container.sh $CMD experiments/mlx/04-ldm-vae-no-norm/container-train-job.sh 
```


#### 13-qrl-vae-winds-focal-z64x16x16

QRL Variational Autoencoder — variational extension of exp 09.  
Encoder outputs posterior (μ, logvar); LoLASaturation on μ; MMD regularization (InfoVAE).  
Reconstruction: `mse+wqrl+fl_hist` (same as exp 09). Latent: `(64, 16, 16)`.  
Docs: [docs/qrl-vae.md](../qrl-vae.md)

```bash
export CMD=sbatch
CONFIG_FILE=experiments/mlx/13-qrl-vae-winds-focal-z64x16x16/13-qrl-vae-winds-focal-z64x16x16.yaml \
SIF_FILE=_saved/sif_files/container.sif \
./run-container.sh $CMD experiments/mlx/13-qrl-vae-winds-focal-z64x16x16/container-train-job.sh
```




- container sbatch 

- container interactive
```bash
export CMD=bash
CF=mse_kl_04_0 
CONFIG_FILE=experiments/mlx/01-ldm-vae-norm/$CF.yaml SIF_FILE=_saved/sif_files/container.sif ./run-container.sh bash experiments/mlx/01-ldm-vae-norm/container-train-job.sh 
```


- container interactive
```bash
export CMD=bash
CF=mse_kl_04_0 
CONFIG_FILE=experiments/mlx/01-ldm-vae-norm/${FC}_resume.yaml SIF_FILE=_saved/sif_files/container.sif ./run-container.sh bash experiments/mlx/01-ldm-vae-norm/container-train-job.sh 
```