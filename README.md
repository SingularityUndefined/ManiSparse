# ManiSparse
# Lightweight Transformer via Unrolling of Mixed Graph Algorithms for Traffic Forecast
## Requirements

Required packages for this implementation:

```
torch>=2.4.1
tqdm 
numpy 
matplotlib 
networkx>=2.5
pandas
pyyaml
tensorboard
```
Install the dependencies in your Python environment before running training.

## Datasets
**PEMS0X datasets** are from repository [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data). The training code expects them under `../TS_datasets/PEMS0X_data`:

<!-- PEMS-BAY and METR-LA datasets are from repository [DCRNN](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph). -->
```
../TS_datasets/
├── PEMS0X_data/
│   ├── PEMS03/
│   ├── PEMS04/
│   ├── PEMS07/
│   └── PEMS08/
├── PEMS-BAY/
└── METR-LA/
```
**PEMS-BAY** and **METR-LA** dataset are preprocessed from repository [DCRNN](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph). Each folder contains two `.npy` files for adjacency matrix and time series data.

We also provide our dataset together with [Google Drive](https://drive.google.com/drive/folders/1tMgyxzQ_dio73rapQ-LYSOOIXwXFFUJw?usp=share_link). Download from this link and put it outside this repository folder as `../TS_datasets`.

### Dataset durations and split sizes

All traffic datasets below use 5-minute sampling, so one day has 288 time steps.
The current training config uses `t_in=12`, `t_out=12`, `T=24`, and `data_stride=3`.
The code splits each dataset as train/val/test = 60%/20%/20%, then reports dataset
length as `(split_steps - T) // data_stride`.

| Dataset | Data shape `(steps, nodes, channels)` | Total days | Total weeks | Train days / weeks / samples | Val days / weeks / samples | Test days / weeks / samples |
|---|---:|---:|---:|---:|---:|---:|
| PEMS03 | `(26208, 358, 1)` | 91.00 | 13.00 | 54.60 / 7.80 / 5233 | 18.20 / 2.60 / 1739 | 18.20 / 2.60 / 1739 |
| PEMS04 | `(16992, 307, 3)` | 59.00 | 8.43 | 35.40 / 5.06 / 3390 | 11.80 / 1.69 / 1124 | 11.80 / 1.69 / 1125 |
| PEMS07 | `(28224, 883, 1)` | 98.00 | 14.00 | 58.80 / 8.40 / 5636 | 19.60 / 2.80 / 1873 | 19.60 / 2.80 / 1873 |
| PEMS08 | `(17856, 170, 3)` | 62.00 | 8.86 | 37.20 / 5.31 / 3563 | 12.40 / 1.77 / 1182 | 12.40 / 1.77 / 1182 |
| METR-LA | `(34272, 207, 2)` | 119.00 | 17.00 | 71.40 / 10.20 / 6846 | 23.80 / 3.40 / 2276 | 23.80 / 3.40 / 2277 |
| PEMS-BAY | `(52105, 325, 2)` | 180.92 | 25.85 | 108.55 / 15.51 / 10413 | 36.18 / 5.17 / 3465 | 36.18 / 5.17 / 3465 |

## Model Branches and Ablations

Each unrolled block follows the same high-level order:

| Stage | Main tensors / parameters | Notes |
|---|---|---|
| Feature extraction | `FeatureExtractor` | Builds features with shape `(B, T, N, H, F)` from the current sequence and optional ST embeddings. |
| Graph learning | `GraphLearningModule`, `multiM`, `multiQ` | Returns spatial weights `u_ew: (B, T, N, K, H)` and temporal weights `d_ew: (B, T - 1, interval, N, H)`. |
| Theta estimation | `node_covariance`, `glasso_estimation`, `normalize_theta` | Estimates dense node matrix `Theta` from the current output, then normalizes it as `Theta_ij / sqrt(Theta_ii * Theta_jj)`. Skipped entirely for `ablation="Theta"`. |
| ADMM update | `ADMMBlock` | Updates the signal using the branches listed below. |
| Skip connection | `skip_connection_weights[i]` | Blends each block output with the previous block output. |
| Deflation | `DeflationCGSolver` | Runs only when `model.use_deflation=True` and `ablation!="Theta"`. The first deflated mode is the normal ADMM output. |

Current valid ablation names are `None`, `DGLR`, `DGTV`, `UT`, `simple`, and `Theta`.

| Ablation | What changes in the ADMM step | Branches skipped or modified |
|---|---|---|
| `None` | Full split ADMM. | Uses spatial `z_u`, temporal `z_d`, directed temporal L1 `phi/gamma`, directed temporal L2 `cLdr`, dense `Theta`, and optional deflation. |
| `DGLR` | Removes the temporal L2 / graph Laplacian regularization branch. | Skips `z_d` solve and `gamma_d` update. No `mu_d2`, `rho_d`, or `zd_solver`. |
| `DGTV` | Removes the directed temporal L1 / graph TV branch. | Skips `phi/gamma`, `phi_direct`, `gamma` update, and the `Ldr_T(gamma + rho * phi)` term. No `mu_d1` or `rho`. |
| `UT` | Uses undirected temporal graph normalization and an undirected temporal operator. | Graph learning sets `directed_time=False`; `z_d` uses `apply_op_Ln` instead of `apply_op_cLdr`. It also skips `phi/gamma` and the directed temporal L1 update. |
| `simple` | Uses a single-variable ADMM update instead of split auxiliary updates. | Skips `z_u`, `z_d`, `gamma_u`, and `gamma_d` updates. No `rho_u`, `rho_d`, `zu_solver`, or `zd_solver`. |
| `Theta` | Removes the dense Theta regularizer. | Skips `cov_matrix` and `glasso_estimation`; sets `admm_block.Theta=None`; no `lambda_theta`; `LHS_zu` does not apply `Theta`; deflation is skipped even when requested. Directed temporal `phi/gamma` is still active. |

Parameter and solver existence by ablation:

| Ablation | `mu_u` | `mu_d1` | `mu_d2` | `lambda_theta` | `rho` | `rho_u` | `rho_d` | `x_solver` | `zu_solver` | `zd_solver` | Deflation can run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `None` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| `DGLR` | Yes | Yes | No | Yes | Yes | Yes | No | Yes | Yes | No | Yes |
| `DGTV` | Yes | No | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes |
| `UT` | Yes | Yes* | Yes | Yes | Yes* | Yes | Yes | Yes | Yes | Yes | Yes |
| `simple` | Yes | Yes | Yes | Yes | Yes | No | No | Yes | No | No | Yes |
| `Theta` | Yes | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | No |

For `Theta`, `lambda_theta` existence is currently equivalent to `ablation!="Theta"`, but the code uses the explicit ablation state rather than `hasattr(lambda_theta)` for branch decisions. For `UT`, `mu_d1` and `rho` are still created, but the forward path does not use the directed temporal L1 update. `PGD_iters` is currently loaded from config but is not used by the cleaned ADMM forward path.

During training, every 10 batches prints the current-batch per-block Theta sparsity as `nnz / numel` through `tqdm.write`, so the progress bar is preserved. Because `Theta` is re-estimated for each batch rather than stored as a persistent model parameter, this value describes the most recent forward pass.
Before `Theta` enters ADMM, negative diagonal entries are clamped to zero and entries with zero diagonal normalization denominator are set to zero to avoid NaN/Inf values.
`model.glasso_method` and `model.glasso_rho` are configured in `train/config.yaml`; `glasso_rho` is passed to the `glasso_pytorch` ADMM solver and is ignored by `quic` and `sklearn`.

## Training and Testing

The default settings are in `train/config.yaml`. We provide multiple parsers to change the configurations. Run the commands from the repository root.

**Example 1**: running main experiment on PEMS03 dataset:
```
python -m train.train_traffic --dataset PEMS03 --cuda 0 --batchsize 12 --le-emb --neighbors 4
```

**Example 2**: running 'w/o DGLR' experiment on METR-LA dataset:
```
python -m train.train_traffic --dataset METR-LA --cuda 1 --ablation DGLR --batchsize 16 --le-emb
```

**Example 3**: running 'w/o undirected temporal graph' experiment on PEMS-BAY:
```
python -m train.train_traffic --dataset PEMS-BAY --cuda 0 --ablation UT --batchsize 64 --le-emb
```
