# ManiSparse
# Lightweight Transformer via Unrolling of Mixed Graph Algorithms for Traffic Forecast

Chinese version: [README_CN.md](README_CN.md)

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

### UnrollingModel overall flow

`clean_lib.unrolling_model.UnrollingModel` stacks `num_blocks` learned graph-ADMM blocks. The input batch is an observed prefix `y` with shape `(B, t_in, N, C_signal)` and a time-index tensor `t_list` with shape `(B, T)`, where `T = t_in + t_out`. The model first builds a full-horizon initial guess `output: (B, T, N, C_signal)` using `GNNExtrapolation` when `model.use_extrapolation=True`, otherwise `LR_guess`.

If spatial-temporal embedding is enabled, `SpatialTemporalEmbedding(t_list)` is computed once before the block loop and reused in every block because it does not depend on the current signal estimate. Each block then refines the current full-horizon signal:

| Loop step | Data flow | Main shapes / notes |
|---|---|---|
| 1. Build graph-learning input | `output_emb = concat(output, shared_output_emb)` when ST embedding is enabled; otherwise `output_emb = output`. | `output_emb` keeps batch/time/node axes `(B, T, N, *)`. |
| 2. Extract features | `features = FeatureExtractor(output_emb)`. | `features: (B, T, N, H, F)`, where `H=num_heads` and `F=feature_channels`. |
| 3. Learn graph weights | `u_ew, d_ew = GraphLearningModule(features)`. | `u_ew: (B, T, N, K, H)` for spatial neighbors; `d_ew: (B, T - 1, interval, N, H)` for temporal directed edges. These are assigned to the current `ADMMBlock`. |
| 4. Estimate Theta | For block `0` or `ablation="Theta"`, set `Theta=None`. Otherwise compute centered node covariance from `output[..., 0:1]`, or from the previous deflated `multi_x[..., 0:1]` when available, then call `glasso_estimation` and `normalize_theta`. | Plain covariance gives `Theta: (N, N)`; deflated covariance gives batched `Theta: (B, N, N)`. |
| 5. ADMM update | Run `ADMMBlock` on signal channels only. With `use_one_channel=True`, the ADMM input is `output[..., 0:1]`; otherwise it is `output`. | Returns `output_new: (B, T, N, C_admm)`. If `predict_only=True`, the observed prefix is copied back before ADMM. |
| 6. Optional deflation | Deflation runs only when `model.use_deflation=True`, this is not the final block, and `ablation!="Theta"`. | ADMM returns `(output_new, multi_x)`. `multi_x` is stored in `last_deflation_multi_x` and can feed the next block's Theta estimation. The final block skips deflation and returns only `output_new`. |
| 7. Skip blend | `output = p_i * output_new + (1 - p_i) * output_old`, where `p_i = skip_connection_weights[i]`. | `output_old` is the previous block input; in the first one-channel block it uses `output[..., 0:1]` so channel shapes match. |

When `output_graph=True`, the model also returns the stacked learned graph weights:
`undirected_graphs: (B, num_blocks, T, N, K, H)` and
`directed_graphs: (B, num_blocks, T - 1, interval, N, H)`.

`model.use_stable_graph_learning` controls the graph-learning numerical path and is part of the model design for training, validation, and testing. The current default in `train/config.yaml` is `True`, so the model uses the numerical-stability path by default. Setting it to `False` keeps the original formulas: invalid directed temporal intervals are masked after graph weights are computed, `exp` is unclamped, and graph normalization uses the original `torch.where(value > 0, 1 / value, 0)` form. Setting it to `True` enables the numerical-stability path: invalid directed temporal intervals are masked before the `multiQ` projection, Gaussian-kernel exponentials are clamped to avoid extreme underflow/overflow, and graph normalization avoids constructing `1 / 0` inside autograd.

### Numerical Stability Changes

The numerical-stability changes were introduced after training logs showed finite inputs, finite recovered outputs, and a finite loss, but non-finite gradients first appeared in `model_blocks.0.graph_learning_module.multiQ`. This pattern means the forward pass can look valid while the backward graph still creates NaN/Inf values. The affected path is the directed graph-learning chain:

```text
multiQ -> Q_df / Q_i -> exp(...) graph weights -> d_ew -> ADMMBlock -> output -> loss
```

The original graph-learning code is still available through `model.use_stable_graph_learning=False`. The current default `True` applies the following stability changes consistently during training, validation, and testing:

| Location | Original behavior | Stable behavior | Reason |
|---|---|---|---|
| Graph normalization | `torch.where(value > 0, 1 / value, 0)` | `value.clamp_min(eps).reciprocal()` followed by masking invalid entries to zero. | `torch.where` can still build a `1 / 0` branch in autograd even when the forward output is masked, which can produce NaN gradients. |
| Directed temporal mask | Invalid temporal intervals were masked after graph weights were computed. | Invalid temporal intervals are masked before the `multiQ` projection. | Prevents invalid negative-index temporal positions from entering `einsum`, squared distances, and `exp` before the final mask is applied. |
| Gaussian-kernel `exp` | `torch.exp(exp_arg)` without bounds. | Clamp displacement-kernel lower exponent to about `-80` and inner-product upper exponent to about `80`. | Avoids extreme underflow/overflow and backward expressions such as `0 * inf`. |
| Forward checks | The original path checked only NaN in `Q_df` / `Q_i`. | Stable path checks full finite status for `Q_df` / `Q_i`. | Inf values are also unsafe for the later graph normalization and ADMM update. |
| Failure diagnosis | Bad-gradient checks only reported the first bad parameter gradient. | On bad backward, the failing batch is rerun with anomaly detection and tensor/gradient hooks in `UnrollingModel` and `GraphLearningModule`. | Helps locate whether the first non-finite value appears in features, graph weights, ADMM inputs/outputs, or the final parameter gradient. |

This gate is not a testing-only switch. It is part of the model definition: the same value is used for training, validation, testing, and saved model behavior. CLI overrides are available for temporary experiments:

```bash
--stable-graph-learning
--no-stable-graph-learning
```

Each unrolled block follows the same high-level order:

| Stage | Main tensors / parameters | Notes |
|---|---|---|
| Feature extraction | `FeatureExtractor` | Builds features with shape `(B, T, N, H, F)` from the current sequence and optional ST embeddings. |
| Graph learning | `GraphLearningModule`, `multiM`, `multiQ` | Returns spatial weights `u_ew: (B, T, N, K, H)` and temporal weights `d_ew: (B, T - 1, interval, N, H)`. |
| Theta estimation | `node_covariance`, `glasso_estimation`, `normalize_theta` | Estimates dense node matrix `Theta` from the current output, then normalizes it as `Theta_ij / sqrt(Theta_ii * Theta_jj)`. Skipped for the first block and skipped entirely for `ablation="Theta"`. |
| ADMM update | `ADMMBlock` | Updates the signal using the branches listed below. |
| Skip connection | `skip_connection_weights[i]` | Blends each block output with the previous block output. |
| Deflation | `DeflationCGSolver` | Runs only on intermediate blocks when `model.use_deflation=True` and `ablation!="Theta"`. The first deflated mode is the normal ADMM output. The final block skips deflation and returns the normal ADMM output. |

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

During training, every 20 batches prints the current-batch per-block Theta sparsity as `nnz / numel` through `tqdm.write`, so the progress bar is preserved. The same line also reports each block's `lambda_theta`; scalar values are printed directly, while vector/list values are summarized as min / median / max. Floating-point values use scientific notation with 3 significant digits. Because `Theta` is re-estimated for each batch rather than stored as a persistent model parameter, the sparsity value describes the most recent forward pass.
Before `Theta` enters ADMM, negative diagonal entries are clamped to zero and entries with zero diagonal normalization denominator are set to zero to avoid NaN/Inf values.
`model.glasso_method`, `model.glasso_alpha`, `model.glasso_rho`, `model.glasso_eps`, `model.glasso_eigh_shift`, `model.glasso_eigh_shift_retries`, and `model.glasso_fallback` are configured in `train/config.yaml` and can be temporarily overridden from the command line. `glasso_alpha` is the Graphical Lasso regularization strength for all solver backends; `glasso_rho` is passed to the `glasso_pytorch` ADMM solver and is ignored by `quic` and `sklearn`. `glasso_eps` is optional covariance diagonal jitter and defaults to `0.0` because it changes the GLASSO problem. The default numerical protection is instead eigensolver-only shift recovery: when `torch.linalg.eigh(A)` fails in the ADMM Theta update, the solver tries `eigh(A + delta I)` and subtracts `delta` from the returned eigenvalues. This keeps the original ADMM update unchanged in exact arithmetic. `glasso_fallback=True` falls back to sklearn/quic on the original covariance matrix only if the ADMM backend still fails after these spectrum-preserving retries; the ridge/pinv fallback is a final finite-output guard after all GLASSO solvers fail.

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
