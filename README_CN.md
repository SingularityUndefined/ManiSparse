# ManiSparse
# 通过混合图算法展开实现交通预测的轻量级 Transformer

English version: [README.md](README.md)

## 依赖

运行训练前需要在 Python 环境中安装：

```text
torch>=2.4.1
tqdm
numpy
matplotlib
networkx>=2.5
pandas
pyyaml
tensorboard
```

## 数据集

**PEMS0X 数据集**来自 [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data)。训练代码默认在 `../TS_datasets/PEMS0X_data` 下读取：

```text
../TS_datasets/
├── PEMS0X_data/
│   ├── PEMS03/
│   ├── PEMS04/
│   ├── PEMS07/
│   └── PEMS08/
├── PEMS-BAY/
└── METR-LA/
```

**PEMS-BAY** 和 **METR-LA** 由 [DCRNN](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph) 数据预处理得到。每个文件夹包含邻接矩阵和时序数据两个 `.npy` 文件。

也可以从 [Google Drive](https://drive.google.com/drive/folders/1tMgyxzQ_dio73rapQ-LYSOOIXwXFFUJw?usp=share_link) 下载数据，并放在仓库外的 `../TS_datasets`。

### 数据集时长和划分大小

下面的交通数据集都使用 5 分钟采样，所以一天有 288 个时间步。当前训练配置使用 `t_in=12`、`t_out=12`、`T=24`、`data_stride=3`。代码按照 train/val/test = 60%/20%/20% 划分，然后用 `(split_steps - T) // data_stride` 计算样本数。

| Dataset | Data shape `(steps, nodes, channels)` | Total days | Total weeks | Train days / weeks / samples | Val days / weeks / samples | Test days / weeks / samples |
|---|---:|---:|---:|---:|---:|---:|
| PEMS03 | `(26208, 358, 1)` | 91.00 | 13.00 | 54.60 / 7.80 / 5233 | 18.20 / 2.60 / 1739 | 18.20 / 2.60 / 1739 |
| PEMS04 | `(16992, 307, 3)` | 59.00 | 8.43 | 35.40 / 5.06 / 3390 | 11.80 / 1.69 / 1124 | 11.80 / 1.69 / 1125 |
| PEMS07 | `(28224, 883, 1)` | 98.00 | 14.00 | 58.80 / 8.40 / 5636 | 19.60 / 2.80 / 1873 | 19.60 / 2.80 / 1873 |
| PEMS08 | `(17856, 170, 3)` | 62.00 | 8.86 | 37.20 / 5.31 / 3563 | 12.40 / 1.77 / 1182 | 12.40 / 1.77 / 1182 |
| METR-LA | `(34272, 207, 2)` | 119.00 | 17.00 | 71.40 / 10.20 / 6846 | 23.80 / 3.40 / 2276 | 23.80 / 3.40 / 2277 |
| PEMS-BAY | `(52105, 325, 2)` | 180.92 | 25.85 | 108.55 / 15.51 / 10413 | 36.18 / 5.17 / 3465 | 36.18 / 5.17 / 3465 |

## 模型分支和消融

### UnrollingModel 总体流程

`clean_lib.unrolling_model.UnrollingModel` 堆叠了 `num_blocks` 个学习图结构的 graph-ADMM block。输入 batch 包括观测前缀 `y: (B, t_in, N, C_signal)` 和时间索引 `t_list: (B, T)`，其中 `T = t_in + t_out`。模型首先生成完整时间窗口的初始估计 `output: (B, T, N, C_signal)`：当 `model.use_extrapolation=True` 时使用 `GNNExtrapolation`，否则使用 `LR_guess`。

如果启用空间-时间嵌入，`SpatialTemporalEmbedding(t_list)` 会在 block 循环前计算一次，并在所有 block 中复用，因为它不依赖当前信号估计。每个 block 按下面的数据流继续细化完整时间窗口信号：

| 循环步骤 | 数据流 | 主要维度 / 说明 |
|---|---|---|
| 1. 构造图学习输入 | 启用 ST embedding 时 `output_emb = concat(output, shared_output_emb)`，否则 `output_emb = output`。 | `output_emb` 保留 `(B, T, N, *)` 轴。 |
| 2. 提取特征 | `features = FeatureExtractor(output_emb)`。 | `features: (B, T, N, H, F)`，其中 `H=num_heads`，`F=feature_channels`。 |
| 3. 学习图权重 | `u_ew, d_ew = GraphLearningModule(features)`。 | `u_ew: (B, T, N, K, H)` 表示空间邻居权重；`d_ew: (B, T - 1, interval, N, H)` 表示时间有向边权重。它们会被赋给当前 `ADMMBlock`。 |
| 4. 估计 Theta | 第 0 个 block 或 `ablation="Theta"` 时设 `Theta=None`。否则从 `output[..., 0:1]` 计算中心化节点协方差；如果已有上一个 block 的 deflated `multi_x`，则从 `multi_x[..., 0:1]` 计算协方差，然后调用 `glasso_estimation` 和 `normalize_theta`。 | 普通协方差得到 `Theta: (N, N)`；deflated 协方差得到 batched `Theta: (B, N, N)`。 |
| 5. ADMM 更新 | `ADMMBlock` 只接收信号通道。`use_one_channel=True` 时输入为 `output[..., 0:1]`，否则输入为 `output`。 | 返回 `output_new: (B, T, N, C_admm)`。如果 `predict_only=True`，ADMM 前会把观测前缀拷贝回去。 |
| 6. 可选 deflation | 只有当 `model.use_deflation=True`、当前不是最后一个 block、且 `ablation!="Theta"` 时执行。 | ADMM 返回 `(output_new, multi_x)`。`multi_x` 存入 `last_deflation_multi_x`，可用于下一个 block 的 Theta 估计。最后一个 block 跳过 deflation，只返回 `output_new`。 |
| 7. skip 融合 | `output = p_i * output_new + (1 - p_i) * output_old`，其中 `p_i = skip_connection_weights[i]`。 | `output_old` 是上一个 block 的输入；第一个 one-channel block 中使用 `output[..., 0:1]` 以匹配通道数。 |

当 `output_graph=True` 时，模型还会返回堆叠后的图权重：
`undirected_graphs: (B, num_blocks, T, N, K, H)` 和
`directed_graphs: (B, num_blocks, T - 1, interval, N, H)`。

`model.use_stable_graph_learning` 控制图学习的数值路径，是模型设计的一部分，训练、验证和测试都会使用同一个值。当前 `train/config.yaml` 默认值为 `True`，因此默认使用数值稳定版。设为 `False` 时保留原始公式：无效时间 interval 在图权重计算后才 mask，`exp` 不做 clamp，图归一化使用原始 `torch.where(value > 0, 1 / value, 0)`。设为 `True` 时启用数值稳定路径：无效时间 interval 会在 `multiQ` 投影前 mask，高斯核指数会 clamp 以避免极端 underflow/overflow，图归一化会避免在 autograd 中构造 `1 / 0`。

### 数值稳定性修改过程和原因

这些数值稳定性修改来自一次训练报错：输入、恢复后的输出和 loss 都是 finite，但 `loss.backward()` 后第一个坏掉的梯度出现在 `model_blocks.0.graph_learning_module.multiQ`。这说明 forward 表面上可以正常，但 backward 图中仍然可能产生 NaN/Inf。相关路径是有向图学习链路：

```text
multiQ -> Q_df / Q_i -> exp(...) graph weights -> d_ew -> ADMMBlock -> output -> loss
```

原始图学习代码仍可通过 `model.use_stable_graph_learning=False` 使用。当前默认 `True` 会在训练、验证和测试中一致应用下面的稳定化修改：

| 位置 | 原始行为 | 稳定版行为 | 原因 |
|---|---|---|---|
| 图归一化 | `torch.where(value > 0, 1 / value, 0)` | 先用 `value.clamp_min(eps).reciprocal()`，再把无效位置 mask 为 0。 | `torch.where` 的 forward 结果可能被 mask 成有限值，但 autograd 仍可能看到 `1 / 0` 分支，产生 NaN 梯度。 |
| 有向时间 mask | 无效时间 interval 在图权重计算后才 mask。 | 无效时间 interval 在 `multiQ` 投影前就 mask。 | 避免负索引得到的无效时间位置进入 `einsum`、平方距离和 `exp`。 |
| 高斯核 `exp` | 直接 `torch.exp(exp_arg)`。 | displacement kernel 的指数下界约为 `-80`，inner-product kernel 的指数上界约为 `80`。 | 避免极端 underflow/overflow 以及 backward 中类似 `0 * inf` 的表达式。 |
| forward 检查 | 原始路径只检查 `Q_df` / `Q_i` 是否有 NaN。 | 稳定路径检查 `Q_df` / `Q_i` 是否全部 finite。 | Inf 对后续图归一化和 ADMM 更新也不安全。 |
| 错误定位 | 原先 bad-gradient 检查只能报第一个坏掉的参数梯度。 | bad backward 时会用同一个 batch 开启 anomaly detection，并在 `UnrollingModel` 和 `GraphLearningModule` 中记录 forward 张量和 backward 梯度。 | 可以定位第一处 non-finite 值出现在 features、graph weights、ADMM 输入/输出还是最终参数梯度。 |

这个门控不是只用于测试的开关，而是模型定义的一部分。临时实验可以用命令行覆盖：

```bash
--stable-graph-learning
--no-stable-graph-learning
```

### 每个 unrolled block 的高层结构

| Stage | Main tensors / parameters | Notes |
|---|---|---|
| Feature extraction | `FeatureExtractor` | 从当前序列和可选 ST embedding 构造 `(B, T, N, H, F)` 特征。 |
| Graph learning | `GraphLearningModule`, `multiM`, `multiQ` | 返回空间权重 `u_ew: (B, T, N, K, H)` 和时间权重 `d_ew: (B, T - 1, interval, N, H)`。 |
| Theta estimation | `node_covariance`, `glasso_estimation`, `normalize_theta` | 从当前输出估计 dense node matrix `Theta`，并做 `Theta_ij / sqrt(Theta_ii * Theta_jj)` 归一化。第一个 block 和 `ablation="Theta"` 时跳过。 |
| ADMM update | `ADMMBlock` | 用下面的分支更新信号。 |
| Skip connection | `skip_connection_weights[i]` | 将当前 ADMM 输出和前一个 block 输出融合。 |
| Deflation | `DeflationCGSolver` | 只在中间 block、`model.use_deflation=True` 且 `ablation!="Theta"` 时运行。最后一个 block 跳过 deflation。 |

当前有效消融名称为 `None`、`DGLR`、`DGTV`、`UT`、`simple` 和 `Theta`。

| Ablation | ADMM step 中的变化 | 跳过或修改的分支 |
|---|---|---|
| `None` | 完整 split ADMM。 | 使用空间 `z_u`、时间 `z_d`、有向时间 L1 `phi/gamma`、有向时间 L2 `cLdr`、dense `Theta` 和可选 deflation。 |
| `DGLR` | 去掉时间 L2 / graph Laplacian regularization 分支。 | 跳过 `z_d` solve 和 `gamma_d` update。没有 `mu_d2`、`rho_d`、`zd_solver`。 |
| `DGTV` | 去掉有向时间 L1 / graph TV 分支。 | 跳过 `phi/gamma`、`phi_direct`、`gamma` update 和 `Ldr_T(gamma + rho * phi)` 项。没有 `mu_d1` 或 `rho`。 |
| `UT` | 使用无向时间图归一化和无向时间算子。 | Graph learning 设置 `directed_time=False`；`z_d` 使用 `apply_op_Ln` 而不是 `apply_op_cLdr`。同时跳过 `phi/gamma` 和有向时间 L1 更新。 |
| `simple` | 使用单变量 ADMM 更新，不使用 split auxiliary updates。 | 跳过 `z_u`、`z_d`、`gamma_u` 和 `gamma_d` 更新。没有 `rho_u`、`rho_d`、`zu_solver`、`zd_solver`。 |
| `Theta` | 去掉 dense Theta regularizer。 | 跳过 `cov_matrix` 和 `glasso_estimation`；设置 `admm_block.Theta=None`；没有 `lambda_theta`；`LHS_zu` 不使用 `Theta`；即使请求 deflation 也会跳过。 |

参数和 solver 是否存在：

| Ablation | `mu_u` | `mu_d1` | `mu_d2` | `lambda_theta` | `rho` | `rho_u` | `rho_d` | `x_solver` | `zu_solver` | `zd_solver` | Deflation can run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `None` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| `DGLR` | Yes | Yes | No | Yes | Yes | Yes | No | Yes | Yes | No | Yes |
| `DGTV` | Yes | No | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes |
| `UT` | Yes | Yes* | Yes | Yes | Yes* | Yes | Yes | Yes | Yes | Yes | Yes |
| `simple` | Yes | Yes | Yes | Yes | Yes | No | No | Yes | No | No | Yes |
| `Theta` | Yes | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | No |

`Theta` 消融时，`lambda_theta` 是否存在目前等价于 `ablation!="Theta"`，但代码分支使用显式 ablation 状态，而不是 `hasattr(lambda_theta)`。`UT` 中 `mu_d1` 和 `rho` 仍然会创建，但 forward 路径不会使用有向时间 L1 更新。`PGD_iters` 会从配置读取，但当前 cleaned ADMM forward 路径没有使用。

训练时每 20 个 batch 会通过 `tqdm.write` 输出一次当前 batch 的每个 block 的 Theta 稀疏度 `nnz / numel`，不会破坏进度条。同一行还会输出每个 block 的 `lambda_theta`；标量直接打印，向量/list 打印 min / median / max。浮点数使用 3 位有效数字的科学计数法。因为 `Theta` 是每个 batch 重新估计的，不是持久模型参数，所以这个稀疏度描述的是最近一次 forward。

`Theta` 进入 ADMM 前会先把负的对角线元素 clamp 到 0，并把归一化分母为 0 的位置设为 0，以避免 NaN/Inf。

`model.glasso_method`、`model.glasso_alpha`、`model.glasso_rho`、`model.glasso_eps`、`model.glasso_eigh_shift`、`model.glasso_eigh_shift_retries` 和 `model.glasso_fallback` 在 `train/config.yaml` 中配置，也可以用命令行临时覆盖。`glasso_alpha` 是所有 Graphical Lasso 后端的正则强度；`glasso_rho` 只传给 `glasso_pytorch` ADMM solver，`quic` 和 `sklearn` 会忽略它。`glasso_eps` 是可选的协方差对角 jitter，默认是 `0.0`，因为它会改变原始 GLASSO 问题。现在默认使用的是只作用在特征值分解求解器上的 shift recovery：如果 ADMM Theta 更新中的 `torch.linalg.eigh(A)` 失败，solver 会尝试 `eigh(A + delta I)`，然后把返回的特征值减回 `delta`。在精确算术下，这不会改变原本的 ADMM 更新。只有这些不改变谱问题的重试仍失败时，`glasso_fallback=True` 才会在原始协方差矩阵上 fallback 到 sklearn/quic；ridge/pinv 只是所有 GLASSO solver 都失败后的最终有限输出保护。

## 训练和测试

默认设置在 `train/config.yaml` 中。请从仓库根目录运行命令。

**示例 1**：在 PEMS03 上运行主实验：

```bash
python -m train.train_traffic --dataset PEMS03 --cuda 0 --batchsize 12 --le-emb --neighbors 4
```

**示例 2**：在 METR-LA 上运行 `w/o DGLR` 实验：

```bash
python -m train.train_traffic --dataset METR-LA --cuda 1 --ablation DGLR --batchsize 16 --le-emb
```

**示例 3**：在 PEMS-BAY 上运行 `w/o undirected temporal graph` 实验：

```bash
python -m train.train_traffic --dataset PEMS-BAY --cuda 0 --ablation UT --batchsize 64 --le-emb
```
