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
| 4. 估计 Theta | 第 0 个 block 或 `ablation="Theta"` 时设 `Theta=None`。否则 GLASSO 从中心化协方差估计精度矩阵；Kalofolias 直接从当前 `output` 或上一个 block 的 multi-mode `multi_x` 学习平滑信号图。 | GLASSO 和 dense Kalofolias 给出 `(N, N)` 或 `(B, N, N)`；local Kalofolias 返回局部候选边权 `(N, K_theta)` 或 `(B, N, K_theta)`。 |
| 5. ADMM 更新 | `ADMMBlock` 只接收信号通道。`use_one_channel=True` 时输入为 `output[..., 0:1]`，否则输入为 `output`。 | 返回 `output_new: (B, T, N, C_admm)`。如果 `predict_only=True`，ADMM 前会把观测前缀拷贝回去。 |
| 6. 可选 deflation | 只有当 `model.deflation.enabled=True`、当前不是最后一个 block、且 `ablation!="Theta"` 时执行。 | ADMM 返回 `(output_new, multi_x)`。`multi_x` 存入 `last_deflation_multi_x`，可用于下一个 block 的 Theta 估计。默认 detach；可通过可微分支保留完整计算图。最后一个 block 跳过 deflation。 |
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
| Theta estimation | `node_covariance`, `glasso_estimation`, `KalofoliasGraphLearningModule`, `LocalKalofoliasGraphLearning` | GLASSO 从协方差估计 dense Theta；Kalofolias 从平滑信号学习图，`graph=local` 时返回局部候选边权。第一个 block 和 `ablation="Theta"` 时跳过。 |
| ADMM update | `ADMMBlock` | 用下面的分支更新信号。 |
| Skip connection | `skip_connection_weights[i]` | 将当前 ADMM 输出和前一个 block 输出融合。 |
| Deflation | `DeflationCGSolver` | 只在中间 block、`model.deflation.enabled=True` 且 `ablation!="Theta"` 时运行。默认 no-grad；实验性可微模式会保留 fixed-CG、正交化和 multi-mode 的计算图。 |

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

`Theta` 消融时，`lambda_theta` 是否存在目前等价于 `ablation!="Theta"`，但代码分支使用显式 ablation 状态，而不是 `hasattr(lambda_theta)`。`UT` 中 `mu_d1` 和 `rho` 仍然会创建，但 forward 路径不会使用有向时间 L1 更新。cleaned ADMM 路径只暴露实际使用的迭代配置：外层 `num_layers`、内层 `CG_iters` 和可选 deflation CG 设置。

### ADMM 参数初始化、约束与数值投影

空间正则项以 `mu_u L^u + lambda_theta Theta` 的形式进入 ADMM。为了避免两个独立自由参数通过负值相互抵消，`mu_u` 与 `lambda_theta` 不再分别直接学习，而是在每个 block、每个 ADMM iteration 使用共享总强度和 sigmoid 分配：

```text
total = softplus(raw_total)
gate  = sigmoid(raw_gate)
mu_u          = total * gate
lambda_theta  = total * (1 - gate)
```

因此恒有 `mu_u > 0`、`lambda_theta >= 0`，且 `mu_u + lambda_theta = total`。`ADMM_params.mu_u` 与 `ADMM_params.lambda_theta` 仍是初始化值：例如默认 `2` 与 `1` 分别初始化为 `total=3`、`mu_share=2/3`，之后 `raw_total` 与 `raw_gate` 会正常学习。这个初始化远离 sigmoid 的 0/1 饱和区。

其余有明确 ADMM 含义的参数在每次 optimizer step 后投影到合法域：

| 参数 | 约束 | 原因 |
|---|---|---|
| `mu_d1`, `mu_d2` | `> eps` | 时间正则强度必须为正。 |
| `rho`, `rho_u`, `rho_d` | `> eps` | ADMM penalty 必须为正，避免线性系统退化。 |
| CG `alpha` | `[0, clamp]` | 避免负步长和过大的学习步长。 |
| CG `beta` | `[0, +inf)` | 避免负搜索方向系数。 |

普通神经网络权重、`multiM/multiQ`、skip 权重不做这种正性约束。该参数化改变了 state dict 的参数名，因此旧的仅模型权重 `val_<epoch>.pth` 不能直接加载到新结构；应使用新结构重新训练。

### 可微 multi-mode deflation 与 local Kalofolias 分支

默认配置下，deflation 是辅助图估计步骤：`multi_x`、deflation CG 和 local Kalofolias 均在 no-grad/detach 路径中运行。这样最省显存、数值最稳，但 loss 不会经由 `Theta -> multi_x` 回传到前一个 block。

若需要端到端经过 multi-mode Theta 估计反传，必须同时开启下面两个开关：

```yaml
model:
  deflation:
    allow_backward: True
  theta:
    kalofolias:
      allow_backward: True
```

等价 CLI：

```bash
--kalofolias-allow-backward --deflation-allow-backward
```

开启 `kalofolias.allow_backward=True` 而未显式设置 deflation 开关时，代码会自动把 `deflation.allow_backward` 设为 `True`。若显式传 `--no-deflation-allow-backward`，会保留该消融设置并给出 warning，因为此时 `multi_x` 仍 detach，Kalofolias 不能通过 deflation 回传梯度。

可微分支使用固定次数 CG，移除输入 detach/no-grad，并以 `torch.stack` 组合 modes，从而保留：

```text
loss -> 后续 ADMM -> Theta -> local Kalofolias -> multi_x
     -> multi-mode deflation CG / 正交化 -> 前一 ADMM block
```

PEMS03、batch=1 的一次前向加反向测试中，默认 detached 路径峰值 allocated 约 1917 MiB；同时开启两条可微分支约 2431 MiB。该值会随 batch size、mode 数、CG iteration 和 block 数增长。当前没有 gradient checkpoint；可微 deflation 以额外显存换取真实梯度。

训练时每 20 个 batch 会通过 `tqdm.write` 输出一次当前 batch 的每个 block 的 Theta 图指标，不会破坏进度条。除稀疏度、kNN recall/precision 与度统计外，同一行还会输出 `mu_u`、`lambda_theta`、`total=mu_u+lambda_theta` 和 `mu_share=mu_u/total` 的范围与中位数。因为 `Theta` 是每个 batch 重新估计的，不是持久模型参数，所以这些图指标描述的是最近一次 forward。

每个 training epoch 结束时，`Theta epoch summary` 还会对每 20 batch 的采样求平均：除图指标外，汇总 `mu_u`、`lambda_theta`、`total` 与 `mu_share` 的采样中位数均值。每次 validation 后参数不会更新，因此日志会直接打印各 block 当前的完整范围；这与对 validation batch 求均值等价。

`Theta` 进入 ADMM 前会先把负的对角线元素 clamp 到 0，并把归一化分母为 0 的位置设为 0，以避免 NaN/Inf。

Kalofolias 配置位于 `model.theta.kalofolias`，其中 `graph: dense|local` 选择图表示，`local_kNN` 指定 local 候选边宽度。local 模式在 ADMM 中应用归一化局部 Laplacian，而不物化 dense 矩阵。若物理传感器图的某一连通分量无法提供请求数量的邻居，代码保留请求宽度，将缺失 slot mask 为零；它们不会进入求解器、ADMM 算子或图指标。日志会报告每个节点可用邻居数的最小/最大值。

`model.glasso_method`、`model.glasso_alpha`、`model.glasso_rho`、`model.glasso_eps`、`model.glasso_eigh_shift`、`model.glasso_eigh_shift_retries` 和 `model.glasso_fallback` 在 `train/config.yaml` 中配置，也可以用命令行临时覆盖。`glasso_alpha` 是所有 Graphical Lasso 后端的正则强度；`glasso_rho` 只传给 `glasso_pytorch` ADMM solver，`quic` 和 `sklearn` 会忽略它。`glasso_eps` 是可选的协方差对角 jitter，默认是 `0.0`，因为它会改变原始 GLASSO 问题。现在默认使用的是只作用在特征值分解求解器上的 shift recovery：如果 ADMM Theta 更新中的 `torch.linalg.eigh(A)` 失败，solver 会尝试 `eigh(A + delta I)`，然后把返回的特征值减回 `delta`。在精确算术下，这不会改变原本的 ADMM 更新。只有这些不改变谱问题的重试仍失败时，`glasso_fallback=True` 才会在原始协方差矩阵上 fallback 到 sklearn/quic；ridge/pinv 只是所有 GLASSO solver 都失败后的最终有限输出保护。
如果训练中 GLASSO 触发 eigensolver fallback，训练日志会写出 `GLASSO fallback observed`，包括 epoch/batch、block index、Theta 来源、协方差维度、ADMM iteration、backend、shift 数值、retry index，以及 solver 从 ADMM 切到 sklearn/ridge 的路径。

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

**示例 4**：在 PEMS03 上运行 local Kalofolias，并允许完整 multi-mode deflation 与 Theta 反传：

```bash
python -m train.train_traffic \
  --dataset PEMS03 \
  --cuda 1 \
  --batchsize 1 \
  --neighbors 4 \
  --theta-method kalofolias \
  --kalofolias-graph local \
  --theta-neighbors 10 \
  --kalofolias-allow-backward
```

最后一个开关会自动开启 `deflation.allow_backward`。若希望命令完全显式，也可额外传 `--deflation-allow-backward`。batch size 应根据显存调整；上例的 batch=1 是可微分支的保守起点。

### 实验目录、日志与自动断点续训

训练产物按 Theta 方法、数据集、学习率/随机种子和其余超参数组织：

```text
logs_learnable_emb/
└── kalofolias_local/
    └── PEMS03/
        └── lr_5e-04_seed_3407/
            └── diffV_shareQ_kaloBW1_deflateBW1_deflate5_.../
```

目录名中的 `kaloBW0/1` 表示 `kalofolias.allow_backward` 是否关闭/开启，`deflateBW0/1` 对应 `deflation.allow_backward`。这样不会把可微分支与默认 detached 分支的 checkpoint、日志混在同一个目录。

模型目录中有两类 checkpoint：

```text
models/<experiment_name>/Huber/
├── val_<epoch>.pth          # validation 最优时保存；只含 model.state_dict()
└── last_train_state.pth     # 每个完整 epoch 覆盖保存；用于断点续训
```

`last_train_state.pth` 包含模型、optimizer、scheduler、下一 epoch、最佳 validation 状态、loss/Theta 历史以及 Python/NumPy/PyTorch CPU/CUDA 的 RNG 状态。以相同实验目录再次启动时会自动恢复它；可用以下选项控制：

```bash
# 指定一个完整训练状态文件
--resume /path/to/last_train_state.pth

# 忽略自动发现的 last_train_state.pth，从头训练
--no-auto-resume
```

恢复粒度是 epoch 边界：如果进程在一个 epoch 中途退出，会从该 epoch 开始重跑；当前没有 iteration 级 checkpoint 或 gradient checkpoint。
