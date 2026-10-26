# 在正交子空间求解完整时空目标的 deflation

本文推导一种**拟议算法**：第一个模态求解原时空重建问题；此后每个模态在与已有输出模态正交的子空间中，重新求解**同一个包含时间有向 L2 与 L1 项的问题**。这与当前 [`DeflationCGSolver`](../clean_lib/admm_block.py) 的计算不同：当前实现把普通 ADMM 输出作为第一个模态，其余模态只对 `I + mu_u L_u + lambda_theta Theta` 做投影 CG，没有把时间有向算子放进 deflation 求解。

## 1. 固定一个 block 后的问题

考虑一个 batch 样本。记 `T` 为完整时间长度，`N` 为节点数，`C` 为信号通道数，`J` 为图 head 数。把所有 head 的信号堆叠为

$$
X=(X_1,\ldots,X_J),\qquad X_h\in\mathbb R^{T\times N\times C}.
$$

令 `s` 为**传入当前 ADMM block 的完整序列**。`R` 只提取前 `t_in` 个时刻，`Y` 是 `Rs` 在每个 head 上的复制。除 `predict_only=True` 会在调用 ADMM 前把这个前缀重设为真实观测外，后续 block 的 `s` 前缀通常已经是上一 block 的预测结果。因此，这里的数据项严格对应当前 block 的代码，不能无条件把 `Y` 叫作最初观测。`L_u` 是按时间和 head 分块的空间图算子，`D=L_d` 是按 head 分块的**时间有向差分算子**，`Theta` 是空间图正则算子。按当前 ADMM 的系数及二次项因子约定，形式上的原问题为

$$
\boxed{
\begin{aligned}
\min_X F(X)={}&\|RX-Y\|_F^2
 +\mu_u\langle X,L_uX\rangle
 +\lambda_\theta\langle X,\Theta X\rangle\\
 &+\mu_{d2}\|DX\|_F^2
 +\mu_{d1}\|DX\|_1.
\end{aligned}}
\tag{1}
$$

这里 `D` 由已学习的时间有向边权 `d_ew` 构造。在代码中，目标时刻 `t` 的差分是其当前信号减去若干历史时刻信号的有向加权和；首个时刻的差分设为零。`D^T D` 出现在时间 L2 子问题中，`DX` 出现在 L1 软阈值子问题中。参见 [`apply_op_Ldr`](../clean_lib/admm_block.py) 和 [`apply_op_cLdr`](../clean_lib/admm_block.py)。

更具体地，对时间 `t>0`、节点 `i`、head `h`、通道 `c`，代码所用算子可写为

$$
\begin{aligned}
(L_{u,h}X_h)_{t,i,c}
  &=X_{h,t,i,c}-\sum_{j\in\mathcal N_i}u_{t,i,j,h}X_{h,t,j,c},\\
(DX)_{h,t,i,c}
  &=X_{h,t,i,c}
    -\sum_{\ell=1}^{\min(\mathrm{interval},t)}
       d_{t-1,\ell-1,i,h}X_{h,t-\ell,i,c},\\
(DX)_{h,0,i,c}&=0.
\end{aligned}
$$

`D` 因此是沿时间的有向线性算子；`D^T` 是它在欧氏内积下的伴随算子。`L_u` 不混合不同时间，`Theta` 在每个时间点作用于节点维度。

**“求解同一个目标”需要先固定系数与图。** 对一个 block 的全部模态，固定 `u_ew`、`d_ew`、`Theta` 及 `mu_u`、`lambda_theta`、`mu_d1`、`mu_d2`；这些量可以沿用该 block 已估计的图和选定的参数切片。当前展开网络在不同 ADMM 迭代使用不同的可学习参数，其有限步映射本身并不严格等于求解一个固定的式 (1)。`s` 的未来部分在当前 ADMM 中只作初值；数据项 `RX-Y` 只覆盖它的前缀。

## 2. 输出模态的正交约束

当前模型在 ADMM 后以可学习权重 `c_h` 合并 head：

$$
\bar X_t=\mathcal C X_t=\sum_{h=1}^{J}c_h X_{h,t}.
\tag{2}
$$

为与现有 deflation 的输出空间正交定义一致，约束应作用于 **head 合并后的** `N*C` 维信号。先取得第一个输出模态 `m_1=\mathcal C X_1`。第 `k` 个模态开始前，在每个时刻 `t` 把已得的 `m_{1,t},...,m_{k-1,t}` 正交化，得到列正交矩阵 `Q_{k-1,t}`。则第 `k` 个问题是

$$
\boxed{
X_k\in\arg\min_X F(X)
\quad\text{s.t.}\quad
Q_{k-1,t}^{\top}(\mathcal C X)_t=0
\quad(t=0,\ldots,T-1),
\qquad m_k=\mathcal C X_k.}
\tag{3}
$$

多通道时把节点和通道展平后做内积。若希望按整段序列正交，只需把所有 `t` 的信号一起展平并使用一个全局 `Q`；本文以下采用当前默认的**逐时间点正交**。虽然约束可逐时间点投影，`D` 仍耦合不同时间，所以式 (3) 必须联合求解完整时间序列。

第一个模态可以直接使用现有 ADMM 输出，作为式 (3) 的起点。若要求所有模态都是真正的优化解，第一个模态也应以固定目标 (1) 运行到收敛。对后续模态，数据项始终使用同一个 block 前缀 `Y`；完整序列初值可沿用上一模态或原 block 的输入 `s`，它不替代 `Y`。

## 3. ADMM 拆分

定义可行子空间

$$
\mathcal S_{k-1}=\{V:\ Q_{k-1,t}^{\top}(\mathcal C V)_t=0\ \text{对所有 }t\},
$$

并引入 `Z_u`、`Z_d`、`Phi`、`V`：

$$
\begin{aligned}
\min_{X,Z_u,Z_d,\Phi,V}\quad
&\|RX-Y\|_F^2
 +\mu_u\langle Z_u,L_u Z_u\rangle
 +\lambda_\theta\langle Z_u,\Theta Z_u\rangle\\
&+\mu_{d2}\|D Z_d\|_F^2
 +\mu_{d1}\|\Phi\|_1
 +\iota_{\mathcal S_{k-1}}(V),\\
\text{s.t.}\quad
&X=Z_u,\quad X=Z_d,\quad \Phi=DX,\quad X=V.
\end{aligned}
\tag{4}
$$

其中 `iota_S(V)` 在 `V` 属于 `S` 时为零，否则为正无穷。前三个约束是原 ADMM 的空间、时间 L2、时间 L1 拆分；`X=V` 是新增的正交约束拆分。`Z_u` 同时承载 `L_u` 和 `Theta`，对应当前默认 `ablation="None"` 的结构。若使用 `Full` 分支，可把 `Theta` 再拆为独立辅助变量，推导方式相同。实现时可把 `(Z_u,Z_d,Phi,V)` 视为一个联合辅助变量块；在固定 `X` 时，这四个更新彼此可分离。

令 `gamma_u`、`gamma_d`、`gamma`、`eta` 分别对应上式四个约束，惩罚系数为 `rho_u`、`rho_d`、`rho`、`rho_perp`。为与现有 `phi/gamma` 更新的符号一致，第三个约束写成 `Phi-DX=0`。增广拉格朗日函数为

$$
\begin{aligned}
\mathcal L={}&\|RX-Y\|_F^2
 +\mu_u\langle Z_u,L_uZ_u\rangle
 +\lambda_\theta\langle Z_u,\Theta Z_u\rangle
 +\mu_{d2}\|DZ_d\|_F^2
 +\mu_{d1}\|\Phi\|_1
 +\iota_{\mathcal S_{k-1}}(V)\\
&+\langle\gamma_u,X-Z_u\rangle
 +\frac{\rho_u}{2}\|X-Z_u\|_F^2
 +\langle\gamma_d,X-Z_d\rangle
 +\frac{\rho_d}{2}\|X-Z_d\|_F^2\\
&+\langle\gamma,\Phi-DX\rangle
 +\frac{\rho}{2}\|\Phi-DX\|_F^2
 +\langle\eta,X-V\rangle
 +\frac{\rho_\perp}{2}\|X-V\|_F^2.
\end{aligned}
\tag{5}
$$

## 4. 各步的具体推导

以下假设二次图算子对称；不对称情形见第 6 节。上标 `r` 表示模态 `k` 内的 ADMM 迭代。

### 4.1 更新 `X`

在式 (5) 中只保留含 `X` 的项并求导，得到

$$
\begin{aligned}
0={}&2R^\top(RX-Y)
 +\gamma_u^r+\rho_u(X-Z_u^r)
 +\gamma_d^r+\rho_d(X-Z_d^r)\\
&-D^\top\gamma^r+\rho D^\top(DX-\Phi^r)
 +\eta^r+\rho_\perp(X-V^r).
\end{aligned}
$$

整理并除以 `2`：

$$
\boxed{
\begin{aligned}
\Big[R^\top R
 +\tfrac{\rho_u+\rho_d+\rho_\perp}{2}I
 +\tfrac{\rho}{2}D^\top D\Big]X^{r+1}
={}&R^\top Y
 +\tfrac{\rho_uZ_u^r+\rho_dZ_d^r+\rho_\perp V^r}{2}\\
&-\tfrac{\gamma_u^r+\gamma_d^r+\eta^r}{2}
 +\tfrac12D^\top(\gamma^r+\rho\Phi^r).
\end{aligned}}
\tag{6}
$$

这是跨时间的线性系统。`D^T D` 使时间有向图明确进入每个额外模态。若 `rho_perp>0` 且其余项半正定，左端是正定的，可用标准 CG 求到规定残差；当前网络中的可学习步长 `UnrolledCGSolver` 不等同于标准 CG。

### 4.2 更新空间辅助变量 `Z_u`

求导并整理得到

$$
\boxed{
\Big[\mu_u L_u+\lambda_\theta\Theta+\tfrac{\rho_u}{2}I\Big]Z_u^{r+1}
=\tfrac12\big(\gamma_u^r+\rho_u X^{r+1}\big).}
\tag{7}
$$

这个方程与默认 ADMM 的空间辅助变量方程同型；新正交约束不直接改变它的左端。

### 4.3 更新时间 L2 辅助变量 `Z_d`

同样地，

$$
\boxed{
\Big[\mu_{d2}D^\top D+\tfrac{\rho_d}{2}I\Big]Z_d^{r+1}
=\tfrac12\big(\gamma_d^r+\rho_d X^{r+1}\big).}
\tag{8}
$$

这一步再次使用时间有向图，并且仍是跨时间求解。

### 4.4 更新时间 L1 辅助变量 `Phi`

与 `Phi` 有关的项是

$$
\mu_{d1}\|\Phi\|_1
 +\langle\gamma^r,\Phi-DX^{r+1}\rangle
 +\frac{\rho}{2}\|\Phi-DX^{r+1}\|_F^2.
$$

配方后得到逐元素软阈值：

$$
\boxed{
\Phi^{r+1}
=\operatorname{soft}_{\mu_{d1}/\rho}
\left(DX^{r+1}-\frac{\gamma^r}{\rho}\right),\quad
\operatorname{soft}_\tau(a)=\operatorname{sign}(a)\max(|a|-\tau,0).}
\tag{9}
$$

因此完整时空 deflation 确实是一个 **L2–L1 混合问题**；单独投影 CG 无法完成这一步。

### 4.5 更新正交辅助变量 `V`

配方后，`V` 的更新就是对可行子空间的欧氏投影：

$$
\boxed{V^{r+1}=\Pi_{\mathcal S_{k-1}}
\left(X^{r+1}+\frac{\eta^r}{\rho_\perp}\right).}
\tag{10}
$$

令投影输入为 `A=(A_1,...,A_J)`，`\bar A_t=\sum_h c_h A_{h,t}`，并假设 `Q_{k-1,t}` 列正交。由于约束只作用于 head 合并后的输出，逐时刻投影子问题为

$$
\min_{V_{1,t},\ldots,V_{J,t}}
\frac12\sum_h\|V_{h,t}-A_{h,t}\|_F^2
\quad\text{s.t.}\quad
Q_{k-1,t}^\top\sum_h c_hV_{h,t}=0.
$$

令约束乘子为 `xi_t`。对每个 `V_{h,t}` 求导可得 `V_{h,t}=A_{h,t}-c_h Q_{k-1,t}xi_t`；代回约束并利用 `Q_{k-1,t}^T Q_{k-1,t}=I`，得到 `xi_t=Q_{k-1,t}^T\bar A_t/(\sum_j c_j^2)`。于是**保持当前输出正交定义的 head 级投影**为

$$
\boxed{
V_{h,t}=A_{h,t}
-\frac{c_h}{\sum_j c_j^2}
Q_{k-1,t}Q_{k-1,t}^\top\bar A_t,
\qquad \sum_j c_j^2>0.}
\tag{11}
$$

确实有 `\sum_h c_h V_{h,t}=(I-Q_{k-1,t}Q_{k-1,t}^\top)\bar A_t`。若只处理单个 head 且 `c_1=1`，式 (11) 退化为熟悉的 `V_t=(I-Q_tQ_t^T)A_t`。若模态几乎线性相关，先对 `Q` 做稳定的 QR 正交化；若没有足够独立方向，就应停止生成新模态。

### 4.6 更新乘子

$$
\boxed{
\begin{aligned}
\gamma_u^{r+1}&=\gamma_u^r+\rho_u(X^{r+1}-Z_u^{r+1}),\\
\gamma_d^{r+1}&=\gamma_d^r+\rho_d(X^{r+1}-Z_d^{r+1}),\\
\gamma^{r+1}&=\gamma^r+\rho(\Phi^{r+1}-DX^{r+1}),\\
\eta^{r+1}&=\eta^r+\rho_\perp(X^{r+1}-V^{r+1}).
\end{aligned}}
\tag{12}
$$

不能把普通 ADMM 的最终输出简单投影一次当作式 (3) 的解：这样可能破坏数据拟合及时间正则的最优性条件。式 (10) 的投影必须与其他变量和乘子更新反复交替。

## 5. 多模态算法

1. 在当前 block 固定空间图、时间有向图和 `Theta`。确定用于本次求解的一组固定正则系数。
2. 求解无正交约束的式 (1)，得到 `X_1` 与输出模态 `m_1=\mathcal C X_1`；也可以沿用现有 ADMM 输出作为近似第一模态。
3. 对 `k=2,...,K`：逐时间点正交化已有输出模态，得到 `Q_{k-1,t}`；初始化 `X,Z_u,Z_d,Phi,V` 及乘子；重复式 (6)–(12)，直到原始与对偶残差达到容差；返回 `m_k=\mathcal C X_k`。
4. 把 `m_1,...,m_K` 组成 `multi_x`，交给下一个 block 的 `Theta` 估计。若保留当前模型结构，最后一个 block 仍无需生成额外模态。

这比当前每个额外模态只做少量空间投影 CG 昂贵得多：每个模态都要反复求解跨时间的 `X`、`Z_d` 线性系统，并执行有向 L1 软阈值与正交投影。

具体停止量可取原始残差 `r_pri=(X-Z_u, X-Z_d, Phi-DX, X-V)`。把四个辅助变量合成一个 ADMM 块时，按式 (4) 的约束符号，对偶残差的范数可取

$$
\left\|
\rho_u\Delta Z_u+\rho_d\Delta Z_d
+\rho D^\top\Delta\Phi+\rho_\perp\Delta V
\right\|_F,
$$

其中 `Delta` 表示相邻两轮辅助变量之差。两类残差都应达到指定的绝对与相对容差；内部线性系统也应有单独的求解容差。

## 6. 何时能严格称为“优化问题”

上述二次项求导使用了 `L_u=L_u^T`、`Theta=Theta^T`。若这些算子还半正定、正则系数非负，则式 (1) 是凸问题，式 (6)–(8) 可按正定线性系统求解。时间有向差分 `D` **不要求对称**：`D^T D` 自然半正定，而 `\|DX\|_1` 保留了有向差分的方向信息。

当前代码中的空间近邻是逐行选出的；默认 local Kalofolias 也逐行产生权重，**没有强制 `L_u` 或 `Theta` 对称**。对不对称算子，`\nabla\langle X,L_uX\rangle=(L_u+L_u^T)X`，并非 `2L_uX`。因此，若要严格求解本文写出的标量目标，应先构造对称半正定的空间算子；直接沿用当前不对称算子及式 (7) 只能称为按现有线性更新规则运行的算法，不能保证它在最小化式 (1)。同理，要声称“求解”而非“展开有限步近似”，应固定目标系数并使用有收敛准则的线性求解器与 ADMM。
