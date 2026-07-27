import random
import numpy as np
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import argparse

from dataset.dataset_utils import (
    Normalization,
    create_dataloader,
    create_directed_dataloader,
    create_weather_dataloader,
)


def seed_everything(seed=11):
    """固定 Python、NumPy、PyTorch 的随机种子，减少实验复现误差。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name, logfile, level=logging.INFO, to_console=False):
    """创建一个写入文件的 logger，可选同时输出到控制台。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


class WeightedMSELoss(nn.Module):
    """加权 MSE：前 `t` 步重建误差乘权重，后 `T - t` 步预测误差保持原权重。"""

    def __init__(self, t, T, weights=None) -> None:
        super().__init__()
        self.t = t
        self.T = T
        self.weights = t / (T - t) if weights is None else weights

    def forward(self, inputs, target):
        rec_loss = nn.MSELoss()(inputs[:, :self.t], target[:, :self.t])
        pred_loss = nn.MSELoss()(inputs[:, self.t:], target[:, self.t:])
        return rec_loss * self.weights + pred_loss


def plot_loss_curve(train_loss, val_loss, save_path, val_freq=5, use_log=False):
    """保存训练/验证 loss 曲线，验证 loss 的横轴按 `val_freq` 对齐。"""
    train_len, val_len = len(train_loss), len(val_loss)
    if train_len > 1:
        plt.figure()
        plt.plot(list(range(1, train_len + 1)), train_loss, label='train')
    if val_len != 0:
        plt.plot(list(range(val_freq, val_len * val_freq + 1, val_freq)), val_loss, label='val')
    if use_log:
        plt.yscale('log')
    if train_len > 1:
        plt.legend()
        plt.savefig(save_path)
        plt.close()


def _should_log_gradient(iteration_count, interval):
    """判断当前 iteration 是否到达指定日志间隔。"""
    return (iteration_count + 1) % interval == 0


def _is_logged_parameter(model, name):
    """筛选需要记录梯度的参数，目前只看图聚合权重和旧 extrapolation 线性层。"""
    if 'agg' in name and 'weight' in name:
        return True
    uses_old_extrapolation = getattr(model, 'use_extrapolation', False) and getattr(model, 'use_old_extrapolation', False)
    return uses_old_extrapolation and 'linear_extrapolation' in name


def _gradient_summary(name, param):
    """格式化单个参数的取值范围和梯度 L2 范数。"""
    grad_norm = param.grad.data.norm(2).item() if param.grad is not None else float('nan')
    return f'{name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {grad_norm:.4f}'


def log_gradients(epoch, num_epochs, iteration_count, train_loader, model, grad_logger, args):
    """按 `args.loggrad` 周期把关键参数的数值范围和梯度写入日志。"""
    should_log_file = _should_log_gradient(iteration_count, args.loggrad)
    should_print_debug = args.debug and _should_log_gradient(iteration_count, 3 * args.loggrad)

    if should_log_file:
        grad_logger.info(f'[Epoch {epoch}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}]')
    if should_print_debug:
        print(f'[Epoch {epoch}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}]')

    for name, param in model.named_parameters():
        if not _is_logged_parameter(model, name):
            continue
        summary = _gradient_summary(name, param)
        if should_log_file:
            grad_logger.info(summary)
        if should_print_debug:
            print(summary)


def print_gradients(model):
    """直接打印关键参数的数值范围和梯度，用于临时 debug。"""
    for name, param in model.named_parameters():
        if _is_logged_parameter(model, name):
            print(_gradient_summary(name, param))


def change_model_location(model, model_path, device):
    """加载模型参数，并同步带有 `device` 属性的子模块位置标记。"""
    model_params = torch.load(model_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(model_params, strict=False)
    print('missing keys:', missing_keys)
    print('unexpected keys:', unexpected_keys)
    for name, module in model.named_children():
        if hasattr(module, 'device'):
            print(f'Loaded module: {name}')
            module.device = device
        if name == 'model_blocks':
            for block in module:
                block['feature_extractor'].device = device
                block['ADMM_block'].device = device
                block['graph_learning_module'].device = device
    return model


def _loader_iter(loader, use_tqdm):
    """根据配置返回普通 dataloader 迭代器或 tqdm 包装后的迭代器。"""
    return tqdm(loader) if use_tqdm else loader


def _model_uses_one_channel(config, fallback):
    """从 config 读取是否只使用单通道；缺省时使用调用方传入的 fallback。"""
    return config.get('model', {}).get('use_one_channel', fallback)


def _loss_on_requested_span(loss_fn, output, target, masked_flag, t_in):
    """计算训练/验证 loss；masked 模式只在预测区间 `t_in:` 上计算。"""
    if loss_fn is None:
        return None
    if masked_flag:
        return loss_fn(output[:, t_in:], target[:, t_in:])
    return loss_fn(output, target)


def _forward_eval_batch(model, y, x, t_list, data_normalization, masked_flag, config, loss_fn, output_graph=False, use_one_channel=False):
    """完成一个 batch 的评估前向过程。

    输入:
        y: 观测序列前缀，shape 通常为 (B, t_in, N, C)
        x: 完整目标序列，shape 通常为 (B, T, N, C)
        t_list: 时间索引，shape 通常为 (B, T)

    返回:
        output: 恢复到原始数据尺度后的模型输出
        loss: 按配置计算的 loss；如果 `loss_fn is None` 则为 None
        normed_output/normed_x: 归一化尺度上的输出和目标，供 test_series 保存
        undirected_graphs/directed_graphs: `output_graph=True` 时模型额外返回的图序列
    """
    t_in = config['model']['t_in']
    normalize_one_channel = _model_uses_one_channel(config, use_one_channel)

    if data_normalization is None:
        model_result = model(y, t_list, output_graph=True) if output_graph else model(y, t_list)
        if output_graph:
            output, undirected_graphs, directed_graphs = model_result
        else:
            output, undirected_graphs, directed_graphs = model_result, None, None
        loss = _loss_on_requested_span(loss_fn, output, x, masked_flag, t_in)
        return output, loss, output, x, undirected_graphs, directed_graphs

    normed_y = data_normalization.normalize_data(y)
    normed_x = data_normalization.normalize_data(x, normalize_one_channel)
    model_result = model(normed_y, t_list, output_graph=True) if output_graph else model(normed_y, t_list)
    if output_graph:
        normed_output, undirected_graphs, directed_graphs = model_result
    else:
        normed_output, undirected_graphs, directed_graphs = model_result, None, None

    output = data_normalization.recover_data(normed_output, normalize_one_channel)
    if config.get('normed_loss', False):
        loss = _loss_on_requested_span(loss_fn, normed_output, normed_x, masked_flag, t_in)
    else:
        loss = _loss_on_requested_span(loss_fn, output, x, masked_flag, t_in)
    return output, loss, normed_output, normed_x, undirected_graphs, directed_graphs


def _empty_channel_metrics(signal_channels):
    """初始化逐通道 metric 的累加器。"""
    return {
        'rec_mse': np.zeros((signal_channels,)),
        'pred_mse': np.zeros((signal_channels,)),
        'pred_mae': np.zeros((signal_channels,)),
        'pred_mape': np.zeros((signal_channels,)),
        'pred_mape_count': np.zeros((signal_channels,)),
    }


def _update_channel_metrics(sums, output, x, t_in):
    """累加逐通道指标。

    逐通道指标只在 `test` 中使用，最终会变成 `metrics_d`:
        rec_RMSE: 前 `t_in` 步重建区间的逐通道 RMSE
        pred_RMSE: `t_in:` 预测区间的逐通道 RMSE
        pred_MAE: `t_in:` 预测区间的逐通道 MAE
        pred_MAPE: `t_in:` 预测区间的逐通道 MAPE，忽略接近 0 的真实值
    """
    x_pred = x[:, t_in:]
    output_pred = output[:, t_in:]
    sums['rec_mse'] += ((x[:, :t_in] - output[:, :t_in]) ** 2).detach().mean((0, 1, 2)).cpu().numpy()
    sums['pred_mse'] += ((x_pred - output_pred) ** 2).detach().mean((0, 1, 2)).cpu().numpy()
    sums['pred_mae'] += torch.abs(output_pred - x_pred).detach().mean((0, 1, 2)).cpu().numpy()

    for channel_id in range(x_pred.size(-1)):
        mask = torch.abs(x_pred[..., channel_id]) > 1e-8
        if mask.any():
            percentage_error = torch.abs(output_pred[..., channel_id][mask] - x_pred[..., channel_id][mask])
            percentage_error = percentage_error / torch.abs(x_pred[..., channel_id][mask])
            sums['pred_mape'][channel_id] += percentage_error.detach().mean().cpu().item() * 100
            sums['pred_mape_count'][channel_id] += 1


def _finalize_channel_metrics(sums, num_batches):
    """把逐 batch 累加的逐通道误差转换成最终 `metrics_d` 字典。"""
    pred_mape = np.divide(
        sums['pred_mape'],
        sums['pred_mape_count'],
        out=np.full_like(sums['pred_mape'], np.nan),
        where=sums['pred_mape_count'] > 0,
    )
    return {
        'rec_RMSE': np.sqrt(sums['rec_mse'] / num_batches),
        'pred_RMSE': np.sqrt(sums['pred_mse'] / num_batches),
        'pred_MAE': sums['pred_mae'] / num_batches,
        'pred_MAPE': pred_mape,
    }


def _add_rmse_metrics(metrics):
    """基于 `compute_metrics` 的基础 MSE 结果补充 RMSE 和 rNMSE。"""
    metrics['rNMSE_stepwise'] = torch.sqrt(metrics['pred_MSE_stepwise'] / metrics['truth_sq_stepwise'])
    metrics['rNMSE'] = math.sqrt(metrics['pred_MSE'] / metrics['truth_sq'])
    metrics['rec_RMSE'] = math.sqrt(metrics['rec_MSE'])
    metrics['pred_RMSE'] = math.sqrt(metrics['pred_MSE'])
    return metrics


def test(model, val_loader, data_normalization, masked_flag, config, device, signal_channels, mode='test', loss_fn=None, use_one_channel=False, use_tqdm=True):
    """评估模型并返回汇总 metric。

    `test` 做的是“评价指标汇总”，不是保存每个样本的完整过程:
    1. 把每个 batch 的 `(y, x, t_list)` 放到 `device`。
       `y` 是观测前缀，通常为 (B, t_in, N, C)；
       `x` 是完整目标序列，通常为 (B, T, N, C)。
    2. 如果传入 `data_normalization`，先在归一化尺度上跑模型，再把输出恢复
       到原始数据尺度用于 metric。
    3. 如果 `mode == 'val'` 且传入 `loss_fn`，额外返回平均 validation loss。
       loss 的计算尺度由 `config['normed_loss']` 决定；`masked_flag=True` 时
       loss 只看预测区间 `t_in:`。
    4. 拼接所有 batch 后调用 `compute_metrics` 计算全局 metric:
       `rec_MSE/rec_RMSE` 衡量前 `t_in` 步重建；
       `pred_MSE/pred_RMSE/pred_MAE/pred_MAPE/rNMSE` 衡量 `t_in:` 后的预测。
    5. 多通道输入时额外返回 `metrics_d`，即每个 channel 各自的
       `rec_RMSE/pred_RMSE/pred_MAE/pred_MAPE`。
    """
    assert mode in ['val', 'test']
    model.eval()
    t_in = config['model']['t_in']
    output_list = []
    x_list = []
    running_loss = 0.0
    channel_sums = None if use_one_channel else _empty_channel_metrics(signal_channels)

    with torch.no_grad():
        for y, x, t_list in _loader_iter(val_loader, use_tqdm):
            y, x, t_list = y.to(device), x.to(device), t_list.to(device)
            output, loss, _, _, _, _ = _forward_eval_batch(
                model,
                y,
                x,
                t_list,
                data_normalization,
                masked_flag,
                config,
                loss_fn,
                use_one_channel=use_one_channel,
            )
            if loss is not None:
                running_loss += loss.item()

            output_list.append(output.detach().cpu())
            x_list.append(x.detach().cpu())
            if channel_sums is not None:
                _update_channel_metrics(channel_sums, output, x, t_in)

    full_output = torch.cat(output_list, 0)
    full_x = torch.cat(x_list, 0)
    metrics = _add_rmse_metrics(compute_metrics(full_output, full_x, masked_flag, t_in))
    metrics_d = None if use_one_channel else _finalize_channel_metrics(channel_sums, len(val_loader))

    if mode == 'val':
        running_loss = running_loss / len(val_loader) if loss_fn is not None else 0.0
        return (running_loss, metrics) if use_one_channel else (running_loss, metrics, metrics_d)
    return metrics if use_one_channel else (metrics, metrics_d)


def test_series(model, val_loader, data_normalization, masked_flag, config, device, signal_channels, mode='test', loss_fn=None, use_one_channel=False, use_tqdm=True):
    """导出完整输出序列和图序列，用于可视化/诊断，不计算汇总 metric。

    `test_series` 和 `test` 的区别:
        `test` 返回最终评价指标，例如 RMSE、MAE、MAPE、rNMSE。
        `test_series` 返回逐样本的原始结果，包括:
            output/x: 原始数据尺度上的预测序列和真实序列
            normed_output/normed_x: 归一化尺度上的预测序列和真实序列
            undirected_graphs/directed_graphs: 模型每个 batch 输出的图
            connect_list/nearest_nodes/nearest_dists: 模型内部保存的图辅助信息

    所以这里的 `loss_fn`、`masked_flag`、`mode` 主要是为了复用前向 helper 和
    保持旧接口兼容；返回值本身不包含 metric。
    """
    assert mode in ['val', 'test']
    model.eval()
    output_list = []
    x_list = []
    normed_x_list = []
    normed_output_list = []
    undirected_graph_list = []
    directed_graph_list = []

    with torch.no_grad():
        for y, x, t_list in _loader_iter(val_loader, use_tqdm):
            y, x, t_list = y.to(device), x.to(device), t_list.to(device)
            output, _, normed_output, normed_x, undirected_graphs, directed_graphs = _forward_eval_batch(
                model,
                y,
                x,
                t_list,
                data_normalization,
                masked_flag,
                config,
                loss_fn,
                output_graph=True,
                use_one_channel=use_one_channel,
            )

            output_list.append(output.detach().cpu())
            x_list.append(x.detach().cpu())
            normed_output_list.append(normed_output.detach().cpu())
            normed_x_list.append(normed_x.detach().cpu())
            undirected_graph_list.append(undirected_graphs.detach().cpu())
            directed_graph_list.append(directed_graphs.detach().cpu())

    return {
        'output': torch.cat(output_list, 0),
        'x': torch.cat(x_list, 0),
        'normed_output': torch.cat(normed_output_list, 0),
        'normed_x': torch.cat(normed_x_list, 0),
        'connect_list': getattr(model, 'connect_list', None),
        'nearest_nodes': getattr(model, 'nearest_nodes', None),
        'nearest_dists': getattr(model, 'nearest_dists', None),
        'undirected_graphs': torch.cat(undirected_graph_list, 0),
        'directed_graphs': torch.cat(directed_graph_list, 0),
    }


def compute_metrics(output, x, masked_flag, t_in):
    """
    计算全局重建和预测指标。

    output: 模型输出，shape (B, T, N, C)
    x: 真实目标，shape (B, T, N, C)
    masked_flag: 保留旧接口；当前 metric 总是在 `t_in:` 预测区间上计算预测项。

    返回:
        rec_MSE: `:t_in` 重建区间的 MSE
        pred_MSE/pred_MAE/pred_MAPE: `t_in:` 预测区间的误差
        pred_MSE_stepwise: 每个预测步的 MSE，shape (T - t_in,)
        truth_sq_stepwise/truth_sq: 真实值平方均值，用于计算 rNMSE
        nearest_loss: 第一个预测步 `t_in` 的 MSE
    """
    rec_mse = ((x[:, :t_in] - output[:, :t_in]) ** 2).mean().item()
    x_pred = x[:, t_in:]
    output_pred = output[:, t_in:]
    pred_mse = ((x_pred - output_pred) ** 2).mean().item()
    pred_mae = torch.abs(output_pred - x_pred).mean().item()

    mask = torch.abs(x_pred) > 1e-8
    if mask.any():
        pred_mape = (torch.abs(output_pred[mask] - x_pred[mask]) / torch.abs(x_pred[mask])).mean().item() * 100
    else:
        pred_mape = None

    pred_mse_stepwise = ((x_pred - output_pred) ** 2).mean((0, 2, 3))
    truth_sq_stepwise = (x_pred ** 2).mean((0, 2, 3))
    truth_sq = (x_pred ** 2).mean()
    nearest_loss = ((x[:, t_in] - output[:, t_in]) ** 2).mean().item()

    return {
        'rec_MSE': rec_mse,
        'pred_MSE': pred_mse,
        'pred_MAE': pred_mae,
        'pred_MAPE': pred_mape,
        'pred_MSE_stepwise': pred_mse_stepwise,
        'truth_sq_stepwise': truth_sq_stepwise,
        'truth_sq': truth_sq,
        'nearest_loss': nearest_loss
    }


def check_nan_gradients(model: nn.Module):
    """检查模型参数或梯度中是否存在 NaN/Inf，返回第一个异常位置。"""
    for name, param in reversed(list(model.named_parameters())):
        param_detached = param.detach().cpu()
        if torch.isnan(param_detached).any() or torch.isinf(param_detached).any():
            return {"name": name, "kind": "parameter"}

        if param.grad is None:
            continue
        grad_detached = param.grad.detach().cpu()
        if torch.isnan(grad_detached).any() or torch.isinf(grad_detached).any():
            return {"name": name, "kind": "gradient"}
    return None


def log_parameters_scalars(model: nn.Module, name_list: list):
    """收集指定参数名片段对应参数的 min/max 和梯度范数，供日志或 TensorBoard 使用。"""
    param_dicts = {}
    grad_dict = {}
    for check_name in name_list:
        param_dicts[check_name] = {}
        for name, param in model.named_parameters():
            if check_name in name:
                name_split = name.split('.')
                block_id, param_name = name_split[1], name_split[-1]
                param_dicts[check_name][f'{param_name}_{block_id}_min'] = param.detach().cpu().min().item()
                param_dicts[check_name][f'{param_name}_{block_id}_max'] = param.detach().cpu().max().item()
                if param.grad is not None:
                    grad_dict[f'{param_name}_{block_id}'] = param.grad.data.detach().cpu().norm(2).item()
    return param_dicts, grad_dict

# TensorBoard utilities (legacy; not required for core training).
def dataframe_from_tensorboard(log_dir, selected_tag):
    """
    从 TensorBoard event 文件中读取指定 scalar tag，返回包含 step/value 的 DataFrame。
    """
    import pandas as pd
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # 获取所有的 tags
    tags = ea.Tags()
    print("Available tags:", tags)

    # 提取某个 tag 的数据（例如 'train/loss'）
    scalar_data = ea.Scalars(selected_tag)

    # 将数据转换为 DataFrame

    df = pd.DataFrame({
        "step": [e.step for e in scalar_data],
        "value": [e.value for e in scalar_data]
    })

    return df

def plot_dataframe(df, title="Training Loss Curve"):
    """
    绘制由 `dataframe_from_tensorboard` 得到的 step/value 曲线。
    """

    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["value"])
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.show()


def log_tensorboard():
    """TensorBoard 写日志的预留入口，目前没有实际实现。"""
    pass


def generate_experiment_name(args: argparse.Namespace, config:dict):
    """实验命名的预留入口，目前没有实际实现。"""
    pass
