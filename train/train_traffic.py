"""Traffic forecasting training entry point.

This module keeps the high-level training flow visible:
    parse config/args -> build data/model/optimizer/loggers -> train/validate/test.
Most setup code is delegated to `train.train_utils`.
"""

from __future__ import annotations

import copy
import gc
import math
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from train.train_utils import (
    apply_args_to_config,
    build_experiment_names,
    build_loss_fn,
    collect_theta_support_deltas,
    create_data,
    create_loggers,
    create_model,
    create_optimizer_and_scheduler,
    create_training_paths,
    create_writer,
    format_theta_support_batch,
    get_model_config,
    log_run_header,
    parse_args,
    prepare_runtime,
)
from utils import (
    change_model_location,
    check_nan_gradients,
    log_gradients,
    log_parameters_scalars,
    plot_loss_curve,
    test,
)


SIGNAL_NAMES = ["flow", "occupancy", "speed"]


def _print_data_summary(train_set, val_set, test_set, data_normalization, mode, signal_channels):
    """Print quick dataset and normalization sanity checks to stdout."""
    print("dataset size", len(train_set), len(val_set), len(test_set))
    print("number of channels:", signal_channels)
    if mode == "standardize":
        print("mean value of data", data_normalization.mean[..., 0].min(), data_normalization.mean[..., 0].max())
        print("std of data", data_normalization.std[..., 0].min(), data_normalization.std[..., 0].max())
    else:
        print("min value of data", data_normalization.min[..., 0].min(), data_normalization.min[..., 0].max())
        print("max value of data", data_normalization.max[..., 0].min(), data_normalization.max[..., 0].max())


def _compute_loss(loss_fn, output, target, masked_flag, t_in):
    """Compute loss on all time steps, or only prediction steps in masked mode."""
    if masked_flag:
        return loss_fn(output[:, t_in:], target[:, t_in:])
    return loss_fn(output, target)


def _tensor_health(name, tensor):
    """Return a compact numeric summary for NaN/Inf diagnostics."""
    detached = tensor.detach()
    if not torch.is_floating_point(detached) and not torch.is_complex(detached):
        total = detached.numel()
        if total == 0:
            return f"{name}: shape={tuple(detached.shape)}, dtype={detached.dtype}, numel=0"
        values = detached.to(torch.float32)
        return (
            f"{name}: shape={tuple(detached.shape)}, dtype={detached.dtype}, finite={total}/{total}, "
            f"min={values.min().item():.6g}, max={values.max().item():.6g}, "
            f"mean={values.mean().item():.6g}"
        )

    finite_mask = torch.isfinite(detached)
    finite_count = finite_mask.sum().item()
    total = detached.numel()
    nan_count = torch.isnan(detached).sum().item()
    inf_count = torch.isinf(detached).sum().item()

    if finite_count == 0:
        return f"{name}: shape={tuple(detached.shape)}, finite=0/{total}, nan={nan_count}, inf={inf_count}"

    finite_values = detached[finite_mask]
    return (
        f"{name}: shape={tuple(detached.shape)}, finite={finite_count}/{total}, "
        f"nan={nan_count}, inf={inf_count}, min={finite_values.min().item():.6g}, "
        f"max={finite_values.max().item():.6g}, mean={finite_values.mean().item():.6g}"
    )


def _training_location(epoch, num_epochs, iteration_count, train_loader):
    """Format the current training location in one consistent way."""
    return f"epoch {epoch + 1}/{num_epochs}, iter {iteration_count}/{len(train_loader)}"


def _short_log_text(value, limit=180):
    """Compact long exception text for one-line training logs."""
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _format_glasso_fallback_event(event):
    """Format one GLASSO fallback/eigensolver event for the training log."""
    parts = [
        f"stage={event.get('stage', '-')}",
        f"block={event.get('block', '-')}",
        f"source={event.get('theta_source', '-')}",
        f"cov_shape={event.get('cov_shape', '-')}",
    ]
    if "cov_batch_index" in event:
        parts.append(f"cov_batch_index={event['cov_batch_index']}")
    if "solver" in event:
        parts.append(f"solver={event['solver']}")
    if "backend" in event:
        parts.append(f"backend={event['backend']}")
    if "admm_iter" in event:
        parts.append(f"admm_iter={event['admm_iter']}")
    if "shift" in event:
        parts.append(f"shift={float(event['shift']):.3e}")
    if "retry" in event:
        parts.append(f"retry={event['retry']}")
    if "from" in event or "to" in event:
        parts.append(f"path={event.get('from', '-')}->{event.get('to', '-')}")
    if "matrix_shape" in event:
        parts.append(f"matrix_shape={event['matrix_shape']}")
    if "dtype" in event:
        parts.append(f"dtype={event['dtype']}")
    if "device" in event:
        parts.append(f"device={event['device']}")
    if "reason" in event:
        parts.append(f"reason={_short_log_text(event['reason'])}")
    return ", ".join(parts)


def _format_bad_tensor_info(info):
    """Format the bad parameter/gradient record returned by check_nan_gradients."""
    if info is None:
        return "none"
    parts = [
        f"{info.get('kind', '-')}: {info.get('name', '-')}",
        f"shape={info.get('shape', '-')}",
        f"finite={info.get('finite', '-')}/{info.get('numel', '-')}",
        f"nan={info.get('nan', '-')}",
        f"inf={info.get('inf', '-')}",
    ]
    if "min" in info:
        parts.extend(
            [
                f"min={info['min']:.6g}",
                f"max={info['max']:.6g}",
                f"mean={info['mean']:.6g}",
            ]
        )
    return ", ".join(parts)


def _format_glasso_fallback_events(model, max_events=40):
    """Return readable lines for GLASSO events from the latest model forward."""
    events = getattr(model, "last_glasso_events", None) or []
    if not events:
        return []
    lines = [f"GLASSO fallback/eigensolver events in latest forward: count={len(events)}"]
    for event in events[:max_events]:
        lines.append(_format_glasso_fallback_event(event))
    if len(events) > max_events:
        lines.append(f"... {len(events) - max_events} more GLASSO fallback events omitted")
    return lines


def _log_glasso_fallback_events(model, logger, epoch, num_epochs, iteration_count, train_loader):
    """Write latest-forward GLASSO fallback events to the training logger."""
    lines = _format_glasso_fallback_events(model)
    if not lines:
        return
    logger.warning(
        "%s\n%s",
        f"GLASSO fallback observed at {_training_location(epoch, num_epochs, iteration_count, train_loader)}",
        "\n".join(lines),
    )


def _raise_training_numerical_error(reason, epoch, num_epochs, iteration_count, train_loader, tensors, logger, extra_lines=None):
    """Log and raise a readable numerical failure message."""
    lines = [
        f"Numerical failure during training: {reason}",
        f"Location: {_training_location(epoch, num_epochs, iteration_count, train_loader)}",
    ]
    lines.extend(_tensor_health(name, tensor) for name, tensor in tensors)
    if extra_lines:
        lines.extend(extra_lines)
    message = "\n".join(lines)
    logger.error(message)
    raise ValueError(message)


def _set_model_numerical_debug(model, enabled, context=""):
    """Enable optional tensor/gradient hooks on modules that support them."""
    for module_name, module in model.named_modules():
        if hasattr(module, "set_debug_numerics"):
            module_context = f"{context}.{module_name}" if context else module_name
            module.set_debug_numerics(enabled, module_context)


def _collect_model_numerical_debug_records(model, max_lines=80):
    """Collect recent numerical debug records from instrumented modules."""
    records = []
    for module_name, module in model.named_modules():
        module_records = getattr(module, "debug_records", None)
        if module_records:
            records.append(f"Numerical debug records from {module_name}:")
            records.extend(module_records[-max_lines:])
    if not records:
        return ["Numerical debug rerun produced no module-level records."]
    return records[-max_lines:]


def _diagnose_bad_backward(model, y, x, t_list, data_normalization, config, loss_fn, masked_flag, args, optimizer, location):
    """Rerun the failing batch with anomaly detection and graph-learning hooks."""
    optimizer.zero_grad(set_to_none=True)
    _set_model_numerical_debug(model, True, f"diagnostic.{location}")
    lines = [f"Diagnostic rerun for {location}:"]
    try:
        with torch.autograd.detect_anomaly(check_nan=True):
            diag_loss, diag_output = _train_forward(model, y, x, t_list, data_normalization, config, loss_fn, masked_flag, args)
            lines.append(_tensor_health("diagnostic_loss", diag_loss))
            lines.append(_tensor_health("diagnostic_output_recovered", diag_output))
            diag_loss.backward()
            diag_bad = check_nan_gradients(model)
            if diag_bad is None:
                lines.append("Diagnostic backward completed without non-finite parameter gradients.")
            else:
                lines.append(f"Diagnostic backward ended with bad tensor: {_format_bad_tensor_info(diag_bad)}")
    except Exception as diagnostic_error:
        lines.append(f"Diagnostic rerun failed at first detected bad operation: {diagnostic_error}")
    finally:
        lines.extend(_collect_model_numerical_debug_records(model))
        _set_model_numerical_debug(model, False)
        optimizer.zero_grad(set_to_none=True)
    return lines


def _optimizer_state_health(optimizer, max_lines=40):
    """Summarize optimizer state when parameters become non-finite after step."""
    lines = []
    for group_idx, group in enumerate(optimizer.param_groups):
        for param_idx, param in enumerate(group["params"]):
            state = optimizer.state.get(param, {})
            for state_name, state_value in state.items():
                if torch.is_tensor(state_value):
                    lines.append(_tensor_health(f"optimizer_state[group={group_idx}, param={param_idx}, {state_name}]", state_value))
            if len(lines) >= max_lines:
                return lines
    return lines or ["Optimizer has no tensor state to inspect."]


def _train_forward(model, y, x, t_list, data_normalization, config, loss_fn, masked_flag, args):
    """Normalize one training batch, run the model, recover output, and compute loss."""
    model_config = get_model_config(config)
    t_in = model_config["t_in"]
    use_one_channel = model_config["use_one_channel"]

    normed_y = data_normalization.normalize_data(y)
    normed_x = data_normalization.normalize_data(x, use_one_channel)
    normed_output = model(normed_y, t_list)
    if args.mode == "normalize":
        normed_output = nn.ReLU()(normed_output)

    if config["normed_loss"]:
        loss = _compute_loss(loss_fn, normed_output, normed_x, masked_flag, t_in)
        output = data_normalization.recover_data(normed_output, use_one_channel)
    else:
        output = data_normalization.recover_data(normed_output, use_one_channel)
        loss = _compute_loss(loss_fn, output, x, masked_flag, t_in)

    return loss, output


def _empty_train_metrics():
    """Initialize per-epoch training metric accumulators."""
    return {
        "running_loss": 0.0,
        "rec_mse": 0.0,
        "pred_mse": 0.0,
        "pred_mae": 0.0,
        "pred_mape": 0.0,
        "pred_mape_count": 0,
        "nearest_loss": 0.0,
    }


def _update_train_metrics(sums, loss, output, x, t_in, masked_flag):
    """Accumulate training metrics for one batch on recovered data scale."""
    sums["running_loss"] += loss.item()
    sums["rec_mse"] += ((x[:, :t_in] - output[:, :t_in]) ** 2).detach().cpu().mean().item()

    if masked_flag:
        x_pred = x[:, t_in:]
        output_pred = output[:, t_in:]
    else:
        x_pred = x[:, t_in:]
        output_pred = output[:, t_in:]

    sums["pred_mse"] += ((x_pred - output_pred) ** 2).detach().cpu().mean().item()
    sums["pred_mae"] += torch.abs(output_pred - x_pred).detach().cpu().mean().item()
    mask = torch.abs(x_pred) > 1e-8
    if mask.any():
        sums["pred_mape"] += (torch.abs(output_pred[mask] - x_pred[mask]) / torch.abs(x_pred[mask])).detach().cpu().mean().item() * 100
        sums["pred_mape_count"] += 1
    sums["nearest_loss"] += ((x[:, t_in] - output[:, t_in]) ** 2).detach().cpu().mean().item()


def _finalize_train_metrics(sums, num_batches):
    """Convert per-batch sums into epoch-level training metrics."""
    pred_mape = sums["pred_mape"] / sums["pred_mape_count"] if sums["pred_mape_count"] > 0 else float("nan")
    return {
        "loss": sums["running_loss"] / num_batches,
        "rec_RMSE": math.sqrt(sums["rec_mse"] / num_batches),
        "nearest_RMSE": math.sqrt(sums["nearest_loss"] / num_batches),
        "pred_RMSE": math.sqrt(sums["pred_mse"] / num_batches),
        "pred_MAE": sums["pred_mae"] / num_batches,
        "pred_MAPE": pred_mape,
    }


def _maybe_write_tensorboard(writer, model, output, x, loss, rmse_per_time, loss_acc, epoch, iteration_count, train_loader, T, t_in):
    """Write per-step RMSE, loss, parameter range, and gradient summaries."""
    checkpoints_per_epoch = 5
    checkpoint_interval = max(1, len(train_loader) // checkpoints_per_epoch)
    if iteration_count % checkpoint_interval != 0:
        return rmse_per_time, loss_acc

    batch_step = epoch * len(train_loader) + iteration_count
    rmse_checkpoint = torch.sqrt(rmse_per_time / checkpoint_interval)
    writer.add_scalars("rec_RMSE_per_step", {f"time_{i:02d}": rmse_checkpoint[i] for i in range(t_in)}, global_step=batch_step)
    writer.add_scalars("pred_RMSE_per_step", {f"time_{i:02d}": rmse_checkpoint[i] for i in range(t_in, T)}, global_step=batch_step)
    writer.add_scalars("RMSE_per_step", {f"time_{i:02d}": rmse_checkpoint[i] for i in range(T)}, global_step=batch_step)
    writer.add_scalar("Loss_batch", loss_acc / checkpoint_interval, global_step=batch_step)

    param_dicts, grad_dict = log_parameters_scalars(model, ["multiQ", "multiM", "alpha", "beta"])
    for check_name, value_dict in param_dicts.items():
        writer.add_scalars(check_name, value_dict, global_step=batch_step)
    writer.add_scalars("grad", grad_dict, global_step=batch_step)
    return rmse_per_time * 0, 0.0


def _handle_training_value_error(error, model, test_loader, data_normalization, config, device, signal_channels, logger, train_loss_list, val_loss_list, plot_path):
    """Save current loss curve, run a final test pass, then re-raise the training error."""
    plot_loss_curve(train_loss_list, val_loss_list, plot_path)
    use_one_channel = get_model_config(config)["use_one_channel"]
    try:
        if use_one_channel:
            metrics = test(model, test_loader, data_normalization, False, config, device, signal_channels, use_one_channel=True)
            metrics_d = None
        else:
            metrics, metrics_d = test(model, test_loader, data_normalization, False, config, device, signal_channels, use_one_channel=False)

        logger.info(
            "Test (ALL): rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f",
            metrics["rec_RMSE"],
            metrics["pred_RMSE"],
            metrics["pred_MAE"],
            metrics["pred_MAPE"],
        )
        if metrics_d is not None:
            for i in range(signal_channels):
                logger.info(
                    "Test (%s): rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f",
                    SIGNAL_NAMES[i] if i < len(SIGNAL_NAMES) else f"channel_{i}",
                    metrics_d["rec_RMSE"][i],
                    metrics_d["pred_RMSE"][i],
                    metrics_d["pred_MAE"][i],
                    metrics_d["pred_MAPE"][i],
                )
    except Exception as test_error:
        logger.error("Skipped final test after training failure because evaluation also failed: %s", test_error)
    raise ValueError(str(error)) from error


def train_one_epoch(
    model,
    train_loader,
    data_normalization,
    config,
    device,
    optimizer,
    loss_fn,
    logger,
    grad_logger,
    writer,
    args,
    epoch,
    num_epochs,
    paths,
    masked_flag,
):
    """Run one training epoch and return epoch-level metrics."""
    model.train()
    model_config = get_model_config(config)
    T = model_config["t_in"] + model_config["t_out"]
    t_in = model_config["t_in"]
    rmse_per_time = torch.zeros((T,))
    loss_acc = 0.0
    metric_sums = _empty_train_metrics()
    total_batches = len(train_loader)

    for iter_idx, (y, x, t_list) in enumerate(tqdm(train_loader)):
        if iter_idx > 0 and iter_idx % 128 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        iteration_count = iter_idx + 1
        optimizer.zero_grad()
        y, x, t_list = y.to(device), x.to(device), t_list.to(device)

        try:
            loss, output = _train_forward(model, y, x, t_list, data_normalization, config, loss_fn, masked_flag, args)
        except (AssertionError, ValueError) as error:
            _raise_training_numerical_error(
                "model forward or loss computation failed",
                epoch,
                num_epochs,
                iteration_count,
                train_loader,
                [
                    ("input_y_raw", y),
                    ("target_x_raw", x),
                    ("time_list", t_list),
                ],
                logger,
                extra_lines=[
                    f"forward_error: {error}",
                    f"mode={args.mode}, normed_loss={config['normed_loss']}, "
                    f"use_one_channel={model_config['use_one_channel']}",
                ]
                + _format_glasso_fallback_events(model),
            )
        _log_glasso_fallback_events(model, logger, epoch, num_epochs, iteration_count, train_loader)
        if iteration_count % 20 == 0:
            theta_stats = collect_theta_support_deltas(model)
            tqdm.write(format_theta_support_batch(theta_stats, epoch, iteration_count, total_batches))

        if not torch.isfinite(loss).all():
            _raise_training_numerical_error(
                "loss is NaN or Inf before backward",
                epoch,
                num_epochs,
                iteration_count,
                train_loader,
                [
                    ("loss", loss),
                    ("input_y_raw", y),
                    ("target_x_raw", x),
                    ("model_output_recovered", output),
                ],
                logger,
                extra_lines=[
                    f"mode={args.mode}, normed_loss={config['normed_loss']}, "
                    f"use_one_channel={model_config['use_one_channel']}"
                ],
            )

        loss.backward()
        nan_info = check_nan_gradients(model)
        if nan_info is not None:
            diagnostic_lines = _diagnose_bad_backward(
                model,
                y,
                x,
                t_list,
                data_normalization,
                config,
                loss_fn,
                masked_flag,
                args,
                optimizer,
                _training_location(epoch, num_epochs, iteration_count, train_loader),
            )
            _raise_training_numerical_error(
                f"{nan_info['kind']} has NaN or Inf after backward",
                epoch,
                num_epochs,
                iteration_count,
                train_loader,
                [
                    ("loss", loss),
                    ("input_y_raw", y),
                    ("target_x_raw", x),
                    ("model_output_recovered", output),
                ],
                logger,
                extra_lines=[
                    f"first_bad_tensor: {_format_bad_tensor_info(nan_info)}",
                    f"mode={args.mode}, normed_loss={config['normed_loss']}, "
                    f"use_one_channel={model_config['use_one_channel']}",
                ]
                + _format_glasso_fallback_events(model)
                + diagnostic_lines,
            )

        optimizer.step()
        nan_info = check_nan_gradients(model)
        if nan_info is not None:
            _raise_training_numerical_error(
                f"{nan_info['kind']} has NaN or Inf after optimizer step",
                epoch,
                num_epochs,
                iteration_count,
                train_loader,
                [
                    ("loss", loss),
                    ("input_y_raw", y),
                    ("target_x_raw", x),
                    ("model_output_recovered", output),
                ],
                logger,
                extra_lines=[
                    f"first_bad_tensor: {_format_bad_tensor_info(nan_info)}",
                    f"mode={args.mode}, normed_loss={config['normed_loss']}, "
                    f"use_one_channel={model_config['use_one_channel']}",
                    "Bad values appeared after optimizer.step(); optimizer state follows.",
                ]
                + _format_glasso_fallback_events(model)
                + _optimizer_state_health(optimizer),
            )
        if args.loggrad != -1:
            log_gradients(epoch, num_epochs, iteration_count, train_loader, model, grad_logger, args)

        if config["clamp"] > 0:
            model.clamp_param(config["clamp"])
        elif config["clamp"] == 0:
            model.clamp_param()
        if args.debug:
            torch.save(model.state_dict(), paths.debug_model_path)

        rmse_per_time += ((x - output) ** 2).detach().cpu().mean((0, 2, 3))
        loss_acc += loss.detach().cpu().item()
        rmse_per_time, loss_acc = _maybe_write_tensorboard(
            writer,
            model,
            output,
            x,
            loss,
            rmse_per_time,
            loss_acc,
            epoch,
            iteration_count,
            train_loader,
            T,
            t_in,
        )
        _update_train_metrics(metric_sums, loss, output, x, t_in, masked_flag)

    logger.info("output: (%f, %f)", output.detach().cpu().max().item(), output.detach().cpu().min().item())
    return _finalize_train_metrics(metric_sums, len(train_loader))


def evaluate(model, loader, data_normalization, masked_flag, config, device, signal_channels, loss_fn):
    """Run validation/test evaluation and normalize return values across channel modes."""
    try:
        if get_model_config(config)["use_one_channel"]:
            val_loss, metrics = test(
                model,
                loader,
                data_normalization,
                masked_flag,
                config,
                device,
                signal_channels,
                mode="val",
                loss_fn=loss_fn,
                use_one_channel=True,
            )
            return val_loss, metrics, None

        val_loss, metrics, metrics_d = test(
            model,
            loader,
            data_normalization,
            masked_flag,
            config,
            device,
            signal_channels,
            mode="val",
            loss_fn=loss_fn,
            use_one_channel=False,
        )
        return val_loss, metrics, metrics_d
    except Exception as error:
        context = "\n".join(_format_glasso_fallback_events(model))
        raise RuntimeError(f"evaluation failed: {error}\n{context}") from error


def log_eval_metrics(logger, prefix, epoch_text, loss, metrics, metrics_d, signal_channels):
    """Write validation/test metrics to the logger, including per-channel metrics."""
    logger.info(
        "%s: Epoch [%s], Loss:%.4f, rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f",
        prefix,
        epoch_text,
        loss,
        metrics["rec_RMSE"],
        metrics["pred_RMSE"],
        metrics["pred_MAE"],
        metrics["pred_MAPE"],
    )
    if metrics_d is not None:
        for i in range(signal_channels):
            signal_name = SIGNAL_NAMES[i] if i < len(SIGNAL_NAMES) else f"channel_{i}"
            logger.info(
                "Channel %s:\t rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f",
                signal_name,
                metrics_d["rec_RMSE"][i],
                metrics_d["pred_RMSE"][i],
                metrics_d["pred_MAE"][i],
                metrics_d["pred_MAPE"][i],
            )


def maybe_periodic_test(model, test_loader, data_normalization, masked_flag, config, device, signal_channels, loss_fn, logger, model_dir, best_epoch, epoch, num_epochs):
    """Every 10 epochs, evaluate the saved best checkpoint when it is recent."""
    if (epoch + 1) % 10 != 0 or best_epoch <= epoch + 1 - 10:
        return

    gc.collect()
    torch.cuda.empty_cache()
    best_model = copy.deepcopy(model)
    checkpoint_path = os.path.join(model_dir, f"val_{best_epoch}.pth")
    try:
        state_dict = torch.load(checkpoint_path, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path)
    best_model.load_state_dict(state_dict)
    best_model.zero_grad()

    test_loss, test_metrics, test_metrics_d = evaluate(
        best_model,
        test_loader,
        data_normalization,
        masked_flag,
        config,
        device,
        signal_channels,
        loss_fn,
    )
    log_eval_metrics(logger, "Test", f"{best_epoch}/{epoch + 1}/{num_epochs}", test_loss, test_metrics, test_metrics_d, signal_channels)

    gc.collect()
    torch.cuda.empty_cache()
    del best_model


def main(argv=None):
    """CLI entry point for traffic training."""
    args, config = parse_args(argv)
    config = apply_args_to_config(config, args)
    device = prepare_runtime(args)
    loss_fn = build_loss_fn(config)
    names = build_experiment_names(config, args)
    paths = create_training_paths(names)
    writer = create_writer(paths.tensorboard_logdir)
    logger, grad_logger = create_loggers(paths, names, args)

    datasets_and_loaders, data_normalization = create_data(config, args, device)
    train_set, val_set, test_set, train_loader, val_loader, test_loader = datasets_and_loaders
    signal_channels = train_set.signal_channel
    _print_data_summary(train_set, val_set, test_set, data_normalization, args.mode, signal_channels)

    print("args.ablation", args.ablation)
    model, admm_info = create_model(config, args, train_set, device)
    optimizer, scheduler = create_optimizer_and_scheduler(config, args, model)
    model_pretrained_path = None
    log_run_header(logger, args, config, model, train_set, signal_channels, admm_info, model_pretrained_path)
    if grad_logger is not None:
        grad_logger.info("------BEGIN TRAINING PROCESS-------")

    print("log path", os.path.join(paths.log_dir, names.log_filename))
    print("tensorboard log path", paths.tensorboard_logdir)

    masked_flag = False
    best_val_loss = 20
    best_epoch = args.start_epochs
    if args.start_epochs > 0:
        model_pretrained_path = os.path.join(paths.model_dir, f"val_{args.start_epochs}.pth")
        model = change_model_location(model, model_pretrained_path, device)

    train_loss_list = []
    val_loss_list = []
    num_epochs = args.epochs

    try:
        for epoch in range(args.start_epochs, num_epochs):
            try:
                train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    data_normalization,
                    config,
                    device,
                    optimizer,
                    loss_fn,
                    logger,
                    grad_logger,
                    writer,
                    args,
                    epoch,
                    num_epochs,
                    paths,
                    masked_flag,
                )
            except ValueError as error:
                _handle_training_value_error(
                    error,
                    model,
                    test_loader,
                    data_normalization,
                    config,
                    device,
                    signal_channels,
                    logger,
                    train_loss_list,
                    val_loss_list,
                    paths.plot_path,
                )

            train_loss_list.append(train_metrics["loss"])
            logger.info(
                "Training: Epoch [%d/%d], LR:%.2e, Loss:%.4f, rec_RMSE: %.4f, "
                "RMSE_next:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f",
                epoch + 1,
                num_epochs,
                optimizer.param_groups[0]["lr"],
                train_metrics["loss"],
                train_metrics["rec_RMSE"],
                train_metrics["nearest_RMSE"],
                train_metrics["pred_RMSE"],
                train_metrics["pred_MAE"],
                train_metrics["pred_MAPE"],
            )
            val_loss, metrics, metrics_d = evaluate(
                model,
                val_loader,
                data_normalization,
                masked_flag,
                config,
                device,
                signal_channels,
                loss_fn,
            )
            val_loss_list.append(val_loss)
            log_eval_metrics(logger, "Validation", f"{epoch + 1}/{num_epochs}", val_loss, metrics, metrics_d, signal_channels)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                logger.info("saved best params at epoch %d", epoch + 1)
                torch.save(model.state_dict(), os.path.join(paths.model_dir, f"val_{epoch + 1}.pth"))

            if scheduler is not None:
                scheduler.step(val_loss)
                logger.info("Current Learning Rate: %.2e", optimizer.param_groups[0]["lr"])

            maybe_periodic_test(
                model,
                test_loader,
                data_normalization,
                masked_flag,
                config,
                device,
                signal_channels,
                loss_fn,
                logger,
                paths.model_dir,
                best_epoch,
                epoch,
                num_epochs,
            )
    finally:
        plot_loss_curve(train_loss_list, val_loss_list, paths.plot_path)
        writer.close()


if __name__ == "__main__":
    main()
