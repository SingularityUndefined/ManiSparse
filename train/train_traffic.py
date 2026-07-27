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
    create_data,
    create_loggers,
    create_model,
    create_optimizer_and_scheduler,
    create_training_paths,
    create_writer,
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


def _train_forward(model, y, x, t_list, data_normalization, config, loss_fn, masked_flag, args):
    """Normalize one training batch, run the model, recover output, and compute loss."""
    t_in = config["model"]["t_in"]
    use_one_channel = config["model"]["use_one_channel"]

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
    use_one_channel = config["model"]["use_one_channel"]
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
    T = config["model"]["t_in"] + config["model"]["t_out"]
    t_in = config["model"]["t_in"]
    rmse_per_time = torch.zeros((T,))
    loss_acc = 0.0
    metric_sums = _empty_train_metrics()

    for iter_idx, (y, x, t_list) in enumerate(tqdm(train_loader)):
        if iter_idx > 0 and iter_idx % 128 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        iteration_count = iter_idx + 1
        optimizer.zero_grad()
        y, x, t_list = y.to(device), x.to(device), t_list.to(device)

        loss, output = _train_forward(model, y, x, t_list, data_normalization, config, loss_fn, masked_flag, args)
        if torch.isnan(loss).any():
            message = f"Loss is NaN in [Epoch {epoch + 1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}]"
            logger.error(message)
            raise ValueError(message)

        loss.backward()
        nan_name = check_nan_gradients(model)
        if nan_name is not None:
            message = (
                f"Gradient has NaN/Inf value in [Epoch {epoch + 1}/{num_epochs}, "
                f"Iter {iteration_count}/{len(train_loader)}] first in {nan_name}"
            )
            logger.error(message)
            raise ValueError(message)

        optimizer.step()
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
    if config["model"]["use_one_channel"]:
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
    names = build_experiment_names(config, args, args.lr)
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
