"""Utilities used by the traffic training entry point.

The training script should mainly describe the epoch/batch loop.  Argument
parsing, experiment naming, data/model construction, and logging setup live here
so they are easier to inspect or replace.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler

from dataset.dataset_utils import (
    Normalization,
    create_dataloader,
    create_directed_dataloader,
)
from lib.unrolling_model import UnrollingModel
from utils import WeightedMSELoss, seed_everything, setup_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentNames:
    logs_dir: str
    experiment_dir: str
    experiment_name: str
    log_filename: str


@dataclass
class TrainingPaths:
    tensorboard_logdir: str
    log_dir: str
    grad_logger_dir: str
    debug_model_path: str
    model_dir: str
    plot_dir: str
    plot_path: str


def _resolve_path(path):
    """Resolve a user path relative to cwd, project root, then `train/`."""
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    train_path = PROJECT_ROOT / "train" / path
    return train_path if train_path.exists() else path


def load_config(config_path):
    """Load the YAML config used to populate parser defaults and model settings."""
    config_path = _resolve_path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args(argv=None):
    """Parse command-line arguments for traffic training."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    pre_args, _ = pre_parser.parse_known_args(argv)
    config = load_config(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--cuda", help="CUDA device", default=-1, type=int)
    parser.add_argument("--dataset", help="dataset name", type=str, required=True)
    parser.add_argument("--batchsize", help="batch size", type=int, required=True)
    parser.add_argument("--mode", help="normalization mode", default="standardize", type=str)

    parser.add_argument("--neighbors", help="kNN neighbors", default=config["model"]["kNN"], type=int)
    parser.add_argument("--interval", help="intervals for time graph", default=config["model"]["interval"], type=int)
    parser.add_argument("--FElayers", help="feature extractor layers", default=1, type=int)
    parser.add_argument("--ablation", help="operator to eliminate in ablation study", default="None", type=str)

    parser.add_argument("--seed", help="random seed", default=3407, type=int)
    parser.add_argument("--debug", dest="debug", help="save debug model every iteration", action="store_true")
    parser.set_defaults(debug=False)

    parser.add_argument("--stepLR", dest="use_stepLR", action="store_true")
    parser.set_defaults(use_stepLR=False)
    parser.add_argument("--stepsize", help="stepLR stepsize", default=8, type=int)
    parser.add_argument("--gamma", help="stepLR gamma", default=0.2, type=float)

    parser.add_argument("--sharedM", dest="sharedM", action="store_true")
    parser.set_defaults(sharedM=config["model"]["sharedM"])
    parser.add_argument("--sharedQ", dest="sharedQ", action="store_true")
    parser.set_defaults(sharedQ=config["model"]["sharedQ"])
    parser.add_argument("--sharedV", dest="diff_interval", action="store_false")
    parser.set_defaults(diff_interval=config["model"]["diff_interval"])

    parser.add_argument("--epochs", help="running epochs", default=70, type=int)
    parser.add_argument("--start_epochs", help="start epochs", default=0, type=int)
    parser.add_argument("--loggrad", help="log gradient norms; -1 disables it", default=-1, type=int)

    parser.add_argument("--tout", help="t_out", default=config["model"]["t_out"], type=int)
    parser.add_argument("--trunc", dest="trunc", action="store_true")
    parser.set_defaults(trunc=False)
    parser.add_argument("--le-emb", help="learnable embedding", dest="le_emb", action="store_true")
    parser.set_defaults(le_emb=False)
    parser.add_argument("--blocks", help="number of ADMM blocks", default=config["model"]["num_blocks"], type=int)
    parser.add_argument("--layers", help="number of ADMM iterations per block", default=config["model"]["num_layers"], type=int)
    parser.add_argument("--CGiters", help="number of CG iterations", default=config["model"]["CG_iters"], type=int)
    parser.add_argument("--stride", help="sampling stride of dataset", default=config["data_stride"], type=int)
    parser.add_argument("--lr", help="learning rate", default=config["learning_rate"], type=float)
    parser.add_argument("--predonly", dest="pred_only", action="store_true")
    parser.set_defaults(pred_only=False)

    args = parser.parse_args(argv)
    return args, config


def apply_args_to_config(config, args):
    """Apply CLI overrides to the loaded config in-place."""
    config["model"]["kNN"] = args.neighbors
    config["model"]["interval"] = args.interval
    config["model"]["sharedM"] = args.sharedM
    config["model"]["sharedQ"] = args.sharedQ
    config["model"]["diff_interval"] = args.diff_interval
    config["model"]["num_blocks"] = args.blocks
    config["model"]["num_layers"] = args.layers
    config["model"]["CG_iters"] = args.CGiters
    config["model"]["t_out"] = args.tout
    return config


def prepare_runtime(args):
    """Set random seed and choose the device used for the training run."""
    seed_everything(args.seed)
    if args.cuda != -1 and torch.cuda.is_available():
        return torch.device(f"cuda:{args.cuda}")
    return torch.device("cpu")


def build_loss_fn(config):
    """Build the loss function specified by `config['loss_function']`."""
    loss_name = config["loss_function"]
    if loss_name == "MSE":
        return nn.MSELoss()
    if loss_name == "Huber":
        return nn.HuberLoss(delta=1)
    if loss_name == "Mix":
        t_in = config["model"]["t_in"]
        return WeightedMSELoss(t_in, t_in + config["model"]["t_out"])
    raise ValueError("config['loss_function'] should be one of: MSE, Huber, Mix")


def resolve_dataset_dir(dataset_name, base_dir="../TS_datasets/"):
    """Return the dataset root used by the old traffic experiments."""
    dataset_dir = base_dir
    if "PEMS0" in dataset_name:
        dataset_dir = os.path.join(dataset_dir, "PEMS0X_data")
    elif dataset_name in ["gba", "sd"]:
        dataset_dir = os.path.join(dataset_dir, "LargeST")
    return dataset_dir


def build_experiment_names(config, args, learning_rate):
    """Create log/model directory names without touching the filesystem."""
    logs_dir = "logs_learnable_emb" if args.le_emb else "dense_logs_new"
    experiment_dir = f"lr_{learning_rate:.0e}_seed_{args.seed}"

    dataset_name = args.dataset
    num_blocks = config["model"]["num_blocks"]
    num_layers = config["model"]["num_layers"]
    num_heads = config["model"]["num_heads"]
    interval = config["model"]["interval"]
    feature_channels = config["model"]["feature_channels"]
    loss_name = config["loss_function"]

    name = f"{dataset_name}_s{args.stride}_{num_blocks}b{num_layers}_{num_heads}h_{feature_channels}f_{args.FElayers}FE"
    if args.pred_only:
        name = "predOnly_" + name
    if args.trunc:
        name = "trunc_" + name
    if args.ablation != "None":
        name = f"wo_{args.ablation}_" + name
    if not config["model"]["use_extrapolation"]:
        name = "LR_" + name
    if not config["model"]["use_one_channel"]:
        name = "AllChannel_" + name
    if config["model"]["sharedM"]:
        name = "shareM_" + name
    if config["model"]["sharedQ"]:
        name = "shareQ_" + name
    if config["model"]["diff_interval"]:
        name = "diffV_" + name
    name += "_normed_loss" if config["normed_loss"] else "_true_loss"

    experiment_name = os.path.join(experiment_dir, name)
    log_filename = f"nn_{config['model']['kNN']}_int_{interval}_{loss_name}.log"
    return ExperimentNames(logs_dir, experiment_dir, experiment_name, log_filename)


def create_training_paths(names):
    """Create all filesystem paths used by one training run."""
    run_id = names.log_filename[:-4] if names.log_filename.endswith(".log") else names.log_filename
    tensorboard_logdir = f"./{names.logs_dir}/TB_log/{names.experiment_name}/{run_id}"
    log_dir = f"./{names.logs_dir}/train_Logs/{names.experiment_name}"
    grad_logger_dir = f"./{names.logs_dir}/grad_logs/{names.experiment_name}"
    debug_model_path = os.path.join(f"./{names.logs_dir}/debug_models/{names.experiment_name}", f"{run_id}.pth")
    model_dir = os.path.join(f"./{names.logs_dir}/models/{names.experiment_name}", run_id)
    plot_dir = f"./{names.logs_dir}/loss_curves/{names.experiment_name}"
    plot_path = os.path.join(plot_dir, f"{run_id}.png")

    for path in [tensorboard_logdir, log_dir, model_dir, plot_dir, os.path.dirname(debug_model_path)]:
        os.makedirs(path, exist_ok=True)
    return TrainingPaths(tensorboard_logdir, log_dir, grad_logger_dir, debug_model_path, model_dir, plot_dir, plot_path)


def create_writer(logdir):
    """Create TensorBoard writer lazily so imports do not require TensorBoard."""
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(logdir)


def create_loggers(paths, names, args):
    """Create the main logger and, if requested, a separate gradient logger."""
    logger = setup_logger("traffic_train", os.path.join(paths.log_dir, names.log_filename), logging.DEBUG, to_console=True)
    grad_logger = None
    if args.loggrad != -1:
        os.makedirs(paths.grad_logger_dir, exist_ok=True)
        grad_logger = setup_logger("traffic_grad", os.path.join(paths.grad_logger_dir, names.log_filename), logging.INFO)
    return logger, grad_logger


def create_data(config, args, device):
    """Load train/val/test splits and create the normalization object."""
    dataset_dir = resolve_dataset_dir(args.dataset)
    T = config["model"]["t_in"] + config["model"]["t_out"]
    t_in = config["model"]["t_in"]
    return_time = True

    if "PEMS0" in args.dataset:
        datasets_and_loaders = create_dataloader(
            dataset_dir,
            args.dataset,
            T,
            t_in,
            args.stride,
            args.batchsize,
            config["num_workers"],
            return_time,
            use_one_channel=config["model"]["use_one_channel"],
            truncated=args.trunc,
        )
    else:
        datasets_and_loaders = create_directed_dataloader(
            dataset_dir,
            args.dataset,
            T,
            t_in,
            args.stride,
            args.batchsize,
            config["num_workers"],
            return_time,
            use_one_channel=config["model"]["use_one_channel"],
        )

    train_set = datasets_and_loaders[0]
    data_normalization = Normalization(train_set, args.mode, device)
    return datasets_and_loaders, data_normalization


def build_admm_info(config):
    """Collect ADMM hyperparameters in the format expected by `UnrollingModel`."""
    return {
        "ADMM_iters": config["model"]["num_layers"],
        "CG_iters": config["model"]["CG_iters"],
        "PGD_iters": config["model"]["PGD_iters"],
        "mu_u_init": config["ADMM_params"]["mu_u"],
        "mu_d1_init": config["ADMM_params"]["mu_d1"],
        "mu_d2_init": config["ADMM_params"]["mu_d2"],
    }


def create_model(config, args, train_set, device):
    """Build `UnrollingModel` from config, CLI overrides, and dataset graph info."""
    T = config["model"]["t_in"] + config["model"]["t_out"]
    t_in = config["model"]["t_in"]
    admm_info = build_admm_info(config)

    model = UnrollingModel(
        config["model"]["num_blocks"],
        device,
        T,
        t_in,
        config["model"]["num_heads"],
        config["model"]["interval"],
        train_set.signal_channel,
        config["model"]["feature_channels"],
        GNN_layers=2,
        graph_info=train_set.graph_info,
        ADMM_info=admm_info,
        k_hop=config["model"]["kNN"],
        ablation=args.ablation,
        st_emb_info=config["st_emb_info"],
        use_extrapolation=config["model"]["use_extrapolation"],
        extrapolation_agg_layers=args.FElayers,
        use_one_channel=config["model"]["use_one_channel"],
        sharedM=config["model"]["sharedM"],
        sharedQ=config["model"]["sharedQ"],
        diff_interval=config["model"]["diff_interval"],
        predict_only=args.pred_only,
        le_emb=args.le_emb,
    ).to(device)
    return model, admm_info


def create_optimizer_and_scheduler(config, args, model):
    """Create optimizer and optional ReduceLROnPlateau scheduler."""
    if config["optim"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif config["optim"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config["weight_decay"])
    else:
        raise ValueError("config['optim'] should be adam or adamw")

    scheduler = None
    if args.use_stepLR:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.gamma,
            patience=5,
            cooldown=5,
            min_lr=5e-6,
        )
    return optimizer, scheduler


def log_run_header(logger, args, config, model, train_set, signal_channels, admm_info, model_pretrained_path=None):
    """Write command, arguments, config, and model summary to the main log."""
    logger.info("#################################################")
    logger.info("Training CMD:\t" + " ".join(sys.argv))
    logger.info("PARAMETER SETTINGS:")
    for arg, value in vars(args).items():
        logger.info("\t %s: %s", arg, value)

    logger.info("CONFIG SETTINGS:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info("\t%s:", key)
            for k, v in value.items():
                logger.info("\t\t%s: %s", k, v)
        else:
            logger.info("\t%s: %s", key, value)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    logger.info("pretrained path: %s", model_pretrained_path)
    logger.info("feature channels: %d", config["model"]["feature_channels"])
    logger.info("Total parameters: %d", total_params)
    logger.info("PARAMETER SETTINGS:")
    logger.info("ADMM blocks: %d", config["model"]["num_blocks"])
    logger.info("ADMM info: %s", admm_info)
    logger.info(
        "graph info: nodes %d, edges %d, signal channels %d",
        train_set.n_nodes,
        train_set.n_edges,
        signal_channels,
    )
    logger.info("--------BEGIN TRAINING PROCESS------------")
