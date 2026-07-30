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
from clean_lib.unrolling_model import UnrollingModel
from utils import WeightedMSELoss, seed_everything, setup_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


MODEL_CONFIG_PATHS = {
    "t_in": ("sequence", "t_in"),
    "t_out": ("sequence", "t_out"),
    "use_one_channel": ("sequence", "use_one_channel"),
    "kNN": ("graph", "kNN"),
    "theta_method": ("theta", "method"),
    "theta_kNN": ("theta", "local_kNN"),
    "interval": ("graph", "interval"),
    "num_blocks": ("architecture", "num_blocks"),
    "num_layers": ("architecture", "num_layers"),
    "CG_iters": ("architecture", "CG_iters"),
    "num_heads": ("architecture", "num_heads"),
    "feature_channels": ("architecture", "feature_channels"),
    "use_extrapolation": ("architecture", "use_extrapolation"),
    "le_emb": ("architecture", "learnable_embedding"),
    "use_deflation": ("deflation", "enabled"),
    "deflation_samples": ("deflation", "samples"),
    "deflation_CG_iters": ("deflation", "CG_iters"),
    "deflation_tol": ("deflation", "tol"),
    "glasso_backend": ("theta", "glasso", "backend"),
    "glasso_alpha": ("theta", "glasso", "alpha"),
    "glasso_rho": ("theta", "glasso", "rho"),
    "glasso_eps": ("theta", "glasso", "eps"),
    "glasso_eigh_shift": ("theta", "glasso", "eigh_shift"),
    "glasso_eigh_shift_retries": ("theta", "glasso", "eigh_shift_retries"),
    "glasso_fallback": ("theta", "glasso", "fallback"),
    "kalofolias_graph": ("theta", "kalofolias", "graph"),
    "kalofolias_alpha": ("theta", "kalofolias", "alpha"),
    "kalofolias_beta": ("theta", "kalofolias", "beta"),
    "kalofolias_max_iter": ("theta", "kalofolias", "max_iter"),
    "kalofolias_tol": ("theta", "kalofolias", "tol"),
    "kalofolias_threshold": ("theta", "kalofolias", "threshold"),
    "kalofolias_output_mode": ("theta", "kalofolias", "output_mode"),
    "kalofolias_normalize_distances": ("theta", "kalofolias", "normalize_distances"),
    "use_stable_graph_learning": ("graph_learning", "use_stable_graph_learning"),
    "sharedM": ("graph_learning", "sharedM"),
    "sharedQ": ("graph_learning", "sharedQ"),
    "diff_interval": ("graph_learning", "diff_interval"),
}


MODEL_DEFAULTS = {
    "use_deflation": True,
    "le_emb": True,
    "use_stable_graph_learning": False,
    "deflation_samples": 5,
    "deflation_tol": 1e-6,
    "theta_method": "glasso",
    "glasso_backend": "admm",
    "glasso_alpha": 0.2,
    "glasso_rho": 1.0,
    "glasso_eps": 0.0,
    "glasso_eigh_shift": 1e-6,
    "glasso_eigh_shift_retries": 4,
    "glasso_fallback": True,
    "kalofolias_alpha": 0.3,
    "kalofolias_beta": 1.0,
    "kalofolias_graph": "dense",
    "kalofolias_max_iter": 200,
    "kalofolias_tol": 1e-4,
    "kalofolias_threshold": 1e-4,
    "kalofolias_output_mode": "laplacian",
    "kalofolias_normalize_distances": True,
}


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


def _nested_get(mapping, path, default=None):
    """Return a nested config value, falling back to default when absent."""
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _nested_set(mapping, path, value):
    """Set a nested config value, creating intermediate dicts as needed."""
    current = mapping
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _flatten_model_config(model_section):
    """Build the internal flat model config from the grouped YAML section."""
    flat = {}
    for flat_key, path in MODEL_CONFIG_PATHS.items():
        value = _nested_get(model_section, path, None)
        if value is None and flat_key in model_section:
            value = model_section[flat_key]
        if value is not None:
            flat[flat_key] = value
    return flat


def get_model_config(config):
    """Return the internal flat model config generated from grouped YAML."""
    return config.get("_model", _flatten_model_config(config["model"]))


def _validate_model_config(model_config):
    """Validate the grouped model config after defaults and CLI overrides."""
    if model_config["theta_method"] not in {"glasso", "kalofolias"}:
        raise ValueError("model.theta.method must be one of: glasso, kalofolias")
    if model_config["glasso_backend"] not in {"admm", "quic", "sklearn"}:
        raise ValueError("model.theta.glasso.backend must be one of: admm, quic, sklearn")
    if model_config["kalofolias_graph"] not in {"dense", "local"}:
        raise ValueError("model.theta.kalofolias.graph must be one of: dense, local")
    if model_config["theta_method"] == "kalofolias" and model_config["kalofolias_graph"] == "local":
        if model_config["theta_kNN"] <= model_config["kNN"]:
            raise ValueError("model.theta.local_kNN must be larger than model.graph.kNN for local Kalofolias")


def _add_bool_override(parser, name, dest, help_text):
    """Add paired boolean flags that leave config unchanged when omitted."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", default=None, help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"disable {help_text}")


def parse_args(argv=None):
    """Parse command-line arguments for traffic training.

    YAML is the source of persistent defaults.  Most model/data hyperparameters
    therefore default to `None` here and only overwrite the loaded config when
    the user explicitly passes a CLI flag.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    pre_args, _ = pre_parser.parse_known_args(argv)
    config = load_config(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--cuda", help="CUDA device", default=-1, type=int)
    parser.add_argument("--dataset", help="dataset name", type=str, required=True)
    parser.add_argument("--batchsize", help="batch size", type=int, required=True)
    parser.add_argument("--mode", help="normalization mode", default="standardize", type=str)

    parser.add_argument("--neighbors", help="temporary override for model.graph.kNN", default=None, type=int)
    parser.add_argument("--theta-neighbors", help="temporary override for model.theta.local_kNN", default=None, type=int)
    parser.add_argument("--interval", help="temporary override for model.graph.interval", default=None, type=int)
    parser.add_argument("--FElayers", help="feature extractor layers", default=1, type=int)
    parser.add_argument("--ablation", help="operator to eliminate in ablation study", default="None", type=str)

    parser.add_argument("--seed", help="random seed", default=3407, type=int)
    parser.add_argument("--debug", dest="debug", help="save debug model every iteration", action="store_true")
    parser.set_defaults(debug=False)

    parser.add_argument("--stepLR", dest="use_stepLR", action="store_true")
    parser.set_defaults(use_stepLR=False)
    parser.add_argument("--stepsize", help="stepLR stepsize", default=8, type=int)
    parser.add_argument("--gamma", help="stepLR gamma", default=0.2, type=float)

    _add_bool_override(parser, "sharedM", "sharedM", "temporary override for model.graph_learning.sharedM")
    _add_bool_override(parser, "sharedQ", "sharedQ", "temporary override for model.graph_learning.sharedQ")
    _add_bool_override(parser, "diff-interval", "diff_interval", "temporary override for model.graph_learning.diff_interval")
    _add_bool_override(
        parser,
        "stable-graph-learning",
        "use_stable_graph_learning",
        "temporary override for model.graph_learning.use_stable_graph_learning",
    )

    parser.add_argument("--epochs", help="running epochs", default=70, type=int)
    parser.add_argument("--start_epochs", help="start epochs", default=0, type=int)
    parser.add_argument("--loggrad", help="log gradient norms; -1 disables it", default=-1, type=int)

    parser.add_argument("--tout", help="temporary override for model.sequence.t_out", default=None, type=int)
    parser.add_argument("--trunc", dest="trunc", action="store_true")
    parser.set_defaults(trunc=False)
    _add_bool_override(parser, "le-emb", "le_emb", "temporary override for model.architecture.learnable_embedding")
    parser.add_argument("--blocks", help="temporary override for model.architecture.num_blocks", default=None, type=int)
    parser.add_argument("--layers", help="temporary override for model.architecture.num_layers", default=None, type=int)
    parser.add_argument("--CGiters", help="temporary override for model.architecture.CG_iters", default=None, type=int)
    _add_bool_override(parser, "deflation", "deflation", "temporary override for model.deflation.enabled")
    parser.add_argument("--deflation-samples", help="temporary override for model.deflation.samples", default=None, type=int)
    parser.add_argument(
        "--deflation-CGiters",
        help="temporary override for model.deflation.CG_iters",
        default=None,
        type=int,
    )
    parser.add_argument("--deflation-tol", help="temporary override for model.deflation.tol", default=None, type=float)
    parser.add_argument("--theta-method", help="temporary override for model.theta.method", default=None, type=str, choices=["glasso", "kalofolias"])
    parser.add_argument("--glasso-backend", help="temporary override for model.theta.glasso.backend", default=None, type=str, choices=["admm", "quic", "sklearn"])
    parser.add_argument("--glasso-alpha", help="temporary override for model.theta.glasso.alpha", default=None, type=float)
    parser.add_argument("--glasso-rho", help="temporary override for model.theta.glasso.rho", default=None, type=float)
    parser.add_argument("--glasso-eps", help="temporary override for model.theta.glasso.eps", default=None, type=float)
    parser.add_argument("--glasso-eigh-shift", help="temporary override for model.theta.glasso.eigh_shift", default=None, type=float)
    parser.add_argument("--glasso-eigh-shift-retries", help="temporary override for model.theta.glasso.eigh_shift_retries", default=None, type=int)
    _add_bool_override(parser, "glasso-fallback", "glasso_fallback", "temporary override for model.theta.glasso.fallback")
    parser.add_argument("--kalofolias-graph", help="temporary override for model.theta.kalofolias.graph", default=None, type=str, choices=["dense", "local"])
    parser.add_argument("--kalofolias-alpha", help="temporary override for model.theta.kalofolias.alpha", default=None, type=float)
    parser.add_argument("--kalofolias-beta", help="temporary override for model.theta.kalofolias.beta", default=None, type=float)
    parser.add_argument("--kalofolias-max-iter", help="temporary override for model.theta.kalofolias.max_iter", default=None, type=int)
    parser.add_argument("--kalofolias-tol", help="temporary override for model.theta.kalofolias.tol", default=None, type=float)
    parser.add_argument("--kalofolias-threshold", help="temporary override for model.theta.kalofolias.threshold", default=None, type=float)
    parser.add_argument("--kalofolias-output-mode", help="temporary override for model.theta.kalofolias.output_mode", default=None, type=str)
    _add_bool_override(
        parser,
        "kalofolias-normalize-distances",
        "kalofolias_normalize_distances",
        "temporary override for model.theta.kalofolias.normalize_distances",
    )
    parser.add_argument("--stride", help="temporary override for data_stride", default=None, type=int)
    parser.add_argument("--lr", help="temporary override for learning_rate", default=None, type=float)
    parser.add_argument("--predonly", dest="pred_only", action="store_true")
    parser.set_defaults(pred_only=False)

    args = parser.parse_args(argv)
    return args, config


def apply_args_to_config(config, args):
    """Apply CLI overrides to the loaded config in-place."""
    model_config = _flatten_model_config(config["model"])
    arg_to_config_key = {
        "neighbors": "kNN",
        "theta_neighbors": "theta_kNN",
        "interval": "interval",
        "sharedM": "sharedM",
        "sharedQ": "sharedQ",
        "diff_interval": "diff_interval",
        "use_stable_graph_learning": "use_stable_graph_learning",
        "blocks": "num_blocks",
        "layers": "num_layers",
        "CGiters": "CG_iters",
        "tout": "t_out",
        "le_emb": "le_emb",
        "deflation_samples": "deflation_samples",
        "deflation_CGiters": "deflation_CG_iters",
        "deflation_tol": "deflation_tol",
        "deflation": "use_deflation",
        "theta_method": "theta_method",
        "glasso_backend": "glasso_backend",
        "glasso_alpha": "glasso_alpha",
        "glasso_rho": "glasso_rho",
        "glasso_eps": "glasso_eps",
        "glasso_eigh_shift": "glasso_eigh_shift",
        "glasso_eigh_shift_retries": "glasso_eigh_shift_retries",
        "glasso_fallback": "glasso_fallback",
        "kalofolias_graph": "kalofolias_graph",
        "kalofolias_alpha": "kalofolias_alpha",
        "kalofolias_beta": "kalofolias_beta",
        "kalofolias_max_iter": "kalofolias_max_iter",
        "kalofolias_tol": "kalofolias_tol",
        "kalofolias_threshold": "kalofolias_threshold",
        "kalofolias_output_mode": "kalofolias_output_mode",
        "kalofolias_normalize_distances": "kalofolias_normalize_distances",
    }
    for arg_name, config_key in arg_to_config_key.items():
        value = getattr(args, arg_name)
        if value is not None:
            model_config[config_key] = value
            _nested_set(config["model"], MODEL_CONFIG_PATHS[config_key], value)

    for key, value in MODEL_DEFAULTS.items():
        model_config.setdefault(key, value)
    model_config.setdefault("theta_kNN", model_config["kNN"] + 4)
    model_config.setdefault("deflation_CG_iters", model_config["CG_iters"])
    _validate_model_config(model_config)
    for key, value in model_config.items():
        _nested_set(config["model"], MODEL_CONFIG_PATHS[key], value)
    config["_model"] = model_config

    if args.stride is not None:
        config["data_stride"] = args.stride
    if args.lr is not None:
        config["learning_rate"] = args.lr
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
    model_config = get_model_config(config)
    if loss_name == "MSE":
        return nn.MSELoss()
    if loss_name == "Huber":
        return nn.HuberLoss(delta=1)
    if loss_name == "Mix":
        t_in = model_config["t_in"]
        return WeightedMSELoss(t_in, t_in + model_config["t_out"])
    raise ValueError("config['loss_function'] should be one of: MSE, Huber, Mix")


def resolve_dataset_dir(dataset_name, base_dir="../TS_datasets/"):
    """Return the dataset root used by the old traffic experiments."""
    dataset_dir = base_dir
    if "PEMS0" in dataset_name:
        dataset_dir = os.path.join(dataset_dir, "PEMS0X_data")
    elif dataset_name in ["gba", "sd"]:
        dataset_dir = os.path.join(dataset_dir, "LargeST")
    return dataset_dir


def build_experiment_names(config, args):
    """Create log/model directory names without touching the filesystem."""
    learning_rate = config["learning_rate"]
    model_config = get_model_config(config)
    logs_dir = "logs_learnable_emb" if model_config["le_emb"] else "dense_logs_new"

    dataset_name = args.dataset
    theta_method = model_config.get("theta_method", "glasso")
    theta_family = theta_method
    if theta_method == "kalofolias":
        theta_family = f"kalofolias_{model_config['kalofolias_graph']}"
    elif theta_method == "glasso":
        theta_family = f"glasso_{model_config['glasso_backend']}"
    lr_seed_dir = f"lr_{learning_rate:.0e}_seed_{args.seed}"
    num_blocks = model_config["num_blocks"]
    num_layers = model_config["num_layers"]
    num_heads = model_config["num_heads"]
    interval = model_config["interval"]
    feature_channels = model_config["feature_channels"]
    loss_name = config["loss_function"]

    name = (
        f"s{config['data_stride']}_{num_blocks}b{num_layers}_{num_heads}h_"
        f"{feature_channels}f_{args.FElayers}FE_"
        f"k{model_config['kNN']}_thetaK{model_config['theta_kNN']}_int{interval}"
    )
    if args.pred_only:
        name = "predOnly_" + name
    if model_config.get("use_deflation", False):
        name = f"deflate{model_config['deflation_samples']}_" + name
    if args.trunc:
        name = "trunc_" + name
    if args.ablation != "None":
        name = f"wo_{args.ablation}_" + name
    if not model_config["use_extrapolation"]:
        name = "LR_" + name
    if not model_config["use_one_channel"]:
        name = "AllChannel_" + name
    if model_config["sharedM"]:
        name = "shareM_" + name
    if model_config["sharedQ"]:
        name = "shareQ_" + name
    if model_config["diff_interval"]:
        name = "diffV_" + name
    name += "_normed_loss" if config["normed_loss"] else "_true_loss"

    experiment_dir = os.path.join(theta_family, dataset_name, lr_seed_dir)
    experiment_name = os.path.join(experiment_dir, name)
    log_filename = f"{loss_name}.log"
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
    model_config = get_model_config(config)
    dataset_dir = resolve_dataset_dir(args.dataset)
    T = model_config["t_in"] + model_config["t_out"]
    t_in = model_config["t_in"]
    stride = config["data_stride"]
    return_time = True

    if "PEMS0" in args.dataset:
        datasets_and_loaders = create_dataloader(
            dataset_dir,
            args.dataset,
            T,
            t_in,
            stride,
            args.batchsize,
            config["num_workers"],
            return_time,
            use_one_channel=model_config["use_one_channel"],
            truncated=args.trunc,
        )
    else:
        datasets_and_loaders = create_directed_dataloader(
            dataset_dir,
            args.dataset,
            T,
            t_in,
            stride,
            args.batchsize,
            config["num_workers"],
            return_time,
            use_one_channel=model_config["use_one_channel"],
        )

    train_set = datasets_and_loaders[0]
    data_normalization = Normalization(train_set, args.mode, device)
    return datasets_and_loaders, data_normalization


def build_admm_info(config):
    """Collect ADMM hyperparameters in the format expected by `UnrollingModel`."""
    model_config = get_model_config(config)
    return {
        "ADMM_iters": model_config["num_layers"],
        "CG_iters": model_config["CG_iters"],
        "mu_u_init": config["ADMM_params"]["mu_u"],
        "mu_d1_init": config["ADMM_params"]["mu_d1"],
        "mu_d2_init": config["ADMM_params"]["mu_d2"],
        "lambda_init": config["ADMM_params"]["lambda_theta"],
        "deflation_samples": model_config["deflation_samples"],
        "deflation_CG_iters": model_config["deflation_CG_iters"],
        "deflation_tol": model_config["deflation_tol"],
    }


def create_model(config, args, train_set, device):
    """Build `UnrollingModel` from config, CLI overrides, and dataset graph info."""
    model_config = get_model_config(config)
    T = model_config["t_in"] + model_config["t_out"]
    t_in = model_config["t_in"]
    admm_info = build_admm_info(config)

    model = UnrollingModel(
        model_config["num_blocks"],
        device,
        T,
        t_in,
        model_config["num_heads"],
        model_config["interval"],
        train_set.signal_channel,
        model_config["feature_channels"],
        GNN_layers=2,
        graph_info=train_set.graph_info,
        ADMM_info=admm_info,
        k_hop=model_config["kNN"],
        theta_k_hop=model_config["theta_kNN"],
        ablation=args.ablation,
        st_emb_info=config["st_emb_info"],
        use_extrapolation=model_config["use_extrapolation"],
        extrapolation_agg_layers=args.FElayers,
        use_one_channel=model_config["use_one_channel"],
        sharedM=model_config["sharedM"],
        sharedQ=model_config["sharedQ"],
        diff_interval=model_config["diff_interval"],
        use_stable_graph_learning=model_config["use_stable_graph_learning"],
        predict_only=args.pred_only,
        le_emb=model_config["le_emb"],
        use_deflation=model_config.get("use_deflation", False),
        deflation_samples=model_config["deflation_samples"],
        theta_method=model_config["theta_method"],
        glasso_backend=model_config["glasso_backend"],
        glasso_alpha=model_config["glasso_alpha"],
        glasso_rho=model_config["glasso_rho"],
        glasso_eps=model_config["glasso_eps"],
        glasso_eigh_shift=model_config["glasso_eigh_shift"],
        glasso_eigh_shift_retries=model_config["glasso_eigh_shift_retries"],
        glasso_fallback=model_config["glasso_fallback"],
        kalofolias_alpha=model_config["kalofolias_alpha"],
        kalofolias_beta=model_config["kalofolias_beta"],
        kalofolias_graph=model_config["kalofolias_graph"],
        kalofolias_max_iter=model_config["kalofolias_max_iter"],
        kalofolias_tol=model_config["kalofolias_tol"],
        kalofolias_threshold=model_config["kalofolias_threshold"],
        kalofolias_output_mode=model_config["kalofolias_output_mode"],
        kalofolias_normalize_distances=model_config["kalofolias_normalize_distances"],
    ).to(device)
    return model, admm_info


def create_optimizer_and_scheduler(config, args, model):
    """Create optimizer and optional ReduceLROnPlateau scheduler."""
    learning_rate = config["learning_rate"]
    if config["optim"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif config["optim"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config["weight_decay"])
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


def collect_theta_support_deltas(model):
    """Return per-block Theta support changes against the original kNN graph.

    Theta is estimated inside `UnrollingModel.forward`, so these values describe
    the most recent batch that passed through the model.  Block 0 is skipped
    because the first unrolled block intentionally does not receive Theta.
    A `None` Theta means the branch was skipped, for example when
    `ablation="Theta"`.

    For local Kalofolias, the comparison is row-local and directed:
        baseline support: model.nearest_nodes[:, 1:] with shape (N, kNN)
        learned support: Theta over theta_neighbor_list, shape (N, theta.local_kNN)
    The reported values are average added and removed edges per node.
    """
    stats = []
    for block_idx, block in enumerate(model.model_blocks):
        if block_idx == 0:
            continue
        admm_block = block["ADMM_block"]
        theta = getattr(admm_block, "Theta", None)
        lambda_theta = _summarize_lambda_theta(getattr(admm_block, "lambda_theta", None))
        if theta is None:
            stats.append(
                {
                    "block_idx": block_idx,
                    "ratio": None,
                    "nnz": 0,
                    "numel": 0,
                    "shape": None,
                    "support_delta": None,
                    "lambda_theta": lambda_theta,
                }
            )
            continue

        theta = torch.as_tensor(theta).detach()
        nnz = torch.count_nonzero(theta).item()
        numel = theta.numel()
        support_delta = _theta_support_delta_against_knn(model, admm_block, theta)
        stats.append(
            {
                "block_idx": block_idx,
                "ratio": nnz / numel if numel > 0 else float("nan"),
                "nnz": nnz,
                "numel": numel,
                "shape": tuple(theta.shape),
                "support_delta": support_delta,
                "lambda_theta": lambda_theta,
            }
        )
    return stats


def _knn_baseline_mask(model, candidate_nodes, device):
    """Return mask showing which candidate slots are in the original kNN graph."""
    base_neighbors = torch.as_tensor(model.nearest_nodes[:, 1:], device=device, dtype=torch.long)
    candidate_nodes = torch.as_tensor(candidate_nodes, device=device, dtype=torch.long)
    if candidate_nodes.ndim != 2:
        raise ValueError("candidate_nodes must have shape (N, K_candidate)")
    valid_base = base_neighbors >= 0
    return ((candidate_nodes.unsqueeze(-1) == base_neighbors.unsqueeze(1)) & valid_base.unsqueeze(1)).any(dim=-1)


def _theta_support_delta_against_knn(model, admm_block, theta):
    """Compare learned Theta support to the original kNN support.

    Returns average per-node support changes:
        added_per_node: learned Theta edges outside the original kNN list.
        removed_per_node: original kNN edges missing from learned Theta.
    """
    device = theta.device
    uses_local_kalofolias = (
        getattr(model, "theta_method", "").lower() == "kalofolias"
        and getattr(model, "kalofolias_graph", "").lower() == "local"
    )
    threshold = getattr(model, "kalofolias_threshold", 0.0) if uses_local_kalofolias else 0.0
    base_neighbors = torch.as_tensor(model.nearest_nodes[:, 1:], device=device, dtype=torch.long)
    original_baseline_count = (base_neighbors >= 0).sum(dim=-1).to(torch.float32).unsqueeze(0)

    if getattr(admm_block, "theta_neighbor_list", None) is not None and theta.ndim in (2, 3):
        candidate_nodes = torch.as_tensor(admm_block.theta_neighbor_list, device=device, dtype=torch.long)
        baseline_mask = _knn_baseline_mask(model, candidate_nodes, device)
        learned_support = torch.abs(theta) > threshold
        if learned_support.ndim == 2:
            learned_support = learned_support.unsqueeze(0)

        baseline_mask_batched = baseline_mask.unsqueeze(0).expand(learned_support.size(0), -1, -1)
        added_per_node = torch.logical_and(learned_support, ~baseline_mask_batched).sum(dim=-1).to(torch.float32)
        learned_base_count = torch.logical_and(learned_support, baseline_mask_batched).sum(dim=-1).to(torch.float32)
        removed_per_node = (original_baseline_count - learned_base_count).clamp_min(0.0)
        learned_per_node = learned_support.sum(dim=-1).to(torch.float32)

        return {
            "kind": "local",
            "threshold": float(threshold),
            "baseline_edges_per_node": float(original_baseline_count.mean().detach().cpu()),
            "learned_edges_per_node": float(learned_per_node.mean().detach().cpu()),
            "added_per_node": float(added_per_node.mean().detach().cpu()),
            "removed_per_node": float(removed_per_node.mean().detach().cpu()),
        }

    if theta.ndim in (2, 3):
        n_nodes = theta.size(-1)
        candidate_nodes = torch.arange(n_nodes, device=device).repeat(n_nodes, 1)
        baseline_mask = _knn_baseline_mask(model, candidate_nodes, device)
        self_mask = torch.eye(n_nodes, device=device, dtype=torch.bool)
        baseline_mask = torch.logical_and(baseline_mask, ~self_mask)
        learned_support = torch.abs(theta) > threshold
        if learned_support.ndim == 2:
            learned_support = learned_support.unsqueeze(0)
        learned_support = torch.logical_and(learned_support, ~self_mask.unsqueeze(0))

        baseline_mask_batched = baseline_mask.unsqueeze(0).expand(learned_support.size(0), -1, -1)
        added_per_node = torch.logical_and(learned_support, ~baseline_mask_batched).sum(dim=-1).to(torch.float32)
        learned_base_count = torch.logical_and(learned_support, baseline_mask_batched).sum(dim=-1).to(torch.float32)
        removed_per_node = (original_baseline_count - learned_base_count).clamp_min(0.0)
        learned_per_node = learned_support.sum(dim=-1).to(torch.float32)

        return {
            "kind": "dense",
            "threshold": float(threshold),
            "baseline_edges_per_node": float(original_baseline_count.mean().detach().cpu()),
            "learned_edges_per_node": float(learned_per_node.mean().detach().cpu()),
            "added_per_node": float(added_per_node.mean().detach().cpu()),
            "removed_per_node": float(removed_per_node.mean().detach().cpu()),
        }

    return {"kind": "unsupported"}


def _summarize_lambda_theta(lambda_theta):
    """Return scalar or vector summary for the ADMM Theta weight."""
    if lambda_theta is None:
        return {"kind": "missing"}

    values = torch.as_tensor(lambda_theta).detach().to(torch.float32).reshape(-1)
    if values.numel() == 0:
        return {"kind": "empty"}
    if values.numel() == 1:
        return {"kind": "scalar", "value": values.item()}

    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return {"kind": "nonfinite", "numel": values.numel()}

    return {
        "kind": "vector",
        "min": finite_values.min().item(),
        "median": finite_values.median().item(),
        "max": finite_values.max().item(),
        "numel": values.numel(),
    }


def _format_lambda_theta(lambda_stats):
    """Format a `lambda_theta` summary for one-line tqdm output."""
    kind = lambda_stats["kind"]
    if kind == "missing":
        return "missing"
    if kind == "empty":
        return "empty"
    if kind == "scalar":
        return f"{lambda_stats['value']:.4f}"
    if kind == "nonfinite":
        return f"nonfinite(numel={lambda_stats['numel']})"
    return (
        f"[{lambda_stats['min']:.4f}, {lambda_stats['max']:.4f}], "
        f"median={lambda_stats['median']:.4f}"
    )


def format_theta_support_batch(theta_stats, epoch, iteration_count, total_batches):
    """Format current-batch Theta support changes for tqdm-safe training output."""
    prefix = f"Theta support delta [epoch {epoch + 1}, batch {iteration_count}/{total_batches}]"
    if not theta_stats:
        return f"{prefix}: no ADMM blocks"

    parts = []
    for stat in theta_stats:
        block_name = f"block_{stat['block_idx']}"
        lambda_text = _format_lambda_theta(stat["lambda_theta"])
        if stat["ratio"] is None:
            parts.append(f"{block_name}=skipped, lambda_theta={lambda_text}")
            continue
        support_delta = stat.get("support_delta")
        if support_delta is not None and support_delta.get("kind") in {"local", "dense"}:
            parts.append(
                f"{block_name}=+{support_delta['added_per_node']:.2f}/node, "
                f"-{support_delta['removed_per_node']:.2f}/node "
                f"(learned={support_delta['learned_edges_per_node']:.2f}/node, "
                f"base={support_delta['baseline_edges_per_node']:.2f}/node, "
                f"kind={support_delta['kind']}, shape={stat['shape']}), "
                f"lambda_theta={lambda_text}"
            )
            continue
        parts.append(
            f"{block_name}=support_delta_unavailable "
            f"({stat['nnz']}/{stat['numel']}, shape={stat['shape']}), "
            f"lambda_theta={lambda_text}"
        )
    return f"{prefix}: " + "; ".join(parts)


def _log_nested_config(logger, config):
    """Write the effective config after CLI overrides are applied."""
    def _log_mapping(prefix, mapping):
        for key, value in mapping.items():
            if str(key).startswith("_"):
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                logger.info("\t%s:", path)
                _log_mapping(path, value)
            else:
                logger.info("\t%s: %s", path, value)

    for key, value in config.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            logger.info("\t%s:", key)
            _log_mapping(str(key), value)
        else:
            logger.info("\t%s: %s", key, value)


def _log_cli_arguments(logger, args):
    """Write runtime args and only the CLI overrides that were actually used."""
    config_override_args = {
        "neighbors",
        "theta_neighbors",
        "interval",
        "sharedM",
        "sharedQ",
        "diff_interval",
        "use_stable_graph_learning",
        "blocks",
        "layers",
        "CGiters",
        "tout",
        "le_emb",
        "deflation",
        "deflation_samples",
        "deflation_CGiters",
        "deflation_tol",
        "theta_method",
        "glasso_backend",
        "glasso_alpha",
        "glasso_rho",
        "glasso_eps",
        "glasso_eigh_shift",
        "glasso_eigh_shift_retries",
        "glasso_fallback",
        "kalofolias_graph",
        "kalofolias_alpha",
        "kalofolias_beta",
        "kalofolias_max_iter",
        "kalofolias_tol",
        "kalofolias_threshold",
        "kalofolias_output_mode",
        "kalofolias_normalize_distances",
        "stride",
        "lr",
    }
    logger.info("RUNTIME ARGUMENTS AND CLI OVERRIDES:")
    for arg, value in vars(args).items():
        if arg in config_override_args and value is None:
            continue
        logger.info("\t%s: %s", arg, value)


def _log_effective_model_flags(logger, config, model):
    """Log key effective flags, including what reached GraphLearningModule."""
    model_config = get_model_config(config)
    logger.info(
        "Effective graph flags from config: sharedM=%s, sharedQ=%s, diff_interval=%s, use_stable_graph_learning=%s",
        model_config["sharedM"],
        model_config["sharedQ"],
        model_config["diff_interval"],
        model_config["use_stable_graph_learning"],
    )
    logger.info(
        "Effective Theta graph settings: theta_method=%s, glasso_backend=%s, kalofolias_graph=%s, kNN=%s, theta_local_kNN=%s, kalofolias_output_mode=%s",
        model_config["theta_method"],
        model_config["glasso_backend"],
        model_config["kalofolias_graph"],
        model_config["kNN"],
        model_config["theta_kNN"],
        model_config["kalofolias_output_mode"],
    )
    if len(model.model_blocks) == 0:
        return

    graph_learn = model.model_blocks[0]["graph_learning_module"]
    logger.info(
        "GraphLearningModule flags in block_0: sharedM=%s, sharedQ=%s, diff_interval=%s, directed_time=%s, use_stable_graph_learning=%s",
        graph_learn.sharedM,
        graph_learn.sharedQ,
        graph_learn.diff_interval,
        graph_learn.directed_time,
        graph_learn.use_stable_graph_learning,
    )


def log_run_header(logger, args, config, model, train_set, signal_channels, admm_info, model_pretrained_path=None):
    """Write command, arguments, config, and model summary to the main log."""
    logger.info("#################################################")
    logger.info("Training CMD:\t" + " ".join(sys.argv))
    _log_cli_arguments(logger, args)

    logger.info("EFFECTIVE CONFIG SETTINGS:")
    _log_nested_config(logger, config)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_config = get_model_config(config)
    logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    logger.info("MODEL SUMMARY:")
    logger.info("pretrained path: %s", model_pretrained_path)
    logger.info("feature channels: %d", model_config["feature_channels"])
    logger.info("Total parameters: %d", total_params)
    logger.info("ADMM blocks: %d", model_config["num_blocks"])
    logger.info("ADMM info: %s", admm_info)
    _log_effective_model_flags(logger, config, model)
    logger.info(
        "graph info: nodes %d, edges %d, signal channels %d",
        train_set.n_nodes,
        train_set.n_edges,
        signal_channels,
    )
    logger.info("--------BEGIN TRAINING PROCESS------------")
