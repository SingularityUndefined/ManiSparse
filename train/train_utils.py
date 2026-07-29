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

    parser.add_argument("--neighbors", help="temporary override for model.kNN", default=None, type=int)
    parser.add_argument("--interval", help="temporary override for model.interval", default=None, type=int)
    parser.add_argument("--FElayers", help="feature extractor layers", default=1, type=int)
    parser.add_argument("--ablation", help="operator to eliminate in ablation study", default="None", type=str)

    parser.add_argument("--seed", help="random seed", default=3407, type=int)
    parser.add_argument("--debug", dest="debug", help="save debug model every iteration", action="store_true")
    parser.set_defaults(debug=False)

    parser.add_argument("--stepLR", dest="use_stepLR", action="store_true")
    parser.set_defaults(use_stepLR=False)
    parser.add_argument("--stepsize", help="stepLR stepsize", default=8, type=int)
    parser.add_argument("--gamma", help="stepLR gamma", default=0.2, type=float)

    _add_bool_override(parser, "sharedM", "sharedM", "temporary override for model.sharedM")
    _add_bool_override(parser, "sharedQ", "sharedQ", "temporary override for model.sharedQ")
    _add_bool_override(parser, "diff-interval", "diff_interval", "temporary override for model.diff_interval")
    _add_bool_override(
        parser,
        "stable-graph-learning",
        "use_stable_graph_learning",
        "temporary override for model.use_stable_graph_learning",
    )

    parser.add_argument("--epochs", help="running epochs", default=70, type=int)
    parser.add_argument("--start_epochs", help="start epochs", default=0, type=int)
    parser.add_argument("--loggrad", help="log gradient norms; -1 disables it", default=-1, type=int)

    parser.add_argument("--tout", help="temporary override for model.t_out", default=None, type=int)
    parser.add_argument("--trunc", dest="trunc", action="store_true")
    parser.set_defaults(trunc=False)
    parser.add_argument("--le-emb", help="learnable embedding", dest="le_emb", action="store_true")
    parser.set_defaults(le_emb=False)
    parser.add_argument("--blocks", help="temporary override for model.num_blocks", default=None, type=int)
    parser.add_argument("--layers", help="temporary override for model.num_layers", default=None, type=int)
    parser.add_argument("--CGiters", help="temporary override for model.CG_iters", default=None, type=int)
    _add_bool_override(parser, "deflation", "deflation", "temporary override for model.use_deflation")
    parser.add_argument("--deflation-samples", help="temporary override for model.deflation_samples", default=None, type=int)
    parser.add_argument(
        "--deflation-CGiters",
        help="temporary override for model.deflation_CG_iters",
        default=None,
        type=int,
    )
    parser.add_argument("--deflation-tol", help="temporary override for model.deflation_tol", default=None, type=float)
    parser.add_argument("--glasso-method", help="temporary override for model.glasso_method", default=None, type=str)
    parser.add_argument("--glasso-alpha", help="temporary override for model.glasso_alpha", default=None, type=float)
    parser.add_argument("--glasso-rho", help="temporary override for model.glasso_rho", default=None, type=float)
    parser.add_argument("--glasso-eps", help="temporary override for model.glasso_eps", default=None, type=float)
    parser.add_argument("--glasso-eigh-shift", help="temporary override for model.glasso_eigh_shift", default=None, type=float)
    parser.add_argument("--glasso-eigh-shift-retries", help="temporary override for model.glasso_eigh_shift_retries", default=None, type=int)
    _add_bool_override(parser, "glasso-fallback", "glasso_fallback", "temporary override for model.glasso_fallback")
    parser.add_argument("--stride", help="temporary override for data_stride", default=None, type=int)
    parser.add_argument("--lr", help="temporary override for learning_rate", default=None, type=float)
    parser.add_argument("--predonly", dest="pred_only", action="store_true")
    parser.set_defaults(pred_only=False)

    args = parser.parse_args(argv)
    return args, config


def apply_args_to_config(config, args):
    """Apply CLI overrides to the loaded config in-place."""
    model_config = config["model"]
    arg_to_config_key = {
        "neighbors": "kNN",
        "interval": "interval",
        "sharedM": "sharedM",
        "sharedQ": "sharedQ",
        "diff_interval": "diff_interval",
        "use_stable_graph_learning": "use_stable_graph_learning",
        "blocks": "num_blocks",
        "layers": "num_layers",
        "CGiters": "CG_iters",
        "tout": "t_out",
        "deflation_samples": "deflation_samples",
        "deflation_CGiters": "deflation_CG_iters",
        "deflation_tol": "deflation_tol",
        "deflation": "use_deflation",
        "glasso_method": "glasso_method",
        "glasso_alpha": "glasso_alpha",
        "glasso_rho": "glasso_rho",
        "glasso_eps": "glasso_eps",
        "glasso_eigh_shift": "glasso_eigh_shift",
        "glasso_eigh_shift_retries": "glasso_eigh_shift_retries",
        "glasso_fallback": "glasso_fallback",
    }
    for arg_name, config_key in arg_to_config_key.items():
        value = getattr(args, arg_name)
        if value is not None:
            model_config[config_key] = value

    model_config.setdefault("use_deflation", False)
    model_config.setdefault("use_stable_graph_learning", False)
    model_config.setdefault("deflation_samples", 5)
    model_config.setdefault("deflation_CG_iters", model_config["CG_iters"])
    model_config.setdefault("deflation_tol", 1e-6)
    model_config.setdefault("glasso_method", "admm")
    model_config.setdefault("glasso_alpha", 0.2)
    model_config.setdefault("glasso_rho", 1.0)
    model_config.setdefault("glasso_eps", 0.0)
    model_config.setdefault("glasso_eigh_shift", 1e-6)
    model_config.setdefault("glasso_eigh_shift_retries", 4)
    model_config.setdefault("glasso_fallback", True)

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


def build_experiment_names(config, args):
    """Create log/model directory names without touching the filesystem."""
    logs_dir = "logs_learnable_emb" if args.le_emb else "dense_logs_new"
    learning_rate = config["learning_rate"]
    experiment_dir = f"lr_{learning_rate:.0e}_seed_{args.seed}"

    dataset_name = args.dataset
    num_blocks = config["model"]["num_blocks"]
    num_layers = config["model"]["num_layers"]
    num_heads = config["model"]["num_heads"]
    interval = config["model"]["interval"]
    feature_channels = config["model"]["feature_channels"]
    loss_name = config["loss_function"]

    name = f"{dataset_name}_s{config['data_stride']}_{num_blocks}b{num_layers}_{num_heads}h_{feature_channels}f_{args.FElayers}FE"
    if args.pred_only:
        name = "predOnly_" + name
    if config["model"].get("use_deflation", False):
        name = f"deflate{config['model']['deflation_samples']}_" + name
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
            use_one_channel=config["model"]["use_one_channel"],
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
            use_one_channel=config["model"]["use_one_channel"],
        )

    train_set = datasets_and_loaders[0]
    data_normalization = Normalization(train_set, args.mode, device)
    return datasets_and_loaders, data_normalization


def build_admm_info(config):
    """Collect ADMM hyperparameters in the format expected by `UnrollingModel`."""
    model_config = config["model"]
    return {
        "ADMM_iters": model_config["num_layers"],
        "CG_iters": model_config["CG_iters"],
        "PGD_iters": model_config["PGD_iters"],
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
        use_stable_graph_learning=config["model"]["use_stable_graph_learning"],
        predict_only=args.pred_only,
        le_emb=args.le_emb,
        use_deflation=config["model"].get("use_deflation", False),
        deflation_samples=config["model"]["deflation_samples"],
        glasso_method=config["model"]["glasso_method"],
        glasso_alpha=config["model"]["glasso_alpha"],
        glasso_rho=config["model"]["glasso_rho"],
        glasso_eps=config["model"]["glasso_eps"],
        glasso_eigh_shift=config["model"]["glasso_eigh_shift"],
        glasso_eigh_shift_retries=config["model"]["glasso_eigh_shift_retries"],
        glasso_fallback=config["model"]["glasso_fallback"],
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


def collect_theta_nnz_ratios(model):
    """Return per-block Theta sparsity ratios for the current forward state.

    Theta is estimated inside `UnrollingModel.forward`, so these values describe
    the most recent batch that passed through the model.  A `None` Theta means
    the branch was skipped, for example when `ablation="Theta"`.
    """
    stats = []
    for block_idx, block in enumerate(model.model_blocks):
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
                    "lambda_theta": lambda_theta,
                }
            )
            continue

        theta = torch.as_tensor(theta).detach()
        nnz = torch.count_nonzero(theta).item()
        numel = theta.numel()
        stats.append(
            {
                "block_idx": block_idx,
                "ratio": nnz / numel if numel > 0 else float("nan"),
                "nnz": nnz,
                "numel": numel,
                "shape": tuple(theta.shape),
                "lambda_theta": lambda_theta,
            }
        )
    return stats


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
        return f"{lambda_stats['value']:.3e}"
    if kind == "nonfinite":
        return f"nonfinite(numel={lambda_stats['numel']})"
    return (
        f"min={lambda_stats['min']:.3e}, "
        f"median={lambda_stats['median']:.3e}, "
        f"max={lambda_stats['max']:.3e}"
    )


def format_theta_nnz_batch(theta_stats, epoch, iteration_count, total_batches):
    """Format current-batch per-block Theta sparsity for tqdm-safe training output."""
    prefix = f"Theta nnz ratios [epoch {epoch + 1}, batch {iteration_count}/{total_batches}]"
    if not theta_stats:
        return f"{prefix}: no ADMM blocks"

    parts = []
    for stat in theta_stats:
        block_name = f"block_{stat['block_idx']}"
        lambda_text = _format_lambda_theta(stat["lambda_theta"])
        if stat["ratio"] is None:
            parts.append(f"{block_name}=skipped, lambda_theta={lambda_text}")
            continue
        parts.append(
            f"{block_name}={stat['ratio']:.3e} "
            f"({stat['nnz']}/{stat['numel']}, shape={stat['shape']}), "
            f"lambda_theta={lambda_text}"
        )
    return f"{prefix}: " + "; ".join(parts)


def _log_nested_config(logger, config):
    """Write the effective config after CLI overrides are applied."""
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info("\t%s:", key)
            for k, v in value.items():
                logger.info("\t\t%s: %s", k, v)
        else:
            logger.info("\t%s: %s", key, value)


def _log_cli_arguments(logger, args):
    """Write runtime args and only the CLI overrides that were actually used."""
    config_override_args = {
        "neighbors",
        "interval",
        "sharedM",
        "sharedQ",
        "diff_interval",
        "use_stable_graph_learning",
        "blocks",
        "layers",
        "CGiters",
        "tout",
        "deflation",
        "deflation_samples",
        "deflation_CGiters",
        "deflation_tol",
        "glasso_method",
        "glasso_alpha",
        "glasso_rho",
        "glasso_eps",
        "glasso_eigh_shift",
        "glasso_eigh_shift_retries",
        "glasso_fallback",
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
    model_config = config["model"]
    logger.info(
        "Effective graph flags from config: sharedM=%s, sharedQ=%s, diff_interval=%s, use_stable_graph_learning=%s",
        model_config["sharedM"],
        model_config["sharedQ"],
        model_config["diff_interval"],
        model_config["use_stable_graph_learning"],
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
    logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    logger.info("MODEL SUMMARY:")
    logger.info("pretrained path: %s", model_pretrained_path)
    logger.info("feature channels: %d", config["model"]["feature_channels"])
    logger.info("Total parameters: %d", total_params)
    logger.info("ADMM blocks: %d", config["model"]["num_blocks"])
    logger.info("ADMM info: %s", admm_info)
    _log_effective_model_flags(logger, config, model)
    logger.info(
        "graph info: nodes %d, edges %d, signal channels %d",
        train_set.n_nodes,
        train_set.n_edges,
        signal_channels,
    )
    logger.info("--------BEGIN TRAINING PROCESS------------")
