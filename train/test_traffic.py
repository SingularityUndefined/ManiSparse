"""Evaluate a traffic checkpoint on the test split.

Run from the repository root with ``python -m train.test_traffic --help``.
Use the same dataset and model options that were used during training.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch

from train.train_utils import (
    apply_args_to_config,
    build_loss_fn,
    create_data,
    create_model,
    get_model_config,
    parse_args,
    prepare_runtime,
)
from utils import test as evaluate_batches


SIGNAL_NAMES = ("flow", "occupancy", "speed")


def load_checkpoint(model, path, device):
    """Strictly restore model weights from either checkpoint format."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        # Full training-state files include NumPy RNG objects, which cannot be
        # loaded by weights_only. Only load checkpoint files you trust.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint_path} does not contain a model state_dict")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint weights do not match the model. Use the same --dataset, "
            "--config, --ablation and model options as in training."
        ) from error
    return checkpoint_path


def main(argv=None):
    args, config = parse_args(argv, testing=True)
    config = apply_args_to_config(config, args)
    device = prepare_runtime(args)
    datasets_and_loaders, data_normalization = create_data(config, args, device)
    train_set, _, test_set, _, _, test_loader = datasets_and_loaders
    if len(test_set) == 0:
        raise ValueError("The test split is empty")

    model, _ = create_model(config, args, train_set, device)
    checkpoint_path = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    loss_fn = build_loss_fn(config)
    use_one_channel = get_model_config(config)["use_one_channel"]
    result = evaluate_batches(
        model, test_loader, data_normalization, False, config, device,
        train_set.signal_channel, mode="val", loss_fn=loss_fn,
        use_one_channel=use_one_channel,
    )
    if use_one_channel:
        loss, metrics = result
        channel_metrics = None
    else:
        loss, metrics, channel_metrics = result

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {args.dataset} | test samples: {len(test_set)} | device: {device}")
    print(f"Test loss ({config['loss_function']}): {loss:.4f}")
    print(
        "ALL: rec_RMSE={rec_RMSE:.4f}, pred_RMSE={pred_RMSE:.4f}, "
        "pred_MAE={pred_MAE:.4f}, pred_MAPE={pred_MAPE:.4f}%, "
        "rNMSE={rNMSE:.4f}".format(**metrics)
    )
    if channel_metrics is not None:
        for channel in range(train_set.signal_channel):
            name = SIGNAL_NAMES[channel] if channel < len(SIGNAL_NAMES) else f"channel_{channel}"
            print(
                f"{name}: rec_RMSE={channel_metrics['rec_RMSE'][channel]:.4f}, "
                f"pred_RMSE={channel_metrics['pred_RMSE'][channel]:.4f}, "
                f"pred_MAE={channel_metrics['pred_MAE'][channel]:.4f}, "
                f"pred_MAPE={channel_metrics['pred_MAPE'][channel]:.4f}%"
            )


if __name__ == "__main__":
    main()
