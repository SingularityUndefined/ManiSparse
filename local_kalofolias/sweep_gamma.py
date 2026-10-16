"""Compare Local Kalofolias convergence for fixed alpha/beta and varying gamma."""

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from local_kalofolias import learn_local_graph_from_smooth_signals
from local_kalofolias.benchmark_local_kalofolias import (
    checkpoint_sequence,
    local_metrics,
    make_problem,
)


ALPHA = 0.3
BETA = 1.0
STEP_SAFETY = 0.9


@dataclass(frozen=True)
class GammaCase:
    """One constant or per-iteration gamma experiment."""

    label: str
    values: Union[float, np.ndarray]


def default_gamma(k_neighbors: int) -> float:
    """Return the solver's automatic step size for the fixed beta."""
    return STEP_SAFETY / (2.0 * BETA + math.sqrt(k_neighbors) + 1.0)


def parse_gammas(spec: str, automatic_gamma: float) -> List[float]:
    """Parse comma-separated positive step sizes, accepting ``default``."""
    values: List[float] = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        value = automatic_gamma if item in {"auto", "default"} else float(item)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("all gamma values must be finite and positive")
        if not any(math.isclose(value, existing, rel_tol=1e-12, abs_tol=1e-15) for existing in values):
            values.append(value)
    if not values:
        raise ValueError("at least one gamma value is required")
    return values


def parse_gamma_schedules(spec: str, max_iter: int) -> List[GammaCase]:
    """Parse ``name:start:end`` linear gamma schedules."""
    schedules: List[GammaCase] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError("gamma schedules must use name:start:end syntax")
        name, start_text, end_text = parts
        start = float(start_text)
        end = float(end_text)
        if not name or not math.isfinite(start) or not math.isfinite(end) or start <= 0 or end <= 0:
            raise ValueError("gamma schedule names must be non-empty and endpoints finite and positive")
        schedules.append(GammaCase(f"{name} {start:g}->{end:g}", np.linspace(start, end, max_iter)))
    return schedules


def sustained_convergence_iteration(history: List[float], tolerance: float) -> Optional[int]:
    """Return the first iteration after which relative change stays below tolerance."""
    suffix_max = -math.inf
    first: Optional[int] = None
    for index in range(len(history) - 1, -1, -1):
        value = history[index]
        if not math.isfinite(value):
            suffix_max = math.inf
        else:
            suffix_max = max(suffix_max, value)
        if suffix_max < tolerance:
            first = index + 1
    return first


def gamma_label(gamma: float, automatic_gamma: float) -> str:
    """Format a legend label and mark the automatic solver choice."""
    label = f"gamma={gamma:.4g}"
    if math.isclose(gamma, automatic_gamma, rel_tol=1e-9, abs_tol=1e-12):
        label += " (default)"
    return label


def recovery_metrics(weights: torch.Tensor, true_local: torch.Tensor, threshold: float) -> Dict[str, float]:
    """Do not report support or weight recovery for non-finite iterates."""
    if not torch.isfinite(weights).all():
        return {key: math.nan for key in ("nnz", "nnz_ratio", "f1", "rel_fro")}
    return local_metrics(weights, true_local, threshold)


def print_summary(rows: List[Dict[str, object]], convergence_tol: float) -> None:
    """Print one final convergence/recovery row per gamma."""
    headers = ["gamma", "converged", f"iter_to_{convergence_tol:g}", "final_rel_change", "nnz", "f1", "rel_fro"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        iteration = "-" if row["iter_to_tol"] < 0 else str(int(row["iter_to_tol"]))
        print(
            f"{row['label']} | {'yes' if row['iter_to_tol'] >= 0 else 'no'} | "
            f"{iteration} | {row['rel_change']:.6g} | "
            f"{int(row['nnz']) if math.isfinite(row['nnz']) else '-'} | "
            f"{row['f1']:.4f} | {row['rel_fro']:.6g}"
        )


def plot_sweep(
    traces: Dict[str, List[Dict[str, float]]],
    output: Path,
    true_nnz_ratio: float,
) -> None:
    """Plot convergence and recovery paths in the benchmark figure style."""
    cache = Path(f"/tmp/matplotlib-cache-{os.getuid()}")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for label, rows in traces.items():
        iterations = np.asarray([row["iter"] for row in rows])
        axes[0, 0].semilogx(iterations, 100.0 * np.asarray([row["nnz_ratio"] for row in rows]), marker="o", label=label)
        axes[0, 1].semilogx(iterations, [row["f1"] for row in rows], marker="o", label=label)
        axes[1, 0].semilogx(iterations, [row["rel_fro"] for row in rows], marker="o", label=label)
        axes[1, 1].loglog(iterations, np.maximum([row["rel_change"] for row in rows], 1e-16), marker="o", label=label)

    axes[0, 0].axhline(100.0 * true_nnz_ratio, color="black", linestyle="--", linewidth=1.0, label="true slots")
    axes[0, 0].set_ylabel("local nnz ratio (%)")
    axes[0, 0].set_title("Candidate-slot sparsity")
    axes[0, 1].set_ylabel("support F1")
    axes[0, 1].set_title("Local support recovery")
    axes[1, 0].set_ylabel("relative local-weight error")
    axes[1, 0].set_title("Weight recovery")
    axes[1, 1].set_ylabel("relative change")
    axes[1, 1].set_title("Solver convergence")

    for axis in axes.flat:
        axis.set_xlabel("iteration")
        axis.grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"Local Kalofolias gamma sweep (alpha={ALPHA}, beta={BETA})")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Local Kalofolias gamma values with alpha=0.3 and beta=1.0 fixed."
    )
    parser.add_argument("--gammas", type=str, default="0.025,0.05,0.1,default,0.2,0.35,0.4,0.5,0.6")
    parser.add_argument(
        "--gamma-schedules",
        type=str,
        default="increasing:0.05:0.35,decreasing:0.35:0.05",
        help="comma-separated name:start:end linear schedules; use an empty string to disable",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-nodes", type=int, default=40)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--active-neighbors-per-node", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--convergence-tol", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="1,2,5,10,20,30,40,50,75,100,150,200,300,500,1000",
    )
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-weight-min", type=float, default=0.15)
    parser.add_argument("--edge-weight-max", type=float, default=0.35)
    parser.add_argument("--diagonal-shift", type=float, default=0.1)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--output", type=Path, default=Path("local_kalofolias/figures/gamma_convergence_sweep.png"))
    args = parser.parse_args()

    if args.max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if args.convergence_tol <= 0:
        raise ValueError("convergence_tol must be positive")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    automatic_gamma = default_gamma(args.k_neighbors)
    gammas = parse_gammas(args.gammas, automatic_gamma)
    cases = [GammaCase(gamma_label(gamma, automatic_gamma), gamma) for gamma in gammas]
    cases.extend(parse_gamma_schedules(args.gamma_schedules, args.max_iter))
    checkpoints = checkpoint_sequence(args.max_iter, args.checkpoints)
    if checkpoints[-1] != args.max_iter:
        checkpoints.append(args.max_iter)
    signals, neighbor_list, true_local, _ = make_problem(
        n_samples=args.n_samples,
        n_nodes=args.n_nodes,
        k_neighbors=args.k_neighbors,
        active_neighbors_per_node=args.active_neighbors_per_node,
        seed=args.seed,
        dtype=dtype,
        edge_weight_min=args.edge_weight_min,
        edge_weight_max=args.edge_weight_max,
        diagonal_shift=args.diagonal_shift,
    )

    traces: Dict[str, List[Dict[str, float]]] = {}
    summary_rows: List[Dict[str, object]] = []
    for case in cases:
        trace: List[Dict[str, float]] = []
        for checkpoint in checkpoints:
            gamma = case.values[:checkpoint] if isinstance(case.values, np.ndarray) else case.values
            result = learn_local_graph_from_smooth_signals(
                signals,
                neighbor_list,
                alpha=ALPHA,
                beta=BETA,
                max_iter=checkpoint,
                tol=args.tol,
                gamma=gamma,
                threshold=args.threshold,
                early_stopping=False,
                return_info=True,
            )
            trace.append(
                {
                    "iter": float(checkpoint),
                    "rel_change": result.relative_change,
                    **recovery_metrics(result.local_weights, true_local, args.threshold),
                }
            )
        traces[case.label] = trace
        full_result = result
        convergence_iteration = sustained_convergence_iteration(
            full_result.relative_change_history, args.convergence_tol
        )
        summary_rows.append(
            {
                "label": case.label,
                "iter_to_tol": -1.0 if convergence_iteration is None else float(convergence_iteration),
                "rel_change": full_result.relative_change,
                **recovery_metrics(full_result.local_weights, true_local, args.threshold),
            }
        )

    true_nnz_ratio = float((true_local > 1e-12).sum() / true_local.numel())
    print(
        "Local Kalofolias gamma convergence sweep: "
        f"alpha={ALPHA}, beta={BETA}, automatic_gamma={automatic_gamma:.8g}, "
        f"cases={','.join(case.label for case in cases)}, max_iter={args.max_iter}, "
        f"convergence_tol={args.convergence_tol:g}, dtype={args.dtype}"
    )
    print_summary(summary_rows, args.convergence_tol)
    plot_sweep(traces, args.output, true_nnz_ratio)
    print(f"figure={args.output}")


if __name__ == "__main__":
    main()
