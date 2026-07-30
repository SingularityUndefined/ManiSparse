import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from kalofolias_graph_learning import learn_graph_from_smooth_signals
from kalofolias_graph_learning.benchmark_kalofolias import graph_metrics, make_problem


def parse_float_list(value: str) -> List[float]:
    """Parse comma-separated positive floats."""
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("list must contain at least one value")
    if any(item <= 0.0 for item in values):
        raise ValueError("all values must be positive")
    return values


def first_iteration_below(history: List[float], tol: float) -> int:
    """Return the first 1-based iteration whose diagnostic is below tol."""
    for idx, value in enumerate(history, start=1):
        if value <= tol:
            return idx
    return -1


def print_table(rows: List[Dict[str, float]]) -> None:
    """Print markdown table ordered by alpha then beta."""
    headers = [
        "alpha",
        "beta",
        "iter_to_tol",
        "final_rel_change",
        "nnz",
        "nnz_ratio",
        "precision",
        "recall",
        "f1",
        "rel_fro",
        "rel_spec",
        "objective",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        iter_to_tol = "-" if row["iter_to_tol"] < 0 else str(int(row["iter_to_tol"]))
        print(
            f"{row['alpha']:.6g} | "
            f"{row['beta']:.6g} | "
            f"{iter_to_tol} | "
            f"{row['final_rel_change']:.6g} | "
            f"{int(row['nnz'])} | "
            f"{row['nnz_ratio']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{row['rel_fro']:.6g} | "
            f"{row['rel_spec']:.6g} | "
            f"{row['objective']:.6g}"
        )


def heatmap(ax, matrix: np.ndarray, alphas: List[float], betas: List[float], title: str, cbar_label: str, fmt: str):
    """Draw a labeled alpha/beta heatmap."""
    image = ax.imshow(matrix, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(betas)))
    ax.set_yticks(np.arange(len(alphas)))
    ax.set_xticklabels([f"{value:g}" for value in betas])
    ax.set_yticklabels([f"{value:g}" for value in alphas])
    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    ax.set_title(title)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = "-" if not np.isfinite(value) else format(value, fmt)
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8)
    cbar = ax.figure.colorbar(image, ax=ax)
    cbar.ax.set_ylabel(cbar_label)


def plot_sweep(rows: List[Dict[str, float]], alphas: List[float], betas: List[float], output: Path) -> None:
    """Plot convergence speed and graph recovery quality over alpha/beta grid."""
    import os

    cache = Path(f"/tmp/matplotlib-cache-{os.getuid()}")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    shape = (len(alphas), len(betas))
    iter_matrix = np.full(shape, np.nan, dtype=np.float64)
    nnz_matrix = np.full(shape, np.nan, dtype=np.float64)
    f1_matrix = np.full(shape, np.nan, dtype=np.float64)
    rel_change_matrix = np.full(shape, np.nan, dtype=np.float64)
    for row in rows:
        alpha_idx = alphas.index(row["alpha"])
        beta_idx = betas.index(row["beta"])
        iter_matrix[alpha_idx, beta_idx] = row["iter_to_tol"] if row["iter_to_tol"] > 0 else np.nan
        nnz_matrix[alpha_idx, beta_idx] = 100.0 * row["nnz_ratio"]
        f1_matrix[alpha_idx, beta_idx] = row["f1"]
        rel_change_matrix[alpha_idx, beta_idx] = row["final_rel_change"]

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    heatmap(axes[0, 0], iter_matrix, alphas, betas, "Iterations to tolerance", "iteration", ".0f")
    heatmap(axes[0, 1], nnz_matrix, alphas, betas, "Final nnz ratio", "percent", ".1f")
    heatmap(axes[1, 0], f1_matrix, alphas, betas, "Final support F1", "F1", ".2f")
    heatmap(axes[1, 1], rel_change_matrix, alphas, betas, "Final relative change", "relative change", ".1e")
    fig.suptitle("Kalofolias alpha/beta convergence sweep")
    fig.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Kalofolias alpha/beta and compare convergence speed.")
    parser.add_argument("--alphas", type=str, default="0.003,0.01,0.03,0.1")
    parser.add_argument("--betas", type=str, default="1,3,10,30")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--true-nnz", type=int, default=160, help="number of undirected nonzero graph edges")
    parser.add_argument("--edge-weight-min", type=float, default=0.15)
    parser.add_argument("--edge-weight-max", type=float, default=0.35)
    parser.add_argument("--diagonal-shift", type=float, default=0.1)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--output", type=Path, default=Path("kalofolias_graph_learning/figures/alpha_beta_sweep.png"))
    args = parser.parse_args()

    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)
    dtype = np.float64 if args.dtype == "float64" else np.float32
    signals, true_adjacency, true_support = make_problem(
        n_samples=args.n_samples,
        n_features=args.n_features,
        seed=args.seed,
        dtype=dtype,
        true_nnz=args.true_nnz,
        edge_weight_min=args.edge_weight_min,
        edge_weight_max=args.edge_weight_max,
        diagonal_shift=args.diagonal_shift,
    )
    true_nnz = int(np.triu(true_support, k=1).sum())
    rows: List[Dict[str, float]] = []
    for alpha in alphas:
        for beta in betas:
            result = learn_graph_from_smooth_signals(
                signals,
                alpha=alpha,
                beta=beta,
                max_iter=args.max_iter,
                tol=args.tol,
                threshold=args.threshold,
                return_info=True,
            )
            metrics = graph_metrics(result.adjacency, true_adjacency, true_support, args.threshold)
            rows.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "iter_to_tol": float(first_iteration_below(result.relative_change_history, args.tol)),
                    "final_rel_change": result.relative_change,
                    "objective": result.objective,
                    **metrics,
                }
            )

    print(
        "Kalofolias alpha/beta sweep: "
        f"n_samples={args.n_samples}, n_features={args.n_features}, "
        f"requested_true_nnz={args.true_nnz}, true_nnz={true_nnz}, "
        f"max_iter={args.max_iter}, tol={args.tol}, threshold={args.threshold}, dtype={args.dtype}"
    )
    print_table(rows)
    best_f1 = max(rows, key=lambda row: row["f1"])
    fastest = min((row for row in rows if row["iter_to_tol"] > 0), key=lambda row: row["iter_to_tol"], default=None)
    print(
        "best_f1: "
        f"alpha={best_f1['alpha']:.6g}, beta={best_f1['beta']:.6g}, "
        f"f1={best_f1['f1']:.4f}, nnz={int(best_f1['nnz'])}, "
        f"iter_to_tol={best_f1['iter_to_tol'] if best_f1['iter_to_tol'] > 0 else '-'}"
    )
    if fastest is not None:
        print(
            "fastest_to_tol: "
            f"alpha={fastest['alpha']:.6g}, beta={fastest['beta']:.6g}, "
            f"iter_to_tol={int(fastest['iter_to_tol'])}, f1={fastest['f1']:.4f}, "
            f"nnz={int(fastest['nnz'])}"
        )
    else:
        print("fastest_to_tol: none reached tolerance")
    plot_sweep(rows, alphas, betas, args.output)
    print(f"figure={args.output}")


if __name__ == "__main__":
    main()
