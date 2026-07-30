import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from kalofolias_graph_learning import learn_graph_from_smooth_signals
from kalofolias_graph_learning.benchmark_kalofolias import graph_metrics, make_problem


def parse_int_list(value: str) -> List[int]:
    """Parse comma-separated positive integers."""
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("list must contain at least one value")
    if any(item <= 1 for item in values):
        raise ValueError("all sample sizes must be greater than one")
    return values


def print_table(rows: List[Dict[str, float]]) -> None:
    """Print markdown table for sample-size recovery diagnostics."""
    headers = [
        "n_samples",
        "nnz",
        "nnz_ratio",
        "precision",
        "recall",
        "f1",
        "rel_fro",
        "rel_spec",
        "n_iter",
        "time_ms",
        "converged",
        "rel_change",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(
            f"{int(row['n_samples'])} | "
            f"{int(row['nnz'])} | "
            f"{row['nnz_ratio']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{row['rel_fro']:.6g} | "
            f"{row['rel_spec']:.6g} | "
            f"{int(row['n_iter'])} | "
            f"{row['time_ms']:.3f} | "
            f"{bool(row['converged'])} | "
            f"{row['rel_change']:.6g}"
        )


def plot_rows(rows: List[Dict[str, float]], true_nnz: int, output: Path) -> None:
    """Plot nnz and recovery quality versus number of samples."""
    import os

    cache = Path(f"/tmp/matplotlib-cache-{os.getuid()}")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    samples = np.asarray([row["n_samples"] for row in rows], dtype=np.float64)
    nnz = np.asarray([row["nnz"] for row in rows], dtype=np.float64)
    precision = np.asarray([row["precision"] for row in rows], dtype=np.float64)
    recall = np.asarray([row["recall"] for row in rows], dtype=np.float64)
    f1 = np.asarray([row["f1"] for row in rows], dtype=np.float64)
    rel_fro = np.asarray([row["rel_fro"] for row in rows], dtype=np.float64)
    rel_spec = np.asarray([row["rel_spec"] for row in rows], dtype=np.float64)
    n_iter = np.asarray([row["n_iter"] for row in rows], dtype=np.float64)
    time_ms = np.asarray([row["time_ms"] for row in rows], dtype=np.float64)
    converged = np.asarray([row["converged"] for row in rows], dtype=bool)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(samples, nnz, marker="o")
    axes[0].axhline(true_nnz, color="black", linestyle="--", linewidth=1.0, label="true nnz")
    axes[0].set_xlabel("number of samples")
    axes[0].set_ylabel("recovered nnz")
    axes[0].set_title("Recovered graph size")
    axes[0].grid(True, linewidth=0.4, alpha=0.35)
    axes[0].legend()

    axes[1].plot(samples, precision, marker="o", label="precision")
    axes[1].plot(samples, recall, marker="o", label="recall")
    axes[1].plot(samples, f1, marker="o", label="F1")
    axes[1].set_xlabel("number of samples")
    axes[1].set_ylabel("support metric")
    axes[1].set_title("Support recovery")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, linewidth=0.4, alpha=0.35)
    axes[1].legend()

    axes[2].plot(samples, rel_fro, marker="o", label="relative Frobenius")
    axes[2].plot(samples, rel_spec, marker="o", label="relative spectral")
    axes[2].set_xlabel("number of samples")
    axes[2].set_ylabel("relative error")
    axes[2].set_title("Weighted graph error")
    axes[2].grid(True, linewidth=0.4, alpha=0.35)
    axes[2].legend()

    axes[3].plot(samples[converged], n_iter[converged], marker="o", linestyle="-", label="iterations")
    if not np.all(converged):
        axes[3].scatter(samples[~converged], n_iter[~converged], marker="x", color="tab:red", label="hit max_iter")
    axes[3].set_xlabel("number of samples")
    axes[3].set_ylabel("iterations")
    axes[3].set_title("Convergence cost")
    axes[3].grid(True, linewidth=0.4, alpha=0.35)
    time_axis = axes[3].twinx()
    time_axis.plot(samples, time_ms, marker="s", color="tab:orange", linestyle="--", label="runtime")
    time_axis.set_ylabel("runtime (ms)")
    lines, labels = axes[3].get_legend_handles_labels()
    time_lines, time_labels = time_axis.get_legend_handles_labels()
    axes[3].legend(lines + time_lines, labels + time_labels)

    fig.suptitle("Kalofolias sample-size recovery sweep")
    fig.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep sample size for Kalofolias graph recovery.")
    parser.add_argument("--sample-sizes", type=str, default="20,40,60,80,100,120,160,200")
    parser.add_argument("--n-features", type=int, default=40)
    parser.add_argument("--true-nnz", type=int, default=160)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-weight-min", type=float, default=0.15)
    parser.add_argument("--edge-weight-max", type=float, default=0.35)
    parser.add_argument("--diagonal-shift", type=float, default=0.1)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--output", type=Path, default=Path("kalofolias_graph_learning/figures/sample_size_sweep.png"))
    args = parser.parse_args()

    sample_sizes = parse_int_list(args.sample_sizes)
    dtype = np.float64 if args.dtype == "float64" else np.float32
    rows: List[Dict[str, float]] = []
    true_nnz_seen = None
    for n_samples in sample_sizes:
        signals, true_adjacency, true_support = make_problem(
            n_samples=n_samples,
            n_features=args.n_features,
            seed=args.seed,
            dtype=dtype,
            true_nnz=args.true_nnz,
            edge_weight_min=args.edge_weight_min,
            edge_weight_max=args.edge_weight_max,
            diagonal_shift=args.diagonal_shift,
        )
        true_nnz = int(np.triu(true_support, k=1).sum())
        true_nnz_seen = true_nnz if true_nnz_seen is None else true_nnz_seen
        if true_nnz != true_nnz_seen:
            raise RuntimeError("true graph changed across sample sizes")
        start = time.perf_counter()
        result = learn_graph_from_smooth_signals(
            signals,
            alpha=args.alpha,
            beta=args.beta,
            max_iter=args.max_iter,
            tol=args.tol,
            threshold=args.threshold,
            return_info=True,
        )
        time_ms = 1000.0 * (time.perf_counter() - start)
        metrics = graph_metrics(result.adjacency, true_adjacency, true_support, args.threshold)
        rows.append(
            {
                "n_samples": float(n_samples),
                "n_iter": float(result.n_iter),
                "time_ms": float(time_ms),
                "converged": float(result.converged),
                "rel_change": result.relative_change,
                **metrics,
            }
        )

    print(
        "Kalofolias sample-size sweep: "
        f"sample_sizes={sample_sizes}, n_features={args.n_features}, "
        f"requested_true_nnz={args.true_nnz}, true_nnz={true_nnz_seen}, "
        f"alpha={args.alpha}, beta={args.beta}, max_iter={args.max_iter}, "
        f"threshold={args.threshold}, dtype={args.dtype}"
    )
    print_table(rows)
    plot_rows(rows, int(true_nnz_seen), args.output)
    print(f"figure={args.output}")


if __name__ == "__main__":
    main()
