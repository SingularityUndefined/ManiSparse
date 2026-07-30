import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from kalofolias_graph_learning import learn_graph_from_smooth_signals


def make_sparse_adjacency(
    n_features: int,
    true_nnz: int,
    seed: int,
    dtype: np.dtype,
    edge_weight_min: float,
    edge_weight_max: float,
) -> np.ndarray:
    """Create a symmetric nonnegative graph with exactly ``true_nnz`` edges."""
    possible_edges = n_features * (n_features - 1) // 2
    if true_nnz <= 0 or true_nnz > possible_edges:
        raise ValueError(f"true_nnz must be in [1, {possible_edges}] for n_features={n_features}")
    rng = np.random.default_rng(seed)
    src_all, dst_all = np.triu_indices(n_features, k=1)
    chosen = rng.choice(possible_edges, size=true_nnz, replace=False)
    src = src_all[chosen]
    dst = dst_all[chosen]
    weights = rng.uniform(edge_weight_min, edge_weight_max, size=true_nnz).astype(dtype, copy=False)
    adjacency = np.zeros((n_features, n_features), dtype=dtype)
    adjacency[src, dst] = weights
    adjacency[dst, src] = weights
    return adjacency


def graph_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Return combinatorial Laplacian ``L = diag(W 1) - W``."""
    return np.diag(adjacency.sum(axis=1)) - adjacency


def make_problem(
    n_samples: int,
    n_features: int,
    seed: int,
    dtype: np.dtype,
    true_nnz: int,
    edge_weight_min: float,
    edge_weight_max: float,
    diagonal_shift: float,
):
    """Draw smooth graph signals from a Laplacian-based Gaussian model."""
    rng = np.random.default_rng(seed)
    true_adjacency = make_sparse_adjacency(
        n_features=n_features,
        true_nnz=true_nnz,
        seed=seed,
        dtype=dtype,
        edge_weight_min=edge_weight_min,
        edge_weight_max=edge_weight_max,
    )
    true_laplacian = graph_laplacian(true_adjacency)

    # A pure graph Laplacian is singular. The diagonal shift makes the Gaussian
    # precision SPD while preserving the graph smoothness encoded by L.
    true_precision = true_laplacian + diagonal_shift * np.eye(n_features, dtype=dtype)
    true_covariance = np.linalg.inv(true_precision)
    chol_covariance = np.linalg.cholesky(true_covariance)
    samples = rng.standard_normal((n_samples, n_features)).astype(dtype) @ chol_covariance.T
    samples -= samples.mean(axis=0, keepdims=True)

    # Kalofolias expects rows as nodes and columns as observed smooth signals.
    signals = samples.T
    if true_adjacency.max() > 0.0:
        true_adjacency = true_adjacency / true_adjacency.max()
    true_support = true_adjacency > 1e-12
    return signals, true_adjacency, true_support


def upper_values(matrix: np.ndarray) -> np.ndarray:
    """Return upper-triangular off-diagonal values."""
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def graph_metrics(adjacency: np.ndarray, true_adjacency: np.ndarray, true_support: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compare learned adjacency against the true sparse graph."""
    learned = adjacency.copy()
    np.fill_diagonal(learned, 0.0)
    learned_support = learned > threshold

    learned_edges = np.triu(learned_support, k=1)
    true_edges = np.triu(true_support, k=1)
    tp = float(np.logical_and(learned_edges, true_edges).sum())
    fp = float(np.logical_and(learned_edges, ~true_edges).sum())
    fn = float(np.logical_and(~learned_edges, true_edges).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(np.float64).eps)

    learned_upper = upper_values(learned)
    true_upper = upper_values(true_adjacency)
    learned_norm = learned_upper / max(np.linalg.norm(learned_upper), np.finfo(np.float64).eps)
    true_norm = true_upper / max(np.linalg.norm(true_upper), np.finfo(np.float64).eps)
    rel_fro = float(np.linalg.norm(learned_norm - true_norm))
    rel_spec = float(
        np.linalg.norm(
            learned / max(np.linalg.norm(learned, ord="fro"), np.finfo(np.float64).eps)
            - true_adjacency / max(np.linalg.norm(true_adjacency, ord="fro"), np.finfo(np.float64).eps),
            ord=2,
        )
    )
    n_nodes = adjacency.shape[0]
    possible_edges = n_nodes * (n_nodes - 1) // 2
    nnz = int(learned_edges.sum())
    return {
        "nnz": float(nnz),
        "nnz_ratio": float(nnz / possible_edges),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rel_fro": rel_fro,
        "rel_spec": rel_spec,
    }


def checkpoint_sequence(max_iter: int, checkpoints: str) -> List[int]:
    """Parse or generate increasing checkpoint iterations."""
    if checkpoints:
        values = sorted({int(value.strip()) for value in checkpoints.split(",") if value.strip()})
        return [value for value in values if 0 < value <= max_iter]
    base = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    values = [value for value in base if value <= max_iter]
    if not values or values[-1] != max_iter:
        values.append(max_iter)
    return values


def print_table(rows: List[Dict[str, float]]) -> None:
    """Print markdown table for recovery checkpoints."""
    headers = ["iter", "nnz", "nnz_ratio", "precision", "recall", "f1", "rel_fro", "rel_spec", "rel_change"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(
            f"{int(row['iter'])} | "
            f"{int(row['nnz'])} | "
            f"{row['nnz_ratio']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{row['rel_fro']:.6g} | "
            f"{row['rel_spec']:.6g} | "
            f"{row['rel_change']:.6g}"
        )


def plot_rows(rows: List[Dict[str, float]], output: Path, true_nnz_ratio: float) -> None:
    """Plot recovery metrics over checkpoints."""
    import os

    cache = Path(f"/tmp/matplotlib-cache-{os.getuid()}")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    iters = np.asarray([row["iter"] for row in rows], dtype=np.float64)
    nnz_ratio = 100.0 * np.asarray([row["nnz_ratio"] for row in rows], dtype=np.float64)
    f1 = np.asarray([row["f1"] for row in rows], dtype=np.float64)
    rel_fro = np.asarray([row["rel_fro"] for row in rows], dtype=np.float64)
    rel_spec = np.asarray([row["rel_spec"] for row in rows], dtype=np.float64)
    rel_change = np.asarray([row["rel_change"] for row in rows], dtype=np.float64)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    axes[0].semilogx(iters, nnz_ratio, marker="o")
    axes[0].axhline(100.0 * true_nnz_ratio, color="black", linestyle="--", linewidth=1.0, label="true graph")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("nnz ratio (%)")
    axes[0].set_title("Recovered graph sparsity")
    axes[0].grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[0].legend()

    axes[1].semilogx(iters, f1, marker="o", label="support F1")
    axes[1].semilogx(iters, rel_fro, marker="o", label="relative Frobenius error")
    axes[1].semilogx(iters, rel_spec, marker="o", label="relative spectral error")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("metric")
    axes[1].set_title("Recovery quality")
    axes[1].grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[1].legend()

    axes[2].loglog(iters, rel_change, marker="o")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("relative edge-vector change")
    axes[2].set_title("Solver stopping diagnostic")
    axes[2].grid(True, which="both", linewidth=0.4, alpha=0.35)

    fig.suptitle("Kalofolias smooth-signal graph recovery")
    fig.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Kalofolias graph learning on synthetic sparse graph signals.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--checkpoints", type=str, default="")
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--true-nnz", type=int, default=160, help="number of undirected nonzero graph edges")
    parser.add_argument("--edge-weight-min", type=float, default=0.15)
    parser.add_argument("--edge-weight-max", type=float, default=0.35)
    parser.add_argument("--diagonal-shift", type=float, default=0.1)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--output", type=Path, default=Path("kalofolias_graph_learning/figures/recovery_vs_iteration.png"))
    args = parser.parse_args()

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
    checkpoints = checkpoint_sequence(args.max_iter, args.checkpoints)
    true_nnz = int(np.triu(true_support, k=1).sum())
    possible_edges = args.n_features * (args.n_features - 1) // 2
    true_nnz_ratio = true_nnz / possible_edges
    rows: List[Dict[str, float]] = []

    for checkpoint in checkpoints:
        result = learn_graph_from_smooth_signals(
            signals,
            alpha=args.alpha,
            beta=args.beta,
            max_iter=checkpoint,
            tol=args.tol,
            threshold=args.threshold,
            return_info=True,
        )
        metrics = graph_metrics(result.adjacency, true_adjacency, true_support, args.threshold)
        rows.append(
            {
                "iter": float(checkpoint),
                "rel_change": result.relative_change,
                **metrics,
            }
        )

    print(
        "Kalofolias smooth-signal graph recovery: "
        f"n_samples={args.n_samples}, n_features={args.n_features}, "
        f"requested_true_nnz={args.true_nnz}, "
        f"true_nnz={true_nnz}, alpha={args.alpha}, beta={args.beta}, "
        f"threshold={args.threshold}, max_iter={args.max_iter}, dtype={args.dtype}"
    )
    print_table(rows)
    plot_rows(rows, args.output, true_nnz_ratio)
    print(f"figure={args.output}")


if __name__ == "__main__":
    main()
