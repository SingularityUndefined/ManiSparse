import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from local_kalofolias import build_ring_neighbor_list, learn_local_graph_from_smooth_signals, local_weights_to_dense


def circular_distance(i: int, j: int, n_nodes: int) -> int:
    """Return ring distance between two node indices."""
    raw = abs(i - j)
    return min(raw, n_nodes - raw)


def make_local_true_graph(
    neighbor_list: torch.Tensor,
    active_neighbors_per_node: int,
    seed: int,
    edge_weight_min: float,
    edge_weight_max: float,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create local ground-truth weights inside the given candidate slots.

    The benchmark uses a ring-local support. For the default ``N=40, K=10`` and
    ``active_neighbors_per_node=4``, every node is connected only to
    ``i-2, i-1, i+1, i+2`` inside its 10 candidate neighbors.

    Returns:
        ``true_local_weights`` with shape ``(N, K)`` and symmetric dense
        adjacency ``true_adjacency`` with shape ``(N, N)``.
    """
    if active_neighbors_per_node <= 0 or active_neighbors_per_node % 2 != 0:
        raise ValueError("active_neighbors_per_node must be a positive even integer")
    n_nodes, k_neighbors = neighbor_list.shape
    if active_neighbors_per_node > k_neighbors:
        raise ValueError("active_neighbors_per_node cannot exceed K")

    rng = np.random.default_rng(seed)
    radius = active_neighbors_per_node // 2
    dense = torch.zeros(n_nodes, n_nodes, dtype=dtype)

    # Fill each undirected local edge once. The paired directed slots in
    # true_local_weights therefore share the same edge weight.
    for i in range(n_nodes):
        for j in neighbor_list[i].tolist():
            if i < j and circular_distance(i, j, n_nodes) <= radius:
                dense[i, j] = float(rng.uniform(edge_weight_min, edge_weight_max))
                dense[j, i] = dense[i, j]

    true_local = torch.zeros(n_nodes, k_neighbors, dtype=dtype)
    for i in range(n_nodes):
        true_local[i] = dense[i, neighbor_list[i]]
    if true_local.max() > 0:
        true_local = true_local / true_local.max()
        dense = dense / dense.max()
    return true_local, dense


def graph_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Return combinatorial Laplacian ``L = diag(W 1) - W``."""
    return torch.diag(adjacency.sum(dim=-1)) - adjacency


def make_problem(
    n_samples: int,
    n_nodes: int,
    k_neighbors: int,
    active_neighbors_per_node: int,
    seed: int,
    dtype: torch.dtype,
    edge_weight_min: float,
    edge_weight_max: float,
    diagonal_shift: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample smooth node signals from the local ground-truth graph."""
    neighbor_list = build_ring_neighbor_list(n_nodes, k_neighbors)
    true_local, true_adjacency = make_local_true_graph(
        neighbor_list,
        active_neighbors_per_node=active_neighbors_per_node,
        seed=seed,
        edge_weight_min=edge_weight_min,
        edge_weight_max=edge_weight_max,
        dtype=dtype,
    )
    true_precision = graph_laplacian(true_adjacency) + diagonal_shift * torch.eye(n_nodes, dtype=dtype)
    true_covariance = torch.linalg.inv(true_precision)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    samples = torch.randn(n_samples, n_nodes, generator=generator, dtype=dtype) @ torch.linalg.cholesky(true_covariance).T
    samples = samples - samples.mean(dim=0, keepdim=True)

    # The Kalofolias solver expects rows as nodes and columns as smooth signals.
    signals = samples.T.contiguous()
    return signals, neighbor_list, true_local, true_adjacency


def local_metrics(learned_local: torch.Tensor, true_local: torch.Tensor, threshold: float) -> Dict[str, float]:
    """Evaluate support and weight recovery on the candidate local edge slots."""
    learned = learned_local.detach().cpu().to(torch.float64)
    true = true_local.detach().cpu().to(torch.float64)
    learned_support = learned > threshold
    true_support = true > 1e-12

    tp = float(torch.logical_and(learned_support, true_support).sum())
    fp = float(torch.logical_and(learned_support, ~true_support).sum())
    fn = float(torch.logical_and(~learned_support, true_support).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(np.float64).eps)

    learned_vec = learned.reshape(-1)
    true_vec = true.reshape(-1)
    learned_norm = learned_vec / torch.linalg.vector_norm(learned_vec).clamp_min(torch.finfo(torch.float64).eps)
    true_norm = true_vec / torch.linalg.vector_norm(true_vec).clamp_min(torch.finfo(torch.float64).eps)
    rel_fro = float(torch.linalg.vector_norm(learned_norm - true_norm))

    nnz = int(learned_support.sum())
    total_slots = true.numel()
    true_nnz = int(true_support.sum())
    return {
        "nnz": float(nnz),
        "nnz_ratio": float(nnz / total_slots),
        "true_nnz": float(true_nnz),
        "true_nnz_ratio": float(true_nnz / total_slots),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rel_fro": rel_fro,
    }


def checkpoint_sequence(max_iter: int, checkpoints: str) -> List[int]:
    """Parse or generate increasing checkpoint iterations."""
    if checkpoints:
        values = sorted({int(value.strip()) for value in checkpoints.split(",") if value.strip()})
        return [value for value in values if 0 < value <= max_iter]
    base = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    values = [value for value in base if value <= max_iter]
    if not values or values[-1] != max_iter:
        values.append(max_iter)
    return values


def print_table(rows: List[Dict[str, float]]) -> None:
    """Print a markdown table for local recovery checkpoints."""
    headers = ["iter", "time_ms", "nnz", "nnz_ratio", "precision", "recall", "f1", "rel_fro", "rel_change"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(
            f"{int(row['iter'])} | "
            f"{row['time_ms']:.3f} | "
            f"{int(row['nnz'])} | "
            f"{row['nnz_ratio']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{row['rel_fro']:.6g} | "
            f"{row['rel_change']:.6g}"
        )


def plot_rows(rows: List[Dict[str, float]], output: Path, true_nnz_ratio: float) -> None:
    """Plot local recovery metrics over checkpoints."""
    import os

    cache = Path(f"/tmp/matplotlib-cache-{os.getuid()}")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    iters = np.asarray([row["iter"] for row in rows], dtype=np.float64)
    nnz_ratio = 100.0 * np.asarray([row["nnz_ratio"] for row in rows], dtype=np.float64)
    precision = np.asarray([row["precision"] for row in rows], dtype=np.float64)
    recall = np.asarray([row["recall"] for row in rows], dtype=np.float64)
    f1 = np.asarray([row["f1"] for row in rows], dtype=np.float64)
    rel_fro = np.asarray([row["rel_fro"] for row in rows], dtype=np.float64)
    rel_change = np.asarray([row["rel_change"] for row in rows], dtype=np.float64)
    time_ms = np.asarray([row["time_ms"] for row in rows], dtype=np.float64)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].semilogx(iters, nnz_ratio, marker="o")
    axes[0, 0].axhline(100.0 * true_nnz_ratio, color="black", linestyle="--", linewidth=1.0, label="true slots")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel("local nnz ratio (%)")
    axes[0, 0].set_title("Candidate-slot sparsity")
    axes[0, 0].grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[0, 0].legend()

    axes[0, 1].semilogx(iters, precision, marker="o", label="precision")
    axes[0, 1].semilogx(iters, recall, marker="o", label="recall")
    axes[0, 1].semilogx(iters, f1, marker="o", label="F1")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("support metric")
    axes[0, 1].set_title("Local support recovery")
    axes[0, 1].grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[0, 1].legend()

    axes[1, 0].semilogx(iters, rel_fro, marker="o")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("relative local-weight error")
    axes[1, 0].set_title("Weight recovery")
    axes[1, 0].grid(True, which="both", linewidth=0.4, alpha=0.35)

    axes[1, 1].loglog(iters, rel_change, marker="o", label="relative change")
    axes[1, 1].loglog(iters, np.maximum(time_ms, 1e-9), marker="o", label="time (ms)")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_title("Solver diagnostics")
    axes[1, 1].grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[1, 1].legend()

    fig.suptitle("Local-neighbor Kalofolias recovery on a fixed candidate edge set")
    fig.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local-neighbor Kalofolias graph learning.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-nodes", type=int, default=40)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--active-neighbors-per-node", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--checkpoints", type=str, default="")
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-weight-min", type=float, default=0.15)
    parser.add_argument("--edge-weight-max", type=float, default=0.35)
    parser.add_argument("--diagonal-shift", type=float, default=0.1)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--output", type=Path, default=Path("local_kalofolias/figures/local_recovery_vs_iteration.png"))
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    signals, neighbor_list, true_local, true_adjacency = make_problem(
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

    checkpoints = checkpoint_sequence(args.max_iter, args.checkpoints)
    rows: List[Dict[str, float]] = []
    for checkpoint in checkpoints:
        start = time.perf_counter()
        result = learn_local_graph_from_smooth_signals(
            signals,
            neighbor_list,
            alpha=args.alpha,
            beta=args.beta,
            max_iter=checkpoint,
            tol=args.tol,
            threshold=args.threshold,
            return_info=True,
        )
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        metrics = local_metrics(result.local_weights, true_local, args.threshold)
        rows.append(
            {
                "iter": float(checkpoint),
                "time_ms": elapsed_ms,
                "rel_change": result.relative_change,
                **metrics,
            }
        )

    learned_dense = local_weights_to_dense(result.local_weights, neighbor_list)
    dense_nnz = int(torch.triu(learned_dense > args.threshold, diagonal=1).sum())
    true_dense_nnz = int(torch.triu(true_adjacency > 1e-12, diagonal=1).sum())
    total_slots = args.n_nodes * args.k_neighbors
    true_local_nnz = int((true_local > 1e-12).sum())
    true_nnz_ratio = true_local_nnz / total_slots

    print(
        "Local Kalofolias fixed-candidate recovery: "
        f"n_samples={args.n_samples}, n_nodes={args.n_nodes}, K={args.k_neighbors}, "
        f"candidate_slots={total_slots}, active_slots_per_node={args.active_neighbors_per_node}, "
        f"true_local_nnz={true_local_nnz}, true_local_nnz_ratio={true_nnz_ratio:.4f}, "
        f"true_undirected_edges={true_dense_nnz}, learned_undirected_edges={dense_nnz}, "
        f"alpha={args.alpha}, beta={args.beta}, threshold={args.threshold}, "
        f"max_iter={args.max_iter}, dtype={args.dtype}"
    )
    print_table(rows)
    plot_rows(rows, args.output, true_nnz_ratio)
    print(f"figure={args.output}")


if __name__ == "__main__":
    main()
