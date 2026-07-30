"""Kalofolias graph learning from smooth signals.

This implements the primal-dual Algorithm 1 from:

    V. Kalofolias, "How to Learn a Graph from Smooth Signals", AISTATS 2016.

The solver estimates a nonnegative undirected weighted graph from node signals
by solving the vectorized objective

    min_w  2 z^T w - alpha * sum_i log((S w)_i) + beta * ||w||_2^2
    s.t.   w >= 0,

where ``w`` contains the upper-triangular edge weights, ``z`` contains squared
pairwise distances between node signal vectors, and ``S w`` is the node degree
vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class KalofoliasResult:
    """Result returned by ``learn_graph_from_smooth_signals(..., return_info=True)``."""

    adjacency: np.ndarray
    edge_weights: np.ndarray
    edge_index: np.ndarray
    degree: np.ndarray
    n_iter: int
    converged: bool
    relative_change: float
    objective: float
    relative_change_history: List[float]
    objective_history: List[float]
    nnz_ratio_history: List[float]
    alpha: float
    beta: float
    step_size: float


def _validate_signals(signals: Union[np.ndarray, object]) -> np.ndarray:
    """Return finite float64 node-signal matrix with shape ``(n_nodes, n_signals)``."""
    if hasattr(signals, "detach"):
        signals = signals.detach().cpu().numpy()
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("signals must have shape (n_nodes, n_signals)")
    if x.shape[0] < 2:
        raise ValueError("signals must contain at least two nodes")
    if not np.isfinite(x).all():
        raise ValueError("signals contains NaN or Inf")
    return x


def pairwise_squared_distances(signals: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute squared distances between node signal rows.

    Args:
        signals: Node signal matrix with shape ``(n_nodes, n_signals)``.
        normalize: If True, divide distances by their positive median. This
            keeps ``alpha`` and ``beta`` usable across synthetic scales.
    """
    x = _validate_signals(signals)
    gram = x @ x.T
    sq_norm = np.diag(gram)
    dist = sq_norm[:, None] + sq_norm[None, :] - 2.0 * gram
    dist = np.maximum(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    if normalize:
        positive = dist[dist > 0.0]
        if positive.size:
            dist = dist / np.median(positive)
    return dist


def _complete_graph_edges(n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return upper-triangular complete graph edge endpoint vectors."""
    return np.triu_indices(n_nodes, k=1)


def _degree_from_edges(w: np.ndarray, src: np.ndarray, dst: np.ndarray, n_nodes: int) -> np.ndarray:
    """Apply ``S w``: edge weights to node degrees, shape ``(n_nodes,)``."""
    degree = np.zeros(n_nodes, dtype=w.dtype)
    np.add.at(degree, src, w)
    np.add.at(degree, dst, w)
    return degree


def _adjacency_from_edges(w: np.ndarray, src: np.ndarray, dst: np.ndarray, n_nodes: int) -> np.ndarray:
    """Expand upper-triangular edge vector into symmetric adjacency matrix."""
    adjacency = np.zeros((n_nodes, n_nodes), dtype=w.dtype)
    adjacency[src, dst] = w
    adjacency[dst, src] = w
    return adjacency


def _objective(w: np.ndarray, z: np.ndarray, degree: np.ndarray, alpha: float, beta: float) -> float:
    """Evaluate the smooth-graph objective for positive-degree iterates."""
    eps = np.finfo(w.dtype).tiny
    return float(2.0 * z.dot(w) - alpha * np.log(np.maximum(degree, eps)).sum() + beta * w.dot(w))


def _default_step_size(n_nodes: int, beta: float, safety: float) -> float:
    """Conservative Condat-style step size for Algorithm 1."""
    # For the complete graph degree operator S, ||S||_2 = sqrt(2 * (n_nodes - 1)).
    operator_norm = np.sqrt(2.0 * (n_nodes - 1.0))
    lipschitz = 2.0 * beta
    return float(safety / (lipschitz + operator_norm + 1.0))


def learn_graph_from_smooth_signals(
    signals: Union[np.ndarray, object],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-5,
    step_size: Optional[float] = None,
    step_safety: float = 0.9,
    distance_matrix: Optional[np.ndarray] = None,
    normalize_distances: bool = True,
    threshold: float = 1e-8,
    return_info: bool = False,
) -> Union[np.ndarray, KalofoliasResult]:
    """Learn a sparse weighted graph from smooth node signals.

    Args:
        signals: Smooth node signals, shape ``(n_nodes, n_signals)``. Each row
            is one node's signal vector over samples or time.
        alpha: Log-degree barrier weight. Larger values encourage non-isolated
            nodes and generally denser graphs.
        beta: L2 edge-weight penalty. Larger values shrink edge weights.
        max_iter: Maximum primal-dual iterations.
        tol: Stop when the relative edge-vector change drops below this value.
        step_size: Optional primal-dual step. If omitted, a conservative value
            is chosen from the degree operator norm and ``beta``.
        step_safety: Multiplier used only when ``step_size`` is omitted.
        distance_matrix: Optional precomputed squared-distance matrix,
            shape ``(n_nodes, n_nodes)``.
        normalize_distances: If True, median-normalize pairwise distances.
        threshold: Entries at or below this value count as zero in diagnostics.
        return_info: If True, return ``KalofoliasResult``. Otherwise return only
            the adjacency matrix ``W``.

    Returns:
        Symmetric nonnegative adjacency matrix with shape ``(n_nodes, n_nodes)``,
        or ``KalofoliasResult`` when ``return_info=True``.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")

    x = _validate_signals(signals)
    n_nodes = x.shape[0]
    if distance_matrix is None:
        distances = pairwise_squared_distances(x, normalize=normalize_distances)
    else:
        distances = np.asarray(distance_matrix, dtype=np.float64)
        if distances.shape != (n_nodes, n_nodes):
            raise ValueError("distance_matrix must have shape (n_nodes, n_nodes)")
        if not np.isfinite(distances).all():
            raise ValueError("distance_matrix contains NaN or Inf")
        distances = 0.5 * (distances + distances.T)
        np.fill_diagonal(distances, 0.0)

    src, dst = _complete_graph_edges(n_nodes)
    z = distances[src, dst]
    n_edges = z.shape[0]
    gamma = _default_step_size(n_nodes, beta, step_safety) if step_size is None else float(step_size)
    if gamma <= 0:
        raise ValueError("step_size must be positive")

    # Edge weights w are initialized with positive degrees so the log-degree
    # barrier is finite from the first objective evaluation.
    w = np.full(n_edges, 1.0 / max(n_nodes - 1, 1), dtype=np.float64)
    d = np.zeros(n_nodes, dtype=np.float64)
    relative_change_history: List[float] = []
    objective_history: List[float] = []
    nnz_ratio_history: List[float] = []
    converged = False
    relative_change = float("inf")

    for iteration in range(1, max_iter + 1):
        w_prev = w.copy()

        # S^T d maps node dual variables back to edge variables:
        # one value per candidate edge (i, j), equal to d_i + d_j.
        st_d = d[src] + d[dst]
        y = w - gamma * (2.0 * beta * w + st_d)
        y_bar = d + gamma * _degree_from_edges(w, src, dst, n_nodes)

        p = np.maximum(0.0, y - 2.0 * gamma * z)

        # Proximal operator of the conjugate of -alpha * log(degree).
        p_bar = 0.5 * (y_bar - np.sqrt(y_bar * y_bar + 4.0 * alpha * gamma))

        st_p_bar = p_bar[src] + p_bar[dst]
        q = p - gamma * (2.0 * beta * p + st_p_bar)
        q_bar = p_bar + gamma * _degree_from_edges(p, src, dst, n_nodes)

        w = w - y + q
        d = d - y_bar + q_bar

        degree = _degree_from_edges(w, src, dst, n_nodes)
        objective = _objective(w, z, degree, alpha, beta)
        relative_change = float(np.linalg.norm(w - w_prev) / max(np.linalg.norm(w_prev), np.finfo(np.float64).eps))
        nnz_ratio = float(np.count_nonzero(w > threshold) / n_edges)
        objective_history.append(objective)
        relative_change_history.append(relative_change)
        nnz_ratio_history.append(nnz_ratio)

        if relative_change <= tol:
            converged = True
            break

    w = np.maximum(w, 0.0)
    adjacency = _adjacency_from_edges(w, src, dst, n_nodes)
    degree = _degree_from_edges(w, src, dst, n_nodes)
    edge_index = np.stack([src, dst], axis=0)
    result = KalofoliasResult(
        adjacency=adjacency,
        edge_weights=w,
        edge_index=edge_index,
        degree=degree,
        n_iter=iteration,
        converged=converged,
        relative_change=relative_change,
        objective=objective_history[-1],
        relative_change_history=relative_change_history,
        objective_history=objective_history,
        nnz_ratio_history=nnz_ratio_history,
        alpha=float(alpha),
        beta=float(beta),
        step_size=float(gamma),
    )
    return result if return_info else adjacency
