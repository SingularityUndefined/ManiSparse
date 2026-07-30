"""Local-neighbor Kalofolias graph learning in PyTorch.

This module keeps the Kalofolias smooth-signal objective, but restricts the
candidate graph to a fixed neighbor list. The learned graph is represented as a
local sparse weight matrix:

    local_weights: (N, K)

where ``local_weights[i, k]`` is the weight from node ``i`` to
``neighbor_list[i, k]``. This matches the sparse graph operators used by the
unrolled model more closely than a dense ``(N, N)`` matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn


@dataclass
class LocalKalofoliasResult:
    """Result returned by ``learn_local_graph_from_smooth_signals(..., return_info=True)``."""

    local_weights: torch.Tensor
    neighbor_list: torch.Tensor
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


def build_ring_neighbor_list(n_nodes: int = 40, k: int = 10, device=None) -> torch.Tensor:
    """Build a ring-local neighbor list with half neighbors on each side.

    For ``n_nodes=40`` and ``k=10``, row ``i`` contains
    ``i-5, ..., i-1, i+1, ..., i+5`` modulo 40.
    """
    if k <= 0 or k % 2 != 0:
        raise ValueError("k must be a positive even integer")
    if k >= n_nodes:
        raise ValueError("k must be smaller than n_nodes")
    nodes = torch.arange(n_nodes, device=device)
    half = k // 2
    offsets = torch.cat(
        [
            torch.arange(-half, 0, device=device),
            torch.arange(1, half + 1, device=device),
        ]
    )
    return (nodes[:, None] + offsets[None, :]) % n_nodes


def _validate_signals(signals: torch.Tensor) -> torch.Tensor:
    """Return finite floating tensor with shape ``(N, S)`` or ``(B, N, S)``."""
    if not torch.is_tensor(signals):
        signals = torch.as_tensor(signals)
    if not torch.is_floating_point(signals):
        signals = signals.to(torch.float32)
    if signals.ndim not in (2, 3):
        raise ValueError("signals must have shape (N, S) or (B, N, S)")
    if not torch.isfinite(signals).all():
        raise ValueError("signals contains NaN or Inf")
    return signals


def _validate_neighbor_list(neighbor_list: torch.Tensor, n_nodes: int, device) -> torch.Tensor:
    """Return integer neighbor list with shape ``(N, K)``."""
    if not torch.is_tensor(neighbor_list):
        neighbor_list = torch.as_tensor(neighbor_list)
    neighbor_list = neighbor_list.to(device=device, dtype=torch.long)
    if neighbor_list.ndim != 2 or neighbor_list.size(0) != n_nodes:
        raise ValueError("neighbor_list must have shape (N, K)")
    if neighbor_list.numel() == 0:
        raise ValueError("neighbor_list cannot be empty")
    if int(neighbor_list.min()) < 0 or int(neighbor_list.max()) >= n_nodes:
        raise ValueError("neighbor_list contains node indices outside [0, N)")
    return neighbor_list


def local_weights_to_dense(
    local_weights: torch.Tensor,
    neighbor_list: Union[torch.Tensor, object],
    *,
    symmetric: bool = True,
) -> torch.Tensor:
    """Scatter local weights ``(..., N, K)`` into dense adjacency ``(..., N, N)``.

    Args:
        local_weights: Local edge weights with shape ``(N, K)`` or
            ``(B, N, K)``.
        neighbor_list: Candidate neighbor indices with shape ``(N, K)``.
        symmetric: If true, average the directed local slots with their
            transpose so the returned dense adjacency is undirected.

    Returns:
        Dense adjacency with shape ``(N, N)`` or ``(B, N, N)``.
    """
    if not torch.is_tensor(local_weights):
        local_weights = torch.as_tensor(local_weights)
    if local_weights.ndim not in (2, 3):
        raise ValueError("local_weights must have shape (N, K) or (B, N, K)")
    n_nodes, k_neighbors = local_weights.shape[-2:]
    neighbor_list = _validate_neighbor_list(neighbor_list, n_nodes, local_weights.device)
    if neighbor_list.size(1) != k_neighbors:
        raise ValueError("neighbor_list and local_weights must use the same K")

    batch_shape = local_weights.shape[:-2]
    dense = local_weights.new_zeros(*batch_shape, n_nodes, n_nodes)
    row_index = torch.arange(n_nodes, device=local_weights.device).unsqueeze(-1).expand_as(neighbor_list)
    if local_weights.ndim == 2:
        dense[row_index, neighbor_list] = local_weights
    else:
        dense[:, row_index, neighbor_list] = local_weights
    dense.diagonal(dim1=-2, dim2=-1).zero_()
    if symmetric:
        dense = 0.5 * (dense + dense.transpose(-1, -2))
    return dense


def _local_pairwise_squared_distances(
    signals: torch.Tensor,
    neighbor_list: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    """Compute local squared distances, shape ``(N, K)`` or ``(B, N, K)``."""
    if signals.ndim == 2:
        neighbor_signal = signals[neighbor_list]
        dist = (signals[:, None, :] - neighbor_signal).square().sum(dim=-1)
    else:
        batch_size = signals.size(0)
        flat_neighbors = neighbor_list.reshape(-1)
        neighbor_signal = signals[:, flat_neighbors, :].reshape(
            batch_size,
            neighbor_list.size(0),
            neighbor_list.size(1),
            signals.size(-1),
        )
        dist = (signals[:, :, None, :] - neighbor_signal).square().sum(dim=-1)

    if normalize:
        positive = dist[dist > 0]
        if positive.numel() > 0:
            dist = dist / positive.median().clamp_min(torch.finfo(dist.dtype).eps)
    return dist


def _default_step_size(k_neighbors: int, beta: float, safety: float) -> float:
    """Conservative primal-dual step for the local row-sum degree operator."""
    operator_norm = k_neighbors ** 0.5
    lipschitz = 2.0 * beta
    return float(safety / (lipschitz + operator_norm + 1.0))


def _local_objective(w: torch.Tensor, z: torch.Tensor, degree: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Evaluate local smooth-graph objective."""
    eps = torch.finfo(w.dtype).tiny
    return 2.0 * (z * w).sum() - alpha * torch.log(degree.clamp_min(eps)).sum() + beta * (w * w).sum()


def _solve_local(
    distances: torch.Tensor,
    neighbor_list: torch.Tensor,
    alpha: float,
    beta: float,
    max_iter: int,
    tol: float,
    step_size: Optional[float],
    step_safety: float,
    threshold: float,
) -> LocalKalofoliasResult:
    """Run local primal-dual updates on a distance tensor."""
    k_neighbors = distances.size(-1)
    gamma = _default_step_size(k_neighbors, beta, step_safety) if step_size is None else float(step_size)
    if gamma <= 0:
        raise ValueError("step_size must be positive")

    w = torch.full_like(distances, 1.0 / max(k_neighbors, 1))
    d = torch.zeros_like(distances[..., 0])
    relative_change_history: List[float] = []
    objective_history: List[float] = []
    nnz_ratio_history: List[float] = []
    converged = False
    relative_change = float("inf")

    for iteration in range(1, max_iter + 1):
        w_prev = w

        # Local degree operator:
        #   S w = row_sum(w), shape (..., N)
        #   S^T d = d[..., i] broadcast to every candidate edge of node i.
        st_d = d.unsqueeze(-1)
        y = w - gamma * (2.0 * beta * w + st_d)
        y_bar = d + gamma * w.sum(dim=-1)

        p = torch.clamp(y - 2.0 * gamma * distances, min=0.0)
        p_bar = 0.5 * (y_bar - torch.sqrt(y_bar.square() + 4.0 * alpha * gamma))

        q = p - gamma * (2.0 * beta * p + p_bar.unsqueeze(-1))
        q_bar = p_bar + gamma * p.sum(dim=-1)

        w = w - y + q
        d = d - y_bar + q_bar

        degree = w.sum(dim=-1)
        objective = _local_objective(w, distances, degree, alpha, beta)
        relative_change_tensor = torch.linalg.vector_norm(w - w_prev) / torch.linalg.vector_norm(w_prev).clamp_min(
            torch.finfo(w.dtype).eps
        )
        relative_change = float(relative_change_tensor.detach().cpu())
        objective_history.append(float(objective.detach().cpu()))
        relative_change_history.append(relative_change)
        nnz_ratio_history.append(float((w > threshold).to(torch.float64).mean().detach().cpu()))
        if relative_change <= tol:
            converged = True
            break

    w = torch.clamp(w, min=0.0)
    return LocalKalofoliasResult(
        local_weights=w,
        neighbor_list=neighbor_list,
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


def learn_local_graph_from_smooth_signals(
    signals: Union[torch.Tensor, object],
    neighbor_list: Union[torch.Tensor, object],
    *,
    alpha: float = 0.3,
    beta: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
    step_size: Optional[float] = None,
    step_safety: float = 0.9,
    normalize_distances: bool = True,
    threshold: float = 1e-4,
    return_info: bool = False,
) -> Union[torch.Tensor, LocalKalofoliasResult]:
    """Learn local sparse graph weights from smooth node signals.

    Args:
        signals: ``(N, S)`` or ``(B, N, S)`` smooth node signals.
        neighbor_list: ``(N, K)`` candidate neighbor indices.

    Returns:
        Local weights with shape ``(N, K)`` or ``(B, N, K)`` by default.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")

    signals = _validate_signals(signals)
    n_nodes = signals.size(-2)
    neighbor_list = _validate_neighbor_list(neighbor_list, n_nodes, signals.device)
    distances = _local_pairwise_squared_distances(signals, neighbor_list, normalize_distances)
    result = _solve_local(
        distances,
        neighbor_list,
        alpha,
        beta,
        max_iter,
        tol,
        step_size,
        step_safety,
        threshold,
    )
    return result if return_info else result.local_weights


class LocalKalofoliasGraphLearning(nn.Module):
    """PyTorch module wrapper for local-neighbor Kalofolias graph learning.

    The default output is a local sparse matrix with shape ``(N, K)`` or
    ``(B, N, K)``. Autograd through the solver is disabled by default.
    """

    def __init__(
        self,
        neighbor_list: Union[torch.Tensor, object],
        alpha: float = 0.3,
        beta: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        step_size: Optional[float] = None,
        step_safety: float = 0.9,
        normalize_distances: bool = True,
        threshold: float = 1e-4,
        allow_backward: bool = False,
    ):
        super().__init__()
        self.register_buffer("neighbor_list", torch.as_tensor(neighbor_list, dtype=torch.long))
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.step_safety = step_safety
        self.normalize_distances = normalize_distances
        self.threshold = threshold
        self.allow_backward = allow_backward

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Return local graph weights with shape ``(N, K)`` or ``(B, N, K)``."""
        if self.allow_backward:
            return learn_local_graph_from_smooth_signals(
                signals,
                self.neighbor_list,
                alpha=self.alpha,
                beta=self.beta,
                max_iter=self.max_iter,
                tol=self.tol,
                step_size=self.step_size,
                step_safety=self.step_safety,
                normalize_distances=self.normalize_distances,
                threshold=self.threshold,
                return_info=False,
            )
        with torch.no_grad():
            return learn_local_graph_from_smooth_signals(
                signals.detach(),
                self.neighbor_list,
                alpha=self.alpha,
                beta=self.beta,
                max_iter=self.max_iter,
                tol=self.tol,
                step_size=self.step_size,
                step_safety=self.step_safety,
                normalize_distances=self.normalize_distances,
                threshold=self.threshold,
                return_info=False,
            )
