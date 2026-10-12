"""Pure-PyTorch dense Kalofolias smooth-signal graph learning."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _adjacency_to_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Convert adjacency matrix/matrices to graph Laplacian."""
    degree = adjacency.sum(dim=-1)
    return torch.diag_embed(degree) - adjacency


class KalofoliasGraphLearningModule(nn.Module):
    """Estimate a dense graph from smooth node signals entirely in PyTorch.

    ``gamma`` may be a scalar or sequence of control points. Sequences are
    linearly interpolated over ``max_iter``. When ``gamma`` is omitted, the
    existing conservative automatic step-size rule is used.
    """

    VALID_OUTPUT_MODES = {"adjacency", "laplacian"}

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        threshold: float = 1e-4,
        output_mode: str = "adjacency",
        normalize_distances: bool = True,
        allow_backward: bool = False,
        gamma: Optional[Union[float, Sequence[float]]] = None,
    ):
        super().__init__()
        output_mode = output_mode.lower()
        if output_mode not in self.VALID_OUTPUT_MODES:
            raise ValueError("output_mode must be one of: 'adjacency', 'laplacian'")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if gamma is not None:
            gamma_values = torch.as_tensor(gamma)
            if gamma_values.ndim > 1 or gamma_values.numel() == 0:
                raise ValueError("gamma must be a non-empty scalar or one-dimensional sequence")
            if torch.any(~torch.isfinite(gamma_values)) or torch.any(gamma_values <= 0):
                raise ValueError("all gamma values must be finite and positive")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.threshold = threshold
        self.output_mode = output_mode
        self.normalize_distances = normalize_distances
        self.allow_backward = allow_backward

    @staticmethod
    def _pairwise_squared_distances(signals: torch.Tensor, normalize: bool) -> torch.Tensor:
        """Return the squared-distance matrix for one node-signal tensor."""
        differences = signals[:, None, :] - signals[None, :, :]
        distances = differences.square().sum(dim=-1)
        if normalize:
            positive = distances[distances > 0]
            if positive.numel() > 0:
                median = torch.quantile(positive, 0.5, interpolation="linear")
                distances = distances / median.clamp_min(torch.finfo(distances.dtype).eps)
        return distances

    def _gamma_schedule(self, n_nodes: int, reference: torch.Tensor) -> torch.Tensor:
        """Return a scalar gamma or one value per solver iteration."""
        if self.gamma is None:
            operator_norm = (2.0 * (n_nodes - 1.0)) ** 0.5
            return reference.new_tensor(0.9 / (2.0 * self.beta + operator_norm + 1.0))

        values = torch.as_tensor(self.gamma, device=reference.device, dtype=reference.dtype).reshape(-1)
        if values.numel() == 1:
            return values[0]
        if values.numel() == self.max_iter:
            return values
        return F.interpolate(
            values.reshape(1, 1, -1),
            size=self.max_iter,
            mode="linear",
            align_corners=True,
        ).reshape(-1)

    @staticmethod
    def _degree_from_edges(
        edge_weights: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """Apply the complete-graph degree operator to an edge vector."""
        degree = edge_weights.new_zeros(n_nodes)
        degree = degree.index_add(0, src, edge_weights)
        return degree.index_add(0, dst, edge_weights)

    def _estimate_single(self, signals: torch.Tensor) -> torch.Tensor:
        """Run the primal-dual iterations on one ``(N, S)`` tensor."""
        n_nodes = signals.size(0)
        if n_nodes < 2:
            raise ValueError("signals must contain at least two nodes")
        src, dst = torch.triu_indices(n_nodes, n_nodes, offset=1, device=signals.device)
        distances = self._pairwise_squared_distances(signals, self.normalize_distances)
        edge_distances = distances[src, dst]
        gamma_schedule = self._gamma_schedule(n_nodes, signals)

        edge_weights = signals.new_full((src.numel(),), 1.0 / (n_nodes - 1))
        dual_degree = signals.new_zeros(n_nodes)
        epsilon = torch.finfo(signals.dtype).eps

        for iteration in range(self.max_iter):
            gamma = gamma_schedule if gamma_schedule.ndim == 0 else gamma_schedule[iteration]
            previous_weights = edge_weights

            dual_on_edges = dual_degree[src] + dual_degree[dst]
            primal_forward = edge_weights - gamma * (2.0 * self.beta * edge_weights + dual_on_edges)
            dual_forward = dual_degree + gamma * self._degree_from_edges(edge_weights, src, dst, n_nodes)

            primal_prox = torch.clamp(primal_forward - 2.0 * gamma * edge_distances, min=0.0)
            dual_prox = 0.5 * (
                dual_forward - torch.sqrt(dual_forward.square() + 4.0 * self.alpha * gamma)
            )

            prox_dual_on_edges = dual_prox[src] + dual_prox[dst]
            primal_backward = primal_prox - gamma * (
                2.0 * self.beta * primal_prox + prox_dual_on_edges
            )
            dual_backward = dual_prox + gamma * self._degree_from_edges(primal_prox, src, dst, n_nodes)

            edge_weights = edge_weights - primal_forward + primal_backward
            dual_degree = dual_degree - dual_forward + dual_backward

            relative_change = torch.linalg.vector_norm(edge_weights - previous_weights) / torch.linalg.vector_norm(
                previous_weights
            ).clamp_min(epsilon)
            if float(relative_change.detach().cpu()) <= self.tol:
                break

        edge_weights = torch.clamp(edge_weights, min=0.0)
        adjacency = signals.new_zeros(n_nodes, n_nodes)
        adjacency[src, dst] = edge_weights
        adjacency[dst, src] = edge_weights
        if self.output_mode == "laplacian":
            return _adjacency_to_laplacian(adjacency)
        return adjacency

    def forward(self, signals: Union[torch.Tensor, object]) -> torch.Tensor:
        """Estimate graph matrix/matrices from node-signal data."""
        if not torch.is_tensor(signals):
            signals = torch.as_tensor(signals)
        if not torch.is_floating_point(signals):
            signals = signals.to(torch.float32)
        if signals.ndim not in (2, 3):
            raise ValueError("signals must have shape (N, S) or (B, N, S)")
        if not torch.isfinite(signals).all():
            raise ValueError("signals contains NaN or Inf")

        solver_signals = signals if self.allow_backward else signals.detach()
        context = torch.enable_grad() if self.allow_backward else torch.no_grad()
        with context:
            if solver_signals.ndim == 2:
                return self._estimate_single(solver_signals)
            return torch.stack([self._estimate_single(sample) for sample in solver_signals], dim=0)
