"""Torch module wrapper for Kalofolias smooth-signal graph learning."""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from .kalofolias import learn_graph_from_smooth_signals


def _adjacency_to_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Convert adjacency matrix/matrices to graph Laplacian."""
    degree = adjacency.sum(dim=-1)
    return torch.diag_embed(degree) - adjacency


class KalofoliasGraphLearningModule(nn.Module):
    """Estimate a sparse graph from smooth node signals.

    Args:
        alpha: Log-degree barrier weight in the Kalofolias objective.
        beta: L2 edge-weight penalty.
        max_iter: Maximum primal-dual iterations.
        tol: Relative edge-vector change tolerance.
        threshold: Small-weight threshold used only for solver diagnostics.
        output_mode: ``"adjacency"`` returns the learned nonnegative graph
            weights. ``"laplacian"`` returns ``diag(W 1) - W``.
        normalize_distances: Median-normalize pairwise node distances before
            optimization.
        allow_backward: This wrapper is algorithmic and non-differentiable.
            Keep False. Passing True raises ``NotImplementedError``.

    Input:
        ``signals`` with shape ``(N, S)`` or ``(B, N, S)``, where ``N`` is the
        number of nodes and ``S`` is the number of observed smooth signals.

    Output:
        ``(N, N)`` or ``(B, N, N)`` according to ``output_mode``.
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
    ):
        super().__init__()
        output_mode = output_mode.lower()
        if output_mode not in self.VALID_OUTPUT_MODES:
            raise ValueError("output_mode must be one of: 'adjacency', 'laplacian'")
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.threshold = threshold
        self.output_mode = output_mode
        self.normalize_distances = normalize_distances
        self.allow_backward = allow_backward

    def _estimate_single(self, signals: torch.Tensor) -> torch.Tensor:
        """Run the NumPy solver for one ``(N, S)`` signal matrix."""
        result = learn_graph_from_smooth_signals(
            signals,
            alpha=self.alpha,
            beta=self.beta,
            max_iter=self.max_iter,
            tol=self.tol,
            threshold=self.threshold,
            normalize_distances=self.normalize_distances,
            return_info=True,
        )
        adjacency = torch.as_tensor(result.adjacency, dtype=signals.dtype, device=signals.device)
        if self.output_mode == "laplacian":
            return _adjacency_to_laplacian(adjacency)
        return adjacency

    def forward(self, signals: Union[torch.Tensor, object]) -> torch.Tensor:
        """Estimate graph matrix/matrices from node-signal data."""
        if self.allow_backward:
            raise NotImplementedError("KalofoliasGraphLearningModule does not support autograd through the solver")
        if not torch.is_tensor(signals):
            signals = torch.as_tensor(signals)
        if not torch.is_floating_point(signals):
            signals = signals.to(torch.float32)
        if signals.ndim not in (2, 3):
            raise ValueError("signals must have shape (N, S) or (B, N, S)")
        if not torch.isfinite(signals).all():
            raise ValueError("signals contains NaN or Inf")

        with torch.no_grad():
            detached = signals.detach()
            if detached.ndim == 2:
                return self._estimate_single(detached)
            return torch.stack([self._estimate_single(detached[i]) for i in range(detached.size(0))], dim=0)
