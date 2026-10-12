"""Minimal Local Kalofolias module with only model-required functionality."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalLocalKalofolias(nn.Module):
    """Learn local edge weights directly from a local distance tensor ``z``."""

    def __init__(
        self,
        neighbor_list=None,
        neighbor_mask=None,
        alpha=0.3,
        beta=1.0,
        gamma=None,
        max_iter=200,
        normalize_distances=True,
    ):
        super().__init__()
        if neighbor_list is None:
            self.register_buffer("neighbor_list", None)
            self.register_buffer("neighbor_mask", None)
        else:
            neighbor_list = torch.as_tensor(neighbor_list, dtype=torch.long)
            self.register_buffer("neighbor_list", neighbor_list)
            if neighbor_mask is None:
                neighbor_mask = torch.ones_like(neighbor_list, dtype=torch.bool)
            self.register_buffer("neighbor_mask", torch.as_tensor(neighbor_mask, dtype=torch.bool))

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.normalize_distances = normalize_distances

    def compute_z(self, signals):
        """Compute local squared distances from ``(N,S)`` or ``(B,N,S)`` signals."""
        neighbors = self.neighbor_list
        if signals.ndim == 2:
            neighbor_signals = signals[neighbors]
            z = (signals[:, None, :] - neighbor_signals).square().sum(dim=-1)
        else:
            batch_size = signals.size(0)
            neighbor_signals = signals[:, neighbors.reshape(-1), :].reshape(
                batch_size,
                neighbors.size(0),
                neighbors.size(1),
                signals.size(-1),
            )
            z = (signals[:, :, None, :] - neighbor_signals).square().sum(dim=-1)

        mask = self.neighbor_mask if z.ndim == 2 else self.neighbor_mask.unsqueeze(0)
        z = z * mask
        if self.normalize_distances:
            positive = z[(z > 0) & mask]
            if positive.numel() > 0:
                z = z / positive.median().clamp_min(torch.finfo(z.dtype).eps)
        return z

    def _gamma_schedule(self, z):
        if self.gamma is None:
            k_neighbors = z.size(-1)
            return z.new_tensor(0.9 / (2.0 * self.beta + k_neighbors**0.5 + 1.0))

        gamma = torch.as_tensor(self.gamma, device=z.device, dtype=z.dtype).reshape(-1)
        if gamma.numel() == 1:
            return gamma[0]
        if gamma.numel() == self.max_iter:
            return gamma
        return F.interpolate(
            gamma.reshape(1, 1, -1),
            size=self.max_iter,
            mode="linear",
            align_corners=True,
        ).reshape(-1)

    def forward(self, z):
        """Return local weights with the same ``(N,K)`` or ``(B,N,K)`` shape as ``z``."""
        if self.neighbor_mask is None:
            mask = torch.ones_like(z, dtype=torch.bool)
        else:
            mask = self.neighbor_mask if z.ndim == 2 else self.neighbor_mask.unsqueeze(0)

        valid_counts = mask.sum(dim=-1).to(z.dtype)
        W = valid_counts.reciprocal().unsqueeze(-1) * mask
        if z.ndim == 3 and W.size(0) == 1:
            W = W.expand_as(z)
        d = torch.zeros_like(z[..., 0])
        gamma_schedule = self._gamma_schedule(z)

        for iteration in range(self.max_iter):
            gamma = gamma_schedule if gamma_schedule.ndim == 0 else gamma_schedule[iteration]

            y = (W - gamma * (2.0 * self.beta * W + d.unsqueeze(-1))) * mask
            y_bar = d + gamma * W.sum(dim=-1)

            p = torch.relu(y - 2.0 * gamma * z) * mask
            p_bar = 0.5 * (
                y_bar
                - torch.sqrt(y_bar.square() + 4.0 * self.alpha * gamma)
            )

            q = (p - gamma * (2.0 * self.beta * p + p_bar.unsqueeze(-1))) * mask
            q_bar = p_bar + gamma * p.sum(dim=-1)

            W = (W - y + q) * mask
            d = d - y_bar + q_bar

        return torch.relu(W) * mask
