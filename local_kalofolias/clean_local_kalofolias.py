"""Clean Local Kalofolias module with only model-required functionality."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanLocalKalofolias(nn.Module):
    """Learn local edge weights directly from a local distance tensor ``z``."""

    alpha_min = 0.1
    alpha_max = 2.0
    beta_min = 0.5
    beta_max = 1.5
    parameter_share_interval = 1

    def __init__(
        self,
        neighbor_list=None,
        neighbor_mask=None,
        alpha=0.3,
        beta=1.0,
        gamma=None,
        max_iter=200,
        normalize_distances=True,
        learnable_alpha_beta=False,
        learnable_gamma=False,
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

        self.learnable_alpha_beta = learnable_alpha_beta
        self.learnable_gamma = learnable_gamma
        self._fixed_alpha = float(alpha)
        self._fixed_beta = float(beta)
        self._fixed_gamma = None if gamma is None else float(gamma)
        self.max_iter = max_iter
        self.normalize_distances = normalize_distances

        if learnable_alpha_beta:
            alpha_ratio = (alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)
            beta_ratio = (beta - self.beta_min) / (self.beta_max - self.beta_min)
            eps = torch.finfo(torch.float32).eps
            alpha_ratio = min(max(alpha_ratio, eps), 1.0 - eps)
            beta_ratio = min(max(beta_ratio, eps), 1.0 - eps)
            alpha_logit = math.log(alpha_ratio / (1.0 - alpha_ratio))
            beta_logit = math.log(beta_ratio / (1.0 - beta_ratio))
            self.alpha_logits = nn.Parameter(torch.full((max_iter,), alpha_logit))
            self.beta_logits = nn.Parameter(torch.full((max_iter,), beta_logit))

        if learnable_gamma:
            if gamma is None:
                k_neighbors = self.neighbor_list.size(1)
                gamma = 0.9 / (2.0 * beta + k_neighbors**0.5 + 1.0)
            gamma_raw = math.log(math.expm1(float(gamma)))
            self.gamma_raw = nn.Parameter(torch.full((max_iter,), gamma_raw))

    @property
    def alpha(self):
        if not self.learnable_alpha_beta:
            return self._fixed_alpha
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_logits)

    @property
    def beta(self):
        if not self.learnable_alpha_beta:
            return self._fixed_beta
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.beta_logits)

    @property
    def gamma(self):
        if not self.learnable_gamma:
            return self._fixed_gamma
        return F.softplus(self.gamma_raw)

    def _iteration_parameter(self, value, iteration):
        if not torch.is_tensor(value) or value.ndim == 0:
            return value
        return value[iteration]

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

        for iteration in range(self.max_iter):
            alpha = self._iteration_parameter(self.alpha, iteration)
            beta = self._iteration_parameter(self.beta, iteration)
            gamma_parameter = self._iteration_parameter(self.gamma, iteration)
            gamma = (
                0.9 / (2.0 * beta + z.size(-1) ** 0.5 + 1.0)
                if gamma_parameter is None
                else gamma_parameter
            )

            y = (W - gamma * (2.0 * beta * W + d.unsqueeze(-1))) * mask
            y_bar = d + gamma * W.sum(dim=-1)

            p = torch.relu(y - 2.0 * gamma * z) * mask
            p_bar = 0.5 * (
                y_bar
                - torch.sqrt(y_bar.square() + 4.0 * alpha * gamma)
            )

            q = (p - gamma * (2.0 * beta * p + p_bar.unsqueeze(-1))) * mask
            q_bar = p_bar + gamma * p.sum(dim=-1)

            W = (W - y + q) * mask
            d = d - y_bar + q_bar

        return torch.relu(W) * mask
