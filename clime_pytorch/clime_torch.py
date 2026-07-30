"""PyTorch CLIME approximation.

This file implements a GPU-capable split-ADMM solver for

    min ||Omega||_1
    s.t. ||S Omega - I||_max <= lambda.

It is not an exact LP solver.  It is intended as an experiment-friendly GPU
approximation that can be compared against ``clime_cpu``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import torch


@dataclass
class ClimeTorchResult:
    """Result returned by ``clime_torch(..., return_info=True)``."""

    precision: torch.Tensor
    raw_precision: torch.Tensor
    n_iter: int
    converged: bool
    primal_residual: float
    dual_residual: float
    objective: float
    residual_history: List[float]
    dual_residual_history: List[float]
    constraint_history: List[float]
    objective_history: List[float]
    nnz_ratio_history: List[float]
    lambda_: float
    rho: float
    eta: float
    symmetrized: bool
    linear_solver: str


def _soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Elementwise L1 proximal operator."""
    return torch.sign(x) * torch.clamp(x.abs() - threshold, min=0)


def _symmetrize_clime_torch(precision: torch.Tensor) -> torch.Tensor:
    """CLIME symmetrization: keep the smaller-magnitude directed estimate."""
    lower_or_equal = precision.abs() <= precision.transpose(-1, -2).abs()
    return torch.where(lower_or_equal, precision, precision.transpose(-1, -2))


def _validate_covariance(emp_cov: torch.Tensor) -> torch.Tensor:
    """Validate and symmetrize a torch covariance matrix."""
    if not torch.is_tensor(emp_cov):
        emp_cov = torch.as_tensor(emp_cov)
    if not torch.is_floating_point(emp_cov):
        emp_cov = emp_cov.to(torch.float32)
    if emp_cov.ndim != 2 or emp_cov.shape[0] != emp_cov.shape[1]:
        raise ValueError("emp_cov must have shape (p, p)")
    if not torch.isfinite(emp_cov).all():
        raise ValueError("emp_cov contains NaN or Inf")
    return 0.5 * (emp_cov + emp_cov.transpose(-1, -2))


def _factor_spd_system(lhs: torch.Tensor, linear_solver: str):
    """Precompute the constant linear-system factor used by every X-update."""
    if linear_solver == "cholesky":
        return "cholesky", torch.linalg.cholesky(lhs)
    if linear_solver == "solve":
        return "solve", lhs
    if linear_solver == "auto":
        try:
            return "cholesky", torch.linalg.cholesky(lhs)
        except RuntimeError:
            return "solve", lhs
    raise ValueError("linear_solver must be one of: 'auto', 'cholesky', 'solve'")


def _solve_factored_system(factor_kind: str, factor: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve ``lhs @ x = rhs`` from a precomputed factor."""
    if factor_kind == "cholesky":
        return torch.cholesky_solve(rhs, factor)
    return torch.linalg.solve(factor, rhs)


def clime_torch(
    emp_cov: torch.Tensor,
    lambda_: float = 0.01,
    *,
    rho: float = 1.0,
    eta: float = 10.0,
    max_iter: int = 500,
    tol: float = 1e-4,
    symmetrize: bool = True,
    linear_solver: str = "auto",
    return_info: bool = False,
) -> Union[torch.Tensor, ClimeTorchResult]:
    """Estimate a CLIME precision matrix with a GPU-capable ADMM approximation.

    Args:
        emp_cov: Empirical covariance matrix, shape ``(p, p)``. CUDA tensors
            stay on CUDA.
        lambda_: Infinity-norm feasibility radius.
        rho: ADMM penalty for ``S X - Y = I``.
        eta: ADMM penalty for ``X - Z = 0``. Larger values force the sparse
            variable ``Z`` to track ``X`` more aggressively.
        max_iter: Maximum ADMM iterations.
        tol: Stop when both primal and dual residuals are below this value.
        symmetrize: If True, apply the standard CLIME post-symmetrization rule.
        linear_solver: ``"auto"``, ``"cholesky"``, or ``"solve"`` for the
            closed-form X-update linear system.
        return_info: If True, return ``ClimeTorchResult``.

    Returns:
        Precision estimate as a torch tensor, or ``ClimeTorchResult`` when
        ``return_info=True``.
    """
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    if rho <= 0:
        raise ValueError("rho must be positive")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    cov = _validate_covariance(emp_cov)
    p = cov.shape[0]
    eye = torch.eye(p, dtype=cov.dtype, device=cov.device)
    cov_t = cov.transpose(-1, -2)
    lhs = rho * (cov_t @ cov) + eta * eye
    factor_kind, factor = _factor_spd_system(lhs, linear_solver)

    # ADMM variables:
    #   X is the dense precision estimate used by the linear constraint.
    #   Y is the constrained residual S X - I, projected into [-lambda, lambda].
    #   Z is the sparse precision estimate after the L1 proximal step.
    #   U is the scaled dual variable for S X - Y = I.
    #   V is the scaled dual variable for X - Z = 0.
    x = torch.zeros_like(cov)
    y = torch.zeros_like(cov)
    z = torch.zeros_like(cov)
    u = torch.zeros_like(cov)
    v = torch.zeros_like(cov)
    residual_history: List[float] = []
    dual_residual_history: List[float] = []
    constraint_history: List[float] = []
    objective_history: List[float] = []
    nnz_ratio_history: List[float] = []
    converged = False
    primal_residual = float("inf")
    dual_residual = float("inf")

    for iteration in range(1, max_iter + 1):
        y_old = y
        z_old = z

        # X-update:
        #   min_X rho/2 ||S X - I - Y + U||_F^2
        #       + eta/2 ||X - Z + V||_F^2
        #
        # This is a closed-form linear solve, shared by all columns:
        #   (rho S^T S + eta I) X
        #       = rho S^T(I + Y - U) + eta(Z - V).
        rhs = rho * cov_t @ (eye + y - u) + eta * (z - v)
        x = _solve_factored_system(factor_kind, factor, rhs)

        sx_minus_i = cov @ x - eye
        y = torch.clamp(sx_minus_i + u, min=-lambda_, max=lambda_)
        z = _soft_threshold(x + v, 1.0 / eta)
        u = u + sx_minus_i - y
        v = v + x - z

        primal_1 = sx_minus_i - y
        primal_2 = x - z
        dual = rho * cov_t @ (y - y_old) + eta * (z - z_old)
        primal_residual = max(
            float(primal_1.abs().max().detach().cpu()),
            float(primal_2.abs().max().detach().cpu()),
        )
        dual_residual = float(dual.abs().max().detach().cpu())
        constraint_max = float((cov @ z - eye).abs().max().detach().cpu())
        objective = float(z.abs().sum().detach().cpu())
        nnz_ratio = float((z.abs() > 1e-8).to(torch.float64).mean().detach().cpu())
        residual_history.append(primal_residual)
        dual_residual_history.append(dual_residual)
        constraint_history.append(constraint_max)
        objective_history.append(objective)
        nnz_ratio_history.append(nnz_ratio)
        if primal_residual <= tol and dual_residual <= tol:
            converged = True
            break

    raw_precision = z
    precision = _symmetrize_clime_torch(raw_precision) if symmetrize else raw_precision
    if return_info:
        return ClimeTorchResult(
            precision=precision,
            raw_precision=raw_precision,
            n_iter=iteration,
            converged=converged,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            objective=float(raw_precision.abs().sum().detach().cpu()),
            residual_history=residual_history,
            dual_residual_history=dual_residual_history,
            constraint_history=constraint_history,
            objective_history=objective_history,
            nnz_ratio_history=nnz_ratio_history,
            lambda_=float(lambda_),
            rho=float(rho),
            eta=float(eta),
            symmetrized=symmetrize,
            linear_solver=factor_kind,
        )
    return precision
