"""CPU CLIME solver using column-wise linear programming.

CLIME estimates a sparse precision matrix by solving

    min ||Omega||_1
    s.t. ||S Omega - I||_max <= lambda

which separates into one linear program per precision-matrix column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ClimeCPUResult:
    """Result returned by ``clime_cpu(..., return_info=True)``."""

    precision: np.ndarray
    raw_precision: np.ndarray
    statuses: List[int]
    messages: List[str]
    objectives: List[float]
    lambda_: float
    symmetrized: bool


def _symmetrize_clime(precision: np.ndarray) -> np.ndarray:
    """CLIME symmetrization: keep the smaller-magnitude directed estimate."""
    p = precision.shape[0]
    out = precision.copy()
    for i in range(p):
        for j in range(i + 1, p):
            value = precision[i, j] if abs(precision[i, j]) <= abs(precision[j, i]) else precision[j, i]
            out[i, j] = value
            out[j, i] = value
    return out


def _as_symmetric_numpy(emp_cov) -> np.ndarray:
    """Convert input covariance to a finite symmetric float64 numpy matrix."""
    if hasattr(emp_cov, "detach"):
        emp_cov = emp_cov.detach().cpu().numpy()
    cov = np.asarray(emp_cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("emp_cov must have shape (p, p)")
    if not np.isfinite(cov).all():
        raise ValueError("emp_cov contains NaN or Inf")
    return 0.5 * (cov + cov.T)


def _solve_column_lp(
    cov: np.ndarray,
    column_idx: int,
    lambda_: float,
    linprog_options: Optional[Dict[str, object]],
):
    """Solve one CLIME column LP after beta = u - v, u/v >= 0."""
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise ImportError("clime_cpu requires scipy.optimize.linprog") from exc

    p = cov.shape[0]
    eye_col = np.zeros(p, dtype=np.float64)
    eye_col[column_idx] = 1.0

    objective = np.ones(2 * p, dtype=np.float64)
    signed_cov = np.concatenate([cov, -cov], axis=1)
    constraints = np.concatenate([signed_cov, -signed_cov], axis=0)

    # S (u - v) - e_j <= lambda  ->  S (u - v) <= lambda + e_j
    # e_j - S (u - v) <= lambda  -> -S (u - v) <= lambda - e_j
    bounds_rhs = np.concatenate(
        [
            lambda_ + eye_col,
            lambda_ - eye_col,
        ],
        axis=0,
    )

    result = linprog(
        c=objective,
        A_ub=constraints,
        b_ub=bounds_rhs,
        bounds=[(0.0, None)] * (2 * p),
        method="highs",
        options=linprog_options,
    )
    if not result.success:
        raise RuntimeError(
            f"CLIME LP failed for column {column_idx}: status={result.status}, message={result.message}"
        )
    beta = result.x[:p] - result.x[p:]
    return beta, result


def clime_cpu(
    emp_cov,
    lambda_: float = 0.01,
    *,
    symmetrize: bool = True,
    linprog_options: Optional[Dict[str, object]] = None,
    return_info: bool = False,
):
    """Estimate a precision matrix with the exact CPU CLIME linear programs.

    Args:
        emp_cov: Empirical covariance matrix, shape ``(p, p)``.
        lambda_: Infinity-norm feasibility radius.
        symmetrize: If True, apply the standard CLIME post-symmetrization rule.
        linprog_options: Optional options passed to ``scipy.optimize.linprog``.
        return_info: If True, return ``ClimeCPUResult``. Otherwise return only
            the precision estimate as a numpy array.
    """
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")

    cov = _as_symmetric_numpy(emp_cov)
    p = cov.shape[0]
    raw_precision = np.zeros((p, p), dtype=np.float64)
    statuses: List[int] = []
    messages: List[str] = []
    objectives: List[float] = []

    for column_idx in range(p):
        beta, result = _solve_column_lp(cov, column_idx, lambda_, linprog_options)
        raw_precision[:, column_idx] = beta
        statuses.append(int(result.status))
        messages.append(str(result.message))
        objectives.append(float(result.fun))

    precision = _symmetrize_clime(raw_precision) if symmetrize else raw_precision.copy()
    if return_info:
        return ClimeCPUResult(
            precision=precision,
            raw_precision=raw_precision,
            statuses=statuses,
            messages=messages,
            objectives=objectives,
            lambda_=float(lambda_),
            symmetrized=symmetrize,
        )
    return precision
