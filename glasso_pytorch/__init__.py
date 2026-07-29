from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


TensorLike = Union[float, torch.Tensor]


@dataclass
class GraphicalLassoResult:
    precision: torch.Tensor
    covariance: torch.Tensor
    n_iter: int
    converged: bool
    primal_residual: float
    dual_residual: float
    dual_gap: float


def _as_batch_matrix(emp_cov: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if emp_cov.ndim == 2:
        return emp_cov.unsqueeze(0), True
    if emp_cov.ndim >= 3:
        return emp_cov, False
    raise ValueError("emp_cov must have shape (n_features, n_features) or (..., n_features, n_features)")


def _restore_batch(matrix: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return matrix.squeeze(0) if squeezed else matrix


def _offdiag_mask(n_features: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return ~torch.eye(n_features, device=device, dtype=torch.bool)


def _soft_threshold(x: torch.Tensor, threshold: TensorLike) -> torch.Tensor:
    threshold = torch.as_tensor(threshold, dtype=x.dtype, device=x.device)
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)


def _regularization_tensor(alpha: TensorLike, ref: torch.Tensor) -> torch.Tensor:
    alpha = torch.as_tensor(alpha, dtype=ref.dtype, device=ref.device)
    if torch.any(alpha < 0):
        raise ValueError("alpha must be non-negative")
    return alpha


def _matrix_eye_like(matrix: torch.Tensor) -> torch.Tensor:
    """Return an identity matrix broadcastable to a batch of square matrices."""
    n_features = matrix.shape[-1]
    eye = torch.eye(n_features, dtype=matrix.dtype, device=matrix.device)
    return eye.expand(matrix.shape)


def _eigh_with_shift_recovery(
    matrix: torch.Tensor,
    *,
    initial_shift: float,
    shift_retries: int,
    cpu_fallback: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run symmetric eigendecomposition with spectrum-preserving fallbacks.

    The shift fallback uses eigh(A + delta I), then subtracts delta from the
    returned eigenvalues. It preserves the original eigensystem in exact
    arithmetic and does not change the ADMM graphical-lasso objective.
    """
    try:
        return torch.linalg.eigh(matrix)
    except RuntimeError as direct_error:
        last_error = direct_error

    # CPU fallback necessarily copies the matrix and is therefore only used
    # when gradients are not being tracked through the eigensolve.
    can_cpu_fallback = cpu_fallback and not matrix.requires_grad

    if can_cpu_fallback and (matrix.device.type != "cpu" or matrix.dtype != torch.float64):
        try:
            cpu_matrix = matrix.detach().to(device="cpu", dtype=torch.float64)
            eigvals, eigvecs = torch.linalg.eigh(cpu_matrix)
            return eigvals.to(device=matrix.device, dtype=matrix.dtype), eigvecs.to(device=matrix.device, dtype=matrix.dtype)
        except RuntimeError as error:
            last_error = error

    if initial_shift > 0:
        eye = _matrix_eye_like(matrix)
        for retry_idx in range(shift_retries + 1):
            shift = initial_shift * (10.0**retry_idx)
            shifted = matrix + shift * eye
            try:
                eigvals, eigvecs = torch.linalg.eigh(shifted)
                return eigvals - shift, eigvecs
            except RuntimeError as error:
                last_error = error

            if can_cpu_fallback and (shifted.device.type != "cpu" or shifted.dtype != torch.float64):
                try:
                    cpu_shifted = shifted.detach().to(device="cpu", dtype=torch.float64)
                    eigvals, eigvecs = torch.linalg.eigh(cpu_shifted)
                    eigvals = eigvals.to(device=matrix.device, dtype=matrix.dtype) - shift
                    eigvecs = eigvecs.to(device=matrix.device, dtype=matrix.dtype)
                    return eigvals, eigvecs
                except RuntimeError as error:
                    last_error = error

    raise RuntimeError(
        "torch.linalg.eigh failed for the original matrix and all "
        "spectrum-preserving fallbacks"
    ) from last_error


def _dual_gap(
    emp_cov: torch.Tensor,
    precision: torch.Tensor,
    alpha: TensorLike,
    *,
    penalize_diagonal: bool = False,
) -> torch.Tensor:
    alpha = torch.as_tensor(alpha, dtype=precision.dtype, device=precision.device)
    gap = (emp_cov * precision).sum(dim=(-2, -1)) - precision.shape[-1]
    penalty = precision.abs().sum(dim=(-2, -1))
    if not penalize_diagonal:
        penalty = penalty - precision.diagonal(dim1=-2, dim2=-1).abs().sum(dim=-1)
    return gap + alpha * penalty


def empirical_covariance_from_samples(
    samples: torch.Tensor,
    *,
    center: bool = False,
    unbiased: bool = True,
) -> torch.Tensor:
    """Build empirical covariance matrices from raw samples.

    Args:
        samples: Raw samples with shape ``(batch, n_features)`` or
            ``(batch, n_samples, n_features)``.
        center: If True, subtracts the feature mean before forming covariance.
            For ``(batch, n_features)``, this subtracts each row's mean.
        unbiased: If True, divides by ``n_samples - 1`` for 3D inputs. For 2D
            inputs there is only one sample per batch, so the denominator is 1.

    Returns:
        Batched covariance matrices with shape ``(batch, n_features, n_features)``.
    """
    if not torch.is_floating_point(samples):
        raise TypeError("samples must be a floating point tensor")
    if samples.ndim == 2:
        if center:
            samples = samples - samples.mean(dim=-1, keepdim=True)
        return samples.unsqueeze(-1) @ samples.unsqueeze(-2)
    if samples.ndim == 3:
        n_samples = samples.shape[-2]
        if n_samples < 2:
            raise ValueError("3D samples require at least two samples per batch")
        if center:
            samples = samples - samples.mean(dim=-2, keepdim=True)
        denominator = n_samples - 1 if unbiased else n_samples
        return samples.transpose(-1, -2) @ samples / denominator
    raise ValueError("samples must have shape (batch, n_features) or (batch, n_samples, n_features)")


def graphical_lasso(
    emp_cov: torch.Tensor,
    alpha: TensorLike = 0.01,
    *,
    rho: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    rtol: float = 1e-4,
    eps: float = 0.0,
    eigh_shift: float = 1e-6,
    eigh_shift_retries: int = 4,
    eigh_cpu_fallback: bool = True,
    penalize_diagonal: bool = False,
    return_info: bool = False,
) -> Union[torch.Tensor, GraphicalLassoResult]:
    """Estimate a sparse precision matrix from an empirical covariance matrix.

    This solves the graphical lasso problem with ADMM using only PyTorch ops:

        minimize  trace(S @ Theta) - logdet(Theta) + alpha * ||Theta||_1
        subject to Theta positive definite

    By default the diagonal is not L1-penalized, matching the common graphical
    lasso convention used by packages such as scikit-learn.

    Args:
        emp_cov: Empirical covariance matrix with shape ``(p, p)`` or a batch
            of covariance matrices with shape ``(..., p, p)``.
        alpha: L1 regularization strength. Larger values produce sparser
            off-diagonal entries. Can be a scalar tensor on the same device.
        rho: ADMM penalty parameter.
        max_iter: Maximum ADMM iterations.
        tol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        eps: Small diagonal jitter added to ``emp_cov``. This changes the
            graphical-lasso problem and should be used only when desired.
        eigh_shift: Initial eigensolver-only spectral shift. If direct
            ``eigh(A)`` fails, the solver tries ``eigh(A + delta I)`` and then
            subtracts ``delta`` from the returned eigenvalues.
        eigh_shift_retries: Number of additional 10x shift retries.
        eigh_cpu_fallback: If True, retry eigendecomposition on CPU float64
            before/while applying spectral shifts.
        penalize_diagonal: If True, also applies L1 penalty to the diagonal.
        return_info: If True, returns a ``GraphicalLassoResult`` with residuals
            and convergence metadata. Otherwise returns only the precision.

    Returns:
        The sparse precision matrix with the same batch shape, dtype, and device
        as ``emp_cov``. If ``return_info`` is True, returns a
        ``GraphicalLassoResult`` instead.
    """
    if not torch.is_floating_point(emp_cov):
        raise TypeError("emp_cov must be a floating point tensor")
    if emp_cov.shape[-1] != emp_cov.shape[-2]:
        raise ValueError("emp_cov must be square in its last two dimensions")
    if rho <= 0:
        raise ValueError("rho must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if eigh_shift < 0:
        raise ValueError("eigh_shift must be non-negative")
    if eigh_shift_retries < 0:
        raise ValueError("eigh_shift_retries must be non-negative")

    cov, squeezed = _as_batch_matrix(emp_cov)
    n_features = cov.shape[-1]
    alpha_t = _regularization_tensor(alpha, cov)
    eye = _matrix_eye_like(cov)

    # Symmetrize the covariance to avoid tiny asymmetric numerical noise causing
    # complex behavior in the eigensolver. ADMM keeps all iterates symmetric.
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    if eps > 0:
        cov = cov + eps * eye

    theta = torch.linalg.inv(cov + alpha_t * eye)
    theta = 0.5 * (theta + theta.transpose(-1, -2))
    z = theta.clone()
    u = torch.zeros_like(theta)

    offdiag = _offdiag_mask(n_features, device=cov.device, dtype=cov.dtype)
    converged = False
    primal_residual = float("inf")
    dual_residual = float("inf")

    for n_iter in range(1, max_iter + 1):
        z_old = z.clone()

        # Theta update:
        # argmin trace(S Theta) - logdet(Theta) + rho/2 ||Theta - Z + U||_F^2.
        # If Q diag(d) Q^T = rho * (Z - U) - S, the closed-form eigenvalues are
        # (d + sqrt(d^2 + 4 rho)) / (2 rho).
        eig_input = rho * (z - u) - cov
        eig_input = 0.5 * (eig_input + eig_input.transpose(-1, -2))
        eigvals, eigvecs = _eigh_with_shift_recovery(
            eig_input,
            initial_shift=eigh_shift,
            shift_retries=eigh_shift_retries,
            cpu_fallback=eigh_cpu_fallback,
        )
        theta_eigvals = (eigvals + torch.sqrt(eigvals.square() + 4.0 * rho)) / (2.0 * rho)
        theta = (eigvecs * theta_eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        theta = 0.5 * (theta + theta.transpose(-1, -2))

        theta_plus_u = theta + u
        if penalize_diagonal:
            z = _soft_threshold(theta_plus_u, alpha_t / rho)
        else:
            z = theta_plus_u.clone()
            z[..., offdiag] = _soft_threshold(z[..., offdiag], alpha_t / rho)
        z = 0.5 * (z + z.transpose(-1, -2))

        u = u + theta - z

        primal = theta - z
        dual = rho * (z - z_old)
        primal_residual_t = torch.linalg.matrix_norm(primal, ord="fro", dim=(-2, -1))
        dual_residual_t = torch.linalg.matrix_norm(dual, ord="fro", dim=(-2, -1))

        theta_norm = torch.linalg.matrix_norm(theta, ord="fro", dim=(-2, -1))
        z_norm = torch.linalg.matrix_norm(z, ord="fro", dim=(-2, -1))
        u_norm = torch.linalg.matrix_norm(rho * u, ord="fro", dim=(-2, -1))
        abs_scale = n_features * tol
        primal_tol = abs_scale + rtol * torch.maximum(theta_norm, z_norm)
        dual_tol = abs_scale + rtol * u_norm

        primal_residual = float(primal_residual_t.max().detach().cpu())
        dual_residual = float(dual_residual_t.max().detach().cpu())
        if torch.all(primal_residual_t <= primal_tol) and torch.all(dual_residual_t <= dual_tol):
            converged = True
            break

    precision = _restore_batch(z, squeezed)

    if return_info:
        covariance = torch.linalg.inv(0.5 * (z + z.transpose(-1, -2)))
        covariance = _restore_batch(0.5 * (covariance + covariance.transpose(-1, -2)), squeezed)
        dual_gap = _dual_gap(cov, z, alpha_t, penalize_diagonal=penalize_diagonal)
        return GraphicalLassoResult(
            precision=precision,
            covariance=covariance,
            n_iter=n_iter,
            converged=converged,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            dual_gap=float(dual_gap.mean().detach().cpu()),
        )
    return precision


class GraphicalLassoModule(nn.Module):
    """Parameter-free ``nn.Module`` wrapper for PyTorch graphical lasso.

    This module is intended for inserting the solver into a larger PyTorch
    network. Set ``allow_backward=False`` to use it as a GPU-capable algorithmic
    layer without retaining gradients through the ADMM iterations.

    By default, the module accepts empirical covariance matrices:

    * ``(batch, n_features, n_features)`` -> ``(batch, n_features, n_features)``

    Set ``input_mode="samples"`` to pass raw samples instead.
    """

    def __init__(
        self,
        alpha: TensorLike = 0.01,
        *,
        rho: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        rtol: float = 1e-4,
        eps: float = 0.0,
        eigh_shift: float = 1e-6,
        eigh_shift_retries: int = 4,
        eigh_cpu_fallback: bool = True,
        penalize_diagonal: bool = False,
        allow_backward: bool = False,
        input_mode: str = "covariance",
        center: bool = False,
        unbiased: bool = True,
    ) -> None:
        super().__init__()
        input_mode = input_mode.lower()
        if input_mode not in {"samples", "covariance"}:
            raise ValueError("input_mode must be either 'samples' or 'covariance'")
        self.alpha = alpha
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.rtol = rtol
        self.eps = eps
        self.eigh_shift = eigh_shift
        self.eigh_shift_retries = eigh_shift_retries
        self.eigh_cpu_fallback = eigh_cpu_fallback
        self.penalize_diagonal = penalize_diagonal
        self.allow_backward = allow_backward
        self.input_mode = input_mode
        self.center = center
        self.unbiased = unbiased

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, rho={self.rho}, max_iter={self.max_iter}, "
            f"tol={self.tol}, rtol={self.rtol}, eps={self.eps}, "
            f"eigh_shift={self.eigh_shift}, "
            f"eigh_shift_retries={self.eigh_shift_retries}, "
            f"eigh_cpu_fallback={self.eigh_cpu_fallback}, "
            f"penalize_diagonal={self.penalize_diagonal}, "
            f"allow_backward={self.allow_backward}, input_mode={self.input_mode}, "
            f"center={self.center}, unbiased={self.unbiased}"
        )

    def _prepare_covariance(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "covariance":
            return x
        return empirical_covariance_from_samples(x, center=self.center, unbiased=self.unbiased)

    def solve(self, x: torch.Tensor, *, return_info: bool = False) -> Union[torch.Tensor, GraphicalLassoResult]:
        if self.allow_backward:
            emp_cov = self._prepare_covariance(x)
            return graphical_lasso(
                emp_cov,
                alpha=self.alpha,
                rho=self.rho,
                max_iter=self.max_iter,
                tol=self.tol,
                rtol=self.rtol,
                eps=self.eps,
                eigh_shift=self.eigh_shift,
                eigh_shift_retries=self.eigh_shift_retries,
                eigh_cpu_fallback=self.eigh_cpu_fallback,
                penalize_diagonal=self.penalize_diagonal,
                return_info=return_info,
            )

        with torch.no_grad():
            emp_cov = self._prepare_covariance(x)
            result = graphical_lasso(
                emp_cov,
                alpha=self.alpha,
                rho=self.rho,
                max_iter=self.max_iter,
                tol=self.tol,
                rtol=self.rtol,
                eps=self.eps,
                eigh_shift=self.eigh_shift,
                eigh_shift_retries=self.eigh_shift_retries,
                eigh_cpu_fallback=self.eigh_cpu_fallback,
                penalize_diagonal=self.penalize_diagonal,
                return_info=return_info,
            )
        if return_info:
            result.precision = result.precision.detach()
            result.covariance = result.covariance.detach()
            return result
        return result.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.solve(x)


class GraphicalLasso:
    """Small sklearn-like wrapper around ``graphical_lasso``.

    The ``fit`` method accepts an empirical covariance matrix directly and sets
    ``precision_`` to the estimated sparse precision matrix.
    """

    def __init__(
        self,
        alpha: TensorLike = 0.01,
        *,
        rho: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        rtol: float = 1e-4,
        eps: float = 0.0,
        eigh_shift: float = 1e-6,
        eigh_shift_retries: int = 4,
        eigh_cpu_fallback: bool = True,
        penalize_diagonal: bool = False,
    ) -> None:
        self.alpha = alpha
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.rtol = rtol
        self.eps = eps
        self.eigh_shift = eigh_shift
        self.eigh_shift_retries = eigh_shift_retries
        self.eigh_cpu_fallback = eigh_cpu_fallback
        self.penalize_diagonal = penalize_diagonal

        self.precision_: Optional[torch.Tensor] = None
        self.covariance_: Optional[torch.Tensor] = None
        self.n_iter_: Optional[int] = None
        self.converged_: Optional[bool] = None
        self.primal_residual_: Optional[float] = None
        self.dual_residual_: Optional[float] = None
        self.dual_gap_: Optional[float] = None

    def fit(self, emp_cov: torch.Tensor) -> "GraphicalLasso":
        result = graphical_lasso(
            emp_cov,
            alpha=self.alpha,
            rho=self.rho,
            max_iter=self.max_iter,
            tol=self.tol,
            rtol=self.rtol,
            eps=self.eps,
            eigh_shift=self.eigh_shift,
            eigh_shift_retries=self.eigh_shift_retries,
            eigh_cpu_fallback=self.eigh_cpu_fallback,
            penalize_diagonal=self.penalize_diagonal,
            return_info=True,
        )
        self.precision_ = result.precision
        self.covariance_ = result.covariance
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        self.primal_residual_ = result.primal_residual
        self.dual_residual_ = result.dual_residual
        self.dual_gap_ = result.dual_gap
        return self


__all__ = [
    "GraphicalLasso",
    "GraphicalLassoModule",
    "GraphicalLassoResult",
    "empirical_covariance_from_samples",
    "graphical_lasso",
]
