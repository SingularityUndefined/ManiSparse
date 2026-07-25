import torch
import torch.nn as nn

class CG_Solver(nn.Module):
    def __init__(self, max_iter=100, tol=1e-6):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, A_func, b, x0=None):
        # Solve the linear system Ax = b using the Conjugate Gradient method
        # A_func: function that computes the matrix-vector product Ax
        # b: (n_batchs, n_nodes,) right-hand side vector
        # x0: (n_batchs, n_nodes,) initial guess for the solution
        if not torch.is_tensor(b):
            b = torch.as_tensor(b, dtype=torch.get_default_dtype())
        elif not b.is_floating_point():
            b = b.to(dtype=torch.get_default_dtype())
        n_batchs, n_nodes = b.shape
        if x0 is None:
            x = torch.zeros((n_batchs, n_nodes), dtype=b.dtype, device=b.device)
        else:
            if not torch.is_tensor(x0):
                x0 = torch.as_tensor(x0, dtype=b.dtype, device=b.device)
            x = x0.to(dtype=b.dtype, device=b.device).clone()
        
        r = b - A_func(x)  # initial residual, shape (n_batchs, n_nodes)
        p = r.clone()
        rsold = (r * r).sum(dim=1) # shape (n_batchs)
        eps = torch.finfo(b.dtype).eps
        if torch.all(torch.sqrt(rsold) < self.tol):
            return x

        for i in range(self.max_iter):
            Ap = A_func(p)  # shape (n_batchs, n_nodes)
            denom = (p * Ap).sum(dim=1)
            valid_denom = torch.abs(denom) > eps
            denom_safe = torch.where(valid_denom, denom, torch.ones_like(denom))
            alpha = torch.where(valid_denom, rsold / denom_safe, torch.zeros_like(rsold))
            x = x + alpha[:, None] * p
            r = r - alpha[:, None] * Ap
            rsnew = (r * r).sum(dim=1)
            if torch.all(torch.sqrt(rsnew) < self.tol):
                break
            valid_rsold = rsold > eps
            rsold_safe = torch.where(valid_rsold, rsold, torch.ones_like(rsold))
            beta = torch.where(valid_rsold, rsnew / rsold_safe, torch.zeros_like(rsold))
            p = r + beta[:, None] * p
            rsold = rsnew
        
        return x


class KrylovSubspaceSolver(nn.Module):
    def __init__(self, max_dim=None, tol=1e-6, reorthogonalize=True):
        super().__init__()
        self.max_dim = max_dim
        self.tol = tol
        self.reorthogonalize = reorthogonalize

    def _as_float_tensor(self, x):
        if not torch.is_tensor(x):
            return torch.as_tensor(x, dtype=torch.get_default_dtype())
        if not x.is_floating_point():
            return x.to(dtype=torch.get_default_dtype())
        return x

    def project_orthogonal(self, v, basis, n_basis):
        if n_basis == 0:
            return v
        active_basis = basis[:, :n_basis, :]
        coeff = torch.einsum("bd,bkd->bk", v, active_basis)
        return v - torch.einsum("bk,bkd->bd", coeff, active_basis)

    def batch_matvec(self, matrix, vector):
        return torch.einsum("bij,bj->bi", matrix, vector)

    def coordinates(self, V, x):
        return torch.einsum("bmn,bn->bm", V, x)

    def reconstruct(self, V, coeff):
        return torch.einsum("bm,bmn->bn", coeff, V)

    def lanczos(self, A_func, b, max_dim=None):
        # Build an orthonormal Krylov basis for span{b, A b, ..., A^{m-1} b}
        # using only the linear operator A_func.
        b = self._as_float_tensor(b)
        n_batch, n_nodes = b.shape
        max_dim = self.max_dim if max_dim is None else max_dim
        if max_dim is None:
            max_dim = n_nodes
        max_dim = max(1, min(max_dim, n_nodes))

        eps = torch.finfo(b.dtype).eps
        beta0 = torch.linalg.norm(b, dim=1)
        active = beta0 > self.tol
        q = torch.where(
            active[:, None],
            b / beta0.clamp_min(eps)[:, None],
            torch.zeros_like(b),
        )
        q_prev = torch.zeros_like(q)
        beta_prev = torch.zeros(n_batch, dtype=b.dtype, device=b.device)

        basis = []
        alphas = []
        betas = []

        for step in range(max_dim):
            basis.append(q)
            w = A_func(q) - beta_prev[:, None] * q_prev
            alpha = (q * w).sum(dim=1)
            w = w - alpha[:, None] * q

            if self.reorthogonalize:
                V_step = torch.stack(basis, dim=1)
                w = self.project_orthogonal(w, V_step, step + 1)

            beta_next = torch.linalg.norm(w, dim=1)
            alphas.append(alpha)
            if step == max_dim - 1:
                break

            betas.append(beta_next)
            active = beta_next > self.tol
            q_prev = q
            q = torch.where(
                active[:, None],
                w / beta_next.clamp_min(eps)[:, None],
                torch.zeros_like(w),
            )
            beta_prev = beta_next

        V = torch.stack(basis, dim=1)
        alpha_diag = torch.stack(alphas, dim=1)
        n_krylov = V.shape[1]
        T = torch.zeros((n_batch, n_krylov, n_krylov), dtype=b.dtype, device=b.device)
        diag_idx = torch.arange(n_krylov, device=b.device)
        T[:, diag_idx, diag_idx] = alpha_diag

        if n_krylov > 1:
            beta_offdiag = torch.stack(betas[: n_krylov - 1], dim=1)
            offdiag_idx = torch.arange(n_krylov - 1, device=b.device)
            T[:, offdiag_idx, offdiag_idx + 1] = beta_offdiag
            T[:, offdiag_idx + 1, offdiag_idx] = beta_offdiag

        return V, T, beta0
