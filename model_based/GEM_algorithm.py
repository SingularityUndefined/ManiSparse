import torch
import numpy as np
import torch.nn as nn

from glasso_pytorch import graphical_lasso as torch_graphical_lasso

try:
    from .utils import CG_Solver, KrylovSubspaceSolver
except ImportError:
    from utils import CG_Solver, KrylovSubspaceSolver


class GEM(nn.Module):
    # NOTE: This is a non-parametric implementation for now
    def __init__(
        self,
        n_nodes,
        alpha=0.05,
        max_iter=20,
        lambda_L=0.5,
        lambda_theta=0.5,
        deflation_samples=None,
        sparse_version=False,
        neighbor_list=None,
        glasso_solver="quic",
        glasso_tol=1e-4,
        glasso_rtol=1e-4,
        glasso_rho=1.0,
        deflation_method="cg",
        krylov_dim=None,
        krylov_tol=1e-6,
        krylov_reorthogonalize=True,
        krylov_small_max_iter=None,
    ):
        super(GEM, self).__init__()
        self.n_nodes = n_nodes
        if deflation_samples is None:
            self.deflation_samples = n_nodes - 1
        else:
            self.deflation_samples = deflation_samples
        self.alpha = alpha
        self.max_iter = max_iter # maximum number of iterations for the graphical lasso optimization
        
        self.lambda_L = lambda_L # regularization parameter for the graph Laplacian
        self.lambda_theta = lambda_theta # regularization parameter for the precision matrix

        self.glasso_solver = glasso_solver.lower()
        if self.glasso_solver not in {"quic", "sklearn", "admm"}:
            raise ValueError("glasso_solver must be one of: 'quic', 'sklearn', 'admm'")
        self.glasso_tol = glasso_tol
        self.glasso_rtol = glasso_rtol
        self.glasso_rho = glasso_rho
        self.deflation_method = deflation_method.lower()
        if self.deflation_method not in {"cg", "krylov"}:
            raise ValueError("deflation_method must be one of: 'cg', 'krylov'")
        self.krylov_dim = krylov_dim
        self.graphical_lasso = None
        if self.glasso_solver == "quic":
            from inverse_covariance import QuicGraphicalLasso

            self.graphical_lasso = QuicGraphicalLasso(
                lam=self.alpha,
                max_iter=self.max_iter,
                tol=self.glasso_tol,
                init_method="cov",
                auto_scale=False,
            )
        self.sparse_version = sparse_version
        self.cg_solver = CG_Solver(max_iter=100, tol=1e-6)
        
        self.deflation_solver_list = nn.ModuleList([CG_Solver(max_iter=10, tol=1e-6) for _ in range(self.deflation_samples)])
        self.krylov_solver = KrylovSubspaceSolver(
            max_dim=krylov_dim,
            tol=krylov_tol,
            reorthogonalize=krylov_reorthogonalize,
        )
        if krylov_small_max_iter is None:
            krylov_small_max_iter = max(20, 2 * self.deflation_samples)
        self.krylov_small_solver = CG_Solver(max_iter=krylov_small_max_iter, tol=krylov_tol)
        self.neighbor_list = neighbor_list

    def compute_covariance(self, x):
        # compute the covariance matrix of the input data x
        # x shape: (n_batch, n_samples, n_nodes)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        N_samples = x.size(1)
        if N_samples < 2:
            raise ValueError("at least two samples are required to compute covariance")
        x_mean = x.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        covariance = torch.matmul(x_centered.transpose(-1, -2), x_centered) / (N_samples - 1) # shape (n_batch, n_nodes, n_nodes), unbiased estimator
        return covariance

    def _to_numpy(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _to_tensor(self, value, like=None):
        if torch.is_tensor(value):
            tensor = value
        else:
            kwargs = {}
            if like is not None:
                kwargs["device"] = like.device
            tensor = torch.as_tensor(value, **kwargs)

        if like is not None:
            return tensor.to(device=like.device, dtype=like.dtype)
        if not tensor.is_floating_point():
            return tensor.to(dtype=torch.get_default_dtype())
        return tensor

    def _right_linear_operator(self, x, operator):
        if callable(operator):
            return operator(x)

        x = self._to_tensor(x)
        operator = self._to_tensor(operator, like=x)
        if operator.ndim == 2:
            return torch.matmul(x, operator)
        if operator.ndim == 3:
            if x.ndim != 2 or x.shape[0] != operator.shape[0]:
                raise ValueError("batched operator must match x shape (n_batch, n_nodes)")
            return torch.einsum("bi,bij->bj", x, operator)
        raise ValueError("operator must be callable, 2-D, or 3-D")

    def apply_L(self, x, W=None):
        # apply the graph Laplacian L to the input data x
        # NOTE: here only compute batch-level operation. If samples requried, please multiply the first 2 dimensions first.
        # x shape: (n_batch, n_nodes)
        # W shape: (n_nodes, n_neighbors) for sparse graph, (n_nodes, n_nodes) for dense graph
        x = self._to_tensor(x)
        # k = self.neighbor_list.size(-1) if self.sparse_version else self.n_nodes
        if not self.sparse_version:
            assert W is not None, "W must be provided for the dense version of the GEM algorithm"
            W = self._to_tensor(W, like=x)
            if W.ndim == 2:
                assert torch.allclose(W, W.T), "W must be symmetric for the dense version of the GEM algorithm"
                assert torch.all(W >= 0), "W must be non-negative for the dense version of the GEM algorithm"
                degree = W.sum(dim=1).reshape(1, -1) # shape (1, n_nodes)
            elif W.ndim == 3:
                assert torch.allclose(W, W.transpose(-1, -2)), "W must be symmetric for the dense version of the GEM algorithm"
                assert torch.all(W >= 0), "W must be non-negative for the dense version of the GEM algorithm"
                degree = W.sum(dim=2) # shape (n_batch, n_nodes)
            else:
                raise ValueError("dense W must have shape (n_nodes, n_nodes) or (n_batch, n_nodes, n_nodes)")
            # apply the transformation W to the input data x
            return degree * x - self._right_linear_operator(x, W) # shape (n_batch, n_nodes)
        else: # sparse version
            W = self._to_tensor(W, like=x)
            neighbor_list = self._to_tensor(self.neighbor_list).to(device=x.device, dtype=torch.long)
            assert W.shape[-2] == self.n_nodes, "W must have the same number of rows as the number of nodes"
            assert W.shape[-1] == neighbor_list.shape[-1], "W must have the same number of columns as the number of neighbors"
            neighbor_x = x[:, neighbor_list] # shape (n_batch, n_nodes, n_neighbors)
            edge_weight = W if W.ndim == 3 else W.reshape(1, self.n_nodes, -1)
            delta_x = edge_weight * (x[:, :, None] - neighbor_x) # shape (n_batch, n_nodes, n_neighbors)
            delta_x = delta_x.sum(dim=-1) # shape (n_batch, n_nodes)
            return delta_x

    def apply_Theta(self, x, Theta):
        # apply the precision matrix Theta to the input data x
        # x shape: (n_batch, n_nodes)
        # Theta shape: (n_nodes, n_nodes)
        return self._right_linear_operator(x, Theta) # shape (n_batch, n_nodes)

    def E_step_LHS(self, x, W, Theta):
        # compute the left-hand side of the E-step optimization problem
        # LHS: (I + lambda_L * L + lambda_theta * Theta) * x
        # x shape: (n_batch, n_nodes)
        return x + self.lambda_L * self.apply_L(x, W) + self.lambda_theta * self.apply_Theta(x, Theta) # shape (n_batch, n_nodes)

    def E_step(self, y, W, Theta):
        # solve the optimization problem to find the optimal solutions x
        # (I + lambda_L * L + lambda_theta * Theta) * x = y
        y = self._to_tensor(y)

        def A_func(x):
            return self.E_step_LHS(x, W, Theta)

        return self.cg_solver.solve(A_func, y) # shape (n_batch, n_nodes)

    def _basis_projection(self, v, q):
        return (v * q).sum(dim=1, keepdim=True) * q

    def _remove_basis_projection(self, v, q):
        return v - self._basis_projection(v, q)

    def _project_orthogonal(self, v, orth_basis, n_basis):
        if n_basis == 0:
            return v
        basis = orth_basis[:, :n_basis, :]
        coeff = torch.einsum("bn,bkn->bk", v, basis)
        return v - torch.einsum("bk,bkn->bn", coeff, basis)

    def _normalize_basis(self, x):
        norm = torch.linalg.norm(x, dim=1, keepdim=True)
        return torch.where(norm > 0, x / norm.clamp_min(torch.finfo(x.dtype).eps), torch.zeros_like(x))

    def _validate_multisignal_inputs(self, y, e_step_x):
        y = self._to_tensor(y)
        e_step_x = self._to_tensor(e_step_x, like=y)
        if y.ndim != 2 or e_step_x.ndim != 2:
            raise ValueError("MultiSignal expects y and e_step_x with shape (n_batch, n_nodes)")

        if e_step_x.shape != y.shape:
            raise ValueError("e_step_x must have the same shape as y")
        return y, e_step_x

    def _initial_multisignal_state(self, y, e_step_x):
        n_batch, n_nodes = y.shape

        n_modes = self.deflation_samples
        multi_x = torch.zeros((n_batch, n_modes, n_nodes), dtype=y.dtype, device=y.device)
        if n_modes == 0:
            return multi_x, None, None

        first_x = e_step_x.clone()
        multi_x[:, 0, :] = first_x
        first_q = self._normalize_basis(first_x)
        orth_basis = torch.zeros((n_batch, n_modes, n_nodes), dtype=y.dtype, device=y.device)
        orth_basis[:, 0, :] = first_q
        projected_y = self._remove_basis_projection(y, first_q)
        return multi_x, orth_basis, projected_y

    def _MultiSignal_cg(self, y, e_step_x, L, Theta):
        multi_x, orth_basis, projected_y = self._initial_multisignal_state(y, e_step_x)
        if self.deflation_samples == 0:
            return multi_x

        for mode_idx in range(1, self.deflation_samples):
            def projected_A_func(v, basis=orth_basis, n_basis=mode_idx):
                pv = self._project_orthogonal(v, basis, n_basis)
                return self._project_orthogonal(self.E_step_LHS(pv, L, Theta), basis, n_basis)

            z0 = torch.zeros_like(projected_y)
            z = self.deflation_solver_list[mode_idx - 1].solve(projected_A_func, projected_y, x0=z0)
            x_k = self._project_orthogonal(z, orth_basis, mode_idx)

            multi_x[:, mode_idx, :] = x_k
            q_k = self._normalize_basis(x_k)
            orth_basis[:, mode_idx, :] = q_k
            projected_y = self._remove_basis_projection(projected_y, q_k)

        return multi_x # shape (n_batch, deflation_samples, n_nodes)

    def _project_reduced_orthogonal(self, v, reduced_basis, n_basis):
        return self.krylov_solver.project_orthogonal(v, reduced_basis, n_basis)

    def _normalize_reduced_basis(self, h):
        norm = torch.linalg.norm(h, dim=1, keepdim=True)
        return torch.where(norm > 0, h / norm.clamp_min(torch.finfo(h.dtype).eps), torch.zeros_like(h))

    def _MultiSignal_krylov(self, y, e_step_x, L, Theta, krylov_dim=None):
        multi_x, orth_basis, _ = self._initial_multisignal_state(y, e_step_x)
        if self.deflation_samples == 0:
            return multi_x

        def A_func(v):
            return self.E_step_LHS(v, L, Theta)

        n_batch = y.shape[0]
        n_modes = self.deflation_samples
        if krylov_dim is None:
            krylov_dim = self.krylov_dim
        if krylov_dim is None:
            krylov_dim = min(self.n_nodes, max(n_modes + 5, 2 * n_modes))

        V, T, beta = self.krylov_solver.lanczos(A_func, y, max_dim=krylov_dim)
        n_krylov = V.shape[1]
        reduced_basis = torch.zeros((n_batch, n_modes, n_krylov), dtype=y.dtype, device=y.device)
        reduced_rhs = torch.zeros((n_batch, n_krylov), dtype=y.dtype, device=y.device)
        reduced_rhs[:, 0] = beta

        first_h = self.krylov_solver.coordinates(V, orth_basis[:, 0, :])
        first_h = self._normalize_reduced_basis(first_h)
        reduced_basis[:, 0, :] = first_h
        projected_rhs = self._remove_basis_projection(reduced_rhs, first_h)

        for mode_idx in range(1, n_modes):
            def reduced_projected_A(v, basis=reduced_basis, n_basis=mode_idx):
                pv = self._project_reduced_orthogonal(v, basis, n_basis)
                Apv = self.krylov_solver.batch_matvec(T, pv)
                return self._project_reduced_orthogonal(Apv, basis, n_basis)

            r0 = torch.zeros_like(projected_rhs)
            r = self.krylov_small_solver.solve(reduced_projected_A, projected_rhs, x0=r0)
            s_k = self._project_reduced_orthogonal(r, reduced_basis, mode_idx)
            x_k = self.krylov_solver.reconstruct(V, s_k)
            x_k = self._project_orthogonal(x_k, orth_basis, mode_idx)

            multi_x[:, mode_idx, :] = x_k
            q_k = self._normalize_basis(x_k)
            orth_basis[:, mode_idx, :] = q_k

            h_k = self.krylov_solver.coordinates(V, q_k)
            h_k = self._project_reduced_orthogonal(h_k, reduced_basis, mode_idx)
            h_k = self._normalize_reduced_basis(h_k)
            reduced_basis[:, mode_idx, :] = h_k
            projected_rhs = self._remove_basis_projection(projected_rhs, h_k)

        return multi_x

    def MultiSignal(self, y, e_step_x, L, Theta, deflation_method=None, krylov_dim=None):
        # Generate K deflated modes following deflation.tex.
        # The first mode is supplied by E_step; later modes use either projected
        # CG in the original space or an approximate Krylov reduced solve.
        y, e_step_x = self._validate_multisignal_inputs(y, e_step_x)
        if deflation_method is None:
            deflation_method = self.deflation_method
        deflation_method = deflation_method.lower()
        if deflation_method == "cg":
            return self._MultiSignal_cg(y, e_step_x, L, Theta)
        if deflation_method == "krylov":
            return self._MultiSignal_krylov(y, e_step_x, L, Theta, krylov_dim=krylov_dim)
        raise RuntimeError(f"unsupported deflation_method: {deflation_method}")
        

    def _output_spec(self, multi_x):
        if torch.is_tensor(multi_x):
            return multi_x.dtype, multi_x.device
        multi_x_tensor = torch.as_tensor(multi_x)
        return multi_x_tensor.dtype, torch.device("cpu")

    def _as_numpy_batch(self, multi_x):
        output_dtype, output_device = self._output_spec(multi_x)
        if torch.is_tensor(multi_x):
            multi_x_np = multi_x.detach().cpu().numpy()
        else:
            multi_x_np = np.asarray(multi_x)
        return multi_x_np, output_dtype, output_device

    def _stack_precision(self, precision_list, output_dtype, output_device):
        return torch.stack(precision_list, dim=0).to(device=output_device, dtype=output_dtype)

    def _m_step_quic(self, multi_x, output_dtype, output_device):
        # QUIC wrapper accepts signal samples X with shape (n_samples, n_nodes).
        precision_list = []
        for batch_idx in range(multi_x.shape[0]):
            self.graphical_lasso.fit(multi_x[batch_idx])
            precision_list.append(torch.as_tensor(self.graphical_lasso.precision_))
        return self._stack_precision(precision_list, output_dtype, output_device)

    def _m_step_sklearn(self, covariance, output_dtype, output_device):
        # sklearn.graphical_lasso accepts empirical covariance S with shape (n_nodes, n_nodes).
        from sklearn.covariance import graphical_lasso as sklearn_graphical_lasso

        covariance_np = covariance.detach().cpu().numpy()
        precision_list = []
        for batch_idx in range(covariance_np.shape[0]):
            _, precision = sklearn_graphical_lasso(
                covariance_np[batch_idx],
                alpha=self.alpha,
                tol=self.glasso_tol,
                max_iter=self.max_iter,
            )
            precision_list.append(torch.as_tensor(precision))
        return self._stack_precision(precision_list, output_dtype, output_device)

    def _m_step_admm(self, covariance):
        # Native PyTorch ADMM accepts batched empirical covariance directly.
        return torch_graphical_lasso(
            covariance,
            alpha=self.alpha,
            rho=self.glasso_rho,
            max_iter=self.max_iter,
            tol=self.glasso_tol,
            rtol=self.glasso_rtol,
        )

    def M_step(self, multi_x):
        # generate one precision matrix per batch
        # multi_x shape: (n_batch, n_samples, n_nodes)
        # output shape: (n_batch, n_nodes, n_nodes)
        if self.glasso_solver == "quic":
            multi_x_np, output_dtype, output_device = self._as_numpy_batch(multi_x)
            return self._m_step_quic(multi_x_np, output_dtype, output_device)

        covariance = self.compute_covariance(multi_x)
        if self.glasso_solver == "sklearn":
            output_dtype, output_device = self._output_spec(multi_x)
            return self._m_step_sklearn(covariance, output_dtype, output_device)
        if self.glasso_solver == "admm":
            return self._m_step_admm(covariance)

        raise RuntimeError(f"unsupported glasso_solver: {self.glasso_solver}")

    def single_step(self, y, L, Theta, deflation_method=None, krylov_dim=None):
        # perform a single iteration of the GEM algorithm
        
        # E-step: solve the optimization problem to find the optimal solutions x
        x = self.E_step(y, L, Theta) # shape (n_batch, n_nodes)
        # generate deflations from the optimal solutions x
        multi_x = self.MultiSignal(y, x, L, Theta, deflation_method=deflation_method, krylov_dim=krylov_dim) # shape (n_batch, n_samples, n_nodes)
        # M-step: generate one precision matrix per batch
        Theta_new = self.M_step(multi_x) # shape (n_batch, n_nodes, n_nodes)
        return x, Theta_new

    def forward(self, y, L_init=None, Theta_init=None):
        # y in shape (n_batch, n_nodes)
        if Theta_init is None:
            # generate the initial precision matrix Theta = (YY^T/(n - 1) + alpha * I)^{-1}
            S_init = self.compute_covariance(y.unsqueeze(0)) # shape (1, n_nodes, n_nodes)
            Theta_init = (S_init + self.alpha * np.eye(self.n_nodes)).squeeze(0).inverse() # shape (n_nodes, n_nodes)
            Theta_init = (Theta_init + Theta_init.T) / 2 # make sure the precision matrix is symmetric
        for i in range(self.max_iter):
            x, Theta_new = self.single_step(y, L_init, Theta_init)
            Theta_init = Theta_new
        return x, Theta_new
