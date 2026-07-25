import torch
import numpy as np
import torch.nn as nn

from glasso_pytorch import graphical_lasso as torch_graphical_lasso

try:
    from .utils import CG_Solver
except ImportError:
    from utils import CG_Solver


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
        if self.sparse_version:
            self.cg_solver = CG_Solver(max_iter=100, tol=1e-6)
        
        self.deflation_solver_list = nn.ModuleList([CG_Solver(max_iter=10, tol=1e-6) for _ in range(self.deflation_samples)])
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

    def apply_L(self, x, W=None):
        # apply the graph Laplacian L to the input data x
        # NOTE: here only compute batch-level operation. If samples requried, please multiply the first 2 dimensions first.
        # x shape: (n_batch, n_nodes)
        # W shape: (n_nodes, n_neighbors) for sparse graph, (n_nodes, n_nodes) for dense graph
        n_batch = x.size(0)
        # k = self.neighbor_list.size(-1) if self.sparse_version else self.n_nodes
        if not self.sparse_version:
            assert W is not None, "W must be provided for the dense version of the GEM algorithm"
            assert W == W.T, "W must be symmetric for the dense version of the GEM algorithm"
            assert W >= 0, "W must be non-negative for the dense version of the GEM algorithm"
            degree = np.sum(W, axis=1) # shape (n_nodes,)
            # apply the transformation W to the input data x
            return (degree * x - np.dot(x, W)).sum(-1) # shape (n_batch, n_samples)
        else: # sparse version
            assert W.size(0) == self.n_nodes, "W must have the same number of rows as the number of nodes"
            assert W.size(1) == self.neighbor_list.size(-1), "W must have the same number of columns as the number of neighbors"
            delta_x = W * (x - x[:, :, self.neighbor_list.view(-1)].view(n_batch, self.n_nodes, -1)) # shape (n_batch, n_samples, n_nodes)
            delta_x = delta_x.sum(-1) # shape (n_batch, n_samples, n_nodes)
            return delta_x

    def apply_Theta(self, x, Theta):
        # apply the precision matrix Theta to the input data x
        # x shape: (n_batch, n_nodes)
        # Theta shape: (n_nodes, n_nodes)
        return np.dot(x, Theta) # shape (n_batch, n_nodes)

    def E_step_LHS(self, x, W, Theta):
        # compute the left-hand side of the E-step optimization problem
        # LHS: (I + lambda_L * L + lambda_theta * Theta) * x
        # x shape: (n_batch, n_nodes)
        return x + self.lambda_L * self.apply_L(x, W) + self.lambda_theta * self.apply_Theta(x, Theta) # shape (n_batch, n_nodes)

    def E_step(self, y, W, Theta):
        # solve the optimization problem to find the optimal solutions x
        # (I + lambda_L * L + lambda_theta * Theta) * x = y
        if not self.sparse_version:
            # use the dense version of the optimization problem
            Lap = np.diag(np.sum(W, axis=1)) - W # compute the graph Laplacian
            A = np.eye(self.n_nodes) + self.lambda_L * Lap + self.lambda_theta * Theta # shape (n_nodes, n_nodes)
            x = np.linalg.solve(A, y.T).T # shape (n_batch, n_nodes)
            return x
        else:
            # use the sparse version of the optimization problem
            # hand_craft conjugated gradients
            def A_func(x):
                return self.E_step_LHS(x, W, Theta)
            return self.cg_solver.solve(A_func, y) # shape (n_batch, n_nodes)

    def MultiSignal(self, y, x, L, Theta):
        # generate deflations from the optimal solutions x
        # solving the optimization problem to find the deflations
        # (I + lambda_L * L + lambda_theta * Theta) * z = y
        # NOTE: here we assume that the deflations are independent of each other, so we can solve the optimization problem for each deflation separately
        # y shape: (n_batch, n_samples, n_nodes)
        # x shape: (n_batch, n_nodes)
        # L shape: (n_nodes, n_nodes)
        # Theta shape: (n_nodes, n_nodes)
        n_batch, n_samples, n_nodes = y.size(0), y.size(1), y.size(2)
        multi_x = np.zeros((n_batch, n_samples, n_nodes))
        
        for i in range(n_samples):
            # solve the optimization problem for each deflation
            multi_x[:, i, :] = self.E_step(y[:, i, :], L, Theta) # shape (n_batch, n_nodes)
        return multi_x # shape (n_batch, n_samples, n_nodes)    
        pass
        

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

    def M_step(self, multi_x, Theta):
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

    def single_step(self, y, L, Theta):
        # perform a single iteration of the GEM algorithm
        
        # E-step: solve the optimization problem to find the optimal solutions x
        x = self.E_step(y, L, Theta) # shape (n_batch, n_nodes)
        # generate deflations from the optimal solutions x
        multi_x = self.MultiSignal(y, x, L, Theta) # shape (n_batch, n_samples, n_nodes)
        # M-step: generate one precision matrix per batch
        Theta_new = self.M_step(multi_x, Theta) # shape (n_batch, n_nodes, n_nodes)
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
