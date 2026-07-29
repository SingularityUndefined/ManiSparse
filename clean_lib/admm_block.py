import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


DEFAULT_ADMM_INFO = {
    "ADMM_iters": 50,
    "CG_iters": 3,
    "mu_u_init": 10,
    "mu_d1_init": 10,
    "mu_d2_init": 10,
    "lambda_init": 5,
}


class UnrolledCGSolver(nn.Module):
    """Unrolled conjugate-gradient-style solver.

    All solver tensors follow the ADMM block convention:
        RHS, x0, residuals: (B, T, N, H, C)

    The learnable step sizes are indexed by outer ADMM iteration and inner CG
    iteration:
        alpha, beta: (ADMM_iters, CG_iters, H, 1)
    """

    def __init__(
        self,
        ADMM_iters,
        CG_iters,
        n_heads,
        device,
        alpha_init=0.08,
        beta_init=0.08,
        name=None,
    ):
        super().__init__()
        self.ADMM_iters = ADMM_iters
        self.CG_iters = CG_iters
        self.n_heads = n_heads
        self.device = device
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.name = name

        param_shape = (ADMM_iters, CG_iters, n_heads, 1)
        self.alpha = Parameter(
            torch.ones(param_shape, device=device) * alpha_init,
            requires_grad=True,
        )
        self.beta = Parameter(
            torch.ones(param_shape, device=device) * beta_init,
            requires_grad=True,
        )

    def forward(self, LHS_func, RHS, x0, ADMM_iters, args=None):
        """Approximately solve LHS_func(x) = RHS for one ADMM iteration."""
        if x0 is None:
            x0 = RHS.clone()

        # r and p are the CG residual/search direction. They keep the same
        # shape as the signal estimate: (B, T, N, H, C).
        r = RHS - self._apply_lhs(LHS_func, x0, ADMM_iters, args)
        p = r.clone()

        for i in range(self.CG_iters):
            Ap = self._apply_lhs(LHS_func, p, ADMM_iters, args)
            x0 = x0 + self.alpha[ADMM_iters, i] * p
            r = r - self.alpha[ADMM_iters, i] * Ap
            p = r + self.beta[ADMM_iters, i] * p

        return x0

    @staticmethod
    def _apply_lhs(LHS_func, x, ADMM_iters, args):
        if args is None:
            return LHS_func(x, ADMM_iters)
        return LHS_func(x, args, ADMM_iters)


class DeflationCGSolver(nn.Module):
    """Non-parametric CG solver used by multi-signal deflation.

    Unlike `UnrolledCGSolver`, this solver has no learnable step sizes and is
    always executed under `torch.no_grad()`.  It works on ADMM output tensors
    shaped (B, T, N, C), and treats all dimensions except batch as one
    flattened vector space.
    """

    def __init__(self, max_iter=10, tol=1e-6):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, A_func, b, x0=None):
        with torch.no_grad():
            return self._solve_no_grad(A_func, b.detach(), None if x0 is None else x0.detach())

    def _solve_no_grad(self, A_func, b, x0=None):
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()

        reduce_dims = tuple(range(1, b.ndim))
        r = b - A_func(x)
        p = r.clone()
        rsold = (r * r).sum(dim=reduce_dims)
        eps = torch.finfo(b.dtype).eps

        if torch.all(torch.sqrt(rsold) < self.tol):
            return x

        for _ in range(self.max_iter):
            Ap = A_func(p)
            denom = (p * Ap).sum(dim=reduce_dims)
            valid_denom = torch.abs(denom) > eps
            denom_safe = torch.where(valid_denom, denom, torch.ones_like(denom))
            alpha = torch.where(valid_denom, rsold / denom_safe, torch.zeros_like(rsold))
            view_shape = (b.size(0),) + (1,) * (b.ndim - 1)

            x = x + alpha.view(view_shape) * p
            r = r - alpha.view(view_shape) * Ap
            rsnew = (r * r).sum(dim=reduce_dims)
            if torch.all(torch.sqrt(rsnew) < self.tol):
                break

            valid_rsold = rsold > eps
            rsold_safe = torch.where(valid_rsold, rsold, torch.ones_like(rsold))
            beta = torch.where(valid_rsold, rsnew / rsold_safe, torch.zeros_like(rsold))
            p = r + beta.view(view_shape) * p
            rsold = rsnew

        return x


class ADMMBlock(nn.Module):
    """Unrolled ADMM block.

    Tensor convention inside the block is:
        (batch, time, node, head, channel)

    `u_ew`, `d_ew`, and optionally `Theta` are populated by the caller before
    `forward`.

    External input/output convention:
        y:      (B, observed_or_full_T, N, C)
        output: (B, T, N, C)

    Internal convention after expanding graph heads:
        x, y, zu, zd, gamma_*: (B, T, N, H, C)
        u_ew: (B, T, N, K, H), where K is nearest_nodes.size(1) - 1
        d_ew: (B, T - 1, interval, N, H)
        Theta: optional dense node precision/affinity matrix, (N, N)

    Each ADMMBlock owns its own x/zu/zd CG solvers. Since UnrollingModel
    constructs one ADMMBlock per unrolled layer, CG parameters are not shared
    across layers. Within each solver, alpha/beta are indexed by ADMM
    iteration, so different outer iterations use different parameter slices.
    """

    VALID_ABLATIONS = {"None", "DGLR", "DGTV", "UT", "simple", "Theta"}

    def __init__(
        self,
        T,
        n_nodes,
        n_heads,
        n_channels,
        interval,
        connect_list,
        nearest_nodes,
        device,
        ADMM_info=None,
        ablation="None",
    ):
        super().__init__()
        if ADMM_info is None:
            ADMM_info = DEFAULT_ADMM_INFO

        self.device = device
        self.T = T
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.n_channels = n_channels
        self.interval = interval
        self.connect_list = connect_list
        self.nearest_nodes = nearest_nodes.to(torch.int64)
        self.ablation = ablation

        assert self.ablation in self.VALID_ABLATIONS, (
            "ablation should be one of: None, Theta, DGLR, DGTV, UT, simple"
        )

        # temp_indice[t-1, v] gives the source time index t - (v + 1) for
        # directed temporal edges into target time t. Shape: (T - 1, interval).
        self.temp_indice = torch.arange(1, T).reshape(-1, 1) - torch.arange(1, interval + 1)

        self.u_ew = None
        self.d_ew = None
        self.Theta = None

        self.ADMM_iters = ADMM_info["ADMM_iters"]
        self.CG_iters = ADMM_info["CG_iters"]
        self.deflation_samples = ADMM_info.get("deflation_samples", 2)
        self.deflation_CG_iters = ADMM_info.get("deflation_CG_iters", self.CG_iters)
        self.deflation_tol = ADMM_info.get("deflation_tol", 1e-6)

        self.mu_u_init = ADMM_info["mu_u_init"]
        self.mu_d1_init = ADMM_info["mu_d1_init"]
        self.mu_d2_init = ADMM_info["mu_d2_init"]
        self.lambda_init = ADMM_info["lambda_init"]

        self._init_admm_parameters()
        self._init_cg_solvers()
        self._init_deflation_solver()

        self.comb_weights = Parameter(
            torch.ones((self.n_heads,), device=self.device) / self.n_heads,
            requires_grad=True,
        )

    @property
    def x_solver(self):
        return self._x_solver

    @property
    def zu_solver(self):
        return self._zu_solver

    @property
    def zd_solver(self):
        return self._zd_solver

    @property
    def alpha_x(self):
        return self.x_solver.alpha

    @property
    def beta_x(self):
        return self.x_solver.beta

    @property
    def alpha_zu(self):
        return self.zu_solver.alpha

    @property
    def beta_zu(self):
        return self.zu_solver.beta

    @property
    def alpha_zd(self):
        return self.zd_solver.alpha

    @property
    def beta_zd(self):
        return self.zd_solver.beta

    def _init_admm_parameters(self):
        self.mu_u = self._vector_parameter(self.mu_u_init)

        if self.ablation != "DGTV":
            self.mu_d1 = self._vector_parameter(self.mu_d1_init)
        if self.ablation != "DGLR":
            self.mu_d2 = self._vector_parameter(self.mu_d2_init)
        if self._uses_theta_regularizer():
            self.lambda_theta = self._vector_parameter(self.lambda_init)

        rho_init = math.sqrt(self.n_nodes / self.T)
        self.rho_init = rho_init
        self.rho_u_init = rho_init
        self.rho_d_init = rho_init
        self.rho_theta_init = rho_init

        if self.ablation != "DGTV":
            self.rho = self._vector_parameter(self.rho_init)
        if self.ablation != "simple":
            self.rho_u = self._vector_parameter(self.rho_u_init)
        if self.ablation not in ["DGLR", "simple"]:
            self.rho_d = self._vector_parameter(self.rho_d_init)

    def _init_cg_solvers(self):
        alpha_init = 0.08
        beta_init = 0.08
        self.alpha_x_init = alpha_init
        self.alpha_zu_init = alpha_init
        self.alpha_zd_init = alpha_init
        self.beta_x_init = beta_init
        self.beta_zu_init = beta_init
        self.beta_zd_init = beta_init

        # One ADMM update has multiple linear solves. Keep x, z_u, and z_d as
        # distinct child modules so they have independent alpha/beta
        # parameters. Each solver still has an ADMM_iters dimension, so every
        # outer ADMM iteration uses its own parameter slice.
        self._x_solver = self._make_cg_solver("x", self.alpha_x_init, self.beta_x_init)

        if self.ablation != "simple":
            self._zu_solver = self._make_cg_solver(
                "zu",
                self.alpha_zu_init,
                self.beta_zu_init,
            )
        if self.ablation not in ["DGLR", "simple"]:
            self._zd_solver = self._make_cg_solver(
                "zd",
                self.alpha_zd_init,
                self.beta_zd_init,
            )

    def _init_deflation_solver(self):
        self.deflation_solver = DeflationCGSolver(
            max_iter=self.deflation_CG_iters,
            tol=self.deflation_tol,
        )

    def _make_cg_solver(self, name, alpha_init, beta_init):
        return UnrolledCGSolver(
            ADMM_iters=self.ADMM_iters,
            CG_iters=self.CG_iters,
            n_heads=self.n_heads,
            device=self.device,
            alpha_init=alpha_init,
            beta_init=beta_init,
            name=name,
        )

    def _vector_parameter(self, init_value):
        return Parameter(
            torch.ones((self.ADMM_iters,), device=self.device) * init_value,
            requires_grad=True,
        )

    def _uses_theta_regularizer(self):
        """Whether this block includes the dense Theta regularizer."""
        return self.ablation != "Theta"

    def apply_op_Lu(self, x):
        """Apply the learned spatial graph operator.

        Args:
            x: (B, T, N, H, C)

        Returns:
            (I - W_u) x with shape (B, T, N, H, C).
        """
        batch_size, time_steps = x.size(0), x.size(1)
        pad_x = torch.zeros_like(x[:, :, 0], device=self.device).unsqueeze(2)
        pad_x = torch.cat((x, pad_x), dim=2)

        # nearest_nodes[:, 0] is the node itself; columns 1: are spatial
        # neighbors. Missing neighbors are expected to point to the padded zero
        # node prepared above.
        neighbor_x = pad_x[:, :, self.nearest_nodes[:, 1:].reshape(-1)].view(
            batch_size,
            time_steps,
            self.n_nodes,
            -1,
            self.n_heads,
            self.n_channels,
        )
        return x - (self.u_ew.unsqueeze(-1) * neighbor_x).sum(3)

    def apply_op_Theta(self, x):
        """Apply the optional dense node-level Theta matrix.

        Args:
            x: (B, T, N, H, C)
            self.Theta: (N, N), shared by the batch, or (B, N, N), one dense
                node matrix per batch sample.

        Returns:
            Theta @ x over the node dimension, also (B, T, N, H, C).
        """
        if self.Theta is None:
            return torch.zeros_like(x)
        Theta = torch.as_tensor(self.Theta, device=x.device, dtype=x.dtype)
        if Theta.ndim == 2:
            return torch.einsum("ij,btjhc->btihc", Theta, x)
        if Theta.ndim == 3:
            if Theta.size(0) != x.size(0):
                raise ValueError("batched Theta must have the same batch size as x")
            return torch.einsum("bij,btjhc->btihc", Theta, x)
        raise ValueError("Theta must have shape (N, N) or (B, N, N)")

    def apply_op_Ldr(self, x):
        """Apply directed temporal difference operator L_d.

        For each target time t=1..T-1, each node receives `interval` previous
        time states from temp_indice. The first time slice has no history and is
        forced to zero.

        Args:
            x: (B, T, N, H, C)

        Returns:
            L_d x with shape (B, T, N, H, C).
        """
        batch_size, time_steps = x.size(0), x.size(1)
        history = x[:, self.temp_indice.view(-1)].reshape(
            batch_size,
            time_steps - 1,
            self.interval,
            self.n_nodes,
            -1,
            self.n_channels,
        )
        features = self.d_ew.unsqueeze(-1) * history

        y = x.clone()
        y[:, 1:] = x[:, 1:] - features.sum(2)
        y[:, 0] = x[:, 0] * 0
        return y

    def apply_op_Ldr_T(self, x):
        """Apply transpose of directed temporal operator L_d^T.

        Args:
            x: (B, T, N, H, C)

        Returns:
            L_d^T x with shape (B, T, N, H, C).
        """
        time_steps = x.size(1)
        features = self.d_ew.unsqueeze(-1) * x[:, 1:].unsqueeze(2)
        features = torch.stack(
            [
                features.diagonal(offset=-offset, dim1=1, dim2=2).sum(-1)
                for offset in range(0, time_steps - 1)
            ],
            dim=1,
        )

        y = x.clone()
        y[:, 0] = x[:, 0] * 0
        y[:, :-1] = y[:, :-1] - features
        return y

    def apply_op_cLdr(self, x):
        """Apply L_d^T L_d."""
        return self.apply_op_Ldr_T(self.apply_op_Ldr(x))

    def apply_op_Ln(self, x):
        """Apply the undirected temporal operator used by the UT ablation."""
        batch_size, time_steps = x.size(0), x.size(1)
        history = x[:, self.temp_indice.view(-1)].reshape(
            batch_size,
            time_steps - 1,
            self.interval,
            self.n_nodes,
            -1,
            self.n_channels,
        )
        in_features = self.d_ew.unsqueeze(-1) * history

        out_features = self.d_ew.unsqueeze(-1) * x[:, 1:].unsqueeze(2)
        out_features = torch.stack(
            [
                out_features.diagonal(offset=-offset, dim1=1, dim2=2).sum(-1)
                for offset in range(0, time_steps - 1)
            ],
            dim=1,
        )

        y = x.clone()
        y[:, 1:] = y[:, 1:] - in_features.sum(2)
        y[:, :-1] = y[:, :-1] - out_features
        return y

    def LHS_simple_x(self, x, y, iters):
        """Left-hand side for the single-variable/simple ADMM update."""
        HtHx = x.clone()
        HtHx[:, y.size(1) :] = torch.zeros_like(x[:, y.size(1) :])
        return (
            HtHx
            + self.mu_u[iters] * self.apply_op_Lu(x)
            + self.lambda_theta[iters] * self.apply_op_Theta(x)
            + (self.mu_d2[iters] + self.rho[iters] / 2) * self.apply_op_cLdr(x)
        )

    def LHS_x(self, x, y, iters):
        """Left-hand side for the x update in the split ADMM formulation."""
        HtHx = x.clone()
        HtHx[:, y.size(1) :] = torch.zeros_like(x[:, y.size(1) :])

        if self.ablation in ["DGTV", "UT"]:
            return HtHx + (self.rho_u[iters] + self.rho_d[iters]) / 2 * x

        if self.ablation in ["Theta", "None"]:
            return (
                HtHx
                + (self.rho_u[iters] + self.rho_d[iters]) / 2 * x
                + self.rho[iters] / 2 * self.apply_op_cLdr(x)
            )

        if self.ablation == "DGLR":
            return (
                HtHx
                + self.rho[iters] / 2 * self.apply_op_cLdr(x)
                + self.rho_u[iters] / 2 * x
            )

        raise ValueError(f"Unsupported ablation: {self.ablation}")

    def LHS_zu(self, zu, iters):
        """Left-hand side for the spatial auxiliary variable z_u."""
        output = self.mu_u[iters] * self.apply_op_Lu(zu) + self.rho_u[iters] / 2 * zu
        if self.ablation != "Theta":
            output = output + self.lambda_theta[iters] * self.apply_op_Theta(zu)
        return output

    def LHS_zd(self, zd, iters):
        """Left-hand side for the temporal auxiliary variable z_d."""
        if self.ablation == "UT":
            return self.mu_d2[iters] * self.apply_op_Ln(zd) + self.rho_d[iters] / 2 * zd
        return self.mu_d2[iters] * self.apply_op_cLdr(zd) + self.rho_d[iters] / 2 * zd

    def phi_direct(self, x, gamma, ADMM_iters):
        """Closed-form soft-threshold update for temporal sparsity variable phi."""
        s = self.apply_op_Ldr(x) - gamma / self.rho[ADMM_iters]
        d = self.mu_d1[ADMM_iters] / self.rho[ADMM_iters]
        u = torch.abs(s) - d
        return torch.sign(s) * u * (u > 0)

    def _expand_heads(self, x):
        """Expand a returned signal (B, T, N, C) to ADMM head space."""
        return x.unsqueeze(-2).repeat(1, 1, 1, self.n_heads, 1)

    def _combine_heads(self, x):
        """Collapse ADMM head space (B, T, N, H, C) back to (B, T, N, C)."""
        return torch.einsum("btnhc,h->btnc", x, self.comb_weights)

    def _apply_deflation_Lu(self, x):
        """Apply the same learned Lu operator used by ADMM to output-scale x."""
        return self._combine_heads(self.apply_op_Lu(self._expand_heads(x)))

    def _apply_deflation_Theta(self, x):
        """Apply the block's current Theta parameter to output-scale x."""
        if self.Theta is None:
            return torch.zeros_like(x)
        Theta = torch.as_tensor(self.Theta, device=x.device, dtype=x.dtype)
        if Theta.ndim == 2:
            return torch.einsum("ij,btjc->btic", Theta, x)
        if Theta.ndim == 3:
            return torch.einsum("bij,btjc->btic", Theta, x)
        raise ValueError("Theta must have shape (N, N) or (B, N, N)")

    def _deflation_lhs(self, x, iteration):
        """Apply I + mu_u Lu + lambda_theta Theta for deflation solves."""
        output = x + self.mu_u[iteration] * self._apply_deflation_Lu(x)
        if self._uses_theta_regularizer():
            output = output + self.lambda_theta[iteration] * self._apply_deflation_Theta(x)
        return output

    @staticmethod
    def _mode_inner(x, y):
        """Batchwise inner product over all non-batch dimensions."""
        return (x * y).sum(dim=tuple(range(1, x.ndim)), keepdim=True)

    def _normalize_mode(self, x):
        norm = torch.sqrt(self._mode_inner(x, x))
        eps = torch.finfo(x.dtype).eps
        return torch.where(norm > 0, x / norm.clamp_min(eps), torch.zeros_like(x))

    def _remove_mode_projection(self, x, q):
        return x - self._mode_inner(x, q) * q

    def _project_orthogonal(self, x, basis, n_basis):
        if n_basis == 0:
            return x
        output = x
        for basis_idx in range(n_basis):
            output = self._remove_mode_projection(output, basis[:, basis_idx])
        return output

    def _initial_deflation_state(self, rhs, e_step_x, n_modes):
        """Initialize multi_x so mode 0 is exactly the original ADMM output."""
        multi_x = torch.zeros(
            (e_step_x.size(0), n_modes) + tuple(e_step_x.shape[1:]),
            dtype=e_step_x.dtype,
            device=e_step_x.device,
        )
        multi_x[:, 0] = e_step_x

        basis = torch.zeros_like(multi_x)
        first_q = self._normalize_mode(e_step_x)
        basis[:, 0] = first_q
        projected_rhs = self._remove_mode_projection(rhs, first_q)
        return multi_x, basis, projected_rhs

    def _run_deflation(self, rhs, e_step_x, n_modes, iteration):
        """Generate multi-signal deflation modes from the original ADMM output.

        Args:
            rhs: full sequence used as the deflation right-hand side, (B, T, N, C).
            e_step_x: original ADMM forward output, (B, T, N, C).
            n_modes: number of modes to return. Mode 0 is always e_step_x.
            iteration: ADMM parameter index used for mu_u/lambda_theta.

        Returns:
            multi_x: (B, n_modes, T, N, C).
        """
        if n_modes < 1:
            raise ValueError("deflation requires at least one mode")

        multi_x, basis, projected_rhs = self._initial_deflation_state(rhs, e_step_x, n_modes)
        for mode_idx in range(1, n_modes):
            def projected_A_func(v, current_basis=basis, current_mode=mode_idx):
                pv = self._project_orthogonal(v, current_basis, current_mode)
                Apv = self._deflation_lhs(pv, iteration)
                return self._project_orthogonal(Apv, current_basis, current_mode)

            z = self.deflation_solver.solve(projected_A_func, projected_rhs)
            x_k = self._project_orthogonal(z, basis, mode_idx)
            multi_x[:, mode_idx] = x_k

            q_k = self._normalize_mode(x_k)
            basis[:, mode_idx] = q_k
            projected_rhs = self._remove_mode_projection(projected_rhs, q_k)

        return multi_x

    def forward(self, y, mask=None, deflation=False, deflation_samples=None, deflation_iteration=None):
        """Run one ADMM block over a sequence.

        Args:
            y: full-horizon sequence, shape (B, T, N, C). The caller is
                responsible for extrapolating the observed prefix before this
                block; UnrollingModel does that once before the ADMM loop.
            mask: observed prefix length `t_in`. The data term H^T H x keeps
                only this prefix and treats future steps as unobserved.
            deflation: when True, also run multi-signal deflation after the
                original ADMM forward result is computed.
            deflation_samples: number of deflated modes. The first mode is
                exactly the original ADMM output.
            deflation_iteration: ADMM parameter index for mu_u/lambda_theta in
                the deflation linear operator. Defaults to the final iteration.

        Returns:
            If deflation=False:
                output: (B, T, N, C).
            If deflation=True:
                output, multi_x where multi_x is (B, K, T, N, C) and
                multi_x[:, 0] is exactly output.
        """
        if mask is None:
            raise ValueError("ADMMBlock.forward requires mask=t_in for the observed prefix length")
        if y.size(1) != self.T:
            raise ValueError(f"ADMMBlock.forward expects full horizon T={self.T}, got y.shape={tuple(y.shape)}")
        self._assert_finite(y, "input_y", -1)
        self._assert_finite(self.u_ew, "u_ew", -1)
        self._assert_finite(self.d_ew, "d_ew", -1)

        x = y[:, 0 : self.T]
        y = y[:, 0:mask]
        deflation_rhs = x.clone()

        # ADMM operates on H learned graph heads. The final head dimension is
        # collapsed by comb_weights before returning.
        y = y.unsqueeze(-2).repeat(1, 1, 1, self.n_heads, 1)
        x = x.unsqueeze(-2).repeat(1, 1, 1, self.n_heads, 1)

        # Dual variables for x = z_u and x = z_d constraints.
        gamma_u = torch.ones_like(x) * 0.05
        gamma_d = torch.ones_like(x) * 0.1
        zu = x.clone()
        zd = x.clone()

        # phi/gamma are used by the directed temporal L1 term. Ablating Theta
        # removes only the dense Theta regularizer, not this temporal term.
        if self.ablation in ["None", "DGLR", "simple", "Theta"]:
            gamma = torch.ones_like(x) * 0.1
            phi = self.apply_op_Ldr(x)

        for i in range(self.ADMM_iters):
            Hty = torch.zeros_like(x)
            Hty[:, 0 : y.size(1)] = y

            if self.ablation == "simple":
                x = self._update_simple_x(x, y, Hty, phi, gamma, i)
            else:
                x, zu, zd, gamma_u, gamma_d = self._update_split_variables(
                    x,
                    y,
                    Hty,
                    zu,
                    zd,
                    gamma_u,
                    gamma_d,
                    i,
                    gamma=gamma if self.ablation in ["None", "DGLR", "Theta"] else None,
                    phi=phi if self.ablation in ["None", "DGLR", "Theta"] else None,
                )

            if self.ablation in ["None", "DGLR", "simple", "Theta"]:
                phi = self.phi_direct(x, gamma, i)
                gamma = gamma + self.rho[i] * (phi - self.apply_op_Ldr(x))
                self._assert_finite(gamma, "gamma", i)
                self._assert_finite(phi, "phi", i)

        output = self._combine_heads(x)
        if not deflation or not self._uses_theta_regularizer():
            return output

        if deflation_samples is None:
            deflation_samples = self.deflation_samples
        if deflation_iteration is None:
            deflation_iteration = self.ADMM_iters - 1
        if not 0 <= deflation_iteration < self.ADMM_iters:
            raise ValueError("deflation_iteration must be in [0, ADMM_iters)")
        multi_x = self._run_deflation(
            deflation_rhs.detach(),
            output.detach(),
            deflation_samples,
            deflation_iteration,
        )

        # This is assigned directly in _initial_deflation_state. `multi_x` is
        # detached from autograd, while `output` keeps the original forward
        # gradient path.
        assert torch.equal(multi_x[:, 0], output), "multi_x[:, 0] must be the original ADMM output"
        return output, multi_x

    def _update_simple_x(self, x, y, Hty, phi, gamma, iteration):
        """Update x for the simplified formulation without z_u/z_d solves."""
        RHS_x = self.apply_op_Ldr_T(self.rho[iteration] * phi + gamma) / 2 + Hty
        self._assert_finite(RHS_x, "RHS_x", iteration)
        x = self.x_solver(self.LHS_simple_x, RHS_x, x, iteration, args=y)
        self._assert_finite(x, "x", iteration)
        return x

    def _update_split_variables(
        self,
        x,
        y,
        Hty,
        zu,
        zd,
        gamma_u,
        gamma_d,
        iteration,
        gamma=None,
        phi=None,
    ):
        """Update x, z_u, z_d and their dual variables for one ADMM iteration."""
        RHS_x = self._build_RHS_x(Hty, zu, zd, gamma_u, gamma_d, iteration, gamma, phi)
        self._assert_finite(RHS_x, "RHS_x", iteration)

        x = self.x_solver(self.LHS_x, RHS_x, x, iteration, args=y)
        self._assert_finite(x, "x", iteration)

        RHS_zu = gamma_u / 2 + self.rho_u[iteration] / 2 * x
        zu = self.zu_solver(self.LHS_zu, RHS_zu, zu, iteration)
        self._assert_finite(RHS_zu, "RHS_zu", iteration)
        self._assert_finite(zu, "zu", iteration)

        if self.ablation != "DGLR":
            RHS_zd = gamma_d / 2 + self.rho_d[iteration] / 2 * x
            zd = self.zd_solver(self.LHS_zd, RHS_zd, zd, iteration)
            self._assert_finite(RHS_zd, "RHS_zd", iteration)
            self._assert_finite(zd, "zd", iteration)

        gamma_u = gamma_u + self.rho_u[iteration] * (x - zu)
        self._assert_finite(gamma_u, "gamma_u", iteration)
        if self.ablation != "DGLR":
            gamma_d = gamma_d + self.rho_d[iteration] * (x - zd)
            self._assert_finite(gamma_d, "gamma_d", iteration)

        return x, zu, zd, gamma_u, gamma_d

    def _build_RHS_x(self, Hty, zu, zd, gamma_u, gamma_d, iteration, gamma, phi):
        """Build the right-hand side of the x linear solve for each ablation."""
        if self.ablation in ["DGTV", "UT"]:
            return (
                (self.rho_u[iteration] * zu + self.rho_d[iteration] * zd) / 2
                - (gamma_u + gamma_d) / 2
                + Hty
            )

        if self.ablation in ["None", "Theta"]:
            return (
                self.apply_op_Ldr_T(gamma + self.rho[iteration] * phi) / 2
                + (self.rho_u[iteration] * zu + self.rho_d[iteration] * zd) / 2
                - (gamma_u + gamma_d) / 2
                + Hty
            )

        if self.ablation == "DGLR":
            return (
                self.apply_op_Ldr_T(gamma + self.rho[iteration] * phi) / 2
                + self.rho_u[iteration] * zu / 2
                - gamma_u / 2
                + Hty
            )

        raise ValueError(f"Unsupported ablation: {self.ablation}")

    @staticmethod
    def _assert_finite(tensor, name, iteration):
        if tensor is None:
            raise AssertionError(f"{name} is None in ADMM iteration {iteration}")
        if torch.isfinite(tensor).all():
            return
        finite_count = torch.isfinite(tensor).sum().item()
        total = tensor.numel()
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise AssertionError(
            f"{name} is not finite in ADMM iteration {iteration}: "
            f"shape={tuple(tensor.shape)}, finite={finite_count}/{total}, "
            f"nan={nan_count}, inf={inf_count}"
        )
