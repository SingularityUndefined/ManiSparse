import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from clean_lib.backup_modules import LR_guess


DEFAULT_ADMM_INFO = {
    "ADMM_iters": 50,
    "CG_iters": 3,
    "PGD_iters": 3,
    "mu_u_init": 10,
    "mu_d1_init": 10,
    "mu_d2_init": 10,
    "lambda_init": 5,
}


class ADMMBlock(nn.Module):
    """Unrolled ADMM block.

    Tensor convention inside the block is:
        (batch, time, node, head, channel)

    `u_ew` and `d_ew` are populated by the caller before `forward`.
    The theta-related parameters and operation hook are intentionally kept as
    interfaces only; theta logic is not implemented here.
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

        self.temp_indice = torch.arange(1, T).reshape(-1, 1) - torch.arange(1, interval + 1)

        self.u_ew = None # place holders
        self.d_ew = None
        self.Theta = None

        self.ADMM_iters = ADMM_info["ADMM_iters"]
        self.CG_iters = ADMM_info["CG_iters"]
        self.PGD_iters = ADMM_info["PGD_iters"]

        self.mu_u_init = ADMM_info["mu_u_init"]
        self.mu_d1_init = ADMM_info["mu_d1_init"]
        self.mu_d2_init = ADMM_info["mu_d2_init"]
        self.lambda_init = ADMM_info["lambda_init"]

        self._init_admm_parameters()
        self._init_cg_parameters()

        self.comb_weights = Parameter(
            torch.ones((self.n_heads,), device=self.device) / self.n_heads,
            requires_grad=True,
        )

    def _init_admm_parameters(self):
        self.mu_u = self._vector_parameter(self.mu_u_init)

        if self.ablation != "DGTV":
            self.mu_d1 = self._vector_parameter(self.mu_d1_init)
        if self.ablation != "DGLR":
            self.mu_d2 = self._vector_parameter(self.mu_d2_init)
        if self.ablation != "Theta":
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
        # if self.ablation != "Theta":
        #     self.rho_theta = self._vector_parameter(self.rho_theta_init)
        if self.ablation not in ["DGLR", "simple"]:
            self.rho_d = self._vector_parameter(self.rho_d_init)

    def _init_cg_parameters(self):
        alpha_init = 0.08
        self.alpha_x_init = alpha_init
        self.alpha_zu_init = alpha_init
        self.alpha_zd_init = alpha_init
        self.beta_x_init = alpha_init
        self.beta_zu_init = alpha_init
        self.beta_zd_init = alpha_init

        self.alpha_x = self._cg_parameter(self.alpha_x_init)
        self.beta_x = self._cg_parameter(self.beta_x_init)

        if self.ablation != "simple":
            self.alpha_zu = self._cg_parameter(self.alpha_zu_init)
            self.beta_zu = self._cg_parameter(self.beta_zu_init)

        if self.ablation not in ["DGLR", "simple"]:
            self.alpha_zd = self._cg_parameter(self.alpha_zd_init)
            self.beta_zd = self._cg_parameter(self.beta_zd_init)

    def _vector_parameter(self, init_value):
        return Parameter(
            torch.ones((self.ADMM_iters,), device=self.device) * init_value,
            requires_grad=True,
        )

    def _cg_parameter(self, init_value):
        return Parameter(
            torch.ones(
                (self.ADMM_iters, self.CG_iters, self.n_heads, 1),
                device=self.device,
            )
            * init_value,
            requires_grad=True,
        )

    def apply_op_Lu(self, x):
        """Spatial graph operator."""
        batch_size, time_steps = x.size(0), x.size(1)
        pad_x = torch.zeros_like(x[:, :, 0], device=self.device).unsqueeze(2)
        pad_x = torch.cat((x, pad_x), dim=2)

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
        """Theta operator interface. Implementation is intentionally deferred."""
        # Theta in (n_node, n_nodes), x in (B, T, n_nodes, n_head, n_channels)
        if self.Theta is None:
            return 0 # no theta component if Theta is not set
        return torch.einsum("ij,btjhc->btihc", self.Theta, x)

    def apply_op_Ldr(self, x):
        """Directed temporal graph operator."""
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
        """Transpose of the directed temporal graph operator."""
        _, time_steps = x.size(0), x.size(1)
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
        return self.apply_op_Ldr_T(self.apply_op_Ldr(x))

    def apply_op_Ln(self, x):
        """Undirected temporal graph operator used by UT ablation."""
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

    def CG_solver(self, LHS_func, RHS, x0, ADMM_iters, alpha, beta, args=None):
        """Conjugate-gradient-style unrolled solver for LHS_func(x) = RHS."""
        if x0 is None:
            x0 = RHS.clone()

        if args is None:
            r = RHS - LHS_func(x0, ADMM_iters)
        else:
            r = RHS - LHS_func(x0, args, ADMM_iters)

        p = r.clone()
        for i in range(self.CG_iters):
            if args is None:
                Ap = LHS_func(p, ADMM_iters)
            else:
                Ap = LHS_func(p, args, ADMM_iters)

            x0 = x0 + alpha[ADMM_iters, i] * p
            r = r - alpha[ADMM_iters, i] * Ap
            p = r + beta[ADMM_iters, i] * p

        return x0

    def LHS_simple_x(self, x, y, iters):
        HtHx = x.clone()
        HtHx[:, y.size(1) :] = torch.zeros_like(x[:, y.size(1) :])
        return (
            HtHx
            + self.mu_u[iters] * self.apply_op_Lu(x)
            + self.lambda_theta[iters] * self.apply_op_Theta(x)
            + (self.mu_d2[iters] + self.rho[iters] / 2) * self.apply_op_cLdr(x)
        )

    def LHS_x(self, x, y, iters):
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

    def LHS_zu(self, zu, iters): # added theta support
        if self.ablation == "Theta":
            return self.mu_u[iters] * self.apply_op_Lu(zu) + self.rho_u[iters] / 2 * zu
        return self.mu_u[iters] * self.apply_op_Lu(zu) + self.lambda_theta[iters] * self.apply_op_Theta(zu) + self.rho_u[iters] / 2 * zu

    def LHS_zd(self, zd, iters):
        if self.ablation == "UT":
            return self.mu_d2[iters] * self.apply_op_Ln(zd) + self.rho_d[iters] / 2 * zd

        return self.mu_d2[iters] * self.apply_op_cLdr(zd) + self.rho_d[iters] / 2 * zd

    def soft_threshold(self, phi, lambda_):
        u = torch.abs(phi) - lambda_
        return torch.sign(phi) * u * (u > 0)

    def Phi_PGD(self, phi, x, gamma, ADMM_iters):
        for i in range(self.PGD_iters):
            df = gamma + self.rho[ADMM_iters] * (phi - self.apply_op_Ldr(x))
            phi = self.soft_threshold(
                phi - self.epsilon[ADMM_iters, i] * df,
                self.epsilon[ADMM_iters, i] * self.mu_d1[ADMM_iters],
            )
        return phi

    def phi_direct(self, x, gamma, ADMM_iters):
        s = self.apply_op_Ldr(x) - gamma / self.rho[ADMM_iters]
        d = self.mu_d1[ADMM_iters] / self.rho[ADMM_iters]
        u = torch.abs(s) - d
        return torch.sign(s) * u * (u > 0)

    def forward(self, y, mask=None):
        """Run the ADMM block.

        Args:
            y: Input tensor in (batch, time, node, channel).
            mask: Number of observed time steps when y already contains T steps.
        """
        if y.size(1) < self.T:
            x = LR_guess(y, self.T, self.device)
        else:
            assert mask is not None, "mask should be t for sequential inputs"
            x = y[:, 0 : self.T]
            y = y[:, 0:mask]

        y = y.unsqueeze(-2).repeat(1, 1, 1, self.n_heads, 1)
        x = x.unsqueeze(-2).repeat(1, 1, 1, self.n_heads, 1)

        gamma_u = torch.ones_like(x) * 0.05
        gamma_d = torch.ones_like(x) * 0.1

        if self.ablation in ["None", "DGLR", "simple"]:
            gamma = torch.ones_like(x) * 0.1
            phi = self.apply_op_Ldr(x)

        zu = x.clone()
        zd = x.clone()

        for i in range(self.ADMM_iters):
            Hty = torch.zeros_like(x)
            Hty[:, 0 : y.size(1)] = y

            if self.ablation == "simple":
                RHS_x = self.apply_op_Ldr_T(self.rho[i] * phi + gamma) / 2 + Hty
                self._assert_finite(RHS_x, "RHS_x", i)
                x = self.CG_solver(
                    self.LHS_simple_x,
                    RHS_x,
                    x,
                    i,
                    self.alpha_x,
                    self.beta_x,
                    args=y,
                )
                self._assert_finite(x, "x", i)
            else:
                if self.ablation in ["DGTV", "UT"]:
                    RHS_x = (
                        (self.rho_u[i] * zu + self.rho_d[i] * zd) / 2
                        - (gamma_u + gamma_d) / 2
                        + Hty
                    )
                elif self.ablation in ["None", "Theta"]:
                    RHS_x = (
                        self.apply_op_Ldr_T(gamma + self.rho[i] * phi) / 2
                        + (self.rho_u[i] * zu + self.rho_d[i] * zd) / 2
                        - (gamma_u + gamma_d) / 2
                        + Hty
                    )
                elif self.ablation == "DGLR":
                    RHS_x = (
                        self.apply_op_Ldr_T(gamma + self.rho[i] * phi) / 2
                        + self.rho_u[i] * zu / 2
                        - gamma_u / 2
                        + Hty
                    )
                else:
                    raise ValueError(f"Unsupported ablation: {self.ablation}")

                self._assert_finite(RHS_x, "RHS_x", i)
                x = self.CG_solver(
                    self.LHS_x,
                    RHS_x,
                    x,
                    i,
                    self.alpha_x,
                    self.beta_x,
                    args=y,
                )
                self._assert_finite(x, "x", i)

                RHS_zu = gamma_u / 2 + self.rho_u[i] / 2 * x
                zu = self.CG_solver(self.LHS_zu, RHS_zu, zu, i, self.alpha_zu, self.beta_zu)
                self._assert_finite(RHS_zu, "RHS_zu", i)

                if self.ablation != "DGLR":
                    RHS_zd = gamma_d / 2 + self.rho_d[i] / 2 * x
                    zd = self.CG_solver(
                        self.LHS_zd,
                        RHS_zd,
                        zd,
                        i,
                        self.alpha_zd,
                        self.beta_zd,
                    )
                    self._assert_finite(RHS_zd, "RHS_zd", i)

                gamma_u = gamma_u + self.rho_u[i] * (x - zu)
                if self.ablation != "DGLR":
                    gamma_d = gamma_d + self.rho_d[i] * (x - zd)

            if self.ablation in ["None", "DGLR", "simple"]:
                phi = self.phi_direct(x, gamma, i)
                gamma = gamma + self.rho[i] * (phi - self.apply_op_Ldr(x))
                self._assert_finite(gamma, "gamma", i)
                self._assert_finite(phi, "phi", i)

        return torch.einsum("btnhc, h -> btnc", x, self.comb_weights)

    @staticmethod
    def _assert_finite(tensor, name, iteration):
        assert not torch.isnan(tensor).any(), f"{name} has NaN value in loop {iteration}"
        assert not torch.isinf(tensor).any(), f"{name} has inf value in loop {iteration}"
