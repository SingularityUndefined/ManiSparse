import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphLearningModule(nn.Module):
    """Learn spatial and temporal graph weights from extracted features."""

    def __init__(
        self,
        T,
        n_nodes,
        connect_list,
        nearest_nodes,
        n_heads,
        interval,
        device,
        n_channels=None,
        sigma=6,
        Q1_init=1.2,
        M_init=1.5,
        sharedM=True,
        sharedQ=True,
        diff_interval=True,
        directed_time=True,
        use_m_disp=True,
        use_stable_graph_learning=False,
    ) -> None:
        super().__init__()
        self.directed_time = directed_time
        self.use_m_disp = use_m_disp
        self.use_stable_graph_learning = use_stable_graph_learning
        self.T = T
        self.n_nodes = n_nodes
        self.device = device
        self.connect_list = connect_list
        self.nearest_nodes = nearest_nodes
        self.n_heads = n_heads
        self.interval = interval
        self.temp_indice = torch.arange(1, T).reshape(-1, 1) - torch.arange(1, interval + 1)

        self.n_channels = n_channels
        self.n_out = (self.n_channels + 1) // 2
        self.sharedM = sharedM
        self.sharedQ = sharedQ
        self.diff_interval = diff_interval
        self.Q1_init = Q1_init
        self.M_init = M_init

        self.multiQ = Parameter(self._init_multiQ(), requires_grad=True)
        self.multiM = Parameter(self._init_multiM(), requires_grad=True)
        self.debug_numerics = False
        self.debug_context = ""
        self.debug_records = []

    def set_debug_numerics(self, enabled, context=""):
        """Enable detailed forward/backward tensor diagnostics for one rerun."""
        self.debug_numerics = enabled
        self.debug_context = context
        self.debug_records = []

    @staticmethod
    def _safe_inverse_positive(values):
        """Compute 1 / values for positive entries without creating 1 / 0."""
        eps = torch.finfo(values.dtype).eps
        safe_values = values.clamp_min(eps)
        return torch.where(values > 0, safe_values.reciprocal(), torch.zeros_like(values))

    def _inverse_positive(self, values):
        """Use the selected graph normalization formula."""
        if self.use_stable_graph_learning:
            return self._safe_inverse_positive(values)
        return torch.where(
            values > 0,
            torch.ones((1,), device=self.device) / values,
            torch.zeros((1,), device=self.device),
        )

    def _graph_exp(self, exp_arg, clamp_min=None, clamp_max=None):
        """Apply exp with optional numerical-stability clamps."""
        if self.use_stable_graph_learning:
            if clamp_min is not None:
                exp_arg = exp_arg.clamp_min(clamp_min)
            if clamp_max is not None:
                exp_arg = exp_arg.clamp_max(clamp_max)
        return torch.exp(exp_arg)

    @staticmethod
    def _tensor_summary(name, tensor):
        """Summarize tensor health without keeping autograd history."""
        detached = tensor.detach()
        total = detached.numel()
        if total == 0:
            return f"{name}: shape={tuple(detached.shape)}, numel=0"

        finite_mask = torch.isfinite(detached)
        finite_count = finite_mask.sum().item()
        nan_count = torch.isnan(detached).sum().item()
        inf_count = torch.isinf(detached).sum().item()
        if finite_count == 0:
            return (
                f"{name}: shape={tuple(detached.shape)}, finite=0/{total}, "
                f"nan={nan_count}, inf={inf_count}"
            )

        finite_values = detached[finite_mask]
        return (
            f"{name}: shape={tuple(detached.shape)}, finite={finite_count}/{total}, "
            f"nan={nan_count}, inf={inf_count}, min={finite_values.min().item():.3e}, "
            f"max={finite_values.max().item():.3e}, mean={finite_values.mean().item():.3e}"
        )

    def _debug_tensor(self, name, tensor):
        """Record forward tensor health and attach a backward grad hook."""
        if not self.debug_numerics:
            return tensor

        label = f"{self.debug_context}.{name}" if self.debug_context else name
        self.debug_records.append("forward " + self._tensor_summary(label, tensor))

        if tensor.requires_grad:
            def _grad_hook(grad, grad_label=label):
                summary = "backward " + self._tensor_summary(f"grad({grad_label})", grad)
                self.debug_records.append(summary)
                if not torch.isfinite(grad).all():
                    raise RuntimeError(f"Non-finite gradient first observed at {grad_label}\n{summary}")
                return grad

            tensor.register_hook(_grad_hook)
        return tensor

    def _init_multiQ(self):
        multiQ = torch.diag_embed(torch.ones((self.n_heads, self.n_channels), device=self.device))

        if not self.sharedQ:
            multiQ = multiQ.unsqueeze(0).repeat(self.T - 1, 1, 1, 1)

        if self.diff_interval:
            interval_scale = torch.linspace(1, self.Q1_init, steps=self.interval, device=self.device)
            if self.sharedQ:
                multiQ = multiQ.unsqueeze(0).repeat(self.interval, 1, 1, 1)
                multiQ = multiQ * interval_scale.reshape(self.interval, 1, 1, 1)
            else:
                multiQ = multiQ.unsqueeze(1).repeat(1, self.interval, 1, 1, 1)
                multiQ = multiQ * interval_scale.reshape(1, self.interval, 1, 1, 1)

        return multiQ

    def _init_multiM(self):
        multiM = (
            torch.diag_embed(torch.ones((self.n_heads, self.n_channels), device=self.device))
            * self.M_init
        )
        if not self.sharedM:
            multiM = multiM.unsqueeze(0).repeat(self.T, 1, 1, 1)
        return multiM

    def undirected_graph_from_features(self, features):
        features = self._debug_tensor("undirected.features", features)
        batch_size, time_steps = features.size(0), features.size(1)
        pad_features = torch.zeros_like(features[:, :, 0], device=self.device).unsqueeze(2)
        pad_features = torch.cat((features, pad_features), dim=2)

        feature_j = pad_features[:, :, self.nearest_nodes[:, 1:].reshape(-1)].view(
            batch_size,
            time_steps,
            self.n_nodes,
            -1,
            self.n_heads,
            self.n_channels,
        )
        df = self._debug_tensor("undirected.df", features.unsqueeze(3) - feature_j)

        if self.sharedM:
            Mdf = torch.einsum("hij, btnehj -> btnehi", self.multiM, df)
        else:
            Mdf = torch.einsum("thij, btnehj -> btnehi", self.multiM, df)
        Mdf = self._debug_tensor("undirected.Mdf", Mdf)

        exp_arg = self._debug_tensor("undirected.exp_arg", -((Mdf**2).sum(-1)))
        weights = self._debug_tensor("undirected.weights_raw", self._graph_exp(exp_arg, clamp_min=-80.0))
        mask = (self.nearest_nodes[:, 1:] == -1).unsqueeze(0).unsqueeze(1).unsqueeze(4)
        mask = mask.repeat(batch_size, time_steps, 1, 1, self.n_heads)
        weights = self._debug_tensor("undirected.weights_masked", weights * (~mask))

        degree = weights.sum(3)
        degree_j = degree[:, :, self.nearest_nodes[:, 1:].reshape(-1)].view(
            batch_size,
            time_steps,
            self.n_nodes,
            -1,
            self.n_heads,
        )
        degree_multiply = self._debug_tensor("undirected.degree_multiply", degree.unsqueeze(3) * degree_j)
        inv_degree_multiply = self._inverse_positive(degree_multiply)
        if not self.use_stable_graph_learning:
            inv_degree_multiply = torch.where(inv_degree_multiply == torch.inf, 0, inv_degree_multiply)
        return self._debug_tensor("undirected.weights_normalized", weights * torch.sqrt(inv_degree_multiply))

    def directed_graph_from_features(self, features):
        features = self._debug_tensor("directed.features", features)
        batch_size, time_steps = features.size(0), features.size(1)
        features_j = features[:, 1:]
        features_i = features[:, self.temp_indice.view(-1)].view(
            batch_size,
            time_steps - 1,
            self.interval,
            self.n_nodes,
            -1,
            self.n_channels,
        )
        features_i = self._debug_tensor("directed.features_i", features_i)
        features_j = self._debug_tensor("directed.features_j", features_j)

        valid_edge_mask = torch.ones(time_steps - 1, self.interval, device=self.device).tril_(diagonal=0)
        feature_mask = valid_edge_mask.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)
        weight_mask = valid_edge_mask.unsqueeze(0).unsqueeze(3).unsqueeze(4)

        if self.use_m_disp:
            weights = self._directed_weights_from_displacement(features, features_i, features_j, feature_mask)
        else:
            weights = self._directed_weights_from_inner_product(features_i, features_j, feature_mask)

        weights = self._debug_tensor("directed.weights_masked", weights * weight_mask)

        if self.directed_time:
            in_degree = self._debug_tensor("directed.in_degree", weights.sum(2, keepdim=True))
            inv_in_degree = self._inverse_positive(in_degree)
            return self._debug_tensor("directed.weights_normalized", weights * inv_in_degree)

        in_degree = weights.sum(2)
        out_degree = torch.stack(
            [
                weights.diagonal(offset=-offset, dim1=1, dim2=2).sum(-1)
                for offset in range(time_steps - 1)
            ],
            dim=1,
        )
        pad_degree = torch.zeros_like(in_degree[:, 0:1], device=self.device)
        degree = torch.cat((pad_degree, in_degree), dim=1) + torch.cat((out_degree, pad_degree), dim=1)

        degree_i = degree[:, 1:].unsqueeze(2)
        degree_j = degree[:, self.temp_indice.view(-1)].view(
            batch_size,
            time_steps - 1,
            self.interval,
            self.n_nodes,
            self.n_heads,
        )
        degree_multiply = self._debug_tensor("directed.degree_multiply", degree_i * degree_j)
        inv_degree_multiply = self._inverse_positive(degree_multiply)
        return self._debug_tensor("directed.weights_normalized", weights * torch.sqrt(inv_degree_multiply))

    def _directed_weights_from_displacement(self, features, features_i, features_j, valid_feature_mask):
        feature_diff = features_i - features_j.unsqueeze(2)
        if self.use_stable_graph_learning:
            feature_diff = feature_diff * valid_feature_mask
        feature_diff = self._debug_tensor("directed.feature_diff", feature_diff)
        if self.sharedQ and self.diff_interval:
            Q_df = torch.einsum("vhij, btvnhj -> btvnhi", self.multiQ, feature_diff)
        elif self.sharedQ:
            Q_df = torch.einsum("hij, btvnhj -> btvnhi", self.multiQ, feature_diff)
        elif self.diff_interval:
            Q_df = torch.einsum("tvhij, btvnhj -> btvnhi", self.multiQ, feature_diff)
        else:
            Q_df = torch.einsum("thij, btvnhj -> btvnhi", self.multiQ, feature_diff)

        Q_df = self._debug_tensor("directed.Q_df", Q_df)
        if self.use_stable_graph_learning:
            assert torch.isfinite(Q_df).all(), (
                f"Q_df has non-finite value: Q in ({self.multiQ.max().item():.4f}, "
                f"{self.multiQ.min().item():.4f}; features in "
                f"({features.max().item()}, {features.min().item()}))"
            )
        else:
            assert not torch.isnan(Q_df).any(), (
                f"Q_df has NaN value: Q in ({self.multiQ.max().item():.4f}, "
                f"{self.multiQ.min().item():.4f}; features in "
                f"({features.max().item()}, {features.min().item()}))"
            )
        exp_arg = self._debug_tensor("directed.exp_arg_displacement", -((Q_df**2).sum(-1)))
        return self._debug_tensor("directed.weights_raw", self._graph_exp(exp_arg, clamp_min=-80.0))

    def _directed_weights_from_inner_product(self, features_i, features_j, valid_feature_mask):
        if self.use_stable_graph_learning:
            features_i = features_i * valid_feature_mask
        features_i = self._debug_tensor("directed.masked_features_i", features_i)
        if self.sharedQ and self.diff_interval:
            Q_i = torch.einsum("vhij, btvnhj -> btvnhi", self.multiQ, features_i)
        elif self.sharedQ:
            Q_i = torch.einsum("hij, btvnhj -> btvnhi", self.multiQ, features_i)
        elif self.diff_interval:
            Q_i = torch.einsum("tvhij, btvnhj -> btvnhi", self.multiQ, features_i)
        else:
            Q_i = torch.einsum("thij, btvnhj -> btvnhi", self.multiQ, features_i)

        Q_i = self._debug_tensor("directed.Q_i", Q_i)
        if self.use_stable_graph_learning:
            assert torch.isfinite(Q_i).all(), (
                f"Q_i has non-finite value: Q in ({self.multiQ.max().item():.4f}, "
                f"{self.multiQ.min().item():.4f}; features in "
                f"({features_i.max().item()}, {features_i.min().item()}))"
            )
        else:
            assert not torch.isnan(Q_i).any(), (
                f"Q_i has NaN value: Q in ({self.multiQ.max().item():.4f}, "
                f"{self.multiQ.min().item():.4f}; features in "
                f"({features_i.max().item()}, {features_i.min().item()}))"
            )
        exp_arg = self._debug_tensor("directed.exp_arg_inner_product", -((Q_i * features_j.unsqueeze(2)).sum(-1)))
        return self._debug_tensor("directed.weights_raw", self._graph_exp(exp_arg, clamp_max=80.0))

    def forward(self, features=None):
        assert features is not None, "feature cannot be none"
        return self.undirected_graph_from_features(features), self.directed_graph_from_features(features)
