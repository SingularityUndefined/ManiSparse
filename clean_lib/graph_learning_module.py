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
    ) -> None:
        super().__init__()
        self.directed_time = directed_time
        self.use_m_disp = use_m_disp
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
        df = features.unsqueeze(3) - feature_j

        if self.sharedM:
            Mdf = torch.einsum("hij, btnehj -> btnehi", self.multiM, df)
        else:
            Mdf = torch.einsum("thij, btnehj -> btnehi", self.multiM, df)

        weights = torch.exp(-((Mdf**2).sum(-1)))
        mask = (self.nearest_nodes[:, 1:] == -1).unsqueeze(0).unsqueeze(1).unsqueeze(4)
        mask = mask.repeat(batch_size, time_steps, 1, 1, self.n_heads)
        weights = weights * (~mask)

        degree = weights.sum(3)
        degree_j = degree[:, :, self.nearest_nodes[:, 1:].reshape(-1)].view(
            batch_size,
            time_steps,
            self.n_nodes,
            -1,
            self.n_heads,
        )
        degree_multiply = degree.unsqueeze(3) * degree_j
        inv_degree_multiply = torch.where(
            degree_multiply > 0,
            torch.ones((1,), device=self.device) / degree_multiply,
            torch.zeros((1,), device=self.device),
        )
        inv_degree_multiply = torch.where(inv_degree_multiply == torch.inf, 0, inv_degree_multiply)
        return weights * torch.sqrt(inv_degree_multiply)

    def directed_graph_from_features(self, features):
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

        if self.use_m_disp:
            weights = self._directed_weights_from_displacement(features, features_i, features_j)
        else:
            weights = self._directed_weights_from_inner_product(features_i, features_j)

        mask = torch.ones(time_steps - 1, self.interval, device=self.device).tril_(diagonal=0)
        mask = mask.unsqueeze(0).unsqueeze(3).unsqueeze(4)
        mask = mask.repeat(batch_size, 1, 1, self.n_nodes, self.n_heads)
        weights = weights * mask

        if self.directed_time:
            in_degree = weights.sum(2, keepdim=True)
            inv_in_degree = torch.where(
                in_degree > 0,
                torch.ones((1,), device=self.device) / in_degree,
                torch.zeros((1,), device=self.device),
            )
            return weights * inv_in_degree

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
        degree_multiply = degree_i * degree_j
        inv_degree_multiply = torch.where(
            degree_multiply > 0,
            torch.ones((1,), device=self.device) / degree_multiply,
            torch.zeros((1,), device=self.device),
        )
        return weights * torch.sqrt(inv_degree_multiply)

    def _directed_weights_from_displacement(self, features, features_i, features_j):
        if self.sharedQ and self.diff_interval:
            Q_df = torch.einsum("vhij, btvnhj -> btvnhi", self.multiQ, features_i - features_j.unsqueeze(2))
        elif self.sharedQ:
            Q_df = torch.einsum("hij, btvnhj -> btvnhi", self.multiQ, features_i - features_j.unsqueeze(2))
        elif self.diff_interval:
            Q_df = torch.einsum("tvhij, btvnhj -> btvnhi", self.multiQ, features_i - features_j.unsqueeze(2))
        else:
            Q_df = torch.einsum("thij, btvnhj -> btvnhi", self.multiQ, features_i - features_j.unsqueeze(2))

        assert not torch.isnan(Q_df).any(), (
            f"Q_df has NaN value: Q in ({self.multiQ.max().item():.4f}, "
            f"{self.multiQ.min().item():.4f}; features in "
            f"({features.max().item()}, {features.min().item()}))"
        )
        return torch.exp(-((Q_df**2).sum(-1)))

    def _directed_weights_from_inner_product(self, features_i, features_j):
        if self.sharedQ and self.diff_interval:
            Q_i = torch.einsum("vhij, btvnhj -> btvnhi", self.multiQ, features_i)
        elif self.sharedQ:
            Q_i = torch.einsum("hij, btvnhj -> btvnhi", self.multiQ, features_i)
        elif self.diff_interval:
            Q_i = torch.einsum("tvhij, btvnhj -> btvnhi", self.multiQ, features_i)
        else:
            Q_i = torch.einsum("thij, btvnhj -> btvnhi", self.multiQ, features_i)

        assert not torch.isnan(Q_i).any(), (
            f"Q_i has NaN value: Q in ({self.multiQ.max().item():.4f}, "
            f"{self.multiQ.min().item():.4f}; features in "
            f"({features_i.max().item()}, {features_i.min().item()}))"
        )
        return torch.exp(-((Q_i * features_j.unsqueeze(2)).sum(-1)))

    def forward(self, features=None):
        assert features is not None, "feature cannot be none"
        return self.undirected_graph_from_features(features), self.directed_graph_from_features(features)
