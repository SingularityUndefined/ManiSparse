import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self, beta=0.8):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def spatial_gcn_aggregation(x, nearest_nodes, nearest_dist, n_heads, device, sigma):
    batch_size, time_steps, n_nodes, n_in = x.size(0), x.size(1), x.size(2), x.size(-1)
    pad_x = torch.zeros_like(x[:, :, 0]).unsqueeze(2)
    pad_x = torch.cat((x, pad_x), dim=2)

    lambda_ = torch.arange(1, n_heads + 1, 1, dtype=torch.float32, device=device) / n_heads
    nearest_dist = nearest_dist.view(-1)
    nearest_nodes = nearest_nodes.view(-1)

    weights = torch.exp(-(nearest_dist[:, None] ** 2) * lambda_ / (sigma**2))
    weights[nearest_nodes == -1, :] = 0
    weights[weights < 1e-5] = 0
    assert not weights.isnan().any(), "GCN weights contain NaN"

    if x.ndim == 4:
        agg = (pad_x[:, :, nearest_nodes, None] * weights[:, :, None]).view(
            batch_size,
            time_steps,
            n_nodes,
            -1,
            n_heads,
            n_in,
        )
    elif x.ndim == 5:
        agg = (pad_x[:, :, nearest_nodes] * weights[:, :, None]).view(
            batch_size,
            time_steps,
            n_nodes,
            -1,
            n_heads,
            n_in,
        )
    else:
        raise ValueError(f"Invalid tensor shape: {x.shape}, dimension needs to be 4 or 5")

    agg = agg.sum(3)
    assert not agg.isnan().any(), "GCN aggregation contain NaN"

    nearest_dist[nearest_dist == torch.inf] = 0
    dist_agg = (weights * nearest_dist[:, None]).view(n_nodes, -1, n_heads).sum(1)
    return agg, dist_agg


class SpatialGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, nearest_nodes, nearest_dist, n_heads, device, sigma):
        super().__init__()
        self.nearest_nodes = nearest_nodes
        self.nearest_dist = nearest_dist
        self.n_heads = n_heads
        self.device = device
        self.sigma = sigma
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        agg, dist_agg = spatial_gcn_aggregation(
            x,
            self.nearest_nodes,
            self.nearest_dist,
            self.n_heads,
            self.device,
            self.sigma,
        )
        return agg + dist_agg[None, None, :, :, None]


class GraphSAGELayer(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        nearest_nodes,
        n_heads,
        in_heads,
        device,
        use_out_fc=False,
        use_multihead_fc=True,
        use_single_fc=True,
    ):
        super().__init__()
        self.nearest_nodes = nearest_nodes
        self.n_nodes = nearest_nodes.size(0)
        self.k = nearest_nodes.size(1) - 1
        self.n_heads = n_heads
        self.in_heads = in_heads
        self.device = device
        self.use_out_fc = use_out_fc
        self.use_multihead_fc = use_multihead_fc
        self.use_single_fc = use_single_fc

        if self.use_single_fc:
            in_features = self.in_heads * (self.k + 1) if self.use_multihead_fc else self.k + 1
            out_features = self.n_heads if self.use_multihead_fc else 1
            self.agg_fc = nn.Linear(in_features, out_features)
        else:
            self.agg_fc = nn.Linear(self.k + 1, 1)
            if self.use_multihead_fc:
                self.swish1 = Swish()
                self.multihead_fc = nn.Linear(self.in_heads, self.n_heads)

        if self.use_out_fc:
            self.swish2 = Swish()
            self.out_fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        assert not x.isnan().any(), "Input x contains NaN"
        batch_size, time_steps, n_nodes, n_channels = x.size(0), x.size(1), x.size(2), x.size(-1)

        pad_x = torch.zeros_like(x[:, :, 0]).unsqueeze(2)
        pad_x = torch.cat((x, pad_x), dim=2)
        if pad_x.ndim == 4:
            pad_x = pad_x.unsqueeze(-2)

        in_heads = pad_x.size(-2)
        assert not torch.isnan(self.agg_fc.weight).any(), "Aggregation weights contain NaN"
        assert not torch.isinf(self.agg_fc.weight).any(), "Aggregation weights contain inf"

        if self.use_single_fc and self.use_multihead_fc:
            x_nn = pad_x[:, :, self.nearest_nodes.view(-1)].reshape(
                batch_size,
                time_steps,
                n_nodes,
                -1,
                n_channels,
            )
            x_agg = self.agg_fc(x_nn.transpose(-1, -2)).transpose(-1, -2)
        elif self.use_single_fc:
            x_nn = pad_x[:, :, self.nearest_nodes.view(-1)].reshape(
                batch_size,
                time_steps,
                n_nodes,
                -1,
                in_heads,
                n_channels,
            )
            x_agg = self.agg_fc(x_nn.transpose(-1, -3)).squeeze(-1).transpose(-1, -2)
        else:
            x_nn = pad_x[:, :, self.nearest_nodes.view(-1)].reshape(
                batch_size,
                time_steps,
                n_nodes,
                -1,
                in_heads,
                n_channels,
            )
            x_agg = self.agg_fc(x_nn.transpose(-1, -3)).squeeze(-1)
            if self.use_multihead_fc:
                x_agg = self.multihead_fc(self.swish1(x_agg)).transpose(-1, -2)

        assert not x_agg.isnan().any(), "Aggregation contain NaN"

        if self.use_out_fc:
            x_agg = self.out_fc(self.swish2(x_agg))
        return x_agg


class GNNExtrapolation(nn.Module):
    def __init__(self, n_nodes, t_in, T, nearest_nodes, nearest_dists, n_heads, device, sigma_ratio=400, alpha=0.2):
        super().__init__()
        self.device = device
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.t_in = t_in
        self.T = T
        self.nearest_nodes = nearest_nodes
        self.nearest_dists = nearest_dists
        self.sigma = self.nearest_dists.max() / sigma_ratio
        self.shrink = nn.Linear(t_in * n_heads, T - t_in)
        self.swish = Swish()
        self.alpha = alpha

    def forward(self, x):
        batch_size, _, n_nodes, n_channels = x.size()
        agg, _ = spatial_gcn_aggregation(
            x,
            self.nearest_nodes,
            self.nearest_dists,
            self.n_heads,
            self.device,
            self.sigma,
        )
        agg = agg.permute(0, 2, 4, 1, 3).reshape(batch_size, n_nodes, n_channels, -1)
        y = self.shrink(agg).permute(0, 3, 1, 2)
        return torch.cat([x, self.swish(y)], dim=1)


class TemporalHistoryLayer(nn.Module):
    def __init__(self, in_features, out_features, interval):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.interval = interval
        self.linear = nn.Linear(interval * in_features, out_features)

    def forward(self, x):
        batch_size, time_steps, n_nodes = x.size(0), x.size(1), x.size(2)
        pad_x = torch.zeros_like(x[:, 0 : self.interval])
        pad_x = torch.cat((pad_x, x), dim=1)
        inputs = torch.stack([pad_x[:, i : i + self.interval] for i in range(time_steps)], dim=1)

        if x.ndim == 4:
            inputs = inputs.transpose(2, 3).reshape(batch_size, time_steps, n_nodes, -1)
        elif x.ndim == 5:
            n_heads = x.size(-2)
            inputs = inputs.transpose(2, 3).reshape(batch_size, time_steps, n_nodes, n_heads, -1)
        else:
            raise ValueError(f"Invalid tensor shape: {x.shape}, dimension needs to be 4 or 5")

        return self.linear(inputs)


class GraphSAGEExtrapolation(nn.Module):
    def __init__(
        self,
        n_nodes,
        t_in,
        T,
        nearest_nodes,
        n_in,
        n_heads,
        device,
        interval,
        n_layers=2,
        parallel=False,
    ):
        super().__init__()
        self.device = device
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.t_in = t_in
        self.T = T
        self.n_in = n_in
        self.nearest_nodes = nearest_nodes
        self.interval = interval

        self.input_layer = FElayer(
            n_in,
            n_in,
            nearest_nodes,
            n_heads,
            device,
            interval,
            parallel=parallel,
            use_out_fc=False,
        )
        self.n_layers = n_layers
        if self.n_layers > 1:
            self.SAGEs = nn.Sequential(
                *[
                    FElayer(
                        self.n_in,
                        self.n_in,
                        self.nearest_nodes,
                        self.n_heads,
                        self.device,
                        self.interval,
                        parallel=parallel,
                        in_heads=self.n_heads,
                        use_out_fc=False,
                    )
                    for _ in range(self.n_layers - 1)
                ]
            )

        self.shrink = nn.Linear(t_in * n_heads, T - t_in)
        self.swish = Swish()

    def forward(self, x):
        batch_size, _, n_nodes, n_channels = x.size()
        agg = self.input_layer(x)
        if self.n_layers > 1:
            agg = self.SAGEs(agg)
        agg = agg.permute(0, 2, 4, 1, 3).reshape(batch_size, n_nodes, n_channels, -1)
        y = self.shrink(agg).permute(0, 3, 1, 2)
        return torch.cat([x, self.swish(y)], dim=1)


class FElayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        nearest_nodes,
        n_heads,
        device,
        interval,
        parallel=True,
        in_heads=1,
        use_out_fc=False,
    ):
        super().__init__()
        self.nearest_nodes = nearest_nodes
        self.n_heads = n_heads
        self.device = device
        self.parallel = parallel
        self.in_heads = in_heads
        self.swish1 = Swish()
        self.swish2 = Swish()

        self.graph_sage = GraphSAGELayer(
            in_features,
            out_features,
            self.nearest_nodes,
            self.n_heads,
            in_heads,
            self.device,
            use_out_fc=use_out_fc,
        )

        if self.parallel:
            temporal_out = out_features * n_heads if in_heads == 1 else out_features
            self.temporal_hist = TemporalHistoryLayer(in_features, temporal_out, interval)
        else:
            self.temporal_hist = TemporalHistoryLayer(out_features, out_features, interval)

    def forward(self, x):
        batch_size, time_steps, n_nodes = x.size(0), x.size(1), x.size(2)
        spatial_features = self.swish1(self.graph_sage(x))

        if not self.parallel:
            return self.swish2(self.temporal_hist(spatial_features))

        temporal_features = self.temporal_hist(x)
        if self.in_heads == 1:
            temporal_features = temporal_features.unsqueeze(-2).reshape(
                batch_size,
                time_steps,
                n_nodes,
                self.n_heads,
                -1,
            )
        temporal_features = self.swish2(temporal_features)
        return spatial_features + temporal_features


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        nearest_nodes,
        n_heads,
        device,
        interval,
        parallel=False,
        n_layers=2,
    ):
        super().__init__()
        self.nearest_nodes = nearest_nodes
        self.n_heads = n_heads
        self.device = device
        self.n_layers = n_layers

        self.input_layer = FElayer(
            in_features,
            out_features,
            self.nearest_nodes,
            self.n_heads,
            self.device,
            interval,
            parallel=parallel,
            use_out_fc=True,
        )
        if self.n_layers > 1:
            self.fe_layers = nn.Sequential(
                *[
                    FElayer(
                        out_features,
                        out_features,
                        self.nearest_nodes,
                        self.n_heads,
                        self.device,
                        interval,
                        parallel=parallel,
                        in_heads=self.n_heads,
                        use_out_fc=False,
                    )
                    for _ in range(self.n_layers - 1)
                ]
            )

    def forward(self, x):
        features = self.input_layer(x)
        if self.n_layers > 1:
            features = self.fe_layers(features)
        return features
