import heapq
import math

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SimpleLinearExtrapolation(nn.Module):
    def __init__(self, n_nodes, t_in, T):
        super().__init__()
        assert T > t_in, "t_in > T"
        self.t_in = t_in
        self.T = T
        self.fc = nn.Linear(n_nodes, (T - t_in) * n_nodes)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, n_nodes, n_channels = x.size()
        y = self.fc(x[:, -1].transpose(-1, -2)).reshape(
            batch_size,
            n_channels,
            -1,
            n_nodes,
        )
        y = self.relu(y.permute(0, 2, 3, 1))
        return torch.cat([x, y], dim=1)


def laplacian_embeddings(k, n_nodes, edges, u_dist, device, sigma, eps=1e-10, normalized=False):
    assert 0 < k < n_nodes, f"0 < k < {n_nodes}"

    adj = torch.zeros((n_nodes, n_nodes), device=device)
    for i in range(edges.size(0)):
        adj[edges[i, 0], edges[i, 1]] = math.exp(-(u_dist[i] ** 2) / (sigma**2))

    diagonals = adj.sum(0)
    if normalized:
        diagonal_scale = torch.sqrt(diagonals[:, None] * diagonals[None, :])
        laplacian = torch.eye(n_nodes, device=device) - adj / diagonal_scale
    else:
        laplacian = torch.diag(diagonals) - adj

    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    _ = (eigenvalues > eps).sum()
    topk_index = torch.topk(eigenvalues, k, largest=False).indices
    embeddings = eigenvectors[:, topk_index]
    return embeddings.real if embeddings.is_complex() else embeddings


def position_embedding(time_list, half_t_dim, half_tid_dim, half_diw_dim, device, t_emb_only=False):
    batch_size, time_steps = time_list.size(0), time_list.size(1)
    t_emb = torch.zeros((batch_size, time_steps, 2 * half_t_dim), device=device)
    tid_emb = torch.zeros((batch_size, time_steps, 2 * half_tid_dim), device=device)
    diw_emb = torch.zeros((batch_size, time_steps, 2 * half_diw_dim), device=device)

    tid_list = time_list % (12 * 24)
    diw_list = (time_list // (12 * 24)) % 7

    t_multiplier = torch.pow(10000, torch.arange(0, half_t_dim, device=device) / half_t_dim)
    t_emb[:, :, 0::2] = torch.sin(time_list[:, :, None] / t_multiplier)
    t_emb[:, :, 1::2] = torch.cos(time_list[:, :, None] / t_multiplier)

    if t_emb_only:
        return t_emb

    tid_multiplier = torch.pow(
        10000,
        torch.arange(0, half_tid_dim, device=device) / half_tid_dim,
    )
    tid_emb[:, :, 0::2] = torch.sin(tid_list[:, :, None] / tid_multiplier)
    tid_emb[:, :, 1::2] = torch.cos(tid_list[:, :, None] / tid_multiplier)

    diw_multiplier = torch.pow(
        10000,
        torch.arange(0, half_diw_dim, device=device) / half_diw_dim,
    )
    diw_emb[:, :, 0::2] = torch.sin(diw_list[:, :, None] / diw_multiplier)
    diw_emb[:, :, 1::2] = torch.cos(diw_list[:, :, None] / diw_multiplier)

    return torch.cat((t_emb, tid_emb, diw_emb), dim=-1)


class SpatialTemporalEmbedding(nn.Module):
    def __init__(
        self,
        n_nodes,
        edges,
        u_dist,
        sigma_ratio,
        device,
        s_dim,
        t_dim=10,
        tid_dim=10,
        diw_dim=2,
        learnable=False,
    ):
        super().__init__()
        assert t_dim % 2 == 0, "t_dim should be even"
        assert tid_dim % 2 == 0, "tid_dim should be even"
        assert diw_dim % 2 == 0, "diw_dim should be even"

        self.learnable = learnable
        self.s_dim = s_dim
        self.n_nodes = n_nodes
        self.edges = edges
        self.u_dist = u_dist.to(device)
        self.sigma = self.u_dist.std() / 50
        self.device = device
        self.half_t_dim = t_dim // 2
        self.half_tid_dim = tid_dim // 2
        self.half_diw_dim = diw_dim // 2

        if self.learnable:
            self.spatial_emb = Parameter(torch.randn(n_nodes, s_dim))
            self.tid_emb = nn.Embedding(12 * 24, tid_dim)
            self.diw_emb = nn.Embedding(7, diw_dim)
        else:
            self.spatial_emb = laplacian_embeddings(
                self.s_dim,
                self.n_nodes,
                self.edges,
                self.u_dist,
                self.device,
                self.sigma,
            )

    def forward(self, t_list=None):
        batch_size, time_steps = t_list.size(0), t_list.size(1)
        s_emb = self.spatial_emb.unsqueeze(0).unsqueeze(1).repeat(
            batch_size,
            time_steps,
            1,
            1,
        )
        if t_list is None:
            return s_emb

        if not self.learnable:
            t_emb = position_embedding(
                t_list,
                self.half_t_dim,
                self.half_tid_dim,
                self.half_diw_dim,
                self.device,
            )
            t_emb = t_emb.unsqueeze(2).repeat(1, 1, self.n_nodes, 1)
            return torch.cat((s_emb, t_emb), -1)

        t_emb = position_embedding(
            t_list,
            self.half_t_dim,
            0,
            0,
            self.device,
            t_emb_only=True,
        )
        t_emb = t_emb.unsqueeze(2).repeat(1, 1, self.n_nodes, 1)
        tid_emb = self.tid_emb(t_list % (12 * 24)).unsqueeze(2).repeat(1, 1, self.n_nodes, 1)
        diw_emb = self.diw_emb((t_list // (12 * 24)) % 7).unsqueeze(2).repeat(
            1,
            1,
            self.n_nodes,
            1,
        )
        return torch.cat((s_emb, t_emb, tid_emb, diw_emb), -1)


def LR_guess(y, T, device):
    batch_size, time_steps, n_nodes, n_channels = y.size()
    if time_steps == 0:
        return torch.zeros((batch_size, T, n_nodes, n_channels), device=device)
    if time_steps == 1:
        return y.repeat(1, T, 1, 1)

    y_flat = y.transpose(0, 1).reshape(time_steps, -1)
    x_idx = torch.arange(0, time_steps, 1, dtype=torch.float, device=device)
    bar_x = (time_steps - 1) / 2
    bar_y = y_flat.mean(0)

    numerator = time_steps * y_flat.T @ x_idx - x_idx.sum() * y_flat.sum(0)
    denominator = time_steps * x_idx.dot(x_idx) - x_idx.sum() ** 2
    w = numerator / denominator
    b = bar_y - bar_x * w

    x_out = torch.arange(time_steps, T, 1, dtype=torch.float, device=device)
    y_out = torch.cat([y_flat, x_out[:, None] * w + b], 0)
    return y_out.view(T, batch_size, n_nodes, n_channels).transpose(0, 1)


def connect_list(n_nodes, edges, device):
    counts = torch.zeros(n_nodes, dtype=torch.int)
    for edge in edges:
        counts[edge[0]] += 1

    max_degree = counts.max()
    neighbors = -torch.ones(n_nodes, max_degree + 1, dtype=torch.int, device=device)
    for edge in edges:
        neighbors[edge[0], counts[edge[0]]] = edge[1]
        counts[edge[0]] -= 1

    assert torch.all(counts == 0), "Counts should be zero after processing all edges"
    assert torch.all(neighbors[:, 0] == -1), "First column should be empty before self assignment"
    neighbors[:, 0] = torch.arange(n_nodes, device=device)
    return neighbors


def k_hop_neighbors(n_nodes, edges: torch.Tensor, k):
    graph = nx.DiGraph()
    graph.add_edges_from(edges.detach().cpu().numpy())

    new_edges = set()
    for node in range(n_nodes):
        k_hop = nx.single_source_shortest_path_length(graph, node, cutoff=k).keys()
        for neighbor in k_hop:
            new_edges.add((node, neighbor))

    return torch.LongTensor(np.array(list(new_edges)))


def visualise_graph(edges: torch.Tensor, distances: torch.Tensor, dataset_name, fig_name):
    import matplotlib.pyplot as plt

    edges = edges.detach().cpu().numpy()
    distances = distances.detach().cpu().numpy()

    graph = nx.DiGraph()
    for i in range(len(edges)):
        graph.add_edge(edges[i, 0], edges[i, 1], weight=distances[i])

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=False, node_size=7, node_color="lightblue", arrowsize=2)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=2)
    plt.title(dataset_name)
    plt.savefig(fig_name, dpi=800)


def find_k_nearest_neighbors(n_nodes, edges: torch.Tensor, distances: torch.Tensor, k, device):
    edges = edges.detach().cpu().numpy()
    distances = distances.detach().cpu().numpy()

    graph = nx.DiGraph()
    for i in range(len(edges)):
        graph.add_edge(edges[i, 0], edges[i, 1], weight=distances[i])

    nearest_nodes = -torch.ones((n_nodes, k + 1), dtype=torch.int, device=device)
    nearest_distance = torch.full((n_nodes, k + 1), float("inf"), device=device)

    for node in range(n_nodes):
        path_lengths = nx.single_source_dijkstra_path_length(graph, node)
        closest_nodes = heapq.nsmallest(k + 1, path_lengths.items(), key=lambda x: x[1])
        k_true = len(closest_nodes)
        nearest_nodes[node, :k_true] = torch.tensor([i for (i, _) in closest_nodes], device=device)
        nearest_distance[node, :k_true] = torch.tensor([j for (_, j) in closest_nodes], device=device)

    return nearest_nodes, nearest_distance


def layer_norm_on_data(x: torch.Tensor, norm_shape):
    norm_dims = len(norm_shape)
    assert torch.Size(norm_shape) == x.shape[-norm_dims:], f"get {x[-norm_dims].size()} for {norm_shape}"

    dims = list(range(x.ndim - norm_dims, x.ndim))
    mean = x.mean(dim=dims, keepdim=True)
    mean_x2 = (x**2).mean(dim=dims, keepdim=True)
    std = torch.sqrt(mean_x2 - mean**2 + 1e-6)
    return (x - mean) / std, mean, std


def layer_recovery_on_data(x, norm_shape, gain, bias):
    x_norm, _, _ = layer_norm_on_data(x, norm_shape)
    return x_norm * bias + gain
