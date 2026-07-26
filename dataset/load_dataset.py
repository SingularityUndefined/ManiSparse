"""Dataset helpers for traffic/weather forecasting experiments.

All dataset classes return sliding windows:
    y: observed prefix, shape (t, n_nodes, n_channels)
    x: reconstruction target, shape (T, n_nodes, n_channels)
    time: optional absolute time index, shape (T,)

Each dataset also exposes `graph_info`, consumed by the unrolling model:
    {
        "n_nodes": int,
        "u_edges": LongTensor, shape (n_edges, 2),
        "u_dist": FloatTensor, shape (n_edges,),
    }
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from collections import Counter
import pickle


SPLITS = ("train", "val", "test")
TRAFFIC_SPLIT_RATIOS = {
    "train": (0.0, 0.6),
    "val": (0.6, 0.8),
    "test": (0.8, 1.0),
}
WEATHER_SPLIT_RATIOS = {
    "train": (0.0, 0.35),
    "val": (0.35, 0.5),
    "test": (0.5, 1.0),
}


def _slice_by_split(data, split, ratios):
    """Return split data and the absolute offset of its first time step."""
    assert split in SPLITS, "split should in train, val or test"
    data_len = data.shape[0]
    begin_ratio, end_ratio = ratios[split]
    begin = int(data_len * begin_ratio)
    end = int(data_len * end_ratio)
    return data[begin:end], begin


def _to_graph_info(n_nodes, u_edges, u_distance):
    return {
        "n_nodes": n_nodes,
        "u_edges": u_edges,
        "u_dist": u_distance,
    }


def _window_length(data_len, T, stride):
    return (data_len - T) // stride


def _time_index(T, index, stride, data_begin):
    return torch.arange(0, T, dtype=torch.long) + index * stride + data_begin


def physical_graph(df, sensor_dict=None):
    """Build a bidirectional physical graph from a CSV dataframe.

    Expected columns are at least `from`, `to`, and a distance-like last column.
    If `sensor_dict` is provided, raw sensor ids are remapped to dense node
    indices used by the data tensor.
    """
    if sensor_dict is None:
        from_list, to_list = list(df["from"].values), list(df["to"].values)
    else:
        from_list = [sensor_dict[i] for i in df["from"].values]
        to_list = [sensor_dict[i] for i in df["to"].values]

    n_edges = len(from_list) * 2
    u_edges = np.array([from_list + to_list, to_list + from_list]).T

    edge_counts = Counter([(u_edges[i, 0], u_edges[i, 1]) for i in range(n_edges)])
    assert max(list(edge_counts.values())), "distance graph asymmetric"

    ew1 = df[df.columns[-1]].values
    u_distance = np.stack([ew1, ew1]).reshape(-1)
    return n_edges, u_edges, u_distance


class TrafficDataset(Dataset):
    """Traffic dataset with CSV physical graph.

    Raw data file is expected to be `.npz` with key `data` and shape:
        (T_total, n_nodes, n_channels)

    Splits use train/val/test = 6:2:2.
    """

    def __init__(
        self,
        data_folder,
        graph_csv,
        data_file,
        T,
        t,
        stride,
        split="train",
        n_nodes=None,
        id_file=None,
        return_time=False,
        use_one_channel=False,
        truncated=False,
    ):
        super().__init__()
        self.T = T
        self.t = t
        self.stride = stride
        self.truncated = truncated
        self.return_time = return_time
        self.use_one_channel = use_one_channel

        data = np.load(os.path.join(data_folder, data_file))["data"]
        print("nan_count", len(data[np.isnan(data)]))
        self.signal_channel = data.shape[-1]

        # `truncated` keeps the same sampled timestamps but reduces dataset
        # length by stride and then indexes the truncated sequence densely.
        data_len = data.shape[0]
        if truncated:
            data_len = data_len // self.stride
            self.stride = 1

        self.data, self.data_begin = _slice_by_split(
            data[:data_len],
            split,
            TRAFFIC_SPLIT_RATIOS,
        )

        self.df = pd.read_csv(os.path.join(data_folder, graph_csv), index_col=None)
        if id_file is not None:
            sensor_id = np.loadtxt(os.path.join(data_folder, id_file), dtype=int)
            self.n_nodes = sensor_id.shape[0]
            self.sensor_dict = dict([(sensor_id[k], k) for k in range(self.n_nodes)])
        else:
            self.sensor_dict = None
            self.n_nodes = max(max(self.df["from"].values), max(self.df["to"].values)) + 1

        self.n_edges, self.u_edges, self.u_distance = physical_graph(self.df, self.sensor_dict)
        self.u_edges = torch.Tensor(self.u_edges).type(torch.long)
        self.u_distance = torch.Tensor(self.u_distance)

        # d_edges adds self edges for temporal/directed helper logic in older code.
        self.d_edges = torch.cat(
            [self.u_edges, torch.arange(0, self.n_nodes)[:, None] + torch.zeros((2,), dtype=torch.long)],
            0,
        )
        self.graph_info = _to_graph_info(self.n_nodes, self.u_edges, self.u_distance)

    def __len__(self):
        return _window_length(self.data.shape[0], self.T, self.stride)
    
    def __getitem__(self, index):
        start = index * self.stride
        y = self.data[start : start + self.t]
        x = self.data[start : start + self.T]

        if self.use_one_channel:
            x = x[..., 0:1]

        time = _time_index(self.T, index, self.stride, self.data_begin)
        if self.return_time:
            return torch.Tensor(y), torch.Tensor(x), time
        return torch.Tensor(y), torch.Tensor(x)


def directed_physical_graph(adj_mat, squared_dist=False):
    """Convert a weighted adjacency matrix to directed edge list and distances.

    Positive off-diagonal entries are interpreted as similarities. Distances are
    transformed by `-log(weight)`, or `sqrt(-log(weight))` when squared_dist is
    True. Isolated nodes are connected to adjacent indices with unit distance so
    downstream graph routines do not receive empty neighborhoods.
    """
    u_edges = []
    u_distance = []
    print("original edges", (adj_mat > 0).sum())
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if i != j and adj_mat[i, j] > 0: 
                u_edges.append([i, j])
                if squared_dist: 
                    u_distance.append(np.sqrt(-np.log(adj_mat[i, j])))
                else:
                    u_distance.append(-np.log(adj_mat[i, j]))
    
    print("uedges original", len(u_distance))
    for i in range(adj_mat.shape[0]):
        adj_mat[i, i] = 0
    # isolated nodes
    n_isolated = 0
    for i in range(adj_mat.shape[0]):
        if np.sum(adj_mat[i]) == 0:
            n_isolated += 1
            if i != 0:
                u_edges.append([i, i-1])
                u_edges.append([i-1, i])
                u_distance.append(1)
                u_distance.append(1)
            if i != adj_mat.shape[0] - 1:
                u_edges.append([i, i+1])
                u_edges.append([i+1, i])
                u_distance.append(1)
                u_distance.append(1)
    print("n_isolated", n_isolated)

    n_edges = len(u_edges)
    u_edges = np.array(u_edges)
    u_distance = np.array(u_distance)
    return n_edges, u_edges, u_distance


class DirectedTrafficDataset(Dataset):
    """Traffic dataset backed by a dense directed adjacency matrix.

    Raw data shape:
        (T_total, n_nodes) or (T_total, n_nodes, n_channels)

    Splits use train/val/test = 6:2:2.
    """

    def __init__(
        self,
        data_folder,
        adj_mat_file,
        data_file,
        T,
        t,
        stride,
        split="train",
        n_nodes=None,
        return_time=False,
        use_one_channel=False,
    ):
        super().__init__()
        self.T = T
        self.t = t
        self.stride = stride
        self.return_time = return_time
        self.use_one_channel = use_one_channel

        data = np.load(os.path.join(data_folder, data_file))
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)

        if torch.isnan(torch.Tensor(data)).any():
            print("data has nan")
            nan_indices = np.argwhere(np.isnan(data))
            print("nan indices:", nan_indices)

        print("nan_count", len(data[np.isnan(data)]))
        self.signal_channel = data.shape[-1]
        print("signal channel", self.signal_channel)
        self.data, self.data_begin = _slice_by_split(data, split, TRAFFIC_SPLIT_RATIOS)

        self.adj_mat = np.load(os.path.join(data_folder, adj_mat_file))
        # Symmetrize traffic adjacency before converting it to directed edges.
        self.adj_mat = np.maximum.reduce([self.adj_mat, self.adj_mat.T])
        self.n_nodes = self.adj_mat.shape[0]
        self.n_edges, self.u_edges, self.u_distance = directed_physical_graph(self.adj_mat, squared_dist=True)
        self.u_edges = torch.Tensor(self.u_edges).type(torch.long)

        n_edges = self.u_edges.size(0)
        edge_counts = Counter([(self.u_edges[i, 0], self.u_edges[i, 1]) for i in range(n_edges)])
        assert max(list(edge_counts.values())), "distance graph asymmetric"

        self.u_distance = torch.Tensor(self.u_distance)
        self.d_edges = torch.cat(
            [self.u_edges, torch.arange(0, self.n_nodes)[:, None] + torch.zeros((2,), dtype=torch.long)],
            0,
        )
        self.graph_info = _to_graph_info(self.n_nodes, self.u_edges, self.u_distance)

    def __len__(self):
        return _window_length(self.data.shape[0], self.T, self.stride)

    def __getitem__(self, index):
        start = index * self.stride
        y = self.data[start : start + self.t]
        x = self.data[start : start + self.T]

        if self.use_one_channel:
            x = x[..., 0:1]

        time = torch.arange(0, self.T, dtype=torch.long) + index + self.data_begin
        if self.return_time:
            return torch.Tensor(y), torch.Tensor(x), time
        return torch.Tensor(y), torch.Tensor(x)
        

class WeatherDataset(Dataset):
    """Weather dataset with adjacency matrix graph.

    Pickle file is expected to contain key `all` with shape:
        (T_total, n_nodes)

    Splits use train/val/test = 35%/15%/50%, matching the original experiment.
    """

    def __init__(
        self,
        data_folder,
        adj_file,
        data_file,
        T,
        t=10,
        stride=1,
        split="train",
        return_time=False,
        use_one_channel=False,
    ):
        super().__init__()
        self.T = T
        self.t = t
        assert T - t < 6, "T - t should be in 1, 2, 3, 4, 5"
        self.stride = stride
        self.return_time = return_time
        self.use_one_channel = use_one_channel

        data = pickle.load(open(os.path.join(data_folder, data_file), "rb"))
        data = np.expand_dims(data["all"], axis=-1)
        
        print("nan_count", len(data[np.isnan(data)]))
        self.signal_channel = data.shape[-1]
        self.data, self.data_begin = _slice_by_split(data, split, WEATHER_SPLIT_RATIOS)

        self.adj_mat = np.load(os.path.join(data_folder, adj_file))
        self.n_nodes = self.adj_mat.shape[0]
        self.n_edges, self.u_edges, self.u_distance = directed_physical_graph(self.adj_mat, squared_dist=False)
        self.u_edges = torch.Tensor(self.u_edges).type(torch.long)
        self.u_distance = torch.Tensor(self.u_distance)
        self.d_edges = torch.cat(
            [self.u_edges, torch.arange(0, self.n_nodes)[:, None] + torch.zeros((2,), dtype=torch.long)],
            0,
        )
        self.graph_info = _to_graph_info(self.n_nodes, self.u_edges, self.u_distance)
    
    def __len__(self):
        return self.data.shape[0] - self.T
    
    def __getitem__(self, index):
        start = index * self.stride
        y = self.data[start : start + self.t]
        x = self.data[start : start + self.T]

        if self.use_one_channel:
            x = x[..., 0:1]

        time = _time_index(self.T, index, self.stride, self.data_begin)
        if self.return_time:
            return torch.Tensor(y), torch.Tensor(x), time
        return torch.Tensor(y), torch.Tensor(x)
        
