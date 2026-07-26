"""Dataset construction and normalization utilities.

These helpers are separated from the project-level `utils.py` so dataset
loading code lives next to the dataset classes.  The public function signatures
match the old root `utils.py` versions.
"""

import os

import torch
from torch.utils.data import DataLoader

from dataset.load_dataset import DirectedTrafficDataset, TrafficDataset, WeatherDataset


def create_dataloader(
    dataset_dir,
    dataset_name,
    T,
    t_in,
    stride,
    batch_size,
    num_workers,
    return_time,
    use_one_channel=False,
    truncated=False,
):
    data_folder = os.path.join(dataset_dir, dataset_name)
    graph_csv = dataset_name + ".csv"
    data_file = dataset_name + ".npz"
    if dataset_name == "PEMS03":
        id_file = dataset_name + ".txt"
    else:
        id_file = None

    train_set = TrafficDataset(
        data_folder,
        graph_csv,
        data_file,
        T,
        t_in,
        stride,
        "train",
        id_file=id_file,
        return_time=return_time,
        use_one_channel=use_one_channel,
        truncated=truncated,
    )
    val_set = TrafficDataset(
        data_folder,
        graph_csv,
        data_file,
        T,
        t_in,
        stride,
        "val",
        id_file=id_file,
        return_time=return_time,
        use_one_channel=use_one_channel,
        truncated=truncated,
    )
    test_set = TrafficDataset(
        data_folder,
        graph_csv,
        data_file,
        T,
        t_in,
        stride,
        "test",
        id_file=id_file,
        return_time=return_time,
        use_one_channel=use_one_channel,
        truncated=truncated,
    )

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def create_directed_dataloader(
    dataset_dir,
    dataset_name,
    T,
    t_in,
    stride,
    batch_size,
    num_workers,
    return_time,
    use_one_channel=False,
):
    data_folder = os.path.join(dataset_dir, dataset_name)

    if dataset_name == "PEMS-BAY":
        adj_mat_name = "pems_adj_mat.npy"
        data_file_name = "pems_node_values.npy"
    elif dataset_name == "METR-LA":
        adj_mat_name = "adj_mat.npy"
        data_file_name = "node_values.npy"
    else:
        adj_mat_name = dataset_name + "_rn_adj.npy"
        data_file_name = dataset_name + "_his_2019.npy"

    train_set = DirectedTrafficDataset(
        data_folder,
        adj_mat_name,
        data_file_name,
        T,
        t_in,
        stride,
        "train",
        return_time=return_time,
        use_one_channel=use_one_channel,
    )
    val_set = DirectedTrafficDataset(
        data_folder,
        adj_mat_name,
        data_file_name,
        T,
        t_in,
        stride,
        "val",
        return_time=return_time,
        use_one_channel=use_one_channel,
    )
    test_set = DirectedTrafficDataset(
        data_folder,
        adj_mat_name,
        data_file_name,
        T,
        t_in,
        stride,
        "test",
        return_time=return_time,
        use_one_channel=use_one_channel,
    )

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def create_weather_dataloader(
    dataset_dir,
    dataset_name,
    T,
    t_in,
    stride,
    batch_size,
    num_workers,
    return_time,
    use_one_channel=False,
):
    data_folder = os.path.join(dataset_dir, dataset_name, "dataset/processed")
    assert dataset_name in ["Molene", "NOAA"]
    if dataset_name == "Molene":
        data_filename = "dataset_w=10_steps=[1, 2, 3, 4, 5]_splits=[0.35, 0.15, 0.5].pickle"
        adj_filename = "weighted_adjacency.npy"
    else:
        data_filename = "NOA_w=10_steps=[1, 2, 3, 4, 5]_splits=[0.35, 0.15, 0.5].pickle"
        adj_filename = "weighted_adj.npy"

    train_set = WeatherDataset(
        data_folder,
        adj_filename,
        data_filename,
        T,
        t_in,
        stride,
        "train",
        return_time=return_time,
        use_one_channel=use_one_channel,
    )
    val_set = WeatherDataset(
        data_folder,
        adj_filename,
        data_filename,
        T,
        t_in,
        stride,
        "val",
        return_time=return_time,
        use_one_channel=use_one_channel,
    )
    test_set = WeatherDataset(
        data_folder,
        adj_filename,
        data_filename,
        T,
        t_in,
        stride,
        "test",
        return_time=return_time,
        use_one_channel=use_one_channel,
    )

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_set, val_set, test_set, train_loader, val_loader, test_loader


class Normalization:
    """Normalize or standardize tensors using statistics from a dataset.

    Expected data shape in `dataset.data`:
        (time, n_nodes, n_channels)

    Runtime tensor shapes are usually:
        (batch, T, n_nodes, n_channels)
    """

    def __init__(self, dataset: TrafficDataset, mode: str, device):
        assert mode in ["normalize", "standardize"], "mode should be in [normalize, standardize]"
        self.mode = mode
        if mode == "standardize":
            self.mean = torch.Tensor(dataset.data.mean(0)).to(device)
            self.std = torch.Tensor(dataset.data.std(0)).to(device)
        elif mode == "normalize":
            self.min = torch.Tensor(dataset.data.min(0)).to(device)
            self.max = torch.Tensor(dataset.data.max(0)).to(device)

    def normalize_data(self, x, use_one_channel=False):
        if self.mode == "standardize":
            if use_one_channel:
                return torch.where(
                    self.std[..., 0:1] != 0,
                    (x - self.mean[..., 0:1]) / self.std[..., 0:1],
                    torch.zeros_like(x),
                )
            return torch.where(self.std != 0, (x - self.mean) / self.std, torch.zeros_like(x))

        if use_one_channel:
            return (x - self.min[..., 0:1]) / (self.max[..., 0:1] - self.min[..., 0:1])
        return (x - self.min) / (self.max - self.min)

    def recover_data(self, x, use_one_channel=False):
        if use_one_channel:
            if self.mode == "standardize":
                return x * self.std[..., 0:1] + self.mean[..., 0:1]
            return x * (self.max[..., 0:1] - self.min[..., 0:1]) + self.min[..., 0:1]

        if self.mode == "standardize":
            return x * self.std + self.mean
        return x * (self.max - self.min) + self.min
