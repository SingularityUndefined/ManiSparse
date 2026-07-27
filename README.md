# ManiSparse
# Lightweight Transformer via Unrolling of Mixed Graph Algorithms for Traffic Forecast
## Requirements

Required packages for this implementation:

```
torch>=2.4.1
tqdm 
numpy 
matplotlib 
networkx>=2.5
pandas
pyyaml
tensorboard
```
Install the dependencies in your Python environment before running training.

## Datasets
**PEMS0X datasets** are from repository [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data). The training code expects them under `../TS_datasets/PEMS0X_data`:

<!-- PEMS-BAY and METR-LA datasets are from repository [DCRNN](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph). -->
```
../TS_datasets/
├── PEMS0X_data/
│   ├── PEMS03/
│   ├── PEMS04/
│   ├── PEMS07/
│   └── PEMS08/
├── PEMS-BAY/
└── METR-LA/
```
**PEMS-BAY** and **METR-LA** dataset are preprocessed from repository [DCRNN](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph). Each folder contains two `.npy` files for adjacency matrix and time series data.

We also provide our dataset together with [Google Drive](https://drive.google.com/drive/folders/1tMgyxzQ_dio73rapQ-LYSOOIXwXFFUJw?usp=share_link). Download from this link and put it outside this repository folder as `../TS_datasets`.

## Training and Testing

The default settings are in `train/config.yaml`. We provide multiple parsers to change the configurations. Run the commands from the repository root.

**Example 1**: running main experiment on PEMS03 dataset:
```
python -m train.train_traffic --dataset PEMS03 --cuda 0 --batchsize 12 --le-emb --neighbors 4
```

**Example 2**: running 'w/o DGLR' experiment on METR-LA dataset:
```
python -m train.train_traffic --dataset METR-LA --cuda 1 --ablation DGLR --batchsize 16 --le-emb
```

**Example 3**: running 'w/o undirected temporal graph' experiment on PEMS-BAY:
```
python -m train.train_traffic --dataset PEMS-BAY --cuda 0 --ablation UT --batchsize 64 --le-emb
```
