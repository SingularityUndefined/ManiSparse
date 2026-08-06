python -m train.train_traffic --dataset PEMS08 --cuda 0 --batchsize 16 --lr 5e-4 --theta-method glasso --glasso-backend admm

python -m train.train_traffic --dataset PEMS08 --cuda 0 --batchsize 20 --lr 5e-4 --theta-method kalofolias --kalofolias-graph local --theta-neighbors 10