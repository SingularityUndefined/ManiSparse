#!/usr/bin/env bash
# Run only condition C with the independent mu_u/lambda_theta ablation.
set -euo pipefail

cd "$(dirname "$0")"

# Full backward retains the Kalofolias and multi-mode deflation graphs. Batch
# size 8 is the known completed PEMS08 setting and is the safe starting point.
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
EPOCHS="${EPOCHS:-70}"
SEED="${SEED:-3407}"
THETA_NEIGHBORS="${THETA_NEIGHBORS:-8}"
LOGS_DIR="${LOGS_DIR:-logs_projection_ablation_c_only}"
# 0 restores independent, unconstrained mu_u and lambda_theta. Set to 1 for
# the positive-total/sigmoid-split baseline.
COUPLE_SPATIAL_PENALTIES="${COUPLE_SPATIAL_PENALTIES:-0}"

if [[ "$COUPLE_SPATIAL_PENALTIES" == "1" ]]; then
  SPATIAL_PENALTY_FLAG="--couple-spatial-penalties"
else
  SPATIAL_PENALTY_FLAG="--no-couple-spatial-penalties"
fi

python -m train.train_traffic \
  --config train/config.yaml \
  --logs-dir "$LOGS_DIR" \
  --dataset PEMS08 \
  --cuda "$CUDA_DEVICE" \
  --batchsize "$BATCH_SIZE" \
  --lr "$LEARNING_RATE" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --neighbors 4 \
  --theta-neighbors "$THETA_NEIGHBORS" \
  --project-admm-penalties \
  "$SPATIAL_PENALTY_FLAG"
