#!/usr/bin/env bash
# Run the ADMM-penalty projection ablation sequentially on one GPU.
# All runs use the same current YAML optimizer and weight decay. The ablation
# varies backward mode, post-step ADMM-penalty projection, and the spatial
# mu_u/lambda_theta parameterization.
set -euo pipefail

cd "$(dirname "$0")"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-3407}"
THETA_NEIGHBORS="${THETA_NEIGHBORS:-10}"
LOGS_DIR="${LOGS_DIR:-logs_projection_ablation}"

COMMON=(
  --config train/config.yaml
  --logs-dir "$LOGS_DIR"
  --dataset PEMS08
  --cuda "$CUDA_DEVICE"
  --batchsize "$BATCH_SIZE"
  --lr "$LEARNING_RATE"
  --epochs "$EPOCHS"
  --seed "$SEED"
  --neighbors 4
  --theta-method kalofolias
  --kalofolias-graph local
  --theta-neighbors "$THETA_NEIGHBORS"
)

run_experiment() {
  local label="$1"
  shift
  echo "===== ${label} ====="
  python -m train.train_traffic "${COMMON[@]}" "$@"
}

# A/B: detached, coupled spatial penalties.
run_experiment "A: detached + projection + coupled spatial penalties" \
  --no-kalofolias-allow-backward --no-deflation-allow-backward \
  --project-admm-penalties --couple-spatial-penalties
run_experiment "B: detached + no projection + coupled spatial penalties" \
  --no-kalofolias-allow-backward --no-deflation-allow-backward \
  --no-project-admm-penalties --couple-spatial-penalties

# C/D: full backward, coupled spatial penalties.
run_experiment "C: full backward + projection + coupled spatial penalties" \
  --kalofolias-allow-backward --deflation-allow-backward \
  --project-admm-penalties --couple-spatial-penalties
run_experiment "D: full backward + no projection + coupled spatial penalties" \
  --kalofolias-allow-backward --deflation-allow-backward \
  --no-project-admm-penalties --couple-spatial-penalties

# E-H repeat the same four conditions with independent, unconstrained mu_u
# and lambda_theta. These runs can become numerically unstable; the training
# diagnostics will stop and report the failing tensors if that occurs.
run_experiment "E: detached + projection + independent spatial penalties" \
  --no-kalofolias-allow-backward --no-deflation-allow-backward \
  --project-admm-penalties --no-couple-spatial-penalties
run_experiment "F: detached + no projection + independent spatial penalties" \
  --no-kalofolias-allow-backward --no-deflation-allow-backward \
  --no-project-admm-penalties --no-couple-spatial-penalties
run_experiment "G: full backward + projection + independent spatial penalties" \
  --kalofolias-allow-backward --deflation-allow-backward \
  --project-admm-penalties --no-couple-spatial-penalties
run_experiment "H: full backward + no projection + independent spatial penalties" \
  --kalofolias-allow-backward --deflation-allow-backward \
  --no-project-admm-penalties --no-couple-spatial-penalties
