#!/usr/bin/env bash
# Sweep Local Kalofolias solver iterations and deflation mode counts on PEMS08.
#
# Required environment variables:
#   CUDA_DEVICE, BATCH_SIZE, KNN, THETA_KNN, INTERVAL
#
# Example:
#   CUDA_DEVICE=0 BATCH_SIZE=8 KNN=4 THETA_KNN=8 INTERVAL=6 \
#     bash exp_pems08_kalofolias_iter_modes_sweep.sh

set -uo pipefail

cd "$(dirname "$0")"

: "${CUDA_DEVICE:?Set CUDA_DEVICE, for example CUDA_DEVICE=0}"
: "${BATCH_SIZE:?Set BATCH_SIZE, for example BATCH_SIZE=8}"
: "${KNN:?Set KNN, for example KNN=4}"
: "${THETA_KNN:?Set THETA_KNN, for example THETA_KNN=8}"
: "${INTERVAL:?Set INTERVAL, for example INTERVAL=6}"

CONFIG_PATH="${CONFIG_PATH:-train/config.yaml}"
LOGS_DIR="${LOGS_DIR:-logs_pems08_kalofolias_iter_modes_sweep}"

if (( THETA_KNN <= KNN )); then
  echo "THETA_KNN must be larger than KNN for Local Kalofolias." >&2
  exit 2
fi

KALOFOLIAS_ITERS=(100 150 200)
DEFLATION_MODES=(3 5 10)
STATUS_FILE="${LOGS_DIR}/sweep_status.tsv"
failure_count=0
success_count=0

mkdir -p "$LOGS_DIR"
printf 'max_iter\tdeflation_modes\tstatus\texit_code\treason\tlauncher_log\n' > "$STATUS_FILE"

for max_iter in "${KALOFOLIAS_ITERS[@]}"; do
  for deflation_modes in "${DEFLATION_MODES[@]}"; do
    echo "Running PEMS08: max_iter=${max_iter}, deflation_modes=${deflation_modes}"

    # max_iter is separated at the log-root level because the common
    # experiment-name builder does not include this solver setting.
    run_logs_dir="${LOGS_DIR}/max_iter_${max_iter}"
    launcher_log="${run_logs_dir}/launcher_mode_${deflation_modes}.log"
    mkdir -p "$run_logs_dir"

    python -m train.train_traffic \
      --config "$CONFIG_PATH" \
      --logs-dir "$run_logs_dir" \
      --dataset PEMS08 \
      --cuda "$CUDA_DEVICE" \
      --batchsize "$BATCH_SIZE" \
      --neighbors "$KNN" \
      --theta-neighbors "$THETA_KNN" \
      --interval "$INTERVAL" \
      --kalofolias-alpha 0.3 \
      --kalofolias-beta 1.0 \
      --no-kalofolias-learnable-alpha-beta \
      --kalofolias-max-iter "$max_iter" \
      --deflation-samples "$deflation_modes" \
      2>&1 | tee "$launcher_log"
    exit_code=${PIPESTATUS[0]}

    if (( exit_code == 0 )); then
      status="success"
      reason="none"
      ((success_count += 1))
      echo "Completed: max_iter=${max_iter}, deflation_modes=${deflation_modes}"
    else
      status="failed"
      if grep -Eqi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "$launcher_log"; then
        reason="cuda_oom"
      elif grep -Eqi 'Numerical failure|gradient has NaN or Inf|SqrtBackward0.*nan' "$launcher_log"; then
        reason="numerical_nan"
      elif (( exit_code == 137 )); then
        reason="killed_or_host_oom"
      elif (( exit_code == 143 )); then
        reason="terminated"
      else
        reason="process_error"
      fi
      ((failure_count += 1))
      echo "Failed (${reason}, exit=${exit_code}): max_iter=${max_iter}, deflation_modes=${deflation_modes}" >&2
      echo "See launcher log: ${launcher_log}" >&2
      echo "Continuing with the next experiment." >&2
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$max_iter" "$deflation_modes" "$status" "$exit_code" "$reason" "$launcher_log" \
      >> "$STATUS_FILE"
  done
done

echo "Sweep finished: ${success_count} succeeded, ${failure_count} failed."
echo "Summary: ${STATUS_FILE}"

if (( failure_count > 0 )); then
  exit 1
fi
