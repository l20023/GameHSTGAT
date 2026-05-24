#!/usr/bin/env bash
# Submit throttled SLURM array for one grid (fair_1bit or vector).
#
# Usage:
#   bash scripts/slurm/run_batch_slurms.sh [MAX_PARALLEL] [COMM_MODE]
#   bash scripts/slurm/run_batch_slurms.sh 20 fair_1bit
#   bash scripts/slurm/run_batch_slurms.sh 10 vector
#
# Smoke subset:
#   SEEDS=0,1 NUM_NODES_LIST=10 SIGNAL_QUALITY_LIST=0.6 \
#     bash scripts/slurm/run_batch_slurms.sh 4 fair_1bit
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MAX_PARALLEL="${1:-20}"
COMMUNICATION_MODE="${2:-fair_1bit}"
CONFIG="${CONFIG:-configs/default.yaml}"
SBATCH_FILE="scripts/slurm/job.sh"

SEEDS="${SEEDS:-}"
NUM_NODES_LIST="${NUM_NODES_LIST:-10,100,1000}"
SIGNAL_QUALITY_LIST="${SIGNAL_QUALITY_LIST:-0.55,0.6,0.7,0.8}"

export WANDB_PROJECT="${WANDB_PROJECT:-game-theory-project}"
export WANDB_ENTITY="${WANDB_ENTITY:-GameHSTGAT}"
export CONFIG COMMUNICATION_MODE SEEDS NUM_NODES_LIST SIGNAL_QUALITY_LIST

case "$COMMUNICATION_MODE" in
  fair_1bit)
    ARTIFACTS_ROOT="artifacts/training_metrics_fair/grid_runs"
    SUMMARY_JSON="artifacts/grid_summary_fair.json"
    METRICS_CSV="artifacts/metrics_summary_fair.csv"
    METRICS_AGG_CSV="artifacts/metrics_summary_fair_aggregated.csv"
    ;;
  vector)
    ARTIFACTS_ROOT="artifacts/training_metrics_vector/grid_runs"
    SUMMARY_JSON="artifacts/grid_summary_vector.json"
    METRICS_CSV="artifacts/metrics_summary_vector.csv"
    METRICS_AGG_CSV="artifacts/metrics_summary_vector_aggregated.csv"
    export COMMUNICATION_DIM="${COMMUNICATION_DIM:-32}"
    ;;
  *)
    echo "[ERROR] Unknown communication mode: $COMMUNICATION_MODE"
    exit 1
    ;;
esac

export ARTIFACTS_ROOT SUMMARY_JSON METRICS_CSV METRICS_AGG_CSV

count_args=(
  python scripts/run_grid_task.py
  --count
  --config "$CONFIG"
  --num-nodes-list "$NUM_NODES_LIST"
  --signal-quality-list "$SIGNAL_QUALITY_LIST"
)
if [[ -n "$SEEDS" ]]; then
  count_args+=(--seeds "$SEEDS")
fi

NUM_TASKS="$("${count_args[@]}")"
if [[ "$NUM_TASKS" -lt 1 ]]; then
  echo "[ERROR] Grid has zero tasks."
  exit 1
fi

mkdir -p logs

echo "=========================================="
echo "SLURM grid submission"
echo "  mode:          ${COMMUNICATION_MODE}"
echo "  config:        ${CONFIG}"
echo "  tasks:         ${NUM_TASKS}"
echo "  max parallel:  ${MAX_PARALLEL}"
echo "  artifacts:     ${ARTIFACTS_ROOT}"
echo "=========================================="

ARRAY_JOB="$(sbatch --parsable \
  --array="1-${NUM_TASKS}%${MAX_PARALLEL}" \
  --export=ALL \
  "$SBATCH_FILE")"

echo "[SUCCESS] Array job: ${ARRAY_JOB}"

if [[ "${SUBMIT_FINALIZE:-1}" == "1" ]]; then
  FINALIZE_JOB="$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB}" \
    --export=ALL,ARRAY_JOB_ID="${ARRAY_JOB}" \
    scripts/slurm/finalize_job.sh)"
  echo "[SUCCESS] Finalize job: ${FINALIZE_JOB}"
fi
