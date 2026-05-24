#!/usr/bin/env bash
# Submit throttled SLURM array for one grid (fair_1bit or vector).
#
# Submits one array job per network size n (shorter walltime per tier -> faster queue).
#
# Usage:
#   bash scripts/slurm/run_batch_slurms.sh [MAX_PARALLEL] [COMM_MODE]
#   bash scripts/slurm/run_batch_slurms.sh 20 fair_1bit
#   bash scripts/slurm/run_batch_slurms.sh 10 vector
#
# Smoke subset:
#   SEEDS=0,1 NUM_NODES_LIST=10 SIGNAL_QUALITY_LIST=0.6 \
#     bash scripts/slurm/run_batch_slurms.sh 4 fair_1bit
#
# Walltime overrides (optional; bump only the tier that times out):
#   SBATCH_TIME_N10=02:00:00 SBATCH_TIME_N100=04:00:00 SBATCH_TIME_N1000=08:00:00 \
#     bash scripts/slurm/run_batch_slurms.sh 20 fair_1bit
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MAX_PARALLEL="${1:-20}"
COMMUNICATION_MODE="${2:-fair_1bit}"
CONFIG="${CONFIG:-configs/default.yaml}"
SBATCH_FILE="scripts/slurm/job.sh"

SEEDS="${SEEDS:-}"
NUM_NODES_LIST="${NUM_NODES_LIST:-10,100,1000}"
FULL_NUM_NODES_LIST="${NUM_NODES_LIST}"
SIGNAL_QUALITY_LIST="${SIGNAL_QUALITY_LIST:-0.55,0.6,0.7,0.8}"

export WANDB_PROJECT="${WANDB_PROJECT:-game-theory-project}"
export WANDB_ENTITY="${WANDB_ENTITY:-GameHSTGAT}"
export CONFIG COMMUNICATION_MODE SEEDS SIGNAL_QUALITY_LIST

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

walltime_for_n() {
  local n="$1"
  case "$n" in
    1000) echo "${SBATCH_TIME_N1000:-08:00:00}" ;;
    100) echo "${SBATCH_TIME_N100:-04:00:00}" ;;
    10) echo "${SBATCH_TIME_N10:-02:00:00}" ;;
    *) echo "${SBATCH_TIME:-02:00:00}" ;;
  esac
}

mkdir -p logs

IFS=',' read -ra NODES_ARR <<< "$NUM_NODES_LIST"
ARRAY_JOBS=()

echo "=========================================="
echo "SLURM grid submission (one array per n)"
echo "  mode:          ${COMMUNICATION_MODE}"
echo "  config:        ${CONFIG}"
echo "  max parallel:  ${MAX_PARALLEL}"
echo "  artifacts:     ${ARTIFACTS_ROOT}"
echo "=========================================="

for n in "${NODES_ARR[@]}"; do
  n="${n// /}"
  if [[ -z "$n" ]]; then
    continue
  fi

  export NUM_NODES_LIST="$n"
  SBATCH_TIME="$(walltime_for_n "$n")"

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
    echo "[WARN] Skipping n=${n}: zero tasks."
    continue
  fi

  echo ""
  echo "[INFO] Submitting n=${n}: ${NUM_TASKS} tasks, walltime=${SBATCH_TIME}"

  ARRAY_JOB="$(sbatch --parsable \
    --job-name="HSTGAT_n${n}" \
    --time="${SBATCH_TIME}" \
    --array="1-${NUM_TASKS}%${MAX_PARALLEL}" \
    --export=ALL \
    "$SBATCH_FILE")"

  echo "[SUCCESS] Array job n=${n}: ${ARRAY_JOB}"
  ARRAY_JOBS+=("$ARRAY_JOB")
done

if [[ "${#ARRAY_JOBS[@]}" -eq 0 ]]; then
  echo "[ERROR] No array jobs submitted."
  exit 1
fi

if [[ "${SUBMIT_FINALIZE:-1}" == "1" ]]; then
  DEPENDENCY="afterok"
  for job_id in "${ARRAY_JOBS[@]}"; do
    DEPENDENCY="${DEPENDENCY}:${job_id}"
  done

  export NUM_NODES_LIST="${FULL_NUM_NODES_LIST}"

  FINALIZE_JOB="$(sbatch --parsable \
    --dependency="${DEPENDENCY}" \
    --export=ALL,ARRAY_JOB_ID="${ARRAY_JOBS[*]}",NUM_NODES_LIST="${FULL_NUM_NODES_LIST}" \
    scripts/slurm/finalize_job.sh)"
  echo ""
  echo "[SUCCESS] Finalize job: ${FINALIZE_JOB} (depends on ${#ARRAY_JOBS[@]} array job(s))"
fi
