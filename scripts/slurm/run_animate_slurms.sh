#!/usr/bin/env bash
# Submit SLURM arrays to train one seed, save checkpoints, and render GIFs
# for all q=0.55 grid cells (3 topologies × each n tier).
#
# One array job per network size n (3 tasks = complete, ws_p_0.0, ws_p_0.1).
#
# Usage:
#   SEED=0 bash scripts/slurm/run_animate_slurms.sh [MAX_PARALLEL] [COMM_MODE]
#   SEED=0 bash scripts/slurm/run_animate_slurms.sh 3 fair_1bit
#   SEED=4 bash scripts/slurm/run_animate_slurms.sh 3 vector
#
# Subset (single n):
#   SEED=0 NUM_NODES_LIST=10 bash scripts/slurm/run_animate_slurms.sh 3 fair_1bit
#
# Walltime overrides:
#   SBATCH_TIME_N10=03:00:00 SBATCH_TIME_N100=06:00:00 SBATCH_TIME_N1000=12:00:00 \
#     SEED=0 bash scripts/slurm/run_animate_slurms.sh 3 fair_1bit
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MAX_PARALLEL="${1:-3}"
COMMUNICATION_MODE="${2:-fair_1bit}"
SEED="${SEED:-0}"
SIGNAL_QUALITY="${SIGNAL_QUALITY:-0.55}"
NUM_NODES_LIST="${NUM_NODES_LIST:-10,100,1000}"
SBATCH_FILE="scripts/slurm/animate_job.sh"

export SEED SIGNAL_QUALITY COMMUNICATION_MODE

case "$COMMUNICATION_MODE" in
  fair_1bit)
    ;;
  vector)
    export COMMUNICATION_DIM="${COMMUNICATION_DIM:-32}"
    ;;
  *)
    echo "[ERROR] Unknown communication mode: $COMMUNICATION_MODE"
    exit 1
    ;;
esac

walltime_for_n() {
  local n="$1"
  case "$n" in
    1000) echo "${SBATCH_TIME_N1000:-12:00:00}" ;;
    100) echo "${SBATCH_TIME_N100:-06:00:00}" ;;
    10) echo "${SBATCH_TIME_N10:-03:00:00}" ;;
    *) echo "${SBATCH_TIME:-03:00:00}" ;;
  esac
}

mkdir -p logs

IFS=',' read -ra NODES_ARR <<< "$NUM_NODES_LIST"
ARRAY_JOBS=()
NUM_TOPOS=3

echo "=========================================="
echo "SLURM animation submission (one seed)"
echo "  seed:          ${SEED}"
echo "  q:             ${SIGNAL_QUALITY}"
echo "  mode:          ${COMMUNICATION_MODE}"
echo "  max parallel:  ${MAX_PARALLEL}"
echo "  nodes list:    ${NUM_NODES_LIST}"
echo "=========================================="

for n in "${NODES_ARR[@]}"; do
  n="${n// /}"
  if [[ -z "$n" ]]; then
    continue
  fi

  export NUM_NODES="$n"
  SBATCH_TIME="$(walltime_for_n "$n")"

  echo ""
  echo "[INFO] Submitting n=${n}: ${NUM_TOPOS} topology tasks, walltime=${SBATCH_TIME}"

  ARRAY_JOB="$(sbatch --parsable \
    --job-name="HSTGAT_anim_n${n}_s${SEED}" \
    --time="${SBATCH_TIME}" \
    --array="1-${NUM_TOPOS}%${MAX_PARALLEL}" \
    --export=ALL,NUM_NODES="${n}" \
    "$SBATCH_FILE")"

  echo "[SUCCESS] Array job n=${n}: ${ARRAY_JOB}"
  ARRAY_JOBS+=("$ARRAY_JOB")
done

if [[ "${#ARRAY_JOBS[@]}" -eq 0 ]]; then
  echo "[ERROR] No array jobs submitted."
  exit 1
fi

echo ""
echo "Submitted ${#ARRAY_JOBS[@]} array job(s): ${ARRAY_JOBS[*]}"
echo "Outputs:"
echo "  checkpoints -> artifacts/checkpoints/${COMMUNICATION_MODE}/n_*/q_*/<topo>/seed_${SEED}.pt"
echo "  animations  -> artifacts/animations/${COMMUNICATION_MODE}/n_*/q_*/<topo>/seed_${SEED}.gif"
