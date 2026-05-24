#!/bin/bash
#SBATCH -J HSTGAT_finalize
#SBATCH --account LIO-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

echo "=== Finalize Start: $(date) ==="
echo "Array job id: ${ARRAY_JOB_ID:-unknown}"

REPO_DIR="${REPO_DIR:-/rds/user/lmk66/hpc-work/GameHSTGAT}"
cd "$REPO_DIR"

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source ~/.bashrc
conda activate poly-shgnn-gpu

cmd=(
  python scripts/finalize_grid.py
  --config "${CONFIG:-configs/default.yaml}"
  --artifacts-root "${ARTIFACTS_ROOT:-artifacts/training_metrics_fair/grid_runs}"
  --output "${SUMMARY_JSON:-artifacts/grid_summary_fair.json}"
  --metrics-csv "${METRICS_CSV:-artifacts/metrics_summary_fair.csv}"
  --aggregate-csv "${METRICS_AGG_CSV:-artifacts/metrics_summary_fair_aggregated.csv}"
  --communication-mode "${COMMUNICATION_MODE:-fair_1bit}"
  --num-nodes-list "${NUM_NODES_LIST:-10,100,1000}"
  --signal-quality-list "${SIGNAL_QUALITY_LIST:-0.55,0.6,0.7,0.8}"
)

if [[ -n "${SEEDS:-}" ]]; then
  cmd+=(--seeds "$SEEDS")
fi

if [[ -n "${COMMUNICATION_DIM:-}" ]]; then
  cmd+=(--communication-dim "$COMMUNICATION_DIM")
fi

"${cmd[@]}"
status=$?

echo "=== Finalize End: $(date) (exit=${status}) ==="
exit "$status"
