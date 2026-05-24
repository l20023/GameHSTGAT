#!/bin/bash
#SBATCH -J HSTGAT
#SBATCH --account LIO-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Fallback; run_batch_slurms.sh overrides --time per n (2h / 4h / 8h).
#SBATCH --array=1-1

echo "=== Job Start: $(date) ==="
echo "Node: $(hostname)"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

REPO_DIR="${REPO_DIR:-/rds/user/lmk66/hpc-work/GameHSTGAT}"
cd "$REPO_DIR"

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/12.1

source ~/.bashrc
conda activate poly-shgnn-gpu

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
echo "Task index (0-based): ${TASK_INDEX}"
echo "Communication mode: ${COMMUNICATION_MODE:-fair_1bit}"
echo "Artifacts root: ${ARTIFACTS_ROOT:-artifacts/training_metrics_fair/grid_runs}"

cmd=(
  python scripts/run_grid_task.py
  --config "${CONFIG:-configs/default.yaml}"
  --task-index "$TASK_INDEX"
  --communication-mode "${COMMUNICATION_MODE:-fair_1bit}"
  --artifacts-root "${ARTIFACTS_ROOT:-artifacts/training_metrics_fair/grid_runs}"
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

echo "=== Job End: $(date) (exit=${status}) ==="
exit "$status"
