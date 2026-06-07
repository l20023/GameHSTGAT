#!/bin/bash
#SBATCH -J HSTGAT_anim
#SBATCH --account LIO-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Fallback; run_animate_slurms.sh overrides --time per n.
#SBATCH --array=1-3

echo "=== Animate job start: $(date) ==="
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

SEED="${SEED:-0}"
SIGNAL_QUALITY="${SIGNAL_QUALITY:-0.55}"
NUM_NODES="${NUM_NODES:?NUM_NODES must be set}"
COMMUNICATION_MODE="${COMMUNICATION_MODE:-fair_1bit}"
EPISODE_SEED_BASE="${EPISODE_SEED_BASE:-4242}"

TOPOS=(complete ws_p_0.0 ws_p_0.1)
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#TOPOS[@]} )); then
  echo "[ERROR] task index ${TASK_INDEX} out of range for topologies."
  exit 1
fi
TOPO="${TOPOS[$TASK_INDEX]}"

case "$NUM_NODES" in
  10)  TRAIN_EPISODES=5000 ;;
  100) TRAIN_EPISODES=7000 ;;
  1000) TRAIN_EPISODES=10000 ;;
  *)
    echo "[ERROR] Unsupported NUM_NODES=${NUM_NODES}"
    exit 1
    ;;
esac

CHANNEL="${COMMUNICATION_MODE}"
FRAME_STEP=1
if [[ "$TOPO" != "complete" ]]; then
  FRAME_STEP=2
fi

Q_KEY="$(python - <<PY
q = float("${SIGNAL_QUALITY}")
print(f"{q:.2f}".replace(".", "p"))
PY
)"

CKPT="artifacts/checkpoints/${CHANNEL}/n_${NUM_NODES}/q_${Q_KEY}/${TOPO}/seed_${SEED}.pt"
OUT="artifacts/animations/${CHANNEL}/n_${NUM_NODES}/q_${Q_KEY}/${TOPO}/seed_${SEED}.html"

cmd=(
  python scripts/animate_episode.py
  --seed "$SEED"
  --num-nodes "$NUM_NODES"
  --signal-quality "$SIGNAL_QUALITY"
  --topology "$TOPO"
  --communication-mode "$COMMUNICATION_MODE"
  --train-episodes "$TRAIN_EPISODES"
  --episode-seed $((EPISODE_SEED_BASE + SEED))
  --frame-step "$FRAME_STEP"
  --format html
  --save-checkpoint "$CKPT"
  --output "$OUT"
)

if [[ "$COMMUNICATION_MODE" == "vector" ]]; then
  cmd+=(--communication-dim "${COMMUNICATION_DIM:-32}")
fi

echo "Running: n=${NUM_NODES} topo=${TOPO} seed=${SEED} q=${SIGNAL_QUALITY}"
"${cmd[@]}"
status=$?

echo "=== Animate job end: $(date) (exit=${status}) ==="
exit "$status"
