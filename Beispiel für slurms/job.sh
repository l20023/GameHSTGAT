#!/bin/bash
#SBATCH -J QLearnable
#SBATCH --account LIO-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Note: The --array line here is just a fallback; the manager script overrides it.
#SBATCH --array=1-1

echo "=== Job Start: $(date) ==="
echo "Node: $(hostname)"
echo "Sweep ID: $SWEEP_ID"

# ---------------- Environment ----------------
REPO_DIR=/rds/user/lmk66/hpc-work/Directional-Sheaf-Hypergraphs-Journal
cd "$REPO_DIR"

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/12.1

source ~/.bashrc
conda activate poly-shgnn-gpu

# ---------------- Execution ----------------
if [ -z "$SWEEP_ID" ]; then
    echo "[ERROR] No SWEEP_ID found in environment!"
    exit 1
fi

echo "Launching W&B Agent..."
# Each job in the array processes 1 configuration and then exits.
# This ensures that if a run takes too long, it doesn't get killed mid-way.
export WANDB_PROJECT="PolyDirectedSheaf"
export WANDB_ENTITY="sheaf_hypergraphs"

wandb agent --count 1 "$SWEEP_ID"

echo "=== Job End: $(date) ==="