#!/bin/bash

# Configuration
CONFIG_DIR="./configs/expanded"
SBATCH_FILE="job.sh"

# Arguments with Defaults
NUM_RUNS=${1:-150}        # Total hyperparameter combinations to try
MAX_PARALLEL=${2:-20}    # Max jobs running at the same time
METHODS_CSV=${3:-"SheafGeDiDiagLQglobalLight,SheafGeDiDiagLQperedgeLight"}
DATASETS_CSV=${4:-"roman,squirrel,telegram,EU,chameleon,enroll,cora,cornell,texas,wisconsin,citeseer,amazon,node:500_subset-min=3_subset-max=10_N-intra=30_N-inter=10,node:500_subset-min=3_subset-max=10_N-intra=30_N-inter=30,500_subset-min=3_subset-max=10_N-intra=30_N-inter=50"}

# WandB credentials
export WANDB_PROJECT="Learnable_Q"
export WANDB_ENTITY="sheaf_hypergraphs"

# Parse Filters
IFS=',' read -ra ALLOWED_METHODS <<< "$METHODS_CSV"
IFS=',' read -ra ALLOWED_DATASETS <<< "$DATASETS_CSV"

echo "[INFO] Starting Sweeps: $NUM_RUNS total runs, $MAX_PARALLEL parallel jobs max."

for CONFIG_YAML in "$CONFIG_DIR"/*.yaml "$CONFIG_DIR"/*.yml; do
    [[ ! -f "$CONFIG_YAML" ]] && continue

    # Parse filename: method_dataset.yaml
    BASENAME=$(basename "$CONFIG_YAML")
    NAME_NO_EXT="${BASENAME%.*}"
    METHOD="${NAME_NO_EXT%%_*}"
    DATASET="${NAME_NO_EXT#*_}"

    # === Filtering Logic ===
    MATCHED_METHOD=false
    for m in "${ALLOWED_METHODS[@]}"; do [[ "$METHOD" == "$m" ]] && MATCHED_METHOD=true; done
    MATCHED_DATASET=false
    for d in "${ALLOWED_DATASETS[@]}"; do [[ "$DATASET" == "$d" || "$DATASET" == $d* ]] && MATCHED_DATASET=true; done

    if [[ "$MATCHED_METHOD" == false || "$MATCHED_DATASET" == false ]]; then
        # echo "[SKIP] $BASENAME does not match filters."
        continue
    fi

    # === 1. Initialize W&B Sweep ===
    echo "[INFO] Creating W&B Sweep for $BASENAME..."
    SWEEP_OUTPUT=$(wandb sweep "$CONFIG_YAML" 2>&1)
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "ID:" | awk '{print $NF}')

    if [ -z "$SWEEP_ID" ]; then
        echo "[ERROR] Could not extract Sweep ID for $BASENAME. Output: $SWEEP_OUTPUT"
        continue
    fi
    echo "[SUCCESS] Sweep ID: $SWEEP_ID"

    # === 2. Submit Slurm Job Array ===
    # We override the #SBATCH --array from the file using the command line flag
    echo "[INFO] Submitting Slurm Array for $METHOD on $DATASET..."
    
    sbatch --array=1-"$NUM_RUNS"%"$MAX_PARALLEL" \
           --export=ALL,SWEEP_ID="$SWEEP_ID" \
           "$SBATCH_FILE"
done