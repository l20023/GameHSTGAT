#!/usr/bin/env bash
# Full proposal grid with vector communication (ablation; not the strict HST benchmark).
#
# Usage:
#   bash scripts/run_experiment_vector.sh
#   bash scripts/run_experiment_vector.sh --seeds 0,1,2
#   COMMUNICATION_DIM=64 bash scripts/run_experiment_vector.sh
#
# Defaults: same matrix as fair run (5 seeds, n={10,100,1000}, q={0.55,0.6,0.7,0.8}); vector channel dim 32.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/_experiment_grid_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_experiment_grid_common.sh"

export CONFIG="${CONFIG:-configs/default.yaml}"
export COMMUNICATION_DIM="${COMMUNICATION_DIM:-32}"

run_proposal_grid_experiment \
  "vector" \
  "artifacts/training_metrics_vector" \
  "artifacts/grid_summary_vector.json" \
  "artifacts/metrics_summary_vector.csv" \
  "artifacts/metrics_summary_vector_aggregated.csv" \
  "$@"
