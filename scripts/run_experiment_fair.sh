#!/usr/bin/env bash
# Full proposal grid with fair 1-bit communication (HST benchmark).
#
# Usage:
#   bash scripts/run_experiment_fair.sh
#   bash scripts/run_experiment_fair.sh --seeds 0,1,2          # smoke / subset
#   CONFIG=configs/default.yaml bash scripts/run_experiment_fair.sh
#
# Defaults: 3 seeds (0..2), T=100, n in {10,50,100}, q in {0.6,0.8}, fair_1bit.
# Pass any extra run_grid.py flags after the script name (see examples above).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/_experiment_grid_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_experiment_grid_common.sh"

export CONFIG="${CONFIG:-configs/default.yaml}"

run_proposal_grid_experiment \
  "fair_1bit" \
  "artifacts/training_metrics_fair" \
  "artifacts/grid_summary_fair.json" \
  "artifacts/metrics_summary_fair.csv" \
  "artifacts/metrics_summary_fair_aggregated.csv" \
  "$@"
