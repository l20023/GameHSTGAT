#!/usr/bin/env bash
# Shared helper for full proposal-grid experiments (sourced by run_experiment_*.sh).
set -euo pipefail

run_proposal_grid_experiment() {
  local communication_mode="$1"
  local artifacts_dir="$2"
  local summary_json="$3"
  local metrics_csv="$4"
  local metrics_agg_csv="$5"
  shift 5

  local config="${CONFIG:-configs/default.yaml}"
  local grid_args=(
    --config "$config"
    --communication-mode "$communication_mode"
    --artifacts-dir "$artifacts_dir"
    --output "$summary_json"
    --num-nodes-list "${NUM_NODES_LIST:-10,50,100}"
    --signal-quality-list "${SIGNAL_QUALITY_LIST:-0.6,0.8}"
  )

  if [[ "$communication_mode" == "vector" ]]; then
    grid_args+=(--communication-dim "${COMMUNICATION_DIM:-32}")
  fi

  echo "=========================================="
  echo "Proposal grid experiment"
  echo "  mode:          ${communication_mode}"
  echo "  config:        ${config}"
  echo "  artifacts:     ${artifacts_dir}/grid_runs"
  echo "  summary:       ${summary_json}"
  echo "  extra args:    $*"
  echo "=========================================="

  python scripts/run_grid.py "${grid_args[@]}" "$@"

  local grid_root="${artifacts_dir}/grid_runs"
  echo ""
  echo "Summarizing metrics under ${grid_root} ..."
  python scripts/summarize_metrics.py \
    --root "$grid_root" \
    --csv "$metrics_csv" \
    --aggregate-csv "$metrics_agg_csv"

  echo ""
  echo "Done."
  echo "  Grid summary:     ${summary_json}"
  echo "  Per-run table:    ${metrics_csv}"
  echo "  Aggregated table: ${metrics_agg_csv}"
}
