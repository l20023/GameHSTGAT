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
    --num-nodes-list "${NUM_NODES_LIST:-10,100,1000}"
    --signal-quality-list "${SIGNAL_QUALITY_LIST:-0.55,0.65,0.8}"
    --train-episodes "${TRAIN_EPISODES:-5000}"
    --test-episodes "${TEST_EPISODES:-1000}"
    --max-horizon "${MAX_HORIZON:-50}"
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
  metrics_agg_json="${metrics_agg_csv%.csv}.json"
  metrics_agg_plots="${grid_root}/aggregate_plots"
  python scripts/summarize_metrics.py \
    --root "$grid_root" \
    --csv "$metrics_csv" \
    --aggregate-csv "$metrics_agg_csv" \
    --aggregate-json "$metrics_agg_json" \
    --aggregate-plots-dir "$metrics_agg_plots"

  echo ""
  echo "Done."
  echo "  Grid summary:     ${summary_json}"
  echo "  Per-run table:    ${metrics_csv}"
  echo "  Aggregated table: ${metrics_agg_csv}"
  echo "  Aggregate JSON:   ${metrics_agg_json}"
  echo "  Aggregate plots:  ${metrics_agg_plots}/"
}
