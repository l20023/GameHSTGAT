#!/usr/bin/env bash
# Run train.py for replication seeds 0..(N-1). Default N=5 from configs/default.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/default.yaml}"
NUM_NODES="${2:-}"
NUM_SEEDS="${3:-5}"

extra_args=()
if [[ -n "$NUM_NODES" ]]; then
  extra_args+=(--num-nodes "$NUM_NODES")
fi

for seed in $(seq 0 $((NUM_SEEDS - 1))); do
  echo "=== seed=${seed} ==="
  python scripts/train.py --config "$CONFIG" --seed "$seed" "${extra_args[@]}"
done

ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-artifacts/training_metrics}"
AGG_DIR="${AGG_DIR:-${ARTIFACTS_ROOT}/aggregate}"
echo ""
echo "Summarizing metrics under ${ARTIFACTS_ROOT} ..."
python scripts/summarize_metrics.py \
  --root "$ARTIFACTS_ROOT" \
  --csv "${AGG_DIR}/metrics_summary.csv" \
  --aggregate-csv "${AGG_DIR}/metrics_summary_aggregated.csv" \
  --aggregate-json "${AGG_DIR}/metrics_summary_aggregated.json" \
  --aggregate-plots-dir "${AGG_DIR}/plots"

echo "Finished ${NUM_SEEDS} replication runs (seeds 0..$((NUM_SEEDS - 1)))."
echo "  Per-run table:    ${AGG_DIR}/metrics_summary.csv"
echo "  Aggregated table: ${AGG_DIR}/metrics_summary_aggregated.csv"
echo "  Aggregate plots:  ${AGG_DIR}/plots/"
