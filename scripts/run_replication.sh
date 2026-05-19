#!/usr/bin/env bash
# Run train.py for replication seeds 0..(N-1). Default N=10 from configs/default.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/default.yaml}"
NUM_NODES="${2:-}"
NUM_SEEDS="${3:-10}"

extra_args=()
if [[ -n "$NUM_NODES" ]]; then
  extra_args+=(--num-nodes "$NUM_NODES")
fi

for seed in $(seq 0 $((NUM_SEEDS - 1))); do
  echo "=== seed=${seed} ==="
  python scripts/train.py --config "$CONFIG" --seed "$seed" "${extra_args[@]}"
done

echo "Finished ${NUM_SEEDS} replication runs (seeds 0..$((NUM_SEEDS - 1)))."
