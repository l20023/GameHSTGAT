#!/usr/bin/env bash
# Render interactive HTML viewers for every checkpoint under artifacts/checkpoints/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

total=0
ok=0
fail=0

while IFS= read -r ckpt; do
  total=$((total + 1))
  rel="${ckpt#artifacts/checkpoints/}"
  mode=$(echo "$rel" | cut -d/ -f1)
  n_part=$(echo "$rel" | cut -d/ -f2)
  q_part=$(echo "$rel" | cut -d/ -f3)
  topo=$(echo "$rel" | cut -d/ -f4)
  seed="${ckpt##*seed_}"
  seed="${seed%.pt}"
  num_nodes="${n_part#n_}"
  signal_quality=$(python -c "print('${q_part}'.replace('q_','').replace('p','.'))")
  out="artifacts/animations/${mode}/n_${num_nodes}/${q_part}/${topo}/seed_${seed}.html"
  frame_step=1
  if [[ "$topo" != "complete" ]]; then
    frame_step=2
  fi

  echo ""
  echo "========================================"
  echo "[$(date '+%H:%M:%S')] ($total) $ckpt"
  echo "-> $out"
  echo "========================================"

  cmd=(
    python scripts/animate_episode.py
    --checkpoint "$ckpt"
    --skip-train
    --seed "$seed"
    --num-nodes "$num_nodes"
    --signal-quality "$signal_quality"
    --topology "$topo"
    --communication-mode "$mode"
    --episode-seed $((4242 + seed))
    --episode-variants 12
    --frame-step "$frame_step"
    --format html
    --output "$out"
  )
  if [[ "$mode" == "vector" ]]; then
    cmd+=(--communication-dim 32)
  fi

  if "${cmd[@]}"; then
    ok=$((ok + 1))
    echo "[$(date '+%H:%M:%S')] OK ($ok/$total)"
  else
    fail=$((fail + 1))
    echo "[$(date '+%H:%M:%S')] FAILED: $ckpt"
  fi
done < <(find artifacts/checkpoints -name '*.pt' | sort)

echo ""
echo "Done: $ok succeeded, $fail failed, $total total"
if (( fail > 0 )); then
  exit 1
fi
