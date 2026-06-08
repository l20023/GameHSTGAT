#!/usr/bin/env bash
# Render interactive HTML viewers for majority-vote baseline (q=0.55, seed 0).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

total=0
ok=0
fail=0

for num_nodes in 10 100 1000; do
  for topo in complete ws_p_0.0 ws_p_0.1; do
    total=$((total + 1))
    out="artifacts/animations/majority_vote/n_${num_nodes}/q_0p55/${topo}/seed_0.html"
    frame_step=1
    if [[ "$topo" != "complete" ]]; then
      frame_step=2
    fi

    echo ""
    echo "========================================"
    echo "[$(date '+%H:%M:%S')] ($total) majority_vote n=${num_nodes} ${topo}"
    echo "-> $out"
    echo "========================================"

    if python scripts/animate_majority_episode.py \
      --seed 0 \
      --num-nodes "$num_nodes" \
      --signal-quality 0.55 \
      --topology "$topo" \
      --episode-seed 4242 \
      --episode-variants 12 \
      --frame-step "$frame_step" \
      --output "$out"; then
      ok=$((ok + 1))
      echo "[$(date '+%H:%M:%S')] OK ($ok/$total)"
    else
      fail=$((fail + 1))
      echo "[$(date '+%H:%M:%S')] FAILED: n=${num_nodes} ${topo}"
    fi
  done
done

echo ""
echo "Done: $ok succeeded, $fail failed, $total total"
if (( fail > 0 )); then
  exit 1
fi
