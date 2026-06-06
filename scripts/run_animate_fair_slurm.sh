#!/usr/bin/env bash
# Fair channel: one seed, q=0.55, all n × topologies via SLURM.
#
# Usage:
#   bash scripts/run_animate_fair_slurm.sh
#   SEED=4 bash scripts/run_animate_fair_slurm.sh
#   SEED=0 NUM_NODES_LIST=1000 bash scripts/run_animate_fair_slurm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_PARALLEL="${MAX_PARALLEL:-3}"
exec bash scripts/slurm/run_animate_slurms.sh "$MAX_PARALLEL" fair_1bit
