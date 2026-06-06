#!/usr/bin/env bash
# Vector channel: one seed, q=0.55, all n × topologies via SLURM.
#
# Usage:
#   bash scripts/run_animate_vector_slurm.sh
#   SEED=4 bash scripts/run_animate_vector_slurm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_PARALLEL="${MAX_PARALLEL:-3}"
exec bash scripts/slurm/run_animate_slurms.sh "$MAX_PARALLEL" vector
