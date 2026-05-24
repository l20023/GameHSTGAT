#!/usr/bin/env bash
# Submit fair_1bit grid via SLURM array.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash scripts/slurm/run_batch_slurms.sh "${1:-20}" fair_1bit "${@:2}"
