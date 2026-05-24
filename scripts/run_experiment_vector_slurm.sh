#!/usr/bin/env bash
# Submit vector communication grid via SLURM array.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export COMMUNICATION_DIM="${COMMUNICATION_DIM:-32}"
bash scripts/slurm/run_batch_slurms.sh "${1:-20}" vector "${@:2}"
