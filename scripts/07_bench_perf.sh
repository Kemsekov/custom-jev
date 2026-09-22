#!/usr/bin/env bash
# Measure latency and cache behaviour by decision shape:
#   decide (text/image, cold/warm), decide_many, index_inputs (photos/mixed/text)
# Writes results/artifacts/perf/ (wiped per run) and prints a table.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }
. .venv/bin/activate
export PYTHONUNBUFFERED=1

python -m jev.cli perf --config config.yaml "$@"
