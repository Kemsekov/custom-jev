#!/usr/bin/env bash
# Head-to-head comparison of reasoning-skipping strategies.
# Writes results/thinking-bench.md and prints the ranking.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }
. .venv/bin/activate

python -m jev.cli thinking-bench --config config.yaml "$@"
