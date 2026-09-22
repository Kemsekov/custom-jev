#!/usr/bin/env bash
# Run the evaluation suites and write results/<timestamp>/report.md.
#
#   ./scripts/04_eval.sh                    # decisions + animals + maze + latency
#   ./scripts/04_eval.sh --suites decisions # one suite
#   ./scripts/04_eval.sh --thinking-bench   # compare reasoning-skipping strategies
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }
. .venv/bin/activate
export PYTHONUNBUFFERED=1

python -m jev.cli evaluate --config config.yaml "$@"
