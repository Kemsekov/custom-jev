#!/usr/bin/env bash
# Start the JEV HTTP API (which starts llama-server itself).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }
. .venv/bin/activate
export PYTHONUNBUFFERED=1

# Graceful shutdown matters: the API owns a llama-server child process. On
# SIGTERM/SIGINT/SIGHUP, forward the signal so uvicorn runs its lifespan
# shutdown and stops llama-server too (otherwise it would be orphaned).
python -m jev.cli serve --config config.yaml "$@" &
PID=$!
trap 'kill -TERM "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; exit 0' TERM INT HUP
wait "$PID"
