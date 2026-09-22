#!/usr/bin/env bash
# Fetch animal photos for the vision evaluation from Wikimedia Commons.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. .venv/bin/activate

python -m jev.cli assets "$@"
