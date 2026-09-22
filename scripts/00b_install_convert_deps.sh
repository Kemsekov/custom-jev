#!/usr/bin/env bash
# Optional: dependencies for converting HuggingFace safetensors to GGUF
# (only needed for scripts/02b_convert_precision.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. .venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install "transformers>=5.0" safetensors sentencepiece

echo "[ok] conversion deps installed"
