#!/usr/bin/env bash
# Download the model artifacts selected in .env (see .env.example):
#
#   JEV_GGUF_REPO / JEV_GGUF_FILE       text brain (default: Qwen3.8-4B Q8_0)
#   JEV_MMPROJ_REPO / JEV_MMPROJ_FILE   vision tower (Qwen3.5 VL projector)
#
# Files land in models/gguf/ with their repo names, which is exactly what the
# runtime loads: config.py turns JEV_GGUF_FILE / JEV_MMPROJ_FILE into those
# paths, so the API and the CLI pick up whatever was downloaded here.
#
# Why Q8_0 and not FP8? GGUF/ggml has no FP8 storage type at all - the format
# only defines integer/block quants (Q8_0, ...), BF16, F16 and MXFP4/NVFP4.
# Q8_0 is the closest thing that exists: 8-bit weights with FP16 block scales,
# ~4.6 GB for the 4B (9.2 GB for the 9B), which keeps the text brain intact
# for single-shot JEV readout.
#
# The vision tower comes from the Qwen3.5 base repo (the Distill inherits the
# Qwen3.5-VL architecture). To build a projector from the Distill's own
# safetensors instead, run scripts/02b_convert_precision.sh --mmproj-only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }
. .venv/bin/activate

GGUF_REPO="${JEV_GGUF_REPO:-${GGUF_REPO:-empero-ai/Qwen3.8-4B-Distill-GGUF}}"
GGUF_FILE="${JEV_GGUF_FILE:-${GGUF_FILE:-Qwen3.8-4B-Q8_0.gguf}}"
MMPROJ_REPO="${JEV_MMPROJ_REPO:-${FALLBACK_REPO:-lmstudio-community/Qwen3.5-4B-GGUF}}"
MMPROJ_FILE="${JEV_MMPROJ_FILE:-mmproj-Qwen3.5-4B-BF16.gguf}"
GGUF_DIR="$ROOT/models/gguf"

mkdir -p "$GGUF_DIR"

echo "==> downloading text model: $GGUF_REPO / $GGUF_FILE"
hf download "$GGUF_REPO" "$GGUF_FILE" --local-dir "$GGUF_DIR"

if [[ -n "$MMPROJ_FILE" ]]; then
  echo "==> downloading vision projector (mmproj): $MMPROJ_REPO / $MMPROJ_FILE"
  hf download "$MMPROJ_REPO" "$MMPROJ_FILE" --local-dir "$GGUF_DIR"
fi

# Optional full-precision artifact for benchmarking on big cards:
#   EXTRA_GGUF=1 ./scripts/02_download_model.sh
if [[ "${EXTRA_GGUF:-0}" == "1" ]]; then
  EXTRA_FILE="${GGUF_FILE%Q8_0.gguf}BF16.gguf"
  if [[ "$EXTRA_FILE" == "$GGUF_FILE" ]]; then
    echo "[warn] cannot derive a BF16 filename from '$GGUF_FILE'; skipping EXTRA_GGUF"
  else
    echo "==> downloading extra: $GGUF_REPO / $EXTRA_FILE"
    hf download "$GGUF_REPO" "$EXTRA_FILE" --local-dir "$GGUF_DIR"
  fi
fi

echo "[ok] model artifacts:"
ls -la "$GGUF_DIR"
