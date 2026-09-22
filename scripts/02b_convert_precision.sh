#!/usr/bin/env bash
# Optional full-precision path for larger cards.
#
#   * converts the Distill safetensors to an F16 GGUF (tensor-core friendly),
#   * optionally builds an mmproj from the Distill's OWN vision tower instead
#     of the Qwen3.5-4B base projector:
#         MATCHED_MMPROJ=1 ./scripts/02b_convert_precision.sh
#   * or only converts the mmproj: ./scripts/02b_convert_precision.sh --mmproj-only
#
# Requires scripts/00b_install_convert_deps.sh first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }
. .venv/bin/activate

SRC_REPO="${SRC_REPO:-empero-ai/Qwen3.8-4B-Distill}"
SRC_DIR="$ROOT/models/Qwen3.8-4B-Distill"
GGUF_DIR="$ROOT/models/gguf"
CONVERT="$ROOT/third_party/llama.cpp/convert_hf_to_gguf.py"

MMPROJ_ONLY=0
[[ "${1:-}" == "--mmproj-only" ]] && MMPROJ_ONLY=1

mkdir -p "$SRC_DIR" "$GGUF_DIR"

if [[ ! -f "$SRC_DIR/model.safetensors" ]]; then
  echo "==> downloading $SRC_REPO (full precision)"
  hf download "$SRC_REPO" --local-dir "$SRC_DIR" \
    --exclude "training_args.bin" --exclude "*.md"
fi

if [[ "$MMPROJ_ONLY" == "0" ]]; then
  echo "==> converting F16 text GGUF"
  python "$CONVERT" "$SRC_DIR" \
    --outtype f16 --outfile "$GGUF_DIR/Qwen3.8-4B-Distill-F16.gguf"
fi

if [[ "${MATCHED_MMPROJ:-0}" == "1" || "$MMPROJ_ONLY" == "1" ]]; then
  echo "==> converting mmproj from the Distill's own vision tower"
  python "$CONVERT" "$SRC_DIR" \
    --mmproj --outtype f16 --outfile "$GGUF_DIR/mmproj-Qwen3.8-4B-Distill-F16.gguf"
fi

echo "[ok] precision artifacts:"
ls -la "$GGUF_DIR"
