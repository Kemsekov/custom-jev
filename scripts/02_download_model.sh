#!/usr/bin/env bash
# Download the default model artifacts for JEV:
#
#   * Qwen3.8-4B-Q8_0.gguf   (4.61 GB)  text brain, near-lossless 8-bit
#   * mmproj-Qwen3.5-4B-BF16.gguf (0.68 GB)  vision tower (Qwen3.5-VL projector)
#
# Why Q8_0 and not FP8? GGUF/ggml has no FP8 storage type at all - the format
# only defines integer/block quants (Q8_0, ...), BF16, F16 and MXFP4/NVFP4.
# The upstream repo has no FP8 file either. Q8_0 is the closest thing that
# exists: 8-bit weights with FP16 block scales, ~4.6 GB, which keeps the text
# brain intact for single-shot JEV readout while still fitting 6-8 GB cards.
#
# The vision tower is loaded from the Qwen3.5-4B base (the Distill inherits the
# Qwen3.5-VL architecture). To build a projector from the Distill's own
# safetensors instead, run scripts/02b_convert_precision.sh --mmproj-only.
#
# Requires HF_TOKEN in the environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }
. .venv/bin/activate

GGUF_REPO="${GGUF_REPO:-empero-ai/Qwen3.8-4B-Distill-GGUF}"
GGUF_DIR="$ROOT/models/gguf"
FALLBACK_REPO="${FALLBACK_REPO:-lmstudio-community/Qwen3.5-4B-GGUF}"

mkdir -p "$GGUF_DIR"

echo "==> downloading Q8_0 text brain: $GGUF_REPO"
hf download "$GGUF_REPO" Qwen3.8-4B-Q8_0.gguf --local-dir "$GGUF_DIR"

echo "==> downloading vision projector (mmproj): $FALLBACK_REPO"
hf download "$FALLBACK_REPO" mmproj-Qwen3.5-4B-BF16.gguf --local-dir "$GGUF_DIR"

# Optional larger/full-precision artifacts, for benchmarking on big cards:
#   EXTRA_GGUF=1 ./scripts/02_download_model.sh
if [[ "${EXTRA_GGUF:-0}" == "1" ]]; then
  hf download "$GGUF_REPO" Qwen3.8-4B-BF16.gguf --local-dir "$GGUF_DIR"
  hf download "$FALLBACK_REPO" Qwen3.5-4B-Q8_0.gguf --local-dir "$GGUF_DIR"
fi

echo "[ok] model artifacts:"
ls -la "$GGUF_DIR"
