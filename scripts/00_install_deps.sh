#!/usr/bin/env bash
# Install the runtime Python dependencies. Intentionally light: no torch and no
# transformers are needed because llama-server owns tokenization, chat-template
# rendering and inference. Torch is only required for the optional GGUF
# conversion step (see 00b_install_convert_deps.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
. .venv/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt

echo "[ok] runtime deps installed into $ROOT/.venv"
