#!/usr/bin/env bash
# Build llama.cpp with CUDA support.
#
# Volta (GV100, sm_70) is targeted explicitly: Xeon + Quadro GV100 boxes are
# common, and building only for the installed arch cuts compile time a lot.
# Override with CUDA_ARCH=... (e.g. 89 for Ada, 90 for Hopper) or "native".
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }
LLAMA_DIR="${LLAMA_DIR:-$ROOT/third_party/llama.cpp}"
CUDA_ARCH="${CUDA_ARCH:-70}"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LLAMA_DIR")"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
if [[ -f build/CMakeCache.txt ]] && ! grep -q "CMAKE_CUDA_ARCHITECTURES:STRING=$CUDA_ARCH" build/CMakeCache.txt; then
  echo "[warn] existing build dir has a different CUDA arch; reconfiguring"
  rm -rf build
fi
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_CURL=OFF

cmake --build build -j "$JOBS" --target llama-server llama-cli llama-tokenize llama-bench

echo "[ok] built: $LLAMA_DIR/build/bin"
ls -la "$LLAMA_DIR/build/bin" | grep -E "llama-(server|cli|tokenize|bench)"
