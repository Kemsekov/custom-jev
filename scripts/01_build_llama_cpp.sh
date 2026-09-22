#!/usr/bin/env bash
# Build llama.cpp. CUDA is used when an nvcc is available; otherwise the build
# falls back to a CPU-only build (still fully functional, just slower).
#
# The CUDA arch defaults to "native" (the GPU visible on this machine, e.g.
# 86 for an RTX 3070, 70 for a GV100). Override with CUDA_ARCH=... (70, 86,
# 89, 90, "all", ...) to produce a build for other hardware. Building only for
# the installed arch cuts compile time a lot.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }
LLAMA_DIR="${LLAMA_DIR:-$ROOT/third_party/llama.cpp}"
CUDA_ARCH="${CUDA_ARCH:-native}"
JOBS="${JOBS:-$(nproc)}"

# nvcc is often installed but not on PATH (e.g. /usr/local/cuda/bin). Find it
# wherever the distro put it and point both CMake and the compiler at it.
if [[ -z "${CUDACXX:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDACXX="$(command -v nvcc)"
  else
    for d in "${CUDA_HOME:-}" /usr/local/cuda /opt/cuda /usr/local/cuda-*; do
      if [[ -n "$d" && -x "$d/bin/nvcc" ]]; then
        CUDACXX="$d/bin/nvcc"
        export PATH="$d/bin:$PATH"
        break
      fi
    done
  fi
fi

if [[ -n "${CUDACXX:-}" && -x "$CUDACXX" ]]; then
  BUILD_CUDA=ON
  export CUDACXX
  echo "[info] CUDA compiler: $CUDACXX (arch=$CUDA_ARCH)"
else
  BUILD_CUDA=OFF
  echo "[warn] nvcc not found; building CPU-only."
  echo "[warn] install the CUDA toolkit (or set CUDACXX/CUDA_HOME) for GPU support."
fi

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LLAMA_DIR")"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
if [[ -f build/CMakeCache.txt ]]; then
  cached_cuda="$(grep -E '^GGML_CUDA:BOOL=' build/CMakeCache.txt | cut -d= -f2 || true)"
  cached_arch="$(grep -E '^CMAKE_CUDA_ARCHITECTURES[^=]*=' build/CMakeCache.txt | cut -d= -f2 || true)"
  stale=0
  [[ "$cached_cuda" != "$BUILD_CUDA" ]] && stale=1
  if [[ "$BUILD_CUDA" == "ON" && -n "$cached_arch" && "$cached_arch" != "$CUDA_ARCH" ]]; then
    stale=1
  fi
  if [[ "$stale" == "1" ]]; then
    echo "[warn] existing build dir has different CUDA settings; reconfiguring"
    rm -rf build
  fi
fi

CMAKE_ARGS=(
  -B build
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_CUDA="$BUILD_CUDA"
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_CURL=OFF
)
if [[ "$BUILD_CUDA" == "ON" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
fi
cmake "${CMAKE_ARGS[@]}"

cmake --build build -j "$JOBS" --target llama-server llama-cli llama-tokenize llama-bench

echo "[ok] built: $LLAMA_DIR/build/bin"
ls -la "$LLAMA_DIR/build/bin" | grep -E "llama-(server|cli|tokenize|bench)"
