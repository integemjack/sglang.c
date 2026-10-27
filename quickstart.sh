#!/usr/bin/env bash
# ===========================================================================
#  quickstart.sh -- zero-setup launcher for sglang-c (Linux / macOS / Git Bash)
#
#  Usage:
#    ./sglang-c/quickstart.sh                  build if needed, start the
#                                             OpenAI-compatible HTTP server,
#                                             wait for /health, demo request
#    ./sglang-c/quickstart.sh cli  [model] [n]   one-shot CLI generation
#    ./sglang-c/quickstart.sh gpu  [model] [port]  CUDA HTTP server (if built)
#    ./sglang-c/quickstart.sh test                run the CTest suite
#    ./sglang-c/quickstart.sh build               build the CPU stack
#    ./sglang-c/quickstart.sh help
#
#  Environment overrides:
#    SGL_MODEL=<dir>   model directory (default: first usable dir under models/)
#    SGL_PORT=<port>   HTTP port (default 30000)
#    SGL_NO_BUILD=1    never build, only run what is already built
# ===========================================================================
set -u

MODE="${1:-serve}"
PORT="${3:-${SGL_PORT:-30000}}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SGC="$SCRIPT_DIR"
ROOT="$(cd "$SGC/.." && pwd)"
CPU_BIN="$SGC/build/bin/sglc"
GPU_BIN="$SGC/build-cuda/bin/Release/cuda_http"
LOG="${TMPDIR:-/tmp}/sglang-c-server-$PORT.log"

die() { echo "[quickstart] ERROR: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
  sglang-c quickstart  (pure-C CPU/CUDA inference stack)
  ---------------------------------------------------------------
  quickstart.sh                 build if needed + start server + demo request
  quickstart.sh cli  [model] [n]     one-shot CLI generation
  quickstart.sh gpu  [model] [port]  CUDA HTTP server (cuda_http)
  quickstart.sh test                 run the CTest suite
  quickstart.sh build                build the CPU stack (CUDA too, if nvcc exists)
  quickstart.sh help                 this text

  env: SGL_MODEL=<dir>   SGL_PORT=<port>   SGL_NO_BUILD=1
EOF
}

pick_model() {
  local cands=( "$ROOT/models/qwen05" "$ROOT/models/qwen3-06" "$ROOT/models/tinyllama"
                "$ROOT/models/gpt2" )
  if [ -n "${SGL_MODEL:-}" ] && [ -f "$SGL_MODEL/config.json" ]; then
    echo "$SGL_MODEL"; return 0
  fi
  for c in "${cands[@]}"; do
    [ -f "$c/config.json" ] && { echo "$c"; return 0; }
  done
  for c in "$ROOT"/models/*/ "$SGC"/models/*/; do
    [ -f "$c/config.json" ] && { echo "${c%/}"; return 0; }
  done
  return 1
}

ensure_cpu_build() {
  [ -x "$CPU_BIN" ] && return 0
  [ "${SGL_NO_BUILD:-}" = "1" ] && die "$CPU_BIN is missing and SGL_NO_BUILD=1"
  command -v cmake >/dev/null || die "cmake not found (install cmake + a C compiler)"
  echo "[quickstart] first run: building the CPU stack ..."
  cmake -S "$SGC" -B "$SGC/build" >/dev/null || die "cmake configure failed"
  cmake --build "$SGC/build" -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" \
    || die "build failed"
}

ensure_gpu_build() {
  [ -x "$GPU_BIN" ] && return 0
  [ "${SGL_NO_BUILD:-}" = "1" ] && die "$GPU_BIN is missing and SGL_NO_BUILD=1"
  command -v nvcc >/dev/null || die "nvcc not found (CUDA toolkit required for the GPU stack)"
  echo "[quickstart] building the CUDA stack ..."
  cmake -S "$SGC" -B "$SGC/build-cuda" -DSGL_CUDA=ON || die "cmake configure failed"
  cmake --build "$SGC/build-cuda" -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" \
    --target cuda_cli cuda_http || die "CUDA build failed"
}

wait_health() {
  local i=0 limit="${SGL_HEALTH_TIMEOUT:-180}"
  while [ "$i" -lt "$limit" ]; do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "[quickstart] server answered on port $PORT after ${i}s"
      return 0
    fi
    i=$((i + 1)); sleep 1
  done
  echo "[quickstart] ERROR: server did not become healthy in ${limit}s - see $LOG" >&2
  tail -n 5 "$LOG" 2>/dev/null
  return 1
}

demo_chat() {
  local payload="${TMPDIR:-/tmp}/sglang-c-demo-$$.json"
  cat >"$payload" <<'JSON'
{"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":24}
JSON
  echo "[quickstart] /health ok -- sending a demo chat request ..."
  echo
  curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
       -H "Content-Type: application/json" --data-binary "@$payload"
  echo
  rm -f "$payload"
}

# runs in the background next to a foreground server: wait, demo, print endpoints
trickle() {
  wait_health || return 1
  [ "$MODE" = "serve" ] && demo_chat
  cat <<EOF

[quickstart] server is up. Endpoints:
    GET  http://127.0.0.1:$PORT/health
    POST http://127.0.0.1:$PORT/generate
    POST http://127.0.0.1:$PORT/v1/completions
    POST http://127.0.0.1:$PORT/v1/chat/completions
[quickstart] server log: $LOG
[quickstart] press Ctrl+C to stop the server.
EOF
}

case "$MODE" in
  -h|--help|help)
    usage
    exit 0
    ;;

  build)
    ensure_cpu_build
    echo "[quickstart] CPU build OK: $CPU_BIN"
    if command -v nvcc >/dev/null; then ensure_gpu_build && echo "[quickstart] CUDA build OK: $GPU_BIN"
    else echo "[quickstart] nvcc not found - skipped the CUDA stack"; fi
    ;;

  test)
    ensure_cpu_build
    ctest --test-dir "$SGC/build" --output-on-failure || die "ctest failed"
    echo "[quickstart] ctest OK"
    ;;

  cli)
    ensure_cpu_build
    [ -n "${2:-}" ] && SGL_MODEL="$2"
    MODEL="$(pick_model)" || die "no model found (put an HF-format model under $ROOT/models/ or set SGL_MODEL)"
    N="${3:-32}"
    QUANT="--w8 --kv8"; case "$MODEL" in *gpt2*) QUANT="" ;; esac
    echo "[quickstart] $MODEL (max-new=$N $QUANT)"
    exec "$CPU_BIN" --model "$MODEL" --prompt "The meaning of life is" --max-new "$N" $QUANT
    ;;

  gpu)
    ensure_gpu_build
    [ -n "${2:-}" ] && SGL_MODEL="$2"
    MODEL="$(pick_model)" || die "no model found"
    echo "[quickstart] CUDA server: model=$MODEL port=$PORT (log: $LOG)"
    trickle &
    HELPER=$!
    "$GPU_BIN" "$MODEL" "$PORT" >"$LOG" 2>&1
    RC=$?
    kill "$HELPER" 2>/dev/null
    [ "$RC" -ne 0 ] && { tail -n 5 "$LOG"; exit "$RC"; }
    ;;

  serve)
    ensure_cpu_build
    [ -n "${2:-}" ] && SGL_MODEL="$2"
    MODEL="$(pick_model)" || die "no model found (put an HF-format model under $ROOT/models/ or set SGL_MODEL)"
    QUANT="--w8 --kv8"; case "$MODEL" in *gpt2*) QUANT="" ;; esac
    echo "[quickstart] model : $MODEL"
    echo "[quickstart] server: http://127.0.0.1:$PORT"
    # the server runs in the foreground (so Ctrl+C stops it); this helper waits for
    # /health, sends one demo request and prints the endpoint list
    trickle &
    HELPER=$!
    "$CPU_BIN" --model "$MODEL" --serve --port "$PORT" $QUANT >"$LOG" 2>&1
    RC=$?
    kill "$HELPER" 2>/dev/null
    echo "[quickstart] server stopped (exit $RC). Log: $LOG"
    [ "$RC" -ne 0 ] && tail -n 5 "$LOG"
    ;;

  *)
    die "unknown mode '$MODE' (try: help)"
    ;;
esac