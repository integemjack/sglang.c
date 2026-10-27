# sglang-c

[![build](https://github.com/integemjack/sglang.c/actions/workflows/build.yml/badge.svg)](https://github.com/integemjack/sglang.c/actions/workflows/build.yml)

**The [SGLang](https://github.com/sgl-project/sglang) inference engine, rewritten in pure C.**

A CPU/CUDA inference stack rebuilt from scratch on top of the ideas in SGLang (whose Python core is ~1.38M
lines), written as portable C11 with **zero third-party dependencies** — no PyTorch, no CUDA runtime
dependency in the CPU build, no BLAS requirement (it can use OpenBLAS if you happen to have it). The only
non-system library in the CPU binary is the MinGW-w64 runtime, linked statically, so a single `.exe` runs on
a clean Windows box.

> **Status:** milestones M0–M26 and t1–t12 are complete (see [ROADMAP.md](ROADMAP.md) /
> [CHANGELOG.md](CHANGELOG.md)). Test suite: **26/26 CTest suites green** on the CPU stack, `cuda_test` reports
> `test_cuda: all OK` on the GPU stack (RTX 2060, sm_75). Real checkpoints validated end to end: GPT-2,
> TinyLlama-1.1B, Qwen2.5-0.5B, **Qwen3-0.6B**, LLaVA (CLIP-L/14-336 + `mm_projector`), Qwen3-Next (hybrid
> linear attention + MoE).

**English** | [简体中文](README.zh-CN.md)

---

## ⚡ One-click quick start (start here)

Two scripts do everything for you: detect the toolchain, build what is missing, start the
OpenAI-compatible HTTP server, wait for `/health`, send one demo chat request, and print the endpoint list.

```bat
:: Windows — double-click it, or from a terminal:
sglang-c\quickstart.bat
```

```bash
# Linux / macOS / Git Bash
bash ./sglang-c/quickstart.sh      # or: chmod +x sglang-c/quickstart.sh && ./sglang-c/quickstart.sh
```

What you get (real output from a first run on `models/qwen05`, Windows CPU stack):

```text
[quickstart] model : models\qwen05
[quickstart] server: http://127.0.0.1:30000   (window "sglang-c server :30000")
[quickstart] server answered on port 30000 after 2s
[quickstart] /health ok -- sending a demo chat request ...
{"id":"chatcmpl-1","object":"chat.completion",...,"choices":[{"index":0,"message":{"role":"assistant",
 "content":"Paris ..."},"finish_reason":"length"}],"usage":{"prompt_tokens":20,...}}

[quickstart] server is up. Endpoints:
    GET  http://127.0.0.1:30000/health
    POST http://127.0.0.1:30000/generate
    POST http://127.0.0.1:30000/v1/completions
    POST http://127.0.0.1:30000/v1/chat/completions
```

### Modes

| Command | What it does |
| --- | --- |
| `quickstart.bat` / `quickstart.sh` | Build if needed → serve on port 30000 → wait for `/health` → demo request |
| `... cli [model] [n]` | One-shot CLI generation (`--w8 --kv8` enabled automatically) |
| `... gpu [model] [port]` | Start the CUDA HTTP server (`cuda_http`), building it first if missing |
| `... test` | Run the CTest suite (`ctest --output-on-failure`) |
| `... build` | Full build: CPU (MinGW/Ninja, or any host C compiler) + CUDA (MSVC + nvcc) |
| `... help` | Usage summary |

### Knobs

| Variable | Meaning |
| --- | --- |
| `SGL_MODEL=<dir>` | Model directory (default: first usable HF-format dir under `models/`) |
| `SGL_PORT=<port>` | HTTP port (default `30000`) |
| `SGL_NO_BUILD=1` | Never build; fail if the binary is missing |
| `SGL_NO_PAUSE=1` | Windows only: don't wait for a key press at the end (CI / scripting) |
| `SGL_HEALTH_TIMEOUT=<s>` | Unix only: seconds to wait for `/health` (default 180) |

Both scripts drive the same underlying commands, and both were exercised against this repository's CPU
build, CUDA build and the 26-suite CTest run (Windows `quickstart.bat`: all modes; `quickstart.sh`: under Git
Bash — `help`, `cli`, `serve`, `gpu`). The server they start is a plain foreground process (on Windows the
CPU server opens its own console window), so `Ctrl+C` or closing that window stops it; `server log:` lines
point at the captured output on Unix.

**Prerequisites** — one C toolchain and one model directory:

- Windows CPU: CMake ≥ 3.20, Ninja, MinGW gcc (`winget install BrechtSanders.WinLibs.POSIX.UCRT`)
- Windows GPU: VS2022 Build Tools + CUDA Toolkit 13.x (`nvcc`)
- Linux/macOS: `cmake`, a C compiler, `curl`
- Models: any Hugging Face export (`config.json`, `tokenizer.json`, `model.safetensors`) under `models/`

Nothing else is needed — no Python, no virtualenv, no PyTorch.

---

## Contents

- [Downloads (prebuilt binaries)](#downloads-prebuilt-binaries)
- [Feature matrix](#feature-matrix)
- [Build from source](#build-from-source)
- [CLI reference](#cli-reference)
- [HTTP API](#http-api)
- [Models and directory format](#models-and-directory-format)
- [Performance](#performance)
- [Verified toolchain versions](#verified-toolchain-versions)
- [Environment variables](#environment-variables)
- [Engineering log](#engineering-log)
- [Architecture](#architecture)
- [Validation methodology](#validation-methodology)
- [Known limitations](#known-limitations)
- [Repository layout](#repository-layout)
- [License](#license)

## Downloads (prebuilt binaries)

[Releases](https://github.com/integemjack/sglang.c/releases) ships CI-built artifacts (tagging `v*` builds and
publishes them automatically):

| Package | Contents |
| --- | --- |
| `linux-x64` / `linux-arm64` | `sglc` (CPU; OpenMP multithreading when built with GNU compilers) |
| `linux-x64-cuda` / `linux-arm64-cuda` | `sglc` + `cuda_cli` + `cuda_http` (+ the bundled `libcublas*.so`); needs an NVIDIA GPU and driver at runtime |
| `macos-arm64` | `sglc` (CPU, **single-threaded**: AppleClang has no `-fopenmp` and CI does not install libomp) |
| `macos-arm64-mlx` | same binary plus `tools/ref_llava_mlx.py`: the LLaVA numeric reference recomputed with `mlx.core` on the Metal GPU (`test_llava` then compares C against MLX) |
| `windows-x64` | `sglc.exe` (CPU, MSVC `/MT` static, single-threaded) |
| `windows-x64-cuda` | `sglc.exe` + `cuda_cli.exe` + `cuda_http.exe` (RTX acceleration, sm_75+; needs a driver, ships the cuBLAS DLLs) |
| `windows-arm64-cuda` | **preview**: Windows on Arm build made with the CUDA 13.4.0 *Developer Preview* toolkit (sm_75 + sm_120 cubins, aimed at Blackwell-class Windows-on-Arm machines such as RTX Spark with driver 616.41+) |

Build matrix (`.github/workflows/build.yml`): `linux-x64`, `linux-arm64`, `linux-x64-cuda`, `linux-arm64-cuda`,
`macos-arm64`, `macos-arm64-mlx`, `windows-x64`, `windows-x64-cuda`, plus a separate `windows-arm64-cuda` job
on GitHub's arm64 Windows runner. Every push builds and packages all of them; the CUDA and arm64 entries
cannot run on the GPU-less runners, so they are smoke-checked instead (binaries present, `ldd`/`dumpbin`
architecture and dependencies, `sglc` usage exit code) and left to be run on real hardware.

What was verified on real hardware and how: [docs/PROGRESS_g11_g19.md](docs/PROGRESS_g11_g19.md).
`linux-x64`, `linux-arm64`, `windows-x64` and `windows-x64-cuda` have all been run for real; **macos-arm64
was compiled on an iMac (macOS 27.0.1 / AppleClang 21) and passed ctest 24/24**
([docs/g_macos_arm64.txt](docs/g_macos_arm64.txt)). `macos-x64` (Intel) was **dropped on 2026-10-27**: it
needed cross-compilation from an arm64 host plus Rosetta to run the tests, and Intel runners are gone.

## Feature matrix

| Capability | SGLang counterpart | Status |
| --- | --- | --- |
| GPT-2 / Llama / Qwen2 / **Qwen3** backends | `srt/models/` | ✅ numerically matched against numpy/HF (maxerr ≤ 2e-4; 1e-8 with SIMD) |
| **Mistral sliding window / Gemma normalization** | `srt/models/` | ✅ identical to a numpy reference on randomized structures (`tests/gen_random_*.py`) |
| Continuous-batching scheduler | `srt/managers/scheduler.py` | ✅ batched output token-identical to serial; 3-way concurrency +68% |
| HTTP service (OpenAI-compatible, CPU + GPU) | `entrypoints/http_server.py` | ✅ `/generate` `/v1/completions` `/v1/chat/completions`, queueing past the limit, faithful `finish_reason` |
| RadixAttention prefix cache | `srt/mem_cache/radix_cache.py` | ✅ paged KV + refcounts + LRU eviction; ~40% faster on long repeated prompts |
| W8A16 quantization (CPU) / **W8A8 dp4a (CUDA)** | `srt/layers/quantization` | ✅ CPU decode 1.76x; best VRAM/speed trade-off on GPU |
| int8 KV cache | fp8/int8 KV path | ✅ half the memory, +20% throughput |
| Beam search | `srt/beam_search/` | ✅ W ≤ 8, hard assertion that W=1 == greedy |
| Speculative decoding v2 | `srt/speculative/` | ✅ batched n-token verify; 13% acceptance with a 160M draft; `spec == greedy` guaranteed |
| Multi-LoRA hot swap | `srt/lora/` | ✅ runtime add/remove, pick 1 of N |
| **CUDA FP16 / graphs / long context** | `srt/` | ✅ 1.1B fp16 fits a 6 GB card; graph batching +89%; ctx 16384 |
| **LLaVA multimodal (CLIP + fusion + HTTP)** | `srt/models/llava.py` | ✅ CLIP ViT + **two-layer MLP projector** (layer count auto-detected, GELU) + visual tokens **inserted at the `<image>` position** + `image_url` in `/v1/chat/completions` (**PNG/JPEG accepted directly**); numpy maxerr 9.3e-9; real CLIP-L/14-336 + real `mm_projector` match transformers at rel **1.4e-05 / 1.8e-05** |
| **Qwen3-Next (hybrid layers + MoE)** | `srt/models/qwen3_next.py` | ✅ linear attention (gated deltanet) + full layers with q-gating / partial rotary / `(1+w)` RMSNorm / MoE (top-k routing + shared experts + routed scaling); vs float64 numpy **2.5e-08–5.8e-08**; recurrent state keyed per sequence (prefix sharing cannot collide) |
| **CPU OpenBLAS / AVX2 dot** | backend | ✅ 3.1x on large-M GEMM (195 GFLOPs), optional at runtime, zero hard dependency |
| SSE streaming | `--stream` | ✅ streamed output strictly identical to non-streamed |

**Not covered:** tensor parallelism; the full LLaVA **7B language model** end to end (CLIP-L/14-336 and
`mm_projector` were verified against real weights piecewise; the 7B LM was not downloaded due to disk/VRAM);
GPU-side sliding window; the CUDA path for Qwen3-Next (CPU reference implementation only).

## Build from source

The quickstart scripts do this for you; for a manual build use:

```bash
# CPU (MinGW-w64 on Windows, or any host C compiler elsewhere)
cmake -S sglang-c -B sglang-c/build -G Ninja
cmake --build sglang-c/build
ctest --test-dir sglang-c/build          # 26 suites (some need real weights under models/)

# CUDA backend (separate build dir; MSVC + nvcc on Windows)
cmake -S sglang-c -B sglang-c/build-cuda -G "Visual Studio 17 2022" -A x64 -DSGL_CUDA=ON
cmake --build sglang-c/build-cuda --config Release --target cuda_test cuda_cli cuda_http
```

One-shot helpers in this directory: `build_all.bat` (full CPU+CUDA build + tests, exits 0 only when
everything is green) and `quickstart.bat` (build-if-needed + serve).

Optional OpenBLAS: unpack `OpenBLAS 0.3.29 x64.zip` into `third_party/openblas` and configure with
`-DSGL_OPENBLAS=ON`. **Use the `x64` package, not `x64_64`** — the latter is the 64-bit-integer ABI and
dynamic calls silently read garbage with it.

## CLI reference

```bash
# interactive generation
sglang-c/build/bin/sglc --model models/gpt2 --prompt "The meaning of life is" --max-new 32
sglang-c/build/bin/sglc --model models/tinyllama --w8 --kv8 --max-new 64   # quantized, faster

# HTTP service (OpenAI compatible)
sglang-c/build/bin/sglc --model models/qwen05 --serve --port 30000 --w8
```

| Flag | Meaning |
| --- | --- |
| `--model <dir>` | Model directory (required) |
| `--prompt <text>` | Prompt for one-shot generation |
| `--max-new N` | Max new tokens (default 32) |
| `--temperature F` / `--top-k N` / `--top-p F` / `--seed N` | Sampling controls |
| `--w8` | int8 weights (W8A16; decode ~1.8x faster) |
| `--kv8` | int8 KV cache (half the memory) |
| `--serve` | Start the HTTP server (`/health` `/generate` `/v1/completions` `/v1/chat/completions`) |
| `--host H` / `--port P` | Bind address / port (default port 30000) |
| `--vision <clip_dir>` / `--projector <file>` | Enable the LLaVA vision path |
| `--lora name=dir[,name2=dir2,...]` | LoRA instance pool (each adapter pre-merged into its own model copy) |
| `--verbose` | Debug logging |

Note: MSVC builds of `sglc` have no `--serve`; use `cuda_http.exe` for the GPU HTTP server.

The CUDA CLI is positional: `cuda_cli.exe <model_dir> "<prompt>" [max_new] [seed]`
(`@file` reads a long prompt from disk). The CUDA server is `cuda_http.exe <model_dir> [port]`.

## HTTP API

All three POST routes take JSON; `/health` is a plain GET.

```bash
# health + counters
curl -s http://127.0.0.1:30000/health
{"ok":true,"gpu":true,"ctx":16384,"seg":2048,"conc":8,"prefix":{"slots":16,"hits":0,"misses":0,...}}

# chat completion
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":24}'

# text completion / raw prompt
curl -s http://127.0.0.1:30000/v1/completions -d '{"prompt":"The meaning of life is","max_tokens":32}'

# streaming (SSE)
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Because the wire format is OpenAI's, the usual SDKs work unchanged:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="none")
print(client.chat.completions.create(
    model="sglang-c",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
).choices[0].message.content)
```

On the GPU server, `/health` also reports `gpu`, `ctx`, `seg` (KV segment length), `conc` (concurrency) and
the prefix-cache counters, and it switches to `{"ok":false,"gpu":false,"degraded":true,"reason":"..."}`
when the VRAM pre-check fails (requests then get 503 with a suggestion instead of the process dying).

## Models and directory format

A model directory is a standard Hugging Face export:

```text
models/qwen05/
├── config.json
├── tokenizer.json          # and optionally tokenizer_config.json
└── model.safetensors       # F32 / F16 / BF16
```

Tested fixtures used by the suite: `models/gpt2`, `models/tinyllama`, `models/qwen05`, `models/qwen3-06`,
`models/llama-160m`, plus randomly generated structures (`tests/gen_random_*.py` → `mistral-rand`,
`gemma-rand`, `gemma2-rand`, `llava-rand`, `qwen3next-rand`) and real LLaVA pieces
(`clip-vit-large-patch14-336`, `llava-mlp`, `llava-real`).

## Performance

All numbers below are measured, not modelled. Absolute times were taken on the machines listed; the "same
process A/B" numbers come from `cuda_perf --ab` because WDDM spill-over can inflate absolute times 3–20x
when free VRAM is tight — cross-process comparisons are not valid on that box.

### CPU (i7-10700F, 8 cores, TinyLlama-1.1B fp32)

| Configuration | Throughput |
| --- | --- |
| Baseline | 1.5 tok/s |
| + SIMD (hand-written SSE2/SSE4.1 intrinsics) | ~3 tok/s |
| + W8 weights + int8 KV | **7.4 tok/s** (4.9x cumulative) |
| GEMM peak | `matmul_nt` 8.4 GFLOPs (4x scalar) |

Qwen2.5-0.5B chat: 4.2 tok/s (W8 + KV8 + SIMD; its 151k-vocabulary `lm_head` dominates the cost).

### CUDA (RTX 2060, sm_75; M23 adaptive GEMM: hand-written kernel for M=1, cuBLAS for M>1)

Qwen2.5-0.5B: single stream **45.7 tok/s** (10.9x the CPU stack); **batched (4 sequences) 105.2 tok/s**
(hand-written 64.9 → cuBLAS 93.6 → adaptive + fused `lm_head` 105.2, +62% cumulative). Prefill maxerr
3.5e-5 vs CPU, greedy identical in all three implementations, batch output identical to single-stream runs,
KV segments isolated.

Current snapshot ([bench_results.txt](../bench_results.txt), 2026-10-26): CUDA single-stream 42.2 tok/s,
batch-4 102.1 tok/s, numerics maxerr 3.5e-05, batch-4 invariant PASS, CPU suite 100% passed.

Highlights from the optimization log (details and logs in [CHANGELOG.md](CHANGELOG.md),
[docs/BENCH.md](docs/BENCH.md)):

- **Flash prefill (G1/G16)**: single-pass chunked attention kernel; the switch-over threshold dropped from
  4096 to 64 tokens. Same-process A/B on a clean machine: 8192-token prefill 66.1 s → **2.7 s (23.6x)**,
  2048-token 11.8x. (An earlier run under VRAM pressure measured 8.3x — the two ratios must not be quoted
  interchangeably.)
- **W8 large-M on tensor cores (G2)**: int8×int8→int32 IMMA for M>1, dp4a GEMV kept for M=1.
  [2048,2048] M=8: 0.925 ms → **0.069 ms (13.4x)**; IMMA vs dp4a prefill maxerr = 0.
- **Decode CUDA graphs (G3)**: 4-sequence batch 177 → **335 tok/s (+89%)**, output identical with graphs
  off. fp16 decode B=1 23.0 → 9.7 ms, B=4 36.6 → 8.9 ms, B=8 38.5 → 11.3 ms.
- **Prefix caching (G6/G15)**: token-exact longest-common-prefix reuse by copying KV rows; 756-token shared
  prefix 0.76 s → **0.34 s (2.2x)**; repeating the same prompt 0.262 s → **0.103 s (2.5x)**.
- **MSVC CPU path (G18)**: previously SSE2-only and 10.2x slower than GCC; a separate AVX2 translation unit
  with CPUID dispatch brought 930-token prefill 326.2 s → **33.6 s (9.7x)** and 8192-token prefill
  3733.0 s → **611.8 s (6.1x)**, numerics unchanged (maxerr 1.43e-04 at 8K, greedy identical).

## Verified toolchain versions

| Component | Version | Notes |
| --- | --- | --- |
| Windows CPU stack | WinLibs GCC **16.2** (`winget install BrechtSanders.WinLibs.POSIX.UCRT`) | CMake ≥ 3.20 + Ninja; artifacts are statically linked |
| CUDA stack (x64) | VS2022 Build Tools (MSVC 14.44) + CUDA Toolkit **13.3** | sm_75+ (RTX 20 series and newer); Windows x64 and Linux (x64/arm64-sbsa) both build with 13.3 |
| CUDA stack (Windows on Arm) | arm64 MSVC + CUDA Toolkit **13.4.0 Developer Preview** | the first CUDA release with Windows-on-Arm host tools; only a preview installer exists, so the arm64 package is experimental |
| Optional OpenBLAS | **0.3.29 x64.zip** | see the warning above about `x64` vs `x64_64`; `-DSGL_OPENBLAS=ON` |
| Source encoding | UTF-8 **with BOM** | MSVC/nvcc read BOM-less files as GBK and swallow struct declarations after Chinese comments; CMake warns during configure |
| Test models | `models/` | real weights: gpt2 / tinyllama / qwen05 / qwen3-06 / llama-160m; generated: mistral-rand / gemma-rand / llava-rand |

## Environment variables

Everything is optional; defaults are production settings.

| Variable | Scope | Meaning |
| --- | --- | --- |
| `SGL_W8` / `SGL_KV8` | CPU | Same as `--w8` / `--kv8` |
| `SGL_POOL_SLOTS` | CPU | KV page count of the pool (large contexts need a larger pool) |
| `SGL_LIN_SLOTS` | CPU | Recurrent-state slots for Qwen3-Next linear layers |
| `SGL_BLAS` / `SGL_BLAS_MINMAC` / `SGL_OPENBLAS_PATH` | CPU | Enable/limit OpenBLAS GEMM use and point at the DLL |
| `SGL_LORA` / `SGL_LORA_SCALE` | CPU | LoRA adapter directory and extra scale |
| `SGL_DRAFT_LAYERS` | CPU | Truncated draft depth for speculative decoding |
| `SGL_SPEC_PROF` | CPU | Print the spec-decoding stage breakdown |
| `SGL_CLIP_LAYER` | Multimodal | Override `mm_vision_select_layer` for CLIP |
| `SGL_CUDA_CTX` | GPU | Context cap (default 16384, bounded by VRAM) |
| `SGL_CUDA_FP16` / `SGL_CUDA_W8` | GPU | fp16 weights / int8 weights on device |
| `SGL_CUDA_GRAPH` | GPU | `0` disables decode CUDA graphs |
| `SGL_ATTN_TWOPASS` | GPU | Force the two-pass attention kernel |
| `SGL_ATTN_DEBUG` / `SGL_CUDA_PROF` / `SGL_CUDA_DBG` | GPU | Print the kernel actually used / per-layer timing / debug counters |
| `SGL_FORCE_CUBLAS`, `SGL_SMALLM_CUBLAS`, `SGL_W8_DP4A`, `SGL_FORCE_LONG_ATTN` | GPU | Force a specific GEMM/attention path (A/B experiments, slower in production) |
| `SGL_HTTP_CONC` | GPU server | Concurrency (1–8, default 8; segment length = ctx / concurrency) |
| `SGL_HTTP_TRACE` | both servers | Per-connection tracing (accept / read / reply / close) |
| `SGL_PFX_MAX` | GPU server | Prefix-cache slots; `0` disables the cache |

Debug-only: `SGL_IMG_DEBUG` (scheduler image path), `SGL_DEBUG_IDS` / `SGL_DEBUG_GAP` (token ids / top1-top2
gap per step in `cuda_cli`), `SGL_FULL_RECOMP` (disable the cached-KV fast path in `cuda_cli`).

## Engineering log

Chronological notes on the hard parts. Everything here was measured; the full write-ups with logs live in
[CHANGELOG.md](CHANGELOG.md), [docs/NOTES_gaps.md](docs/NOTES_gaps.md),
[docs/NOTES_g8.md](docs/NOTES_g8.md), [docs/NOTES_g9.md](docs/NOTES_g9.md) and
[docs/PROGRESS_g11_g19.md](docs/PROGRESS_g11_g19.md).

### M26 — long context + CUDA graphs (2026-02)

**Long context (two-pass attention lifts the 4096 clamp).** The original design used the shared-scores
kernel up to 4096 tokens and a two-pass kernel above it (1024-token chunks, online softmax, K recomputed
twice for fixed shared memory). After G1 routed long rows through flash and G16 dropped the threshold to 64
tokens, **the default path only uses the two-pass kernel when `SGL_ATTN_TWOPASS=1` or the head dimension has
no specialization (`head_dim % 32 != 0`)**; the llama-family kernel choice lives in one place
(`attn_dispatch()`). GPT-2 has its own forward loop and still uses the shared-scores kernel, so its long
context was not raised by G1/G16. `SGL_CUDA_CTX` overrides the context cap (default 16384, VRAM bound);
tied `lm_head` aliases `wte` on device (saves 545 MB for Qwen2.5-0.5B); activation workspaces grow on demand.
Verified: two-pass vs original kernel maxerr 2.2e-5 with identical greedy output; a 4300-token prompt with
batched serving (two KV segments + graph replay) matches single-stream token by token; kernel determinism
(two identical prefills are bit-exact) is part of `cuda_test`. HTTP end to end: a 5000+ token prompt
(chunked prefill, 3 segments) answers the in-text needle ("The answer is 42.").

**CUDA graphs (decode).** `decode_batch` captures and replays every step (H2D input copies captured into the
graph, a graph-safe KV-write kernel computes addresses from a device array, so changing lens needs no
re-capture). `SGL_CUDA_GRAPH=0` turns it off. 4-sequence batch **177 → 335 tok/s (+89%)**, output identical
with graphs disabled.

**A real bug fixed on the way:** the two-pass kernel's block reduction had no barrier between "read
`red[0]`" and "write `red` for the next chunk" — found with compute-sanitizer racecheck (~390k hazards);
reproducing 10 launches corrupted bit-exactness twice and occasionally produced all-NaN. Two
`__syncthreads()` calls eliminated it.

### Gaps G1–G6 (2026-10-27)

All performance numbers are same-process A/B (`cuda_perf --ab`).

- **G1 flash prefill** — long rows no longer recompute K twice; the single-pass chunked kernel
  (`k_attn_flash`: Q blocks resident in shared memory, streamed K/V, online softmax). Initially enabled only
  above 4096 rows (8192-token prefill 72.4 s → 8.8 s, 8.3x, on a VRAM-pressured box); G16 lowered the
  threshold to 64 tokens, and a clean same-process A/B gives 8192-token **66.1 s → 2.7 s (23.6x)** and
  2048-token 11.8x.
- **G2 W8 large-M on tensor cores** — see performance highlights above.
- **G3 decode graphs on by default** — a plain warm-up step runs before capture (cuBLAS algorithm selection
  cannot happen inside capture); failures only mark **that batch size** instead of flipping a process-wide
  switch. `cuda_http` pre-warms workspaces/images and pre-captures B=1..conc before serving.
- **G4 HTTP hang fix** — `active_conns` got its own `conns_lock` instead of sharing `srv->lock` with the
  compute loop, which starved the acceptor under load (connection handshaked but no thread created for a
  long time). Re-verified: 600/600 `GET /health` at 12-way concurrency with no timeout.
- **G5 VRAM pre-check** — weights + KV + workspace + logits are summed before upload; if it does not fit the
  server **degrades instead of dying**: `/health` returns `{"ok":false,"gpu":false,"degraded":true,
  "reason":"..."}` and `/generate` answers 503 with a suggestion (lower `SGL_CUDA_CTX`, enable
  `SGL_CUDA_FP16`/`SGL_CUDA_W8`).
- **G6 GPU prefix cache** — the KV segment of a finished request is leased to a prefix cache; a new request
  compares tokens for the longest common prefix, copies matching rows with `sgl_cuda_copy_kv` and prefills
  only the suffix (LRU eviction, leases released on reuse). Output is **byte-identical** to the uncached path
  (`cuda_test [prefix]` maxerr 1.43e-05, greedy 6/6 identical); 756-token shared prefix 0.76 s → 0.34 s.

### Gaps G11–G19 (2026-10-27, all measured)

- **G16 flash from 64 tokens** — qwen05 int8, same-process best-of-3: 128 tok 1.3x, 512 tok 3.7x, 2048 tok
  **11.8x**, 4096 tok **20.2x**; below 64 tokens the short kernel stays slightly faster. Added a D=96
  specialization and a `D % 32` guard (a non-divisible ND silently computed only the first 64 dims).
  `SGL_ATTN_DEBUG=1` prints the kernel actually used.
- **G15 prefix leases cover generated tokens** — registered rows changed from "prompt only" to "rows that
  really hold valid KV" (`n_prompt + n_gen - 1`), so multi-turn chat reuses the whole segment (matched rows
  552 → **573**); image requests now register their text portion too. Same prompt twice: 0.262 s → 0.103 s.
- **G11 GPU-side PNG/JPEG** — `cuda_http` dispatches on the data-URL mime and runs the same stb_image +
  PIL-semantics resize/crop + CLIP + projector path as the CPU build (it previously accepted only raw-float
  base64).
- **G12 real LLaVA** — two-layer MLP projector, visual tokens inserted at the `<image>` position (suffix text
  positions shifted so RoPE follows), `mm_vision_select_layer=-2` honored; this also fixed a tokenizer bug
  where added tokens matched only at pre-token boundaries (so `USER: <image> ...` never produced an
  `<image>` id).
- **G13 Qwen3-Next** — the recurrent-state ownership key moved from "first KV page" to a 64-bit sequence key
  (two sequences sharing a radix prefix used to fight over one slot and corrupt each other's state).
- **G14 speculative decoding** — pipelined (prefill n-1, park the last prompt token, fixed n=K+1 verify, no
  per-round target graph step) plus small-M GEMV templates (MT=2/4/8). K=1 0.44x → 0.73x, **still a net
  slowdown**: it needs >88% acceptance to pay off, measured 39.1% (the draft model is too weak).
  `SGL_SPEC_PROF=1` prints the stage breakdown.
- **G17 GPU HTTP hang (different bug from G4)** — `read_request` ran `strstr("Content-Length:")` on a buffer
  that was not NUL-terminated, so under load it hit stale bytes from a previous POST and hung body-less
  `GET /health` until the client timed out (12-way load: **0/600** before, **600/600** after).
  `SGL_HTTP_TRACE=1` traces each connection.
- **G18 CPU performance** — the MSVC build previously ran SSE2 only (AVX2 dot existed for GCC only) and was
  10.2x slower than GCC at the same thread count; a separate AVX2 translation unit plus CPUID dispatch
  brought 930-token prefill 326.2 s → 33.6 s. 8K GPU-vs-CPU absolute check: **maxerr 1.48e-04 < 1e-3**,
  greedy identical ([docs/g18_8192.log](docs/g18_8192.log)).
- **G19 CI runs real model families** — CI generates five random families itself (byte-identical to the repo
  fixtures) and the gate changed from "ctest exit code 0" to "`test_family_*` must be Passed and `test_llava`
  must print all OK".

### LoRA adapters (M21)

```bash
export SGL_LORA=path/to/adapter_dir   # adapter.safetensors + adapter_config.json
export SGL_LORA_SCALE=1.0             # optional extra scale
sglc --model models/tinyllama --prompt "..."
```

Standard PEFT naming (q/k/v/o/gate/up/down all supported); merge semantics `W += B·A·(alpha/r)`, verified
greedy-identical to a numpy merge reference (`test_lora`); mmap'd weights are copied to a writable heap
copy automatically. CUDA×LoRA verified identical (α=1 and α=1/16, GPU == numpy merge per token). The
formerly suspected "LoRA×CUDA divergence" turned out to be an off-by-one in `cuda_cli` decode positions
(the first decode wrote at `np+1`, skipping `np`, so attention read an unwritten page) — fixed.

## Architecture

```text
sglc CLI ── HTTP (http.c) ── Scheduler (scheduler.c) ── RadixCache (radix.c)
                                  │  continuous batching / streaming / stop sequences
                                  ▼
                          Model abstraction (model.c)  ←── gpt2.c / llama.c (incl. Qwen2 differences)
                                  │  paged KV pool (refcounted)
                                  ▼
                     Operators (ops.c, quant.c)  ← SIMD dot products / W8A16 / int8 KV
                                  ▼
              Foundation: safetensors (mmap zero-copy) / BPE+SP tokenizer / JSON / log / alloc
```

The CUDA backend (`src/cuda_backend.cu`) plugs in behind the same operator layer: adaptive GEMM (hand-written
kernel for M=1, cuBLAS for M>1), flash prefill, decode CUDA graphs, W8 dp4a/IMMA, and the GPU prefix cache.

## Validation methodology

Every component is checked at three levels:

1. **Unit** — hand-computed values and property assertions (e.g. beam W=1 == greedy, scheduler batch output
   == serial output).
2. **Independent reference** — a from-scratch numpy forward pass sharing no code with the C implementation,
   plus HF tokenizers (Rust) and tiktoken for tokenization.
3. **End to end** — real weights plus factual checks ("the capital of France is **Paris**").

The suite is 26 CTest targets (`ctest --test-dir sglang-c/build`), including randomly generated model
families for regression coverage of the llama/mistral/gemma/gemma2/qwen3/qwen3-next code paths.

## Known limitations

- WinLibs GCC 16.2's AVX2 auto-vectorization crashes intermittently on this machine (minimal reproducer
  isolated), so the build uses `-march=x86-64-v2` + hand-written SSE (see the comment in `CMakeLists.txt`);
  the AVX2 dot product is opened per-function / per-file with CPUID dispatch at runtime.
- Activation outliers are heavy-tailed (row maxabs/median up to 1311x), which rules out W8A8 on CPU; W8A16
  is used instead (see ROADMAP M12).
- Untrained truncated drafts accept ~0% of tokens; even llama-160m against tinyllama only reaches 39% at
  K=1 — so speculative decoding is still a net slowdown today (G14).
- **MSVC builds don't enable OpenMP** (`-fopenmp` is GNU-only): measured CPU prefill gains only 2.05x from
  8 threads with identical numerics; `/openmp` can be added to MSVC when needed (G18).
- The prefix cache is a **segment lease** (one slot per concurrent segment, token-exact comparison), not the
  CPU-side radix tree; multiple images still collapse into the **same** `<image>` position (LLaVA is one
  image per placeholder, G12/G15).
- CI does not run CUDA or real LLaVA weights (runners have no NVIDIA driver and don't download 1.7 GB
  checkpoints); `test_llava_real` and `cuda_test` only run locally when the fixtures are present.
- The GPT-2 path does not use the G1/G16 attention optimizations (still the shared-scores kernel, so long
  context is bounded by shared memory), and only the llama-family `cuda_http` has the shared-prefix cache
  (the CPU build uses the radix tree).

## Repository layout

```text
sglang-c/
├── apps/           sglc.c (CLI + HTTP), cuda_cli.c, cuda_http.c
├── include/        public headers (sgl/*.h)
├── src/            model backends, scheduler, radix cache, ops, quantization, tokenizer, HTTP, CUDA backend
├── tests/          CTest targets, numpy/HF reference scripts, perf harnesses, fixture generators
├── docs/           benchmark logs, gap notes, progress reports
├── models/         small generated fixtures used by the suite
├── quickstart.bat  one-click launcher (Windows)
├── quickstart.sh   one-click launcher (Linux / macOS / Git Bash)
├── build_all.bat   full CPU + CUDA build and test
├── ROADMAP.md      milestone plan
└── CHANGELOG.md    per-change log with measurements
```

## License

Apache 2.0 — see [LICENSE](LICENSE). Portions of the design are derived from
[sgl-project/sglang](https://github.com/sgl-project/sglang).