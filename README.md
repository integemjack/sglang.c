# sglang-c

[![build](https://github.com/integemjack/sglang.c/actions/workflows/build.yml/badge.svg)](https://github.com/integemjack/sglang.c/actions/workflows/build.yml)

**用纯 C 重写的 SGLang 推理引擎** —— 从 [sgl-project/sglang](https://github.com/sgl-project/sglang)（核心 Python 约 138 万行）重构而来的 CPU 推理栈，零第三方依赖（仅 MinGW-w64 运行库静态链接）。

> 进度：M0–M26 + t1–t12 完成（见 [ROADMAP.md](ROADMAP.md) / [CHANGELOG.md](CHANGELOG.md)）。CPU 20/20 + CUDA 全绿；GPT-2 / TinyLlama-1.1B / Qwen2.5-0.5B / **Qwen3-0.6B** 真实权重验证。

## 下载（预编译）

[Releases](https://github.com/integemjack/sglang.c/releases) 提供 CI 构建的全平台产物（打 `v*` tag 自动构建发布）：

| 包 | 内容 |
|---|---|
| `linux-x64 / linux-arm64 / macos-arm64 / macos-x64` | `sglc`（CPU，OpenMP 多线程） |
| `windows-x64` | `sglc.exe`（CPU，MSVC /MT 静态） |
| `windows-x64-cuda` | `sglc.exe` + `cuda_cli.exe` + `cuda_http.exe`（RTX 显卡加速，sm_75+；需驱动，附带 cublas64 DLL） |

## 能力一览

| 功能 | 对应 sglang 模块 | 状态 |
|---|---|---|
| GPT-2 / Llama / Qwen2 / **Qwen3** 后端 | `srt/models/` | ✅ 与 numpy/HF 数值对照（maxerr ≤ 2e-4，SIMD 后 1e-8） |
| **Mistral 滑窗 / Gemma 归一化** | `srt/models/` | ✅ 随机结构 numpy 全等（tests/gen_random_*.py） |
| 连续批处理调度器 | `srt/managers/scheduler.py` | ✅ 批处理输出与串行逐 token 全等，3 并发 +68% |
| HTTP 服务（OpenAI 兼容，CPU+GPU 双栈） | `entrypoints/http_server.py` | ✅ /generate /v1/completions /v1/chat/completions + 超限排队 + finish_reason 如实 |
| RadixAttention 前缀缓存 | `srt/mem_cache/radix_cache.py` | ✅ 分页 KV + 引用计数 + LRU 驱逐，长 prompt 命中提速 ~40% |
| W8A16 量化（CPU）/ **W8A8 dp4a（CUDA）** | `srt/layers/quantization` | ✅ CPU decode 1.76x；CUDA 显存最优档（free VRAM 3947MB） |
| int8 KV cache | fp8/int8 KV 路线 | ✅ 内存减半，吞吐 +20% |
| Beam search | `srt/beam_search/` | ✅ W≤8，W=1 == greedy 硬断言 |
| Speculative decoding v2 | `srt/speculative/` | ✅ n-token 批量 verify；160m draft 接受率 13%，spec==greedy 保证 |
| 多 LoRA 热切换 | `srt/lora/` | ✅ 运行时 add/remove，N 卡 N 选 1 |
| **CUDA FP16 / Graph / 长上下文** | `srt/` | ✅ 1.1B fp16 入 6GB 卡；graph 批处理 +89%；ctx 16384 |
| **LLaVA 多模态（CLIP+融合+HTTP）** | `srt/models/llava.py` | ✅ CLIP ViT + projector + embedding 级融合 + `/v1/chat/completions` image_url（raw float base64；JPEG 解码不支持）；numpy 全等 9.3e-9，GPU 融合 vs CPU 融合 1.2e-7 |
| **CPU OpenBLAS / AVX2 dot** | backend | ✅ 大 M GEMM 3.1x（195 GFLOPs），运行时分派零依赖可选 |
| SSE 流式 | `--stream` | ✅ 流式拼接与非流式严格全等 |

**未覆盖**：张量并行、真实 LLaVA 权重端到端（结构已验证）、GPU 侧 sliding window。

## 快速开始

```bash
# 构建（MinGW-w64 + CMake + Ninja）
cmake -S sglang-c -B sglang-c/build -G Ninja
cmake --build sglang-c/build
ctest --test-dir sglang-c/build          # 20 套测试（部分需 models/ 下真实权重）

# 交互生成（GPT-2 124M / TinyLlama-1.1B / Qwen2.5-0.5B）
sglang-c/build/bin/sglc --model models/gpt2 --prompt "The meaning of life is" --max-new 32
sglang-c/build/bin/sglc --model models/tinyllama --w8 --kv8 --max-new 64   # 量化提速

# HTTP 服务（OpenAI 兼容）
sglang-c/build/bin/sglc --model models/qwen05 --serve --port 30000 --w8
curl localhost:30000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

模型目录格式（HF 标准）：`config.json` / `tokenizer.json` / `model.safetensors`（F32/F16/BF16）。

## 环境要求（本机验证过的工具链版本）

| 组件 | 版本 | 说明 |
|---|---|---|
| Windows CPU 栈 | WinLibs GCC **16.2**（winget `BrechtSanders.WinLibs.POSIX.UCRT`）| CMake ≥ 3.20 + Ninja；产物静态链接不依赖运行库 DLL |
| CUDA 栈 | VS2022 Build Tools（MSVC 14.44）+ CUDA Toolkit **13.3** | sm_75+（RTX 20 系起）；与 MinGW 栈共存于独立 build 目录 |
| 可选 OpenBLAS | **0.3.29 x64.zip**（Release 页） | 注意不是 `x64_64`（那是 64 位 int ABI，动态调用会静默读出垃圾值）；解压到 `third_party/openblas` 后 `-DSGL_OPENBLAS=ON` |
| 源码编码 | UTF-8 **带 BOM** | MSVC/nvcc 按 GBK 读无 BOM 的中文注释会吞结构体声明；CMake configure 阶段有自检警告 |
| 测试模型 | `models/` 目录 | 真实权重：gpt2 / tinyllama / qwen05 / qwen3-06 / llama-160m；随机生成（tests/gen_random_*.py）：mistral-rand / gemma-rand / llava-rand |

## 性能（i7-10700F 8 核，TinyLlama-1.1B fp32）

| 配置 | 吞吐 |
|---|---|
| 基线 | 1.5 tok/s |
| + SIMD（SSE2/SSE4.1 手写 intrinsics） | ~3 tok/s |
| + W8 权重 + int8 KV | **7.4 tok/s**（累计 4.9x） |
| GEMM 峰值 | matmul_nt 8.4 GFLOPs（4x 于标量） |

Qwen2.5-0.5B chat 实测 4.2 tok/s（W8+KV8+SIMD；其 151k 大词表的 lm_head 是主要成本）。

**CUDA 后端（RTX 2060，sm_75；M23 自适应 GEMM：M=1 手写 kernel / M>1 cuBLAS）**：
Qwen2.5-0.5B：单流 **45.7 tok/s**（10.9x CPU）；**批处理（4 序列）105.2 tok/s**
（手写 64.9 → cuBLAS 93.6 → 自适应+lm_head 合并 105.2，累计 +62%）；
prefill maxerr 3.5e-5（vs CPU），greedy 三方一致；批处理输出与各自单流全等；KV 段隔离。
独立 CLI：`cuda_cli.exe models/qwen05 "prompt" 40 0`（自适应后单流回到 45 tok/s；
prompt 支持 `@file` 读长文本）。
GPU HTTP 服务：`cuda_http.exe <model_dir> [port]` —— /health /generate
/v1/completions /v1/chat/completions（ChatML）；KV 段调度（ctx/N 段）+ 批处理
worker；已验证：单请求正确（Paris 事实性回答）+ **两轮 8 并发全部成功且服务存活**
（根因：req_free 用 free() 释放 sgl_xmalloc 的 text_out——对齐分配/free 不匹配）。
`SGL_HTTP_CONC` 覆盖并发数（1-8，默认 8；段长 = ctx/并发）。
构建：`cmake -S sglang-c -B sglang-c/build-cuda -G "Visual Studio 17 2022" -A x64 -DSGL_CUDA=ON`
`cmake --build sglang-c/build-cuda --config Release --target cuda_test cuda_cli cuda_http`

## M26：长上下文 + CUDA Graph（2026-02）

**长上下文（两段式 attention 解除 4096 钳制）**：≤4096 走原 shared-scores kernel
（数值路径不变），>4096 自动切两段式 kernel（chunk 1024 online-softmax，K 重算两遍
换固定 shared 内存）；`SGL_CUDA_CTX` 覆盖上下文上限（默认 16384，受显存约束）；
tied lm_head 在设备端别名 wte（Qwen2.5-0.5B 省 545MB）；激活 workspace 按需增长。
验证：两段式 vs 原 kernel maxerr 2.2e-5 且 greedy 一致；4300-token prompt 批处理
（双 KV 段 + graph 重放）与单流逐 token 全等；kernel 确定性检查（同输入两次 prefill
bit-exact）进 cuda_test。HTTP 端到端：5000+ token prompt（chunked prefill，3 段）
正确答出文中 needle（"The answer is 42."）。
**CUDA Graph（decode）**：`decode_batch` 全步 capture/replay（H2D 输入拷贝入图 +
graph-safe KV 写 kernel 从 device 数组算地址，lens 每步变化无需重建图），
`SGL_CUDA_GRAPH=0` 关闭。4 序列批处理 **177 → 335 tok/s（+89%）**，输出与关闭时全等。
**修一个真 bug**：两段式 kernel 的 block 归约在「读 red[0]」与「下一 chunk 写 red」
之间缺 barrier —— compute-sanitizer racecheck 定位（39 万 hazards），复现 10 次
launch 2 次 bit-exact 破坏、偶发全 NaN；补 2 个 `__syncthreads()` 后消除。

## LoRA 适配器（M21）

```bash
export SGL_LORA=path/to/adapter_dir   # 含 adapter.safetensors + adapter_config.json
export SGL_LORA_SCALE=1.0             # 可选额外缩放
sglc --model models/tinyllama --prompt "..."   # 加载时合并，零推理开销
```
PEFT 标准命名（q/k/v/o/gate/up/down 全支持）；合并语义 `W += B·A·(alpha/r)`，
与 numpy 合并参考 greedy 全等验证（test_lora）；mmap 权重自动换堆可写副本。
CUDA×LoRA：已验证全等（α=1/16 两档，GPU == numpy 合并参考逐 token）；
此前误报的「LoRA×CUDA 分叉」实为 cuda_cli decode 位置 off-by-one
（首 decode 写在 np+1 跳过位置 np，attention 读到未写页垃圾）——已修复，
并附带澄清：numpy 参考含 qkv-bias 修复后的新 truth，旧 [553,...] 为 M11 期错误参考。

## 架构（对应关系）

```
sglc CLI ── HTTP (http.c) ── Scheduler (scheduler.c) ── RadixCache (radix.c)
                                  │  连续批处理 / 流式 / stop/n
                                  ▼
                          Model 抽象层 (model.c)  ←── gpt2.c / llama.c（含 Qwen2 差异项）
                                  │  分页 KV 池（引用计数）
                                  ▼
                    算子层 (ops.c, quant.c)  ← SIMD 内积 / W8A16 / int8 KV
                                  ▼
              基础库：safetensors(mmap 零拷贝) / BPE+SP tokenizer / JSON / log / alloc
```

## 验证方法学

每个组件三层对照：
1. **单元**：手算/性质断言（如 beam W=1 == greedy）
2. **独立参考**：numpy 从零实现的前向（与 C 零共享代码）、HF tokenizers（Rust）、tiktoken
3. **端到端**：真实权重生成 + 事实性检查（"capital of France is **Paris**"）

## 已知问题

- WinLibs GCC 16.2 的 AVX2 自动向量化在本机间歇崩溃（最小复现已定位），故用 `-march=x86-64-v2` + 手写 SSE（见 CMakeLists 注释）
- 激活重尾 outlier（行内 maxabs/median 达 1311x）使 W8A8 不可行，采 W8A16（见 ROADMAP M12）
- 未训练的截层 draft 接受率 ~0%（spec 加速需专门训练的小模型，M19）
