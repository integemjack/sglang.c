> **中文版（备选）** · [English (default) ▸](README.md)
>
> 英文版是主 README，内容更新、结构更完整；本文件保留中文说明作为备选参考。

## ⚡ 一键脚本（快速上手）

脚本会自己检测工具链、缺什么就编译什么，然后起 OpenAI 兼容 HTTP 服务、等 `/health`、发一条 demo 请求，最后打印端点：

```bat
:: Windows：双击即可，或在终端里
sglang-c\quickstart.bat
```

```bash
# Linux / macOS / Git Bash
bash sglang-c/quickstart.sh
```

| 用法 | 作用 |
| --- | --- |
| `quickstart.bat` / `quickstart.sh` | 缺则编译 → 端口 30000 起服务 → 发 demo 请求 |
| `... cli [model] [n]` | 单次命令行生成 |
| `... gpu [model] [port]` | 起服务（同一个 `sglc --serve`，有显卡走 GPU） |
| `... test` / `... build` / `... help` | 跑 ctest / 全量 CPU+CUDA 编译 / 用法 |

环境变量：`SGL_MODEL`（模型目录）、`SGL_PORT`（端口）、`SGL_NO_BUILD=1`（只跑不编）、`SGL_NO_PAUSE=1`（Windows 结束不按任意键）。

# sglang-c

[![build](https://github.com/integemjack/sglang.c/actions/workflows/build.yml/badge.svg)](https://github.com/integemjack/sglang.c/actions/workflows/build.yml)

**用纯 C 重写的 SGLang 推理引擎** —— 从 [sgl-project/sglang](https://github.com/sgl-project/sglang)（核心 Python 约 138 万行）重构而来的 CPU 推理栈，零第三方依赖（仅 MinGW-w64 运行库静态链接）。

> 进度：M0–M26 + t1–t12 完成（见 [ROADMAP.md](ROADMAP.md) / [CHANGELOG.md](CHANGELOG.md)）。CPU 26/26 + CUDA 全绿；GPT-2 / TinyLlama-1.1B / Qwen2.5-0.5B / **Qwen3-0.6B** 真实权重验证。

## 下载（预编译）

[Releases](https://github.com/integemjack/sglang.c/releases) 提供 CI 构建的全平台产物（打 `v*` tag 自动构建发布）：

| 包 | 内容 |
|---|---|
| `linux-x64 / linux-arm64` | `sglc`（CPU，GNU 编译器下开 OpenMP 多线程） |
| `linux-x64-cuda / linux-arm64-cuda` | `sglc` + `cuda_cli` + `cuda_http`（附 `libcublas*.so`）；运行时需 NVIDIA 显卡与驱动 |
| `macos-arm64` | `sglc`（CPU，**单线程**：AppleClang 无 `-fopenmp`，CI 不装 libomp） |
| `macos-arm64-mlx` | 同上，另附 `tools/ref_llava_mlx.py`：数值参考改用 `mlx.core`（Metal GPU）重算，`test_llava` 变成「C 实现 vs MLX」 |
| `windows-x64` | `sglc.exe`（CPU + HTTP 服务，MSVC /MT 静态，单线程；`--serve` 可用） |
| `windows-x64-cuda` | `sglc.exe`（CPU+GPU 双引擎 + HTTP 服务，`--device auto`）+ `cuda_cli.exe` + `cuda_http.exe`（RTX 显卡加速，sm_75+；需驱动，附带 cublas DLL） |
| `windows-arm64-cuda` | **preview**：Windows on Arm 版，用 CUDA 13.4.0 Developer Preview 工具链构建（同时含 sm_75/sm_120 cubin，面向 RTX Spark 这类 Blackwell 级 WoA 机器，驱动 616.41+） |

构建矩阵（`.github/workflows/build.yml`）：`linux-x64`、`linux-arm64`、`linux-x64-cuda`、`linux-arm64-cuda`、
`macos-arm64`、`macos-arm64-mlx`、`windows-x64`、`windows-x64-cuda`，外加独立的 `windows-arm64-cuda` job
（GitHub 的 arm64 Windows runner）。每次 push 全量构建并打包；CUDA 与 arm64 项在无 GPU 的 runner 上跑不起来，
只做 smoke（产物存在 + `ldd`/`dumpbin` 架构与依赖 + `sglc` usage 退出码），实际推理留给真机。
v0.4.0 时 9 个目标全绿（run
[40456249230](https://github.com/integemjack/sglang.c/actions/runs/40456249230)）。

真机实测过的项与验法：`docs/PROGRESS_g11_g19.md`。linux-x64 / linux-arm64 / windows-x64 /
windows-msvc-cuda 均已实跑；**macos-arm64 已在真机 iMac（macOS 27.0.1 / AppleClang 21）上编译并
ctest 24/24 通过**（`docs/g_macos_arm64.txt`）。`macos-x64`（Intel）已于 **2026-10-27 删除**：需要从
arm64 宿主交叉编译 + Rosetta 跑测试，而 Intel runner 已退场。

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
| **LLaVA 多模态（CLIP+融合+HTTP）** | `srt/models/llava.py` | ✅ CLIP ViT + **两层 MLP projector**（自动判层数，GELU）+ 视觉 token **按 `<image>` 位置插入** + `/v1/chat/completions` image_url（**PNG/JPEG 直收**，raw-float base64 仍兼容）；随机结构 numpy 全等 9.3e-9；**真实 CLIP-L/14-336 + 真实 llava mm_projector** 对 transformers 分段对照 rel **1.4e-05 / 1.8e-05** |
| **Qwen3-Next（混合层 + MoE）** | `srt/models/qwen3_next.py` | ✅ linear_attention（gated deltanet）+ full 层的 q 门控 / partial rotary / `(1+w)` RMSNorm / MoE（top-k 路由 + 共享专家 + routed scaling）；随机结构 vs float64 numpy **2.5e-08–5.8e-08**；递归状态按**序列 key** 归属（前缀共享不互踩） |
| **CPU OpenBLAS / AVX2 dot** | backend | ✅ 大 M GEMM 3.1x（195 GFLOPs），运行时分派零依赖可选 |
| SSE 流式 | `--stream` | ✅ 流式拼接与非流式严格全等 |

**未覆盖**：张量并行、真实 LLaVA **7B 语言模型**端到端（CLIP-L/14-336 与 mm_projector
已用真实权重分段对照，7B LM 因磁盘/显存未下）、GPU 侧 sliding window、
Qwen3-Next 的 CUDA 路径（仅 CPU 参考实现）。

## 快速开始

```bash
# 构建（MinGW-w64 + CMake + Ninja）
cmake -S sglang-c -B sglang-c/build -G Ninja
cmake --build sglang-c/build
ctest --test-dir sglang-c/build          # 26 套测试（部分需 models/ 下真实权重）

# 交互生成（GPT-2 124M / TinyLlama-1.1B / Qwen2.5-0.5B）
sglang-c/build/bin/sglc --model models/gpt2 --prompt "The meaning of life is" --max-new 32
sglang-c/build/bin/sglc --model models/tinyllama --w8 --kv8 --max-new 64   # 量化提速

# HTTP 服务（OpenAI 兼容）；--device auto 默认 cuda → mlx → cpu
sglang-c/build/bin/sglc --model models/qwen05 --serve --port 30000 --w8
sglang-c/build-cuda/bin/Release/sglc.exe --model models/qwen05 --serve --port 30000   # 有显卡走 GPU
sglang-c/build-cuda/bin/Release/sglc.exe --model models/qwen05 --serve --device cpu   # 强制 CPU 引擎
curl localhost:30000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'

# Apple Silicon（Metal / MLX）：-DSGL_MLX=ON 产出 mlx_test / mlx_q4_test / mlx_http
cmake -S sglang-c -B sglang-c/build-mlx -G Ninja -DSGL_MLX=ON \
      -DSGL_MLX_ROOT=$(python3 -m mlx --cmake-dir)
cmake --build sglang-c/build-mlx --target mlx_test mlx_q4_test mlx_http -j 8
sglang-c/build-mlx/bin/mlx_test models/qwen05 16              # 与 CPU 金标准对拍（13 项）
sglang-c/build-mlx/bin/mlx_q4_test models/qwen05-4bit 9707    # 4-bit logits 对拍 mlx_lm
sglang-c/build-mlx/bin/sglc --model models/qwen05-4bit --serve --device mlx --port 30001
```

量化检查点直接用 `mlx-community` 的 4-bit（`.weight` 是 U32 打包 + `.scales`/`.biases`），
不做成 fp16：Qwen2.5-0.5B 磁盘 276 MB（fp16 是 951 MB），decode 75 tok/s。

模型目录格式（HF 标准）：`config.json` / `tokenizer.json` / `model.safetensors`（F32/F16/BF16）。

## 环境要求（本机验证过的工具链版本）

| 组件 | 版本 | 说明 |
|---|---|---|
| Windows CPU 栈 | WinLibs GCC **16.2**（winget `BrechtSanders.WinLibs.POSIX.UCRT`）| CMake ≥ 3.20 + Ninja；产物静态链接不依赖运行库 DLL |
| CUDA 栈（x64） | VS2022 Build Tools（MSVC 14.44）+ CUDA Toolkit **13.3** | sm_75+（RTX 20 系起）；Windows x64 与 Linux（x64 / arm64-sbsa）都用 13.3 |
| CUDA 栈（Windows on Arm） | arm64 MSVC + CUDA Toolkit **13.4.0 Developer Preview** | 第一个带 Windows-on-Arm 主机工具的 CUDA；目前只有 preview 安装器，所以 arm64 包是实验性产物 |
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

**MLX 后端（Apple M1 / 8 核 / 16 GB，Metal 4；MLX 0.29.3）**：Qwen2.5-0.5B fp16
单流 **43.2 tok/s**（23.2 ms/步）、批量 4 序列 **119.6 tok/s**；
**同机同模型官方 mlx_lm 为 41.9 tok/s**（23.89 ms/步，`mlx_lm.generate` 128 token 实测）。
4-bit 量化检查点：单流 **75.0 tok/s**、并发 4 条 **111 tok/s**（磁盘 276 MB vs fp16 的 951 MB），
官方 mlx_lm 同检查点 80.7 tok/s。对拍：fixture 13/13，真模型 fp16 12/12、
fp32 13/13（跨路径概率误差 8.9e-7），4-bit 的 logits 与 `mlx_lm` top10 **逐位一致**。
定位过程与实测数据见 [docs/NOTES_mlx_t2.md](docs/NOTES_mlx_t2.md) 与
[docs/NOTES_mlx_q4.md](docs/NOTES_mlx_q4.md)。

**CUDA 后端（RTX 2060，sm_75；M23 自适应 GEMM：M=1 手写 kernel / M>1 cuBLAS）**：
Qwen2.5-0.5B：单流 **45.7 tok/s**（10.9x CPU）；**批处理（4 序列）105.2 tok/s**
（手写 64.9 → cuBLAS 93.6 → 自适应+lm_head 合并 105.2，累计 +62%）；
prefill maxerr 3.5e-5（vs CPU），greedy 三方一致；批处理输出与各自单流全等；KV 段隔离。
独立 CLI：`cuda_cli.exe models/qwen05 "prompt" 40 0`（自适应后单流回到 45 tok/s；
prompt 支持 `@file` 读长文本）。
GPU HTTP 服务：
**就是同一个 `sglc.exe`** —— `sglc --serve --port 30000`（`--device auto` 默认：cuda → mlx → cpu）。
`-DSGL_CUDA=ON` 的构建把 GPU 引擎（原 `apps/cuda_http.c`，现在只留 `sgl_cuda_http_serve()`）
一起链进 `sglc.exe`，因此 Windows/MSVC 版本也直接带 `--serve`（CPU 版 http.c 走
`include/sgl/thread_compat.h` 的 Win32 线程垫片，不再依赖 MinGW 的 winpthreads）。
端点：/health /generate /v1/completions /v1/chat/completions（ChatML）；KV 段调度（ctx/N 段）+ 批处理
worker；已验证：单请求正确（Paris 事实性回答）+ **两轮 8 并发全部成功且服务存活**
（根因：req_free 用 free() 释放 sgl_xmalloc 的 text_out——对齐分配/free 不匹配）。
`SGL_HTTP_CONC` 覆盖并发数（1-8，默认 8；段长 = ctx/并发）。
旧习惯仍可用：`cuda_http.exe <model_dir> [port]` 仍然编、仍然能跑（CI smoke 也还测它），
只是不必再带两个 exe 了。
构建：`cmake -S sglang-c -B sglang-c/build-cuda -G "Visual Studio 17 2022" -A x64 -DSGL_CUDA=ON`
`cmake --build sglang-c/build-cuda --config Release --target sglc`

## M26：长上下文 + CUDA Graph（2026-02）

**长上下文（两段式 attention 解除 4096 钳制）**：当时的设计是 ≤4096 走 shared-scores
kernel、>4096 切两段式 kernel（chunk 1024 online-softmax，K 重算两遍换固定 shared 内存）；
**G1 之后长行走 flash，G16 又把阀值降到 64 token**，所以**默认路径下两段式核只在
`SGL_ATTN_TWOPASS=1` 或 head_dim 无特化（不整除 32）时才用**；llama 族的 kernel 选择逻辑
集中在 `attn_dispatch()` 一处（GPT-2 有自己的前向循环，仍只用 shared-scores 核，
所以它的长上下文能力没被 G1/G16 抬高）。`SGL_CUDA_CTX` 覆盖上下文上限（默认 16384，受显存约束）；
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

## 缺口关闭 G1–G6（2026-10-27）

性能数字全部是**同进程 A/B**（`cuda_perf --ab`）：本卡空闲显存常常小于模型常驻量，
WDDM 溢出会把绝对时间放大 3–20x，跨进程的数字不能直接比。详见 docs/NOTES_gaps.md。

- **G1 flash prefill**：长行不再把 K 重算两遍，改走单遍分块核 `k_attn_flash`（Q 块驻
  shared、K/V 流式、online softmax）。初版只在 >4096 行切换，8192-token prefill
  72.4 → 8.8s（8.3x，**当时机器显存吃紧**）；**G16 把阀值降到 64 token**（短行上 short 核
  只在 64 以下略快），干净状态下同进程 A/B：8192-token **66.1 → 2.7s（23.6x）**、
  2048-token 11.8x。两个倍率不要互相引用（测量时的显存/并发状态不同）。
  回退/对比开关：`SGL_ATTN_TWOPASS=1`；逐层注意力计时：`SGL_CUDA_PROF=1`；
  实际用了哪个核：`SGL_ATTN_DEBUG=1`。
- **G2 W8 大 M 走 tensor core**：int8 权重模式下 M>1 的投影改用 `cublasGemmEx` IMMA
  （int8×int8→int32，整数累加与顺序无关），M=1 仍用手写 dp4a GEMV（实测更快）。
  [2048,2048] M=8：**0.925 → 0.069ms（13.4x）**；IMMA vs dp4a prefill **maxerr = 0**。
  强制旧路径：`SGL_W8_DP4A=1`。
- **G3 decode graph 默认开启**：捕获前先跑一次 plain step 预热（CUDA 13 默认 lazy 内核
  加载、cuBLAS 首次选算法都不能发生在 capture 里），失败只标记**这一个 batch 尺寸**
  （旧版一个进程级开关会让所有尺寸永久走 plain）；`cuda_http` 起服务前 `prewarm`
  把工作区/镜像撑到位并预捕获 B=1..conc。fp16 decode：B=1 23.0→9.7ms、B=4
  36.6→8.9ms、B=8 38.5→11.3ms（3–4x）。`SGL_CUDA_GRAPH=0` 仍可关。
- **G4 HTTP 卡死修复**：`active_conns` 从 `srv->lock` 拆出独立 `conns_lock`——之前
  acceptor 登记连接要和 worker 的每步计算抢同一把锁，高负载下饿死（连接已握手但
  长时间不建线程）。复验：12 并发负载下 600/600 次 GET /health 无超时。
- **G5 显存预检**：上传前先算账（权重+KV+工作区+logits+workspace），不够就**降级
  服务**而不是进程退出：`/health` 返回 `{"ok":false,"gpu":false,"degraded":true,
  "reason":"..."}`，`/generate` 直接 503 + 建议（降 `SGL_CUDA_CTX` 或开
  `SGL_CUDA_FP16=1`/`SGL_CUDA_W8=1`）。
- **G6 GPU 前缀缓存**：同一段 system prompt 的后续请求不再重算前缀——请求结束后它的
  KV 段被「租让」给前缀缓存，新请求按 token 精确比对最长公共前缀，命中部分用
  `sgl_cuda_copy_kv` 搬行、只 prefill 后缀（LRU 驱逐，段被复用时先释放租让）。
  输出与无缓存**逐字一致**（`cuda_test [prefix]` maxerr 1.43e-05、greedy 6/6 全同）；
  端到端 756-token 共享前缀：0.76s → **0.34s（2.2x）**。`/health` 的 `prefix` 字段给出
  hits/hit_rows/leased_rows（G15 又扩成 slots/hits/misses/hit_rate/flushes，并且租让
  现在也覆盖**生成 token 的行**与**带图请求的 text 部分**，同 prompt 重发 0.262s → 0.103s）。
- 新工具：`tests/cuda_perf.c`（同进程 A/B 基准）、`tests/conc_test.ps1`、
  `tests/prefix_e2e.ps1`、`tests/soak_curve.ps1`（RSS/VRAM 斜率）、`tests/g4_probe.py`、
  `tests/bom_fix.ps1`（批量补 UTF-8 BOM）。

## 缺口关闭 G11–G19（2026-10-27，全部实测）

数据与口径：`CHANGELOG.md` 的 `[Unreleased]` 节（每项都带实测数字与日志路径）、
`docs/BENCH.md`（G14/G15/G16/G18 四节）、`docs/NOTES_g8.md §8`（真实 llava）、
`docs/NOTES_g9.md §5`（Qwen3-Next）、`docs/PROGRESS_g11_g19.md`（逐轮过程）。

- **G16 flash 从 64 token 起就用**（阀值原 4096）：qwen05 int8 同进程 best-of-3 下
  128 tok 快 1.3x、512 tok 3.7x、2048 tok **11.8x**、4096 tok **20.2x**；64 tok 以下
  short 核仍略快，保留。head_dim 新增 D=96 特化 + `D % 32` 守卫（ND=D/32 不整除会静默
  只算前 64 维）。`SGL_ATTN_DEBUG=1` 打出实际用的核。
- **G15 前缀租让覆盖生成 token**：登记行数从「只算 prompt」改成「段里真有有效 KV 的行」
  （`n_prompt + n_gen - 1` → 多轮对话/续写整段复用：命中 552 → **573** 行）；带图请求也
  登记 text 部分（以前完全跳过）。同 prompt 重发 0.262s → **0.103s（2.5x）**。
  `/health` 的 `prefix` 字段给 `slots/hits/misses/hit_rate/hit_rows/leased_rows/flushes`；
  `SGL_PFX_MAX=0` 彻底关缓存。
- **G11 GPU 侧直收 PNG/JPEG**：`cuda_http` 按 data URL 的 mime 分派，走与 CPU 版同一条
  stb_image + PIL 语义 resize/crop + CLIP + projector 路径（以前只认 raw float base64）。
- **G12 真实 llava**：两层 MLP projector、视觉 token 按 `<image>` 位置插入（后缀文本位置
  顺延）、支持 `mm_vision_select_layer=-2`；**顺带修了 tokenizer 只在 pre-token 边界匹配
  added token 的 bug**（`USER: <image> ...` 里拿不到 `<image>` id）。
- **G13 Qwen3-Next**：见上方能力表；递归状态归属键从「首 KV 页」换成 64 位序列 key
  （radix 前缀共享时两条序列会抢同一个槽 → 状态互踩）。
- **G14 投机解码**：流水线化（预填 np-1、prompt 末 token 挂起、固定 n=K+1 verify，
  去掉每轮一次 target 图步）+ small-M GEMV（模板 MT=2/4/8）。K=1 **0.44x → 0.73x**，
  **但仍净减速**：回本需 K=1 接受率 >88%，实测 39.1%（draft 太弱）。`SGL_SPEC_PROF=1`
  打阶段分解。
- **G17 GPU 版 HTTP 卡死（与 G4 不同病）**：`read_request` 在**未 NUL 截断**的缓冲上用
  `strstr("Content-Length:")`，负载一大会撞上前一个 POST 留在堆里的残值，把无 body 的
  `GET /health` 挂到客户端超时（修前 12 并发 load：**0/600**）。修后 GPU **600/600**。
  `SGL_HTTP_TRACE=1` 可逐连接打印入口/收包/应答/退出。
- **G18 CPU 侧性能**：MSVC 构建之前**只跑 SSE2**（AVX2 内积只有 GCC 有），同线程下比
  GCC 慢 **10.2x**；新增单独编译单元的 AVX2 dot + CPUID 分派后 930-token prefill
  326.2s → **33.6s（9.7x）**。8K GPU-vs-CPU 绝对对照：**maxerr 1.48e-04 < 1e-3**、
  greedy 相同（日志 `docs/g18_8192.log`）。
- **G19 CI 真跑模型族**：CI 自己生成 5 个随机族（与仓库夹具逐字节相同），验收判据从
  「ctest 退码 0」改成「`test_family_*` 必须 Passed、`test_llava` 必须 all OK」。

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

- WinLibs GCC 16.2 的 AVX2 自动向量化在本机间歇崩溃（最小复现已定位），故用 `-march=x86-64-v2` + 手写 SSE（见 CMakeLists 注释）；AVX2 内积单独用函数级 target / 单文件 `/arch:AVX2` 开后运行时 CPUID 分派
- 激活重尾 outlier（行内 maxabs/median 达 1311x）使 W8A8 不可行，采 W8A16（见 ROADMAP M12）
- 未训练的截层 draft 接受率 ~0%（spec 加速需专门训练的小模型，M19）；
  llama-160m 对 tinyllama 也只有 39%（K=1）——所以 spec 目前仍净减速（G14）
- **MSVC 构建不开 OpenMP**（`-fopenmp` 只给 GNU）：实测 CPU prefill 开 8 线程只值 2.05x，
  数值路径完全一致，需要时再给 MSVC 加 `/openmp`（G18 补记）
- 前缀缓存是**段级租让**（一个并发段一个槽，token 精确比对），不是 CPU 侧的 radix 树；
  多图仍拼在**同一个** `<image>` 位置（llava 是一图一占位，G12/G15）
- CI 不跑 CUDA 与真实 llava 权重（runner 无 NVIDIA 驱动 / 不下 1.7GB 权重）；
  `test_llava_real` 与 `cuda_test` 只在本地有夹具时跑
- GPT-2 路径没接 G1/G16 的 attention 优化（仍是 shared-scores 核，长上下文受 shared 上限约束）；
  共享前缀缓存也只有 llama 族的 `cuda_http` 有（CPU 版是 radix 树）
