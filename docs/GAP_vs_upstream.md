# sglang-c vs 上游 sglang：能力差距盘点（2026-11-05 第四轮复核，HEAD `8a78c27`）

> 本版按**当前 HEAD**（v0.5.1）重新核过，对照 `../` 的 sgl-project/sglang（HEAD `169c88be`，2026-11-05，已 fetch fast-forward）。
> 上一版（2026-10-29，HEAD `264ab89`）之后：本仓 +22 提交（v0.5.0 发版批次：G20/G21/G22a/b、
> conformance 套件、hf_ref_check、CI 五处根因修复）；**上游 +802 提交**（10-25→11-05：
> 262 Fix / 183 Perf / 105 Feat / 79 Refactor / 70 CI / 53 Doc / 29 Test / 17 Chore，
> srt 下 1021 文件改动 +86,390/−28,254 行）。两边都动了，所以这版两头重数。
> 数字全部现数（第 6 节给了命令），不引用旧数字。

---

## 0. 量化基线

| 维度 | 上游 sglang（`169c88be`，11-05） | sglang-c（`8a78c27` + 本提交，v0.5.1） |
| --- | --- | --- |
| 代码量 | `python/` **1,687,095 行**（`srt/` 869,115 行）；另有 Rust router/gateway/server、`dllm/`、`rust_extensions` | `src`+`include`+`apps` **77 文件 36,453 行**（含 MLX 3 个 .cpp 共 3,311 行；约 1/46） |
| 模型实现 | `srt/models/` **259** 个顶层 `*.py`（**233** 处 `EntryClass`） | **3 个后端**：gpt2 / llama 系 / deepseek；llama 系内含 Qwen3-Next 风格 MoE 与 GDN hybrid；另有 MLX(Metal) 后端（macOS） |
| 层与算子 | `srt/layers/` 866 文件（quantization 118） | `ops.c` `tensor.c` `quant.c` `mla.c` `qweight.c` + 自写 CUDA kernel（`cuda_backend.cu`、`cuda_deepseek.cu`） |
| HTTP 路由 | `http_server.py` **79** 个静态唯一路径（装饰器 87 处，含动态注册；另有 Ollama 动态路由、gRPC、rust-server 原生前端） | CPU **≈46** 个实现 + 33 个明确 501；加速侧（`accel_http.c`，CUDA/MLX 共用）**≈46** 个实现 + 31 个 501（§6 给了数法） |
| 调度 / 缓存 | `mem_cache/` **229** 文件、`speculative/` **70** 文件 | `scheduler.c` `radix.c` `spec.c` `beam.c` `cuda_spec.c` + G20 页级 KV（零拷贝挂页） |
| 多模态 | `srt/multimodal/` 104 文件 + `multimodal_gen/` 1472 文件 | `clip.c` + `image.c`（LLaVA，多图） |
| 分布式 | `distributed/` 30、`disaggregation/` 40、`elastic_ep/`、`eplb/` | **0**（`tensor_parallel` 零命中） |
| 平台 | cuda / rocm / npu / xpu / mps / musa / cpu / sail + rust_server | x86-64 CPU、CUDA sm_75+、arm64 CPU（macOS）、MLX(Metal) |
| 测试 | nightly 大模型回归 + `bench_serving` | **51 个 ctest** + CI **9 目标全绿**（linux x64/arm64 ±CUDA、macos(-mlx)、windows ±CUDA；run 41900893470）；runner 无 GPU，CUDA 目标只构建 + smoke |

---

## 1. 结论

和 10-29 那版比，四块硬骨头里的 **GPU KV 语义（G20）** 也已补掉；现在真正卡住「替代原版跑生产负载」的是：

1. **并行与分布式为零**：TP / PP / EP / NCCL / PD 分离全无。70B 级单卡放不下就加载不了，多卡机器只能用 1 张。
   `--dp N` 是单进程内 N 份 KV 池 + radix 的负载分摊，不是真正的 DP attention。
   上游这 10 天在把这块继续拉远（Engram all-to-all、DCP 上下文并行、KVShard PD、decode host receive）。
2. **模型覆盖面差距在扩大**：上游 10 天 +6 个新模型（Kimi-K3.1、Xing4.0、Beam、Flint、Nemotron-3-Embed、
   Unlimited-OCR v2），且 DeepSeek-V4/V4.1 成为主线（Engram 记忆表、MegaAttention、DSpark 投机全平台）。
   本仓 3 个后端 + MLX；**ds41（DeepSeek-V4.1）已开工（树上未提交的 WIP：`include/sgl/ds41.h`、
   `src/ds41_iq_tables.h`、`src/mlx_ds41*.cpp`、`tests/mlx_ds41_test.c`、`tools/dsv41_ref/`、`docs/PLAN_dsv41.md`）**。
3. **服务面**：CPU ≈46 / 加速侧 ≈46 个实现端点，501 表 33+31 个（明示原因，不假装成功）。
   上游静态路由 79（装饰器 87），本版新增 `/get_hicache_l3_cache_layout`（hicache 组，本仓本就 501）。
   与上游语义的对拍固化成 `tests/conformance_upstream.py`（双引擎）+ `tools/hf_ref_check.py`（vs HF transformers 4/4）。
4. **没有和原版（Python sglang / vLLM）的性能对照基准**：`docs/BENCH.md` 全部是自己 vs 自己的 A/B。
   生成正确性已有第三方对照（HF transformers，4/4），性能 A/B 在本机不可行（原版 Linux-only 依赖栈），
   脚本已引擎无关，待 Linux+CUDA 环境直接补。
5. **权重格式只差冷门分支**：W8A8、FP8 blockwise、NVFP4 落盘、marlin W4A8（激活 8bit）没做；
   主流落盘（AWQ / GPTQ v1+v2 / CT-W4A16 / MXFP4 / GGUF / FP8 / marlin ABI 4/8bit）都能打开。

主干推理闭环（模型前向 → paged KV → radix 前缀复用 → continuous batching → OpenAI 兼容 HTTP）在 CPU 上完整；
GPU 上是「llama 系 dense + GDN hybrid + MoE + DeepSeek MLA/MoE + 页级 KV + MTP 混批投机」。

---

## 2. 已补齐（不再算差距）

### 2.1 2026-10-27 → 10-29 批次（v0.5.0 收录，摘要）

GPU qwen3_5 hybrid + GDN 门控修正、MTP 接入 GPU 服务、GPU 并发 8→32、int4 权重、量化分页 KV、
`cuda_deepseek.cu`（MLA+MoE GPU 前向）、CPU llama 系 MoE、GGUF（v2/v3 12 种量化）、AWQ/GPTQ、
分片 safetensors、FP8、Jinja、约束解码、采样全量、chunked prefill、抢占、BERT embedding/rerank、YaRN、
reasoning/tool_calls、CPU 服务面 40+（tokenize/messages/responses/Ollama/…）、优先级队列、
Unigram tokenizer、运行时 LoRA 增删、33 端点明确 501、GPU 服务面 50+、GPU BERT、
DeepSeek GPU 服务路径、GPU MoE 层前向、多图 LLaVA。验证见各 ctest 与 `tests/ext_endpoints_test.py`。

### 2.2 2026-10-29 → 11-05 批次（本次新增，9 项）

| 能力 | 提交 / 位置 | 验证 |
| --- | --- | --- |
| **G20 页级 KV**：kvq 恒等页表 + 页区间分配器 + 零拷贝挂页（COW 统计）+ 动态池准入；`page_sz>0` 即 pagemap | `accel_http.c`（`pagemap` `pg0/npg` `g_pgm_*`） | `/health` 暴露 map/cow/alloc/fail 计数；G20 验收卡 |
| **G21 量化 KV 数值边界**：字节级 KV dump 定位并文档化 | `docs`（G21 卡） | quantized-KV 数值边界有据可查 |
| **G22a GPU 服务 parity**：stop strings、流式 usage 帧；conformance 套件跑双引擎 | `accel_http.c` | `tests/conformance_upstream.py` 双引擎全绿，顺带抓出 3 个真 bug 修掉 |
| **G22b 混批 MTP 投机**：并发请求下逐轮轮转一个 speculator（hlast 按行快照，ABI 4） | `accel.h` `sgl_cuda_mtp_hlast_save/restore` | 混批下投机不中断；顺带修单文件 safetensors 上下文 bug |
| **投机采样 temperature>0**（CPU+GPU）：产出分布与非投机严格相等 | `spec.c` `cuda_spec.c` | 数学等价推导 + 实测 |
| **与 HF 参考实现端到端对拍**：qwen3-06 贪心 24 token，CPU/GPU 双栈 vs HF transformers (CPU fp32) | `tools/hf_ref_check.py` | **4/4 提示词逐字一致**（此前从未与任何参考实现对过生成结果） |
| **MLX(Metal) 后端**：macOS 侧数值参考链 + 多分片 `.index.json` | `src/mlx_*.cpp` | CI `macos-arm64-mlx` 目标；7B-4bit 接受 |
| **v0.5.0 发版 + CI 九目标全绿**：Windows/arm64/CUDA 全进矩阵；五处 CI 根因（workflow 缺夹具、reload UAF、arm64 UB、nvcc goto-over-init、cp1252） | `.github/workflows/build.yml` | run [41900893470](https://github.com/integemjack/sglang.c/actions/runs/41900893470) |
| **发版盯 CI 工具**：`tools/ci_watch.ps1` 轮询 run 到完成并列各 job 结论 | `tools/` | v0.5.0 CI 全程盯完 |

---

## 3. 仍缺的（按「挡住什么」分组）

### A. 模型与权重 —— 挡住「能不能跑」

**A1 权重格式（已补齐主流，剩冷门分支）**

已支持：safetensors（fp16/bf16/fp32/fp8）、GGUF（12 种量化）、AWQ、GPTQ（v1+v2、desc_act/g_idx）、
compressed-tensors W4A16/W8A16、MXFP4（gpt-oss）、marlin ABI（4/8bit）、自研 int8/W8、GPU int4。
仍缺：W8A8、FP8 blockwise、NVFP4 **落盘**、marlin W4A8（激活侧 8bit）；HF Hub 直连下载与缓存；
CPU 侧权重统一 f32 驻留（大模型内存约 2 倍）。

**A2 GPU 模型覆盖**

MoE 层前向（Qwen3-MoE / Qwen3-Next / Mixtral）与 DeepSeek MLA+MoE 的 GPU 前向/服务路径都已落地。
仍缺：TP/PP/EP（见结论 1）、多模态 GPU 侧、CUDA graph 覆盖不到 MoE 分支。
**上游 10 天主线 = DeepSeek-V4/V4.1**（Engram MXFP4 表 + all-to-all 查找、MegaAttention、indexer TP 分片、
DSpark 投机 NV/AMD/NPU 全铺、deepseek-recipe 提示词渲染进 Rust 栈）——本仓 **ds41 WIP 已开工**。

**A3 架构覆盖面（P1）**

上游 259 个模型实现 vs 3 个后端（+MLX）。10 天内上游新增：Kimi-K3.1、Xing4.0、Beam、Flint、
Nemotron-3-Embed、Unlimited-OCR v2（16 文件支撑子目录）、MiniMax-M3.1-Flash。
缺口里最常被点名的：Qwen3.5 全家（上游重点优化）、GLM-5/DSA、Kimi 系、VLM（Qwen-VL/InternVL/Kimi-VL）、
Llama-4、Mamba（非 GDN）、dLLM、whisper 语音、**diffusion/视频生成子系统**（上游已成体系：
MiniMax-H3 / SANA-Video / Wan-Animate-2 / HiDream / LTX / Qwen-Image / Cosmos3 / ComfyUI 节点 /
OpenAI 兼容视频任务）。

**A4 tokenizer 覆盖面**

byte-level BPE + WordPiece + GGUF 内嵌 + Unigram（SentencePiece）都有了。
仍缺：T5 `Precompiled`（NFKC）归一化、added tokens 完整语义、多模态 processor。

**A5 多模态深度**

LLaVA 单图/多图 OK。仍缺：视频/音频、Qwen-VL 式动态分辨率切块、多模态 processor、GPU 侧视觉。

### B. 服务与调度 —— 挡住「当服务用」

**B1 路由**

CPU 版实现 ≈46 个端点 + 33 个明确 501；加速侧（CUDA/MLX 共用 `accel_http.c`）≈46 + 31 个 501。
上游静态 79（装饰器 87）。仍属无后端、永久 501 的：`/hicache/*`（含上游新增 `/get_hicache_l3_cache_layout`）、
external corpus、`/v1/audio/transcriptions`、`/v1/classify` `/v1/decisions`、expert distribution、
`/pd_role_switch`、`/update_weights_*` 整组、`/vertex_generate`。缺 gRPC 与 Vertex。

**B2 GPU 的 KV / 缓存语义（G20 后剩的尾巴）**

页级 KV 已落地（零拷贝挂页 + COW + 动态池准入），但：并发仍硬顶 `MAX_CONC 32`
（`ACTIVE_CONC` 默认 8，`SGL_HTTP_CONC` 可调）；前缀表仍 `PFX_MAX 16` 个槽
（挂页是零拷贝映射，不是 CPU 侧那种全局 radix 树）；上下文越长每页区间越紧。
CPU 侧是完整的页级 radix + refcount + LRU + chunked prefill + 抢占。

**B3 排队与公平性**

优先级（HTTP `priority` 字段、等待队列插队、抢占先抢低优先级）CPU/GPU 都有且实测。
仍未做：prefill delayer、`min_free_slots_delayer`、请求分池（上游 `schedule_policy.py` 那套）。

**B4 KV 分层（P2 视场景）**

`hicache` / Mooncake / NIXL 全无。上游 mem_cache 这 10 天 +41 文件（统一 KV 容量计价与准入、
HiCache fast_file 本地后端、L3 存 MXFP8 KV 页、joint_lru 统一驱逐、Mooncake 会话标签/对象键）——拉远中。

### C. 解码与性能

| 项 | 上游 | C 版 |
| --- | --- | --- |
| spec | EAGLE/3、Medusa、MTP、ngram、**DSpark/DFLASH（全平台）**、SpecForge、DP draft…（70 文件） | 模型自带 MTP 头：**混批轮转投机**（G22b，W=4 时贪心 3.16 token/步、1.5×）+ 朴素 draft/verify（K≤8，含 temperature>0 严格等价采样；无蒸馏 draft 仍净减速） |
| 图/编译 | torch.compile + piecewise/full CUDA graph + megakernels/monokernels（Qwen3.5 绑定、EAGLE 入口）+ XPU full prefill graph | decode CUDA graph（B=1..conc）；无 prefill graph、无编译层、图不覆盖 MoE |
| batch-invariant ops | 有 | 无需专门实现；`tests/batch_invariant.py` 4/4 |
| 量化算子 | fp8 per-tensor/token/block、W4A16、Marlin、DeepGEMM、MXFP4/NVFP4、块状 int8、HALOps MXFP8（118 文件） | W8A16（CPU）、W8A8 dp4a/IMMA（CUDA）、int8 KV（CPU）、NVFP4 风格 int4（默认关）、K4V3 KV（默认关） |
| attention kernel | FlashInfer、FA3/FA4、FlashMLA、TRT-LLM、CUTLASS、MSA-v1 稀疏 | 自写 flash + 两段式长上下文 + cuBLAS |
| 与原版 A/B | — | **仍无**。`docs/BENCH.md` 全是自身 A/B；`DESIGN_1M_32C.md` 264 tok/s 是推算。正确性已有 HF 对照（4/4）；性能对照脚本已引擎无关，待 Linux+CUDA 机 |

### D. 工程与验证

- CI 9 目标全绿且含 CUDA 构建/打包，但 runner 无 GPU（CUDA 只构建 + smoke），大权重数值回归仍只在本地；
  上游有 nightly 大模型回归。
- 无 Adapter/插件体系（`plugins/` `kernel_plugins/` `rust_extensions/`）；上游 rust_server/rust_router 这 10 天成长快。
- 无 Python 包 / `sglang.lang` 前端 / Ray / `sgl-model-gateway`。
- 平台广度：无 ROCm/NPU/XPU/MUSA/SAIL/Intel AMX（差异化定位是「零依赖单二进制 + 自写 kernel」，可不追）。

### E. 明确不追（写清理由，避免被当成遗漏）

- 10 个硬件平台的广度：C 版价值在可控内存与零依赖，不在平台数量。
- Ray / gateway / 多实例缓存感知路由：需要单独的服务生态。
- 训练侧与 RL 训练框架本身（但 `update_weights_*` 组若想进 RL 栈，属 B1 必补）。
- `sglang.lang` 前端语言（gen/select/fork/join）。
- diffusion/视频生成子系统：与「文本推理引擎的 C 重写」定位不同，不排期（上游动向记录在 §3-A3）。

---

## 4. C 版有、上游没有的（差异的另一半）

- **单可执行文件**：`sglc` 一个二进制含 CPU 引擎 + CUDA 引擎 + HTTP 服务（`--device auto` 自动选），
  零 Python、零运行时依赖；上游要 pip 一整套 torch。
- **CPU 路径是完整实现**：页级 radix + int8 KV + 连续批处理 + chunked prefill + 抢占 + W8A16，
  能在无 GPU 机器上真跑（上游 CPU 后端能力有限）。
- **直接读 GGUF**：容器 + 12 种量化 + 内嵌 tokenizer + 从元数据合成 config，一个 `.gguf` 文件就能起服务。
- **自写 kernel**：不依赖 FlashInfer / CUTLASS / TRT-LLM，Windows + MSVC 原生构建（上游基本 Linux/CUDA only）。
- **MLX(Metal) 后端**（macOS）与 arm64 CPU 构建都在 CI 里。
- **发版即打包**：tag 推上去 CI 出 8 个平台包并挂 GitHub Release。

---

## 5. 建议推进顺序（按 投入/解锁能力 比）

| 序 | 事项 | 为什么排这里 | 量级 |
| --- | --- | --- | --- |
| 1 | **ds41（DeepSeek-V4.1）收尾落地**（WIP 已在树上） | 上游 10 天的主线就是 V4/V4.1；方向已开工，沉没成本最低 | M |
| 2 | **TP（≥2 卡）+ NCCL** | 70B 级模型的唯一出路；挡住的面最大，成本也最高 | L |
| 3 | **GPU 并发去 32 顶**（全局页级前缀树 / PFX_MAX 扩容） | G20 已把页表铺好，剩下的是数据结构与策略 | M |
| 4 | Linux+CUDA 机上的**原版性能 A/B** | `hf_ref_check` / `conformance` 脚本已引擎无关，只差环境 | S |
| 5 | Qwen3.5 家族 / GLM DSA 等新热点架构 | 上游重点投入的模型族，C 版架构面要跟上 | M |
| 6 | CUDA graph 覆盖 MoE / prefill graph | decode 提速的下一档 | M |

---

## 6. 本文每个数字怎么核的

```powershell
# 上游（对照方）——在 works/sglang 根目录（HEAD 169c88be，git log -1 可验）
$py = (Get-ChildItem python -Recurse -File -Filter *.py | Get-Content | Measure-Object -Line).Lines   # 1687095
$srt = (Get-ChildItem python/sglang/srt -Recurse -File -Filter *.py | Get-Content | Measure-Object -Line).Lines  # 869115
(Get-ChildItem python/sglang/srt/models -File -Filter *.py).Count              # 259
(git grep -o EntryClass -- python/sglang/srt/models | Measure-Object).Count    # 233
foreach($d in 'mem_cache','layers','layers/quantization','distributed',
              'disaggregation','multimodal','speculative') {
  (Get-ChildItem "python/sglang/srt/$d" -Recurse -File).Count }
# mem_cache=229 layers=866 quantization=118 distributed=30 disaggregation=40 multimodal=104 speculative=70
$txt = Get-Content python/sglang/srt/entrypoints/http_server.py -Raw
$ms = [regex]::Matches($txt, '(?s)@app\.(?:get|post|api_route)\([^\)]{0,200}?"(/[A-Za-z0-9_/\.\-]+)"')
($ms | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique).Count      # 79（装饰器 87：含变量路径/循环注册）
git rev-list --count ad950cdc..origin/main                                     # 802（10-25..11-05）

# C 版——在 sglang-c
$f = Get-ChildItem src,include,apps -Recurse -Include *.c,*.h,*.cu,*.cpp -File
$f.Count; ($f | Get-Content | Measure-Object -Line).Lines                      # 77 / 36453（cpp: 3/3311）
ctest --test-dir build -N                                                      # Total Tests: 51
# 端点数：path 字符串唯一值 − 501 表行数（表 = 「{"/…", 原因}」行）
(Select-String -Path src/http.c -Pattern '"/[a-zA-Z0-9_/\.\-]+"' -AllMatches |
  ForEach-Object { $_.Matches.Value } | Sort-Object -Unique).Count             # 79
(Select-String -Path src/http.c -Pattern '^\s*\{"/' | Measure-Object).Count     # 33 → 实现 ≈46
(Select-String -Path apps/accel_http.c -Pattern '"/[a-zA-Z0-9_/\.\-]+"' -AllMatches |
  ForEach-Object { $_.Matches.Value } | Sort-Object -Unique).Count             # 77
(Select-String -Path apps/accel_http.c -Pattern '^\s*\{"/' | Measure-Object).Count  # 31 → 实现 ≈46

# 关键缺口（全部应输出 0）
foreach($k in 'tensor_parallel','hicache','mooncake','nixl','flashinfer','cutlass') {
  (Select-String -Path src/*.c,src/*.cu,src/*.h,apps/*.c -Pattern $k -SimpleMatch | Measure-Object).Count }
```
