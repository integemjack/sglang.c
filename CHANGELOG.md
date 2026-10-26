# Changelog

本项目所有重要变更。里程碑详情见 [ROADMAP.md](ROADMAP.md)；性能快照见 [docs/BENCH.md](docs/BENCH.md)。

## [v0.3.0] — 2026-10-27 「缺口关闭 G1–G10」

本批处理 TODO_PROMPTS.md v0.2 列出的缺口。测量口径与全部数字见
[docs/NOTES_gaps.md](docs/NOTES_gaps.md)（重要：本机空闲显存常小于模型常驻量，
WDDM 溢出会把绝对时间放大 3–20x，所以性能结论一律用**同进程 A/B**）。

### 性能
- **G1 flash prefill**：新增单遍分块注意力核 `k_attn_flash`（Q 块驻 shared、K/V 流式、
  online softmax），长上下文路径优先走它，`SGL_ATTN_TWOPASS=1` 可回退 A/B。
  8192-token prefill **72.4s → 8.7s（8.3x）**，2048 为 1.7x、512 为 1.5x；
  `[long]` 各阈值不动（two-pass vs shared 2.19e-05）。
- **G2 W8 大 M 走 tensor core**：`k_quant_rows` + `cublasGemmEx` int8 IMMA +
  `k_w8_dequant`（M>1 用 IMMA，M=1 保留 dp4a：实测 0.062 vs 0.068ms）。实测
  [2048,2048] M=8：**0.925ms → 0.069ms（13.4x）**；整数点积与顺序无关，
  IMMA vs dp4a prefill **maxerr = 0**，10/10 greedy 与 fp32 全同（`cuda_test [w8]`）。
- **G3 decode graph 稳定并默认开启**：捕获前先跑一次 plain step 预热（lazy 内核加载与
  cuBLAS 选算法不能发生在 capture 里），失败只标记**这一个 B**（旧版一个进程级开关
  让所有 batch 永久走 plain）；新增 `sgl_cuda_prewarm()` 起服务前把 ws/镜像撑到位并
  预捕获 B=1..conc（实测 8/8，首请求从 ~20s 降到与后续同量级）。fp16 decode A/B：
  B=1 23.0→9.7ms、B=4 36.6→8.9ms、B=8 38.5→11.3ms（3–4x）。
- **G6 GPU 前缀缓存**：`sgl_cuda_copy_kv()` + cuda_http 段租让表（最长公共前缀按 token
  精确比对、LRU、段复用时释放租让）。`[prefix]` 复用 vs 重算 maxerr 1.43e-05、
  greedy 6/6 全同、prefill 50.4 vs 127.1ms；端到端 756-token 共享 system prompt
  0.76s → **0.34s（2.2x）**且输出逐字一致。
- **G7 GPU 投机解码**：新增 `sgl_cuda_forward_seg_rows`（lm_head 一张 M=n GEMM 出全部
  位置 logits）+ `src/cuda_spec.c` 的 `sgl_cuda_spec_generate`（draft/target 各占 KV 段，
  接受逻辑与 CPU spec.c v2 逐 token 等价）。**正确性全过**：`[spec]` draft==target 接受率
  100%、真实组合（tinyllama+llama-160m）13.3%，两者输出均 == 非 spec greedy 逐 token 全同。
  **性能为净减速**（K=1 0.44x / K=4 0.27x）：未蒸馏同源 draft 接受率上限 ~13–15%（每轮仅产
  ~1.6 tok），且 verify/catchup 的 plain 路径在 WDDM 桌面 GPU 下不能图化。正收益需接受率
  ≥50%（同源蒸馏 draft）或 verify 固定 n 图化 + 批量 spec——已如实记入 BENCH.md / NOTES_t7.md。

### 健壮性
- **G4 HTTP 无 body 请求间歇卡死修复**：根因不是 CUDA，是 `srv->lock` 一把锁同时被
  worker（每步计算持锁）和 acceptor（登记连接）争用，高负载下 acceptor 饿死——
  连接被内核握手却长时间不建线程。`active_conns` 拆出独立 `conns_lock`。复验：
  12 并发 x256 token 负载下 600/600 次 GET /health 无超时（修前同形状 20/300 = 6.7%
  在 10s 内无首字节）。
- **G5 显存预检 + 降级服务**：`sgl_cuda_estimate_vram_mb()` / `sgl_cuda_avail_vram_mb()`，
  上传前算账；不够时不再 `return 2` 死进程，而是照常起服务：`/health` 报
  `degraded` + 原因，`/generate` 直接 503 + 可执行建议（降 ctx 或开 fp16/W8）。

### 广度
- **G8 真实多模态**：`openai/clip-vit-base-patch32` 真实权重（pytorch_model.bin →
  safetensors，抽查 4 张量 maxdiff 0.0），C 视觉塔按真实 CLIP 语义修正
  （quick_gelu、post-LN 只作用 pooled、输出 last_hidden_state），**分层 vs torch fp32
  maxerr ≤ 1.24e-05**；新增 stb_image PNG/JPEG 解码接进 CPU HTTP 的
  `image_url` base64 路径（PNG 解码与 PIL 0 像素级差异，端到端 200 且输出随图变）。
- **G9 模型族补完**：Gemma `rope_scaling`（线性/dynamic NTK）+ `hidden_softcap`
  embedding tanh 软上限 + 每层 `layer_types` 滑窗；Qwen3-Next `layer_types` 混合层，
  linear_attention 按 HF 真实公式（Gated DeltaNet + 因果 depthwise conv1d + gated
  RMSNorm）实现，对照独立 float64 numpy 参考 **maxerr ≤ 8e-08**。
  ctest 从 20 套增至 23 套。

### 工具
`tests/cuda_perf.c`（同进程 A/B 基准）、`tests/conc_test.ps1`、`tests/prefix_e2e.ps1`、
`tests/soak_curve.ps1`（RSS/VRAM 斜率）、`tests/g4_probe.py`、`tests/bom_fix.ps1`
（批量补 UTF-8 BOM）、`tests/build_all2.ps1`。

### 已知限制（本批未解决）
- 前缀缓存是**段级**租让（一条序列一个槽），不是 CPU 侧那种 radix 树；跳段部分共享、
  显存压力下的主动回收、带 visual token 的请求未覆盖。
- 真实 Qwen3-Next 的 full_attention 层还有 q 门控 / partial rotary / (1+w) norm / MoE
  未实现（加载真实权重会报缺张量，不会静默算错）；linear 状态按序列首 KV 页归属，
  radix 前缀共享场景会互踩（见 docs/NOTES_g9.md §4）。
- 真实 llava-1.5 的 projector 是两层 MLP + CLIP-336/1024→4096，本仓库仍是单层 linear；
  `<image>` 只作占位，视觉 token 接在 prompt 末尾。
- **G10 AVX2 自动向量化（条件任务，条件不成立→关闭）**：winget 无 GCC≥16.3（最高即已装的
  16.2.0-14.0.0-r3）。整库 `-march=native` 实测 25/25 轮有崩溃、99/500 实例失败（gdb 定位：
  GCC AVX 下 ymm 栈溢出按 32B 对齐、而进程入口栈仅 16B，间歇性随环境块长度变化）；同配置
  x86-64-v2 对照 20/20 全绿。`-mstackrealign` 旁路不完整（test_w8/test_family_qwen3 仍
  0xc0000409），不采纳。维持 `-march=x86-64-v2`，实测记录见 ROADMAP.md。主树合并态另跑
  10 轮×23/23 全绿复验（0 SegFault / 0 fastfail）。

## [Unreleased] — 2026-10-27 「t 系列收尾」

本批完成 TODO_PROMPTS.md 的 13 项任务（t0–t13），全程 numpy 独立参考对照。

### 新增
- **CUDA FP16 权重**（t1，`SGL_CUDA_FP16=1`）：tinyllama-1.1B fp16（~2.2GB）装入 6GB 卡
  39.4 tok/s；qwen05 批处理 627 tok/s 聚合（1.86x）。翻转可解释判定 + gemm-bench。
- **GPT-2 CUDA 后端**（t4）：Conv1D 转置上传、wpe、带 bias LayerNorm、GELU；
  fp32 maxerr 4.6e-5，MinGW CPU == MSVC GPU fp32 == GPU fp16 三方逐 token 一致。
- **CUDA int8 权重**（t8，`SGL_CUDA_W8=1`）：W8A8 dp4a（sm_75 原生）；
  显存最优档（free VRAM 3947MB），16 步 greedy 与 fp32 全同（gap ±0.1 内无翻转）。
- **模型族**（t9）：Qwen3（head_dim 128 + QK-Norm，真实 0.6B 权重）、
  Mistral（sliding window 带状掩码）、Gemma（√E embedding 归一化）——
  全部 numpy 全等（含 >window 长序列 case），ctest 注册 test_family_*。
- **LLaVA 多模态路径**（t10）：CLIP ViT 编码器（conv patch embed + transformer +
  projection）+ mm_projector + embedding 级融合前向入口 `sgl_llama_forward_emb`；
  numpy 全等 9.3e-9，融合入口 vs token 前向逐位一致（maxerr=0）。
- **CPU OpenBLAS**（t12，`SGL_OPENBLAS=ON` + `third_party/openblas`）：运行时
  LoadLibrary 分派（M≥2 且 ≥4M MAC），大 M GEMM 34→11ms（195 GFLOPs，3.1x）；
  AVX2 手写 intrinsics dot（运行时 `__builtin_cpu_supports` 检测，绕开 GCC 16.2
  自动向量化崩溃 bug）；`SGL_BLAS=0` 一键回退内置。
- **多 LoRA 热切换**（t6，plan B 实例池）：运行时 add/remove，N 适配器 N 选 1，
  合并语义 numpy 全等（test_lora_multi）。
- **spec v2 收尾验收**（t7）：160m draft 接受率 13%，spec==greedy 数学保证保持；
  CPU 上净减速的诚实结论（bottleneck=draft 前向，收益场景在 GPU）。

### 修复
- `sgl_llama_decode_batch` 的 RoPE 只旋转第 1 个头（H/HK 个头全错位）——真 1.1B
  权重下 CPU 调度器输出与 numpy 分叉；已修 + test_llama 加 decode_batch==forward
  断言防回归（曾因 0.5B 随机权重鲁棒性假绿）。
- spec v2 prefill 堆溢出（n>1 时 forward_n 需 n 行 logits 缓冲，借临时缓冲）。
- KV scatter 39 万次逐 token D2D copy → 单次 cudaMemcpy2DAsync：GPU 8K prefill
  174→66s。
- cuda_http：ensure_io/reserve 重分配后捕获图悬空指针（buffer realloc 时全部
  invalidate）；accept 瞬时错误杀死 acceptor；respond 分段 send 的
  WSAEWOULDBLOCK 丢包（send_all 循环重发）；SSE 快速完成丢增量（DONE 时补发）。
- http.c（CPU）：无 TCP_NODELAY 时每响应固定卡 ~21-24s（Nagle×延迟 ACK，2×RTO）。
- 200KB 单字符 prompt 挂死（BPE O(n²)）：两栈 64KB 预检 + 503 明确报错。

### 验收（soak+fuzz，t11）
GPU 栈 20 轮 soak drift -0.8%、fuzz 8/8、断连/并发同请求/ctx 溢出 finish_reason=length
/全程存活；CPU 栈 fuzz 7/8 + 断连（环境相关项以 -SkipEnvDependent 隔离，详见
docs/NOTES_t11.md——本机 winsock 对无 body 请求的间歇卡与代码无关，监控用 POST /health）。

### 已知限制
- 8K+ prefill 在 GPU 上 66s（k_attn_long 两遍 kernel，chunk 1024）——数值正确
  （vs CPU maxerr 1.6e-4），性能待闪存式重写。
- cuda_http 的 decode graph 默认关闭（动态 batch 下捕获图不稳定，env 可开）。
- WinLibs GCC 16.2 的 AVX2 **自动向量化**在本机间歇崩溃（x86-64-v2 + 手写
  intrinsics 稳定，见 CMakeLists 注释）；winget 尚无 16.3。

## [v0.1.0] — 2026-02

M0–M26：基础库 → 算子 → tokenizer → 模型 → 采样 → 调度器 → HTTP 服务全链路；
CUDA 后端（自适应 GEMM/长上下文/CUDA Graph）、W8A16/KV8、beam search、
RadixAttention、LoRA、spec v1。详见 ROADMAP.md。
