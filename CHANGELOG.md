# Changelog

本项目所有重要变更。里程碑详情见 [ROADMAP.md](ROADMAP.md)；性能快照见 [docs/BENCH.md](docs/BENCH.md)。

## [Unreleased] — TODO_PROMPTS v0.4 的缺口 G11/G14/G15/G16/G17/G19

### 性能
- **G15 前缀缓存扩容**（`apps/cuda_http.c`）：租让登记的 **行数从「只有 prompt」改成
  「这个段里真有有效 KV 的行」**——非视觉请求登记 `n_prompt + (n_gen-1)`，于是多轮对话
  / 续写场景（下一个 prompt = 上轮 prompt + 生成结果 + 新后缀）能整段复用。实测
  552 → **573 行**（多出的就是生成 token 的 KV）。同 prompt 重发的端到端：
  0.262s → **0.103s（2.5x）**，命中 552/558 行（单轮 prefill 的 99%）。
- **G15 带图请求也登记**（只到 text 末）：视觉行依赖图片、位置在 text 之后，不归
  这段登记管，所以同文本不同图片的请求能复用 text 部分 —— 实测命中 **135 行**
  （以前带图请求直接跳过登记 ⇒ 0）。
- **G15 `/health` 统计**：`prefix.{slots,hits,misses,hit_rate,hit_rows,leased_rows,flushes}`
  （leased_rows 是 gauge：当前可复用的行数）。
- **G15 运维杆**：`SGL_PFX_MAX=0` 彻底关缓存（1..16 限表大小）；请求超过一段容量时
  主动 `pfx_flush_all()`，避免后续命中半截现场。
- **G15 验收**：`tests/prefix_e2e.ps1` 重写成四组（同前缀 / 续写 / 带图 / 显存紧张）
  **全部 PASS**，且每组都断言输出与「关缓存」逐字一致。
  至于任务里的「跳段共享两轮扫描」：扁平表里每条登记都从第 0 行开始，可复用行数的
  上界就是最长的一条 —— 拼接永远不会更长，所以没实现（实测收益为零），改去真正能多
  拿行数的「生成 token 行」；b) 在本服务的静态分段设计里退化为「超容量时清现场」，
  已在组 4 验（也被实测证明：租让不额外占显存）。

### 性能
- **G14 投机解码流水线化**（`src/cuda_spec.c`）：预填 `np-1` 行，把 prompt 末 token
  当“挂起 token”由下一轮 verify 第 0 行吞掉 —— 每轮 verify 固定 n=K+1 行，
  reject/bonus token 的 KV 不再需要单独一次 target 图步（旧结构每轮固定 11.9–12.8ms，
  占总时长 23–46%）。tinyllama+llama-160m（ctx=1024，new=32，best-of-3）：
  K=1 0.44x→**0.73x**（35.1→60.5 tok/s）、K=2 0.35x→**0.64x**、K=4 0.27x→**0.40x**。
  输出语义不变：`cuda_test [spec]` 两例（draft==target 100% 接受率、真实组合 13.3%）
  依旧逐 token 等于非 spec greedy。
  仍末达到 ≥1.0x 的门槛：按实测 t=11.4ms、d=3.35ms 算，回本需 K=1 接受率 >88%、
  K=2 >69.5%，而实测只有 39.1% / 28.6% —— 未蒸馏 draft 太弱，已如实记入 BENCH/NOTES。
- **G14 small-M GEMV**（`k_gemm_nt_mrow` / `k_gemm_nt_h_mrow`）：M=2..8 时每输出行
  一个 block，权重行只读一遍、同时累 2/4/8 个 query 行（MT 用模板实例化，动态 M
  会把累加器打回 local memory：M=2 实测 22.1→16.1ms）。verify 成本因此变成
  n=2 16.1ms / n=3 17.2 / n=4 18.5（旧值 22.1/24.0/25.8）；跨线程归约改 warp
  shuffle（单次同步），n=5..8 的额外惩罚从 33ms 降到 27.7ms。同样受益于批量
  decode B=2..8 与 2–8 token 的短 prefill。
- **G14 诊断开关**：`SGL_SPEC_PROF=1` 打每轮的 draft/verify/补齐耗时分解；
  `SGL_SMALLM_CUBLAS=1` 把 M=2..8 退回 cublas（A/B 用）。

### 性能
- **G16 flash 不再只服务长行：short 核阀值 4096 → 64**（`SGL_ATTN_FLASH_MIN`）。
  qwen05 int8 同进程 best-of-3：128 tok 1.3x、256 tok 1.7x、512 tok 3.7x、
  1024 tok 7.7x、2048 tok **11.8x**、4096 tok **20.2x**（short 核是 O(n²) 且每行
  重读 KV；16/32 tok 两者持平、64 tok short 快 8%，所以留着它做极短 prompt 的快路）。
  默认路径因此变快：fp32 prefill 512/2048 tok 207.8/2385.9 ms → 44.8/200.0 ms，
  int8 512/2048 tok 169.1/2201.0 ms → 48.6/184.5 ms。dispatch 从 prof/非 prof
  两份抄写收成单个 `attn_dispatch()`；未特化的 head_dim 行为不变（先试 flash，
  不成再按行长安回 short / 两遍）。
- **G16 W8 IMMA 后重测**：int8 prefill 512/2048/8192 = 48.6 / 184.5 / 1923.8 ms
  （10533 / 11100 / 4258 tok/s），fp16 52.2 / 198.4 / 1971.9 ms，fp32
  85.4 / 386.7 / 2748.9 ms（ctx=16384，rounds=2）。dp4a vs IMMA 同进程 A/B：
  n=128/256/512/1024 → 46.7x / 91.5x / 151.6x / 155.3x，**dp4a 只在 M=1 持平**
  （decode 仍然选它），所谓“拐点”不存在：M≥2 一路输且越输越多。
  fp32 同进程 A/B：two-pass vs flash = 8.6x / 19.3x / 33.3x（512/2048/8192）。

### 功能/健壮性
- **G16 flash 泛化**：head_dim 新增 **D=96** 特化（QPW=2/TK=32，shared 30.3KB）；
  加 `D % 32` 守卫——ND=D/32 要求整除，否则像 D=80 会静默只算前 64 维
  （实测：加守卫前 D=80 特化 maxerr 0.83、greedy DIFF；加守卫后回退，maxerr 7.15e-07）。
  D=96 随机模型（`..\models\d96-rand`，L=2 H=2 KVH=1 E=192）在 n=37/200/500 上
  flash vs two-pass ≤5.7e-07、GPU vs CPU ≤8.4e-07、greedy SAME。
- `tests/long_vcpu.c` 多了「同进程 flash vs two-pass」一项，配 `SGL_FORCE_LONG_ATTN=1`
  可在小 n 上验任意 head_dim；`SGL_ATTN_DEBUG=1` 会逐层打出实际用的核
  （`[attn] flash D=96 QPW=2 TK=32`），证明确实走特化而不是静默回退。
- `cuda_test [long]` 的三路对比修正：以前 4300 行的 "flash vs two-pass" 两边实际都是
  flash（阀值降到 64 后 FORCE_LONG 不再等于两遍），变成自己跟自己比；现在显式设
  `SGL_ATTN_TWOPASS=1`，并新增 short(32 tok) vs 两遍一项保证 short 核仍被覆盖。
  实测：short vs 两遍 1.08e-05、default(flash) vs 两遍 2.00e-05、flash vs 两遍
  @4300 1.75e-05，全 SAME；`cuda_test ..\models\qwen05` → `test_cuda: all OK`。
- `tests/gen_random_mistral.py` 支持 `--hidden/--heads/--kv-heads/--layers/--interp/
  --window/--vocab`，用来造任意 head_dim 的随机模型；新增 `tests/bench_prefill.ps1`
  一键重跑三精度 prefill + dp4a/IMMA + flash/两遍 A/B。


### 功能
- **G11 GPU 版收标准图片**：`cuda_http` 的 image_url 以前只认 raw float32 CHW，真实
  客户端发的 JPEG/PNG 一律 400。现在按 data URL 的 mime 分派：`image/{png,jpeg}` 走
  `sgl_clip_project_bytes`（stb_image 解码 + PIL 语义 resize/crop + CLIP + projector，
  与 CPU 版同一条代码路径），其余仍走 t10 的 raw 兼容分支。`src/image.c` 进 `sgl_cuda`。
  验收：`tests/http_image_e2e.py --url`（新增：指向已在跑的服务、不自起进程）对
  `cuda_http models/mm-test 30024` 发 jpeg / png / 纯文本 → 全 200，jpeg≠png、有图≠无图，
  坏 base64 → 400；同一套图 CPU 版输出逐 token 前缀完全相同；`cuda_test [mm]`
  GPU 融合 vs CPU 融合 maxerr **1.19e-07**、greedy SAME（无回归）。

### 健壮性
- **G17 GPU 版「/health 偶发卡死」根因修复**（和 CPU 版 G4 是两个不同的病）：
  `cuda_http.c` 的 `read_request` 在**未 NUL 截断**的请求缓冲上用 `strstr` 找
  `Content-Length:`，负载一大就会撞上前一个 POST 请求在堆里残留的
  `Content-Length: 105`，把 body 长度当成 105 后一直等永不会来的字节 —— 表现就是
  无 body 的 GET /health 挂到客户端超时断开为止（实测：12 并发 load 下 **100%
  timeout_read**，挂的时长恰好等于客户端超时；空闲时堆恰好为 0 才看不出来）。
  修复：每次 recv 后 `buf[len]=0`，Content-Length 只在本请求 header 段内做有界搜索
  （与 `src/http.c` 的 t5 修法一致）。复验：`g4_probe load 600 12 256` 对
  cuda_http（GPU）与 sglc（CPU）均 **ok=600 / timeout_read=0**；修复前同样负载为 0/600。
  另加 `SGL_HTTP_TRACE=1` 入口/收包/应答/退出逐行诊断（env 开关，默认静默），
  以及 `g4_probe.py` 的 `--host/--port`（以前写死 30000，没法指向 GPU 版）。
- **G17 soak 误报**：`soak_curve.ps1` 的 stderr 判定加了 INFO 白名单（模型加载时
  `safetensors.c` 会打一堆 `ERROR tensor ... not found` 的可选张量探测，以前会被
  当成异常）。复验：cuda_http + mm-test 跑 2 分钟 → RSS 斜率 0.020 MB/min（0.0044%/min）、
  VRAM 0.000 MB/min，PASS，且如实打出「无 ERROR/FATAL 行」。

### 基础设施
- **G19 CI 真跑模型族与多模态**：`gen_random_{mistral,gemma,qwen3next}.py` 加
  `--out/--tokenizer`（`gen_random_gemma.py` 另有 `--variant dynamic` 出 gemma2），
  CI 里生成 mistral/gemma/gemma2/qwen3next/llava 五套随机模型 + 跑 `ref_llava.py`，
  这样 CMake 的 「模型存在才注册 test」 条件都成立，family 测试与 test_llava 真跑起来；
  验收判据改成正向（`greedy6=一致` 行数 ≥ 4、`test_family_*` 全部 Passed、
  `test_llava: all OK`），不再只是「ctest 退出码 0」。本地核验：新参数生成的 6 个
  safetensors 与仓库现有 `models/*-rand` **逐字节相同**（保证与 `_ref_*.json` 对得上）。

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
