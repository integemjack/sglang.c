# Changelog

本项目所有重要变更。里程碑详情见 [ROADMAP.md](ROADMAP.md)；性能快照见 [docs/BENCH.md](docs/BENCH.md)。

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
