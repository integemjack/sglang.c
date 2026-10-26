# NOTES_t8 — CUDA int8 权重（W8A8 dp4a）

日期：2026-10-27 ｜ RTX 2060 6GB（sm_75，原生 dp4a）｜ qwen2.5-0.5B

## 实现

- `SGL_CUDA_W8=1`：7 类隐藏投影（q/k/v/o/gate/up/down）上传时 per-output-channel
  int8 量化（absmax/127，scale[N] 向量另存）；**fp32 原件不再上传**（显存省的来源）；
  wte/lm_head 走 fp16（t1 路径），norm/bias fp32。
- kernel `k_gemm_w8a8`：激活 per-row absmax 动态量化 int8 → `__dp4a` 4 路整数 MAC
  （Turing 原生）→ ×a_scale×w_scale[n] + bias。全 M 走 dp4a（prefill 含）。
- 分派宏 `PROJ`：`use_w8 && W8 存在` → dp4a；否则原 gemm_nt_any（fp32/fp16 路径
  完全不变）。gpt2 后端无 W8（llama 系专用）。
- lm_head 不量化（精度敏感）。

## 验收（全过）

- **top1 16 步与 fp32 全同**（含 step 11 gap=0.0435、step 13 gap=0.0207 的极小
  间隙均无翻转）；gap 数值偏差 ±0.1 以内 = 量化噪声，远小于最小决策间隙。
- 输出文本与 fp32 逐 token 一致（"Paris. It is the largest city..."）。
- 吞吐（qwen05 单流 greedy 40 tok best of 2）：fp32 33.5 / fp16 33.7 / **int8 33.8
  tok/s**——decode 是带宽+launch 混合瓶颈，三档持平（dp4a kernel 未做
  vectorized ld.128 优化，留作后续）。
- 显存（cuda_cli 上传后 free VRAM）：fp32 剩 216 / fp16 剩 3611 / **int8 剩
  3947 MB**（三档最优：投影 0.55GB + wte fp16 0.27GB ≈ 0.82GB 总权重）。
- **价值结论**：6GB 卡上 int8 的意义是显存（大模型装入/更大 ctx），不是速度。

## 铁律遵守情况

只改 src/cuda_backend.cu（struct/kernel/上传/分派/free）、docs/BENCH.md；
W8 默认关闭（env 显式开启），fp32/fp16 行为 bit-identical（cuda_test 无 W8 全绿
验证）；W8 数值验收独立于 cuda_test 的严格断言（该断言设计给 bit-exact 模式）。

## 补记 (2026-10-27)：cublas IMMA vs dp4a 对比（t8 原文「先 bench 两者」）

`sgl_cuda_gemm_bench` 新增 int8 段（RTX 2060 实测，best of 20）：

- **M=1（decode GEMV）**：两者持平——qkv/ow [896,896] dp4a 0.178ms vs IMMA 0.251ms（dp4a 快 1.41x）；lm_head [151936,896] dp4a 10.96 vs IMMA 12.02（dp4a 快 1.10x）。**选 dp4a 正确**。
- **M>1**：IMMA 全面胜出（M=8 快 6-16x，M=64 快 30-100x）——手写 dp4a kernel 的 dim3(N,M) 网格每块独立重复量化同一行激活，大 M 严重低效。W8 的 prefill（大 M）受此影响，但 W8 主收益是显存（decode 路径），验收已过；**后续优化**：PROJ 宏 M>1 分支切 cublasGemmEx IMMA（需预量化激活到 int8 缓冲）。
