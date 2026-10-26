# sglang-c 剩余任务 · 续作提示词

> 用法：新会话里先粘贴「公共上下文」块，再粘贴某条任务提示词。
> 按价值排序；任务间无强依赖，可任意顺序（标注除外）。

---

## 公共上下文（每条任务前必粘贴）

```
继续开发 sglang-c：把 sglang（LLM 推理框架）用纯 C 重写的项目，位于
C:/Users/ultra/Desktop/works/sglang/sglang-c/（约 5800 行 C+CUDA，M0–M25 已完成）。
现状：
- CPU 栈（MinGW GCC 16.2 / Ninja / x86-64-v2）：三后端 gpt2 / llama(含 tinyllama、qwen2
  含 qkv-bias 与 QK-Norm) / 模型抽象 sgl_model；连续批处理调度器；HTTP（/health
  /generate /v1/completions /v1/chat/completions，含 ChatML、SSE 流式、stop、n）；
  RadixCache 分页 KV + LRU 驱逐；W8A16 权重量化；int8 KV；SSE2/SSE4.1 手写 SIMD；
  beam search；speculative 框架；LoRA 加载时合并。15 套 ctest 全绿。
- CUDA 栈（VS2022 BuildTools + nvcc 13.3，sm_75 = RTX 2060 4GB，build-cuda 目录，
  CMake 用自定义命令直驱 nvcc——MSBuild 的 CUDA 集成不稳勿用）：kernels 含
  rmsnorm/rope/attention(GQA+shared softmax)/silu；GEMM 自适应分派（M=1 手写 k_gemm_nt，
  M>1 cuBLAS，helper gemm_nt_cublas）；decode_batch（B≤32，KV 段隔离 forward_seg/decode_batch）；
  cuda_cli（单发）与 cuda_http（8 并发 HTTP 服务）。
- 实测：GPU 单流 45.7 tok/s、批处理 105-118 tok/s（qwen2.5-0.5B）；CPU 7.4 tok/s（tinyllama W8+KV8+SIMD）。
- 磁上模型：models/gpt2(124M f32)、models/tinyllama(1.1B bf16)、models/qwen05(0.5B bf16,
  qwen2)、models/tiny-llama(2层随机,数值测试)、models/llama-160m(draft)、models/test-lora*
  （LoRA 测试适配器，gen_lora.py 可再生）。
铁律（历史踩坑，务必遵守）：
1) sgl_xmalloc 是 _aligned_malloc——只能 sgl_xfree 释放，绝不 free()；反过来 _strdup/malloc
   的指针绝不 sgl_xfree。两处都犯过堆损坏。
2) CPU 编译 -march=x86-64-v2（本机 WinLibs GCC 16.2 的 AVX2 自动向量化有间歇崩溃 bug）。
3) 头文件带 UTF-8 BOM；MSVC 加 /utf-8；.cu 源文件注释纯 ASCII（GBK 误读会吞代码行）。
4) RoPE 位置必须按 token 计，行号不是位置（CPU/GPU 各犯过一次）；KV 多层布局 [L][ctx][KD]
   的读写都要层偏移 l*ctx*KD，分配要乘 L（两个互掩 bug）。
5) 每次改完必跑：CPU `ctest --test-dir sglang-c/build`（15 套）+ CUDA
   `cuda_test.exe models/qwen05`（三方数值 GPU/CPU/numpy + 批处理不变量）。
   数值验证一律用 numpy 独立参考（tests/ref_llama.py 模式），不信旧输出。
6) HuggingFace 直连可用（hf-mirror 反而常挂）；ModelScope 国内源备选。
构建命令：
- CPU：PATH 加 …WinLibs…/mingw64/bin 后 cmake --build sglang-c/build
- CUDA：PATH 加 CUDA/v13.3/bin 后 cmake --build sglang-c/build-cuda --config Release --target cuda_test cuda_cli cuda_http
- 一键：build_all.bat；性能快照：powershell -File bench_all.ps1
```

---

## 任务 1 · CUDA FP16 权重与计算（性能，首选，大）

```
为 sglang-c CUDA 后端加 FP16 路径（现在全 fp32，4GB 显存只能装 0.5B 模型）：
1) cuda_backend.cu：新增 half 权重上传（上传时 fp32→fp16），k_gemm_nt 加 __half2 版
   （fp16 乘加、fp32 累加），attention/RoPE 可保 fp32（精度敏感）；大 GEMM 走
   cublasHgemm（先 bench：Turing 卡 hgemm vs sgemm 实测多少倍再定分派阈值）。
2) 混合精度预算：权重/输入 fp16，累加与 softmax fp32（对标 sglang 的 fp16 推理）。
3) 目标：TinyLlama-1.1B（fp16 后 ≈2.2GB）能整卡装入并推理；qwen05 吞吐对比 fp32。
验收：
- cuda_test：qwen05 fp16 vs fp32 GPU 的 greedy 逐 token 一致或差异可解释
  （打印 top1/top2 间隙，间隙 <1e-2 的分步翻转可接受）；
- tinyllama-1.1B 在 GPU 上跑通 'The meaning of life is' 续写（连贯文本），
  与 CPU(MinGW) greedy 逐 token 对照（fp16 容差内）；
- bench_all.ps1 新增 fp16 行，吞吐/显存数字写入 README。
注意：显存预算公式先算清（权重 + KV[24层][ctx][KD] + scratch），ctx 钳制逻辑沿用。
```

## 任务 2 · 长上下文：两段式（flash 式）attention（能力，中）

```
解除 CUDA 侧 ctx=4096 钳制（k_attn 的 scores 全驻 shared memory，48KB 上限）：
1) k_attn 改分块在线 softmax：KV 按 512/块迭代，维护 running max 与 running sum
   （标准 flash-attention 的 rescale 技巧），shared 只驻当前块；输出同样分块累积。
2) forward_seg/decode_batch 同步解除钳制；ctx 上限回到模型配置（qwen05=32K）。
3) prefill 顺带支持长 prompt 分块（n 大时 attention O(n²) 逐块走）。
验收：
- 构造 8K token prompt（重复文本即可），GPU prefill 输出 logits 与
  CPU（llama.c 本就支持任意 ctx）maxerr < 1e-3；
- 显存不随 ctx 超线性爆（打印峰值）；
- 原 4096 内路径回归不破坏（cuda_test 全绿）。
```

## 任务 3 · CUDA Graph（性能，中）

```
decode 循环每步 ~20 个 kernel launch，用 CUDA Graph 消除：
1) 形状固定的 decode 步（B、各 seq len 每步变）→ 把 pos/base/lens 放设备内存
   （kernel 从设备指针读），输入输出 buffer 固定地址，即可 cudaStreamBeginCapture
   捕获整个 decode_batch 的一次迭代，实例化后每步只 memcpy 参数 + cudaGraphLaunch。
2) 注意 cuBLAS 在 capture 中的行为（cublasSetStream 指向捕获流；workspace 预分配）。
验收：捕获路径 vs 非捕获路径同输入 logits 全等（bit 级或 1e-6）；
decode 吞吐提升实测（B=1 与 B=4 各报一个数字，目标 +15% 以上）；cuda_test 全绿。
```

## 任务 4 · GPT-2 的 CUDA 后端（对等性，小）

```
CUDA 后端目前只支持 llama 族；给 gpt2 补上（sgl_gpt2* 为源）：
1) gpt2 权重是 Conv1D 布局 [K,N]（llama 是 [N,K]）：上传时转置一次成 [N,K] 复用
   现有 gemm/attention 路径（gpt2 有绝对位置编码 wpe、LayerNorm 带 bias、GELU——
   上传 wpe，kernel 组合差异在 forward_seg 内分支处理，或写 gpt2 专用 forward）。
验收：cuda_cli 对 models/gpt2 的 greedy 输出与 MinGW sglc 逐 token 一致；
cuda_test 加 gpt2 用例（三方对照沿用现有断言风格）。
```

## 任务 5 · GPU HTTP 服务对等 + RadixCache（对等性，中）

```
cuda_http 目前 8 静态 KV 段、无前缀复用、流式未验证：
1) 验证/修复 SSE 流式（wait_done 已有流式分支雏形）——curl -N 流式拼接必须严格
   等于非流式响应（CPU 服务的 test_http 有同款断言可抄）。
2) 并发 >8 时排队而非拒绝；段满 finish_reason=length；空闲段回收复用。
3) 进阶（可选）：GPU 版前缀复用——prompt hash → KV 段缓存，重复 system prompt
   的 chat 请求跳过 prefill（段内 memcpyAsync 复制 KV 或记录段租约）。
验收：两轮 8 并发全过且服务存活（build-cuda/conc_test.ps1 现成）；
流式==非流式；排队请求最终完成。
```

## 任务 6 · 多 LoRA 热切换（功能，中）

```
LoRA 现为加载时合并（一份权重一个 adapter）。改运行时多适配器：
方案 A（显存允许时）：base fp16 + 每个 adapter 预乘 delta [N,K] 上传，gemm 前用
  cublasAxpy 临时叠加（请求级），或 forward 分支 W+delta 的 cublasGemmEx beta 技巧；
方案 B（省显存）：只在 CPU 栈做——model.c 支持按请求选择已合并的多份实例副本
  （调度器给每个 sgl_model 实例建池，请求带 lora 名路由）。
先估算显存：qwen05 fp16 base ≈0.5GB + 每 adapter delta（r=8×7投影≈18MB）→ 方案 A 可行。
验收：同一服务两个请求用不同 adapter，输出各自 == 单独加载该 adapter 的 CPU 参考
（tests/gen_lora.py 造 2 个不同 seed 的 adapter）。
```

## 任务 7 · Speculative Decoding v2（性能，中）

```
M18/M19 结论：框架正确但未训练 draft 接受率 13%、CPU 逐步验证无批收益 → 净减速。
v2 两件事：
1) 验证批 n-outer GEMM：CPU 侧 llama.c 加「一次前向吃 n 个 token 的分块 GEMM」
   （权重驻留复用，n 路激活按块算），使 verify 成本 ≈ 单步 ×1.2 而非 ×n；
2) 找真 draft：qwen2.5-0.5B 的官方小 draft 无——改用 llama-160m 已验（13%），
   或下载 Qwen2.5-0.5B 蒸馏版/自测 Qwen2-0.5B 当 target 时用 random-init 微调不可行——
   务实路径：target=llama-160m + draft=截层自草稿（160m 上测接受率），或直接
   在 160m/1.1B 组合上重测（llama-160m 是 Llama-2 词表 32000，tinyllama 同词表——
   M19 已配好 models/llama-160m）。
验收：报告接受率与端到端 tok/s（vs 非 spec 基线）；数学保证保持
（spec greedy == target-only greedy 逐 token，test_spec 现成断言）。
```

## 任务 8 · CUDA int8 权重（性能，中大）

```
GPU 侧 W8A16（CPU 已有 sgl_quant 基建）：
1) cublasGemmEx：A CUDA_R_8I × B CUDA_R_8I → C CUDA_R_32F（Turing int8 tensor core，
   理论 2x fp16）；权重加载时 per-channel 量化上传，scale 向量另存，输出乘 scale；
2) 或手写 dp4a kernel（__dp4a 指令，Turing 原生），先 bench 两者。
验收：qwen05 int8 vs fp32：top5 重叠 ≥4/5、greedy 允许低概率分步翻转（打印间隙）；
吞吐/显存数字 vs fp16/fp32 三列对比进 README。与任务 1 的 FP16 可组合（fp16 激活
+ int8 权重的混合可后续再议，先各自独立）。
```

## 任务 9 · 更多模型族：Mistral / Gemma / Qwen3（广度，中）

```
llama.c 加新族差异（各自 config 探测 + 权重命名适配）：
- Mistral：sliding window attention（window 4096 的 banded mask；CUDA k_attn 加
  window 参数跳过窗外块，CPU 侧 attention 循环加窗口判断）；
- Gemma：pre-feedforward/post 布局差异、normalizer sqrt(hidden) 缩放、
  RoPE 8k 基频 + 线性缩放（rope_scaling 配置）；embedding 软上限 tanh(30x)（可选）；
- Qwen3：QK-Norm 已支持、layer_types（sliding/full 混合层）。
每个族：下小模型（hf 直连）→ 照 tests/ref_llama.py 写 numpy 参考 → greedy 全等断言。
验收：各族 greedy6 == numpy；ctest 各加一 case；README 模型支持表更新。
```

## 任务 10 · 多模态：LLaVA 式视觉输入（广度，大）

```
给 /v1/chat/completions 加 image_url(base64 data URI)：
1) CLIP ViT-B/32 encoder 的 C 移植（patch embed conv + 12 层 transformer，
   ~150M 参数 safetensors，openai/clip-vit-base-patch32）；image → 49 patch tokens；
2) projector（llava 的 MLP）映射到 llama 词表维度，插入 prompt token 序列；
3) HTTP 解析 content 数组形式（[{type:text},{type:image_url}]）。
验收：encoder 每层输出 vs HF transformers numpy 参考 maxerr<1e-4（分层数值对照，
沿用 ref_llama.py 模式）；端到端：llava-1.5-7b 太大装不下——用
llava tiny 随机权重测试数值通路 + 真实小组合（qwen2-vl-2b 若可拆）人工检查连贯性。
范围提示：这是最大的单项任务，可拆 encoder→projector→HTTP 三步分轮做。
```

## 任务 11 · 健壮性硬化（质量，中）

```
1) 泄漏：循环 1000 请求（CPU sglc --serve 与 cuda_http 各一轮）记录 RSS/VRAM 曲线，
   修到平稳；工具写进 bench_all.ps1（--soak 模式）。
2) Fuzz：畸形 JSON、超长 prompt（>ctx）、max_tokens=10^9、非法 UTF-8、二进制 body、
   Connection 立即断开、并发同 id——全部不崩、有合理状态码。
3) 优雅降级：KV pool 满（CPU 调度器已会等待重试，验证）、GPU 显存不足
   （上传前 free_mem 预检 + 明确报错）、ctx 满时 finish_reason=length。
验收：soak 曲线平坦；fuzz 清单全过；两种服务的错误路径各有断言测试。
```

## 任务 12 · CPU 性能二阶段（性能，中）

```
1) OpenBLAS 集成：sgl_matmul_nt 加运行时分派（预编译 openblas.dll + CMake 可选
   SGL_OPENBLAS=ON），注意 OpenBLAS 是 [N,K] 列主 Fortran 习惯——CblasRowMajor
   直接支持；bench 决定是否默认开。
2) AVX2 重试：装新版 WinLibs/GCC（>=16.3）重跑当年 25 次崩溃复现
   （tests 里有历史：x86-64-v2 稳、avx2 间歇 0xC0000005）——修复则换
   -march=native 重 bench（预计 GEMM 再 2x）。
3) prefill n-outer 分块 GEMM（与任务 7.1 同一基建，先做这边可复用）。
验收：bench_all.ps1 前后对比写入 README；ctest 全绿（OpenBLAS 与内置两路都要过）。
```

## 任务 13 · 收尾发布（文档，小）

```
1) README 完整化：英文版或双语、全部端点+参数表、模型支持矩阵、性能表
   （CPU/GPU × fp32/fp16/int8 各行）、已知限制清单（对照各任务完成情况更新）。
2) CHANGELOG.md（M0–M25+ 时间线已在 ROADMAP.md，转正式格式）；
   LICENSE 核查（原 sglang Apache-2.0，注明派生关系）。
3) 从零验证：删除 build*/ 后 build_all.bat 一把过；新机器清单（工具链版本）写进
   README「环境要求」。
4) 最终全套回归 + bench_all.ps1 快照存档。
```

---

## 附注

- **张量并行**未列入：本机单卡 4GB，真 TP 无从验证；若未来双卡，任务 9 的 Mistral
  路径（MoE 的 expert 并行）是更自然的切入点。
- 顺序建议：1 → 2 → 3（性能主线）→ 4/5（对等性）→ 11（随时可插入）→ 其余按需。
  任务 7.1 与 12.3 共享「n-outer 分块 GEMM」基建，做哪个先都行、注明复用。
