# sglang-c 剩余任务 · 续作提示词

> 用法：新会话里先粘贴「公共上下文」块，再粘贴某条任务提示词。
> 按价值排序；任务间无强依赖，可任意顺序（标注除外）。
> v0.2（2026-10-27）：原 t1–t13 已全部完成并验收，见文末索引；本文件现列出
> 验收时发现但未修的性能/健壮性/广度缺口。

---

## 公共上下文（每条任务前必粘贴）

```
继续开发 sglang-c：把 sglang（LLM 推理框架）用纯 C 重写的项目，位于
C:/Users/ultra/Desktop/works/sglang/sglang-c/（约 1.2 万行 C+CUDA，M0–M26 + t1–t13 已完成，tag v0.2.0）。
现状：
- CPU 栈（MinGW GCC 16.2 / Ninja / x86-64-v2）：gpt2 / llama族(gpt2、llama、qwen2、qwen3、
  mistral 滑窗、gemma 归一化) / sgl_model 抽象；连续批处理调度器；HTTP（CPU http.c + GPU
  cuda_http.c 双实现：/health /generate /v1/completions /v1/chat/completions，ChatML、SSE、
  排队、finish_reason 如实、多模态 image_url）；RadixCache 分页 KV + LRU；W8A16 权重 +
  int8 KV；SSE2/SSE4.1 手写 + AVX2 dot（运行时检测）+ OpenBLAS 运行时分派（大 M GEMM 3.1x）；
  beam search；spec v2（n-token 批量 verify，160m draft 接受率 13%，spec==greedy 保证）；
  多 LoRA 热切换（plan B 实例池）；CLIP ViT 编码器 + mm_projector + embedding 级融合前向。
  20 套 ctest 全绿（含 t9 三族随机结构 numpy 对照、t10 llava 随机结构对照）。
- CUDA 栈（VS2022 BuildTools + nvcc 13.3，sm_75 = RTX 2060 6GB，build-cuda 目录，
  CMake 自定义命令直驱 nvcc）：llama 族 + gpt2 双后端；三精度（fp32 / fp16 SGL_CUDA_FP16 /
  W8A8 dp4a SGL_CUDA_W8，显存最优档 free VRAM 3947MB）；CUDA Graph（cuda_cli 场景 B=1 +174%/
  B=4 +89%；cuda_http 场景默认关——动态 batch 不稳定）；两遍式长上下文 attention（ctx 16384，
  KV scatter 2D memcpy）；滑窗（4 kernel 全带 win 参数）；sgl_cuda_forward_seg_emb
  （视觉 embedding 直进层循环，GPU vs CPU 融合 1.2e-7）。
- 实测（docs/BENCH.md 终版快照）：qwen05 GPU 单流 fp32 33.5 / fp16 36.3 / int8 33.8 tok/s、
  批处理 540.8 tok/s（B=4）；tinyllama-1.1B fp16 39.4 tok/s 入 6GB 卡；CPU 4.1 tok/s（OpenBLAS
  大 M GEMM 195 GFLOPs）。
- 磁上模型：真实权重 gpt2(124M)/tinyllama(1.1B)/qwen05(0.5B qwen2)/qwen3-06(0.6B)/
  llama-160m(draft)/tiny-llama(2层随机)；随机结构 mistral-rand/gemma-rand/llava-rand
  （tests/gen_random_*.py 可再生，含 numpy 参考生成器）；mm-test=mistral-rand+llava-rand 组合。
- 构建铁律（历史踩坑，务必遵守）：
  1) sgl_xmalloc/_aligned_malloc 只能配 sgl_xfree；malloc/_strdup 只能配 free。混用即堆损坏。
  2) 所有 .c/.h/.cu 必须带 UTF-8 BOM——MSVC/nvcc 按 GBK 读无 BOM 的中文注释会吞结构体
     声明（报 has no member）。CMake configure 阶段有自检警告；edit_file 改完可能丢 BOM，
     重编前查第一字节 EF BB BF。
  3) CMake 自定义命令编的 cuda_backend.obj 不跟踪头文件依赖——改 include/sgl/*.h 后必须
     删 build-cuda/cuda_backend.obj 强制重编，否则旧偏移访问新 struct = cuda_http 启动即崩。
  4) CUDA graph 捕获绑定设备指针和 pinned host 地址：任何 buffer realloc（reserve/ensure_io）
     必须先 invalidate_graphs()（已有 helper）。
  5) 数值问题先写独立 numpy 参考（tests/ref_*.py 模式）裁决，不猜；随机小模型
     （gen_random_*.py）优先于下载大权重。
  6) edit_file 的 old_text 必须与文件原文一字不差（缩进、空格、换行）；先 read_file 看准再改。
```

---

## 任务 G1 · GPU 长上下文 prefill 提速（性能，中大）

```
现状：k_attn_long 两遍式（K 点积算两遍换固定 shared 内存），8192-token prefill 66.2s
（KV scatter 已优化过：174→66s，见 git log）。CPU 同长度 4096s，数值验收已过
（maxerr 1.56e-4 < 1e-3），纯性能问题。
1) 先 profile 定位：nsys 或手打每层计时（k_attn_long vs GEMM vs KV scatter 占比），
   别猜。怀疑点：q 加载重复、无 K/V 分块复用 L2、两遍扫描本身。
2) 块级 Flash Attention 重写 k_attn_long：Q 块驻 shared，K/V 块流式过 global，
   online softmax 已有（run_max/run_sum 框架在），缺的是 Q 分块复用与向量化
   （half2/float4 加载）。
3) 目标：8K prefill 66s → <10s（6.6x）；验收用 tests/long_vcpu.c models/qwen05 8192
   （maxerr < 1e-3 不回退）+ cuda_test [long] 段全绿。
```

## 任务 G2 · W8 大 M 切 cublas IMMA（性能，中）

```
现状：PROJ 宏所有 M 都走手写 dp4a kernel。gemm-bench 实测（NOTES_t8.md 补记）：
M=1 两者持平（dp4a 正确选择）；M=8 IMMA 快 6-16x、M=64 快 30-100x——dp4a 的
dim3(N,M) 网格每块独立重复量化同一行激活。
1) 写 k_quant_rows（激活 [M,K] → int8 + 每行 scale，一次 kernel）；
2) PROJ 宏 M>1 分支：量化 → cublasGemmEx(CUDA_R_8I×8I→32I, CUBLAS_COMPUTE_32I)
   → 反量化 kernel（Ci32 × scale_A[row]×scale_W[n] + bias）；
3) 验收：cuda_test SGL_CUDA_W8=1 全绿（fp32 对照容差已有）；bench：qwen05 W8 prefill
   （长 prompt）前后对比进 BENCH.md。注意 cublasGemmEx 8I 要求 K%4==0 且列主布局——
   gemm-bench 里已有可抄的正确调用姿势。
```

## 任务 G3 · cuda_http decode graph 稳定性（稳定/性能，中）

```
现状：cuda_http 默认 SGL_CUDA_GRAPH=0（t5 决定）。开启时动态 batch（B 逐请求变化 +
ensure_io/reserve realloc）偶发 sticky CUDA error，即使已加 invalidate_graphs()。
已修：buffer 悬空、graph 缓存失效；未修：偶发 illegal memory access（WDDM 下
cudaGraphExecUpdate/跨 B 的 graph 竞态待查）。
1) 复现脚本：conc_test.ps1 循环 + SGL_CUDA_GRAPH=1，盯 nvidia-smi 与 err log；
2) 尝试：按 (B, cap代数) 双键缓存 graph；或 capture 时固定最大 B=8、空槽 pos=0 掩码；
3) 验收：graph ON 下 conc_test 5 轮全过 + stream_test 全 PASS；BENCH 加 graph ON 的
   cuda_http 批处理吞吐行（预期恢复 t3 的 +89%）。
```

## 任务 G4 · HTTP 无 body 请求间歇卡（稳定，中）

```
现状：GET /health（无 body）在本机间歇挂起 conn_thread——CPU http.c 与 GPU cuda_http.c
两套独立实现同现象（NOTES_t5.md 详记实验矩阵：短请求挂、带 body/加 pad 全通）。
已规避：监控用 POST /health（已支持）；SO_RCVTIMEO/SO_SNDTIMEO/TCP_NODELAY 都加了。
怀疑方向：本机 LSP/杀软钩子或 winsock 对特定 recv 模式的 bug。
1) 抓包定位（pktmon/netsh trace）：卡住时包到底到没到内核；
2) 旁证实验：另一台干净 Win 机器跑同 binary 是否复现（区分环境 vs 代码）；
3) 若是环境：README 已知限制节写明+指引；若是代码：修。
验收：复现路径明确（或干净机器不复现的结论），写入 NOTES_t5.md 更新。
```

## 任务 G5 · t11 遗留：CUDA 显存预检 + soak 资源曲线（质量，小）

```
t11 做了 soak+fuzz+降级（soak_fuzz.ps1 全过），但两项没做：
1) GPU 上传前 free_mem 预检：sgl_cuda_from_llama 前 cudaMemGetInfo 估
   （权重 + KV[ctx] + scratch）需求，不够则明确报错（现在是大半路 cudaMalloc 失败退出）；
2) soak 的 RSS/VRAM 曲线：soak_fuzz.ps1 只测了延迟漂移，原要求「循环 1000 请求记录
   RSS/VRAM 修到平稳」——加曲线记录与斜率判定（Get-Process WorkingSet64 +
   nvidia-smi --query-gpu=memory.used 采样）。
验收：显存不足场景返回明确 503 文案；1000 轮 soak 曲线末段斜率 <5%（无泄漏）。
```

## 任务 G6 · GPU 前缀缓存 RadixCache（功能，中）

```
t5 的可选项没做。cuda_http 每 KV 段独立，重复 system prompt 的 chat 全量重 prefill。
1) prompt hash → 段号租约表：hash(prompt_ids) 相同的请求复用已完成 prefill 的段
   （引用计数，同 CPU 栈 radix 语义的最小子集）；
2) 或轻量版：段内 memcpyAsync 复制 KV（src 段 → dst 段 + 跳过 prefill）；
3) 验收：同 system prompt 两连发，第二发 prefill 时间 ~0 且输出与第一发逐 token 一致；
   conc_test 回归。注意与 G3 的 graph 交互。
```

## 任务 G7 · Speculative decoding GPU 侧（性能，中）

```
CPU 侧已验证框架正确（spec==greedy 保证 + 13% 接受率），但 CPU 净减速（draft 前向
成本>verify 节省，NOTES_t7.md）。GPU 才是收益场景：
1) sgl_spec_generate 的 CUDA 版：draft/target 都 sgl_cuda_forward_seg（同段隔离），
   verify 用 n-token forward_seg（GPU 已支持 n>1）；
2) lm_head 每 verify 步一次（n 行 logits）→ 主机逐步比对；
3) 验收：tinyllama-1.1B(target)+llama-160m(draft) GPU 上报告接受率与 tok/s vs 非 spec
   基线（BENCH.md 加行）；spec==greedy 断言沿用 test_spec。
```

## 任务 G8 · 真实 LLaVA 端到端 + 图片解码（广度，中）

```
t10 用随机结构验证了 CLIP→projector→融合→HTTP 全链路（numpy 9.3e-9）。缺真实权重：
1) openai/clip-vit-base-patch32 真实 CLIP（~150M，hf 直连）+ ref 分层对照
   （每层输出 maxerr<1e-4，ref_llava.py 扩展）；
2) 真实小 LM 组合：clip + projector(随机) + mistral-rand 已通，换 qwen3-06 当 LM 的
   维度适配（projector 输出 E=1024）；llava-1.5-7b 6GB 卡装不下不做；
3) JPEG/PNG 解码（可选）：stb_image.h 单头集成，HTTP 接受真实图片格式而非 raw float
   base64（现约定已文档化）。
验收：真实 CLIP 分层对照全过；带图 chat 请求 200 且输出随图变（人工检查）。
```

## 任务 G9 · 模型族补完（广度，小）

```
t9 各族主干已通（随机结构 numpy 全等 + qwen3-06 真实权重），三个特性没实现：
1) Gemma：rope_scaling（8k 基频+线性缩放，config.rope_scaling 字段）、embedding
   tanh(30x) 软上限（可选开关）；
2) Qwen3：layer_types（sliding/full 混合层）——现 qwen3-0.6B 全 full 未暴露；
   下载 qwen3-06 之外带 sliding 层的变体或改 gen 脚本生成混合层随机模型对照；
3) 真实 Mistral-7B/Gemma-2B 太大，不下；用随机结构覆盖特性即可。
验收：各特性 numpy 参考对照（ref_llama.py 已有分支框架）；ctest 各加一 case。
```

## 任务 G10 · AVX2 自动向量化重试（性能，小，条件任务）

```
t12 时 winget 最新 WinLibs 16.2（16.3 不存在），当年 25 次 0xC0000005 复现没重跑；
现用手写 AVX2 dot + 运行时检测绕开（M=1 GEMV 已吃满带宽，收益有限）。
条件：winget 出 GCC>=16.3 后。步骤：装新版 → tests 历史复现脚本跑 25 次 →
稳则 -march=native 重 bench（预计大 M GEMM 再 2x，与 OpenBLAS 195 GFLOPs 对比）。
验收：25/25 稳定 + BENCH.md 更新，或仍崩则关闭并在 ROADMAP 记录版本号。
```

---

## 附注

- **张量并行**仍未列入：单卡 6GB 无从验证；双卡时 Mistral 路径（MoE expert 并行）切入。
- 顺序建议：G1（性能主线最大缺口）→ G2 → G3（恢复 graph 收益）→ G5（小而快）→ 其余按需。
  G3 与 G6 都动 cuda_http 引擎，先 G3 后 G6。

---

## 已完成索引（v0.2.0，2026-10-27）

原 t1–t13 全部完成并验收，细节见 docs/NOTES_t*.md 与 git log（t1: t-series commits）：

| 任务 | 结果 |
|---|---|
| t1 FP16 | 1.1B 入 6GB 卡 39.4 tok/s；gap 对照可解释翻转 |
| t2 长上下文 | ctx 16384；8K GPU vs CPU 1.56e-4 PASS（66s 慢 → G1） |
| t3 CUDA Graph | B=1 +174% / B=4 +89%（cuda_cli；cuda_http 场景 → G3） |
| t4 GPT-2 CUDA | 三方（MinGW CPU==GPU fp32==fp16）逐 token 一致 |
| t5 GPU HTTP 对等 | 排队/finish_reason/SSE==plain/12 并发全过（前缀缓存 → G6） |
| t6 多 LoRA | 运行时热切换（plan B 实例池） |
| t7 Spec v2 | 13% 接受率、spec==greedy 保持；CPU 净减速如实记录（GPU → G7） |
| t8 CUDA int8 | dp4a 显存最优档（free 3947MB）greedy 全同（IMMA 对比已补，大 M → G2） |
| t9 模型族 | qwen3-06 真实 + mistral 滑窗/gemma 随机结构 numpy 全等（补特性 → G9） |
| t10 LLaVA | CLIP+projector+融合+HTTP image_url 全链路（真实权重 → G8） |
| t11 健壮性 | soak+fuzz 全过；修 5 个服务 bug（预检/曲线 → G5） |
| t12 CPU 性能 | OpenBLAS 分派 3.1x + AVX2 dot 手写（向量化重试 → G10） |
| t13 发布 | README/CHANGELOG/LICENSE（含 sglang 派生声明）/BOM 自检/从零两栈验证 |
