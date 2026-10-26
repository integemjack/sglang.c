# sglang-c 剩余任务 · 续作提示词

> 用法：新会话里先粘贴「公共上下文」块，再粘贴某条任务提示词。
> 按价值排序；任务间无强依赖，可任意顺序（标注除外）。
> v0.2（2026-10-27）：原 t1–t13 已全部完成并验收，见文末索引；本文件现列出
> 验收时发现但未修的性能/健壮性/广度缺口。
> v0.3（2026-10-27）：G1–G10 全部处理完毕（G7 净减速如实记录、G10 条件不成立关闭）。
> 数据见 docs/NOTES_gaps.md、NOTES_g8.md、NOTES_g9.md、NOTES_t7.md、ROADMAP.md，CHANGELOG 有总结。
> v0.4（2026-10-27，tag v0.3.0）：G1–G10 已发布；本文件改为记录**验收时明确留下、
> 本批没做的遗留**，重编为 G11–G18。每条都写了「现状证据出处 + 为什么没做 + 怎么做 + 验收」。

## 状态速查（v0.4）

G1–G10 已全部完成并发版，结果见文末「已完成索引」。下面是**尚未做的**：

| 任务 | 类别 | 现状证据 | 一句话 |
|---|---|---|---|
| G11 GPU 版图片解码 | 功能缺口 | NOTES_g8 §7-1 | cuda_http 只收 raw-float CHW base64，不收 PNG/JPEG |
| G12 真实 llava-1.5 链路 | 功能缺口 | NOTES_g8 §7-2/3/5 | projector 单层 linear、`<image>` 不按位置插入、无真实端到端 |
| G13 Qwen3-Next full 层 + 递归状态 | 模型广度 | NOTES_g9 §4 | full_attn 缺 q 门控/partial rotary/(1+w)/MoE；lin 状态按首页归属 |
| G14 投机解码正收益路径 | 性能 | NOTES_t7 G7 补记 | 只支持 greedy；verify 不可图化；现为 0.27–0.44x 净减速 |
| G15 前缀缓存升 radix | 性能 | NOTES_gaps “未完成”2 | 段级租让，跳段共享/显存回收/带图请求都没覆盖 |
| G16 W8 IMMA 后端到端重测 + flash 泛化 | 性能/广度 | NOTES_gaps G2、G1 | BENCH int8 行还是 dp4a 时代的数；flash 只特化 D=64/128 |
| G17 G4 修复的 GPU 版复验 + soak 误报 | 健壮性 | NOTES_t5 G4、soak_curve.ps1 | g4_probe 只在 CPU 版跑过；soak_curve 把 INFO 当错误报 |
| G18 8K GPU-vs-CPU 绝对对照 | 验收补齐 | NOTES_gaps “未完成”1 | long_vcpu 本批没跑完，用了传递论证代替 |
| G19 CI 覆盖随机模型族 | 基础设施 | .github/build.yml 与 CMake 条件注册 | CI 只下载 gpt2，模型族/多模态测试全部走 SKIP |

小事不单独立项（做上面任一任务时顺手）：`test_clip_real` 只在 `models/clip-vit-base-patch32/ref/`
存在时注册；G3 的 prewarm 会向段 0 写一次 64-token 垃圾 KV（首个真实 prefill 会覆盖，
且 attention 只读 `[kv_base, kv_base+len)`，安全，但将来做段间共享时要重新审视）。

---

## 公共上下文（每条任务前必粘贴）

```
继续开发 sglang-c：把 sglang（LLM 推理框架）用纯 C 重写的项目，位于
C:/Users/ultra/Desktop/works/sglang/sglang-c/（约 1.3 万行 C+CUDA，M0–M26 + t1–t13 + G1–G10 已完成，tag v0.3.0）。
现状：
- CPU 栈（MinGW GCC 16.2 / Ninja / x86-64-v2）：gpt2 / llama族(gpt2、llama、qwen2、qwen3、
  mistral 滑窗、gemma 归一化) / sgl_model 抽象；连续批处理调度器；HTTP（CPU http.c + GPU
  cuda_http.c 双实现：/health /generate /v1/completions /v1/chat/completions，ChatML、SSE、
  排队、finish_reason 如实、多模态 image_url）；RadixCache 分页 KV + LRU；W8A16 权重 +
  int8 KV；SSE2/SSE4.1 手写 + AVX2 dot（运行时检测）+ OpenBLAS 运行时分派（大 M GEMM 3.1x）；
  beam search；spec v2（n-token 批量 verify，160m draft 接受率 13%，spec==greedy 保证）；
  多 LoRA 热切换（plan B 实例池）；CLIP ViT 编码器 + mm_projector + embedding 级融合前向。
  23 套 ctest 全绿（含 t9 三族随机结构 numpy 对照、t10 llava、G8 真实 CLIP、G9 gemma2/qwen3next）。
- CUDA 栈（VS2022 BuildTools + nvcc 13.3，sm_75 = RTX 2060 6GB，build-cuda 目录，
  CMake 自定义命令直驱 nvcc）：llama 族 + gpt2 双后端；三精度（fp32 / fp16 SGL_CUDA_FP16 /
  W8A8 dp4a SGL_CUDA_W8，显存最优档 free VRAM 3947MB；M>1 已切 cblas IMMA）；CUDA Graph
  （cuda_http 默认开：capture 前预热 + 按 B 隔离失败 + 起服务前 prewarm；B=1/4/8 2.4/4.1/3.4x）；
  单遍 flash 分块 attention（8K prefill 8.3x）+ 两遍核保留作 A/B（SGL_ATTN_TWOPASS=1）；
  sgl_cuda_forward_seg_emb；段级前缀缓存（sgl_cuda_copy_kv + 租让表，输出逐字一致）；
  GPU 侧投机解码（sgl_cuda_forward_seg_rows + src/cuda_spec.c）；显存预检与降级 503。
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

## 任务 G1 · GPU 长上下文 prefill 提速（性能，中大） ✅ 已完成（v0.3.0）

> 下面 G1–G10 是已完成的原始提示词，保留作历史参考；结果见文末已完成索引。

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

## 任务 G2 · W8 大 M 切 cublas IMMA（性能，中） ✅ 已完成（v0.3.0）

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

## 任务 G3 · cuda_http decode graph 稳定性（稳定/性能，中） ✅ 已完成（v0.3.0）

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

## 任务 G4 · HTTP 无 body 请求间歇卡（稳定，中） ✅ 已完成（v0.3.0）

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

## 任务 G5 · t11 遗留：CUDA 显存预检 + soak 资源曲线（质量，小） ✅ 已完成（v0.3.0）

```
t11 做了 soak+fuzz+降级（soak_fuzz.ps1 全过），但两项没做：
1) GPU 上传前 free_mem 预检：sgl_cuda_from_llama 前 cudaMemGetInfo 估
   （权重 + KV[ctx] + scratch）需求，不够则明确报错（现在是大半路 cudaMalloc 失败退出）；
2) soak 的 RSS/VRAM 曲线：soak_fuzz.ps1 只测了延迟漂移，原要求「循环 1000 请求记录
   RSS/VRAM 修到平稳」——加曲线记录与斜率判定（Get-Process WorkingSet64 +
   nvidia-smi --query-gpu=memory.used 采样）。
验收：显存不足场景返回明确 503 文案；1000 轮 soak 曲线末段斜率 <5%（无泄漏）。
```

## 任务 G6 · GPU 前缀缓存 RadixCache（功能，中） ✅ 已完成（v0.3.0）

```
t5 的可选项没做。cuda_http 每 KV 段独立，重复 system prompt 的 chat 全量重 prefill。
1) prompt hash → 段号租约表：hash(prompt_ids) 相同的请求复用已完成 prefill 的段
   （引用计数，同 CPU 栈 radix 语义的最小子集）；
2) 或轻量版：段内 memcpyAsync 复制 KV（src 段 → dst 段 + 跳过 prefill）；
3) 验收：同 system prompt 两连发，第二发 prefill 时间 ~0 且输出与第一发逐 token 一致；
   conc_test 回归。注意与 G3 的 graph 交互。
```

## 任务 G7 · Speculative decoding GPU 侧（性能，中） ✅ 已完成（v0.3.0）

```
CPU 侧已验证框架正确（spec==greedy 保证 + 13% 接受率），但 CPU 净减速（draft 前向
成本>verify 节省，NOTES_t7.md）。GPU 才是收益场景：
1) sgl_spec_generate 的 CUDA 版：draft/target 都 sgl_cuda_forward_seg（同段隔离），
   verify 用 n-token forward_seg（GPU 已支持 n>1）；
2) lm_head 每 verify 步一次（n 行 logits）→ 主机逐步比对；
3) 验收：tinyllama-1.1B(target)+llama-160m(draft) GPU 上报告接受率与 tok/s vs 非 spec
   基线（BENCH.md 加行）；spec==greedy 断言沿用 test_spec。
```

## 任务 G8 · 真实 LLaVA 端到端 + 图片解码（广度，中） ✅ 已完成（v0.3.0）

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

## 任务 G9 · 模型族补完（广度，小） ✅ 已完成（v0.3.0）

```
t9 各族主干已通（随机结构 numpy 全等 + qwen3-06 真实权重），三个特性没实现：
1) Gemma：rope_scaling（8k 基频+线性缩放，config.rope_scaling 字段）、embedding
   tanh(30x) 软上限（可选开关）；
2) Qwen3：layer_types（sliding/full 混合层）——现 qwen3-0.6B 全 full 未暴露；
   下载 qwen3-06 之外带 sliding 层的变体或改 gen 脚本生成混合层随机模型对照；
3) 真实 Mistral-7B/Gemma-2B 太大，不下；用随机结构覆盖特性即可。
验收：各特性 numpy 参考对照（ref_llama.py 已有分支框架）；ctest 各加一 case。
```

## 任务 G10 · AVX2 自动向量化重试（性能，小，条件任务） ✅ 已完成（v0.3.0）

```
t12 时 winget 最新 WinLibs 16.2（16.3 不存在），当年 25 次 0xC0000005 复现没重跑；
现用手写 AVX2 dot + 运行时检测绕开（M=1 GEMV 已吃满带宽，收益有限）。
条件：winget 出 GCC>=16.3 后。步骤：装新版 → tests 历史复现脚本跑 25 次 →
稳则 -march=native 重 bench（预计大 M GEMM 再 2x，与 OpenBLAS 195 GFLOPs 对比）。
验收：25/25 稳定 + BENCH.md 更新，或仍崩则关闭并在 ROADMAP 记录版本号。
```

---

## 任务 G11 · GPU 版收标准图片（功能，中） ✅ 已完成（未发版，见 docs/PROGRESS_g11_g19.md）

```
现状（doc/NOTES_g8.md §7-1）：CPU 版 HTTP 已能收 OpenAI vision 风格的
`image_url` base64 PNG/JPEG（src/image.c + third_party/stb/stb_image.h，PNG 解码与 PIL
像素级一致）；GPU 版 apps/cuda_http.c 仍只接受 t10 的 raw float CHW base64
（cuda_http.c:182-226 附近，直接 memcpy 当 float 用），真实客户端（发的都是 JPEG）用不了。
为什么没做：G8 分工只动 CPU 路径。
1) CMakeLists.txt:203 的 SGL_CORE_SRC 末尾加 src/image.c（stb include 已给 sgl_cuda 加好，
   CMakeLists.txt:206-207）；
2) cuda_http.c 的 vision 回调：按 data URL 的 mime 分派：image/png|image/jpeg 走
   sgl_image_decode → resize 到 clip->img → CLIP → projector；保留 raw float CHW 分支
   作为兼容（端到端脚本还在用）；
3) 验收：tests/http_image_e2e.py 指向 cuda_http 端口跑 jpeg/png/text 三种，全 200 且
   同 seed 输出互不相同；环 base64 → 400；再加一条「GPU 融合前向 vs CPU 融合前向」逐 token
   对照（已有 sgl_cuda_forward_seg_emb，历史 maxerr 1.2e-7）。
```

## 任务 G12 · 真实 llava-1.5 链路（功能，中大） ✅ 已完成（两层 MLP projector + 占位符位置插入 + 真实 CLIP-L/14-336 与真实 mm_projector 分段对照；7B LM 端到端因磁盘不够未做，任务本身允许分段）——见 docs/NOTES_g8.md §8

```
现状（NOTES_g8.md §7-2/3/5）：只对齐了 CLIP 视觉塔本身（真实 clip-vit-base-patch32，
分层 maxerr ≤1.24e-05）；剩下的都是「假」的：
  a) projector 仍是单层 linear；真实 llava-1.5 是 linear_1 → GELU → linear_2，且用
     CLIP large-patch14-336（proj_dim 1024 → 语言模型 4096）；
  b) `<image>` 只当占位符，视觉 token 接在 prompt 末尾而不是插到占位位置；多图只是拼接；
  c) 没有真实 LLaVA 端到端（CLIP + projector + 语言模型一起）。
为什么没做：需真实 llava-1.5 权重（~4GB）与本机显存/磁盘安排。
1) projector 泛化：从 mm_projector 权重形状自动判断层数（单层 linear vs 两层 MLP），
   sgl_clip_encode 后接 MLP；numpy 参考同步（tests/ref_llava.py 已有分支框架）；
2) prompt 模板：在 tokenize 后的序列里找 `<image>` 对应 token 位置，把视觉 token
   （576/576 个）插在那里（而不是末尾）；位置编码按插入后的真实位置算（RoPE 位置偏移）；
3) 权重：用 ModelScope/hf-mirror 拉 llava-1.5-7b（若 6GB 卡装不下，用 llava-1.5 的
   Vicuna-1.5-7B 量化版或只抽 projector + CLIP 做分段对照）；
4) 验收：真实图片 + 文字提问，端到端输出与 Python transformers（同一权重）逐 token 对齐
   （greedy 前 16 步允许 top1/top2 gap < 1e-2 的可解释翻转）；no-image 时行为与纯文本全等。
```

## 任务 G13 · Qwen3-Next full 层与递归状态（模型广度，中大） ✅ 已完成（a–e 全做；CUDA 未支持 = 文档允许的选项。见 docs/NOTES_g9.md 第 5 节）

```
现状（docs/NOTES_g9.md §4）：linear_attention（Gated DeltaNet）已按 HF 真实公式实现并
vs float64 numpy ≤8e-08；但 full_attention 层还是普通 qwen3 写法，缺：
  a) q 门控（q_proj 输出 2×，半幅 sigmoid 门控 attn 输出）；
  b) partial rotary（rope 只作用 25% 维度）；
  c) output norm 是 (1+w)*x/sqrt(var+eps)，不是普通 RMSNorm；
  d) MoE 专家层 + 共享专家 + routed scaling；
  e) dynamic rope 取的是静态 NTK，不是 HF 的 dynamic_rope_update 逐步插值。
为什么没做：真实 Qwen3-Next 是 80B，本机跑不动；但加载时应报缺张量而不是静默算错（已如此）。
1) 先扩 gen_random_qwen3next.py 造一个带上述四个特性的小模型（把 hybrid 层配置成 1-2 个
   full 层 + MoE 2 专家），ref_qwen3next.py 同步；
2) llama.c 实现（每个特性一个小步，每步跑 numpy 对照）；
3) 递归状态（lin_slot）：改为按序列分配而不是按首 KV 页——radix 前缀共享时两条序列共用
   首页会互踩状态。验证：两条序列相同前缀、不同后缀并发 decode，各自状态与单独跑全等；
4) CUDA 路径可选：CUDA 后端目前只有 llama 族普通 attention，Qwen3-Next 可只保证 CPU。
验收：ctest 新增 case 全绿；真实 qwen3-06 回归不变；带 MoE 的随机模型 numpy maxerr < 1e-6。
```

## 任务 G14 · 投机解码正收益路径（性能，中大） ✅ 已做完（结构成本已砍；验收门槛未达，如实记“仍净减速”，见 docs/NOTES_t7.md G14 补记）

```
现状（NOTES_t7.md G7 补记）：GPU 侧实现已正确（spec == 非 spec greedy 逐 token 全同），
但性能是**净减速**：K=1 0.44x / K=2 0.35x / K=4 0.27x。实测成本分解（K=4）：每轮
≈ 4 次 draft 图步 ~20ms + verify n=4 plain ~30ms + 补齐 2 次 plain ~45ms；补齐改走
图步后 11.9→22.0 tok/s。根因：① 未蒸馏同源 draft 接受率上限 13-15%；② verify/catchup
的 plain 路径在 WDDM 下不能图化（n 可变）。
为什么没做：需要新 draft 权重（蒸馏）或图化改造，超出当轮范围。
三个可独立验证的方向（按性价比排序）：
1) **verify 固定 n 图化**：每轮总是验 n=K+1 行（不够就用 mask/重复 token 填满），这样
   verify 能走 cudaGraph（一张固定 n 的图），预期收回 2-3 倍成本；
2) **批量 spec**：cuda_http 里把多个请求的 spec 轮合到一次 batch forward（B>1 时 verify
   成本本就被摊薄）；
3) temperature>0 的修正采样（speculative sampling：接受概率 min(1, p/q)，拒绝时从
   (p-q)+ 重采样）——当前只支持 greedy，这是功能缺口。
验收门槛：至少一个方向能把 K=1/K=2 做到 ≥1.0x（接受率不变的前提下），且 spec 输出与
非 spec greedy 逐 token 全同；BENCH.md 更新，做不到就如实记“该组合仍净减速”。
```

## 任务 G15 · 前缀缓存升 radix（性能，中大） ✅ 已做完（a 用“登记生成 token 行”替代两轮扫描并实测无收益；b/c 完成）——见 docs/BENCH.md 的 G15 节

```
现状（NOTES_gaps.md “未完成”2、G6 节）：现在是**段级租让**：一条序列一个槽，
最长公共前缀按 token 精确比对，命中就把 KV 行搬过来。已验：reuse vs 全算 maxerr
1.43e-05、greedy 6/6 全同，端到端 0.76→0.35s。没做的：
  a) 跳段共享（A/B 共享前缀但排在第三位这种，只能命中一个）；
  b) 显存压力下主动回收租让（现在只有表满 LRU，段从不因显存紧张被让出）；
  c) 带 visual token 的请求不登记（cuda_http 里 n_img>0 直接跳过）；
  d) 前缀树（CPU 侧 RadixCache 那种），而不是扁平表。
1) 先做 a 的最小版：租让表按「前缀长度降序」扫，命中后如果源段还有更长匹配的兄弟项
   就继续比（两轮扫描），不改存储结构就能拿到大部分收益；
2) b：在 reserve/prewarm 失败或 /health 的 degraded 阈值接近时，按 LRU 主动 drop 租让
   并重试；/
3) c：把视觉 embedding 的 hash 也算进前缀（或干脆单独一张 multimodal 租让表）；
4) d 只在 1-3 都做完且测出瓶颈后再考虑。
验收：prefix_e2e.ps1 扩成四组（同前缀/跳段/带图/显存紧张），全 PASS；
命中率/省下 prefill 行数写进 /health 与 BENCH；输出始终与无缓存逐字一致。
```

## 任务 G16 · W8 IMMA 后重测 + flash 泛化（性能，小） ✅ 已完成（未发版，见 docs/PROGRESS_g11_g19.md）

```
现状：G2 把 M>1 的 int8 投影切到了 cublas IMMA（[2048,2048] M=8 0.925→0.069ms，13.4x），
但 **BENCH.md 里的 int8 行还是 dp4a 时代的数**（qwen05 int8 33.8 tok/s 是 decode，
影响不大，但长 prompt prefill 没重测）；G1 的 flash 核只特化了 D=64（qwen05）与 D=128
（tinyllama），其它 head_dim 仍回退两遍核。
1) 用 tests/cuda_perf.c --ab 重测三档（fp32/fp16/int8）的 prefill n=512/2048/8192，
   把 int8 的 prefill 拐点找出来（dp4a 在 M 多大时开始输），写进 BENCH.md；
2) flash 泛化：把 D 做成模板参数的第三个维度（64/80/96/128），或加一个 D 不整除 32 的
   慢路径（现在 ND=D/32 要求整除）；用 mistral-rand（D=64）之外的模型造一个 D=96 的
   随机模型验数值。
验收：BENCH 更新；cuda_test [long] 全绿；D=96 随机模型 GPU vs CPU maxerr < 5e-3。
```

## 任务 G17 · G4 的 GPU 版复验 + soak 误报（健壮性，小） ✅ 已完成（未发版，见 docs/PROGRESS_g11_g19.md）

```
现状：G4 的根因（srv->lock 被 worker 计算和 acceptor 登记争用）修在 **CPU 版** src/http.c
（拆 conns_lock），并已在 CPU 版验到 600/600 无超时。GPU 版 apps/cuda_http.c 是另一套
实现（acceptor 不取 E.lock，理论上不该有同病），但 **g4_probe.py 从未打过 cuda_http**，
而 t5/NOTES_t5 最初的症状描述就是 GPU 版。另：tests/soak_curve.ps1 把 server stderr 的
任何行都当错误报（qwen2 加载器的 q_norm/k_norm not found 是无害 INFO），会误导排查。
1) g4_probe.py 加 --port 参数（已有 HOST/PORT 常量，改成可传），对 cuda_http 跑
   load 600 12 256，要求 0 个 timeout_read；
2) 若 GPU 版也能复现：查 cuda_http.c 的 accept 循环（第 860-875 行附近）与
   EnterCriticalSection(&E.lock) 的持有段（worker 第 205-300 行）；
3) soak_curve.ps1：stderr 只把匹配 ERROR|FATAL|abort 且不含已知白名单的行算错误，
   并把误报写进输出注释。
验收：g4_probe 两版（CPU/GPU）都是 0 超时；soak_curve 对干净服务不再报错。
```

## 任务 G18 · 8K GPU-vs-CPU 绝对对照补跑（验收，小但慢） ✅ 已完成（maxerr 1.47581e-04 < 1e-3，greedy 同；日志 docs/g18_8192.log）

```
现状（NOTES_gaps.md “未完成”1）：t2 那条 `long_vcpu models/qwen05 8192`（GPU prefill
logits vs CPU 同 ids，要求 maxerr < 1e-3）本批没跑完——CPU 侧 8192-token prefill 在这台
机器上 40+ 分钟没结束（且当时有并发争用），已主动终止。G1 的长上下文正确性改用
传递论证（flash == 两遍核 maxerr=0，两遍核 t2 已对 CPU 1.56e-4）代替。
1) 在**干净、无其他任务**的时段跑一次（需要 SGL_POOL_SLOTS=8208 才不报 cpu pages fail），
   预计 40-60 分钟；
2) 或把 CPU 侧改成只跑最后 1 层 logits 的对照（搝 8K 但只算最后一层 attention 后的
   hidden）来缩短时间——但这就不叫端到端对照了，不推荐；
3) 记录实际 maxerr 并把 NOTES_gaps 的“未完成”1 改为已完成（附时间、空闲内存）。
验收：verdict: PASS (<1e-3)，原始输出存 docs/ 或 NOTES_gaps 引用的日志路径。
```

## 任务 G19 · CI 真跑模型族与多模态测试（基础设施，中） ✅ 已完成（未发版，见 docs/PROGRESS_g11_g19.md）

```
现状（2026-10-27 查 v0.2.0 的 CI 日志时发现）：CI 只 curl 了 models/gpt2/model.safetensors，
其余模型全缺。CMake 的 G9 段（CMakeLists.txt:141-155）与 G8 段（:133-137）都是
「模型存在才注册 test」，所以 CI 上 test_family_{mistral,gemma,gemma2,qwen3,qwen3next}
与 test_clip_real 根本不跑；而 test_llama/test_qwen/test_tinyllama/test_w8/test_spec 走了
SKIP。结果是 CI 实际只验了「能编译 + 不依赖模型的几个测试」，项目最核心的数值正确性
（numpy 对照、模型族、多模态）完全没被 CI 覆盖。
为什么没做：当时为了不恒红而选了条件注册（v0.2.0 前 5 个 job 就是被 test_llava 拖红的）。
1) gen_random_*.py 共 5 个脚本硬编码从 models/qwen3-06/ 拷 tokenizer——加一个
   --tokenizer <dir> 参数（CI 传 models/gpt2；gpt2 的 vocab.json/merges.txt 已在仓库），
   并把输出目录也参数化（默认保持 ../models/<fam>-rand，CI 传 models/）；
2) workflow 在 Configure 前加一步（仅 CPU job）：pip install numpy &&
   对 mistral/gemma/gemma2/qwen3next/llava 逐个 python tests/gen_random_<fam>.py；
3) 注意 _ref_*.json 已在仓库且与被生成权重必须一致（脚本固定 seed；若改过 seed，
   要同时重生 ref 并提交）；
4) 验收：CI 日志里 test_family_mistral/gemma/gemma2/qwen3next 与 test_llava 的输出是
   Passed 且不是 SKIP 字样；总测试数从 18 涨到 ~23。
若生成步太慢/太不稳，退而求其次：只生成 gemma2 + qwen3next（G9 新加的两个）
先把新特性的回归盯住。
```

---

## 附注

- **张量并行**仍未列入：单卡 6GB 无从验证；双卡时 Mistral 路径（MoE expert 并行）切入。
- 顺序建议：G11（最小、承接 G8 的真实缺口）→ G17（小，先把 G4 结论坐实）→ G19
  （CI 覆盖，影响后续每次改动的信心）→ G16（重测已有实现，不改设计）→ G14 → G12 → G15 → G13 → G18。
  G12 与 G15 都动多模态前缀语义，先 G12 后 G15；G14 与 G3 都掺 cuda graph，先 G3（已完成）。

---

## 已完成索引（v0.3.0，2026-10-27）

### G1–G10（缺口关闭批）

| 任务 | 结果 |
|---|---|
| G1 prefill 提速 | 单遍 flash 核；8K prefill 72.4→8.8s（**8.3x**，同进程 A/B）；flash vs 两遍 @4300 行 maxerr=0 |
| G2 W8 大 M IMMA | [2048,2048] M=8 0.925→0.069ms（**13.4x**）；imma vs dp4a maxerr=0、10/10 greedy 同 fp32 |
| G3 decode graph 稳定 | capture 前预热 + 按 B 隔离失败 + 起服务前 prewarm；B=1/4/8 2.4/4.1/3.4x；默认开 |
| G4 无 body 间歇卡 | 根因 `srv->lock` 争用饿死 acceptor → 拆 `conns_lock`；CPU 版 600/600 无超时 |
| G5 显存预检 + soak | 不够时降级服务（/health degraded、/generate 503）；8min soak RSS 斜指 0.0046%/min |
| G6 GPU 前缀缓存 | `[prefix]` 1.43e-05 + greedy 6/6 全同；端到端 0.79→0.35s 输出逐字一致 |
| G7 GPU 投机解码 | 正确性全过（== 非 spec greedy）；性能 K=1 0.44x 净减速，如实记录 |
| G8 真实 LLaVA | 真实 CLIP 分层 vs torch **≤1.24e-05**；stb_image PNG/JPEG 进 CPU HTTP |
| G9 模型族补完 | Gemma rope_scaling/softcap/layer_types；Qwen3-Next Gated DeltaNet **≤8e-08**；ctest 20→23 |
| G10 AVX2 重试 | 条件不成立（无 GCC≥16.3）关闭；`-march=native` 25/25 轮崩，x86-64-v2 对照 20/20 |

### t1–t13（v0.2.0，细节见 docs/NOTES_t*.md 与 git log）

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
