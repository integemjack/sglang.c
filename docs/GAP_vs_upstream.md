# sglang-c vs 上游 sglang：能力差距盘点（2026-10-27）

> **2026-10-27 更新（P0/P1/P2 补齐）**：本文列出的差距中，第一轮已实现并验证的是：
> 分片 safetensors + FP8 权重（P0-1/P1）、Jinja chat template（P0-2）、采样参数全量 + `input_ids` + logprobs（P0-3）、
> chunked prefill + 抢占重算（P0-4）、DeepSeek MLA 注意力（P0-5 内核）、约束解码 JSON Schema/正则（P1）、
> 服务面 `/ready /v1/models /get_server_info /metrics /flush_cache /abort_request /update_weights_from_disk`（P1）、
> 数据并行副本 `--dp N`（P1）、embedding/rerank/score 编码器与端点（P2）、reasoning/tool_call 解析器（P2）、
> YaRN RoPE + mscale（P1，对 HF 逐值验证）。
> 下文表格保留原貌，**读的时候以本段为准**；仍未做的集中在：其余非 Llama 系架构（各类 VLM、扩散/语音）、
> `.bin`/GGUF/AWQ/GPTQ/W4A16、HiCache 分层、TP。
> （DeepSeek-V2/V3 已有**完整模型后端**：`src/deepseek.c` + MLA 内核，并用 transformers 的
> `DeepseekV2ForCausalLM` 作金标准对到 2.5e-07；YaRN 也已实现，见 README 的 Feature matrix。）

对照对象：本目录父级 `../` 的 sgl-project/sglang（Python 核心约 138 万行 + C++/CUDA kernel 约 16 万行），
被对照方：`sglang-c/`（C11 + 可选 CUDA，`src` + `include` + `apps` 合计 **11,440 行**）。

一句话结论：**主干推理闭环（模型前向 → paged KV → radix 前缀复用 → continuous batching → OpenAI 兼容 HTTP）已经跑通并做过数值验证，
但离"能替代原版跑真实生产负载"还差五块：非 Llama 系模型架构、权重/对话模板的通用性、服务端调度与 API 参数面、
KV 缓存分层、以及并行/分布式。** 前端语言、多硬件平台、网关这一类属于"刻意不追"，应单列为目的问题。

---

## 0. 已有的（对照基准，避免重复劳动）

| C 版已有 | 位置 |
| --- | --- |
| GPT-2 / Llama 系（llama、qwen2、qwen3、mistral、gemma、gemma2、qwen3-next 混合层 + MoE） | `src/gpt2.c`、`src/llama.c`（`model.c` 只分 gpt2/llama 两类后端） |
| 权重：HF 单文件 `model.safetensors`，F32/F16/BF16 → fp32，mmap 零拷贝 | `src/safetensors.c` |
| Tokenizer：GPT-2 byte-level BPE + Llama Metaspace | `src/bpe.c` |
| Paged KV 池 + CPU 侧 radix 前缀树（页级、token 精确、refcount + LRU） | `src/llama.c`、`src/radix.c` |
| GPU 侧前缀缓存（16 个 segment lease，KV 行拷贝，`SGL_PFX_MAX`） | `apps/cuda_http.c` |
| continuous batching 调度器（GPU 并发 1–8，段长 = ctx/conc） | `src/scheduler.c` |
| HTTP：`/health` `/generate` `/v1/completions` `/v1/chat/completions`（SSE 流式） | `src/http.c`、`apps/cuda_http.c` |
| 量化：CPU W8A16、CUDA W8A8（dp4a + IMMA）、int8 KV、fp16 权重 | `src/quant.c`、`src/cuda_backend.cu` |
| CUDA：自写 flash prefill + 两段式长上下文 + decode CUDA graph（B=1..conc 预捕获） | 同上 |
| beam（W≤8）、speculative（draft/verify，K≤8，净减速）、多 LoRA（预合并）、LLaVA 单图 | `src/beam.c`、`src/spec.c`、`src/model.c`、`src/clip.c` + `src/image.c` |

---

## 1. 模型覆盖面（最大缺口）

| 项 | 上游 | C 版 |
| --- | --- | --- |
| 模型实现文件 | **253 个** `srt/models/*.py` | 2 个后端（gpt2 / llama 系） |
| DeepSeek 系（V2/V3/V4、MLA、DSA/NSA 稀疏注意力、nextn/MTP） | 有（`deepseek*.py` 十余个） | 无 |
| MoE 大规模（Mixtral、DBRX、Qwen3-MoE、Cohere2-MoE、bailing_moe、gpt-oss/mxfp4…） | 有 | 只有 Qwen3-Next 的 top-k + shared expert 单机版本 |
| 多模态（Qwen-VL/InternVL/Kimi-VL/GLM-V/Ernie-VL、视频、音频、encoder 独立调度、ViT CUDA graph） | 有（`multimodal/` + `parser/`） | 只有 LLaVA 单图（CLIP-L/14-336 + MLP projector） |
| embedding / rerank / reward / 分类模型（bert、embedding_gemma2…） | 有，含 `/v1/embeddings` `/v1/rerank` `/v1/score` | 无 |
| 扩散 / 语音（`multimodal_gen/`、whisper、`/v1/audio/transcriptions`） | 有 | 无 |
| dLLM（扩散式 LLM）、线性/混合架构扩展（Mamba、SWA、Minimax-sparse…） | 有 | 仅 Qwen3-Next 一种混合层 |

影响：C 版目前只能吃"Llama 同构 + 标准 MHA/GQA"的 checkpoint；当下主流权重（DeepSeek、Qwen3-MoE、各类 VLM）全部打不开。
这是"能不能用"级别的缺口，不是"好不好用"。

## 2. 权重加载与对话模板（第二缺口，成本低但影响直接）

| 项 | 上游 | C 版 |
| --- | --- | --- |
| 分片 safetensors（`model.safetensors.index.json`） | 有 | **无**，只读单个 `model.safetensors`（`src/safetensors.c` 无 index 处理） |
| PyTorch `.bin`、GGUF、AWQ/GPTQ/compressed-tensors checkpoint | 有 | 无（GEMM 里没有 dequant 元数据路径） |
| HF Hub 直连下载 / 权重缓存（`weight_cache/`） | 有 | 无，必须本地目录 |
| dtype | fp32/fp16/bf16/fp8/int4 全链 | F32/F16/BF16，统一转 fp32（大模型内存 ×2） |
| chat template | Jinja（`tokenizer_config.json` 的 `chat_template`，`parser/template_manager.py` + 模板检测） | **硬编码 ChatML**：`<|im_start|>role\n…`（`src/http.c:395`），非 Qwen 系模型的对话格式是错的 |
| tokenizer | HF tokenizers 全类型（BPE/Unigram/WordPiece）、added tokens、多模态 processor | byte-BPE + Metaspace 两类 |
| 长 prompt 输入 | `text` / `input_ids` / 多模态 embedding | 只有 `text`（无 `input_ids`） |

## 3. 调度、KV 缓存与并发

| 项 | 上游 | C 版 |
| --- | --- | --- |
| chunked prefill（按 token budget 跨 step 切片，与 decode 混跑） | 有 | 无：CPU 侧 `sgl_sched_step` 每个 step 严格「先整段 prefill 一个请求 → 再 batch decode」，prefill 期间整批 decode 停摆；GPU 侧只是把长 prompt 拆到多个 segment 顺序 prefill |
| 抢占 / 重算 / 换出 | 有（`schedule_policy.py`、prefill delayer、`min_free_slots_delayer`） | **无**（全仓 `preempt` 0 处命中），池满时只能排队/失败 |
| 优先级队列、请求分池、DP attention、PP 调度 | 有 | 无（`priority` 0 处命中） |
| KV 分层缓存 HiCache（host / 磁盘 / Mooncake / NIXL，`mem_cache/` 168 个文件） | 有 | 无，纯设备内存 |
| 前缀缓存形态 | 页级 radix + 多种 evict policy | CPU：页级 radix；GPU：**16 个 segment lease**（整段租约，不是页级树） |
| 缓存运维 | `/flush_cache`、`/hicache/*`、缓存命中报告 | 只有 `/health` 里的计数，无 flush |
| session / 多轮会话对象、streaming session | 有（`session/`） | 无（靠前缀缓存自然命中） |
| 并发上限 | 由显存/调度动态决定，可上百 | GPU 固定 `SGL_HTTP_CONC` 1–8，段长 = ctx/conc（`SGL_CUDA_CTX` 默认 16384） |

## 4. 解码与采样

| 项 | 上游 | C 版 |
| --- | --- | --- |
| 约束解码 / 结构化输出（xgrammar、outlines、llguidance + jump-forward、JSON schema、EBNF/正则） | 有（`constrained/`） | **无**（`grammar`/`json_schema`/`response_format` 0 处命中） |
| speculative：EAGLE/EAGLE3、Medusa、MTP(nextn)、ngram、DP 并行 draft（`speculative/` 66 个文件） | 有 | 只有朴素 draft/verify：接受率 13%（截层）～39%（llama-160m），**仍是净减速** |
| 采样参数 | temperature/top_p/top_k/min_p/typical、repetition/frequency/presence penalty、logit_bias、stop/stop_token_ids/ignore_eos、n/best_of、`spaces_between_special_tokens` | temperature/top_p/top_k/seed/n/stop（`src/http.c` 476–484） |
| logprob 回传 | `return_logprob`、`logprob_start_len`、`top_logprobs_num`、`meta_info` 全套 | 无（`logprobs` 0 处命中；`meta_info` 只有 finish_reason + completion_tokens） |
| reasoning parser / tool-call parser / code completion parser（`parser/` 20+ 文件，`/parse_function_call`、`/separate_reasoning`） | 有 | 无：请求里的 `tools`/`functions` 字段完全没被解析（全仓仅 `sglc.c` 一处注释提到 tools），也没有 reasoning 分离 |

## 5. 服务与 API 面

| 项 | 上游 | C 版 |
| --- | --- | --- |
| 路由数 | **40 余个**（`/generate`、`/v1/{completions,chat/completions,embeddings,rerank,score,models,responses,messages}`、`/v1/audio/transcriptions`、`/flush_cache`、`/abort_request`、`/pause_generation`、`/continue_generation`、`/get_server_info`、`/get_model_info`、`/health_generate`、`/ready`、`/metrics`、`/update_weights_from_{disk,distributed,tensor,ipc}`、`/init_weights_update_group`、`/hicache/*`、`/invocations`、`/parse_function_call`、`/separate_reasoning`） | **4 个**：`/health` `/generate` `/v1/completions` `/v1/chat/completions` |
| OpenAI 兼容深度（tools/function calling、response_format、logprobs、echo、suffix、多模态 content part、`/v1/models`） | 完整 | 只有 text/image_url → ChatML 字符串 |
| 请求生命周期 | abort / pause / continue / 权重热更新 / 优雅退出 | 无中止（只能等生成完） |
| gRPC、Anthropic `/v1/messages`、Ollama 适配 | 有（`entrypoints/{grpc,anthropic,ollama}`） | 无 |
| 多实例路由 / 缓存感知负载均衡 / PD 分离 router | 有（`sgl-router` Rust、`sgl-model-gateway`、`disaggregation/`） | 无 |
| 可观测性（Prometheus `/metrics`、OTel trace、req-time stats、CPU monitor、startup timing、warmup） | 有（`observability/` 15 个文件） | 只有 `/health` 计数 + `SGL_HTTP_TRACE` 环境变量；无 metrics 导出 |
| 前端语言（`sglang.lang`：gen/select/fork/join、正则约束原语、RadixAttention 作为语言原语、interpreter/tracer） | 有（sglang 最具辨识度的部分） | 无，只有 CLI + HTTP |

## 6. 并行与分布式（完全空缺）

* TP / PP / DP / EP / CP、`parallel_state`、NCCL 通信、Ray 后端、弹性 EP/EPLB（`elastic_ep/`、`eplb/`）、
  batch overlap、PD（prefill/decode）分离与 Mooncake/NIXL 传输、`disaggregation/` 全套 → **C 版一项都没有**，
  README 自己写了 "Not covered: tensor parallelism"。单进程、单卡、单机。
* RL/训练侧接口：`/update_weights_from_*`、checkpoint engine、weight sync → 无（sglang 现在大量当 rollout 引擎用，
  这块缺失等于进不了 RL 栈）。

## 7. 量化与算子后端

| 项 | 上游 | C 版 |
| --- | --- | --- |
| 量化方案 | fp8（per-tensor/per-token/per-block/per-channel）、W4A16、AWQ/GPTQ/Marlin、GGUF、compressed-tensors、modelopt、bitsandbytes、MXFP4/NVFP4、mxfp8、块状 int8 —— `layers/quantization/` **109 个文件** | W8A16（CPU）、W8A8 dp4a/IMMA（CUDA）、int8 KV、fp16 权重 |
| attention 后端 | **138 个文件**：FlashInfer、FA3/FA4、TRT-LLM、CUTLASS、FlashMLA、DeepGEMM、NSA/DSA、Mamba/线性、Intel AMX、ROCm aiter… | 1 套自写 kernel（shared-scores / flash / 两段式）+ cuBLAS |
| 编译/图 | torch.compile、piecewise + full CUDA graph、TBO、batch-invariant ops | decode CUDA graph（B=1..conc），无 prefill graph、无编译层 |
| 硬件平台 | 10 个（cuda / rocm / npu / xpu / mps / musa / cpu / sail…） | x86-64 CPU、CUDA sm_75+、arm64 CPU（macOS 单线程） |

## 8. 工程与验证

* 上游有 `bench_serving.py` / `bench_one_batch_server.py` 等标准压测、nightly 大模型回归；C 版只有本地 `bench_all.ps1`、
  26 套 ctest，**CI 不跑 CUDA、不跑真实 LLaVA 权重**（README 明写）。
* C 版的正确性对照是"自写 numpy 参考 + 少量真模型"，**没有与原版 sglang / vLLM 端到端吞吐与时延的对照基准**，
  目前所有性能数字都是"自己 vs 自己"的 A/B。
* 上游的 138 万行里有大量边界处理（chat template、special tokens、多模态 processor、stop 语义、partial 流式切分）；
  C 版这些语义靠自己实现，缺一份"和原版逐请求对拍"的 conformance 套件。

---

## 建议的推进顺序（按"离可用还差多少"排）

**P0（不做就没法吃主流权重）**
1. `model.safetensors.index.json` 分片加载 + bf16/fp16 直算（避免统一 fp32 的内存翻倍）。
2. Jinja chat template（读 `tokenizer_config.json`）——现在对任何非 ChatML 模型对话都是错的。
3. HTTP 参数补齐：`input_ids`、`logprobs`/`return_logprob`、`stop_token_ids`/`ignore_eos`、`min_p`/repetition/frequency/presence、`response_format`（先接 JSON schema 约束）、`tools` 透传 + `/parse_function_call`。
4. chunked prefill + 池满抢占，解除"长 prompt 卡住整批 decode"和"并发固定 1–8"。
5. DeepSeek 系（MLA + MoE + fp8 权重）——单个架构就能覆盖当前最主流的一批权重。

**P1（决定能不能当服务用）**
6. TP（≥2 卡）+ NCCL，或至少多进程 DP + cache-aware 前端；`/update_weights_from_disk|tensor` 进 RL 栈。
7. KV 分层：host offload + GPU 侧页级 radix（替换 16 segment lease）+ `/flush_cache` + 命中率报告。
8. fp8 权重/KV（E4M3，per-tensor + block-wise）、W4A16。
9. 约束解码用现成库（llguidance / xgrammar 都是 Rust/C++，可直接链进来，不必自研）。
10. `/metrics`（Prometheus）+ 请求级时延统计；CI 加 GPU smoke。

**P2（拉开体验差距，但不阻塞使用）**
11. EAGLE/Medusa/MTP 类 spec（需要配套 draft 权重；当前朴素 draft 实测净减速）。
12. 更多 VLM / embedding / rerank 模型，`/v1/embeddings` `/v1/rerank` `/score`。
13. 前端语言（`sglang.lang` 语义）——若目标是"替代原版写程序"，这项权重很高；若只做推理服务端可明确放弃。
14. ROCm/NPU 平台、PD 分离 + router。

**可明确不追**：多硬件平台的广度、Ray、`sgl-model-gateway` 的全套路由生态、训练相关子系统。
C 版的差异化价值在"零依赖单二进制 + 自写 kernel + 可控内存"，功能数量不该是目标。