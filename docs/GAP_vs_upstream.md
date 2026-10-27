# sglang-c vs 上游 sglang：能力差距盘点（2026-10-27 第二轮复核）

> 本版是**按当前 HEAD（`afa3269`）重新核过的清单**，不是旧表加批注。
> 上一版（同日早些时候）列的 P0/P1/P2 大部分已在 `4f03936` / `12011e8` / `afa3269` 三个提交里补掉，
> 那些项挪到第 2 节「已补齐」并附证据位置，剩下的才是真差距。
> 数字全部在仓库里现数（见第 5 节的命令），不引用任何旧数字。

对照对象：父目录 `../` 的 sgl-project/sglang。
被对照方：`sglang-c/`（C11 + 可选 CUDA）。

---

## 0. 量化基线

| 维度 | 上游 sglang | sglang-c |
| --- | --- | --- |
| 代码量 | `python/` 1,592,411 行（`srt/` 817,628 行）；另有 Rust router / gateway | `src`+`include`+`apps` 62 文件 **19,919 行**（约 1/80） |
| 模型实现 | `srt/models/` 302 文件（顶层 253 个 `*.py`，223 处 `EntryClass`） | **3 个后端**：gpt2 / llama 系（含 qwen2、qwen3、qwen3-next、mistral、gemma、gemma2）/ deepseek(V2V3)。GPU 侧只有 llama 系 1 个 |
| 层与算子 | `srt/layers/` 841 文件（量化 111） | `ops.c` `tensor.c` `quant.c` `mla.c` + `cuda_backend.cu` 一套自写 kernel |
| HTTP 路由 | `entrypoints/http_server.py` **86 个**路由装饰器（全 `entrypoints/` 92 个），另有 Ollama 动态路由 | **18 个**（含 `/server_info` 这类别名） |
| 调度 / 缓存 | `mem_cache/` 191 文件、`speculative/` 67 文件 | `scheduler.c` `radix.c` `spec.c` `beam.c` |
| 多模态 | `srt/multimodal/` 100 文件 + `multimodal_gen/` 1472 文件 | `clip.c` + `image.c`（LLaVA 单图） |
| 分布式 | `distributed/` 31、`disaggregation/` 38、`elastic_ep/`、`eplb/` 14 | **0** |
| 平台 | `platforms/` cuda / rocm / npu / xpu / mps / musa / cpu / sail… | x86-64 CPU、CUDA sm_75+、arm64 CPU（macOS 单线程） |
| 测试 | nightly 大模型回归 + `bench_serving` | 39 个 ctest 用例（`ctest -N` = 39，CPU 39/39、MSVC+CUDA 38/38 见 CHANGELOG） |

---

## 1. 结论

主干推理闭环（模型前向 → paged KV → radix 前缀复用 → continuous batching → OpenAI 兼容 HTTP）
在 CPU 与 CUDA 两条路径上都已跑通且做过数值验证（对 transformers 本体到 1e-7 量级）。
但离「替代原版跑生产负载」还差 **三块硬骨头 + 一层服务面**，其中任何一块都足以让它在真实场景里用不起来：

1. **权重格式太窄**：只有 safetensors（fp16/bf16/fp32/fp8）+ 自研 int8。
   GGUF / AWQ / GPTQ / compressed-tensors / W4A16 / MXFP4 / NVFP4 **全无**（源码里这些词零命中）。
   下载量最大的那批权重（gguf Q4_K_M、AWQ、GPTQ-Int4）现在**一个都打不开**。
2. **GPU 后端只覆盖标准 llama 系**。`src/cuda_backend.cu` 里没有任何 MoE / 专家路由、MLA、线性注意力的实现
   （`moe|expert` 零命中），`apps/cuda_http.c` 只调 `sgl_llama_load()`，不认 deepseek 后端。
   于是 DeepSeek-V2/V3、Qwen3-Next、所有 MoE 模型只能在 CPU 上跑 —— 而 CPU 单机吞吐当不了服务。
   （`12011e8`「GPU 服务路径对齐 CPU」对齐的是 chat 模板 / 约束解码 / logprobs / 监控端点，**不是模型架构**，
   很容易误读成 GPU 已经能跑 DeepSeek。）
3. **并行与分布式为零**：TP / PP / EP / DP-attention / NCCL / PD 分离全无（`tensor_parallel` 零命中）。
   单卡放不下的模型（70B 级）无法加载；多卡机器只能用上 1 张。

服务面（第 3 节 B 组）是「像不像原版」的问题，不是「能不能跑」的问题，优先级排在三块硬骨头之后。

---

## 2. 已补齐（不再算差距，列出来避免重复劳动）

| 能力 | 实现位置 | 验证 |
| --- | --- | --- |
| 分片 safetensors（`index.json` → `weight_map`，≤128 片） | `safetensors.c` `sgl_st_open_dir` | `test_st_index` 逐元素 |
| FP8 权重（E4M3/E5M2） | `safetensors.c` 加载时解到 f32 | `test_fp8` 15 种位型 |
| Jinja chat template（含 `tools` / `namespace` / 切片 / `is defined`） | `jinja.c` | 与 Python jinja2 3.1.6 **逐字节**一致（Qwen2.5/Qwen3/TinyLlama 真实模板） |
| 采样全量（`min_p`、三种 penalty、`ignore_eos`、`stop_token_ids`、`logprobs`/`top_logprobs`、`n`、`input_ids`） | `sampling.c`、`http.c`、`cuda_http.c` | 两侧引擎都已接 |
| chunked prefill（`SGL_PREFILL_CHUNK`，默认 512）+ 池满抢占重算 | `scheduler.c` | `n_preempt` 统计 |
| 约束解码（JSON Schema / regex → 字节级自动机，token 掩码） | `grammar.c` | `test_grammar` 213 断言；GPU 侧实测出 `{"name":"John Doe","age":30}` |
| DeepSeek-V2/V3 后端（MLA + MoE + 共享专家 + V3 路由） | `deepseek.c`、`mla.c` | 对 `DeepseekV2ForCausalLM` logits 2.5e-07（YaRN 3.0e-07） |
| YaRN RoPE（llama + deepseek，旧 `rope_scaling` 与新 `rope_parameters` 都读） | `llama.c`、`deepseek.c` | 频率表对 HF 1.5e-07 |
| BERT/RoBERTa embedding + rerank（`/v1/embeddings` `/v1/rerank` `/v1/score`） | `bert.c`、`wordpiece.c` | 对 `all-MiniLM-L6-v2` 真实权重 1.3e-07 |
| reasoning 分离 / `<tool_call>` → OpenAI `tool_calls` | `parser.c` | GPU 侧同样接上 |
| 服务面：`/ready` `/v1/models` `/get_server_info` `/metrics` `/flush_cache` `/abort_request` `/update_weights_from_disk` | `http.c`、`cuda_http.c` | GPU 侧 `/update_weights_from_disk` 明确 501，不假装成功 |
| 数据并行副本 `--dp N`（单进程内 N 份 KV 池 + radix，按负载路由） | `scheduler.c` | `test_dp` |
| 长 prompt 提前 400（不再给含糊 500） | `cuda_http.c` | `afa3269` |

---

## 3. 仍缺的（按「挡住什么」分组）

### A. 模型与权重 —— 挡住「能不能跑」

**A1 权重格式（P0，成本最低、解锁最多）**

缺：GGUF（含 K-quant）、AWQ、GPTQ、compressed-tensors、W4A16、MXFP4/NVFP4、legacy `.bin`、HF Hub 直连下载与权重缓存。
影响：上游 `layers/quantization/` 111 个文件对应十余种量化方案，C 版只有 W8A16（CPU）/ W8A8 dp4a+IMMA（CUDA）
和 fp8 权重解包。绝大多数人手上现成的量化模型加载即失败。
补法：GGUF / AWQ / GPTQ 只需**只读解码**（load 时 dequant 成 f16/f32，复用现有 GEMM），
不动 kernel，是投入产出比最高的一项。

**A2 GPU 只覆盖 llama 系（P0）**

缺：GPU 上的 MoE 专家路由、MLA、线性注意力（gated deltanet）、混合层。
证据：`src/cuda_backend.cu` 中 `moe` / `expert` 零命中；`apps/cuda_http.c` 只 `sgl_llama_load()`。
影响：GPU 服务实际只能吃「标准 MHA/GQA + SwiGLU 的 llama 系」checkpoint。
DeepSeek 系、Qwen3-Next、Mixtral、Qwen3-MoE 这类主流权重全部退回 CPU。
补法：先把 CPU 侧 `deepseek.c` 的前向（MLA + 分组路由）搬成 kernel，再补 Qwen3-Next 的线性注意力与 MoE。

**A3 架构覆盖面（P1）**

上游 253 个模型实现 vs 3 个后端。缺口里最常被点名的：
Mixtral / Qwen3-MoE / DBRX / gpt-oss（mxfp4）/ Llama-4 系、
Qwen-VL / InternVL / Kimi-VL / GLM-V / Ernie-VL（含视频、音频）、
Mamba 与各种线性/混合架构、dLLM（扩散式）、扩散生成（`multimodal_gen`）、whisper 语音。
影响：只能吃 Llama 同构 + DeepSeek 两类。

**A4 tokenizer 覆盖面（P1，成本低）**

现有：byte-level BPE（`bpe.c`，读 `tokenizer.json` 的 `model.vocab` + `model.merges`，含 LLaMA Metaspace）、WordPiece（`wordpiece.c`，BERT）。
缺：SentencePiece **Unigram**（`unigram` 零命中；`bpe.c` 注释自己写明「先支持 GPT-2 系 BPE」，Unigram 的 score 表根本不读）、added tokens 的完整语义、多模态 processor（图像切块/resize 策略）。
影响：T5 类、部分 ChatGLM/Baichuan 权重打不开。

**A5 dtype 直算（P2）**

权重统一转 fp32 驻留（除 CUDA 侧 fp16 路径），大模型内存翻倍，且 CPU 侧没有 bf16/f16 直算。

### B. 服务与调度 —— 挡住「当服务用」

**B1 路由 18 vs 86（P1）**

C 版现有：`/health` `/health_generate` `/ready` `/ping` `/generate` `/v1/{completions,chat/completions,models,embeddings,rerank,score}`
`/get_server_info` `/get_model_info` `/server_info` `/metrics` `/flush_cache` `/abort_request` `/update_weights_from_disk`。
缺的按类：
- **前端协议**：`/v1/responses`（OpenAI Responses API）、`/v1/messages` + `/v1/messages/count_tokens`（Anthropic）、
  Ollama（`/api/chat` `/api/generate` `/api/tags` `/api/show`）、gRPC、`/invocations`（SageMaker）、`/vertex_generate`。
- **分词**：`/tokenize` `/v1/tokenize` `/detokenize` `/v1/detokenize`。
- **分类/编码**：`/classify` `/v1/classify` `/encode` `/v1/decisions`。
- **生命周期**：`/pause_generation` `/continue_generation` `/slow_down`。
- **运维/诊断**：`/start_profile` `/stop_profile` `/set_trace_level` `/freeze_gc` `/weights_checker`
  `/release_memory_occupation` `/resume_memory_occupation` `/set_internal_state` `/dumper/{method}` `/configure_logging`。
- **LoRA 运行时**：`/load_lora_adapter` `/load_lora_adapter_from_tensors` `/unload_lora_adapter`
  （C 版只有启动时 `SGL_LORA` 预合并，不能运行中增删；且 CPU 路径对 alpha 不生效的 bug 仍未修）。
- **会话**：`/open_session` `/close_session`（`session` 零命中）。
- **KV 分层**：`/hicache/*`、`/add_external_corpus` `/remove_external_corpus` `/list_external_corpora`。
- **权重同步（RL 栈）**：`/init_weights_update_group` `/update_weights_from_tensor|distributed|ipc` `/update_weight_version`
  等一整组 —— 原版现在大量当 rollout 引擎用，这块缺失等于进不了 RL 栈。
- **音频**：`/v1/audio/transcriptions`。

**B2 GPU 并发生存方式（P1）**

GPU 服务是固定 `MAX_CONC 8` 的**整段租约**：段长 = `SGL_CUDA_CTX / conc`，前缀缓存是 16 个 segment lease，
不是页级 radix 树。CPU 侧倒是页级 radix + refcount + LRU。
影响：并发上限硬顶 8；上下文越长每段越短；长 prompt + 大并发时利用率差。
补法：把 CPU 侧 `radix.c` 的页表搬到设备显存，改成动态页分配（原版 `mem_cache/` 191 个文件干的就是这件事）。

**B3 排队与公平性（P1）**

无优先级队列、请求分池、DP attention、PP 调度（`priority` 零命中）。
原版有 `schedule_policy.py` + prefill delayer + `min_free_slots_delayer`。

**B4 KV 分层（P2 视场景）**

无 host/磁盘/Mooncake/NIXL 分层（`hicache` 零命中）。
注意：CPU 版 KV 池本身就是 host 内存，对它做 offload 无意义；这项只在 GPU 版上才有价值。

### C. 解码与性能

| 项 | 上游 | C 版 |
| --- | --- | --- |
| spec | EAGLE/EAGLE3、Medusa、MTP(nextn)、ngram、DP draft、DFlash、Uno…（67 文件） | 朴素 draft/verify（K≤8），接受率 13%（截层）～39%（llama-160m），**仍是净减速** |
| 图/编译 | torch.compile + piecewise CUDA graph + full graph + TBO | 只有 decode CUDA graph（B=1..conc），无 prefill graph、无编译层 |
| batch-invariant ops | 有（`batch_invariant_ops/`） | 无 → 同一请求不同 batch 组成下 bit 级不可复现 |
| 量化算子 | fp8 per-tensor/token/block、W4A16、Marlin、DeepGEMM、MXFP4/NVFP4、块状 int8（111 文件） | W8A16（CPU）、W8A8 dp4a/IMMA（CUDA）、int8 KV、fp16 权重 |
| attention kernel | FlashInfer、FA3/FA4、FlashMLA、TRT-LLM、CUTLASS… | 自写 flash + 两段式长上下文 + cuBLAS |
| KV 量化 | fp8/int8/fp4 KV | int8 KV |

`docs/BENCH.md` 里所有性能数字都是**自己 vs 自己**的 A/B，没有一条与原版 sglang / vLLM 的端到端对照。

### D. 工程与验证

- CI **不跑 CUDA、不跑真实大权重**（README 明写）；`bench_all.ps1` 是本机脚本。
- 无「与原版逐请求对拍」的 conformance 套件：模板、special tokens、stop 语义、partial 流式切分的边界处理
  全靠自己实现，缺一份和上游行为的对照。
- 39 个 ctest 用例 vs 上游 nightly 大模型回归，量级差两档。
- 无 Adapter/插件体系（`plugins/` `kernel_plugins/` `rust_extensions/`）：C 版加一种新模型要改 `model.c` 的分发。

### E. 明确不追（写清理由，避免被当成遗漏）

- 10 个硬件平台的广度（ROCm/NPU/XPU/MUSA/SAIL/Intel AMX）：C 版差异化价值在「零依赖单二进制 + 自写 kernel + 可控内存」，平台数量不是目标。
- Ray / `sgl-model-gateway` / 多实例缓存感知路由：需要单独的服务生态。
- 训练侧与 RL 训练框架本身（但 `update_weights_*` 这组 API 若想进 RL 栈，仍属 B1 必补）。
- `sglang.lang` 前端语言（gen/select/fork/join、RadixAttention 作为语言原语）：只在「替代原版写程序」时才有意义；若定位是推理服务端，可永久放弃。

---

## 4. 建议推进顺序（按 投入/解锁能力 比）

| 序 | 事项 | 为什么排这里 | 量级 |
| --- | --- | --- | --- |
| 1 | GGUF + AWQ/GPTQ 读取（load 时 dequant） | 纯加载侧改动，不动 kernel，直接解锁市面上大多数现成权重 | S–M |
| 2 | GPU 补 MoE + DeepSeek MLA kernel | 把 CPU 侧已验证的两套前向搬到 GPU，GPU 服务可用模型从 1 类变 3 类 | L |
| 3 | GPU 页级 radix + 动态并发（去掉 8 段租约） | 解除并发上限与段长耦合，长 prompt + 高并发下的利用率 | M–L |
| 4 | `/v1/tokenize` `/v1/detokenize` `/v1/responses` `/v1/messages` | 让 OpenAI/Anthropic SDK 能直连；分词端点还便于对拍 | S |
| 5 | `/pause_generation` `/continue_generation`、`/start_profile` `/stop_profile`、`/release_memory_occupation`、运行时 LoRA 增删 | 运维与 RL 集成的必备开关 | S–M |
| 6 | TP（≥2 卡）+ NCCL | 70B 级模型的唯一出路；成本最高，放最后 | L |
| 7 | conformance 对拍套件 + CI GPU smoke | 之后每加一项能力都要靠它兜底 | M |

---

## 5. 本文每个数字怎么核的

```powershell
# 上游规模
cd python/sglang; (Get-ChildItem -Recurse -File -Filter *.py | Get-Content | Measure-Object -Line).Lines
cd srt; (Get-ChildItem models -File -Filter *.py | Measure-Object).Count
(Get-ChildItem -Path layers,mem_cache,speculative,multimodal,parser -Recurse -File | Measure-Object).Count
(Select-String -Path entrypoints/http_server.py -Pattern '@app\.(get|post|api_route)\(').Count

# C 版规模与路由
cd sglang-c; (Get-ChildItem src,include,apps -Recurse -Include *.c,*.h,*.cu | Get-Content | Measure-Object -Line).Lines
Select-String -Path src/http.c,apps/cuda_http.c -Pattern '"/[a-zA-Z0-9_/\.\-]+"' -AllMatches |
  ForEach-Object { $_.Matches } | ForEach-Object { $_.Value } | Sort-Object -Unique

# 关键缺口（全部应输出 0）
Select-String -Path src/*.c,src/*.cu,apps/*.c -Pattern 'gguf|awq|gptq|w4a16|unigram|tensor_parallel|hicache|eagle|medusa|session|priority' -SimpleMatch
Select-String -Path src/cuda_backend.cu -Pattern 'moe|expert'
```