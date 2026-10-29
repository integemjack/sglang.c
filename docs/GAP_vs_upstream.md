# sglang-c vs 上游 sglang：能力差距盘点（2026-10-29 第三轮复核，HEAD `264ab89`）

> 本版按**当前 HEAD** 重新核过，对照 `../` 的 sgl-project/sglang（HEAD `ad950cdc`，2026-10-25）。
> 上一版（2026-10-27，HEAD `afa3269`）之后有 17 个提交，其中 8 个直接改变了清单：
> GPU 跑 qwen3_5 hybrid、MTP 接入 GPU 服务、GPU 并发 8→32、GPU int4 权重、GPU 量化分页 KV、
> `cuda_deepseek.cu`（MLA+MoE 的 GPU 前向）。这些从「缺」挪到第 2 节。
> 数字全部现数（第 5 节给了命令），不引用旧数字。

---

## 0. 量化基线

| 维度 | 上游 sglang（`ad950cdc`） | sglang-c（`264ab89`） |
| --- | --- | --- |
| 代码量 | `python/` 1,592,648 行（`srt/` 817,628 行）；另有 Rust router / gateway | `src`+`include`+`apps` **68 文件 27,395 行**（约 1/58） |
| 模型实现 | `srt/models/` 253 个顶层 `*.py`（223 处 `EntryClass`） | **3 个后端**：gpt2 / llama 系 / deepseek；llama 系内已含 Qwen3-Next 风格 MoE 与 GDN hybrid |
| 层与算子 | `srt/layers/` 841 文件（量化 111） | `ops.c` `tensor.c` `quant.c` `mla.c` `qweight.c` + 自写 CUDA kernel（`cuda_backend.cu` 6087 行、`cuda_deepseek.cu` 1045 行） |
| HTTP 路由 | `http_server.py` **78 个**（另有 Ollama 动态路由、gRPC bridge） | CPU **40+ 个**、GPU **50+ 个**（共用 `apps/accel_http.c` 的 `route_ext()`；另有 30 个无后端的端点明确 501） |
| 调度 / 缓存 | `mem_cache/` 191 文件、`speculative/` 67 文件 | `scheduler.c` `radix.c` `spec.c` `beam.c` `cuda_spec.c` |
| 多模态 | `srt/multimodal/` 100 文件 + `multimodal_gen/` 1472 文件 | `clip.c` + `image.c`（LLaVA 单图，CPU/GPU 都接了） |
| 分布式 | `distributed/` 31、`disaggregation/` 38、`elastic_ep/`、`eplb/` 14 | **0**（`tensor_parallel` 零命中） |
| 平台 | cuda / rocm / npu / xpu / mps / musa / cpu / sail… | x86-64 CPU、CUDA sm_75+、arm64 CPU（macOS 单线程） |
| 测试 | nightly 大模型回归 + `bench_serving` | 44 个 ctest（`ctest -N`），无 CUDA/大权重 CI |

---

## 1. 结论

和 10-27 那版比，三块硬骨头里 **权重格式、GPU 模型覆盖** 都已补完（含 GPU 侧 MoE 层前向与
DeepSeek 的 MLA+MoE 服务路径），**并行与分布式一字未动**。现在真正卡住「替代原版跑生产负载」的是：

1. **并行与分布式为零**：TP / PP / EP / NCCL / PD 分离全无。70B 级单卡放不下就加载不了，多卡机器只能用 1 张。
   `--dp N` 是单进程内 N 份 KV 池 + radix 的负载分摊，不是真正的 DP attention。
2. **服务面**：CPU 40+ / GPU 50+ 个端点（两边共用同一批形状与语义），剩下的 33 个是
   **明确 501 + 原因**（hicache / audio / reward / expert 分布 / `update_weights_from_tensor|distributed|ipc`
   整组 / Vertex / dumper / PD 角色切换）—— 这些不是「没写」，是本仓没有对应后端
   （分层缓存、reward head、NCCL transport）。CPU/GPU 逐字节对拍固化成 `tests/conformance.py`。
3. **没有和原版（Python sglang / vLLM）的对照基准**：`docs/BENCH.md` 全部是自己 vs 自己的 A/B。
   `DESIGN_1M_32C.md` 里的 264 tok/s 是**算出来的预测**，不是实测对比。
   CPU↔GPU 一致性有实测（`tests/conformance.py` 逐字节一致、`tests/batch_invariant.py` 4/4），
   但那证明的是「两边一样」，不是「和上游一样快」。
4. **权重格式只差冷门分支**：W8A8、FP8 blockwise、NVFP4 落盘、marlin 的 8bit 分支没做；
   真实主流落盘（AWQ / GPTQ v1+v2 / CT-W4A16 / MXFP4 / GGUF / FP8 / marlin ABI）都能打开。

主干推理闭环（模型前向 → paged KV → radix 前缀复用 → continuous batching → OpenAI 兼容 HTTP）在 CPU 上完整，
GPU 上是「llama 系 dense + GDN hybrid + MoE + DeepSeek MLA/MoE + 32 段租约 + MTP 投机」，
两条路径的缓存语义并不相同（见 B2）。

---

## 2. 自 10-27 起已补齐（不再算差距）

| 能力 | 提交 / 位置 | 验证 |
| --- | --- | --- |
| **GPU 跑 qwen3_5 hybrid**（GDN 线性注意力 + q_gate + qk-norm + partial/mrope rope，Qwen3.8-27B） | `19420ca`、`cuda_backend.cu` | 对 CPU f32 逐层对拍（`SGL_CUDA_DBG` vs `SGL_LLAMA_DBG`），输出质量修复前后可比 |
| **GDN 门控修正**（之前 α 与输入无关，所有 qwen3_5/Qwen3-Next 的 GPU 输出都是错的） | `7c13a8e` | 逐 token 计算 `h·W_ba^T`；27B 输出全变，decode 35.5 tok/s 不变 |
| **MTP（模型自带 MTP 头）投机解码接入 GPU 服务** | `f15ae1e`、`8612906`、`264ab89` | 贪心 W=4：**3.16 token/步、53–54 tok/s（off 35.5）= 1.5×**；MTP 自有 KV 按段回填（草稿接受率 0.82/0.74/0.71/0.76）；强制错误草稿与正常分叉点一致 |
| **GPU 并发 8 → 32**（batch GDN decode + 每请求一个线性注意力状态槽） | `bbd168d` | `MAX_CONC 32`，`ACTIVE_CONC` 默认 8、`SGL_HTTP_CONC` 可调；hybrid B>1 不再逐层串行 |
| **GPU int4 权重路径（NVFP4 码书，e2m1 + 16 元素一块缩放）** | `3376a3d` | `SGL_W4=1`；实测有损，默认关 |
| **GPU 量化分页 KV（K/V 分块 scale，页 64 位置）** | `cuda_backend.cu` `sgl_kvq` | `SGL_KVQ=1`；262K 上下文仍未达到 M2 目标，默认关 |
| **DeepSeek MLA + MoE 的 GPU 前向** | `19420ca`、`cuda_deepseek.cu` | 对 CPU 引擎（本身对 numpy 1e-7）< 2e-4；覆盖 sigmoid/softmax 打分、`e_score_correction_bias`、group_limited、topk_group、norm_topk、routed_scale、共享专家；**未接服务** |
| **CPU llama 后端支持 Qwen3-Next 风格 MoE** | `llama.c`（`experts.gate_up_proj` 融合 + 共享专家 + `decoder_sparse_step` + `mlp_only_layers`） | `test_family_qwen3next`（44 个 ctest 内） |
| GGUF（v2/v3、12 种量化、内嵌 tokenizer、q/k 逆置换） | `gguf.c`（1262 行） | 对 numpy 独立实现 96 张量 rel_rms = 0；3 个真实 gguf 端到端正确 |
| AWQ（gemm/gemv、zero_point）/ GPTQ（desc_act、g_idx、4/8bit） | `qweight.c` | 真实检查点切片 rel_rms = 0；三套夹具端到端 5e-8 |
| 分片 safetensors、FP8 权重、Jinja 模板、约束解码（json_schema/regex/structural_tag）、采样全量、chunked prefill、抢占、BERT embedding/rerank、YaRN、reasoning/tool_calls | 见 CHANGELOG | 各有 ctest |
| `/v1/completions` 流式改成 OpenAI `text_completion` 形状；`k_rope_p` 越界修复 | `fc42d32` | 流式块为 `choices[0].text`；`SGL_PREFILL_CHUNK=1` 不再 illegal access |
| **CPU 服务面补齐**：`/tokenize` `/v1/tokenize` `/detokenize` `/v1/detokenize` `/encode`、`/v1/messages/count_tokens`、`/v1/responses`、`/v1/messages`（含 SSE 流式）、Ollama 四个、`/invocations`、`/pause_generation` `/continue_generation` `/slow_down`、`/start_profile` `/stop_profile`、`/release_memory_occupation` `/resume_memory_occupation`、`/open_session` `/close_session`、`/load_lora_adapter` `/unload_lora_adapter`、`/model_info`、`/separate_reasoning` `/parse_function_call` | `http.c` `handle_extended()` | `tests/ext_endpoints_test.py` 37 项全过（起服务实测） |
| **优先级队列**（`priority` 字段；同优先级 FIFO；抢占先抢低优先级） | `scheduler.c` `submit_prio` / `step` / `preempt_one` | `/generate` 带 `priority` 实测接受；ctest 45 全绿 |
| **Unigram（SentencePiece）tokenizer**（T5 类可打开） | `bpe.c`（style=2：Viterbi + 字符边界 + `<0xXX>`/`<unk>` 回退） | `tests/gen_unigram_ref.py` 独立实现生成参考；`test_unigram` 7 case 字符级一致（T5 真实词表 32100） |
| **运行时卸载 LoRA / 暂停继续 / 慢速** | `scheduler.c` `remove_lora` `set_paused`；`http.c` `step_delay_ms` | 实测 load→route→unload、pause/continue |
| **33 个无后端的上游端点明确 501**（不假装成功） | `http.c` `ni[]` 表 / `accel_http.c` `g_ni[]` 表 | 实测 6 类均 501 + 原因 |
| **GPU 服务面补齐**（15 → 50+，与 CPU 同一套形状）：tokenize/detokenize/encode、responses/messages（真流式）、Ollama、invocations、pause/continue/slow_down、profile、release/resume、session、model_info、separate_reasoning/parse_function_call | `accel_http.c` `route_ext()` | `tests/ext_endpoints_test.py 30112` 全过；`tests/conformance.py` CPU vs GPU 逐字节一致 |
| **GPU 侧 BERT embedding 模式**（同一二进制服务 embedding/rerank/score，不需设备） | `accel_http.c` `dir_is_encoder()` | `bert-rand` 实测：embedding 对 numpy 参考 max|diff| 5e-07，rerank 分数一致 |
| **DeepSeek（MLA+MoE）GPU 服务路径** | `accel.h` `ds_*`（ABI 2）、`cuda_http.c`、`accel_http.c` `ds_gen()`/`route_ds()` | `cuda_deepseek_test` 贪婪 16 步 token 序列一致；服务层 6 prompt × 2 长度 = 12/12 与 CPU 文本相同 |
| **GPU 侧 MoE 层前向**（route/top-k/norm、每专家 gate_up+silu*mul、down 累加、共享专家 sigmoid 门控） | `cuda_backend.cu` `k_moe_*` + `moe_forward()`（四处接入） | `cuda_moe_test` 4 组形状 GPU vs CPU maxerr ≤ 7.2e-09；`qwen3next-rand` CPU vs GPU 服务逐字节一致 |

---

## 3. 仍缺的（按「挡住什么」分组）

### A. 模型与权重 —— 挡住「能不能跑」

**A1 权重格式（已补齐主流，剩冷门分支）**

已支持：safetensors（fp16/bf16/fp32/fp8）、GGUF（12 种量化）、AWQ（gemm/gemv）、
GPTQ（desc_act / g_idx、4/8bit、**gptq_v2**）、**compressed-tensors W4A16/W8A16**、**MXFP4（gpt-oss）**、
**marlin 布局（4bit 与 8bit，AWQ/GPTQ 的 marlin ABI）**、自研 int8/W8、GPU int4。
每种都有夹具端到端对照（参考由独立 numpy 实现按官方落盘语义生成）：
`test_family_qw_{awq,gptq,gptq_act,gptq_v2,ct4,mxfp4,marlin,marlin8}`。

marlin 的布局靠独立实现钉死：4bit 是 **Marlin 作者官方 Python `Layer.pack`** +
**vLLM `awq_marlin_repack.cu`** 两方；8bit 用 vLLM 的**两份独立 kernel**
（`awq_marlin_repack.cu` 从 [K, N/pack] 重排、`gptq_marlin_repack.cu` 从 [K/pack, N] 重排）
互相对账再加我们的闭式索引，`tests/marlin_layout.py` 对 4 种形状逐字节相同
（Marlin 官方 Python 只做 4bit，所以 8bit 没有第三方参考）。两者都含 `marlin_permute_scales`
的 64 列置换与 `awq_to_marlin_zero_points` 的零点重排（8bit 的零点 interleave 是 [0,2,1,3]）。
注意 marlin 不是落盘格式（是 vLLM 加载期布局），本仓支持它是为了吃 vLLM 产出的 ABI 张量、
以及让 C 的解包路径能被端到端验证。

仍缺：
- W8A8、FP8 blockwise、NVFP4 落盘（GPU 侧只有内部 e2m1 量化）、marlin 的 `is_a_8bit`
  （W4A8：**激活** 8bit，marlin 的 tile 变成 32(k)x32(n)，跟权重的 4/8bit 是两回事）；
- **HF Hub 直连下载与权重缓存**：只认本地目录；
- CPU 侧权重统一转 f32 驻留（无 bf16/f16 直算），大模型内存约 2 倍。

**A2 GPU 模型覆盖（MoE 与 DeepSeek 已补齐）**

- `cuda_backend.cu` 已有 MoE 层前向（`k_moe_route` / `k_moe_gu` / `k_moe_down` + 共享专家三件套，
  四处前向路径接入），Qwen3-MoE / Qwen3-Next / Mixtral 在 GPU 上可跑；
  `cuda_moe_test` 4 组形状 GPU vs CPU maxerr ≤ 7.2e-09，`qwen3next-rand` CPU vs GPU 服务逐字节一致。
- `apps/cuda_http.c` 已按 `deepseek_detect()` 分发到 `sgl_cuda_deepseek_*`（MLA 吸收式 + MoE gather/scatter），
  服务层 6 prompt × 2 长度 = 12/12 与 CPU 文本相同。
- 仍缺：TP/PP/EP（见结论 1）、多模态的 GPU 侧、CUDA graph 覆盖不到 MoE 分支。


**A3 架构覆盖面（P1）**

上游 253 个模型实现 vs 3 个后端。缺口里最常被点名的：
- VLM：Qwen-VL / InternVL / Kimi-VL / GLM-V / Ernie-VL（含视频、音频）；
- MoE 的非 Qwen3-Next 布局（Mixtral / DBRX / gpt-oss 的 mxfp4）；
- Llama-4 系、Mamba（非 GDN）、各类线性/混合架构；
- dLLM（扩散式语言模型）、`multimodal_gen`（1472 文件）、whisper 语音。

**A4 tokenizer 覆盖面（已补齐 Unigram，剩归一化差异）**

现有 byte-level BPE（`bpe.c`）+ WordPiece（`wordpiece.c`）+ GGUF 内嵌词表 + **Unigram（SentencePiece）**：
T5/部分 ChatGLM 类现在能打开（`test_unigram` 对 T5 真实词表 32100 个 piece 字符级一致）。
仍缺：T5 的 `Precompiled`（NFKC）归一化（本版只做空格→`▁` + 段首 `▁`，全角/组合字符会与 HF 不同）、
added tokens 的完整语义、多模态 processor。

**A5 多模态深度（多图已补齐，剩视频/动态分辨率）**

`clip.c` + `image.c` 支持 LLaVA。**多图 2026-10-29 已补齐（G21）**：一个 prompt 里多个 `<image>`
按出现顺序各插自己的视觉段（图片数与占位符数不匹配时退回旧行为）。
验证两条：`test_llava_multi`（手工拼 emb 参考与调度器结果首 token 一致；两图 vs 单图 logits 差 0.178）
与 `tests/llava_multi_check.py`（起服务：红/蓝 vs 红/红 输出不同，三图不崩）。
仍缺：视频/音频、Qwen-VL 式动态分辨率切块、多模态 processor。

### B. 服务与调度 —— 挡住「当服务用」

**B1 路由：CPU 已到 40+（对齐上游主要面），GPU 仍 15（P1）**

CPU 版（`handle_extended()`）已实现：`/tokenize` `/v1/tokenize` `/detokenize` `/v1/detokenize` `/encode`、
`/v1/messages/count_tokens`、`/v1/responses`、`/v1/messages`（SSE 流式）、Ollama 四个、`/invocations`、
`/pause_generation` `/continue_generation` `/slow_down`、`/start_profile` `/stop_profile`、
`/release_memory_occupation` `/resume_memory_occupation`、`/open_session` `/close_session`、
`/load_lora_adapter` `/unload_lora_adapter`、`/model_info`、`/separate_reasoning` `/parse_function_call`；
另外 33 个无后端支撑的端点明确 501（不假装成功）。

**GPU 侧（`apps/cuda_http.c`）仍只有 15 个**：缺 tokenize/detokenize、responses/messages、Ollama、
invocations、pause/continue/slow_down、profile、release/resume、session、运行时 LoRA，
以及 `/v1/embeddings` `/v1/rerank` `/v1/score`（GPU 服务没加载 BERT，但 BERT 是纯 CPU 实现，
接进来即可）。

仍属无后端、永久 501 的：`/hicache/*`、external corpus、`/v1/audio/transcriptions`、
`/v1/classify` `/v1/decisions`（无 reward head）、expert distribution、`/pd_role_switch`、
`/update_weights_from_tensor|distributed|ipc` 与 init/destroy group、`/vertex_generate`。

缺的按类：
- **前端协议**：`/v1/responses`、`/v1/messages` + `/v1/messages/count_tokens`（Anthropic）、
  Ollama（`/api/chat` `/api/generate` `/api/tags` `/api/show`）、gRPC、`/invocations`（SageMaker）、`/vertex_generate`。
- **分词**：`/tokenize` `/v1/tokenize` `/detokenize` `/v1/detokenize`（对拍也靠它）。
- **分类/编码**：`/classify` `/v1/classify` `/encode` `/v1/decisions`。
- **生命周期**：`/pause_generation` `/continue_generation` `/slow_down`。
- **运维/诊断**：`/start_profile` `/stop_profile` `/set_trace_level` `/freeze_gc` `/weights_checker`
  `/release_memory_occupation` `/resume_memory_occupation` `/set_internal_state` `/dumper/{method}`
  `/configure_logging` `/model_info` `/parse_function_call` `/separate_reasoning` `/get_weights_by_name`
  `/start|stop_expert_distribution_record` `/dump_expert_distribution_record`。
- **LoRA 运行时**：`/load_lora_adapter` `/load_lora_adapter_from_tensors` `/unload_lora_adapter`
  （C 版只有启动时 `--lora` 预合并；alpha 缩放 10-27 后已按 `alpha/r` 生效）。
- **会话**：`/open_session` `/close_session`（`session` 零命中）。
- **KV 分层**：`/hicache/*`、`/add_external_corpus` `/remove_external_corpus` `/list_external_corpora`。
- **权重同步（RL 栈）**：`/init_weights_update_group` `/destroy_weights_update_group`
  `/begin_weight_update` `/end_weight_update` `/update_weights_from_tensor|distributed|ipc`
  `/update_weight_version`、remote-instance 传输组 —— 缺了进不了 RL 栈。
- **音频**：`/v1/audio/transcriptions`。

**B2 GPU 的 KV / 缓存语义（P1）**

GPU 侧现在是**固定 32 段的整段租约**（`MAX_CONC 32`，段长 = `SGL_CUDA_CTX / 32`，`ACTIVE_CONC` 默认 8），
前缀缓存是 **16 个段槽（`PFX_MAX 16`）、token 级精确比较**，不是页级 radix 树；
KV 默认仍是连续 FP32 池，`SGL_KVQ=1` 才切到分页量化池（M2，默认关）。
影响：并发上限硬顶 32、上下文越长每段越短、无 COW 共享、无抢占/驱逐策略。
CPU 侧倒是完整的页级 radix + refcount + LRU + chunked prefill + 抢占。
补法：把 `radix.c` 的页表语义搬到设备显存（`DESIGN_1M_32C.md` §5.1/§7 已有设计）。

**B3 排队与公平性（CPU 已补，GPU 无）**

CPU 调度器现在有优先级：`sgl_sched_submit_prio`（HTTP body 的 `priority` 字段），等待队列按优先级取
（同优先级 FIFO），抢占先抢优先级最低、生成最少的。仍未做：prefill delayer、`min_free_slots_delayer`、
请求分池（上游 `schedule_policy.py` 里那套）。GPU 服务的段租约调度器仍然没有优先级。

**B4 KV 分层（P2 视场景）**

`hicache` / Mooncake / NIXL 全无。GPU 版才有价值（CPU 的 KV 池本身就是 host 内存）。

### C. 解码与性能

| 项 | 上游 | C 版 |
| --- | --- | --- |
| spec | EAGLE/EAGLE3、Medusa、MTP(nextn)、ngram、DP draft、DFlash、Uno…（67 文件） | **模型自带 MTP 头（GPU，W=4，3.16 token/步、1.5×，只对单请求贪心）** + 朴素 draft/verify（K≤8，接受率 13%–39%，仍是净减速） |
| 图/编译 | torch.compile + piecewise/full CUDA graph + TBO | decode CUDA graph（B=1..conc）；无 prefill graph、无编译层 |
| batch-invariant ops | 有（`batch_invariant_ops/`） | 无需专门实现：每行前向独立、chunked prefill 逐 token 叠 KV。`tests/batch_invariant.py` 4/4 实测（单独 vs 并发 vs 重发 vs 分块混批，逐字节一致） |
| 量化算子 | fp8 per-tensor/token/block、W4A16、Marlin、DeepGEMM、MXFP4/NVFP4、块状 int8（111 文件） | W8A16（CPU）、W8A8 dp4a/IMMA（CUDA）、int8 KV（CPU）、fp16 权重、NVFP4 风格 int4（默认关）、K4V3 KV（默认关）；**落盘格式已支持 AWQ/GPTQ(v1+v2)/CT-W4A16/MXFP4/GGUF/FP8，以及 marlin ABI（4bit 与 8bit）** |
| attention kernel | FlashInfer、FA3/FA4、FlashMLA、TRT-LLM、CUTLASS | 自写 flash + 两段式长上下文 + cuBLAS |
| MTP 的边界 | nextn 支持批内多序列、与 EAGLE 混用 | 仅贪心、单请求（`SGL_SPEC_W=4`）；无 MTP 头的模型无收益 |

`docs/BENCH.md` 里所有性能数字都是**自己 vs 自己**的 A/B，没有一条与原版 sglang / vLLM 的端到端对照；
`docs/DESIGN_1M_32C.md` 的 264 tok/s 是带宽模型推算的预测值。

### D. 工程与验证

- CI **不跑 CUDA、不跑真实大权重**（README 明写）；44 个 ctest 全在 CPU 侧，GPU 只靠本地脚本。
- 无「与原版逐请求对拍」的 conformance 套件：模板、special tokens、stop 语义、流式切分边界全靠自己实现
  （Jinja 已做到与 jinja2 3.1.6 逐字节一致，但整体行为没有对照）。
- 无 Adapter/插件体系（`plugins/` `kernel_plugins/` `rust_extensions/`）：C 版加一种新模型要改
  `model.c` 的分发和对应后端。
- 无 Python 包 / `sglang.lang` 前端 / Ray / `sgl-model-gateway`。
- 平台广度：无 ROCm/NPU/XPU/MUSA/SAIL/Intel AMX（差异化定位是「零依赖单二进制 + 自写 kernel」，可不追）。

### E. 明确不追（写清理由，避免被当成遗漏）

- 10 个硬件平台的广度：C 版价值在可控内存与零依赖，不在平台数量。
- Ray / gateway / 多实例缓存感知路由：需要单独的服务生态。
- 训练侧与 RL 训练框架本身（但 `update_weights_*` 这组 API 若想进 RL 栈，属 B1 必补）。
- `sglang.lang` 前端语言（gen/select/fork/join）：只在「替代原版写程序」时有意义。

---

## 4. C 版有、上游没有的（差异的另一半）

- **单可执行文件**：`sglc` 一个二进制含 CPU 引擎 + CUDA 引擎 + HTTP 服务（`--device auto` 自动选），
  零 Python、零运行时依赖；上游要 pip 一整套 torch。
- **CPU 路径是完整实现**：页级 radix + int8 KV + 连续批处理 + chunked prefill + 抢占 + W8A16，
  能在无 GPU 机器上真跑（上游 CPU 后端能力有限）。
- **直接读 GGUF**：容器 + 12 种量化 + 内嵌 tokenizer + 从元数据合成 config，一个 `.gguf` 文件就能起服务；
  上游要另走 llama.cpp 或转换。
- **自写 kernel**：不依赖 FlashInfer / CUTLASS / TRT-LLM，Windows + MSVC 原生构建（上游基本 Linux/CUDA only）。
- **arm64 CPU（macOS）构建**在 CI 里（单线程）。

---

## 5. 建议推进顺序（按 投入/解锁能力 比）

| 序 | 事项 | 为什么排这里 | 量级 |
| --- | --- | --- | --- |
| 1 | **把 `cuda_deepseek.cu` 接进 `cuda_http`**（加载分发 + 前向/采样/流式对接） | 代码已写好并验证过，只差接线；GPU 解锁 DeepSeek-V2/V3 整个家族 | S–M |
| 2 | **GPU 补 MoE 层前向**（Qwen3-Next-80B / Qwen3-MoE / Mixtral 的 fused `experts.gate_up_proj`） | 复用第 1 项的 gather/scatter + 每专家 GEMM；MoE 是当前主流权重形态 | M |
| 3 | `/v1/tokenize` `/v1/detokenize` `/v1/responses` `/v1/messages` | OpenAI/Anthropic SDK 直连；分词端点还是对拍工具 | S |
| 4 | `/pause_generation` `/continue_generation`、`/start_profile` `/stop_profile`、`/release_memory_occupation`、运行时 LoRA 增删 | 运维与 RL 集成的必备开关 | S–M |
| 5 | **GPU 页级 radix + 动态并发**（去掉 32 段租约） | 解除并发上限与段长耦合，长 prompt + 高并发下的利用率 | M–L |
| 6 | TP（≥2 卡）+ NCCL | 70B 级模型的唯一出路；成本最高，放最后 | L |
| 7 | conformance 对拍套件 + CI GPU smoke + 与原版/vLLM 的端到端基准 | 之后每加一项能力都要靠它兜底；也是现在最缺的「证据」 | M |

---

## 6. 本文每个数字怎么核的

```powershell
# 上游（对照方）——在 works/sglang 根目录
$txt = Get-Content python/sglang/srt/entrypoints/http_server.py -Raw
$ms = [regex]::Matches($txt, '(?s)@app\.(?:get|post|api_route)\([^\)]{0,200}?"(/[A-Za-z0-9_/\.\-]+)"')
($ms | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique).Count      # 78
(Get-ChildItem python -Recurse -File -Filter *.py | Get-Content | Measure-Object -Line).Lines   # 1592648
(Get-ChildItem python/sglang/srt/models -File -Filter *.py).Count              # 253
(Get-ChildItem python/sglang/srt/mem_cache -Recurse -File).Count               # 191
(Get-ChildItem python/sglang/srt/speculative -Recurse -File).Count             # 67

# C 版——在 sglang-c
(Get-ChildItem src,include,apps -Recurse -Include *.c,*.h,*.cu | Get-Content | Measure-Object -Line).Lines  # 27395
(Get-ChildItem src,include,apps -Recurse -Include *.c,*.h,*.cu).Count                                        # 68
Select-String -Path src/http.c -Pattern '"/[a-zA-Z0-9_/\.\-]+"' -AllMatches |
  ForEach-Object { $_.Matches.Value } | Sort-Object -Unique                     # 18
Select-String -Path apps/cuda_http.c -Pattern '"/[a-zA-Z0-9_/\.\-]+"' -AllMatches |
  ForEach-Object { $_.Matches.Value } | Sort-Object -Unique                     # 15
ctest --test-dir build -N                                                       # Total Tests: 44

# 关键缺口（全部应输出 0）
foreach($k in 'is_moe','router','shared_expert','exp_gate_up','n_experts') {
  (Select-String -Path src/cuda_backend.cu -Pattern $k -SimpleMatch | Measure-Object).Count }
foreach($k in 'tensor_parallel','hicache','unigram','priority','session','eagle','medusa',
              'mxfp4','w4a16','compressed-tensors','/v1/responses','/v1/messages',
              '/tokenize','/pause_generation','/start_profile','release_memory_occupation',
              '/load_lora_adapter','mooncake','nixl','flashinfer','cutlass') {
  (Select-String -Path src/*.c,src/*.cu,apps/*.c -Pattern $k -SimpleMatch | Measure-Object).Count }
```