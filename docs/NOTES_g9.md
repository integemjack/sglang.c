# NOTES_g9 — 模型族补完（CPU 路径）

日期：2026-10-26。任务：Gemma 的 rope_scaling / embedding softcap；Qwen3-Next 的
layer_types 混合层（linear_attention = gated deltanet）。构建目录 build-g9（独立，未动 build/、build-cuda/）。

## 1. Gemma：rope_scaling + hidden_softcap + layer_types

**C 侧（src/llama.c / include/sgl/llama.h）**
- `hidden_softcap`：embedding 查表后先 `emb = cap*tanh(emb/cap)`，再乘 sqrt(E) 的
  gemma scale（Gemma-3 顺序）。0 = 关闭，行为与旧版一致。
- `rope_scaling`：支持 `rope_type="linear"`（角度除以 factor，`rope_pos_scale=1/factor`）
  与 `"dynamic"`（静态 NTK：`theta' = theta * factor^(D/(D-2))`）。注意：HF 的
  dynamic_rope_update 是按运行时序列长度插值的另一套公式，这里取静态 NTK 形式，
  numpy 参考按同一公式，随机模型对照自洽；接真实 Gemma-2 权重时需再核对。
- `rope_scaling.global_base_freq` + `rope_local_base_freq`（Gemma-3 风格）：配合
  `layer_types`，sliding_attention 层用局部基频、full_attention 层用全局基频。
- `layer_types`：`["full_attention","sliding_attention",...]` 每层窗口；无该字段时
  全局 `sliding_window` 对所有层生效（mistral 旧行为不变）。
- 每层 rope 逆频率在加载时预计算（`layer_invf`），公式与 `sgl_rope_inplace` 同为
  double pow/cos/sin，无 scaling 时逐位一致 → 既有 family 测试数值不变。
- 不认识的 config 字段（含 layer_types 取值/长度不符）保持宽松忽略。

**随机模型与参考**
- `tests/gen_random_gemma.py` 扩展：gemma-rand 现带 `hidden_softcap=30`、
  `rope_scaling={linear, factor=2, global_base_freq=1e6}`、`rope_local_base_freq=1e4`、
  `sliding_window=8`、`layer_types=[sliding, full]`（权重 seed=42 不变，仅 config 变）。
- 变体 `dynamic` → `models/gemma2-rand`（`hidden_softcap=50`、dynamic NTK、无 layer_types）。
- `tests/ref_llama.py` 同步扩展（softcap / 每层基频 / 每层 mask / 位置缩放），
  重新生成 `tests/_ref_gemma.json`、新增 `tests/_ref_gemma2.json`。

**对照数字（test_llama，阈值 5e-3）**
- test_family_gemma：maxerr ≤ 7.2e-08（4 个 case：7.17e-8 / 5.22e-8 / 2.75e-8 / 4.29e-8），
  greedy6 增量 decode 一致、decode_batch B=1/2 一致。
- test_family_gemma2：maxerr ≤ 7.95e-08。

## 2. Qwen3-Next：layer_types 混合 linear_attention（gated deltanet）

**公式来源（已核实）**：HF transformers
`src/transformers/models/qwen3_next/modeling_qwen3_next.py` 的 Qwen3NextGatedDeltaNet
+ `torch_recurrent_gated_delta_rule`（https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py），
真实 config 参考 https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json
（layer_types 由 full_attention_interval=4 生成：每 4 层 1 个 full，其余 linear）。

**C 侧实现**（`lin_attn_step`，src/llama.c，CPU 单线程逐步递归）：
1. `in_proj_qkvz`/`in_proj_ba` 按 k-head 组交错布局 [q|k|v|z]、[b|a]（同 HF
   fix_query_key_value_ordering）；
2. 拼 mixed_qkv 后做因果 depthwise conv1d（核 `linear_conv_kernel_dim`，零历史）+ SiLU；
3. `beta=sigmoid(b)`，`g=-exp(A_log)*softplus(a+dt_bias)`，`alpha=exp(g)`；
4. q/k l2norm（FLA 式 `x/sqrt(sum x²+1e-6)`，q 再除 `sqrt(kd)`）；
5. 逐 token 递归：`S*=alpha; delta=(v-Sᵀk)·beta; S+=k⊗delta; out=Sᵀq`；
6. gated RMSNorm（先 norm 后 `*silu(z)`，eps=rms_norm_eps）→ `out_proj`。
linear 层不占 KV 页；递归状态按（层，序列首 KV 页号）归属槽位（默认 64 槽，
`SGL_LIN_SLOTS` 可调），位置不连续自动清零（prefill 冷启动、增量 decode 续用）。
权重名与 HF 检查点对齐；conv1d.weight 同时接受 [c,1,K]（真实布局）与 [c,K]。

**随机模型与参考**
- 新增 `tests/gen_random_qwen3next.py` → `models/qwen3next-rand`：L=4（3×linear +
  1×full）、E=64、nk=2/nh=4/kd=vd=16/conv=4，A_log~log U(0.01,16)、dt_bias=1（同 HF 初始化）。
- 新增 `tests/ref_qwen3next.py`：独立 numpy（float64）按同一 HF 公式实现，
  生成 `tests/_ref_qwen3next.json`。
- CMakeLists 注册段扩为 `foreach(fam mistral gemma gemma2 qwen3 qwen3next)`。

**对照数字**
- test_family_qwen3next：prefill top5 maxerr ≤ 6.8e-08（6.61e-8 / 4.71e-8 / 6.76e-8 /
  5.71e-8），greedy6（增量 decode 续用递归状态）与 numpy 全量重算一致，
  decode_batch B=2 一致。

## 3. ctest 结果

- 基线（改动前，build-g9）：`100% tests passed out of 20`。
- 改动后全量：21/22 通过；唯一失败 `test_family_qwen3` 为外部任务（GPU 路径
  long_vcpu.exe 等）挤占内存导致的 `alloc.c:28 out of memory`（0xc0000409），
  非数值失败；单独重跑 `100% tests passed out of 1`（106s）。
  期间另有 test_kv8/test_spec/test_w8/test_tinyllama 也因同样 OOM 失败过，
  单独重跑全部 Passed。即 **22 项（20 原有 + gemma2 + qwen3next）全部验证通过**。
- test_family_qwen3（真实 qwen3-06 + 旧参考 _ref_qwen3.json）maxerr 不变
  （Hello world 2.71501e-06 < 5e-3），证明 rope 重构逐位等价。

## 4. 与真实模型的差距（G13 已关闭，见第 5 节）

- ~~full_attention 层是简化版~~ → **G13 已补齐**：q 门控、partial rotary、(1+w) 归一化、
  MoE（专家 + 共享专家 + routed scaling）都已实现并 vs numpy 验证。
- dynamic rope 取静态 NTK 形式，非 HF 的 dynamic_rope_update 逐步插值。
- ~~递归状态按序列首页归属~~ → **G13 已改成按序列 key 分配**（见第 5 节）。
- CUDA 路径未动（cuda_backend.cu 不属于本任务）。

## 5. G13：Qwen3-Next full 层真实特性 + 递归状态按序列（2026-10-27）

公式来源仍是 HF `modeling_qwen3_next.py`（已下载核对：Qwen3NextAttention /
Qwen3NextRMSNorm / Qwen3NextSparseMoeBlock / Qwen3NextTopKRouter / Qwen3NextExperts）
与真实 80B config.json（`partial_rotary_factor=0.25` 与 `rope_theta` 在**顶层**，
没有 `layer_types`，改由 `full_attention_interval=4` 生成）。

### 实现（src/llama.c + src/ops.c）

| 特性 | 实现 | 验证 |
|---|---|---|
| a) q 门控 | `q_proj` 宽 2*q_dim，按头交错 `[q(D)\ | gate(D)]`（同 HF chunk），attn 输出逐元素 ×sigmoid(gate)（o_proj 之前） | 随机模型 vs numpy |
| b) partial rotary | `rot_dim = head_dim*partial_rotary_factor`，只旋转每头前 rot_dim 维，
配对 (i, i+rot_dim/2)，逆频率分母用 rot_dim；每层预算 `rot_invf` | 同上 |
| c) (1+w) 归一化 | `sgl_rmsnorm_p1()`：所有 layernorm / q_norm / k_norm / final_norm 都走它
（模型级开关 `norm_p1`，只有 qwen3_next 开）；linear 层的 gated RMSNorm 仍是普通权重 | 同上 |
| d) MoE | 层选择同 HF（`(l+1)%decoder_sparse_step==0` 且不在 `mlp_only_layers`）；
router softmax→top-k→可选 `norm_topk_prob` 重归一化；专家 `gate_up_proj`（门/上沿宽切）
→silu×up→`down_proj`，权重 ×(topk 权重)×`routed_scaling_factor`；再叠共享专家
（`shared_expert_{gate,up,down}_proj` × sigmoid(`shared_expert_gate`)） | 同上 |
| config 兼容 | `layer_types` 或 `full_attention_interval` 都支持；`partial_rotary_factor`
在 `rope_parameters` 或顶层都读；`routed_scaling_factor` 缺省（真实 config 为 null）→ 1.0 | 同上 |

新增 API（include/sgl/llama.h）：`sgl_llama_forward_seq` / `sgl_llama_decode_batch_seq`
（显式 64 位序列 key）、`sgl_llama_lin_release`（归还递归槽）、`sgl_llama_lin_slots_in_use`。

### 递归状态按序列分配（替代“首 KV 页”归属）

旧实现 `lin_owner[slot] = slot_idx[0]`：radix 前缀共享时两条序列共用首页 → 抢同一个槽。
现在归属键是 64 位：显式 `seq_id`，不传则退化为 `slot_idx` 数组地址（每序列一份页表）。
位置不连续仍自动清零；新增 `sgl_llama_lin_release` 归还槽位（不归还的话 64 个槽会耗尽）。

**验收：`tests/test_lin_seq.c`（新 ctest：test_lin_seq）** —— 两条序列共享前 8 个物理页
（首页相同）、后缀不同：

- 并发 decode 4 步，两条序列每一步 logits 与「各自单独跑」**逐位相同**（maxerr = 0）；
- 槽位占用：单独跑 A/B 后 = 2，共享前缀的两条序列 = 2（旧实现会是 1），release 后 = 0；
- **反向对照**：故意让两条序列共用同一 seq key，结果必须不同 —— 实测不同，证明这个
  测试真能抓住“状态互踩”（不是白跑）；
- 不共享页（首页不同）时结果同样与单独跑一致。

### 随机模型与对照数字

`tests/gen_random_qwen3next.py` 重写成带全部特性的小模型（字段布局与真实 80B config
对齐）：L=4、layer_types=[linear, full, linear, full]、E=64/D=16（rot_dim=4）、
**所有 norm 权重 ~N(0,0.3)（不是 HF 的 0 初值，故意的：这样 (1+w) 写错就会被抓住）**、
MoE 4 专家 top-2、moe_inter=32、shared_inter=32、`mlp_only_layers=[0]`、
`routed_scaling_factor=1.5`。`tests/ref_qwen3next.py` 同步实现（float64）。

| case | prefill top5 maxerr | greedy6 | 阈值 |
|---|---|---|---|
| Hello world | 2.51e-08 | 一致 | <5e-3（任务要求 MoE 模型 <1e-6，实测 ~1e-8） |
| The meaning of life is | 4.02e-08 | 一致 | 同上 |
| Paris is the capital of | 4.84e-08 | 一致 | 同上 |
| word ×100 | 5.75e-08 | 一致 | 同上 |

`test_llama: all OK`（含 decode_batch B=2 一致性）；全量 `ctest`：**24/24 通过**
（含 test_family_{mistral,gemma,gemma2,qwen3,qwen3next} 与新增 test_lin_seq），
真实 qwen3-06 回归（test_qwen / test_family_qwen3）数值不变。

### 仍未做（如实记录）

- CUDA 路径仍未支持 Qwen3-Next（cuda_backend.cu 只有 llama 族 attention），任务里
  “CUDA 可选”允许只保证 CPU。
- 真实 80B 权重没有下载（~160GB，超出本机 6GB 显存与磁盘预算）；验证用的是等价结构的
  随机模型（同权重名、同 config 字段、同公式）。

