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

## 4. 与真实模型的差距（已知限制）

- linear_attention 已按真实 Qwen3-Next 公式实现，但 **full_attention 层是简化版**：
  真实 Qwen3-Next 的 full 层还有 q 门控（q_proj 输出 2×，半幅 sigmoid 门）、
  partial rotary（rope 只作用 25% 维）、output norm 为 `(1+w)*x/sqrt(var+1)`、
  MoE 专家层、共享专家 + routed scaling —— 本实现均按普通 qwen3 full attention 走，
  且无 MoE。加载真实 Qwen3-Next 权重会缺张量而报错（不会静默算错）。
- dynamic rope 取静态 NTK 形式，非 HF 的 dynamic_rope_update 逐步插值。
- 递归状态按序列首页归属：两条序列共享同一首页（radix 前缀复用）会互相踩状态；
  槽位耗尽报错。CPU 参考路径可接受，调度器接入真实 Qwen3-Next 前需改按序列分配。
- CUDA 路径未动（cuda_backend.cu 不属于本任务）。
