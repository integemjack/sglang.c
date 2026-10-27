/* sgl/mla.h — DeepSeek 系 MLA（Multi-head Latent Attention）注意力（CPU f32）
 * 对应 sglang 的 DeepseekV2Attention / DeepseekV3Attention（HF modeling_deepseek_v2.py
 * 的 MLAttention 公式），DeepSeek-V2 / V3 / V4 系共用的注意力结构：
 *
 *   q 低秩：      x → q_a_proj → q_a_layernorm(RMSNorm) → q_b_proj
 *                    → [heads*(qk_nope_head_dim + qk_rope_head_dim)]
 *   kv 低秩+解耦：x → kv_a_proj_with_mqa → 拆成两段
 *                    [0, kv_lora_rank)             → latent c（过 kv_a_layernorm）
 *                    [kv_lora_rank, +qk_rope_head_dim) → k_pe（过 RoPE）
 *                 c → kv_b_proj → 每头 [k_nope (qk_nope_head_dim) | v (v_head_dim)]
 *   RoPE 只作用在 q/k 的 rope 那一段（q 每头最后 qk_rope_head_dim 维；k 只有一份
 *   k_pe、所有 head 共享，即解耦 RoPE 的 MQA 分量）。式子是 HF 默认的
 *   「非交错 rotate_half」（配对 (i, i+D/2)），直接用 ops.h 的 sgl_rope_inplace。
 *   注意 HF/DeepSeek 的 softmax 缩放是 q_head_dim^-0.5（q_head_dim = qk_nope + qk_rope），
 *   不是每头 v_head_dim 或 kv_lora_rank。
 *
 * 【本实现是吸收式（absorbed）】——把 W_UK/W_UV 折进 q 和输出：
 *   q_abs[h] = W_UK[h]^T q_nope[h]      （[kv_lora_rank]，W_UK[h] = kv_b_proj 每头 k_nope 那几行）
 *   score    = q_abs[h]·c + q_pe[h]·k_pe
 *   o_lat[h] = Σ_p a_p c_p             （在 latent 空间加权求和，**不需要 v 进 cache**）
 *   o[h]     = W_UV[h] o_lat[h]        （[v_head_dim]，W_UV[h] = kv_b_proj 每头 v 那几行）
 *   再拼 head → o_proj。数学上与 HF 的非吸收式（先展开 k_nope/v 再 attention）等价，
 *   差别只在 fp32 舍入（与 numpy 参考的实测最大相对误差见 tests/test_mla.c 输出）。
 *   吸收式的好处正是 MLA 的卖点：KV cache 只存压缩 latent + k_pe，
 *   每位置 stride = kv_lora_rank + qk_rope_head_dim 个数，而不是 heads*(qk_nope+v_head)。
 *
 * 这是数值参考实现（标量/SIMD 复用 sgl/ops.h），不是生产性能路径；
 * 生产上 sglang 用 triton/cutlass 把吸收式融合成一个 kernel。
 */
#ifndef SGL_MLA_H
#define SGL_MLA_H

#include "sgl/safetensors.h"
#include "sgl/tensor.h"

/* 连续 KV 缓冲（调用方持有；与 llama.c 的分页页表无关，MLA 走自己的缓冲）：
 * 每个【绝对位置】一行，行宽 stride = kv_lora_rank + qk_rope_head_dim：
 *   row[0 .. kv_lora_rank)                    : latent c（kv_a_layernorm 之后）
 *   row[kv_lora_rank .. + qk_rope_head_dim)   : k_pe（RoPE 之后，全 head 共享）
 * 位置 p 的 query 走 causal attention 时读位置 0..p 的全部行，
 * 所以要求「cache 里下标 = 该 token 的绝对位置」，且同一序列的位置从 0 开始连续。 */
typedef struct {
  float *data; /* [cap][stride]，sgl_xmalloc 分配 */
  int cap;     /* 位置容量（>= 最长上下文）*/
  int stride;  /* 行宽 = sgl_mla_cache_stride(m) */
  int n_used;  /* 已写入的最大位置 + 1（调试/断言用）*/
} sgl_kv_cache;

typedef struct sgl_mla sgl_mla;

/* 读 <model_dir>/config.json + 权重（sgl_st_open_dir，支持分片）。
 * 权重名按 HF DeepSeek 约定：model.layers.{layer}.self_attn.{q_a_proj,q_a_layernorm,
 * q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,o_proj}.weight
 * （config 里 q_lora_rank 为 null 时改为 q_proj.weight，无 q 低秩）。
 * sgl_mla_load = 第 0 层。失败返回 NULL。 */
sgl_mla *sgl_mla_load(const char *model_dir);
sgl_mla *sgl_mla_load_layer(const char *model_dir, int layer);
void sgl_mla_free(sgl_mla *m);

/* KV 缓冲生命周期（cap = 位置容量，一般取 max_position_embeddings 或序列上限）*/
sgl_kv_cache *sgl_mla_cache_new(const sgl_mla *m, int cap);
void sgl_mla_cache_reset(sgl_kv_cache *c); /* n_used = 0（不擦数据：位置由 pos 决定）*/
void sgl_mla_cache_free(sgl_kv_cache *c);

/* MLA attention 前向：x[n][hidden] → out[n][hidden]（只含 attention 分支，不含残差与
 * MLP，调用方自己加残差）。pos[i] 是第 i 个 token 的绝对位置：写 cache 第 pos[i] 行，
 * 并对第 pos[i] 行 query 做 causal mask（attend 位置 0..pos[i]）。
 * 支持 n>1 的 prefill 与 n==1 的增量 decode（用 cache 里已有的位置）。
 * 要求 out 不与 x 别名。返回 0 成功、-1 失败（pos 越界 / cache 不匹配 / 维度不符）。*/
int sgl_mla_forward(sgl_mla *m, const float *x, int n, const int *pos, float *out,
                    sgl_kv_cache *cache);

/* 维度查询（调用方开 cache / 写测试用）*/
int sgl_mla_hidden(const sgl_mla *m);       /* hidden_size */
int sgl_mla_heads(const sgl_mla *m);        /* num_attention_heads */
int sgl_mla_kv_lora_rank(const sgl_mla *m); /* kv_lora_rank */
int sgl_mla_qk_rope_dim(const sgl_mla *m);  /* qk_rope_head_dim */
int sgl_mla_cache_stride(const sgl_mla *m); /* kv_lora_rank + qk_rope_head_dim */

#endif