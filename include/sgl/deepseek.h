/* sgl/deepseek.h — DeepSeek-V2 / V3 系模型后端（MLA + MoE）完整前向（CPU f32）
 * 对应 sglang srt/models/deepseek_v2.py（DeepseekV2ForCausalLM）的 CPU 单流路径：
 * prefill + 增量 decode + 批量 decode + paged KV 池（与 gpt2/llama 后端同一套语义）。
 *
 * 结构（HF modeling_deepseek_v2.py / modeling_deepseek_v3.py）：
 *   embed_tokens → L × DecoderLayer → model.norm → lm_head
 *   DecoderLayer  Pre-Norm 两段残差（HF DeepseekV2DecoderLayer.forward）：
 *     x = x + Attn(RMSNorm_in(x))
 *     x = x + MLP (RMSNorm_post(x))
 *   Attn = MLA（Multi-head Latent Attention，吸收式实现见 sgl/mla.h）：
 *     q = q_b_proj(q_a_layernorm(q_a_proj(x)))       （q_lora_rank 为 null 时直接 q_proj）
 *     kv_a = kv_a_proj_with_mqa(x) → [latent(kv_lora_rank) | k_pe(qk_rope_head_dim)]
 *     latent 过 kv_a_layernorm 后进 KV cache；kv_b_proj 只在「非吸收」写法里显式展开成
 *     k_nope/v，本实现把它折进 q（W_UK）和输出（W_UV），KV cache 只存
 *     [latent | k_pe]，每位置 stride = kv_lora_rank + qk_rope_head_dim。
 *     softmax 缩放任 HF 的 self.scaling = qk_head_dim^-0.5（qk_head_dim = nope + rope）。
 *   MLP = 前 first_k_dense_replace 层 dense SwiGLU（intermediate_size），
 *         其余层 MoE（HF DeepseekV2MoE）：
 *           gate（[n_routed_experts, E]）→ 打分 → top-k → 专家 gate/up/down 加权求和
 *           + 共享专家（宽度 moe_intermediate_size * n_shared_experts，无门控，直接相加）
 *         打分函数两种（config.scoring_func）：softmax（V2）或 sigmoid（V3，含
 *         e_score_correction_bias）。top-k 选择两种（config.topk_method）：
 *         greedy（V2）/ group_limited_greedy（V3，n_group + topk_group 先选组再选专家）。
 *         离线权重系数 = 打分值（**不加** bias 修正，同 HF DeepSeekV3TopkRouter:
 *         scores.gather(1, topk_indices) 与 DeepSeek 官方 inference 代码），
 *         norm_topk_prob 时归一化（+1e-20），再乘 routed_scaling_factor。
 *
 * 分页 KV 池：与 llama/gpt2 完全同语义（LIFO 空闲栈 + 引用计数），页大小
 * stride = kv_lora_rank + qk_rope_head_dim 个 float，池按层独立
 * [n_layer][n_pool][stride]；页表 slot_idx[p] 给出绝对位置 p 的页号，
 * 前缀共享（radix）下多序列共用同一页、靠 page_ref 计数（radix_evict 驱逐）。
 * 池大小 SGL_DEEPSEEK_POOL_SLOTS（默认 4096），环境变量 SGL_POOL_SLOTS 覆盖。
 *
 * YaRN：config.rope_scaling type=yarn 时用 sgl_rope_yarn_invf() 生成逆频率表，并乘
 * attention_factor（mscale/mscale_all_dim，同 HF）；频率表与 attention factor 对着
 * transformers 的 _compute_yarn_parameters 逐值验证（tests/test_yarn.c）。
 * 注：transformers 5.20 自己的 DeepseekV2RotaryEmbedding 会收下 yarn 配置但不生效，
 * 所以整模型层没法拿 HF 当金标准；整模验证用 gen_random_deepseek.py --yarn 的 numpy 参考。
 * 未实现：rope_type=linear/dynamic（写 warning 退化为普通 RoPE）。rope_interleave 支持。
 */
#ifndef SGL_DEEPSEEK_H
#define SGL_DEEPSEEK_H

#include "sgl/safetensors.h"
#include "sgl/tensor.h"

#ifndef SGL_DEEPSEEK_POOL_SLOTS
#define SGL_DEEPSEEK_POOL_SLOTS 4096
#endif

typedef struct {
  sgl_tensor inp_norm;  /* [E]    input_layernorm.weight */
  sgl_tensor post_norm; /* [E]    post_attention_layernorm.weight */
  /* ---- MLA ---- */
  sgl_tensor q_a_w;    /* [q_lora][E]      q_a_proj（q_lora==0 时为空）*/
  sgl_tensor q_a_norm; /* [q_lora]         q_a_layernorm.weight */
  sgl_tensor q_b_w;    /* [H*qk_head_dim][q_lora]（q_lora==0 时是 [..][E] 的 q_proj）*/
  sgl_tensor kv_a_w;   /* [kv_lora+qk_rope][E]  kv_a_proj_with_mqa */
  sgl_tensor kv_a_norm;/* [kv_lora]        kv_a_layernorm.weight */
  sgl_tensor kv_b_w;   /* [H*(qk_nope+v_head)][kv_lora]  kv_b_proj（W_UK/W_UV 来源）*/
  sgl_tensor o_w;      /* [E][H*v_head]    o_proj */
  sgl_tensor q_a_b, kv_a_b, o_b; /* 可选 bias（attention_bias=true 的 config 才有）*/
  /* ---- MLP：dense（前 first_k_dense_replace 层）---- */
  int is_moe;                        /* 1 = 本层走 MoE（dense 权重为空）*/
  sgl_tensor gate_w, up_w, down_w;   /* [I][E] [I][E] [E][I] */
  /* ---- MLP：MoE ---- */
  sgl_tensor router_w;               /* [n_routed_experts][E]  mlp.gate.weight */
  float *ec_bias;                    /* [n_routed_experts] e_score_correction_bias（或 NULL）*/
  int exp_stacked;                   /* 1 = 三维融合专家布局（HF V3: experts.gate_up_proj）*/
  sgl_tensor exp_gu3;                /* [NE][2*mi][E]  stacked gate|up */
  sgl_tensor exp_dn3;                /* [NE][E][mi]    stacked down */
  sgl_tensor *exp_gate, *exp_up, *exp_down; /* 逐专家布局（HF 常见 / 官方权重）*/
  sgl_tensor sh_gate_w, sh_up_w, sh_down_w; /* 共享专家 [sh][E] [sh][E] [E][sh] */
} sgl_deepseek_layer;

typedef struct sgl_deepseek {
  /* ---- config ---- */
  int n_layer, n_head, n_embd, n_ctx, vocab;
  int q_lora;     /* q_lora_rank（0 = null，无 q 低秩）*/
  int kv_lora;    /* kv_lora_rank */
  int qk_nope;    /* qk_nope_head_dim */
  int qk_rope;    /* qk_rope_head_dim */
  int v_head;     /* v_head_dim */
  int q_head_dim; /* qk_nope + qk_rope */
  int stride;     /* KV 行宽 = kv_lora + qk_rope */
  float softmax_scale; /* q_head_dim^-0.5 */
  float rms_eps;
  double rope_theta;
  int rope_interleave;  /* 1 = HF V3 的交错 RoPE（config.rope_interleave）*/
  /* P1: YaRN（config.rope_scaling type=yarn；DeepSeek-V3 真实权重）*/
  int yarn;             /* 1 = 使用 YaRN 频率表 + rope_attn_factor */
  double yarn_factor, yarn_base, yarn_beta_fast, yarn_beta_slow, yarn_orig;
  float rope_attn_factor; /* mscale：乘到 rope 旋转后的结果（同 HF 的 cos/sin 缩放）*/
  float yarn_mscale_mul;  /* YaRN 的 mscale²（DeepSeek 乘到 softmax_scale 上）*/
  int n_inter;          /* dense intermediate_size */
  int n_experts;        /* n_routed_experts（0 = 全 dense）*/
  int n_topk;           /* num_experts_per_tok */
  int moe_inter;        /* moe_intermediate_size */
  int n_shared;         /* n_shared_experts 个数 */
  int shared_inter;     /* moe_inter * n_shared */
  int first_k_dense;    /* first_k_dense_replace */
  int moe_freq;         /* moe_layer_freq */
  int norm_topk;        /* norm_topk_prob */
  int sigmoid_score;    /* scoring_func == "sigmoid"（V3）*/
  int group_limited;    /* topk_method == "group_limited_greedy"（V3）*/
  int n_group, topk_group;
  float routed_scale;   /* routed_scaling_factor */
  int tie_embeddings;
  int n_moe;            /* MoE 层数（统计/日志用）*/
  double *invf;         /* [qk_rope/2] RoPE 逆频率（theta^(-2i/qk_rope)）*/
  /* ---- 权重 ---- */
  sgl_tensor wte, lm_head, final_norm;
  sgl_deepseek_layer *layers;
  sgl_st_ctx *st;
  /* ---- scratch（按 n_ctx 分配，nthr = omp 线程数）---- */
  float *x, *h;        /* [ctx][E] */
  float *qa, *q;       /* [ctx][q_lora] / [ctx][H*q_head_dim] */
  float *kva, *lat;    /* [ctx][stride] / [ctx][kv_lora] */
  float *att;          /* [ctx][H*v_head]（o_proj 输入）*/
  float *gu, *upd;     /* [ctx][RW] RW=max(I, 2*mi, sh)：dense gate/up 与 MoE 单行 gate_up */
  float *mlp_out;      /* [ctx][E]  MLP 输出 */
  float *rt_out;       /* [E]       MoE 单行 down 输出（**不能**复用 mlp_out：它会逐行覆盖）*/
  float *tmp_abs;      /* [nthr][H*kv_lora + kv_lora] q_abs + o_lat */
  float *scores;       /* [nthr][ctx] attention 分数行 */
  float *rt_sc, *rt_ch;/* [n_experts] 打分 / 选择用（含 bias）*/
  float *rt_gs;        /* [n_group] 组分数 */
  int *rt_gsel;        /* [n_group] 选中的组号 */
  int *rt_sel;         /* [64] 选中的专家号 */
  float *rt_w;         /* [64] 对应权重 */
  int nthr, n_score_rows;
  /* forward 内部的每行位置 / 页表（避免每次调用 malloc）*/
  int *bpos;           /* [ctx] */
  int **btabs;         /* [ctx] */
  /* ---- 分页 KV 池 [n_layer][n_pool][stride] ---- */
  float *kc;
  int n_pool, n_free;
  int *free_pages, *page_refs;
} sgl_deepseek;

/* 读 <model_dir>/config.json + 权重（sgl_st_open_dir，支持分片）。失败返回 NULL。 */
sgl_deepseek *sgl_deepseek_load(const char *model_dir);
void sgl_deepseek_free(sgl_deepseek *m);

/* 模型识别：config.json 里 architectures/model_type 是 DeepSeek-V2/V3 系，或同时有
 * kv_lora_rank + qk_nope_head_dim。返回 1 = 是，0 = 否。-1 = config 打不开。 */
int sgl_deepseek_detect(const char *model_dir);

int sgl_deepseek_vocab(const sgl_deepseek *m);
int sgl_deepseek_ctx(const sgl_deepseek *m);

/* 分页池（语义同 sgl_llama_*）*/
int sgl_deepseek_pages_alloc(sgl_deepseek *m, int *idx, int n);
void sgl_deepseek_pages_unref(sgl_deepseek *m, const int *idx, int n);
void sgl_deepseek_page_ref(sgl_deepseek *m, int page, int delta);
int sgl_deepseek_nfree(const sgl_deepseek *m);
int sgl_deepseek_pool_pages(const sgl_deepseek *m);
int sgl_deepseek_page_refs(const sgl_deepseek *m, int page);

/* prefill / 增量单 token：tokens[n] 写位置 [pos0,pos0+n)；页表 idx 长度 ≥ pos0+n。
 * logits_out 收末位分布（[vocab]）；传 NULL 跳过 lm_head。 */
int sgl_deepseek_prefill(sgl_deepseek *m, const int *tokens, int n, int pos0,
                         const int *idx, float *logits_out);
/* 同 prefill（llama 后端的命名习惯，两者等价）*/
int sgl_deepseek_forward(sgl_deepseek *m, const int *tokens, int n, int pos0,
                         const int *idx, float *logits_out);

/* n-token 前向并取【全部 n 个位置】的分布（logits_all 需 n*vocab；NULL 同 prefill）。
 * 与逐 token 前向数值一致（每元素累加方式不变，只换行数）。 */
int sgl_deepseek_forward_n(sgl_deepseek *m, const int *tokens, int n, int pos0,
                           const int *idx, float *logits_all);

/* emb 版 prefill：直接吃 [n][E] embedding 序列（多模态融合入口），页表语义同 prefill。 */
int sgl_deepseek_prefill_emb(sgl_deepseek *m, const float *emb, int n, int pos0,
                             const int *idx, float *logits_out);

/* 批量 decode：B 序列各 1 token（位置 seq_lens[b]，写页 slot idx[b][seq_lens[b]]）；
 * logits_out 需 B*vocab。 */
int sgl_deepseek_decode_batch(sgl_deepseek *m, const int *tokens, int *const *idx,
                              const int *seq_lens, int B, float *logits_out);

/* 维度查询（测试 / 上层开页表用）*/
int sgl_deepseek_kv_stride(const sgl_deepseek *m); /* kv_lora + qk_rope */
int sgl_deepseek_heads(const sgl_deepseek *m);
int sgl_deepseek_hidden(const sgl_deepseek *m);

/* ---- 接入 sgl_model（model.c 分发）需要上层改的几行（本文件不改现有文件）----
 *  1) model.c：#include "sgl/deepseek.h"；kind 再加 2 = deepseek；
 *     sgl_model_load() 里在 gpt2/llama 分发前插：
 *       if (sgl_deepseek_detect(dir) == 1) { m->kind = 2; m->impl = sgl_deepseek_load(dir); ... }
 *     五个转发函数（vocab/ctx/pages_alloc/pages_unref/page_ref/nfree/pool_pages/page_refs/
 *     prefill/decode_batch/prefill_emb）各加一条 kind == 2 分支，分别调同名 sgl_deepseek_*。
 *  2) CMakeLists：src 下的 c 文件是 GLOB_RECURSE CONFIGURE_DEPENDS，src/deepseek.c 会自动进库；
 *     tests/test_ 开头的 c 文件的 glob 没有 CONFIGURE_DEPENDS，加新测试文件后需重新 cmake 配置一次。
 */

#endif