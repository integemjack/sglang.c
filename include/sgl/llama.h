/* sgl/llama.h — Llama 系模型（Llama/TinyLlama/Qwen 同构：RoPE/RMSNorm/SwiGLU/GQA）
 * 对应 sglang srt/models/llama.py 的 CPU 单流路径（prefill + 增量 decode）。
 * KV 池：分页 + 引用计数（与 gpt2 相同语义，页大小 = kv_dim）。 */
#ifndef SGL_LLAMA_H
#define SGL_LLAMA_H

#include "sgl/safetensors.h"
#include "sgl/tensor.h"

#ifndef SGL_LLAMA_POOL_SLOTS
#define SGL_LLAMA_POOL_SLOTS 4096
#endif

typedef struct {
  sgl_tensor inp_norm;                       /* [E] */
  sgl_tensor q_w, k_w, v_w, o_w;             /* [E,E] [K,E] [K,E] [E,E]（K=kv_dim）*/
  sgl_tensor q_b, k_b, v_b;                   /* Qwen2 投影 bias（q[E] k[K] v[K]；无则 null）*/
  sgl_tensor q_norm, k_norm;                  /* [D] Qwen2 的 QK-Norm；无则 null */
  sgl_tensor post_norm;                      /* [E] */
  sgl_tensor gate_w, up_w, down_w;           /* [I,E] [I,E] [E,I] */
  /* int8 权量化（SGL_W8=1）：与上面 f32 权重并存，前向二选一 */
  int8_t *q8, *k8, *v8, *o8, *g8, *u8, *d8;  /* 同布局 [N,K] */
  float *sq8, *sk8, *sv8, *so8, *sg8, *su8, *sd8; /* per-channel scale 数组 [N]，量化时分配 */
} sgl_llama_layer;

typedef struct {
  int n_layer, n_head, n_kv_head, n_embd, n_inter, n_ctx, vocab;
  int head_dim, kv_dim, q_dim; /* q_dim = n_head*head_dim (q rows & attn rows; == n_embd unless explicit head_dim, e.g. qwen3) */
  int sliding_window;        /* t9 mistral: >0 = banded causal mask (attend max p in [pos-win+1, pos]) */
  float gemma_scale;         /* t9 gemma: >0 = multiply embeddings by sqrt(E) (hidden normalizer) */
  float rms_eps;
  double rope_theta;
  int tie_embeddings;
  int has_qkv_bias;   /* Qwen2 系：qkv 投影带 bias */
  int w8;             /* int8 权重前向（SGL_W8=1）*/
  int kv8;            /* int8 KV cache（SGL_KV8=1）*/
  sgl_tensor wte;        /* [V,E] */
  sgl_tensor lm_head;    /* [V,E]（tie 时复用 wte）*/
  sgl_llama_layer *layers;
  sgl_tensor final_norm; /* [E] */
  sgl_st_ctx *st;
  /* scratch */
  float *x, *h, *q, *k, *v, *attn, *gate, *up, *scores;
  int8_t *x8;         /* 激活 int8（per-token，W8 前向用）*/
  float *x8s;         /* per-token scale */
  int n_score_rows;
  /* 分页 KV 池 [n_layer][n_pool][kv_dim]；kv8=1 时按 int8 存（scale [n_layer][n_pool]）*/
  float *kc, *vc;
  int8_t *kc8, *vc8;  /* kv8=1 时 kc/vc 的实际存储 */
  float *kcs, *vcs;   /* kv8=1 时每层每页的量化 scale */
  int n_pool, n_free;
  int *free_pages, *page_refs;
} sgl_llama;

sgl_llama *sgl_llama_load(const char *model_dir);
void sgl_llama_free(sgl_llama *m);

/* 分页池（语义同 gpt2）*/
int sgl_llama_pages_alloc(sgl_llama *m, int *idx, int n);
void sgl_llama_pages_unref(sgl_llama *m, const int *idx, int n);
void sgl_llama_page_ref(sgl_llama *m, int page, int delta);
int sgl_llama_nfree(const sgl_llama *m);

/* prefill / 增量单 token：tokens[n] 写位置 [pos0,pos0+n)；页表 slot_idx 长度 ≥ pos0+n */
int sgl_llama_forward(sgl_llama *m, const int *tokens, int n, int pos0,
                      const int *slot_idx, float *logits_out);

/* n-token 批量前向（Spec-v2 verify / n-outer 分块 GEMM 复用）：与 sgl_llama_forward
 * 完全同语义（同权重复用路径、KV 一次写 [pos0,pos0+n)），但 logits_out 返回
 * 【全部 n 个位置】的分布 [n][V]（传 NULL 则不算 logits）。数值与逐 token 前向
 * 逐位一致（sgl_matmul_nt 分块不改每元素累加方式），test_spec 有断言。 */
int sgl_llama_forward_n(sgl_llama *m, const int *tokens, int n, int pos0,
                        const int *slot_idx, float *logits_all);

/* t10 llava：emb 版前向——直接吃 [n][E] embedding 序列（视觉 token 融合入口）。
 * 与 token 前向同一路径（同一层循环），数值一致。 */
int sgl_llama_forward_emb(sgl_llama *m, const float *emb, int n, int pos0,
                          const int *slot_idx, float *logits_out);

/* 批量 decode（调度器用）：B 序列各 1 token；logits_out 需 B*vocab */
int sgl_llama_decode_batch(sgl_llama *m, const int *tokens, int *const *slot_idx,
                           const int *seq_lens, int B, float *logits_out);

/* ---- beam search ---- */
typedef struct {
  int *ids;        /* 生成部分（不含 prompt/eos）；sgl_xfree 释放 */
  int n;
  double logprob;  /* 累计 log P(generated | prompt) */
} sgl_llama_beam_result;

/* 返回结果条数（按 lp 降序；≤ beam_width）；失败 -1 */
int sgl_llama_beam_search(sgl_llama *m, const int *prompt, int n_prompt,
                          int beam_width, int max_new, int eos_id,
                          sgl_llama_beam_result *out);

#endif
