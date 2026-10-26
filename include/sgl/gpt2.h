/* sgl/gpt2.h — GPT-2 模型（prefill + 批量 decode，分页 KV 池 + 引用计数）
 * 对应 sglang srt/models/gpt2.py + srt/mem_cache/memory_pool（MHATokenToKVPool）
 * 的 CPU 单卡版：
 *   - KV 池按「页」（= 1 槽 = 某层一个位置的 K/V）分配，附引用计数，
 *     供 RadixCache 共享前缀（M8）。
 *   - 序列的位置 p 对应页表 slot_idx[p]，注意力按页表 gather（非连续）。 */
#ifndef SGL_GPT2_H
#define SGL_GPT2_H

#include "sgl/safetensors.h"
#include "sgl/tensor.h"

#ifndef SGL_GPT2_POOL_SLOTS
#define SGL_GPT2_POOL_SLOTS 4096 /* 默认 KV 池页数（4096×72KB≈288MB；可用 env 覆盖做压力测试）*/
#endif
#define SGL_GPT2_MAX_BATCH 32

typedef struct {
  sgl_tensor ln1_g, ln1_b, ln2_g, ln2_b;           /* [embd] */
  sgl_tensor qkv_w, qkv_b;                         /* [embd,3*embd] [3*embd] */
  sgl_tensor proj_w, proj_b;                       /* [embd,embd] [embd] */
  sgl_tensor fc_w, fc_b, mlp_proj_w, mlp_proj_b;   /* [embd,4*embd] [4*embd] [4*embd,embd] [embd] */
} sgl_gpt2_layer;

typedef struct {
  int n_layer, n_head, n_embd, n_ctx, vocab;
  float ln_eps;
  sgl_tensor wte, wpe; /* [vocab,embd] [ctx,embd] */
  sgl_gpt2_layer *layers;
  sgl_tensor lnf_g, lnf_b;
  sgl_st_ctx *st;
  /* scratch（按 n_ctx 分配一次）*/
  float *x, *h, *qkv, *attn_out, *mlp_h, *scores;
  int n_score_rows; /* scores 行数 = 并行线程数（每线程一行 n_ctx）*/
  /* 批量 decode scratch */
  float *bx, *bh, *bqkv, *battn, *bmlp, *bscores;
  /* KV 池: [n_layer][n_pool][n_embd]，页 = 某层一个位置 */
  float *kc, *vc;
  int n_pool;
  int *free_pages;  /* 空闲页栈（LIFO）*/
  int n_free;
  int *page_refs;   /* 每页引用计数（调度请求 + RadixCache 节点）*/
  int n_seq;        /* 兼容字段：最后一次前向的序列长度（单请求路径）*/
} sgl_gpt2;

sgl_gpt2 *sgl_gpt2_load(const char *model_dir);
void sgl_gpt2_free(sgl_gpt2 *m);

/* ---- 分页 KV 池 ---- */
/* 分配 n 页填入 idx[]（整批成功或整批失败，返回 -1 不改写 idx）*/
int sgl_gpt2_pages_alloc(sgl_gpt2 *m, int *idx, int n);
/* 引用计数 ±delta（delta=-1 减到 0 自动归还空闲栈）*/
void sgl_gpt2_page_ref(sgl_gpt2 *m, int page, int delta);
/* 对一组页整体减引用（等价逐页 page_ref(-1)）*/
void sgl_gpt2_pages_unref(sgl_gpt2 *m, const int *idx, int n);
int sgl_gpt2_nfree(const sgl_gpt2 *m);

/* ---- 前向 ---- */
/* prefill：tokens[n] 写入位置 [pos0, pos0+n)，slot_idx 为该序列从位置 0 起的完整页表 */
int sgl_gpt2_prefill(sgl_gpt2 *m, const int *tokens, int n, int pos0,
                     const int *slot_idx, float *logits_out);
/* 批量 decode：B 个序列各 1 token；slot_idx[b] 为序列 b 的完整页表（长度 ≥ seq_lens[b]+1）*/
int sgl_gpt2_decode_batch(sgl_gpt2 *m, const int *tokens, int *const *slot_idx,
                          const int *seq_lens, int B, float *logits_out);

/* logits = x_last × wte^T */
void sgl_gpt2_logits(const sgl_gpt2 *m, const float *x_last, float *logits);

#endif
