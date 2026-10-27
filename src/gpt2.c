/* sgl gpt2 实现 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/gpt2.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/gguf.h"
#include "sgl/json.h"
#include "sgl/log.h"
#include "sgl/ops.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static char *path_join(const char *dir, const char *name) {
  size_t n = strlen(dir) + strlen(name) + 2;
  char *p = (char *)sgl_xmalloc(n);
  snprintf(p, n, "%s/%s", dir, name);
  return p;
}


sgl_gpt2 *sgl_gpt2_load(const char *model_dir) {
  size_t clen = 0;
  char *cfg_text = sgl_config_read_text(model_dir, &clen);
  if (!cfg_text) return NULL;
  sgl_json *cfg = sgl_json_parse(cfg_text, clen);
  sgl_xfree(cfg_text);
  if (!cfg) return NULL;

  sgl_gpt2 *m = (sgl_gpt2 *)sgl_xcalloc(1, sizeof(sgl_gpt2));
  m->n_layer = sgl_json_get_int(cfg, "n_layer", 12);
  m->n_head = sgl_json_get_int(cfg, "n_head", 12);
  m->n_embd = sgl_json_get_int(cfg, "n_embd", 768);
  m->n_ctx = sgl_json_get_int(cfg, "n_ctx", sgl_json_get_int(cfg, "n_positions", 1024));
  m->vocab = sgl_json_get_int(cfg, "vocab_size", 50257);
  m->ln_eps = (float)sgl_json_get_num(cfg, "layer_norm_epsilon", 1e-5);
  sgl_json_free(cfg);
  if (m->n_embd % m->n_head != 0) {
    SGL_ERROR("n_embd %% n_head != 0");
    sgl_gpt2_free(m);
    return NULL;
  }
  SGL_INFO("GPT-2 config: L=%d H=%d E=%d ctx=%d vocab=%d eps=%g",
           m->n_layer, m->n_head, m->n_embd, m->n_ctx, m->vocab, m->ln_eps);

  /* 打开权重：model.safetensors / model.safetensors.index.json（多分片）*/
  m->st = sgl_st_open_dir(model_dir);
  if (!m->st) {
    sgl_gpt2_free(m);
    return NULL;
  }

  int64_t vocab2 = m->vocab, ctx2 = m->n_ctx, e = m->n_embd;
  int64_t e3 = 3 * e, e4 = 4 * e;
  m->wte = sgl_st_get(m->st, "wte.weight", 2, (int64_t[]){vocab2, e});
  m->wpe = sgl_st_get(m->st, "wpe.weight", 2, (int64_t[]){ctx2, e});
  m->lnf_g = sgl_st_get(m->st, "ln_f.weight", 1, (int64_t[]){e});
  m->lnf_b = sgl_st_get(m->st, "ln_f.bias", 1, (int64_t[]){e});
  if (sgl_tensor_is_null(m->wte) || sgl_tensor_is_null(m->wpe)) {
    sgl_gpt2_free(m);
    return NULL;
  }

  m->layers = (sgl_gpt2_layer *)sgl_xcalloc((size_t)m->n_layer, sizeof(*m->layers));
  for (int l = 0; l < m->n_layer; l++) {
    char name[128];
    snprintf(name, sizeof(name), "h.%d.ln_1.weight", l);
    m->layers[l].ln1_g = sgl_st_get(m->st, name, 1, (int64_t[]){e});
    snprintf(name, sizeof(name), "h.%d.ln_1.bias", l);
    m->layers[l].ln1_b = sgl_st_get(m->st, name, 1, (int64_t[]){e});
    snprintf(name, sizeof(name), "h.%d.attn.c_attn.weight", l);
    m->layers[l].qkv_w = sgl_st_get(m->st, name, 2, (int64_t[]){e, e3});
    snprintf(name, sizeof(name), "h.%d.attn.c_attn.bias", l);
    m->layers[l].qkv_b = sgl_st_get(m->st, name, 1, (int64_t[]){e3});
    snprintf(name, sizeof(name), "h.%d.attn.c_proj.weight", l);
    m->layers[l].proj_w = sgl_st_get(m->st, name, 2, (int64_t[]){e, e});
    snprintf(name, sizeof(name), "h.%d.attn.c_proj.bias", l);
    m->layers[l].proj_b = sgl_st_get(m->st, name, 1, (int64_t[]){e});
    snprintf(name, sizeof(name), "h.%d.ln_2.weight", l);
    m->layers[l].ln2_g = sgl_st_get(m->st, name, 1, (int64_t[]){e});
    snprintf(name, sizeof(name), "h.%d.ln_2.bias", l);
    m->layers[l].ln2_b = sgl_st_get(m->st, name, 1, (int64_t[]){e});
    snprintf(name, sizeof(name), "h.%d.mlp.c_fc.weight", l);
    m->layers[l].fc_w = sgl_st_get(m->st, name, 2, (int64_t[]){e, e4});
    snprintf(name, sizeof(name), "h.%d.mlp.c_fc.bias", l);
    m->layers[l].fc_b = sgl_st_get(m->st, name, 1, (int64_t[]){e4});
    snprintf(name, sizeof(name), "h.%d.mlp.c_proj.weight", l);
    m->layers[l].mlp_proj_w = sgl_st_get(m->st, name, 2, (int64_t[]){e4, e});
    snprintf(name, sizeof(name), "h.%d.mlp.c_proj.bias", l);
    m->layers[l].mlp_proj_b = sgl_st_get(m->st, name, 1, (int64_t[]){e});
    /* 校验：一个关键权重缺失即失败 */
    if (sgl_tensor_is_null(m->layers[l].qkv_w) || sgl_tensor_is_null(m->layers[l].fc_w)) {
      SGL_ERROR("layer %d weights missing", l);
      sgl_gpt2_free(m);
      return NULL;
    }
  }

  /* scratch + KV 槽位池一次性分配（池大小 = n_ctx 槽，即单序列满长）*/
  size_t ctx = (size_t)m->n_ctx;
#ifdef _OPENMP
  m->n_score_rows = omp_get_max_threads();
#else
  m->n_score_rows = 1;
#endif
  m->x = (float *)sgl_xmalloc(ctx * m->n_embd * sizeof(float));
  m->h = (float *)sgl_xmalloc(ctx * m->n_embd * sizeof(float));
  m->qkv = (float *)sgl_xmalloc(ctx * 3 * m->n_embd * sizeof(float));
  m->attn_out = (float *)sgl_xmalloc(ctx * m->n_embd * sizeof(float));
  m->mlp_h = (float *)sgl_xmalloc(ctx * 4 * m->n_embd * sizeof(float));
  m->scores = (float *)sgl_xmalloc((size_t)m->n_score_rows * ctx * sizeof(float));

  size_t mb = SGL_GPT2_MAX_BATCH;
  m->bx = (float *)sgl_xmalloc(mb * m->n_embd * sizeof(float));
  m->bh = (float *)sgl_xmalloc(mb * m->n_embd * sizeof(float));
  m->bqkv = (float *)sgl_xmalloc(mb * 3 * m->n_embd * sizeof(float));
  m->battn = (float *)sgl_xmalloc(mb * m->n_embd * sizeof(float));
  m->bmlp = (float *)sgl_xmalloc(mb * 4 * m->n_embd * sizeof(float));
  m->bscores = (float *)sgl_xmalloc(mb * m->n_head * ctx * sizeof(float));

  /* 分页 KV 池（页 = 某层一个位置；可用环境变量 SGL_POOL_SLOTS 覆盖做压力测试）*/
  m->n_pool = SGL_GPT2_POOL_SLOTS;
  {
    const char *env = getenv("SGL_POOL_SLOTS");
    if (env && atoi(env) > 0) m->n_pool = atoi(env);
  }
  m->kc = (float *)sgl_xmalloc((size_t)m->n_layer * m->n_pool * m->n_embd * sizeof(float));
  m->vc = (float *)sgl_xmalloc((size_t)m->n_layer * m->n_pool * m->n_embd * sizeof(float));
  m->free_pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)m->n_pool);
  m->page_refs = (int *)sgl_xcalloc((size_t)m->n_pool, sizeof(int));
  for (int i = 0; i < m->n_pool; i++) m->free_pages[i] = m->n_pool - 1 - i; /* 栈：LIFO */
  m->n_free = m->n_pool;
  SGL_INFO("GPT-2 loaded: weights mmap'd zero-copy, KV pool %d pages %.1f MB",
           m->n_pool, (m->n_layer * 2.0 * m->n_pool * m->n_embd * sizeof(float)) / 1048576.0);
  return m;
}

void sgl_gpt2_free(sgl_gpt2 *m) {
  if (!m) return;
  sgl_xfree(m->x); sgl_xfree(m->h); sgl_xfree(m->qkv); sgl_xfree(m->attn_out);
  sgl_xfree(m->mlp_h); sgl_xfree(m->scores);
  sgl_xfree(m->bx); sgl_xfree(m->bh); sgl_xfree(m->bqkv);
  sgl_xfree(m->battn); sgl_xfree(m->bmlp); sgl_xfree(m->bscores);
  sgl_xfree(m->kc); sgl_xfree(m->vc);
  sgl_xfree(m->free_pages); sgl_xfree(m->page_refs);
  sgl_xfree(m->layers);
  sgl_st_close(m->st); /* mmap 权重随之释放 */
  sgl_xfree(m);
}

/* ---- 分页 KV 池：LIFO 空闲栈 + 引用计数 ---- */
int sgl_gpt2_pages_alloc(sgl_gpt2 *m, int *idx, int n) {
  if (n <= 0) return 0;
  if (n > m->n_free) return -1; /* 整批成功或整批失败 */
  m->n_free -= n;
  for (int i = 0; i < n; i++) {
    idx[i] = m->free_pages[m->n_free + i];
    m->page_refs[idx[i]] = 1; /* 调用方持有 */
  }
  return 0;
}

void sgl_gpt2_page_ref(sgl_gpt2 *m, int page, int delta) {
  if (page < 0 || page >= m->n_pool) return;
  m->page_refs[page] += delta;
  if (m->page_refs[page] <= 0) {
    m->page_refs[page] = 0;
    m->free_pages[m->n_free++] = page;
  }
}

void sgl_gpt2_pages_unref(sgl_gpt2 *m, const int *idx, int n) {
  for (int i = 0; i < n; i++) sgl_gpt2_page_ref(m, idx[i], -1);
}

int sgl_gpt2_nfree(const sgl_gpt2 *m) { return m->n_free; }

void sgl_gpt2_logits(const sgl_gpt2 *m, const float *x_last, float *logits) {
  /* logits[j] = dot(x, wte[j])（wte 行主序，权重 tying）*/
#pragma omp parallel for
  for (int j = 0; j < m->vocab; j++) {
    const float *row = m->wte.data + (size_t)j * m->n_embd;
    double s = 0;
    for (int k = 0; k < m->n_embd; k++) s += x_last[k] * row[k];
    logits[j] = (float)s;
  }
}

int sgl_gpt2_prefill(sgl_gpt2 *m, const int *tokens, int n, int pos0,
                     const int *slot_idx, float *logits_out) {
  int E = m->n_embd, H = m->n_head, D = E / H;
  int L = pos0 + n; /* 总序列长度 */
  if (n <= 0 || L > m->n_ctx) {
    SGL_ERROR("prefill: n=%d pos0=%d exceeds ctx=%d", n, pos0, m->n_ctx);
    return -1;
  }
  for (int p = 0; p < L; p++) {
    if (slot_idx[p] < 0 || slot_idx[p] >= m->n_pool) {
      SGL_ERROR("prefill: page %d out of pool at pos %d", slot_idx[p], p);
      return -1;
    }
  }

  /* x = wte[tokens] + wpe[pos] */
  sgl_embedding_get(m->wte.data, tokens, n, E, m->x);
  for (int t = 0; t < n; t++) {
    const float *pe = m->wpe.data + (size_t)(pos0 + t) * E;
    float *xr = m->x + (size_t)t * E;
    for (int k = 0; k < E; k++) xr[k] += pe[k];
  }

  for (int l = 0; l < m->n_layer; l++) {
    /* ---- attention block ---- */
    sgl_layernorm(m->x, m->layers[l].ln1_g.data, m->layers[l].ln1_b.data,
                  m->h, n, E, m->ln_eps);
    sgl_matmul(m->h, m->layers[l].qkv_w.data, m->layers[l].qkv_b.data,
               m->qkv, n, 3 * E, E);

    /* 写 KV cache（qkv 布局 [n, 3E]：每行 q|k|v，K/V 块不连续，须逐行拷；
     * 页表索引：位置 pos0+t → 页 slot_idx[pos0+t]）*/
    float *kc = m->kc + (size_t)l * m->n_pool * E;
    float *vc = m->vc + (size_t)l * m->n_pool * E;
    for (int t = 0; t < n; t++) {
      int pg = slot_idx[pos0 + t];
      memcpy(kc + (size_t)pg * E, m->qkv + (size_t)t * 3 * E + E,
             (size_t)E * sizeof(float));
      memcpy(vc + (size_t)pg * E, m->qkv + (size_t)t * 3 * E + 2 * E,
             (size_t)E * sizeof(float));
    }

    /* causal attention：每个 (t, head) 对全历史 [0, pos0+t]；scores 每线程一行 */
#pragma omp parallel for
    for (int t = 0; t < n; t++) {
      float *sc = m->scores;
#ifdef _OPENMP
      sc += (size_t)omp_get_thread_num() * m->n_ctx;
#endif
      const float *q = m->qkv + (size_t)t * 3 * E; /* q 在 qkv[t] 的前 E */
      for (int h = 0; h < H; h++) {
        const float *qh = q + h * D;
        int upto = pos0 + t + 1;
        float maxv = -1e30f;
        for (int p = 0; p < upto; p++) {
          const float *kh = kc + (size_t)slot_idx[p] * E + h * D;
          float s = 0;
          for (int d = 0; d < D; d++) s += qh[d] * kh[d];
          s *= 1.0f / sqrtf((float)D);
          sc[p] = s;
          if (s > maxv) maxv = s;
        }
        double sum = 0;
        for (int p = 0; p < upto; p++) {
          sc[p] = expf(sc[p] - maxv);
          sum += sc[p];
        }
        float *out = m->attn_out + (size_t)t * E + h * D;
        memset(out, 0, sizeof(float) * D);
        for (int p = 0; p < upto; p++) {
          const float *vh = vc + (size_t)slot_idx[p] * E + h * D;
          float w = sc[p] / (float)sum;
          for (int d = 0; d < D; d++) out[d] += w * vh[d];
        }
      }
    }

    sgl_matmul(m->attn_out, m->layers[l].proj_w.data, m->layers[l].proj_b.data,
               m->h, n, E, E);
    for (int t = 0; t < n; t++) {
      float *xr = m->x + (size_t)t * E;
      const float *hr = m->h + (size_t)t * E;
      for (int k = 0; k < E; k++) xr[k] += hr[k];
    }

    /* ---- MLP block ---- */
    sgl_layernorm(m->x, m->layers[l].ln2_g.data, m->layers[l].ln2_b.data,
                  m->h, n, E, m->ln_eps);
    sgl_matmul(m->h, m->layers[l].fc_w.data, m->layers[l].fc_b.data,
               m->mlp_h, n, 4 * E, E);
    sgl_gelu_inplace(m->mlp_h, n * 4 * E);
    sgl_matmul(m->mlp_h, m->layers[l].mlp_proj_w.data, m->layers[l].mlp_proj_b.data,
               m->h, n, E, 4 * E);
    for (int t = 0; t < n; t++) {
      float *xr = m->x + (size_t)t * E;
      const float *hr = m->h + (size_t)t * E;
      for (int k = 0; k < E; k++) xr[k] += hr[k];
    }
  }

  /* ln_f + 最后一位置 logits */
  sgl_layernorm(m->x, m->lnf_g.data, m->lnf_b.data, m->h, n, E, m->ln_eps);
  if (logits_out) sgl_gpt2_logits(m, m->h + (size_t)(n - 1) * E, logits_out);
  return 0;
}

/* ---- 批量 decode：B 个序列各 1 token，共享权重一次前向 ---- */
int sgl_gpt2_decode_batch(sgl_gpt2 *m, const int *tokens, int *const *slot_idx,
                          const int *seq_lens, int B, float *logits_out) {
  int E = m->n_embd, H = m->n_head, D = E / H;
  if (B <= 0 || B > SGL_GPT2_MAX_BATCH) {
    SGL_ERROR("decode_batch: B=%d (max %d)", B, SGL_GPT2_MAX_BATCH);
    return -1;
  }
  for (int b = 0; b < B; b++) {
    if (seq_lens[b] + 1 > m->n_ctx) {
      SGL_ERROR("decode_batch: req %d pos out of ctx", b);
      return -1;
    }
    for (int p = 0; p <= seq_lens[b]; p++) {
      if (slot_idx[b][p] < 0 || slot_idx[b][p] >= m->n_pool) {
        SGL_ERROR("decode_batch: req %d page out of pool at %d", b, p);
        return -1;
      }
    }
  }

  /* 嵌入：x[b] = wte[tok_b] + wpe[pos_b] */
  sgl_embedding_get(m->wte.data, tokens, B, E, m->bx);
  for (int b = 0; b < B; b++) {
    const float *pe = m->wpe.data + (size_t)seq_lens[b] * E;
    float *xr = m->bx + (size_t)b * E;
    for (int k = 0; k < E; k++) xr[k] += pe[k];
  }

  for (int l = 0; l < m->n_layer; l++) {
    sgl_layernorm(m->bx, m->layers[l].ln1_g.data, m->layers[l].ln1_b.data,
                  m->bh, B, E, m->ln_eps);
    sgl_matmul(m->bh, m->layers[l].qkv_w.data, m->layers[l].qkv_b.data,
               m->bqkv, B, 3 * E, E);

    /* 写 KV：行 b 写入页 slot_idx[b][seq_lens[b]] */
    float *kc = m->kc + (size_t)l * m->n_pool * E;
    float *vc = m->vc + (size_t)l * m->n_pool * E;
    for (int b = 0; b < B; b++) {
      int pg = slot_idx[b][seq_lens[b]];
      memcpy(kc + (size_t)pg * E,
             m->bqkv + (size_t)b * 3 * E + E, (size_t)E * sizeof(float));
      memcpy(vc + (size_t)pg * E,
             m->bqkv + (size_t)b * 3 * E + 2 * E, (size_t)E * sizeof(float));
    }

    /* 注意力：并行 (b,h)，每对独占一行 bscores */
#pragma omp parallel for
    for (int bh = 0; bh < B * H; bh++) {
      int b = bh / H, h = bh % H;
      float *sc = m->bscores + (size_t)bh * m->n_ctx;
      const float *qh = m->bqkv + (size_t)b * 3 * E + h * D;
      const int *idx = slot_idx[b];
      int upto = seq_lens[b] + 1;
      float maxv = -1e30f;
      for (int p = 0; p < upto; p++) {
        const float *kh = kc + (size_t)idx[p] * E + h * D;
        float s = 0;
        for (int d = 0; d < D; d++) s += qh[d] * kh[d];
        s *= 1.0f / sqrtf((float)D);
        sc[p] = s;
        if (s > maxv) maxv = s;
      }
      double sum = 0;
      for (int p = 0; p < upto; p++) {
        sc[p] = expf(sc[p] - maxv);
        sum += sc[p];
      }
      float *out = m->battn + (size_t)b * E + h * D;
      memset(out, 0, sizeof(float) * D);
      for (int p = 0; p < upto; p++) {
        const float *vh = vc + (size_t)idx[p] * E + h * D;
        float w = sc[p] / (float)sum;
        for (int d = 0; d < D; d++) out[d] += w * vh[d];
      }
    }

    sgl_matmul(m->battn, m->layers[l].proj_w.data, m->layers[l].proj_b.data,
               m->bh, B, E, E);
    for (int b = 0; b < B; b++) {
      float *xr = m->bx + (size_t)b * E;
      const float *hr = m->bh + (size_t)b * E;
      for (int k = 0; k < E; k++) xr[k] += hr[k];
    }

    sgl_layernorm(m->bx, m->layers[l].ln2_g.data, m->layers[l].ln2_b.data,
                  m->bh, B, E, m->ln_eps);
    sgl_matmul(m->bh, m->layers[l].fc_w.data, m->layers[l].fc_b.data,
               m->bmlp, B, 4 * E, E);
    sgl_gelu_inplace(m->bmlp, B * 4 * E);
    sgl_matmul(m->bmlp, m->layers[l].mlp_proj_w.data, m->layers[l].mlp_proj_b.data,
               m->bh, B, E, 4 * E);
    for (int b = 0; b < B; b++) {
      float *xr = m->bx + (size_t)b * E;
      const float *hr = m->bh + (size_t)b * E;
      for (int k = 0; k < E; k++) xr[k] += hr[k];
    }
  }

  sgl_layernorm(m->bx, m->lnf_g.data, m->lnf_b.data, m->bh, B, E, m->ln_eps);
  if (logits_out)
    for (int b = 0; b < B; b++)
      sgl_gpt2_logits(m, m->bh + (size_t)b * E, logits_out + (size_t)b * m->vocab);
  return 0;
}
