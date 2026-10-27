/* sgl bert.c — BERT / RoBERTa 类编码器前向 + embedding / rerank 接口
 * 对应 sglang srt/models/bert.py：
 *   BertEmbeddingModel        -> sgl_bert_encode（post-LN encoder + mean/cls pooling）
 *   BertForSequenceClassification -> sgl_bert_rerank（[CLS] 过分类头）
 * 结构：embeddings(word+position+token_type -> LayerNorm) -> N × (self-attn + add&LN +
 * FFN(gelu) + add&LN)。全序列前向、无 KV cache、无 padding mask（单条序列不需要）。
 * 权重全部走 sgl_st_*（mmap 零拷贝，F16/BF16 由 safetensors 层转 fp32）。 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/bert.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/log.h"
#include "sgl/ops.h"
#include "sgl/safetensors.h"
#include "sgl/tensor.h"

/* hidden_act 编号 */
enum { ACT_GELU_ERF = 0, ACT_GELU_TANH = 1, ACT_RELU = 2, ACT_SILU = 3, ACT_LINEAR = 4 };

typedef struct {
  sgl_tensor q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
  sgl_tensor ln2_w, ln2_b; /* attention.output.LayerNorm */
  sgl_tensor i_w, i_b;     /* intermediate.dense */
  sgl_tensor o2_w, o2_b;   /* output.dense */
  sgl_tensor ln3_w, ln3_b; /* output.LayerNorm */
} bert_layer;

struct sgl_bert {
  int E, L, NH, D, I, V, P, NT;
  int roberta, pos_offset, pad_id, max_len;
  float eps;
  int act;
  sgl_st_ctx *st;
  sgl_wp *tok;
  const char **names;
  int n_names;
  sgl_tensor wte, wpe, wtt, ln1_w, ln1_b;
  bert_layer *layers;
  sgl_tensor pool_w, pool_b;
  int has_pool;
  sgl_tensor cls_w, cls_b; /* 分类头 [n_labels, E] */
  int n_labels;
};

/* ---------- 小工具 ---------- */

static char *bert_read_file(const char *path, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  if (sz < 0) { fclose(f); return NULL; }
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    fclose(f);
    sgl_xfree(buf);
    return NULL;
  }
  fclose(f);
  buf[sz] = 0;
  if (out_len) *out_len = (size_t)sz;
  return buf;
}

static int bert_has_name(const sgl_bert *m, const char *name) {
  for (int i = 0; i < m->n_names; i++)
    if (strcmp(m->names[i], name) == 0) return 1;
  return 0;
}

/* 按前缀 × 后缀候选找一个张量；都没有 -> ok=0 返回 null tensor */
static sgl_tensor bert_find(sgl_bert *m, const char *const *sfx, int nsfx, int ndim,
                           const int64_t *shape, int *ok, char *found_name,
                           size_t found_cap) {
  static const char *PREF[] = {"", "bert.", "roberta.", "model."};
  if (ok) *ok = 0;
  for (size_t p = 0; p < sizeof(PREF) / sizeof(PREF[0]); p++)
    for (int i = 0; i < nsfx; i++) {
      char name[512];
      snprintf(name, sizeof(name), "%s%s", PREF[p], sfx[i]);
      if (!bert_has_name(m, name)) continue;
      sgl_tensor t = sgl_st_get(m->st, name, ndim, shape);
      if (!sgl_tensor_is_null(t)) {
        if (ok) *ok = 1;
        if (found_name && found_cap) snprintf(found_name, found_cap, "%s", name);
        return t;
      }
    }
  return sgl_tensor_null();
}

static sgl_tensor bert_find1(sgl_bert *m, const char *const *sfx, int nsfx, int ndim,
                             const int64_t *shape, int *ok) {
  return bert_find(m, sfx, nsfx, ndim, shape, ok, NULL, 0);
}

static void bert_act_apply(int act, float *x, int n) {
  switch (act) {
    case ACT_GELU_ERF: /* HF "gelu"：0.5x(1+erf(x/√2)) */
      for (int i = 0; i < n; i++)
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118654752f));
      break;
    case ACT_GELU_TANH: sgl_gelu_inplace(x, n); break; /* gelu_new / gelu_fast */
    case ACT_RELU:
      for (int i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
      break;
    case ACT_SILU: sgl_silu_inplace(x, n); break;
    default: break; /* linear */
  }
}

/* ---------- 多头自注意力（整段、无 mask） ---------- */

static void bert_attend(const float *q, const float *k, const float *v, float *ctx,
                        int n, int NH, int D) {
  const int rows = NH * D;
  const float scale = 1.0f / sqrtf((float)D);
  float *scs = (float *)sgl_xmalloc(sizeof(float) * (size_t)NH * (size_t)n);
#pragma omp parallel for schedule(static)
  for (int h = 0; h < NH; h++) {
    const int base = h * D;
    float *sc = scs + (size_t)h * n;
    for (int i = 0; i < n; i++) {
      const float *qi = q + (size_t)i * rows + base;
      for (int j = 0; j < n; j++) {
        const float *kj = k + (size_t)j * rows + base;
        double acc = 0;
        for (int d = 0; d < D; d++) acc += (double)qi[d] * kj[d];
        sc[j] = (float)(acc * scale);
      }
      sgl_softmax_inplace(sc, n);
      float *ci = ctx + (size_t)i * rows + base;
      for (int d = 0; d < D; d++) ci[d] = 0.0f;
      for (int j = 0; j < n; j++) {
        const float w = sc[j];
        const float *vj = v + (size_t)j * rows + base;
        for (int d = 0; d < D; d++) ci[d] += w * vj[d];
      }
    }
  }
  sgl_xfree(scs);
}

/* ---------- 整段前向：out[n,E] 为最后一层 LayerNorm 输出 ---------- */

static int bert_forward(sgl_bert *m, const int *ids, int n, float *out) {
  const int E = m->E, I = m->I, NH = m->NH, D = m->D;
  const size_t se = (size_t)n * (size_t)E;
  const size_t si = (size_t)n * (size_t)I;
  float *x = (float *)sgl_xmalloc(sizeof(float) * se);
  float *y = (float *)sgl_xmalloc(sizeof(float) * se);
  float *qkv = (float *)sgl_xmalloc(sizeof(float) * 3 * se);
  float *ctx = (float *)sgl_xmalloc(sizeof(float) * se);
  float *inter = (float *)sgl_xmalloc(sizeof(float) * (si ? si : 1));
  float *q = qkv, *k = qkv + se, *v = qkv + 2 * se;

  /* embeddings：word + position + token_type（token_type 恒 0） */
  for (int i = 0; i < n; i++) {
    if (ids[i] < 0 || ids[i] >= m->V) {
      SGL_ERROR("bert forward: token id %d 超出词表 [0,%d)", ids[i], m->V);
      goto fail;
    }
    int64_t pos = (int64_t)i + m->pos_offset;
    for (int j = 0; j < E; j++)
      x[(size_t)i * E + j] = m->wte.data[(size_t)ids[i] * E + j] +
                             m->wpe.data[(size_t)pos * E + j] + m->wtt.data[j];
  }
  sgl_layernorm(x, m->ln1_w.data, m->ln1_b.data, y, n, E, m->eps);
  memcpy(x, y, sizeof(float) * se);

  for (int l = 0; l < m->L; l++) {
    bert_layer *L = &m->layers[l];
    sgl_matmul_nt(x, L->q_w.data, L->q_b.data, q, n, E, E);
    sgl_matmul_nt(x, L->k_w.data, L->k_b.data, k, n, E, E);
    sgl_matmul_nt(x, L->v_w.data, L->v_b.data, v, n, E, E);
    bert_attend(q, k, v, ctx, n, NH, D);
    /* attention.output: dense -> residual -> LayerNorm */
    sgl_matmul_nt(ctx, L->o_w.data, L->o_b.data, y, n, E, E);
    for (size_t i = 0; i < se; i++) y[i] += x[i];
    sgl_layernorm(y, L->ln2_w.data, L->ln2_b.data, x, n, E, m->eps);
    /* intermediate -> act -> output -> residual -> LayerNorm */
    sgl_matmul_nt(x, L->i_w.data, L->i_b.data, inter, n, I, E);
    bert_act_apply(m->act, inter, n * I);
    sgl_matmul_nt(inter, L->o2_w.data, L->o2_b.data, y, n, E, I);
    for (size_t i = 0; i < se; i++) y[i] += x[i];
    sgl_layernorm(y, L->ln3_w.data, L->ln3_b.data, x, n, E, m->eps);
  }
  memcpy(out, x, sizeof(float) * se);
  sgl_xfree(inter);
  sgl_xfree(ctx);
  sgl_xfree(qkv);
  sgl_xfree(y);
  sgl_xfree(x);
  return 0;
fail:
  sgl_xfree(inter);
  sgl_xfree(ctx);
  sgl_xfree(qkv);
  sgl_xfree(y);
  sgl_xfree(x);
  return -1;
}

/* ---------- 加载 ---------- */

static int act_from_name(const char *s) {
  if (!s || !*s) return ACT_GELU_ERF;
  if (!strcmp(s, "gelu")) return ACT_GELU_ERF;
  if (!strcmp(s, "gelu_new") || !strcmp(s, "gelu_fast") ||
      !strcmp(s, "gelu_pytorch_tanh") || !strcmp(s, "gelu_approximate"))
    return ACT_GELU_TANH;
  if (!strcmp(s, "relu")) return ACT_RELU;
  if (!strcmp(s, "silu") || !strcmp(s, "swish")) return ACT_SILU;
  if (!strcmp(s, "linear") || !strcmp(s, "identity")) return ACT_LINEAR;
  SGL_WARN("bert: 未识别的 hidden_act '%s'，按 gelu(erf) 处理", s);
  return ACT_GELU_ERF;
}

sgl_bert *sgl_bert_load(const char *dir) {
  if (!dir || !*dir) return NULL;
  char path[1200];
  snprintf(path, sizeof(path), "%s/config.json", dir);
  size_t clen = 0;
  char *ctext = bert_read_file(path, &clen);
  if (!ctext) {
    SGL_ERROR("bert: 读不到 %s", path);
    return NULL;
  }
  sgl_json *cfg = sgl_json_parse(ctext, clen);
  sgl_xfree(ctext);
  if (!cfg) {
    SGL_ERROR("bert: %s JSON 解析失败", path);
    return NULL;
  }

  sgl_bert *m = (sgl_bert *)sgl_xcalloc(1, sizeof(sgl_bert));
  m->E = sgl_json_get_int(cfg, "hidden_size", 0);
  m->L = sgl_json_get_int(cfg, "num_hidden_layers", 0);
  m->NH = sgl_json_get_int(cfg, "num_attention_heads", 0);
  m->I = sgl_json_get_int(cfg, "intermediate_size", 0);
  m->V = sgl_json_get_int(cfg, "vocab_size", 0);
  m->P = sgl_json_get_int(cfg, "max_position_embeddings", 512);
  m->NT = sgl_json_get_int(cfg, "type_vocab_size", 2);
  m->eps = (float)sgl_json_get_num(cfg, "layer_norm_eps", 1e-12);
  m->act = act_from_name(sgl_json_get_str(cfg, "hidden_act", "gelu"));
  const char *mtype = sgl_json_get_str(cfg, "model_type", "bert");
  m->roberta = strcmp(mtype, "roberta") == 0;
  m->pad_id = sgl_json_get_int(cfg, "pad_token_id", m->roberta ? 1 : 0);
  m->pos_offset = m->roberta ? m->pad_id + 1 : 0;
  const char *pet = sgl_json_get_str(cfg, "position_embedding_type", "absolute");
  if (*pet && strcmp(pet, "absolute") != 0) {
    SGL_ERROR("bert: 只支持 position_embedding_type=absolute（此模型是 '%s'）", pet);
    goto fail_cfg;
  }
  if (m->E <= 0 || m->L <= 0 || m->NH <= 0 || m->I <= 0) {
    SGL_ERROR("bert: config.json 缺 hidden_size/num_hidden_layers/"
              "num_attention_heads/intermediate_size");
    goto fail_cfg;
  }
  if (m->E % m->NH) {
    SGL_ERROR("bert: hidden_size %d 不能被 num_attention_heads %d 整除", m->E, m->NH);
    goto fail_cfg;
  }
  m->D = m->E / m->NH;

  /* tokenizer + 长度上限 */
  m->tok = sgl_wp_load_dir(dir);
  if (!m->tok) goto fail_m;
  int tmax = sgl_wp_max_len(m->tok);
  int pmax = m->P - m->pos_offset;
  m->max_len = tmax < pmax ? tmax : pmax;
  if (m->max_len < 4) {
    SGL_ERROR("bert: max_len 太小（tokenizer=%d, positions=%d）", tmax, pmax);
    goto fail_m;
  }

  /* safetensors */
  m->st = sgl_st_open_dir(dir);
  if (!m->st) goto fail_m;
  m->n_names = sgl_st_names(m->st, &m->names);
  if (m->n_names <= 0) {
    SGL_ERROR("bert: %s 里没有张量", dir);
    goto fail_m;
  }

  int ok = 0;
  { /* word embeddings：vocab_size 可能与 tokenizer 差一个 pad，二者都试 */
    static const char *sfx[] = {"embeddings.word_embeddings.weight",
                                "word_embeddings.weight"};
    int64_t sh[2] = {m->V, m->E};
    m->wte = bert_find1(m, sfx, 2, 2, sh, &ok);
    if (!ok) {
      m->V = sgl_wp_vocab_size(m->tok);
      sh[0] = m->V;
      m->wte = bert_find1(m, sfx, 2, 2, sh, &ok);
    }
    if (!ok) {
      SGL_ERROR("bert: 找不到 embeddings.word_embeddings.weight [V=%d,E=%d]", m->V, m->E);
      goto fail_m;
    }
    m->V = (int)m->wte.shape[0]; /* 以真实权重为准 */
  }
  {
    static const char *sfx[] = {"embeddings.position_embeddings.weight"};
    int64_t sh[2] = {m->P, m->E};
    m->wpe = bert_find1(m, sfx, 1, 2, sh, &ok);
    if (!ok) {
      SGL_ERROR("bert: 找不到 position_embeddings [%d,%d]", m->P, m->E);
      goto fail_m;
    }
  }
  {
    static const char *sfx[] = {"embeddings.token_type_embeddings.weight"};
    int64_t sh[2] = {m->NT, m->E};
    m->wtt = bert_find1(m, sfx, 1, 2, sh, &ok);
    if (!ok) { /* 少数模型省掉 token_type（等价全零） */
      int64_t sh1[2] = {1, m->E};
      m->wtt = sgl_tensor_zeros(2, sh1);
      SGL_WARN("bert: 无 token_type_embeddings，按零向量处理");
    }
  }
  {
    static const char *sfx[] = {"embeddings.LayerNorm.weight", "embeddings.LayerNorm.gamma"};
    int64_t sh[1] = {m->E};
    m->ln1_w = bert_find1(m, sfx, 2, 1, sh, &ok);
    if (!ok) {
      SGL_ERROR("bert: 找不到 embeddings.LayerNorm.weight");
      goto fail_m;
    }
    static const char *sfxb[] = {"embeddings.LayerNorm.bias", "embeddings.LayerNorm.beta"};
    m->ln1_b = bert_find1(m, sfxb, 2, 1, sh, &ok);
    if (!ok) {
      SGL_ERROR("bert: 找不到 embeddings.LayerNorm.bias");
      goto fail_m;
    }
  }

  m->layers = (bert_layer *)sgl_xcalloc((size_t)m->L, sizeof(bert_layer));
  for (int l = 0; l < m->L; l++) {
    bert_layer *L = &m->layers[l];
    int64_t sE[1] = {m->E};
    int64_t m2[2] = {m->E, m->E}, m2i[2] = {m->I, m->E}, m2o[2] = {m->E, m->I};
    /* 6 个投影：attention 的 query/key/value、attention.output.dense、
     * intermediate.dense、output.dense。bsh 是 bias 长度（该层的输出维）。*/
    struct { const char *sfx; sgl_tensor *w; sgl_tensor *b; const int64_t *sh;
             int64_t bsh; } qkvp[] = {
        {"attention.self.query.weight", &L->q_w, &L->q_b, m2, m->E},
        {"attention.self.key.weight", &L->k_w, &L->k_b, m2, m->E},
        {"attention.self.value.weight", &L->v_w, &L->v_b, m2, m->E},
        {"attention.output.dense.weight", &L->o_w, &L->o_b, m2, m->E},
        {"intermediate.dense.weight", &L->i_w, &L->i_b, m2i, m->I},
        {"output.dense.weight", &L->o2_w, &L->o2_b, m2o, m->E},
    };
    for (size_t j = 0; j < sizeof(qkvp) / sizeof(qkvp[0]); j++) {
      char wname[512], bname[512];
      snprintf(wname, sizeof(wname), "encoder.layer.%d.%s", l, qkvp[j].sfx);
      const char *sfxw[1] = {wname};
      *qkvp[j].w = bert_find1(m, sfxw, 1, 2, qkvp[j].sh, &ok);
      if (!ok) {
        SGL_ERROR("bert: 找不到 %s（layer %d）", wname, l);
        goto fail_m;
      }
      /* bias：缺失时视为 0（attention_bias=False 的 RoBERTa 变体没有）*/
      snprintf(bname, sizeof(bname), "%.*sbias",
               (int)(strlen(wname) - strlen("weight")), wname);
      const char *sfxb[1] = {bname};
      int64_t bsh[1] = {qkvp[j].bsh};
      *qkvp[j].b = bert_find1(m, sfxb, 1, 1, bsh, &ok);
      if (!ok) {
        float *zeros = (float *)sgl_xmalloc(sizeof(float) * (size_t)bsh[0]);
        memset(zeros, 0, sizeof(float) * (size_t)bsh[0]);
        *qkvp[j].b = sgl_tensor_from_data(1, bsh, zeros);
        (*qkvp[j].b)._owner = zeros;
      }
    }
    struct { const char *sfx; sgl_tensor *w; sgl_tensor *b; } lns[] = {
        {"attention.output.LayerNorm", &L->ln2_w, &L->ln2_b},
        {"output.LayerNorm", &L->ln3_w, &L->ln3_b},
    };
    for (size_t j = 0; j < sizeof(lns) / sizeof(lns[0]); j++) {
      char wname[512], bname[512];
      snprintf(wname, sizeof(wname), "encoder.layer.%d.%s.weight", l, lns[j].sfx);
      snprintf(bname, sizeof(bname), "encoder.layer.%d.%s.bias", l, lns[j].sfx);
      const char *sfxw[1] = {wname};
      const char *sfxb[1] = {bname};
      *lns[j].w = bert_find1(m, sfxw, 1, 1, sE, &ok);
      if (!ok) {
        SGL_ERROR("bert: 找不到 %s", wname);
        goto fail_m;
      }
      *lns[j].b = bert_find1(m, sfxb, 1, 1, sE, &ok);
      if (!ok) {
        SGL_ERROR("bert: 找不到 %s", bname);
        goto fail_m;
      }
    }
    (void)0;
  }

  /* pooler（可选，BertModel 有；纯 encoder 权重集可能没有）*/
  {
    static const char *sfx[] = {"pooler.dense.weight"};
    int64_t sh[2] = {m->E, m->E};
    m->pool_w = bert_find1(m, sfx, 1, 2, sh, &ok);
    m->has_pool = ok;
    if (ok) {
      static const char *sfxb[] = {"pooler.dense.bias"};
      int64_t sh1[1] = {m->E};
      m->pool_b = bert_find1(m, sfxb, 1, 1, sh1, &ok);
      if (!ok) m->has_pool = 0;
    }
  }
  /* 分类头（rerank）：classifier.weight / score.weight，[n_labels, E] */
  {
    static const char *sfx[] = {"classifier.weight", "score.weight", "classifier.dense.weight"};
    int64_t sh[2] = {1, m->E};
    sgl_tensor w = bert_find1(m, sfx, 3, 2, sh, &ok);
    if (!ok) sh[0] = 2, w = bert_find1(m, sfx, 3, 2, sh, &ok);
    if (ok) {
      m->cls_w = w;
      m->n_labels = (int)sh[0];
      static const char *sfxb[] = {"classifier.bias", "score.bias"};
      int64_t sh1[1] = {m->n_labels};
      m->cls_b = bert_find1(m, sfxb, 2, 1, sh1, &ok);
      if (!ok) {
        float *zeros = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->n_labels);
        memset(zeros, 0, sizeof(float) * (size_t)m->n_labels);
        m->cls_b = sgl_tensor_from_data(1, sh1, zeros);
        m->cls_b._owner = zeros;
      }
    }
  }

  SGL_INFO("bert: %s type=%s E=%d L=%d NH=%d D=%d I=%d V=%d max_len=%d "
           "pos_offset=%d eps=%g act=%d pooler=%d labels=%d",
           dir, m->roberta ? "roberta" : "bert", m->E, m->L, m->NH, m->D, m->I, m->V,
           m->max_len, m->pos_offset, (double)m->eps, m->act, m->has_pool, m->n_labels);
  sgl_json_free(cfg);
  return m;

fail_m:
  sgl_json_free(cfg);
  sgl_bert_free(m);
  return NULL;
fail_cfg:
  sgl_json_free(cfg);
  sgl_bert_free(m);
  return NULL;
}

void sgl_bert_free(sgl_bert *m) {
  if (!m) return;
  if (m->layers) {
    for (int l = 0; l < m->L; l++) {
      bert_layer *L = &m->layers[l];
      sgl_tensor ts[] = {L->q_w, L->q_b, L->k_w, L->k_b, L->v_w, L->v_b,
                         L->o_w, L->o_b, L->ln2_w, L->ln2_b, L->i_w,
                         L->i_b, L->o2_w, L->o2_b, L->ln3_w, L->ln3_b};
      for (size_t i = 0; i < sizeof(ts) / sizeof(ts[0]); i++)
        if (!sgl_tensor_is_null(ts[i]) && ts[i]._owner) sgl_tensor_free(&ts[i]);
    }
    sgl_xfree(m->layers);
  }
  sgl_tensor ts[] = {m->wte, m->wpe, m->wtt, m->ln1_w, m->ln1_b,
                     m->pool_w, m->pool_b, m->cls_w, m->cls_b};
  for (size_t i = 0; i < sizeof(ts) / sizeof(ts[0]); i++)
    if (!sgl_tensor_is_null(ts[i]) && ts[i]._owner) sgl_tensor_free(&ts[i]);
  if (m->tok) sgl_wp_free(m->tok);
  if (m->st) sgl_st_close(m->st);
  sgl_xfree(m);
}

int sgl_bert_hidden(const sgl_bert *m) { return m ? m->E : 0; }

/* ---------- encode ---------- */

int sgl_bert_encode(sgl_bert *m, const char *text, int pooling, float *out,
                    int out_cap) {
  if (!m || !text || !out || out_cap < m->E) {
    SGL_ERROR("bert encode: 参数非法（out_cap=%d < hidden=%d？）", out_cap,
              m ? m->E : 0);
    return -1;
  }
  int cap = m->max_len + 2;
  int *ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap);
  int n = sgl_wp_encode_ids(m->tok, text, ids, cap);
  if (n <= 0) {
    SGL_ERROR("bert encode: tokenize 失败（%s）", text);
    sgl_xfree(ids);
    return -1;
  }
  float *h = (float *)sgl_xmalloc(sizeof(float) * (size_t)n * (size_t)m->E);
  if (bert_forward(m, ids, n, h) != 0) {
    sgl_xfree(h);
    sgl_xfree(ids);
    return -1;
  }
  if (pooling == SGL_BERT_POOL_CLS) {
    for (int j = 0; j < m->E; j++) out[j] = h[j];
  } else {
    /* mean pooling：单条序列无 padding，全体 token（含 [CLS]/[SEP]）求均值，
     * 与 sentence-transformers 用 attention_mask（padding 全 1）的结果一致 */
    for (int j = 0; j < m->E; j++) out[j] = 0.0f;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m->E; j++) out[j] += h[(size_t)i * m->E + j];
    const float inv = 1.0f / (float)n;
    for (int j = 0; j < m->E; j++) out[j] *= inv;
  }
  double ss = 0;
  for (int j = 0; j < m->E; j++) ss += (double)out[j] * out[j];
  double norm = sqrt(ss);
  if (norm < 1e-12) {
    SGL_ERROR("bert encode: 输出范数 ~0，无法 L2 normalize");
    sgl_xfree(h);
    sgl_xfree(ids);
    return -1;
  }
  const float invn = (float)(1.0 / norm);
  for (int j = 0; j < m->E; j++) out[j] *= invn;
  sgl_xfree(h);
  sgl_xfree(ids);
  return 0;
}

/* ---------- rerank ---------- */

int sgl_bert_pair_ids(sgl_bert *m, const char *query, const char *doc, int *ids,
                      int max_len) {
  int cap = m->max_len + 2;
  int *qid = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap);
  int *did = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap);
  int qn = sgl_wp_encode_plain(m->tok, query, qid, cap);
  int dn = sgl_wp_encode_plain(m->tok, doc, did, cap);
  int cls = sgl_wp_cls_id(m->tok), sep = sgl_wp_sep_id(m->tok);
  int n = 0, avail = max_len - 3; /* [CLS] + A + [SEP] + B + [SEP] 里的 3 个特殊位 */
  int extra_sep = m->roberta ? 1 : 0; /* RoBERTa 的 pair 是 </s></s> */
  avail -= extra_sep;
  if (avail < 1) avail = 1;
  int qn2 = qn < avail / 2 + avail % 2 ? qn : avail / 2 + avail % 2;
  int dn2 = dn < avail - qn2 ? dn : avail - qn2;
  ids[n++] = cls;
  for (int i = 0; i < qn2; i++) ids[n++] = qid[i];
  ids[n++] = sep;
  if (extra_sep) ids[n++] = sep;
  for (int i = 0; i < dn2; i++) ids[n++] = did[i];
  ids[n++] = sep;
  sgl_xfree(qid);
  sgl_xfree(did);
  return n;
}

float sgl_bert_rerank(sgl_bert *m, const char *query, const char *doc) {
  if (!m || !query || !doc) return NAN;
  int cap = m->max_len + 4;
  int *ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap);
  int n = sgl_bert_pair_ids(m, query, doc, ids, m->max_len);
  if (n <= 0) {
    sgl_xfree(ids);
    return NAN;
  }
  float *h = (float *)sgl_xmalloc(sizeof(float) * (size_t)n * (size_t)m->E);
  if (bert_forward(m, ids, n, h) != 0) {
    sgl_xfree(h);
    sgl_xfree(ids);
    return NAN;
  }
  const float *cls = h; /* [CLS] 位置的最终隐藏态 */
  float score;
  if (m->n_labels == 1) {
    double acc = m->cls_b.data[0];
    for (int j = 0; j < m->E; j++) acc += (double)cls[j] * m->cls_w.data[j];
    score = (float)acc;
  } else if (m->n_labels >= 2) {
    double l0 = m->cls_b.data[0], l1 = m->cls_b.data[1];
    for (int j = 0; j < m->E; j++) {
      l0 += (double)cls[j] * m->cls_w.data[j];
      l1 += (double)cls[j] * m->cls_w.data[(size_t)m->E + j];
    }
    score = (float)l1;
    (void)l0;
  } else {
    /* 无分类头：退化为 bi-encoder 余弦相似度（两次 encode 的 [CLS] 向量） */
    float *a = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->E);
    float *b = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->E);
    double acc = 0;
    if (sgl_bert_encode(m, query, SGL_BERT_POOL_CLS, a, m->E) != 0 ||
        sgl_bert_encode(m, doc, SGL_BERT_POOL_CLS, b, m->E) != 0) {
      acc = NAN;
    } else {
      for (int j = 0; j < m->E; j++) acc += (double)a[j] * b[j];
    }
    score = (float)acc;
    sgl_xfree(a);
    sgl_xfree(b);
  }
  sgl_xfree(h);
  sgl_xfree(ids);
  return score;
}