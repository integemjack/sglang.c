/* sgl mla 实现：DeepSeek 系 MLA 前向（吸收式 / absorbed，见 mla.h 顶部说明）
 * 数值公式照 HF modeling_deepseek_v2.py 的 DeepseekV2Attention.forward：
 *   q = q_b_proj(q_a_layernorm(q_a_proj(x)))
 *   kv_a = kv_a_proj_with_mqa(x) → [c | k_pe]；c = kv_a_layernorm(c)
 *   kv = kv_b_proj(c) → 每头 [k_nope | v]
 *   RoPE（非交错 rotate_half）只打在 q_pe（每头）和 k_pe（单份、全 head 共享）
 *   score = (q_nope·k_nope + q_pe·k_pe) * q_head_dim^-0.5，causal softmax，再乘 v
 *   out = o_proj(concat_h(attn_h))
 * 吸收式把 k_nope = W_UK[h] c 和 v = W_UV[h] c 的 W 折进 q / 输出，见 mla.h。
 * 只依赖 sgl/ops.h 的算子 + sgl_xmalloc/sgl_xfree。 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/mla.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/log.h"
#include "sgl/ops.h"

struct sgl_mla {
  /* ---- config ---- */
  int hidden;    /* hidden_size */
  int heads;     /* num_attention_heads */
  int q_lora;    /* q_lora_rank；0 = config 为 null（无 q 低秩，直接 q_proj）*/
  int kv_lora;   /* kv_lora_rank（压缩 latent 维；DeepSeek-V2/V3 为 512）*/
  int qk_nope;   /* qk_nope_head_dim（每头不参与 rope 的 q/k 维）*/
  int qk_rope;   /* qk_rope_head_dim（解耦 rope 维；V2/V3 为 64）*/
  int v_head;    /* v_head_dim（每头 v / 输出维；V2 为 128）*/
  int q_head_dim;/* qk_nope + qk_rope（HF 的 softmax_scale = q_head_dim^-0.5）*/
  int stride;    /* KV 行宽 = kv_lora + qk_rope */
  int ctx;       /* max_position_embeddings */
  float rms_eps; /* rms_norm_eps */
  double rope_theta;
  /* rope_scaling：字段照读，【不实现】yarn/linear 频率缩放（非空时打 warning，
   * 加载真实 DeepSeek 权重（rope_scaling=yarn, mscale_all_dim=1）时数值会与官方不同）*/
  int has_rope_scaling;
  double rope_factor;
  double rope_mscale;
  double rope_mscale_all_dim;
  float softmax_scale;
  /* ---- 权重（F32 时零拷贝指向 mmap；F16/BF16 时是 st ctx 持有的 fp32 转换缓冲）---- */
  sgl_tensor w_qa;   /* [q_lora][hidden]     q_a_proj（q_lora==0 时为空）*/
  sgl_tensor w_qan;  /* [q_lora]             q_a_layernorm.weight */
  sgl_tensor w_qb;   /* [heads*q_head_dim][q_lora]  q_b_proj（q_lora==0 时是 [..*][hidden] 的 q_proj）*/
  sgl_tensor w_kva;  /* [kv_lora+qk_rope][hidden]  kv_a_proj_with_mqa */
  sgl_tensor w_kvan; /* [kv_lora]            kv_a_layernorm.weight */
  sgl_tensor w_kvb;  /* [heads*(qk_nope+v_head)][kv_lora]  kv_b_proj（W_UK/W_UV 的来源）*/
  sgl_tensor w_o;    /* [hidden][heads*v_head]  o_proj */
  sgl_tensor b_qa, b_kva, b_o; /* 可选 bias（attention_bias=true 的 config 才有）*/
  sgl_st_ctx *st;
  /* ---- scratch（按 n 懒增长，全部 sgl_xmalloc/_realloc）---- */
  int cap;      /* 各 [cap][*] 缓冲的行容量 */
  int sc_cap;   /* 每线程 score 行长度（= 已见过的最大 position+1）*/
  int nthread;  /* omp 线程数（score / q_abs 按线程切）*/
  float *qa;    /* [cap][q_lora]         q_a_proj 输出 */
  float *q;     /* [cap][heads*q_head_dim]  q_pe 已 rope */
  float *kva;   /* [cap][kv_lora+qk_rope]  kv_a 投影输出（k_pe 已 rope）*/
  float *lat;   /* [cap][kv_lora]       latent（rmsnorm 之后）*/
  float *att;   /* [cap][heads*v_head]   各头 attention 输出（o_proj 之前）*/
  float *sc;    /* [nthread][sc_cap]    attention 分数 */
  float *qa_t;  /* [nthread][heads*kv_lora + kv_lora]  每线程 q_abs + o_lat */
};

/* ---------------- 小工具 ---------------- */

static char *path_join_(const char *a, const char *b) {
  size_t la = strlen(a), lb = strlen(b);
  char *s = (char *)sgl_xmalloc(la + lb + 2);
  memcpy(s, a, la);
  size_t p = la;
  if (la && s[la - 1] != '/' && s[la - 1] != '\\') s[p++] = '/';
  memcpy(s + p, b, lb + 1);
  return s;
}

/* 取 2D 权重；形状不符或缺失时报错并返回 null tensor */
static sgl_tensor get_2d_(sgl_st_ctx *st, const char *prefix, const char *suffix,
                          int64_t rows, int64_t cols) {
  char name[600];
  snprintf(name, sizeof(name), "%s%s", prefix, suffix);
  int64_t sh[2] = {rows, cols};
  sgl_tensor t = sgl_st_get(st, name, 2, sh);
  if (sgl_tensor_is_null(t))
    SGL_ERROR("mla: tensor '%s' missing or shape != [%lld,%lld]", name, (long long)rows,
              (long long)cols);
  return t;
}

static sgl_tensor get_1d_(sgl_st_ctx *st, const char *prefix, const char *suffix, int64_t n) {
  char name[600];
  snprintf(name, sizeof(name), "%s%s", prefix, suffix);
  int64_t sh[1] = {n};
  sgl_tensor t = sgl_st_get(st, name, 1, sh);
  if (sgl_tensor_is_null(t))
    SGL_ERROR("mla: tensor '%s' missing or shape != [%lld]", name, (long long)n);
  return t;
}

/* 权重是否存在（可选 bias 用：不存在时不想让 sgl_st_get 报 ERROR）*/
static int has_tensor_(sgl_st_ctx *st, const char *prefix, const char *suffix) {
  char name[600];
  snprintf(name, sizeof(name), "%s%s", prefix, suffix);
  const char **names = NULL;
  int n = sgl_st_names(st, &names);
  for (int i = 0; i < n; i++)
    if (names[i] && !strcmp(names[i], name)) return 1;
  return 0;
}

static sgl_tensor get_opt_(sgl_st_ctx *st, const char *prefix, const char *suffix, int64_t n) {
  if (!has_tensor_(st, prefix, suffix)) return sgl_tensor_null();
  return get_1d_(st, prefix, suffix, n);
}

/* ---------------- 加载 ---------------- */

sgl_mla *sgl_mla_load(const char *model_dir) { return sgl_mla_load_layer(model_dir, 0); }

sgl_mla *sgl_mla_load_layer(const char *model_dir, int layer) {
  if (!model_dir || !*model_dir) return NULL;
  char *cpath = path_join_(model_dir, "config.json");
  FILE *f = NULL;
  if (fopen_s(&f, cpath, "rb") != 0 || !f) f = fopen(cpath, "rb");
  sgl_xfree(cpath);
  if (!f) {
    SGL_ERROR("mla: cannot open config.json in %s", model_dir);
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0) {
    fclose(f);
    return NULL;
  }
  char *buf = (char *)sgl_xmalloc((size_t)sz + 1);
  size_t got = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[got] = 0;
  sgl_json *cfg = sgl_json_parse(buf, got);
  sgl_xfree(buf);
  if (!cfg) return NULL;

  sgl_mla *m = (sgl_mla *)sgl_xcalloc(1, sizeof(*m));
  m->hidden = sgl_json_get_int(cfg, "hidden_size", 0);
  m->heads = sgl_json_get_int(cfg, "num_attention_heads", 0);
  m->q_lora = sgl_json_get_int(cfg, "q_lora_rank", 0); /* null → 0 */
  m->kv_lora = sgl_json_get_int(cfg, "kv_lora_rank", 0);
  m->qk_nope = sgl_json_get_int(cfg, "qk_nope_head_dim", 0);
  m->qk_rope = sgl_json_get_int(cfg, "qk_rope_head_dim", 0);
  m->v_head = sgl_json_get_int(cfg, "v_head_dim", m->qk_nope);
  m->ctx = sgl_json_get_int(cfg, "max_position_embeddings", 4096);
  m->rms_eps = (float)sgl_json_get_num(cfg, "rms_norm_eps", 1e-6);
  m->rope_theta = sgl_json_get_num(cfg, "rope_theta", 10000.0);
  {
    const sgl_json *rs = sgl_json_get(cfg, "rope_scaling");
    if (rs && rs->type == SGL_JSON_OBJECT) {
      m->has_rope_scaling = 1;
      m->rope_factor = sgl_json_get_num(rs, "factor", 1.0);
      m->rope_mscale = sgl_json_get_num(rs, "mscale", 0.0);
      m->rope_mscale_all_dim = sgl_json_get_num(rs, "mscale_all_dim", 0.0);
      SGL_WARN("mla: config has rope_scaling (factor=%g mscale=%g mscale_all_dim=%g)；"
               "本实现不做 yarn 频率缩放/mscale，真实 DeepSeek 权重数值会与官方不同",
               m->rope_factor, m->rope_mscale, m->rope_mscale_all_dim);
    }
  }
  sgl_json_free(cfg);

  if (m->hidden <= 0 || m->heads <= 0 || m->kv_lora <= 0 || m->qk_nope <= 0 ||
      m->qk_rope <= 0 || m->v_head <= 0 || (m->qk_rope & 1) || (m->q_lora < 0)) {
    SGL_ERROR("mla: bad config hidden=%d heads=%d q_lora=%d kv_lora=%d nope=%d rope=%d v=%d",
              m->hidden, m->heads, m->q_lora, m->kv_lora, m->qk_nope, m->qk_rope, m->v_head);
    sgl_xfree(m);
    return NULL;
  }
  m->q_head_dim = m->qk_nope + m->qk_rope;
  m->stride = m->kv_lora + m->qk_rope;
  /* HF: self.softmax_scale = self.q_head_dim ** (-0.5)（yarn 的 mscale 未实现，见上）*/
  m->softmax_scale = (float)(1.0 / sqrt((double)m->q_head_dim));
  m->nthread = 1;
#ifdef _OPENMP
  m->nthread = omp_get_max_threads();
  if (m->nthread < 1) m->nthread = 1;
#endif

  /* ---- 权重 ---- */
  m->st = sgl_st_open_dir(model_dir);
  if (!m->st) {
    SGL_ERROR("mla: no weights in %s", model_dir);
    sgl_xfree(m);
    return NULL;
  }
  char pfx[512];
  snprintf(pfx, sizeof(pfx), "model.layers.%d.self_attn.", layer);
  int q_dim = m->heads * m->q_head_dim;
  int kvb_rows = m->heads * (m->qk_nope + m->v_head);
  if (m->q_lora > 0) {
    m->w_qa = get_2d_(m->st, pfx, "q_a_proj.weight", m->q_lora, m->hidden);
    m->w_qan = get_1d_(m->st, pfx, "q_a_layernorm.weight", m->q_lora);
    m->w_qb = get_2d_(m->st, pfx, "q_b_proj.weight", q_dim, m->q_lora);
  } else {
    /* 无 q 低秩：HF 用单个 q_proj（存进 w_qb，w_qa/w_qan 为空，前向跳过低秩两跳）*/
    m->w_qb = get_2d_(m->st, pfx, "q_proj.weight", q_dim, m->hidden);
  }
  m->w_kva = get_2d_(m->st, pfx, "kv_a_proj_with_mqa.weight", m->stride, m->hidden);
  m->w_kvan = get_1d_(m->st, pfx, "kv_a_layernorm.weight", m->kv_lora);
  m->w_kvb = get_2d_(m->st, pfx, "kv_b_proj.weight", kvb_rows, m->kv_lora);
  m->w_o = get_2d_(m->st, pfx, "o_proj.weight", m->hidden, m->heads * m->v_head);
  if (sgl_tensor_is_null(m->w_qb) || sgl_tensor_is_null(m->w_kva) ||
      sgl_tensor_is_null(m->w_kvan) || sgl_tensor_is_null(m->w_kvb) ||
      sgl_tensor_is_null(m->w_o) ||
      (m->q_lora > 0 && (sgl_tensor_is_null(m->w_qa) || sgl_tensor_is_null(m->w_qan)))) {
    sgl_mla_free(m);
    return NULL;
  }
  m->b_qa = get_opt_(m->st, pfx, m->q_lora > 0 ? "q_a_proj.bias" : "q_proj.bias",
                     m->q_lora > 0 ? m->q_lora : q_dim);
  m->b_kva = get_opt_(m->st, pfx, "kv_a_proj_with_mqa.bias", m->stride);
  m->b_o = get_opt_(m->st, pfx, "o_proj.bias", m->hidden);

  SGL_INFO("mla: hidden=%d heads=%d q_lora=%d kv_lora=%d nope=%d rope=%d v=%d ctx=%d "
           "softmax_scale=%.6f rope_theta=%g%s",
           m->hidden, m->heads, m->q_lora, m->kv_lora, m->qk_nope, m->qk_rope, m->v_head,
           m->ctx, m->softmax_scale, m->rope_theta,
           m->has_rope_scaling ? " (rope_scaling ignored)" : "");
  return m;
}

void sgl_mla_free(sgl_mla *m) {
  if (!m) return;
  sgl_xfree(m->qa);
  sgl_xfree(m->q);
  sgl_xfree(m->kva);
  sgl_xfree(m->lat);
  sgl_xfree(m->att);
  sgl_xfree(m->sc);
  sgl_xfree(m->qa_t);
  if (m->st) sgl_st_close(m->st); /* F16/BF16 的 fp32 转换缓冲归 st 持有 */
  sgl_xfree(m);
}

/* ---------------- KV 缓冲 ---------------- */

sgl_kv_cache *sgl_mla_cache_new(const sgl_mla *m, int cap) {
  if (!m || cap <= 0) return NULL;
  sgl_kv_cache *c = (sgl_kv_cache *)sgl_xcalloc(1, sizeof(*c));
  c->cap = cap;
  c->stride = m->stride;
  c->n_used = 0;
  c->data = (float *)sgl_xcalloc((size_t)cap * m->stride, sizeof(float));
  return c;
}

void sgl_mla_cache_reset(sgl_kv_cache *c) {
  if (c) c->n_used = 0;
}

void sgl_mla_cache_free(sgl_kv_cache *c) {
  if (!c) return;
  sgl_xfree(c->data);
  sgl_xfree(c);
}

/* ---------------- 前向 ---------------- */

/* scratch 按需增长：q/kv 行数 n，score 行长度 maxpos（= 最大 position+1）*/
static int ensure_scratch_(sgl_mla *m, int n, int maxpos) {
  if (n > m->cap) {
    int ncap = m->cap > 0 ? m->cap : 8;
    while (ncap < n) ncap *= 2;
    m->qa = (float *)sgl_xrealloc(m->qa, sizeof(float) * (size_t)ncap * (m->q_lora > 0 ? m->q_lora : 1));
    m->q = (float *)sgl_xrealloc(m->q, sizeof(float) * (size_t)ncap * m->heads * m->q_head_dim);
    m->kva = (float *)sgl_xrealloc(m->kva, sizeof(float) * (size_t)ncap * m->stride);
    m->lat = (float *)sgl_xrealloc(m->lat, sizeof(float) * (size_t)ncap * m->kv_lora);
    m->att = (float *)sgl_xrealloc(m->att, sizeof(float) * (size_t)ncap * m->heads * m->v_head);
    m->cap = ncap;
  }
  if (maxpos > m->sc_cap) {
    int nc = m->sc_cap > 0 ? m->sc_cap : 16;
    while (nc < maxpos) nc *= 2;
    m->sc = (float *)sgl_xrealloc(m->sc, sizeof(float) * (size_t)m->nthread * nc);
    m->qa_t = (float *)sgl_xrealloc(
        m->qa_t, sizeof(float) * (size_t)m->nthread * ((size_t)m->heads * m->kv_lora + m->kv_lora));
    m->sc_cap = nc;
  }
  return 0;
}

int sgl_mla_forward(sgl_mla *m, const float *x, int n, const int *pos, float *out,
                    sgl_kv_cache *cache) {
  if (!m || !x || !pos || !out || !cache || !cache->data) return -1;
  if (n <= 0) {
    SGL_ERROR("mla forward: n=%d", n);
    return -1;
  }
  if (cache->stride != m->stride) {
    SGL_ERROR("mla forward: cache stride %d != model stride %d", cache->stride, m->stride);
    return -1;
  }
  int maxpos = 0;
  for (int t = 0; t < n; t++) {
    if (pos[t] < 0 || pos[t] >= cache->cap) {
      SGL_ERROR("mla forward: pos=%d out of cache cap %d (token %d)", pos[t], cache->cap, t);
      return -1;
    }
    if (pos[t] + 1 > maxpos) maxpos = pos[t] + 1;
  }
  ensure_scratch_(m, n, maxpos);

  const int H = m->heads, Dn = m->qk_nope, Dr = m->qk_rope, Dc = m->kv_lora;
  const int Dh = m->hidden, stride = m->stride, qhd = m->q_head_dim;
  const int Dv = m->v_head, per_head = (Dn + Dv) * Dc;

  /* 1) q 路：低秩（q_a_proj → RMSNorm → q_b_proj）或直接 q_proj */
  const float *qsrc;
  int qsrc_dim;
  if (m->q_lora > 0) {
    sgl_matmul_nt(x, m->w_qa.data, m->b_qa.data, m->qa, n, m->q_lora, Dh);
    sgl_rmsnorm(m->qa, m->w_qan.data, m->qa, n, m->q_lora, m->rms_eps);
    qsrc = m->qa;
    qsrc_dim = m->q_lora;
  } else {
    qsrc = x;
    qsrc_dim = Dh;
  }
  sgl_matmul_nt(qsrc, m->w_qb.data, m->q_lora > 0 ? NULL : m->b_qa.data, m->q, n, H * qhd,
                qsrc_dim);

  /* 2) kv 路：kv_a_proj_with_mqa → [latent | k_pe] */
  sgl_matmul_nt(x, m->w_kva.data, m->b_kva.data, m->kva, n, stride, Dh);

  /* 3) 逐 token：q_pe / k_pe 的 RoPE（非交错 rotate_half，只打在 rope 那一段），
   *    写 cache 行（latent 过 RMSNorm；k_pe 全 head 共享一份）*/
  for (int t = 0; t < n; t++) {
    float *kva_row = m->kva + (size_t)t * stride;
    float *q_row = m->q + (size_t)t * H * qhd;
    int p = pos[t];
    /* k_pe：kv_a 输出尾部连续 qk_rope 维 */
    sgl_rope_inplace(kva_row + Dc, 1, Dr, p, m->rope_theta);
    /* q_pe：每头 head 内偏移 qk_nope 起、长 qk_rope 的连续切片 */
    for (int h = 0; h < H; h++)
      sgl_rope_inplace(q_row + (size_t)h * qhd + Dn, 1, Dr, p, m->rope_theta);
    /* latent 的 rmsnorm 要连续行，先抽到 [n][Dc] 缓冲 */
    memcpy(m->lat + (size_t)t * Dc, kva_row, sizeof(float) * Dc);
    /* cache 行 = [latent(待 rmsnorm 后回填) | k_pe] */
    memcpy(cache->data + (size_t)p * stride + Dc, kva_row + Dc, sizeof(float) * Dr);
    if (p + 1 > cache->n_used) cache->n_used = p + 1;
  }
  sgl_rmsnorm(m->lat, m->w_kvan.data, m->lat, n, Dc, m->rms_eps);
  for (int t = 0; t < n; t++)
    memcpy(cache->data + (size_t)pos[t] * stride, m->lat + (size_t)t * Dc,
           sizeof(float) * Dc);

  /* 4) 吸收式 attention：q_abs[h] = W_UK[h]^T q_nope[h]（W_UK[h] 是 kv_b_proj 里
   *    该头 k_nope 那 Dn 行，行主序 [Dn][Dc] → sgl_matmul 的 M=1 即 A·W^T 的转置形），
   *    score = q_abs·c + q_pe·k_pe，softmax 后 o_lat = Σ a c，o[h] = W_UV[h]·o_lat。 */
  const float *kvb = m->w_kvb.data;
#pragma omp parallel for schedule(dynamic, 4)
  for (int t = 0; t < n; t++) {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    float *sc = m->sc + (size_t)tid * m->sc_cap;
    float *qabs = m->qa_t + (size_t)tid * ((size_t)H * Dc + Dc);
    float *olat = qabs + (size_t)H * Dc;
    const float *q_row = m->q + (size_t)t * H * qhd;
    float *o_row = m->att + (size_t)t * H * Dv;
    const int np = pos[t] + 1; /* causal：位置 0..pos[t] 全部可见 */
    for (int h = 0; h < H; h++) {
      const float *q_nope = q_row + (size_t)h * qhd;
      const float *q_pe = q_nope + Dn;
      const float *uk = kvb + (size_t)h * per_head;         /* [Dn][Dc] */
      const float *uv = uk + (size_t)Dn * Dc;               /* [Dv][Dc] */
      float *q_abs = qabs + (size_t)h * Dc;
      sgl_matmul(q_nope, uk, NULL, q_abs, 1, Dc, Dn);
      for (int j = 0; j < np; j++) {
        const float *crow = cache->data + (size_t)j * stride;
        float s1 = 0, s2 = 0;
        sgl_dot_f32(q_abs, crow, Dc, &s1);
        sgl_dot_f32(q_pe, crow + Dc, Dr, &s2);
        sc[j] = (s1 + s2) * m->softmax_scale;
      }
      sgl_softmax_inplace(sc, np);
      memset(olat, 0, sizeof(float) * Dc);
      for (int j = 0; j < np; j++) {
        const float a = sc[j];
        if (a == 0.0f) continue;
        const float *crow = cache->data + (size_t)j * stride;
        for (int d = 0; d < Dc; d++) olat[d] += a * crow[d];
      }
      sgl_matmul_nt(olat, uv, NULL, o_row + (size_t)h * Dv, 1, Dv, Dc);
    }
  }

  /* 5) o_proj：拼 head（[heads*v_head]）回 hidden */
  sgl_matmul_nt(m->att, m->w_o.data, m->b_o.data, out, n, Dh, H * Dv);
  return 0;
}

/* ---------------- 维度查询 ---------------- */

int sgl_mla_hidden(const sgl_mla *m) { return m ? m->hidden : 0; }
int sgl_mla_heads(const sgl_mla *m) { return m ? m->heads : 0; }
int sgl_mla_kv_lora_rank(const sgl_mla *m) { return m ? m->kv_lora : 0; }
int sgl_mla_qk_rope_dim(const sgl_mla *m) { return m ? m->qk_rope : 0; }
int sgl_mla_cache_stride(const sgl_mla *m) { return m ? m->stride : 0; }