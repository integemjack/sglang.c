/* sgl llama implementation: RoPE/RMSNorm/SwiGLU/GQA + paged KV pool
 * (llama / tinyllama / qwen2 family; qwen2 adds qkv bias + qk-norm).
 * KV pool: LIFO free list + per-page refcount; optional int8 KV (SGL_KV8=1);
 * optional int8 weights W8A16 (SGL_W8=1); draft mode (SGL_DRAFT_LAYERS=N). */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/llama.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sgl/port.h"
#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/log.h"
#include "sgl/ops.h"
#include "sgl/quant.h"

#define SGL_LLAMA_POOL_SLOTS 4096

/* quantize a [N,K] f32 weight into i8 + per-channel scale */
static void w8_quant(const float *w, int8_t **q8, float **s8, int64_t N, int64_t K) {
  *q8 = (int8_t *)sgl_xmalloc((size_t)N * (size_t)K);
  *s8 = (float *)sgl_xmalloc(sizeof(float) * (size_t)N);
  sgl_quant_rows_i8(w, *q8, *s8, (int)N, (int)K);
}

static char *path_join_(const char *a, const char *b) {
  size_t n = strlen(a) + strlen(b) + 2;
  char *p = (char *)sgl_xmalloc(n);
  snprintf(p, n, "%s/%s", a, b);
  return p;
}

/* logits = x * lm_head^T, lm_head is [V,E] row-major */
static void llama_logits(const sgl_llama *m, const float *x, float *out) {
  const float *W = m->lm_head.data;
#pragma omp parallel for
  for (int j = 0; j < m->vocab; j++) {
    const float *row = W + (size_t)j * m->n_embd;
    double s = 0;
    for (int k = 0; k < m->n_embd; k++) s += x[k] * row[k];
    out[j] = (float)s;
  }
}

/* LoRA adapter (PEFT safetensors): merge W += B @ A * scale into the f32
 * weights at load time (zero inference cost, same as merged models).
 * Enabled via env SGL_LORA=<dir> (adapter.safetensors + adapter_config.json),
 * optional SGL_LORA_SCALE multiplier (default 1). */
static int lora_merge_all(sgl_llama *m, const char *dir) {
  char cpath[1024];
  snprintf(cpath, sizeof(cpath), "%s/adapter_config.json", dir);
  FILE *f = fopen(cpath, "rb");
  if (!f) { SGL_ERROR("lora: no adapter_config.json in %s", dir); return -1; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *cbuf = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(cbuf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); sgl_xfree(cbuf); return -1; }
  fclose(f);
  cbuf[sz] = 0;
  sgl_json *cfg = sgl_json_parse(cbuf, (size_t)sz);
  sgl_xfree(cbuf);
  if (!cfg) { SGL_ERROR("lora: bad adapter_config.json"); return -1; }
  double alpha = sgl_json_get_num(cfg, "lora_alpha", 16.0);
  int r = sgl_json_get_int(cfg, "r", 8);
  sgl_json_free(cfg);
  const char *scaleenv = getenv("SGL_LORA_SCALE");
  double scale = (scaleenv ? atof(scaleenv) : 1.0) * alpha / r;

  char spath[1024];
  snprintf(spath, sizeof(spath), "%s/adapter.safetensors", dir);
  sgl_st_ctx *st = sgl_st_open(spath);
  if (!st) return -1;

  int merged = 0;
  typedef struct { const char *suffix; float *w; int64_t N, K; int owned; } lora_tg;
  for (int l = 0; l < m->n_layer; l++) {
    sgl_llama_layer *L = &m->layers[l];
    lora_tg tg[7];
    tg[0] = (lora_tg){"self_attn.q_proj", L->q_w.data, m->n_embd, m->n_embd, 0};
    tg[1] = (lora_tg){"self_attn.k_proj", L->k_w.data, m->kv_dim, m->n_embd, 0};
    tg[2] = (lora_tg){"self_attn.v_proj", L->v_w.data, m->kv_dim, m->n_embd, 0};
    tg[3] = (lora_tg){"self_attn.o_proj", L->o_w.data, m->n_embd, m->n_embd, 0};
    tg[4] = (lora_tg){"mlp.gate_proj", L->gate_w.data, m->n_inter, m->n_embd, 0};
    tg[5] = (lora_tg){"mlp.up_proj", L->up_w.data, m->n_inter, m->n_embd, 0};
    tg[6] = (lora_tg){"mlp.down_proj", L->down_w.data, m->n_embd, m->n_inter, 0};
    for (int t = 0; t < 7; t++) {
      char an[256], bn[256];
      snprintf(an, sizeof(an), "model.layers.%d.%s.lora_A.weight", l, tg[t].suffix);
      snprintf(bn, sizeof(bn), "model.layers.%d.%s.lora_B.weight", l, tg[t].suffix);
      sgl_tensor A = sgl_st_get(st, an, 2, (int64_t[]){r, tg[t].K});
      sgl_tensor B = sgl_st_get(st, bn, 2, (int64_t[]){tg[t].N, r});
      if (!A.data || !B.data) continue; /* proj not adapted */
      /* target weight may live in the read-only mmap - copy to heap first,
       * then merge into the writable copy (tensor header keeps dims) */
      if (!tg[t].owned) {
        size_t nelem = (size_t)(tg[t].N * tg[t].K);
        float *writable = (float *)sgl_xmalloc(nelem * sizeof(float));
        memcpy(writable, tg[t].w, nelem * sizeof(float));
        tg[t].w = writable;
        tg[t].owned = 1;
        switch (t) {
          case 0: L->q_w.data = writable; break;
          case 1: L->k_w.data = writable; break;
          case 2: L->v_w.data = writable; break;
          case 3: L->o_w.data = writable; break;
          case 4: L->gate_w.data = writable; break;
          case 5: L->up_w.data = writable; break;
          case 6: L->down_w.data = writable; break;
        }
      }
      for (int64_t n = 0; n < tg[t].N; n++) {
        float *wr = tg[t].w + (size_t)n * tg[t].K;
        const float *br = B.data + (size_t)n * r;
        for (int rr = 0; rr < r; rr++) {
          float bs = br[rr] * (float)scale;
          if (bs == 0.0f) continue;
          const float *ar = A.data + (size_t)rr * tg[t].K;
          for (int64_t k = 0; k < tg[t].K; k++) wr[k] += bs * ar[k];
        }
      }
      merged++;
    }
  }
  sgl_st_close(st);
  SGL_INFO("lora: merged %d projections (rank=%d alpha=%g scale=%g)", merged, r, alpha, scale);
  return merged > 0 ? 0 : -1;
}

sgl_llama *sgl_llama_load(const char *dir) {
  char *cpath = path_join_(dir, "config.json");
  FILE *f = NULL;
  if (fopen_s(&f, cpath, "rb") != 0 || !f) {
    /* MinGW path */
    f = fopen(cpath, "rb");
  }
  sgl_xfree(cpath);
  if (!f) { SGL_ERROR("llama: cannot open config.json in %s", dir); return NULL; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); sgl_xfree(buf); return NULL; }
  fclose(f);
  buf[sz] = 0;
  sgl_json *cfg = sgl_json_parse(buf, (size_t)sz);
  sgl_xfree(buf);
  if (!cfg) return NULL;

  sgl_llama *m = (sgl_llama *)sgl_xcalloc(1, sizeof(*m));
  m->n_layer = sgl_json_get_int(cfg, "num_hidden_layers", 0);
  m->n_head = sgl_json_get_int(cfg, "num_attention_heads", 0);
  m->n_kv_head = sgl_json_get_int(cfg, "num_key_value_heads", m->n_head);
  m->n_embd = sgl_json_get_int(cfg, "hidden_size", 0);
  m->n_inter = sgl_json_get_int(cfg, "intermediate_size", 4 * m->n_embd);
  m->n_ctx = sgl_json_get_int(cfg, "max_position_embeddings", 2048);
  m->vocab = sgl_json_get_int(cfg, "vocab_size", 0);
  m->rms_eps = (float)sgl_json_get_num(cfg, "rms_norm_eps", 1e-5);
  m->rope_theta = sgl_json_get_num(cfg, "rope_theta", 10000.0);
  m->tie_embeddings = sgl_json_get_int(cfg, "tie_word_embeddings", 0);
  m->head_dim = m->n_embd / (m->n_head ? m->n_head : 1);
  {
    int hd = sgl_json_get_int(cfg, "head_dim", 0);
    if (hd > 0) m->head_dim = hd;
  }
  m->kv_dim = m->head_dim * m->n_kv_head;
  m->q_dim = m->head_dim * m->n_head;
  m->sliding_window = sgl_json_get_int(cfg, "sliding_window", 0); /* t9 mistral: 0 = off */
  /* t9 gemma: hidden states scaled by sqrt(E) right after embedding */
  {
    const char *mt = sgl_json_get_str(cfg, "model_type", "");
    m->gemma_scale = !strcmp(mt, "gemma") ? sqrtf((float)m->n_embd) : 0.0f;
  }
  /* ---- G9: gemma hidden_softcap / rope_scaling / layer_types / qwen3-next linear dims ---- */
  m->emb_softcap = (float)sgl_json_get_num(cfg, "hidden_softcap", 0.0);
  m->rope_pos_scale = 1.0;
  {
    const sgl_json *rs = sgl_json_get(cfg, "rope_scaling");
    const char *rtype = rs ? sgl_json_get_str(rs, "rope_type", "default") : "default";
    double factor = rs ? sgl_json_get_num(rs, "factor", 1.0) : 1.0;
    double global_base = rs ? sgl_json_get_num(rs, "global_base_freq", 0.0) : 0.0;
    double local_base = sgl_json_get_num(cfg, "rope_local_base_freq", 0.0);
    if (factor <= 0.0) factor = 1.0;
    /* layer_types：按字符串判类型；不认识的字符串宽松忽略（按 full_attention）*/
    const sgl_json *lt = sgl_json_get(cfg, "layer_types");
    if (lt && lt->type == SGL_JSON_ARRAY && lt->count == m->n_layer) {
      m->layer_kind = (int *)sgl_xmalloc(sizeof(int) * (size_t)m->n_layer);
      for (int l = 0; l < m->n_layer; l++) {
        const sgl_json *it = lt->items[l];
        const char *s = (it && it->type == SGL_JSON_STRING) ? it->str : "full_attention";
        if (!strcmp(s, "linear_attention")) m->layer_kind[l] = 2;
        else if (!strcmp(s, "sliding_attention")) m->layer_kind[l] = 1;
        else {
          if (strcmp(s, "full_attention")) SGL_INFO("layer_types[%d]=%s unknown, treat as full", l, s);
          m->layer_kind[l] = 0;
        }
      }
    } else if (lt && lt->type == SGL_JSON_ARRAY && lt->count > 0) {
      SGL_INFO("layer_types count %d != num_hidden_layers %d, ignored", lt->count, m->n_layer);
    }
    /* qwen3-next linear attention 维度（无 layer_types 时这些字段无害，nk 仅备查）*/
    m->lin_nk = sgl_json_get_int(cfg, "linear_num_key_heads", 0);
    m->lin_nh = sgl_json_get_int(cfg, "linear_num_value_heads", m->lin_nk);
    m->lin_kd = sgl_json_get_int(cfg, "linear_key_head_dim", 0);
    m->lin_vd = sgl_json_get_int(cfg, "linear_value_head_dim", m->lin_kd);
    m->lin_conv = sgl_json_get_int(cfg, "linear_conv_kernel_dim", 0);
    m->lin_ratio = m->lin_nk > 0 ? m->lin_nh / m->lin_nk : 1;
    for (int l = 0; m->layer_kind && l < m->n_layer; l++)
      if (m->layer_kind[l] == 2 && (m->lin_nk <= 0 || m->lin_kd <= 0 || m->lin_conv <= 0 ||
          m->lin_nh % m->lin_nk)) {
        SGL_ERROR("linear_attention layer but linear_* config invalid");
        sgl_llama_free(m);
        sgl_json_free(cfg);
        return NULL;
      }
    /* 每层 rope 逆频率：sliding 层用 rope_local_base_freq（Gemma-3 局部 1e4），
     * 其余层用 rope_scaling.global_base_freq（缺省 rope_theta）；
     * rope_type=linear → 角度除以 factor；dynamic → NTK 基频缩放 theta*factor^(D/(D-2))
     *（注：HF 的 dynamic_rope_update 是按运行时序列长度插值的另一套公式，这里取
     *  静态 NTK 形式，随机模型 numpy 参考按同一公式，见 docs/NOTES_g9.md）。*/
    int half = m->head_dim / 2;
    m->layer_invf = (double *)sgl_xmalloc(sizeof(double) * (size_t)m->n_layer * (size_t)half);
    m->layer_window = (int *)sgl_xmalloc(sizeof(int) * (size_t)m->n_layer);
    if (!strcmp(rtype, "linear")) m->rope_pos_scale = 1.0 / factor;
    for (int l = 0; l < m->n_layer; l++) {
      int kind = m->layer_kind ? m->layer_kind[l] : (m->sliding_window > 0 ? 1 : 0);
      double base = m->rope_theta;
      if (m->layer_kind && kind == 1 && local_base > 0) base = local_base;
      else if (m->layer_kind && kind != 2 && global_base > 0) base = global_base;
      if (!strcmp(rtype, "dynamic"))
        base *= pow(factor, (double)m->head_dim / (double)(m->head_dim - 2));
      for (int i = 0; i < half; i++)
        m->layer_invf[(size_t)l * half + i] = pow(base, -(double)(2 * i) / (double)m->head_dim);
      /* 无 layer_types：全局 sliding_window 对所有层生效（mistral 旧行为）；
       * 有 layer_types：仅 sliding_attention 层开窗 */
      m->layer_window[l] = (kind == 1) ? m->sliding_window : 0;
    }
  }
  sgl_json_free(cfg);
  if (m->n_layer <= 0 || m->n_head <= 0 || m->n_embd <= 0 || m->vocab <= 0 ||
      m->n_kv_head <= 0 || m->n_embd % m->n_head) {
    SGL_ERROR("llama config invalid");
    sgl_llama_free(m);
    return NULL;
  }
  SGL_INFO("Llama config: L=%d H=%d KVH=%d E=%d I=%d D=%d ctx=%d vocab=%d eps=%g theta=%g tie=%d",
           m->n_layer, m->n_head, m->n_kv_head, m->n_embd, m->n_inter, m->head_dim,
           m->n_ctx, m->vocab, m->rms_eps, m->rope_theta, m->tie_embeddings);

  char *wpath = path_join_(dir, "model.safetensors");
  m->st = sgl_st_open(wpath);
  sgl_xfree(wpath);
  if (!m->st) { sgl_llama_free(m); return NULL; }

  int64_t E = m->n_embd, K = m->kv_dim, I = m->n_inter, V = m->vocab;
  int64_t QD = (int64_t)m->n_head * m->head_dim; /* q/o width: E unless explicit head_dim (qwen3: 2048) */
  m->wte = sgl_st_get(m->st, "model.embed_tokens.weight", 2, (int64_t[]){V, E});
  m->final_norm = sgl_st_get(m->st, "model.norm.weight", 1, (int64_t[]){E});
  {
    sgl_tensor lm = sgl_st_get(m->st, "lm_head.weight", 2, (int64_t[]){V, E});
    if (lm.data) {
      m->lm_head = lm; /* prefer real lm_head even when tie flag is set */
    } else {
      if (!m->tie_embeddings)
        SGL_INFO("lm_head missing, fallback to tied embeddings");
      m->lm_head = m->wte;
    }
  }
  if (!m->wte.data || !m->final_norm.data) { sgl_llama_free(m); return NULL; }

  m->layers = (sgl_llama_layer *)sgl_xcalloc((size_t)m->n_layer, sizeof(*m->layers));
  for (int l = 0; l < m->n_layer; l++) {
    char name[128];
    sgl_llama_layer *L = &m->layers[l];
    L->lin_idx = -1;
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
    L->inp_norm = sgl_st_get(m->st, name, 1, (int64_t[]){E});
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
    L->q_w = sgl_st_get(m->st, name, 2, (int64_t[]){QD, E});
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", l);
    L->k_w = sgl_st_get(m->st, name, 2, (int64_t[]){K, E});
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", l);
    L->v_w = sgl_st_get(m->st, name, 2, (int64_t[]){K, E});
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
    L->o_w = sgl_st_get(m->st, name, 2, (int64_t[]){E, QD});
    /* qwen2: separate q/k/v biases (HF) or fused qkv bias (vLLM layout) */
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.bias", l);
    {
      sgl_tensor b = sgl_st_get(m->st, name, 1, (int64_t[]){QD});
      if (b.data) { L->q_b = b; m->has_qkv_bias = 1; }
    }
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.bias", l);
    {
      sgl_tensor b = sgl_st_get(m->st, name, 1, (int64_t[]){K});
      if (b.data) L->k_b = b;
    }
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.bias", l);
    {
      sgl_tensor b = sgl_st_get(m->st, name, 1, (int64_t[]){K});
      if (b.data) L->v_b = b;
    }
    if (!L->q_b.data) {
      snprintf(name, sizeof(name), "model.layers.%d.self_attn.qkv_proj.bias", l);
      sgl_tensor b = sgl_st_get(m->st, name, 1, (int64_t[]){E + 2 * K});
      if (b.data) {
        L->q_b = sgl_tensor_from_data(1, (int64_t[]){E}, b.data);
        L->k_b = sgl_tensor_from_data(1, (int64_t[]){K}, b.data + E);
        L->v_b = sgl_tensor_from_data(1, (int64_t[]){K}, b.data + E + K);
        m->has_qkv_bias = 1;
      }
    }
    /* qwen2: qk-norm (head_dim sized rmsnorm, applied before rope) */
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", l);
    {
      sgl_tensor qn = sgl_st_get(m->st, name, 1, (int64_t[]){m->head_dim});
      if (qn.data) L->q_norm = qn;
    }
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", l);
    {
      sgl_tensor kn = sgl_st_get(m->st, name, 1, (int64_t[]){m->head_dim});
      if (kn.data) L->k_norm = kn;
    }
    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
    L->post_norm = sgl_st_get(m->st, name, 1, (int64_t[]){E});
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", l);
    L->gate_w = sgl_st_get(m->st, name, 2, (int64_t[]){I, E});
    snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", l);
    L->up_w = sgl_st_get(m->st, name, 2, (int64_t[]){I, E});
    snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", l);
    L->down_w = sgl_st_get(m->st, name, 2, (int64_t[]){E, I});
    if (!L->inp_norm.data || !L->post_norm.data || !L->gate_w.data || !L->up_w.data ||
        !L->down_w.data) {
      SGL_ERROR("layer %d weights missing", l);
      sgl_llama_free(m);
      return NULL;
    }
    /* G9 linear_attention 层：gated deltanet 权重（名字与 HF Qwen3Next 检查点对齐）*/
    if (m->layer_kind && m->layer_kind[l] == 2) {
      int keyd = m->lin_nk * m->lin_kd, vald = m->lin_nh * m->lin_vd;
      int convd = 2 * keyd + vald;
      snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkvz.weight", l);
      L->lin_qkvz_w = sgl_st_get(m->st, name, 2, (int64_t[]){2 * keyd + 2 * vald, E});
      snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_ba.weight", l);
      L->lin_ba_w = sgl_st_get(m->st, name, 2, (int64_t[]){2 * m->lin_nh, E});
      snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", l);
      L->lin_conv_w = sgl_st_get(m->st, name, 3, (int64_t[]){convd, 1, m->lin_conv});
      if (!L->lin_conv_w.data) /* 允许 2D [convd,K] 简化布局 */
        L->lin_conv_w = sgl_st_get(m->st, name, 2, (int64_t[]){convd, m->lin_conv});
      snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", l);
      L->lin_norm_w = sgl_st_get(m->st, name, 1, (int64_t[]){m->lin_vd});
      snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", l);
      L->lin_out_w = sgl_st_get(m->st, name, 2, (int64_t[]){E, vald});
      sgl_tensor alog, dtb;
      snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", l);
      alog = sgl_st_get(m->st, name, 1, (int64_t[]){m->lin_nh});
      snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", l);
      dtb = sgl_st_get(m->st, name, 1, (int64_t[]){m->lin_nh});
      if (alog.data) {
        L->lin_a_log = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->lin_nh);
        memcpy(L->lin_a_log, alog.data, sizeof(float) * (size_t)m->lin_nh);
      }
      if (dtb.data) {
        L->lin_dt_bias = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->lin_nh);
        memcpy(L->lin_dt_bias, dtb.data, sizeof(float) * (size_t)m->lin_nh);
      }
      if (!L->lin_qkvz_w.data || !L->lin_ba_w.data || !L->lin_conv_w.data ||
          !L->lin_norm_w.data || !L->lin_out_w.data || !L->lin_a_log || !L->lin_dt_bias) {
        SGL_ERROR("layer %d linear_attn weights missing", l);
        sgl_llama_free(m);
        return NULL;
      }
      L->lin_idx = m->n_lin++;
    } else {
      /* full/sliding：标准注意力权重必备 */
      if (!L->q_w.data || !L->k_w.data || !L->v_w.data || !L->o_w.data) {
        SGL_ERROR("layer %d weights missing", l);
        sgl_llama_free(m);
        return NULL;
      }
    }
  }

  /* draft mode: use only first N layers (self-draft for speculative decoding) */
  {
    const char *e = getenv("SGL_DRAFT_LAYERS");
    if (e && atoi(e) > 0 && atoi(e) < m->n_layer) {
      SGL_INFO("draft mode: first %d of %d layers", atoi(e), m->n_layer);
      m->n_layer = atoi(e);
    }
  }

  /* LoRA merge (env SGL_LORA=<dir>): before w8 quantization below */
  {
    const char *e = getenv("SGL_LORA");
    if (e && *e) {
      if (lora_merge_all(m, e) != 0)
        SGL_ERROR("lora merge failed for %s (continuing without)", e);
    }
  }

  /* scratch buffers */
  size_t ctx = (size_t)m->n_ctx;
#ifdef _OPENMP
  m->n_score_rows = omp_get_max_threads();
#else
  m->n_score_rows = 1;
#endif
  m->x = (float *)sgl_xmalloc(ctx * m->n_embd * sizeof(float));
  m->h = (float *)sgl_xmalloc(ctx * m->n_embd * sizeof(float));
  m->q = (float *)sgl_xmalloc(ctx * (size_t)m->q_dim * sizeof(float));
  m->k = (float *)sgl_xmalloc(ctx * (size_t)m->kv_dim * sizeof(float));
  m->v = (float *)sgl_xmalloc(ctx * (size_t)m->kv_dim * sizeof(float));
  m->attn = (float *)sgl_xmalloc(ctx * (size_t)m->q_dim * sizeof(float));
  m->gate = (float *)sgl_xmalloc(ctx * (size_t)m->n_inter * sizeof(float));
  m->up = (float *)sgl_xmalloc(ctx * (size_t)m->n_inter * sizeof(float));
  m->scores = (float *)sgl_xmalloc((size_t)m->n_score_rows * ctx * sizeof(float));

  /* paged KV pool (int8 storage when SGL_KV8=1) */
  m->n_pool = SGL_LLAMA_POOL_SLOTS;
  {
    const char *env = getenv("SGL_POOL_SLOTS");
    if (env && atoi(env) > 0) m->n_pool = atoi(env);
    env = getenv("SGL_KV8");
    if (env && atoi(env) != 0) m->kv8 = 1;
  }
  if (m->kv8) {
    m->kc8 = (int8_t *)sgl_xmalloc((size_t)m->n_layer * m->n_pool * m->kv_dim);
    m->vc8 = (int8_t *)sgl_xmalloc((size_t)m->n_layer * m->n_pool * m->kv_dim);
    m->kcs = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->n_layer * m->n_pool);
    m->vcs = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->n_layer * m->n_pool);
  } else {
    m->kc = (float *)sgl_xmalloc((size_t)m->n_layer * m->n_pool * m->kv_dim * sizeof(float));
    m->vc = (float *)sgl_xmalloc((size_t)m->n_layer * m->n_pool * m->kv_dim * sizeof(float));
  }
  m->free_pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)m->n_pool);
  m->page_refs = (int *)sgl_xcalloc((size_t)m->n_pool, sizeof(int));
  for (int i = 0; i < m->n_pool; i++) m->free_pages[i] = m->n_pool - 1 - i;
  m->n_free = m->n_pool;

  /* G9: linear attention 递归状态池（按序列首 KV 页归属；CPU 参考路径，槽位有限）*/
  if (m->n_lin > 0) {
    int keyd = m->lin_nk * m->lin_kd, vald = m->lin_nh * m->lin_vd;
    int convd = 2 * keyd + vald;
    m->n_lin_slots = 64;
    { const char *e = getenv("SGL_LIN_SLOTS"); if (e && atoi(e) > 0) m->n_lin_slots = atoi(e); }
    size_t sh = (size_t)m->lin_nh * m->lin_kd * m->lin_vd;          /* 每层每槽状态元素 */
    size_t hh = (size_t)convd * (m->lin_conv > 1 ? m->lin_conv - 1 : 0); /* conv 历史 */
    m->lin_state = (float *)sgl_xcalloc((size_t)m->n_lin * m->n_lin_slots, sh * sizeof(float));
    m->lin_hist = (float *)sgl_xcalloc((size_t)m->n_lin * m->n_lin_slots, hh ? hh * sizeof(float) : 1);
    m->lin_owner = (int *)sgl_xmalloc(sizeof(int) * (size_t)m->n_lin_slots);
    m->lin_pos = (int *)sgl_xmalloc(sizeof(int) * (size_t)m->n_lin * m->n_lin_slots);
    m->lin_free = (int *)sgl_xmalloc(sizeof(int) * (size_t)m->n_lin_slots);
    for (int i = 0; i < m->n_lin_slots; i++) { m->lin_owner[i] = -1; m->lin_free[i] = m->n_lin_slots - 1 - i; }
    for (size_t i = 0; i < (size_t)m->n_lin * m->n_lin_slots; i++) m->lin_pos[i] = -1;
    m->lin_nfree = m->n_lin_slots;
    m->ls_qkvz = (float *)sgl_xmalloc(sizeof(float) * (size_t)(2 * keyd + 2 * vald));
    m->ls_ba = (float *)sgl_xmalloc(sizeof(float) * (size_t)(2 * m->lin_nh));
    m->ls_conv = (float *)sgl_xmalloc(sizeof(float) * (size_t)convd);
    m->ls_o = (float *)sgl_xmalloc(sizeof(float) * (size_t)vald);
    m->ls_qn = (float *)sgl_xmalloc(sizeof(float) * (size_t)keyd);
    m->ls_kn = (float *)sgl_xmalloc(sizeof(float) * (size_t)keyd);
    SGL_INFO("G9: %d linear_attention layers, %d recurrent state slots (nh=%d nk=%d kd=%d vd=%d conv=%d)",
             m->n_lin, m->n_lin_slots, m->lin_nh, m->lin_nk, m->lin_kd, m->lin_vd, m->lin_conv);
  }

  /* W8: quantize all projection weights after load (env SGL_W8=1) */
  {
    const char *env = getenv("SGL_W8");
    if (env && atoi(env) != 0) {
      m->w8 = 1;
      clock_t tq = clock();
      for (int l = 0; l < m->n_layer; l++) {
        sgl_llama_layer *Lq = &m->layers[l];
        if (Lq->lin_idx >= 0) continue; /* G9: linear 层不参与 W8（无 qkv/o 权重）*/
        w8_quant(Lq->q_w.data, &Lq->q8, &Lq->sq8, E, E);
        w8_quant(Lq->k_w.data, &Lq->k8, &Lq->sk8, K, E);
        w8_quant(Lq->v_w.data, &Lq->v8, &Lq->sv8, K, E);
        w8_quant(Lq->o_w.data, &Lq->o8, &Lq->so8, E, E);
        w8_quant(Lq->gate_w.data, &Lq->g8, &Lq->sg8, I, E);
        w8_quant(Lq->up_w.data, &Lq->u8, &Lq->su8, I, E);
        w8_quant(Lq->down_w.data, &Lq->d8, &Lq->sd8, E, I);
      }
      SGL_INFO("W8: int8 weights quantized in %.2fs", (double)(clock() - tq) / CLOCKS_PER_SEC);
    }
  }
  if (m->w8) {
    size_t widest = (size_t)m->n_embd > (size_t)m->n_inter ? m->n_embd : m->n_inter;
    m->x8 = (int8_t *)sgl_xmalloc(ctx * widest);
    m->x8s = (float *)sgl_xmalloc(sizeof(float) * ctx);
  }
  SGL_INFO("Llama loaded: KV pool %d pages (%.1f MB f32%s)",
           m->n_pool,
           (m->n_layer * 2.0 * m->n_pool * m->kv_dim * sizeof(float)) / 1048576.0,
           m->kv8 ? ", int8 mode" : "");
  return m;
}

void sgl_llama_free(sgl_llama *m) {
  if (!m) return;
  for (int l = 0; l < m->n_layer; l++) {
    sgl_llama_layer *L = &m->layers[l];
    sgl_xfree(L->q8); sgl_xfree(L->k8); sgl_xfree(L->v8); sgl_xfree(L->o8);
    sgl_xfree(L->g8); sgl_xfree(L->u8); sgl_xfree(L->d8);
    sgl_xfree(L->sq8); sgl_xfree(L->sk8); sgl_xfree(L->sv8); sgl_xfree(L->so8);
    sgl_xfree(L->sg8); sgl_xfree(L->su8); sgl_xfree(L->sd8);
    sgl_xfree(L->lin_a_log); sgl_xfree(L->lin_dt_bias); /* G9 */
  }
  sgl_xfree(m->x); sgl_xfree(m->h); sgl_xfree(m->q); sgl_xfree(m->k); sgl_xfree(m->v);
  sgl_xfree(m->attn); sgl_xfree(m->gate); sgl_xfree(m->up); sgl_xfree(m->scores);
  sgl_xfree(m->x8); sgl_xfree(m->x8s);
  sgl_xfree(m->kc); sgl_xfree(m->vc);
  sgl_xfree(m->kc8); sgl_xfree(m->vc8); sgl_xfree(m->kcs); sgl_xfree(m->vcs);
  sgl_xfree(m->free_pages); sgl_xfree(m->page_refs);
  /* G9 */
  sgl_xfree(m->layer_kind); sgl_xfree(m->layer_invf); sgl_xfree(m->layer_window);
  sgl_xfree(m->lin_state); sgl_xfree(m->lin_hist);
  sgl_xfree(m->lin_owner); sgl_xfree(m->lin_pos); sgl_xfree(m->lin_free);
  sgl_xfree(m->ls_qkvz); sgl_xfree(m->ls_ba); sgl_xfree(m->ls_conv);
  sgl_xfree(m->ls_o); sgl_xfree(m->ls_qn); sgl_xfree(m->ls_kn);
  sgl_xfree(m->layers);
  sgl_st_close(m->st);
  sgl_xfree(m);
}

/* ---- paged KV pool: LIFO free stack + refcounts ---- */
int sgl_llama_pages_alloc(sgl_llama *m, int *idx, int n) {
  if (n <= 0) return 0;
  if (n > m->n_free) return -1;
  m->n_free -= n;
  for (int i = 0; i < n; i++) {
    idx[i] = m->free_pages[m->n_free + i];
    m->page_refs[idx[i]] = 1;
  }
  return 0;
}

void sgl_llama_page_ref(sgl_llama *m, int page, int delta) {
  if (page < 0 || page >= m->n_pool) return;
  m->page_refs[page] += delta;
  if (m->page_refs[page] <= 0) {
    m->page_refs[page] = 0;
    m->free_pages[m->n_free++] = page;
  }
}

void sgl_llama_pages_unref(sgl_llama *m, const int *idx, int n) {
  for (int i = 0; i < n; i++) sgl_llama_page_ref(m, idx[i], -1);
}

int sgl_llama_nfree(const sgl_llama *m) { return m->n_free; }

/* projection dispatch: W8A16 when quantized, else f32 gemm */
static void proj_nt(sgl_llama *m, const float *A, const sgl_tensor *W,
                    int8_t *W8, const float *sW8, const float *bias,
                    float *C, int M, int N, int K) {
  if (m->w8 && W8) {
    sgl_matmul_w8a16(A, W8, sW8, bias, C, M, N, K);
  } else {
    sgl_matmul_nt(A, W->data, bias, C, M, N, K);
  }
}

/* G9: 每层 rope（预计算逆频率，支持 linear 位置缩放 / 局部-全局基频 / dynamic NTK）。
 * 公式与 sgl_rope_inplace 同（double pow/cos/sin，float 乘加）：无 scaling 时
 * 逐位一致，既有 family 测试不受影响。pos 已含 rope_pos_scale。 */
static void rope_layer(const sgl_llama *m, int l, float *qk, int n_heads, int D, double pos) {
  int half = D / 2;
  const double *invf = m->layer_invf + (size_t)l * (size_t)half;
  for (int h = 0; h < n_heads; h++) {
    float *head = qk + (size_t)h * D;
    for (int i = 0; i < half; i++) {
      double ang = pos * invf[i];
      float c = (float)cos(ang), s = (float)sin(ang);
      float a = head[i], b = head[i + half];
      head[i] = a * c - b * s;
      head[i + half] = a * s + b * c;
    }
  }
}

/* G9: 取/初始化某序列在 linear 层 ll 的递归状态槽。
 * 归属键 = 序列首 KV 页号 slot_idx[0]（同一页表内稳定）；位置不连续
 * （新序列/页被复用）则清零状态。局限：两条序列共享首页（radix 前缀复用）
 * 会互相踩状态 —— CPU 参考路径限制，见 docs/NOTES_g9.md。 */
static int lin_slot_get(sgl_llama *m, int ll, int first_page, int pos) {
  int s = -1;
  for (int i = 0; i < m->n_lin_slots; i++)
    if (m->lin_owner[i] == first_page) { s = i; break; }
  if (s < 0) {
    if (m->lin_nfree <= 0) {
      SGL_ERROR("linear attn: no free state slot (SGL_LIN_SLOTS=%d)", m->n_lin_slots);
      return -1;
    }
    s = m->lin_free[--m->lin_nfree];
    m->lin_owner[s] = first_page;
  }
  /* lin_pos 按（层，槽）记录：同一序列在多个 linear 层共享槽号但各自独立推进 */
  int *lp = &m->lin_pos[(size_t)ll * m->n_lin_slots + s];
  if (*lp != pos) {
    int keyd = m->lin_nk * m->lin_kd, vald = m->lin_nh * m->lin_vd;
    size_t sh = (size_t)m->lin_nh * m->lin_kd * m->lin_vd;
    size_t hh = (size_t)(2 * keyd + vald) * (m->lin_conv > 1 ? m->lin_conv - 1 : 0);
    *lp = pos;
    memset(m->lin_state + ((size_t)ll * m->n_lin_slots + s) * sh, 0, sh * sizeof(float));
    if (hh) memset(m->lin_hist + ((size_t)ll * m->n_lin_slots + s) * hh, 0, hh * sizeof(float));
  }
  return s;
}

/* G9 qwen3-next linear_attention 层单 token 步进：按 HF transformers
 * modeling_qwen3_next.py 的 Qwen3NextGatedDeltaNet 公式实现：
 *   1) in_proj_qkvz / in_proj_ba（按 k-head 组交错 [q|k|v|z]、[b|a]）；
 *   2) 因果 depthwise conv1d（核 K，零历史）+ SiLU；
 *   3) beta=sigmoid(b)，g=-exp(A_log)*softplus(a+dt_bias)；
 *   4) q/k l2norm（FLA 式 x/sqrt(sum x^2 + 1e-6)），q 再除 sqrt(kd)；
 *   5) 递归：S*=exp(g); delta=(v-S^T k)*beta; S+=k⊗delta; out=S^T q；
 *   6) gated RMSNorm（先 norm 后 *silu(z)，eps=rms_norm_eps）；out_proj。
 * 与真实模型的差距：未实现 full 层的 q-gate/partial rope/(1+w) norm/MoE（只按
 * layer_types 分支）；状态按序列首页归属（不支持前缀共享）。 */
static int lin_attn_step(sgl_llama *m, const sgl_llama_layer *L, const float *hrow,
                         int first_page, int pos, float *orow) {
  int nk = m->lin_nk, nh = m->lin_nh, kd = m->lin_kd, vd = m->lin_vd;
  int K = m->lin_conv, ratio = m->lin_ratio;
  int keyd = nk * kd, vald = nh * vd, convd = 2 * keyd + vald;
  int ll = L->lin_idx;
  int slot = lin_slot_get(m, ll, first_page, pos);
  if (slot < 0) return -1;
  /* 1) 投影 */
  sgl_matmul_nt(hrow, L->lin_qkvz_w.data, NULL, m->ls_qkvz, 1, 2 * keyd + 2 * vald, m->n_embd);
  sgl_matmul_nt(hrow, L->lin_ba_w.data, NULL, m->ls_ba, 1, 2 * nh, m->n_embd);
  /* 2) 拼 mixed_qkv [q|k|v]（conv 在非交错布局上操作，同 HF）*/
  int gw = 2 * kd + 2 * ratio * vd; /* 每 k-head 组宽 */
  for (int g = 0; g < nk; g++) {
    const float *base = m->ls_qkvz + (size_t)g * gw;
    memcpy(m->ls_conv + (size_t)g * kd, base, sizeof(float) * kd);
    memcpy(m->ls_conv + keyd + (size_t)g * kd, base + kd, sizeof(float) * kd);
    memcpy(m->ls_conv + 2 * keyd + (size_t)g * ratio * vd, base + 2 * kd,
           sizeof(float) * (ratio * vd));
  }
  /* 3) 因果 depthwise conv + SiLU */
  float *hist = m->lin_hist + ((size_t)ll * m->n_lin_slots + slot) * convd * (K > 1 ? K - 1 : 0);
  const float *cw = L->lin_conv_w.data; /* [convd, K] */
  for (int c = 0; c < convd; c++) {
    float x = m->ls_conv[c];
    const float *wc = cw + (size_t)c * K;
    float acc = wc[K - 1] * x;
    for (int j = 0; j < K - 1; j++) acc += wc[j] * hist[(size_t)c * (K - 1) + j];
    for (int j = 0; j + 1 < K - 1; j++) hist[(size_t)c * (K - 1) + j] = hist[(size_t)c * (K - 1) + j + 1];
    if (K > 1) hist[(size_t)c * (K - 1) + K - 2] = x;
    m->ls_conv[c] = acc / (1.0f + expf(-acc));
  }
  /* 4) l2norm q/k（每 k-head 一次；v-head 通过 h/ratio 复用，同 repeat_interleave）*/
  const float *qc = m->ls_conv, *kc = m->ls_conv + keyd, *vc = m->ls_conv + 2 * keyd;
  float inv_sqrt_kd = 1.0f / sqrtf((float)kd);
  for (int g = 0; g < nk; g++) {
    float ss = 0; const float *src = qc + (size_t)g * kd; float *dst = m->ls_qn + (size_t)g * kd;
    for (int d = 0; d < kd; d++) ss += src[d] * src[d];
    float inv = 1.0f / sqrtf(ss + 1e-6f);
    for (int d = 0; d < kd; d++) dst[d] = src[d] * inv * inv_sqrt_kd;
    ss = 0; src = kc + (size_t)g * kd; dst = m->ls_kn + (size_t)g * kd;
    for (int d = 0; d < kd; d++) ss += src[d] * src[d];
    inv = 1.0f / sqrtf(ss + 1e-6f);
    for (int d = 0; d < kd; d++) dst[d] = src[d] * inv;
  }
  /* 5) 每 v-head 递归状态更新 */
  float *S = m->lin_state + ((size_t)ll * m->n_lin_slots + slot) * (size_t)nh * kd * vd;
  float *out = m->ls_o;
  int gw2 = 2 * ratio;
  for (int h = 0; h < nh; h++) {
    const float *qh = m->ls_qn + (size_t)(h / ratio) * kd;
    const float *kh = m->ls_kn + (size_t)(h / ratio) * kd;
    const float *vh = vc + (size_t)h * vd;
    float *Sh = S + (size_t)h * kd * vd;
    int ba_off = (size_t)(h / ratio) * gw2 + (h % ratio);
    float bval = m->ls_ba[ba_off], aval = m->ls_ba[ba_off + ratio];
    float beta = 1.0f / (1.0f + expf(-bval));
    float ax = aval + L->lin_dt_bias[h];
    float sp = ax > 20.0f ? ax : log1pf(expf(ax));
    float alpha = expf(-expf(L->lin_a_log[h]) * sp);
    for (size_t e = 0; e < (size_t)kd * vd; e++) Sh[e] *= alpha;
    float *oh = out + (size_t)h * vd;
    for (int d = 0; d < vd; d++) {
      float mem = 0;
      for (int i = 0; i < kd; i++) mem += Sh[(size_t)i * vd + d] * kh[i];
      float delta = (vh[d] - mem) * beta;
      for (int i = 0; i < kd; i++) Sh[(size_t)i * vd + d] += kh[i] * delta;
      float o = 0;
      for (int i = 0; i < kd; i++) o += Sh[(size_t)i * vd + d] * qh[i];
      oh[d] = o;
    }
  }
  /* 6) gated RMSNorm（先 norm 后 * silu(z)）*/
  for (int h = 0; h < nh; h++) {
    float *oh = out + (size_t)h * vd;
    double ss = 0;
    for (int d = 0; d < vd; d++) ss += (double)oh[d] * oh[d];
    float inv = (float)(1.0 / sqrt(ss / vd + (double)m->rms_eps));
    const float *zh = m->ls_qkvz + (size_t)(h / ratio) * gw + 2 * kd + (size_t)(ratio * vd) + (size_t)(h % ratio) * vd;
    for (int d = 0; d < vd; d++) {
      float zg = zh[d];
      oh[d] = oh[d] * inv * L->lin_norm_w.data[d] * (zg / (1.0f + expf(-zg)));
    }
  }
  /* 7) out_proj */
  sgl_matmul_nt(out, L->lin_out_w.data, NULL, orow, 1, m->n_embd, vald);
  m->lin_pos[(size_t)ll * m->n_lin_slots + slot] = pos + 1;
  return 0;
}

/* 共享实现：logits_rows==0 时只在 logits_out 返回末位分布（旧语义）；
 * logits_rows==n 时在 logits_all 返回全部 n 位分布（sgl_llama_forward_n）。 */
static int llama_forward_emb_impl(sgl_llama *m, const int *tokens, const float *emb,
                                  int n, int pos0, const int *slot_idx,
                                  float *logits_out, int logits_rows);
static int llama_forward_impl(sgl_llama *m, const int *tokens, int n, int pos0,
                              const int *slot_idx, float *logits_out,
                              int logits_rows) {
  return llama_forward_emb_impl(m, tokens, NULL, n, pos0, slot_idx, logits_out, logits_rows);
}

/* t10 llava: emb != NULL 时直接用给定的 embedding 序列 [n][E]（视觉 token 融合），
 * 跳过 token 查表；其余与 token 前向完全一致（同一层循环）。 */
static int llama_forward_emb_impl(sgl_llama *m, const int *tokens, const float *emb,
                                  int n, int pos0, const int *slot_idx,
                                  float *logits_out, int logits_rows) {
  int E = m->n_embd, H = m->n_head, HK = m->n_kv_head, D = m->head_dim;
  int total = pos0 + n;
  if (n <= 0 || total > m->n_ctx) {
    SGL_ERROR("llama forward: n=%d pos0=%d exceeds ctx=%d", n, pos0, m->n_ctx);
    return -1;
  }
  for (int p = 0; p < total; p++) {
    if (slot_idx[p] < 0 || slot_idx[p] >= m->n_pool) {
      SGL_ERROR("llama forward: page %d out of pool at %d", slot_idx[p], p);
      return -1;
    }
  }

  if (emb)
    memcpy(m->x, emb, sizeof(float) * (size_t)n * E);
  else
    sgl_embedding_get(m->wte.data, tokens, n, E, m->x);
  if (m->emb_softcap > 0) /* G9 gemma-3: emb = cap*tanh(emb/cap)，在 sqrt(E) scale 前 */
    for (size_t i = 0; i < (size_t)n * E; i++)
      m->x[i] = m->emb_softcap * tanhf(m->x[i] / m->emb_softcap);
  if (m->gemma_scale > 0) /* t9 gemma: x *= sqrt(E) after embedding */
    for (size_t i = 0; i < (size_t)n * E; i++) m->x[i] *= m->gemma_scale;

  for (int l = 0; l < m->n_layer; l++) {
    sgl_llama_layer *L = &m->layers[l];
    sgl_rmsnorm(m->x, L->inp_norm.data, m->h, n, E, m->rms_eps);
    if (L->lin_idx >= 0) {
      /* G9 linear_attention：无 KV 页，逐 token 递归（串行）；out_proj 写回 h 行 t
       * 时 h 行已被投影消费，不别名冲突（matmul 输入是 ls_o）。 */
      for (int t = 0; t < n; t++) {
        if (lin_attn_step(m, L, m->h + (size_t)t * E, slot_idx[0], pos0 + t,
                          m->h + (size_t)t * E) != 0)
          return -1;
        for (int e = 0; e < E; e++)
          m->x[(size_t)t * E + e] += m->h[(size_t)t * E + e];
      }
    } else {
    proj_nt(m, m->h, &L->q_w, L->q8, L->sq8, L->q_b.data, m->q, n, m->q_dim, E);
    proj_nt(m, m->h, &L->k_w, L->k8, L->sk8, L->k_b.data, m->k, n, m->kv_dim, E);
    proj_nt(m, m->h, &L->v_w, L->v8, L->sv8, L->v_b.data, m->v, n, m->kv_dim, E);
    /* qwen2 qk-norm (before rope) */
    if (L->q_norm.data)
      for (int t = 0; t < n; t++)
        for (int h = 0; h < H; h++)
          sgl_rmsnorm(m->q + (size_t)t * m->q_dim + h * D, L->q_norm.data,
                      m->q + (size_t)t * m->q_dim + h * D, 1, D, 1e-6f);
    if (L->k_norm.data)
      for (int t = 0; t < n; t++)
        for (int hk = 0; hk < HK; hk++)
          sgl_rmsnorm(m->k + (size_t)t * m->kv_dim + hk * D, L->k_norm.data,
                      m->k + (size_t)t * m->kv_dim + hk * D, 1, D, 1e-6f);
    /* rope: per-token position (rows of the same token share pos0+t)；G9 每层预计算逆频率 */
    for (int t = 0; t < n; t++) {
      double pos = (double)(pos0 + t) * m->rope_pos_scale;
      rope_layer(m, l, m->q + (size_t)t * m->q_dim, H, D, pos);
      rope_layer(m, l, m->k + (size_t)t * m->kv_dim, HK, D, pos);
    }

    /* write KV via page table (kv8: per-page quantize) */
    if (m->kv8) {
      int8_t *kc8 = m->kc8 + (size_t)l * m->n_pool * m->kv_dim;
      int8_t *vc8 = m->vc8 + (size_t)l * m->n_pool * m->kv_dim;
      for (int t = 0; t < n; t++) {
        int pg = slot_idx[pos0 + t];
        m->kcs[(size_t)l * m->n_pool + pg] =
            sgl_quant_row_i8(m->k + (size_t)t * m->kv_dim, kc8 + (size_t)pg * m->kv_dim,
                             m->kv_dim);
        m->vcs[(size_t)l * m->n_pool + pg] =
            sgl_quant_row_i8(m->v + (size_t)t * m->kv_dim, vc8 + (size_t)pg * m->kv_dim,
                             m->kv_dim);
      }
    } else {
      float *kc = m->kc + (size_t)l * m->n_pool * m->kv_dim;
      float *vc = m->vc + (size_t)l * m->n_pool * m->kv_dim;
      for (int t = 0; t < n; t++) {
        memcpy(kc + (size_t)slot_idx[pos0 + t] * m->kv_dim,
               m->k + (size_t)t * m->kv_dim, (size_t)m->kv_dim * sizeof(float));
        memcpy(vc + (size_t)slot_idx[pos0 + t] * m->kv_dim,
               m->v + (size_t)t * m->kv_dim, (size_t)m->kv_dim * sizeof(float));
      }
    }

    /* causal attention with GQA, gather by page table */
    {
      const int8_t *kc8l = m->kv8 ? m->kc8 + (size_t)l * m->n_pool * m->kv_dim : NULL;
      const int8_t *vc8l = m->kv8 ? m->vc8 + (size_t)l * m->n_pool * m->kv_dim : NULL;
      const float *kcs = m->kv8 ? m->kcs + (size_t)l * m->n_pool : NULL;
      const float *vcs = m->kv8 ? m->vcs + (size_t)l * m->n_pool : NULL;
      const float *kcf = m->kv8 ? NULL : m->kc + (size_t)l * m->n_pool * m->kv_dim;
      const float *vcf = m->kv8 ? NULL : m->vc + (size_t)l * m->n_pool * m->kv_dim;
#pragma omp parallel for
      for (int t = 0; t < n; t++) {
        float *sc = m->scores;
#ifdef _OPENMP
        sc += (size_t)omp_get_thread_num() * m->n_ctx;
#endif
        const float *qrow = m->q + (size_t)t * m->q_dim;
        float *orow = m->attn + (size_t)t * m->q_dim;
        for (int h = 0; h < H; h++) {
          const float *qh = qrow + h * D;
          size_t hoff = (size_t)(h / (H / HK)) * D;
          int upto = pos0 + t + 1;
          /* t9 mistral / G9 layer_types: sliding 层窗口（每层 layer_window[l]，0=关）*/
          int win = m->layer_window[l];
          int from = win > 0 && upto > win ? upto - win : 0;
          float maxv = -1e30f;
          for (int p = from; p < upto; p++) {
            int pg = slot_idx[p];
            float s = 0;
            if (kc8l) {
              const int8_t *kh = kc8l + (size_t)pg * m->kv_dim + hoff;
              float acc = 0;
              for (int d = 0; d < D; d++) acc += qh[d] * (float)kh[d];
              s = acc * kcs[pg];
            } else {
              float dotv;
              sgl_dot_f32(qh, kcf + (size_t)pg * m->kv_dim + hoff, D, &dotv);
              s = dotv;
            }
            s *= 1.0f / sqrtf((float)D);
            sc[p] = s;
            if (s > maxv) maxv = s;
          }
          double sum = 0;
          for (int p = from; p < upto; p++) { sc[p] = expf(sc[p] - maxv); sum += sc[p]; }
          float *out = orow + h * D;
          memset(out, 0, sizeof(float) * D);
          for (int p = from; p < upto; p++) {
            int pg = slot_idx[p];
            float w = sc[p] / (float)sum;
            if (vc8l) {
              const int8_t *vh = vc8l + (size_t)pg * m->kv_dim + hoff;
              float sv = vcs[pg];
              for (int d = 0; d < D; d++) out[d] += w * ((float)vh[d] * sv);
            } else {
              const float *vh = vcf + (size_t)pg * m->kv_dim + hoff;
              for (int d = 0; d < D; d++) out[d] += w * vh[d];
            }
          }
        }
      }
    }

    proj_nt(m, m->attn, &L->o_w, L->o8, L->so8, NULL, m->h, n, E, m->q_dim);
    for (int t = 0; t < n; t++)
      for (int k = 0; k < E; k++)
        m->x[(size_t)t * E + k] += m->h[(size_t)t * E + k];
    } /* end else (full/sliding attention branch) */

    /* SwiGLU MLP */
    sgl_rmsnorm(m->x, L->post_norm.data, m->h, n, E, m->rms_eps);
    proj_nt(m, m->h, &L->gate_w, L->g8, L->sg8, NULL, m->gate, n, m->n_inter, E);
    proj_nt(m, m->h, &L->up_w, L->u8, L->su8, NULL, m->up, n, m->n_inter, E);
    sgl_silu_inplace(m->gate, n * m->n_inter);
    for (size_t i = 0; i < (size_t)n * m->n_inter; i++) m->gate[i] *= m->up[i];
    proj_nt(m, m->gate, &L->down_w, L->d8, L->sd8, NULL, m->h, n, E, m->n_inter);
    for (int t = 0; t < n; t++)
      for (int k = 0; k < E; k++)
        m->x[(size_t)t * E + k] += m->h[(size_t)t * E + k];
  }

  sgl_rmsnorm(m->x, m->final_norm.data, m->h, n, E, m->rms_eps);
  if (logits_out) {
    if (logits_rows == n) {
      for (int t = 0; t < n; t++)
        llama_logits(m, m->h + (size_t)t * E, logits_out + (size_t)t * m->vocab);
    } else {
      llama_logits(m, m->h + (size_t)(n - 1) * E, logits_out);
    }
  }
  return 0;
}

int sgl_llama_forward(sgl_llama *m, const int *tokens, int n, int pos0,
                      const int *slot_idx, float *logits_out) {
  return llama_forward_impl(m, tokens, n, pos0, slot_idx, logits_out, 0);
}

int sgl_llama_forward_n(sgl_llama *m, const int *tokens, int n, int pos0,
                        const int *slot_idx, float *logits_all) {
  return llama_forward_impl(m, tokens, n, pos0, slot_idx, logits_all, n);
}

int sgl_llama_forward_emb(sgl_llama *m, const float *emb, int n, int pos0,
                          const int *slot_idx, float *logits_out) {
  return llama_forward_emb_impl(m, NULL, emb, n, pos0, slot_idx, logits_out, 0);
}

/* batch decode: B sequences x 1 token, shared weights, one forward */
int sgl_llama_decode_batch(sgl_llama *m, const int *tokens, int *const *slot_idx,
                           const int *seq_lens, int B, float *logits_out) {
  int E = m->n_embd, H = m->n_head, HK = m->n_kv_head, D = m->head_dim;
  if (B <= 0 || B > 64) { SGL_ERROR("decode_batch: B=%d", B); return -1; }
  for (int b = 0; b < B; b++) {
    if (seq_lens[b] + 1 > m->n_ctx) { SGL_ERROR("decode: req %d pos>ctx", b); return -1; }
    for (int p = 0; p <= seq_lens[b]; p++)
      if (slot_idx[b][p] < 0 || slot_idx[b][p] >= m->n_pool) {
        SGL_ERROR("decode: req %d page oob at %d", b, p);
        return -1;
      }
  }

  sgl_embedding_get(m->wte.data, tokens, B, E, m->x);
  if (m->emb_softcap > 0) /* G9 gemma-3 embedding softcap */
    for (size_t i = 0; i < (size_t)B * E; i++)
      m->x[i] = m->emb_softcap * tanhf(m->x[i] / m->emb_softcap);
  if (m->gemma_scale > 0)
    for (size_t i = 0; i < (size_t)B * E; i++) m->x[i] *= m->gemma_scale;

  for (int l = 0; l < m->n_layer; l++) {
    sgl_llama_layer *L = &m->layers[l];
    sgl_rmsnorm(m->x, L->inp_norm.data, m->h, B, E, m->rms_eps);
    if (L->lin_idx >= 0) {
      /* G9 linear_attention：每序列单 token 递归（按各自首页归属状态）*/
      for (int b = 0; b < B; b++) {
        if (lin_attn_step(m, L, m->h + (size_t)b * E, slot_idx[b][0], seq_lens[b],
                          m->h + (size_t)b * E) != 0)
          return -1;
        for (int e = 0; e < E; e++)
          m->x[(size_t)b * E + e] += m->h[(size_t)b * E + e];
      }
    } else {
    proj_nt(m, m->h, &L->q_w, L->q8, L->sq8, L->q_b.data, m->q, B, m->q_dim, E);
    proj_nt(m, m->h, &L->k_w, L->k8, L->sk8, L->k_b.data, m->k, B, m->kv_dim, E);
    proj_nt(m, m->h, &L->v_w, L->v8, L->sv8, L->v_b.data, m->v, B, m->kv_dim, E);
    if (L->q_norm.data)
      for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
          sgl_rmsnorm(m->q + (size_t)b * m->q_dim + h * D, L->q_norm.data,
                      m->q + (size_t)b * m->q_dim + h * D, 1, D, 1e-6f);
    if (L->k_norm.data)
      for (int b = 0; b < B; b++)
        for (int hk = 0; hk < HK; hk++)
          sgl_rmsnorm(m->k + (size_t)b * m->kv_dim + hk * D, L->k_norm.data,
                      m->k + (size_t)b * m->kv_dim + hk * D, 1, D, 1e-6f);
    /* rope at each sequence's own position (heads = H/HK: the op rotates
     * n head rows at one pos — passing 1 left heads 1..H-1 un-rotated,
     * which diverged decode_batch from single-stream on the 2nd token)；
     * G9：与单流同用每层预计算逆频率 */
    for (int b = 0; b < B; b++) {
      double pos = (double)seq_lens[b] * m->rope_pos_scale;
      rope_layer(m, l, m->q + (size_t)b * m->q_dim, H, D, pos);
      rope_layer(m, l, m->k + (size_t)b * m->kv_dim, HK, D, pos);
    }

    if (m->kv8) {
      int8_t *kc8l = m->kc8 + (size_t)l * m->n_pool * m->kv_dim;
      int8_t *vc8l = m->vc8 + (size_t)l * m->n_pool * m->kv_dim;
      for (int b = 0; b < B; b++) {
        int pg = slot_idx[b][seq_lens[b]];
        m->kcs[(size_t)l * m->n_pool + pg] =
            sgl_quant_row_i8(m->k + (size_t)b * m->kv_dim, kc8l + (size_t)pg * m->kv_dim,
                             m->kv_dim);
        m->vcs[(size_t)l * m->n_pool + pg] =
            sgl_quant_row_i8(m->v + (size_t)b * m->kv_dim, vc8l + (size_t)pg * m->kv_dim,
                             m->kv_dim);
      }
    } else {
      float *kc = m->kc + (size_t)l * m->n_pool * m->kv_dim;
      float *vc = m->vc + (size_t)l * m->n_pool * m->kv_dim;
      for (int b = 0; b < B; b++) {
        int pg = slot_idx[b][seq_lens[b]];
        memcpy(kc + (size_t)pg * m->kv_dim, m->k + (size_t)b * m->kv_dim,
               (size_t)m->kv_dim * sizeof(float));
        memcpy(vc + (size_t)pg * m->kv_dim, m->v + (size_t)b * m->kv_dim,
               (size_t)m->kv_dim * sizeof(float));
      }
    }

    {
      const int8_t *kc8l = m->kv8 ? m->kc8 + (size_t)l * m->n_pool * m->kv_dim : NULL;
      const int8_t *vc8l = m->kv8 ? m->vc8 + (size_t)l * m->n_pool * m->kv_dim : NULL;
      const float *kcs = m->kv8 ? m->kcs + (size_t)l * m->n_pool : NULL;
      const float *vcs = m->kv8 ? m->vcs + (size_t)l * m->n_pool : NULL;
      const float *kc = m->kv8 ? NULL : m->kc + (size_t)l * m->n_pool * m->kv_dim;
      const float *vc = m->kv8 ? NULL : m->vc + (size_t)l * m->n_pool * m->kv_dim;
#pragma omp parallel for
      for (int bh = 0; bh < B * H; bh++) {
        int b = bh / H, h = bh % H;
        float *sc = m->scores;
#ifdef _OPENMP
        sc += (size_t)omp_get_thread_num() * m->n_ctx;
#endif
        const float *qh = m->q + (size_t)b * m->q_dim + h * D;
        size_t hoff = (size_t)(h / (H / HK)) * D;
        const int *idx = slot_idx[b];
        int upto = seq_lens[b] + 1;
        /* t9 mistral / G9 layer_types: 每层窗口 */
        int win = m->layer_window[l];
        int from = win > 0 && upto > win ? upto - win : 0;
        float maxv = -1e30f;
        for (int p = from; p < upto; p++) {
          int pg = idx[p];
          float s = 0;
          if (kc8l) {
            const int8_t *kh = kc8l + (size_t)pg * m->kv_dim + hoff;
            float acc = 0;
            for (int d = 0; d < D; d++) acc += qh[d] * (float)kh[d];
            s = acc * kcs[pg];
          } else {
            float dotv;
            sgl_dot_f32(qh, kc + (size_t)pg * m->kv_dim + hoff, D, &dotv);
            s = dotv;
          }
          s *= 1.0f / sqrtf((float)D);
          sc[p] = s;
          if (s > maxv) maxv = s;
        }
        double sum = 0;
        for (int p = from; p < upto; p++) { sc[p] = expf(sc[p] - maxv); sum += sc[p]; }
        float *out = m->attn + (size_t)b * m->q_dim + h * D;
        memset(out, 0, sizeof(float) * D);
        for (int p = from; p < upto; p++) {
          int pg = idx[p];
          float w = sc[p] / (float)sum;
          if (vc8l) {
            const int8_t *vh = vc8l + (size_t)pg * m->kv_dim + hoff;
            float sv = vcs[pg];
            for (int d = 0; d < D; d++) out[d] += w * ((float)vh[d] * sv);
          } else {
            const float *vh = vc + (size_t)pg * m->kv_dim + hoff;
            for (int d = 0; d < D; d++) out[d] += w * vh[d];
          }
        }
      }
    }

    proj_nt(m, m->attn, &L->o_w, L->o8, L->so8, NULL, m->h, B, E, m->q_dim);
    for (int b = 0; b < B; b++)
      for (int k = 0; k < E; k++)
        m->x[(size_t)b * E + k] += m->h[(size_t)b * E + k];
    } /* end else (full/sliding attention branch) */

    sgl_rmsnorm(m->x, L->post_norm.data, m->h, B, E, m->rms_eps);
    proj_nt(m, m->h, &L->gate_w, L->g8, L->sg8, NULL, m->gate, B, m->n_inter, E);
    proj_nt(m, m->h, &L->up_w, L->u8, L->su8, NULL, m->up, B, m->n_inter, E);
    sgl_silu_inplace(m->gate, B * m->n_inter);
    for (size_t i = 0; i < (size_t)B * m->n_inter; i++) m->gate[i] *= m->up[i];
    proj_nt(m, m->gate, &L->down_w, L->d8, L->sd8, NULL, m->h, B, E, m->n_inter);
    for (int b = 0; b < B; b++)
      for (int k = 0; k < E; k++)
        m->x[(size_t)b * E + k] += m->h[(size_t)b * E + k];
  }

  sgl_rmsnorm(m->x, m->final_norm.data, m->h, B, E, m->rms_eps);
  if (logits_out)
    for (int b = 0; b < B; b++)
      llama_logits(m, m->h + (size_t)b * E, logits_out + (size_t)b * m->vocab);
  return 0;
}
