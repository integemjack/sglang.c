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
#include "sgl/gguf.h"
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

/* t11 流式 int8：大矩阵取到 f32 后立刻量化并 drop owned 缓冲。
 * 返回 0 = 正常（含张量缺失跳过），-1 = 量化异常。 */
static int w8_stream_mat(sgl_llama *m, sgl_tensor *t, int8_t **q8, float **s8,
                         int64_t N, int64_t K) {
  if (!t->data || N <= 0 || K <= 0) return 0;
  if (!m->w8) return 0;
  w8_quant(t->data, q8, s8, N, K);
  sgl_st_drop(m->st, t);
  return 0;
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
  /* 配置来源：<dir>/config.json；没有的话从 <dir>/*.gguf 的元数据合成一份
   * （GGUF 单文件下载通常不带 config.json）。*/
  size_t sz = 0;
  char *buf = sgl_config_read_text(dir, &sz);
  if (!buf) {
    SGL_ERROR("llama: no config.json and no GGUF metadata in %s", dir);
    return NULL;
  }
  sgl_json *cfg = sgl_json_parse(buf, sz);
  sgl_xfree(buf);
  if (!cfg) return NULL;

  /* t11 qwen3_5（Qwen3.8 等，VLM 包装）：文本参数都在 text_config 里。
   * 顶层存在 text_config.hidden_size 时，之后所有读取切到 text_config 上
   *（model_type=qwen3_5、rope_parameters、layer_types、linear_* 都在其中）。 */
  sgl_json *cfg_root = cfg;
  int is_vlm_wrap = 0; /* t11：text_config 嵌套 = VLM 包装，权重在 model.language_model.* 下 */
  {
    sgl_json *tc = sgl_json_get(cfg, "text_config");
    if (tc && tc->type == SGL_JSON_OBJECT && sgl_json_get(tc, "hidden_size")) {
      cfg = tc;
      is_vlm_wrap = 1;
      SGL_INFO("llama config: nested text_config detected (VLM wrapper)");
    }
  }

  sgl_llama *m = (sgl_llama *)sgl_xcalloc(1, sizeof(*m));
  m->n_layer = sgl_json_get_int(cfg, "num_hidden_layers", 0);
  m->n_head = sgl_json_get_int(cfg, "num_attention_heads", 0);
  m->n_kv_head = sgl_json_get_int(cfg, "num_key_value_heads", m->n_head);
  m->n_embd = sgl_json_get_int(cfg, "hidden_size", 0);
  m->n_inter = sgl_json_get_int(cfg, "intermediate_size", 4 * m->n_embd);
  m->n_ctx = sgl_json_get_int(cfg, "max_position_embeddings", 2048);
  /* t11：ctx 緩冲（x/h/q/k/v/gate/up/qg ≈ ctx×88KB/f32）随 max_position_embeddings 线性膨胀，
   * 262144 的长上下文 config 会直接炸内存——默认钳到 32768，SGL_CTX 环境变量可改。 */
  {
    const char *e = getenv("SGL_CTX");
    if (e && atoi(e) > 0) m->n_ctx = atoi(e);
    else if (m->n_ctx > 32768) {
      SGL_INFO("ctx %d capped to 32768 (set SGL_CTX to override)", m->n_ctx);
      m->n_ctx = 32768;
    }
  }
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
  m->rope_attn_factor = 1.0; /* P1: 非 yarn 时乘 1 */
  {
    /* P1: 新/旧两种键名都要认：transformers 5.x 的 save_pretrained 写的是
     * `rope_parameters`（已归一化），旧模型/手写 config 写的是 `rope_scaling`。*/
    const sgl_json *rs = sgl_json_get(cfg, "rope_scaling");
    if (!rs || rs->type != SGL_JSON_OBJECT) rs = sgl_json_get(cfg, "rope_parameters");
    const char *rtype = rs ? sgl_json_get_str(rs, "rope_type", "default") : "default";
    double factor = rs ? sgl_json_get_num(rs, "factor", 1.0) : 1.0;
    double global_base = rs ? sgl_json_get_num(rs, "global_base_freq", 0.0) : 0.0;
    double local_base = sgl_json_get_num(cfg, "rope_local_base_freq", 0.0);
    if (factor <= 0.0) factor = 1.0;
    /* P1: YaRN（Llama-3.x / Qwen2.5 长上下文权重的 rope_type=yarn）：
     * 逐层逆频率表改用 sgl_rope_yarn_invf()，并把 attention_factor 乘到 cos/sin 上。
     * 频率表与 attention factor 均对 transformers 的 _compute_yarn_parameters 验过。*/
    int yarn = !strcmp(rtype, "yarn");
    if (yarn) {
      m->rope_attn_factor =
          sgl_rope_yarn_attention_factor(factor, rs ? sgl_json_get_num(rs, "mscale", 0) : 0,
                                         rs ? sgl_json_get_num(rs, "mscale_all_dim", 0) : 0);
      m->yarn_beta_fast = rs ? sgl_json_get_num(rs, "beta_fast", 32.0) : 32.0;
      m->yarn_beta_slow = rs ? sgl_json_get_num(rs, "beta_slow", 1.0) : 1.0;
      m->yarn_factor = factor;
      double orig = rs ? sgl_json_get_num(rs, "original_max_position_embeddings", 0.0) : 0.0;
      m->yarn_orig = orig > 0 ? orig : (double)m->n_ctx;
      SGL_INFO("rope_type=yarn factor=%g orig=%g beta=%.0f/%.0f attention_factor=%.6f",
               m->yarn_factor, m->yarn_orig, m->yarn_beta_fast, m->yarn_beta_slow,
               (double)m->rope_attn_factor);
    }
    /* layer_types：按字符串判类型；不认识的字符串宽松忽略（按 full_attention）*/
    const sgl_json *lt = sgl_json_get(cfg, "layer_types");
    /* G13: 真实 Qwen3-Next config 没有 layer_types，而是 full_attention_interval
     *（HF Qwen3NextConfig.__post_init__：full_attention if (i+1)%interval==0 else
     * linear_attention）。这里两种写法都支持。*/
    int fai = sgl_json_get_int(cfg, "full_attention_interval", 0);
    if ((!lt || lt->type != SGL_JSON_ARRAY) && fai > 0 && m->n_layer > 0) {
      m->layer_kind = (int *)sgl_xmalloc(sizeof(int) * (size_t)m->n_layer);
      for (int l = 0; l < m->n_layer; l++)
        m->layer_kind[l] = ((l + 1) % fai == 0) ? 0 : 2; /* 0=full 2=linear */
      SGL_INFO("layer_types from full_attention_interval=%d", fai);
    }
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
    /* ---- G13 Qwen3-Next full_attention 的真实特性 ---- */
    {
      const char *mt = sgl_json_get_str(cfg, "model_type", "");
      /* t11：qwen3_5（Qwen3.8，dense + gated deltanet）与 qwen3_next 同套
       * full 层特性：(1+w) 归一化 / q_proj 2×输出门控 / partial rotary。
       * 检查点可能不带 model_type（text_config 内），用 attn_output_gate
       *（该家族独有）兑底。*/
      int qnext = !strcmp(mt, "qwen3_next") || !strcmp(mt, "qwen3_5") ||
                  sgl_json_get(cfg, "attn_output_gate") != NULL;
      /* (c) 所有 RMSNorm 是 (1+w)*x/sqrt(var+eps)（Qwen3NextRMSNorm）*/
      m->norm_p1 = qnext;
      /* (a) q_proj 输出 2*q_dim，半幅 sigmoid 门控 attn 输出 */
      m->q_gate = qnext;
      /* (b) partial rotary：rope_parameters.partial_rotary_factor（新格式）或
       * 顶层 partial_rotary_factor（真实 80B config 的写法）*/
      double prf = 1.0;
      const sgl_json *rp = sgl_json_get(cfg, "rope_parameters");
      if (rp) prf = sgl_json_get_num(rp, "partial_rotary_factor", prf);
      prf = sgl_json_get_num(cfg, "partial_rotary_factor", prf);
      if (qnext && prf > 0.0 && prf < 1.0) {
        int rd = (int)((double)m->head_dim * prf);
        if (rd >= 2 && rd < m->head_dim && (rd % 2)) rd -= 1;
        if (rd >= 2 && rd < m->head_dim) m->rot_dim = rd;
      }
      /* (d) MoE：num_experts>0 且层不属 mlp_only_layers 且 (l+1)%decoder_sparse_step==0 */
      m->n_experts = sgl_json_get_int(cfg, "num_experts", 0);
      m->n_exp_topk = sgl_json_get_int(cfg, "num_experts_per_tok", m->n_experts);
      m->moe_inter = sgl_json_get_int(cfg, "moe_intermediate_size", 0);
      m->shared_inter = sgl_json_get_int(cfg, "shared_expert_intermediate_size", 0);
      m->sparse_step = sgl_json_get_int(cfg, "decoder_sparse_step", 1);
      m->norm_topk = sgl_json_get_int(cfg, "norm_topk_prob", 1);
      m->routed_scale = (float)sgl_json_get_num(cfg, "routed_scaling_factor", 1.0);
      if (m->routed_scale <= 0.0f) m->routed_scale = 1.0f; /* 真实 config: None → 1.0 */
      const sgl_json *mo = sgl_json_get(cfg, "mlp_only_layers");
      if (mo && mo->type == SGL_JSON_ARRAY && mo->count > 0) {
        m->n_mlp_only = (int)mo->count;
        m->mlp_only = (int *)sgl_xmalloc(sizeof(int) * (size_t)mo->count);
        for (size_t i = 0; i < mo->count; i++)
          m->mlp_only[i] = (mo->items[i] && mo->items[i]->type == SGL_JSON_NUMBER)
                              ? (int)mo->items[i]->num : -1;
      }
      if (m->n_experts > 0 && (m->moe_inter <= 0 || m->n_exp_topk <= 0 || m->n_exp_topk > m->n_experts)) {
        SGL_ERROR("MoE config invalid (experts=%d topk=%d moe_inter=%d)",
                  m->n_experts, m->n_exp_topk, m->moe_inter);
        sgl_llama_free(m); sgl_json_free(cfg); return NULL;
      }
      if (m->n_experts > 0) SGL_INFO("MoE: %d experts, top-%d, moe_inter=%d, shared_inter=%d, sparse_step=%d, norm_topk=%d, routed_scale=%g",
                                     m->n_experts, m->n_exp_topk, m->moe_inter, m->shared_inter,
                                     m->sparse_step, m->norm_topk, (double)m->routed_scale);
    }
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
    if (m->rot_dim > 0)
      m->rot_invf = (double *)sgl_xmalloc(sizeof(double) * (size_t)m->n_layer *
                                          (size_t)(m->rot_dim / 2));
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
      /* P1: YaRN——整张逆频率表覆盖（attention_factor 已在 rope_layer 里乘上）*/
      if (yarn)
        sgl_rope_yarn_invf(m->layer_invf + (size_t)l * half, m->head_dim, base, m->yarn_factor,
                           m->yarn_beta_fast, m->yarn_beta_slow, m->yarn_orig, 1);
      /* G13 partial rotary：旋转维数 rot_dim，逆频率分母用 rot_dim（同 HF：
       * inv_freq = 1/(base ** (arange(0,dim,2)/dim))，dim = head_dim*factor）*/
      if (m->rot_dim > 0) {
        int hr = m->rot_dim / 2;
        for (int i = 0; i < hr; i++)
          m->rot_invf[(size_t)l * hr + i] = pow(base, -(double)(2 * i) / (double)m->rot_dim);
        if (yarn)
          sgl_rope_yarn_invf(m->rot_invf + (size_t)l * hr, m->rot_dim, base, m->yarn_factor,
                             m->yarn_beta_fast, m->yarn_beta_slow, m->yarn_orig, 1);
      }
      /* 无 layer_types：全局 sliding_window 对所有层生效（mistral 旧行为）；
       * 有 layer_types：仅 sliding_attention 层开窗 */
      m->layer_window[l] = (kind == 1) ? m->sliding_window : 0;
    }
  }
  /* t11：qwen3_5（VLM 包装）的文本权重在 model.language_model.* 下。
   *（is_vlm_wrap 由 text_config 存在性决定，与 model_type 字段位置无关）*/
  sgl_json_free(cfg_root);
  {
    /* t11：显式 head_dim（qwen3 / qwen3_5：D≠E/H，如 5120/24→256）时
     * n_embd % n_head 的旧检查不适用，只要 q_dim/kv_dim 自洽即可。*/
    int hd_explicit = (m->head_dim > 0 && m->q_dim != m->n_embd);
    if (m->n_layer <= 0 || m->n_head <= 0 || m->n_embd <= 0 || m->vocab <= 0 ||
        m->n_kv_head <= 0 || (!hd_explicit && m->n_embd % m->n_head)) {
      SGL_ERROR("llama config invalid");
      sgl_llama_free(m);
      return NULL;
    }
  }
  SGL_INFO("Llama config: L=%d H=%d KVH=%d E=%d I=%d D=%d ctx=%d vocab=%d eps=%g theta=%g tie=%d",
           m->n_layer, m->n_head, m->n_kv_head, m->n_embd, m->n_inter, m->head_dim,
           m->n_ctx, m->vocab, m->rms_eps, m->rope_theta, m->tie_embeddings);

  m->st = sgl_st_open_dir(dir); /* P0-1: 支持 model.safetensors.index.json 多分片（t11：或散分片扫描） */
  if (!m->st) {
    sgl_llama_free(m);
    return NULL;
  }
  if (is_vlm_wrap) sgl_st_set_prefix(m->st, "model.language_model.");

  /* t11 流式 int8：SGL_W8=1 时大矩阵取到 f32 后立刻量化并 drop，
   * 宿存峰值从 ~4.8B 参数×4B 降到 int8 + 单张量瞬态（27B 模型 ~130GB → ~30GB）。
   * LoRA 需要在 f32 上合并，两者互斥（SGL_LORA 存在时不流式）。*/
  {
    const char *e = getenv("SGL_W8");
    if (e && atoi(e) != 0 && !getenv("SGL_LORA")) {
      m->w8 = 1;
      SGL_INFO("t11: streaming int8 load (SGL_W8) enabled");
    }
  }
  SGL_INFO("t11 dbg: loading wte");

  int64_t E = m->n_embd, K = m->kv_dim, I = m->n_inter, V = m->vocab;
  SGL_INFO("t11 dbg: entering layer loop");
  int64_t QD = (int64_t)m->n_head * m->head_dim; /* q/o width: E unless explicit head_dim (qwen3: 2048) */
  m->wte = sgl_st_get(m->st, "model.embed_tokens.weight", 2, (int64_t[]){V, E});
  SGL_INFO("t11 dbg: wte done %lld", (long long)(m->wte.data ? m->wte.shape[0] : -1));
  m->final_norm = sgl_st_get(m->st, "model.norm.weight", 1, (int64_t[]){E});
  SGL_INFO("t11 dbg: final_norm done");
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
    /* G13(d): 哪些层走 MoE（同 HF Qwen3NextDecoderLayer：不在 mlp_only_layers、
     * num_experts>0、(l+1)%decoder_sparse_step==0）。*/
    if (m->n_experts > 0 && (l + 1) % (m->sparse_step > 0 ? m->sparse_step : 1) == 0) {
      int only = 0;
      for (int i = 0; i < m->n_mlp_only; i++)
        if (m->mlp_only[i] == l) only = 1;
      L->is_moe = !only;
    }
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
    L->inp_norm = sgl_st_get(m->st, name, 1, (int64_t[]){E});
    /* G13(a): Qwen3-Next 的 q_proj 宽度是 2*q_dim（前一半 q，后一半 sigmoid 门控）*/
    int qw = m->q_gate ? 2 * m->q_dim : m->q_dim;
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
    L->q_w = sgl_st_get(m->st, name, 2, (int64_t[]){qw, E});
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", l);
    L->k_w = sgl_st_get(m->st, name, 2, (int64_t[]){K, E});
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", l);
    L->v_w = sgl_st_get(m->st, name, 2, (int64_t[]){K, E});
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
    L->o_w = sgl_st_get(m->st, name, 2, (int64_t[]){E, QD});
    /* t11 流式：全注意力层的大矩阵量化后立刻释放 f32（linear 层这些张量缺失，自动跳过）*/
    if (m->w8) {
      if (w8_stream_mat(m, &L->q_w, &L->q8, &L->sq8, qw, E) +
          w8_stream_mat(m, &L->k_w, &L->k8, &L->sk8, K, E) +
          w8_stream_mat(m, &L->v_w, &L->v8, &L->sv8, K, E) +
          w8_stream_mat(m, &L->o_w, &L->o8, &L->so8, E, QD) < 0)
        return NULL;
    }
    /* qwen2: separate q/k/v biases (HF) or fused qkv bias (vLLM layout) */
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.bias", l);
    {
      sgl_tensor b = sgl_st_get(m->st, name, 1, (int64_t[]){qw});
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
    /* qwen2: qk-norm (head_dim sized rmsnorm, applied before rope) —— 已在上方统一读取 */
    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
    L->post_norm = sgl_st_get(m->st, name, 1, (int64_t[]){E});
    if (L->is_moe) {
      /* G13(d) MoE：router + 专家（gate_up 融合）+ 共享专家 + 共享门控 */
      int mi = m->moe_inter, si = m->shared_inter > 0 ? m->shared_inter : mi;
      snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", l);
      L->router_w = sgl_st_get(m->st, name, 2, (int64_t[]){m->n_experts, E});
      snprintf(name, sizeof(name), "model.layers.%d.mlp.experts.gate_up_proj", l);
      L->exp_gate_up = sgl_st_get(m->st, name, 3, (int64_t[]){m->n_experts, 2 * mi, E});
      snprintf(name, sizeof(name), "model.layers.%d.mlp.experts.down_proj", l);
      L->exp_down = sgl_st_get(m->st, name, 3, (int64_t[]){m->n_experts, E, mi});
      snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", l);
      L->sh_gate_w = sgl_st_get(m->st, name, 2, (int64_t[]){si, E});
      snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", l);
      L->sh_up_w = sgl_st_get(m->st, name, 2, (int64_t[]){si, E});
      snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", l);
      L->sh_down_w = sgl_st_get(m->st, name, 2, (int64_t[]){E, si});
      snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", l);
      L->sh_gate = sgl_st_get(m->st, name, 2, (int64_t[]){1, E});
      if (!L->router_w.data || !L->exp_gate_up.data || !L->exp_down.data ||
          !L->sh_gate_w.data || !L->sh_up_w.data || !L->sh_down_w.data || !L->sh_gate.data) {
        SGL_ERROR("layer %d MoE weights missing", l);
        sgl_llama_free(m);
        return NULL;
      }
    } else {
      snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", l);
      L->gate_w = sgl_st_get(m->st, name, 2, (int64_t[]){I, E});
      snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", l);
      L->up_w = sgl_st_get(m->st, name, 2, (int64_t[]){I, E});
      snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", l);
      L->down_w = sgl_st_get(m->st, name, 2, (int64_t[]){E, I});
      if (m->w8) { /* t11 流式：dense SwiGLU 三矩阵 */
        if (w8_stream_mat(m, &L->gate_w, &L->g8, &L->sg8, I, E) +
            w8_stream_mat(m, &L->up_w, &L->u8, &L->su8, I, E) +
            w8_stream_mat(m, &L->down_w, &L->d8, &L->sd8, E, I) < 0)
          return NULL;
      }
    }
    /* qwen2 / G13: qk-norm（G13 的 Qwen3-Next 同样有，宽度 head_dim）*/
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
    if (!L->inp_norm.data || !L->post_norm.data ||
        (!L->is_moe && !((L->gate_w.data || L->g8) && (L->up_w.data || L->u8) &&
                          (L->down_w.data || L->d8)))) { /* t11：流式 int8 后张量为空、量化槽非空即存在 */
      SGL_ERROR("layer %d weights missing", l);
      sgl_llama_free(m);
      return NULL;
    }
    /* G9 linear_attention 层：gated deltanet 权重（名字与 HF Qwen3Next 检查点对齐）*/
    if (m->layer_kind && m->layer_kind[l] == 2) {
      int keyd = m->lin_nk * m->lin_kd, vald = m->lin_nh * m->lin_vd;
      int convd = 2 * keyd + vald;
      if (l == 0) SGL_INFO("t11 dbg: layer 0 linear_attn loading");
      snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkvz.weight", l);
      L->lin_qkvz_w = sgl_st_get(m->st, name, 2, (int64_t[]){2 * keyd + 2 * vald, E});
      snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_ba.weight", l);
      L->lin_ba_w = sgl_st_get(m->st, name, 2, (int64_t[]){2 * m->lin_nh, E});
      /* t11 qwen3_5（Qwen3.8）：in_proj 拆成四份 —— qkv[2keyd+vald]、z[vald]、b[nh]、a[nh]
       *（HF Qwen3_5GatedDeltaNet：qkv 连续 [q|k|v]，无组内交错）。这里在加载期拼回
       * Qwen3-Next 融合布局（qkvz 按 k-head 组 [q|k|v|z]、ba 按 [b|a]），lin_attn_step 零改动。*/
      if (!L->lin_qkvz_w.data || !L->lin_ba_w.data) {
        int nk = m->lin_nk, nh = m->lin_nh, kd = m->lin_kd, vd = m->lin_vd, r = m->lin_ratio;
        sgl_tensor qkv_t, z_t, b_t, a_t;
        snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", l);
        qkv_t = sgl_st_get(m->st, name, 2, (int64_t[]){2 * keyd + vald, E});
        snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", l);
        z_t = sgl_st_get(m->st, name, 2, (int64_t[]){vald, E});
        snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", l);
        b_t = sgl_st_get(m->st, name, 2, (int64_t[]){nh, E});
        snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", l);
        a_t = sgl_st_get(m->st, name, 2, (int64_t[]){nh, E});
        if (!qkv_t.data || !z_t.data || !b_t.data || !a_t.data) {
          SGL_ERROR("layer %d linear_attn in_proj missing (neither fused qkvz/ba nor split qkv/z/b/a)", l);
          sgl_llama_free(m);
          return NULL;
        }
        int64_t fq = 2 * keyd + 2 * vald, fb = 2 * nh;
        float *qkvz = (float *)sgl_xmalloc((size_t)fq * (size_t)E * sizeof(float));
        float *ba = (float *)sgl_xmalloc((size_t)fb * (size_t)E * sizeof(float));
        int gw = 2 * kd + 2 * r * vd; /* 融合布局每 k-head 组的行宽 */
        for (int g = 0; g < nk; g++) {
          /* t11 fix：行偏移要乘行宽 E（之前少了 *E，组 1..n 全部写串到组 0 内部，
           * 导致 in_proj_qkvz 权重错位——CPU/GPU 同时退化、lin 层输出全零的元凶）*/
          float *dst = qkvz + (size_t)g * gw * E;
          memcpy(dst, qkv_t.data + (size_t)g * kd * E,
                 sizeof(float) * (size_t)kd * E); /* q_g */
          memcpy(dst + (size_t)kd * E, qkv_t.data + ((size_t)keyd + (size_t)g * kd) * E,
                 sizeof(float) * (size_t)kd * E); /* k_g */
          memcpy(dst + (size_t)2 * kd * E,
                 qkv_t.data + ((size_t)2 * keyd + (size_t)g * r * vd) * E,
                 sizeof(float) * (size_t)r * vd * E); /* v 组内 r 个 v-head */
          memcpy(dst + (size_t)(2 * kd + r * vd) * E, z_t.data + (size_t)g * r * vd * E,
                 sizeof(float) * (size_t)r * vd * E); /* z 同上 r 个 head */
        }
        for (int g = 0; g < nk; g++)
          for (int j = 0; j < r; j++) {
            int h = g * r + j;
            memcpy(ba + (size_t)(g * 2 * r + j) * E, b_t.data + (size_t)h * E,
                   sizeof(float) * (size_t)E);
            memcpy(ba + (size_t)(g * 2 * r + r + j) * E, a_t.data + (size_t)h * E,
                   sizeof(float) * (size_t)E);
          }
        L->lin_qkvz_w = sgl_tensor_from_data(2, (int64_t[]){fq, E}, qkvz);
        L->lin_ba_w = sgl_tensor_from_data(2, (int64_t[]){fb, E}, ba);
        L->lin_synth = 1;
        /* t11：四个拆分张量的 f32 转换缓冲（~340MB/层）用完即放，
         * 否则 st_close 前一直挂在 owned 里，48 层叠 ~15GB。 */
        sgl_st_drop(m->st, &qkv_t);
        sgl_st_drop(m->st, &z_t);
        sgl_st_drop(m->st, &b_t);
        sgl_st_drop(m->st, &a_t);
        if (m->w8) {
          /* 合成缓冲直接量化后释放（st_drop 管不到堆合成缓冲）*/
          w8_quant(qkvz, &L->q8, &L->sq8, fq, E);
          sgl_xfree(qkvz);
          L->lin_qkvz_w = sgl_tensor_null();
          L->lin_synth = 0; /* 缓冲已释放，free 时不再 xfree */
        }
      } else if (m->w8) {
        /* t11：融合布局检查点（qwen3_next）在流式模式下也直接量化 qkvz */
      }
      /* t11 流式：qkvz（复用 q8 槽）与 out_proj（复用 o8 槽）量化后释放；
       * ba 仅 96×E 留 f32。在下方完整性检查之后执行。*/
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
      if (!((L->lin_qkvz_w.data || L->q8) && L->lin_ba_w.data && L->lin_conv_w.data &&
            L->lin_norm_w.data && (L->lin_out_w.data || L->o8) && L->lin_a_log &&
            L->lin_dt_bias)) { /* t11：流式后看量化槽 */
        SGL_ERROR("layer %d linear_attn weights missing", l);
        sgl_llama_free(m);
        return NULL;
      }
      L->lin_idx = m->n_lin++;
      /* t11 流式（在完整性检查后）：qkvz 复用 q8 槽、out_proj 复用 o8 槽。
       *（合成路径的 qkvz 已在上方量化释放，st_drop 对 null 张量无操作）*/
      if (m->w8) {
        if (w8_stream_mat(m, &L->lin_qkvz_w, &L->q8, &L->sq8, 2 * keyd + 2 * vald, E) != 0 ||
            w8_stream_mat(m, &L->lin_out_w, &L->o8, &L->so8, E, vald) != 0)
          return NULL;
      }
    } else {
      /* full/sliding：标准注意力权重必备（t11：流式 int8 后看量化槽）*/
      if (!((L->q_w.data || L->q8) && (L->k_w.data || L->k8) && (L->v_w.data || L->v8) &&
            (L->o_w.data || L->o8))) {
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
  /* G13: q_proj 2x 的临时区 + MoE scratch */
  if (m->q_gate) {
    m->qg = (float *)sgl_xmalloc(ctx * (size_t)2 * m->q_dim * sizeof(float));
    m->qgate = (float *)sgl_xmalloc(ctx * (size_t)m->q_dim * sizeof(float));
  }
  if (m->n_experts > 0) {
    int mi = m->moe_inter, si = m->shared_inter > 0 ? m->shared_inter : mi;
    int mx = mi > si ? mi : si;
    m->moe_gu = (float *)sgl_xmalloc((size_t)2 * mx * sizeof(float));
    m->moe_mid = (float *)sgl_xmalloc((size_t)mx * sizeof(float));
    m->moe_out = (float *)sgl_xmalloc((size_t)m->n_embd * sizeof(float));
    m->moe_rl = (float *)sgl_xmalloc((size_t)m->n_experts * sizeof(float));
  }

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
    m->lin_owner = (long long *)sgl_xmalloc(sizeof(long long) * (size_t)m->n_lin_slots);
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

  /* W8: quantize all projection weights after load (env SGL_W8=1)
   * t11：流式加载时已在加载点量化过，这里只处理旧路径（无 SGL_W8 提前读到的场景）*/
  {
    const char *env = getenv("SGL_W8");
    if (env && atoi(env) != 0) {
      if (!m->w8) m->w8 = 1;
      clock_t tq = clock();
      for (int l = 0; l < m->n_layer; l++) {
        sgl_llama_layer *Lq = &m->layers[l];
        if (Lq->lin_idx >= 0) continue; /* G9: linear 层不参与 W8（无 qkv/o 权重）*/
        if (Lq->is_moe) continue;        /* G13: MoE 层无 dense gate/up/down，不量化 (走 f32) */
        if (Lq->q8 || Lq->d8) continue;  /* t11：流式已量化 */
        w8_quant(Lq->q_w.data, &Lq->q8, &Lq->sq8, m->q_gate ? 2 * m->q_dim : m->q_dim, E);
        w8_quant(Lq->k_w.data, &Lq->k8, &Lq->sk8, K, E);
        w8_quant(Lq->v_w.data, &Lq->v8, &Lq->sv8, K, E);
        w8_quant(Lq->o_w.data, &Lq->o8, &Lq->so8, E, m->q_dim); /* t11: o 列是 q_dim（显式 head_dim 时 ≠E）*/
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
    if (L->lin_synth) { /* t11：qkvz/ba 合成堆缓冲 */
      sgl_xfree(L->lin_qkvz_w.data);
      sgl_xfree(L->lin_ba_w.data);
    }
  }
  sgl_xfree(m->x); sgl_xfree(m->h); sgl_xfree(m->q); sgl_xfree(m->k); sgl_xfree(m->v);
  sgl_xfree(m->attn); sgl_xfree(m->gate); sgl_xfree(m->up); sgl_xfree(m->scores);
  sgl_xfree(m->x8); sgl_xfree(m->x8s);
  sgl_xfree(m->kc); sgl_xfree(m->vc);
  sgl_xfree(m->kc8); sgl_xfree(m->vc8); sgl_xfree(m->kcs); sgl_xfree(m->vcs);
  sgl_xfree(m->free_pages); sgl_xfree(m->page_refs);
  /* G9 */
  sgl_xfree(m->layer_kind); sgl_xfree(m->layer_invf); sgl_xfree(m->layer_window);
  sgl_xfree(m->rot_invf); sgl_xfree(m->mlp_only); /* G13 */
  sgl_xfree(m->qg); sgl_xfree(m->qgate); sgl_xfree(m->moe_gu); sgl_xfree(m->moe_mid);
  sgl_xfree(m->moe_out); sgl_xfree(m->moe_rl);
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
/* G13: RMSNorm 分派——Qwen3-Next 是 (1+w)*x/sqrt(var+eps)（Qwen3NextRMSNorm），
 * 其余模型是普通的 w*x/sqrt(var+eps)。eps 由调用方给（layernorm 用 rms_eps，
 * qk-norm 用 1e-6，同 HF）。*/
static void rmsn(const sgl_llama *m, const float *x, const float *w, float *y,
                 int rows, int N, float eps) {
  if (m->norm_p1) sgl_rmsnorm_p1(x, w, y, rows, N, eps);
  else sgl_rmsnorm(x, w, y, rows, N, eps);
}

static void rope_layer(const sgl_llama *m, int l, float *qk, int n_heads, int D, double pos) {
  int half = D / 2;
  const double *invf = m->layer_invf + (size_t)l * (size_t)half;
  /* P1: YaRN 的 attention_factor 乘在 cos/sin 上（HF 同做法）*/
  const double sc = m->rope_attn_factor;
  for (int h = 0; h < n_heads; h++) {
    float *head = qk + (size_t)h * D;
    for (int i = 0; i < half; i++) {
      double ang = pos * invf[i];
      float c = (float)(cos(ang) * sc), s = (float)(sin(ang) * sc);
      float a = head[i], b = head[i + half];
      head[i] = a * c - b * s;
      head[i + half] = a * s + b * c;
    }
  }
}

/* G13(b) partial rotary：只旋转每头前 rot_dim 维（rotate_half 配对 (i, i+rot_dim/2)），
 * 逆频率分母用 rot_dim（HF：dim = head_dim*partial_rotary_factor）。
 * rot_dim==0 或 ==D 时直接走整头 rope_layer。*/
static void rope_layer_dyn(const sgl_llama *m, int l, float *qk, int n_heads, int D, double pos) {
  int rd = m->rot_dim;
  if (rd <= 0 || rd >= D) {
    rope_layer(m, l, qk, n_heads, D, pos);
    return;
  }
  int hr = rd / 2;
  const double *invf = m->rot_invf + (size_t)l * (size_t)hr;
  for (int h = 0; h < n_heads; h++) {
    float *head = qk + (size_t)h * D;
    for (int i = 0; i < hr; i++) {
      double ang = pos * invf[i];
      float c = (float)cos(ang), s = (float)sin(ang);
      float a = head[i], b = head[i + hr];
      head[i] = a * c - b * s;
      head[i + hr] = a * s + b * c;
    }
  }
}

/* G9/G13: 取/初始化某序列在 linear 层 ll 的递归状态槽。
 * G13 修正：归属键不再用「首个 KV 页号」。radix 前缀共享时两条序列会共用首页 0，
 * 旧实现让它们抢同一个槽 → 状态互踩（两条流各自的结果都错，而且错得很隐蔽）。
 * 现在用 64 位序列 key：调用方传 seq_id 就用它；不传时由 forward 自己用
 * slot_idx 数组地址当 key（每个序列一份页表，实际调用里天然唯一）。
 * 位置不连续（新序列 / 页被复用）则清零状态。*/
static int lin_slot_get(sgl_llama *m, int ll, long long seq_key, int pos) {
  int s = -1;
  for (int i = 0; i < m->n_lin_slots; i++)
    if (m->lin_owner[i] == seq_key) { s = i; break; }
  if (s < 0) {
    if (m->lin_nfree <= 0) {
      SGL_ERROR("linear attn: no free state slot (SGL_LIN_SLOTS=%d)", m->n_lin_slots);
      return -1;
    }
    s = m->lin_free[--m->lin_nfree];
    m->lin_owner[s] = seq_key;
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
 * 与真实模型的差距（G13 已补齐 full 层）：full_attention 现在有 q 门控 /
 * partial rotary / (1+w) 归一化 / MoE；本函数只处理 linear 层，不受影响。 */
static int lin_attn_step(sgl_llama *m, const sgl_llama_layer *L, const float *hrow,
                         long long seq_key, int pos, float *orow) {
  int nk = m->lin_nk, nh = m->lin_nh, kd = m->lin_kd, vd = m->lin_vd;
  int K = m->lin_conv, ratio = m->lin_ratio;
  int keyd = nk * kd, vald = nh * vd, convd = 2 * keyd + vald;
  int ll = L->lin_idx;
  int slot = lin_slot_get(m, ll, seq_key, pos);
  if (slot < 0) return -1;
  /* 1) 投影 */
  /* t11 流式：qkvz/out_proj 的 int8 走 W8A16 dp4a（L->q8/o8 复用槽）*/
  proj_nt(m, hrow, &L->lin_qkvz_w, L->q8, L->sq8, NULL, m->ls_qkvz, 1,
          2 * keyd + 2 * vald, m->n_embd);
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
  /* 7) out_proj（t11 流式：int8 复用 o8 槽）*/
  proj_nt(m, out, &L->lin_out_w, L->o8, L->so8, NULL, orow, 1, m->n_embd, vald);
  m->lin_pos[(size_t)ll * m->n_lin_slots + slot] = pos + 1;
  return 0;
}

/* G13(a): Qwen3-Next full 层的 q 门控。q_proj 输出 2*q_dim，按头交错
 * [q(D) | gate(D)]（HF：view(*shape, -1, 2*D).chunk(2, -1)），所以分开时
 * 每头取前 D → q、后 D → gate。q 写回 m->q 供后续 q_norm/rope/attn，
 * gate 存到 m->qgate（独立缓冲区，避免与 m->qg 自身重叠读写）。*/
static void q_gate_split(sgl_llama *m, int rows, int H, int D) {
  int qd = m->q_dim;
  for (int t = 0; t < rows; t++) {
    const float *src = m->qg + (size_t)t * 2 * qd;
    float *dstq = m->q + (size_t)t * qd;
    float *dstg = m->qgate + (size_t)t * qd;
    for (int h = 0; h < H; h++) {
      memcpy(dstq + (size_t)h * D, src + (size_t)h * 2 * D, sizeof(float) * D);
      memcpy(dstg + (size_t)h * D, src + (size_t)h * 2 * D + D, sizeof(float) * D);
    }
  }
}

/* G13(a) 后半：attn 输出逐元素乘 sigmoid(gate)（在 o_proj 之前、逐头逐维）。*/
static void q_gate_apply(sgl_llama *m, int rows) {
  int qd = m->q_dim;
  for (int t = 0; t < rows; t++) {
    float *a = m->attn + (size_t)t * qd;
    const float *g = m->qgate + (size_t)t * qd;
    for (int i = 0; i < qd; i++) a[i] *= 1.0f / (1.0f + expf(-g[i]));
  }
}

/* G13(d) MoE 单行：router（softmax → top-k → 可选重归一化）→ 专家 gate_up/down
 * 按权重（× routed_scaling_factor）累加 → 加共享专家（×sigmoid(共享门控)）。
 * 公式照 HF Qwen3NextSparseMoeBlock / Qwen3NextTopKRouter / Qwen3NextExperts。*/
static int moe_row(sgl_llama *m, sgl_llama_layer *L, const float *hrow, float *orow) {
  int E = m->n_embd, NE = m->n_experts, mi = m->moe_inter;
  int si = m->shared_inter > 0 ? m->shared_inter : mi;
  float *rl = m->moe_rl;
  sgl_matmul_nt(hrow, L->router_w.data, NULL, rl, 1, NE, E);
  /* softmax */
  float mx = rl[0];
  for (int i = 1; i < NE; i++) if (rl[i] > mx) mx = rl[i];
  double ssum = 0;
  for (int i = 0; i < NE; i++) { rl[i] = expf(rl[i] - mx); ssum += rl[i]; }
  for (int i = 0; i < NE; i++) rl[i] = (float)(rl[i] / ssum);
  /* top-k（NE 小，直接选）*/
  int sel[32]; float sw[32];
  int k = m->n_exp_topk;
  for (int j = 0; j < k; j++) {
    int best = -1; float bv = -1.0f;
    for (int i = 0; i < NE; i++) {
      int used = 0;
      for (int q = 0; q < j; q++) if (sel[q] == i) used = 1;
      if (!used && rl[i] > bv) { bv = rl[i]; best = i; }
    }
    sel[j] = best; sw[j] = bv;
  }
  if (m->norm_topk) {
    float t = 0;
    for (int j = 0; j < k; j++) t += sw[j];
    if (t > 0) for (int j = 0; j < k; j++) sw[j] /= t;
  }
  memset(orow, 0, sizeof(float) * (size_t)E);
  for (int j = 0; j < k; j++) {
    int e = sel[j];
    if (e < 0) continue;
    const float *gu = L->exp_gate_up.data + (size_t)e * 2 * mi * E;
    sgl_matmul_nt(hrow, gu, NULL, m->moe_gu, 1, 2 * mi, E);
    for (int i = 0; i < mi; i++) {
      float g = m->moe_gu[i];
      m->moe_mid[i] = (g / (1.0f + expf(-g))) * m->moe_gu[mi + i]; /* silu(gate)*up */
    }
    const float *dn = L->exp_down.data + (size_t)e * E * mi;
    sgl_matmul_nt(m->moe_mid, dn, NULL, m->moe_out, 1, E, mi);
    float w = sw[j] * m->routed_scale;
    for (int i = 0; i < E; i++) orow[i] += w * m->moe_out[i];
  }
  /* 共享专家：out += sigmoid(h·sh_gate) * sh_down(silu(sh_gate_w h) * sh_up_w h) */
  if (L->sh_gate_w.data && si > 0) {
    float sg = 0;
    for (int i = 0; i < E; i++) sg += hrow[i] * L->sh_gate.data[i];
    sg = 1.0f / (1.0f + expf(-sg));
    sgl_matmul_nt(hrow, L->sh_gate_w.data, NULL, m->moe_gu, 1, si, E);
    sgl_matmul_nt(hrow, L->sh_up_w.data, NULL, m->moe_mid, 1, si, E);
    for (int i = 0; i < si; i++) {
      float g = m->moe_gu[i];
      m->moe_gu[i] = (g / (1.0f + expf(-g))) * m->moe_mid[i];
    }
    sgl_matmul_nt(m->moe_gu, L->sh_down_w.data, NULL, m->moe_out, 1, E, si);
    for (int i = 0; i < E; i++) orow[i] += sg * m->moe_out[i];
  }
  return 0;
}

/* MLP 块（行级）：dense SwiGLU 或 MoE，结果写 out（[rows][E]，不复用 m->h，避免与调用方别名）。*/
static int mlp_rows(sgl_llama *m, sgl_llama_layer *L, const float *h, int rows, float *out) {
  int E = m->n_embd;
  if (L->is_moe) {
    for (int t = 0; t < rows; t++)
      if (moe_row(m, L, h + (size_t)t * E, out + (size_t)t * E) != 0) return -1;
    return 0;
  }
  /* t11：流式 int8 后 gate/up/down 的 f32 张量为空，统一走 proj_nt（W8A16/f32 分派）*/
  proj_nt(m, h, &L->gate_w, L->g8, L->sg8, NULL, m->gate, rows, m->n_inter, E);
  proj_nt(m, h, &L->up_w, L->u8, L->su8, NULL, m->up, rows, m->n_inter, E);
  sgl_silu_inplace(m->gate, rows * m->n_inter);
  for (size_t i = 0; i < (size_t)rows * m->n_inter; i++) m->gate[i] *= m->up[i];
  proj_nt(m, m->gate, &L->down_w, L->d8, L->sd8, NULL, out, rows, E, m->n_inter);
  return 0;
}

/* 共享实现：logits_rows==0 时只在 logits_out 返回末位分布（旧语义）；
 * logits_rows==n 时在 logits_all 返回全部 n 位分布（sgl_llama_forward_n）。 */
static int llama_forward_emb_impl(sgl_llama *m, const int *tokens, const float *emb,
                                  int n, int pos0, const int *slot_idx, long long seq_key,
                                  float *logits_out, int logits_rows);
static int llama_forward_impl(sgl_llama *m, const int *tokens, int n, int pos0,
                              const int *slot_idx, long long seq_key, float *logits_out,
                              int logits_rows) {
  return llama_forward_emb_impl(m, tokens, NULL, n, pos0, slot_idx, seq_key, logits_out,
                                logits_rows);
}

/* t10 llava: emb != NULL 时直接用给定的 embedding 序列 [n][E]（视觉 token 融合），
 * 跳过 token 查表；其余与 token 前向完全一致（同一层循环）。 */
static int llama_forward_emb_impl(sgl_llama *m, const int *tokens, const float *emb,
                                  int n, int pos0, const int *slot_idx, long long seq_key,
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
    rmsn(m, m->x, L->inp_norm.data, m->h, n, E, m->rms_eps);
    if (L->lin_idx >= 0) {
      /* G9 linear_attention：无 KV 页，逐 token 递归（串行）；out_proj 写回 h 行 t
       * 时 h 行已被投影消费，不别名冲突（matmul 输入是 ls_o）。 G13：递归状态按
       * seq_key 归属（seq_key 由入口决定：显式 seq_id 或 slot_idx 指针）。*/
      for (int t = 0; t < n; t++) {
        if (lin_attn_step(m, L, m->h + (size_t)t * E, seq_key, pos0 + t,
                          m->h + (size_t)t * E) != 0)
          return -1;
        for (int e = 0; e < E; e++)
          m->x[(size_t)t * E + e] += m->h[(size_t)t * E + e];
      }
    } else {
    if (m->q_gate)
      proj_nt(m, m->h, &L->q_w, L->q8, L->sq8, L->q_b.data, m->qg, n, 2 * m->q_dim, E);
    else
      proj_nt(m, m->h, &L->q_w, L->q8, L->sq8, L->q_b.data, m->q, n, m->q_dim, E);
    proj_nt(m, m->h, &L->k_w, L->k8, L->sk8, L->k_b.data, m->k, n, m->kv_dim, E);
    proj_nt(m, m->h, &L->v_w, L->v8, L->sv8, L->v_b.data, m->v, n, m->kv_dim, E);
    if (m->q_gate) q_gate_split(m, n, H, D); /* G13(a) */
    /* qwen2 / G13 qk-norm (before rope)：G13 的 RMSNorm 是 (1+w) */
    if (L->q_norm.data)
      for (int t = 0; t < n; t++)
        rmsn(m, m->q + (size_t)t * m->q_dim, L->q_norm.data,
             m->q + (size_t)t * m->q_dim, H, D, 1e-6f);
    if (L->k_norm.data)
      for (int t = 0; t < n; t++)
        rmsn(m, m->k + (size_t)t * m->kv_dim, L->k_norm.data,
             m->k + (size_t)t * m->kv_dim, HK, D, 1e-6f);
    /* rope: per-token position (rows of the same token share pos0+t)；G9 每层预计算逆频率；
     * G13 partial rotary：只旋转每头前 rot_dim 维（rope_layer_dyn 内部判断）*/
    for (int t = 0; t < n; t++) {
      double pos = (double)(pos0 + t) * m->rope_pos_scale;
      rope_layer_dyn(m, l, m->q + (size_t)t * m->q_dim, H, D, pos);
      rope_layer_dyn(m, l, m->k + (size_t)t * m->kv_dim, HK, D, pos);
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

    if (m->q_gate) q_gate_apply(m, n); /* G13(a): attn *= sigmoid(gate) */
    proj_nt(m, m->attn, &L->o_w, L->o8, L->so8, NULL, m->h, n, E, m->q_dim);
    for (int t = 0; t < n; t++)
      for (int k = 0; k < E; k++)
        m->x[(size_t)t * E + k] += m->h[(size_t)t * E + k];
    } /* end else (full/sliding attention branch) */

    /* MLP：dense SwiGLU 或 G13 MoE（mlp_rows 内部区分）*/
    rmsn(m, m->x, L->post_norm.data, m->h, n, E, m->rms_eps);
    if (mlp_rows(m, L, m->h, n, m->up) != 0) return -1;
    for (int t = 0; t < n; t++)
      for (int k = 0; k < E; k++)
        m->x[(size_t)t * E + k] += m->up[(size_t)t * E + k];
    {
      /* 与 GPU 侧 SGL_CUDA_DBG 对拍：[gpu L%d] 打第 0 行前 4 个，[gpu lin L%d] 打末行前 2 个 */
      static int dbg = -1;
      if (dbg < 0) { const char *e = getenv("SGL_LLAMA_DBG"); dbg = e ? atoi(e) : 0; }
      if (dbg > 0 && l % dbg == 0) {
        const float *xl = m->x + (size_t)(n - 1) * E;
        printf("[cpu L%d] x=[%.7g, %.7g, %.7g, %.7g] last=[%.7g,%.7g]\n", l, m->x[0], m->x[1],
               m->x[2], m->x[3], xl[0], xl[1]);
      }
    }
  }

  rmsn(m, m->x, m->final_norm.data, m->h, n, E, m->rms_eps);
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
  return llama_forward_impl(m, tokens, n, pos0, slot_idx, (long long)(uintptr_t)slot_idx,
                            logits_out, 0);
}

int sgl_llama_forward_seq(sgl_llama *m, const int *tokens, int n, int pos0,
                          const int *slot_idx, long long seq_id, float *logits_out) {
  return llama_forward_impl(m, tokens, n, pos0, slot_idx, seq_id, logits_out, 0);
}

int sgl_llama_forward_n(sgl_llama *m, const int *tokens, int n, int pos0,
                        const int *slot_idx, float *logits_all) {
  return llama_forward_impl(m, tokens, n, pos0, slot_idx, (long long)(uintptr_t)slot_idx,
                            logits_all, n);
}

int sgl_llama_forward_emb(sgl_llama *m, const float *emb, int n, int pos0,
                          const int *slot_idx, float *logits_out) {
  return llama_forward_emb_impl(m, NULL, emb, n, pos0, slot_idx,
                                (long long)(uintptr_t)slot_idx, logits_out, 0);
}

/* G13: 释放某序列占用的递归状态槽（计时/显存都不占，但槽位有限，用完要还）。*/
void sgl_llama_lin_release(sgl_llama *m, long long seq_id) {
  if (!m || m->n_lin <= 0) return;
  for (int s = 0; s < m->n_lin_slots; s++) {
    if (m->lin_owner[s] != seq_id) continue;
    m->lin_owner[s] = -1;
    for (int ll = 0; ll < m->n_lin; ll++) m->lin_pos[(size_t)ll * m->n_lin_slots + s] = -1;
    if (m->lin_nfree < m->n_lin_slots) m->lin_free[m->lin_nfree++] = s;
  }
}

int sgl_llama_lin_slots_in_use(sgl_llama *m) {
  return m && m->n_lin > 0 ? m->n_lin_slots - m->lin_nfree : 0;
}

/* batch decode: B sequences x 1 token, shared weights, one forward
 * G13: 线性层的递归状态按序列 key 归属（不传 seq_ids 时用各 seq 的页表指针）。*/
static int decode_batch_impl(sgl_llama *m, const int *tokens, int *const *slot_idx,
                             const long long *seq_ids, const int *seq_lens, int B,
                             float *logits_out) {
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
    rmsn(m, m->x, L->inp_norm.data, m->h, B, E, m->rms_eps);
    if (L->lin_idx >= 0) {
      /* G9 linear_attention：每序列单 token 递归；G13 按序列 key 归属状态 */
      for (int b = 0; b < B; b++) {
        long long k = seq_ids ? seq_ids[b] : (long long)(uintptr_t)slot_idx[b];
        if (lin_attn_step(m, L, m->h + (size_t)b * E, k, seq_lens[b],
                          m->h + (size_t)b * E) != 0)
          return -1;
        for (int e = 0; e < E; e++)
          m->x[(size_t)b * E + e] += m->h[(size_t)b * E + e];
      }
    } else {
    if (m->q_gate)
      proj_nt(m, m->h, &L->q_w, L->q8, L->sq8, L->q_b.data, m->qg, B, 2 * m->q_dim, E);
    else
      proj_nt(m, m->h, &L->q_w, L->q8, L->sq8, L->q_b.data, m->q, B, m->q_dim, E);
    proj_nt(m, m->h, &L->k_w, L->k8, L->sk8, L->k_b.data, m->k, B, m->kv_dim, E);
    proj_nt(m, m->h, &L->v_w, L->v8, L->sv8, L->v_b.data, m->v, B, m->kv_dim, E);
    if (m->q_gate) q_gate_split(m, B, H, D);
    if (L->q_norm.data)
      for (int b = 0; b < B; b++)
        rmsn(m, m->q + (size_t)b * m->q_dim, L->q_norm.data,
             m->q + (size_t)b * m->q_dim, H, D, 1e-6f);
    if (L->k_norm.data)
      for (int b = 0; b < B; b++)
        rmsn(m, m->k + (size_t)b * m->kv_dim, L->k_norm.data,
             m->k + (size_t)b * m->kv_dim, HK, D, 1e-6f);
    /* rope at each sequence's own position (heads = H/HK: the op rotates
     * n head rows at one pos — passing 1 left heads 1..H-1 un-rotated,
     * which diverged decode_batch from single-stream on the 2nd token)；
     * G9：与单流同用每层预计算逆频率；G13：partial rotary */
    for (int b = 0; b < B; b++) {
      double pos = (double)seq_lens[b] * m->rope_pos_scale;
      rope_layer_dyn(m, l, m->q + (size_t)b * m->q_dim, H, D, pos);
      rope_layer_dyn(m, l, m->k + (size_t)b * m->kv_dim, HK, D, pos);
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

    if (m->q_gate) q_gate_apply(m, B); /* G13(a) */
    proj_nt(m, m->attn, &L->o_w, L->o8, L->so8, NULL, m->h, B, E, m->q_dim);
    for (int b = 0; b < B; b++)
      for (int k = 0; k < E; k++)
        m->x[(size_t)b * E + k] += m->h[(size_t)b * E + k];
    } /* end else (full/sliding attention branch) */

    /* MLP：dense 或 G13 MoE */
    rmsn(m, m->x, L->post_norm.data, m->h, B, E, m->rms_eps);
    if (mlp_rows(m, L, m->h, B, m->up) != 0) return -1;
    for (int b = 0; b < B; b++)
      for (int k = 0; k < E; k++)
        m->x[(size_t)b * E + k] += m->up[(size_t)b * E + k];
  }

  rmsn(m, m->x, m->final_norm.data, m->h, B, E, m->rms_eps);
  if (logits_out)
    for (int b = 0; b < B; b++)
      llama_logits(m, m->h + (size_t)b * E, logits_out + (size_t)b * m->vocab);
  return 0;
}

int sgl_llama_decode_batch(sgl_llama *m, const int *tokens, int *const *slot_idx,
                           const int *seq_lens, int B, float *logits_out) {
  return decode_batch_impl(m, tokens, slot_idx, NULL, seq_lens, B, logits_out);
}

int sgl_llama_decode_batch_seq(sgl_llama *m, const int *tokens, int *const *slot_idx,
                               const long long *seq_ids, const int *seq_lens, int B,
                               float *logits_out) {
  return decode_batch_impl(m, tokens, slot_idx, seq_ids, seq_lens, B, logits_out);
}
