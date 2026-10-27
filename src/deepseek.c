/* sgl deepseek 实现：DeepSeek-V2 / V3 系完整前向（MLA + MoE）
 * 公式照 HF modeling_deepseek_v2.py / modeling_deepseek_v3.py（结构见 sgl/deepseek.h）。
 * 只依赖 sgl/ops.h 的算子 + sgl_xmalloc/sgl_xfree + safetensors。
 *
 * 与 llama.c 的对应关系（同一套后端约定）：
 *   proj_nt / rmsn  → sgl_matmul_nt / sgl_rmsnorm（DeepSeek 无 (1+w) 变体）
 *   moe_row         → ds_moe_row（多两件事：sigmod 打分 + group_limited_greedy 选组、
 *                     共享专家无 sigmoid 门控）
 *   mlp_rows        → ds_mlp_rows（dense SwiGLU / MoE 分派相同）
 *   分页 KV 池      → 完全同语义，只是页大小从 kv_dim 变成 stride=kv_lora+qk_rope
 *   prefill         → ds_forward_impl（统一入口：每行自带绝对位置 pos 与页表 tab，
 *                     prefill（多行同一页表）/ decode_batch（每行自己的页表）共用）
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/deepseek.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sgl/alloc.h"
#include "sgl/gguf.h"
#include "sgl/json.h"
#include "sgl/log.h"
#include "sgl/ops.h"

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

/* 取 2D/1D 权重；形状不符或缺失时报错并返回 null tensor */
static sgl_tensor get_2d_(sgl_st_ctx *st, const char *name, int64_t rows, int64_t cols) {
  int64_t sh[2] = {rows, cols};
  sgl_tensor t = sgl_st_get(st, name, 2, sh);
  if (sgl_tensor_is_null(t))
    SGL_ERROR("deepseek: tensor '%s' missing or shape != [%lld,%lld]", name, (long long)rows,
              (long long)cols);
  return t;
}

static sgl_tensor get_3d_(sgl_st_ctx *st, const char *name, int64_t d0, int64_t d1, int64_t d2) {
  int64_t sh[3] = {d0, d1, d2};
  sgl_tensor t = sgl_st_get(st, name, 3, sh);
  if (sgl_tensor_is_null(t))
    SGL_ERROR("deepseek: tensor '%s' missing or shape != [%lld,%lld,%lld]", name, (long long)d0,
              (long long)d1, (long long)d2);
  return t;
}

static sgl_tensor get_1d_(sgl_st_ctx *st, const char *name, int64_t n) {
  int64_t sh[1] = {n};
  sgl_tensor t = sgl_st_get(st, name, 1, sh);
  if (sgl_tensor_is_null(t))
    SGL_ERROR("deepseek: tensor '%s' missing or shape != [%lld]", name, (long long)n);
  return t;
}

static int has_tensor_(sgl_st_ctx *st, const char *name) {
  const char **names = NULL;
  int n = sgl_st_names(st, &names);
  for (int i = 0; i < n; i++)
    if (names[i] && !strcmp(names[i], name)) return 1;
  return 0;
}

/* 可选张量：不存在时返回 null（不打 ERROR）*/
static sgl_tensor get_opt_1d_(sgl_st_ctx *st, const char *name, int64_t n) {
  if (!has_tensor_(st, name)) return sgl_tensor_null();
  return get_1d_(st, name, n);
}

/* ---------------- 模型识别 ---------------- */

int sgl_deepseek_detect(const char *model_dir) {
  size_t sz = 0;
  char *buf = sgl_config_read_text(model_dir, &sz);
  if (!buf || sz == 0) {
    sgl_xfree(buf);
    return -1;
  }
  sgl_json *cfg = sgl_json_parse(buf, sz);
  sgl_xfree(buf);
  if (!cfg) return -1;
  int yes = 0;
  const char *mt = sgl_json_get_str(cfg, "model_type", "");
  const sgl_json *arch = sgl_json_get(cfg, "architectures");
  int has_arch = 0;
  if (arch && arch->type == SGL_JSON_ARRAY)
    for (int i = 0; i < arch->count; i++) {
      const char *s = (arch->items[i] && arch->items[i]->type == SGL_JSON_STRING)
                          ? arch->items[i]->str
                          : "";
      if (s && !strncmp(s, "Deepseek", 8)) has_arch = 1;
    }
  if (!strncmp(mt, "deepseek_v2", 11) || !strncmp(mt, "deepseek_v3", 11)) yes = 1;
  if (has_arch) yes = 1;
  /* 结构特征：MLA 的三个必需字段同时存在（model_type 可能被改写过）*/
  if (sgl_json_get(cfg, "kv_lora_rank") && sgl_json_get(cfg, "qk_nope_head_dim") &&
      sgl_json_get(cfg, "qk_rope_head_dim"))
    yes = 1;
  sgl_json_free(cfg);
  return yes;
}

/* ---------------- 加载 ---------------- */

static int cfg_int_(const sgl_json *c, const char *k, int def) {
  const sgl_json *v = sgl_json_get(c, k);
  if (!v) return def;
  if (v->type == SGL_JSON_BOOL) return v->boolean; /* "norm_topk_prob": true 这类布尔字段 */
  if (v->type != SGL_JSON_NUMBER) return def;     /* null / 缺失 → def */
  return (int)v->num;
}

static double cfg_num_(const sgl_json *c, const char *k, double def) {
  const sgl_json *v = sgl_json_get(c, k);
  if (!v || v->type != SGL_JSON_NUMBER) return def;
  return v->num;
}

/* 一层 MoE 的专家权重：两种布局（HF V3 三维融合 / 官方权重逐专家）都收 */
static int load_moe_(sgl_deepseek *m, int l, sgl_deepseek_layer *L) {
  int E = m->n_embd, NE = m->n_experts, mi = m->moe_inter, sh = m->shared_inter;
  char nm[600];
  snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.weight", l);
  L->router_w = get_2d_(m->st, nm, NE, E);
  if (sgl_tensor_is_null(L->router_w)) return -1;
  snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.e_score_correction_bias", l);
  {
    sgl_tensor b = get_opt_1d_(m->st, nm, NE);
    if (b.data) {
      L->ec_bias = (float *)sgl_xmalloc(sizeof(float) * (size_t)NE);
      memcpy(L->ec_bias, b.data, sizeof(float) * (size_t)NE);
    }
  }
  /* 布局 A：三维融合 experts.gate_up_proj [NE][2mi][E] + experts.down_proj [NE][E][mi] */
  snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.gate_up_proj.weight", l);
  if (has_tensor_(m->st, nm)) {
    L->exp_stacked = 1;
    L->exp_gu3 = get_3d_(m->st, nm, NE, 2 * mi, E);
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.down_proj.weight", l);
    L->exp_dn3 = get_3d_(m->st, nm, NE, E, mi);
    if (sgl_tensor_is_null(L->exp_gu3) || sgl_tensor_is_null(L->exp_dn3)) return -1;
  } else {
    /* 布局 B：逐专家 experts.{e}.{gate,up,down}_proj.weight */
    L->exp_gate = (sgl_tensor *)sgl_xcalloc((size_t)NE, sizeof(sgl_tensor));
    L->exp_up = (sgl_tensor *)sgl_xcalloc((size_t)NE, sizeof(sgl_tensor));
    L->exp_down = (sgl_tensor *)sgl_xcalloc((size_t)NE, sizeof(sgl_tensor));
    for (int e = 0; e < NE; e++) {
      snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.gate_proj.weight", l, e);
      L->exp_gate[e] = get_2d_(m->st, nm, mi, E);
      snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.up_proj.weight", l, e);
      L->exp_up[e] = get_2d_(m->st, nm, mi, E);
      snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.down_proj.weight", l, e);
      L->exp_down[e] = get_2d_(m->st, nm, E, mi);
      if (sgl_tensor_is_null(L->exp_gate[e]) || sgl_tensor_is_null(L->exp_up[e]) ||
          sgl_tensor_is_null(L->exp_down[e]))
        return -1;
    }
  }
  if (sh > 0) {
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_experts.gate_proj.weight", l);
    L->sh_gate_w = get_2d_(m->st, nm, sh, E);
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_experts.up_proj.weight", l);
    L->sh_up_w = get_2d_(m->st, nm, sh, E);
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_experts.down_proj.weight", l);
    L->sh_down_w = get_2d_(m->st, nm, E, sh);
    if (sgl_tensor_is_null(L->sh_gate_w) || sgl_tensor_is_null(L->sh_up_w) ||
        sgl_tensor_is_null(L->sh_down_w))
      return -1;
  }
  return 0;
}

sgl_deepseek *sgl_deepseek_load(const char *dir) {
  if (!dir || !*dir) return NULL;
  size_t sz = 0;
  char *buf = sgl_config_read_text(dir, &sz);
  if (!buf || sz == 0) {
    SGL_ERROR("deepseek: no config.json and no GGUF metadata in %s", dir);
    sgl_xfree(buf);
    return NULL;
  }
  sgl_json *cfg = sgl_json_parse(buf, sz);
  sgl_xfree(buf);
  if (!cfg) return NULL;

  sgl_deepseek *m = (sgl_deepseek *)sgl_xcalloc(1, sizeof(*m));
  m->n_layer = cfg_int_(cfg, "num_hidden_layers", 0);
  m->n_head = cfg_int_(cfg, "num_attention_heads", 0);
  m->n_embd = cfg_int_(cfg, "hidden_size", 0);
  m->n_ctx = cfg_int_(cfg, "max_position_embeddings", 4096);
  m->vocab = cfg_int_(cfg, "vocab_size", 0);
  m->q_lora = cfg_int_(cfg, "q_lora_rank", 0); /* null → 0（无 q 低秩）*/
  m->kv_lora = cfg_int_(cfg, "kv_lora_rank", 0);
  m->qk_nope = cfg_int_(cfg, "qk_nope_head_dim", 0);
  m->qk_rope = cfg_int_(cfg, "qk_rope_head_dim", 0);
  m->v_head = cfg_int_(cfg, "v_head_dim", m->qk_nope);
  m->rms_eps = (float)cfg_num_(cfg, "rms_norm_eps", 1e-6);
  m->rope_theta = cfg_num_(cfg, "rope_theta", 10000.0);
  /* P1 修正：DeepSeek 的 RoPE 是【交错】配对 (2i, 2i+1)（view_as_complex + freqs_cis 复乘），
   * 与 transformers DeepseekV2 的 apply_rotary_emb 一致；非交错 rotate_half 只用于
   * 显式写了 rope_interleave:false 的旧测试模型（兼容用）。*/
  m->rope_interleave = cfg_int_(cfg, "rope_interleave", 1);
  m->n_inter = cfg_int_(cfg, "intermediate_size", 4 * m->n_embd);
  m->n_experts = cfg_int_(cfg, "n_routed_experts", cfg_int_(cfg, "num_experts", 0));
  m->n_topk = cfg_int_(cfg, "num_experts_per_tok", 0);
  m->moe_inter = cfg_int_(cfg, "moe_intermediate_size", 0);
  m->n_shared = cfg_int_(cfg, "n_shared_experts", 0);
  m->shared_inter = m->n_shared > 0 ? m->n_shared * m->moe_inter : 0;
  m->first_k_dense = cfg_int_(cfg, "first_k_dense_replace", m->n_experts > 0 ? 1 : 0);
  m->moe_freq = cfg_int_(cfg, "moe_layer_freq", 1);
  if (m->moe_freq <= 0) m->moe_freq = 1;
  m->norm_topk = cfg_int_(cfg, "norm_topk_prob", 0);
  m->routed_scale = (float)cfg_num_(cfg, "routed_scaling_factor", 1.0);
  m->n_group = cfg_int_(cfg, "n_group", 0);
  m->topk_group = cfg_int_(cfg, "topk_group", 0);
  m->tie_embeddings = cfg_int_(cfg, "tie_word_embeddings", 0);
  {
    const char *sf = sgl_json_get_str(cfg, "scoring_func", "softmax");
    m->sigmoid_score = sf && !strcmp(sf, "sigmoid");
    const char *tm = sgl_json_get_str(cfg, "topk_method", "greedy");
    m->group_limited = (tm && !strcmp(tm, "group_limited_greedy") && m->n_group > 1 &&
                        m->topk_group > 0);
    const sgl_json *rs = sgl_json_get(cfg, "rope_scaling");
    if (!rs || rs->type != SGL_JSON_OBJECT) rs = sgl_json_get(cfg, "rope_parameters");
    if (rs && rs->type == SGL_JSON_OBJECT) {
      const char *rt = sgl_json_get_str(rs, "type", sgl_json_get_str(rs, "rope_type", "default"));
      double rfac = sgl_json_get_num(rs, "factor", 1.0);
      if (!strcmp(rt, "yarn")) {
        /* P1: YaRN（DeepSeek-V3 真实权重的 rope_scaling）：逆频率表 + mscale
         *（mscale/mscale_all_dim 乘到 cos/sin 上，同 HF）*/
        m->yarn = 1;
        m->yarn_factor = rfac;
        m->yarn_base = cfg_num_(cfg, "rope_theta", 10000.0);
        m->yarn_beta_fast = sgl_json_get_num(rs, "beta_fast", 32.0);
        m->yarn_beta_slow = sgl_json_get_num(rs, "beta_slow", 1.0);
        double orig = sgl_json_get_num(rs, "original_max_position_embeddings", 0.0);
        m->yarn_orig = orig > 0 ? orig : (double)m->n_ctx;
        /* DeepSeek 默认：factor 缺省时用 max_position_embeddings / original */
        if (sgl_json_get(rs, "factor") == NULL && m->yarn_orig > 0)
          m->yarn_factor = (double)m->n_ctx / m->yarn_orig;
        double ms = sgl_json_get_num(rs, "mscale", 0.0);
        double msa = sgl_json_get_num(rs, "mscale_all_dim", 0.0);
        /* DeepSeek 的 mscale 不是乘到 cos/sin 上（HF DeepseekV2Attention.scaling）：
         * softmax_scale *= mscale²，mscale = 0.1*mscale_all_dim*ln(factor)+1。
         * 注意 softmax_scale 在下面才初始化，所以这里先存系数、后面再乘。*/
        m->yarn_mscale_mul = 1.0f;
        if (msa > 0) {
          double mc = sgl_rope_yarn_get_mscale(m->yarn_factor, msa);
          m->yarn_mscale_mul = (float)(mc * mc);
        }
        m->rope_attn_factor = 1.0f; /* 不缩放 cos/sin（与 HF 一致）*/
        SGL_INFO("deepseek: YaRN rope (factor=%g orig=%g beta=%.0f/%.0f mscale=%g/%g) -> "
                 "softmax_scale x %.6f",
                 m->yarn_factor, m->yarn_orig, m->yarn_beta_fast, m->yarn_beta_slow, ms, msa,
                 (double)m->yarn_mscale_mul);
      } else if (!strcmp(rt, "default") || !strcmp(rt, "linear") || !strcmp(rt, "dynamic")) {
        SGL_WARN("deepseek: rope_scaling type=%s 未实现，按普通 RoPE 处理", rt);
      } else {
        SGL_WARN("deepseek: rope_scaling type=%s 未知，按普通 RoPE 处理", rt);
      }
    }
  }
  sgl_json_free(cfg);

  if (m->n_layer <= 0 || m->n_head <= 0 || m->n_embd <= 0 || m->vocab <= 0 ||
      m->kv_lora <= 0 || m->qk_nope <= 0 || m->qk_rope <= 0 || (m->qk_rope & 1) ||
      m->v_head <= 0 || m->q_lora < 0 || m->n_ctx <= 0) {
    SGL_ERROR("deepseek: bad config L=%d H=%d E=%d V=%d q_lora=%d kv_lora=%d nope=%d rope=%d v=%d",
              m->n_layer, m->n_head, m->n_embd, m->vocab, m->q_lora, m->kv_lora, m->qk_nope,
              m->qk_rope, m->v_head);
    sgl_xfree(m);
    return NULL;
  }
  if (m->n_experts > 0 &&
      (m->moe_inter <= 0 || m->n_topk <= 0 || m->n_topk > m->n_experts ||
       (m->group_limited && (m->n_group <= 0 || m->n_experts % m->n_group ||
                             m->topk_group <= 0 || m->topk_group > m->n_group)))) {
    SGL_ERROR("deepseek: MoE config invalid experts=%d topk=%d moe_inter=%d n_group=%d topk_group=%d",
              m->n_experts, m->n_topk, m->moe_inter, m->n_group, m->topk_group);
    sgl_xfree(m);
    return NULL;
  }
  if (m->n_topk > 64) {
    SGL_ERROR("deepseek: num_experts_per_tok=%d > 64（内部选择缓冲上限）", m->n_topk);
    sgl_xfree(m);
    return NULL;
  }
  m->q_head_dim = m->qk_nope + m->qk_rope;
  m->stride = m->kv_lora + m->qk_rope;
  m->softmax_scale = (float)(1.0 / sqrt((double)m->q_head_dim));
  /* P1: YaRN 的 mscale²（DeepSeek 约定，乘在注意力缩放上）*/
  if (m->yarn_mscale_mul > 0) m->softmax_scale *= m->yarn_mscale_mul;
  m->nthr = 1;
#ifdef _OPENMP
  m->nthr = omp_get_max_threads();
  if (m->nthr < 1) m->nthr = 1;
#endif
  m->n_score_rows = m->nthr;
  {
    int half = m->qk_rope / 2;
    m->invf = (double *)sgl_xmalloc(sizeof(double) * (size_t)(half > 0 ? half : 1));
    if (m->yarn) {
      /* P1: YaRN 逆频率表（sgl_rope_yarn_invf 与 HF 逐值一致，见 test_yarn）*/
      sgl_rope_yarn_invf(m->invf, m->qk_rope, m->yarn_base, m->yarn_factor, m->yarn_beta_fast,
                         m->yarn_beta_slow, m->yarn_orig, 1);
    } else {
      for (int i = 0; i < half; i++)
        m->invf[i] = pow(m->rope_theta, -(double)(2 * i) / (double)m->qk_rope);
    }
    m->rope_attn_factor = m->yarn ? m->rope_attn_factor : 1.0f;
  }

  SGL_INFO("DeepSeek config: L=%d H=%d E=%d V=%d q_lora=%d kv_lora=%d nope=%d rope=%d v_head=%d "
           "stride=%d softmax_scale=%.6f theta=%g eps=%g",
           m->n_layer, m->n_head, m->n_embd, m->vocab, m->q_lora, m->kv_lora, m->qk_nope,
           m->qk_rope, m->v_head, m->stride, (double)m->softmax_scale, m->rope_theta,
           (double)m->rms_eps);

  /* ---- 权重 ---- */
  m->st = sgl_st_open_dir(dir);
  if (!m->st) {
    SGL_ERROR("deepseek: no weights in %s", dir);
    sgl_deepseek_free(m);
    return NULL;
  }
  m->wte = get_2d_(m->st, "model.embed_tokens.weight", m->vocab, m->n_embd);
  m->final_norm = get_1d_(m->st, "model.norm.weight", m->n_embd);
  {
    int64_t sh[2] = {m->vocab, m->n_embd};
    sgl_tensor lm = sgl_st_get(m->st, "lm_head.weight", 2, sh);
    if (lm.data) m->lm_head = lm;
    else {
      if (!m->tie_embeddings) SGL_INFO("deepseek: lm_head missing, fallback to tied embeddings");
      m->lm_head = m->wte;
    }
  }
  if (!m->wte.data || !m->final_norm.data || !m->lm_head.data) {
    sgl_deepseek_free(m);
    return NULL;
  }

  int64_t E = m->n_embd;
  int64_t qdim = (int64_t)m->n_head * m->q_head_dim;
  int64_t kvb_rows = (int64_t)m->n_head * (m->qk_nope + m->v_head);
  m->layers = (sgl_deepseek_layer *)sgl_xcalloc((size_t)m->n_layer, sizeof(*m->layers));
  for (int l = 0; l < m->n_layer; l++) {
    char nm[600];
    sgl_deepseek_layer *L = &m->layers[l];
    snprintf(nm, sizeof(nm), "model.layers.%d.input_layernorm.weight", l);
    L->inp_norm = get_1d_(m->st, nm, E);
    snprintf(nm, sizeof(nm), "model.layers.%d.post_attention_layernorm.weight", l);
    L->post_norm = get_1d_(m->st, nm, E);
    if (!L->inp_norm.data || !L->post_norm.data) { sgl_deepseek_free(m); return NULL; }

    /* ---- MLA ---- */
    char ap[128];
    snprintf(ap, sizeof(ap), "model.layers.%d.self_attn.", l);
    if (m->q_lora > 0) {
      snprintf(nm, sizeof(nm), "%sq_a_proj.weight", ap);
      L->q_a_w = get_2d_(m->st, nm, m->q_lora, E);
      snprintf(nm, sizeof(nm), "%sq_a_layernorm.weight", ap);
      L->q_a_norm = get_1d_(m->st, nm, m->q_lora);
      snprintf(nm, sizeof(nm), "%sq_b_proj.weight", ap);
      L->q_b_w = get_2d_(m->st, nm, qdim, m->q_lora);
      snprintf(nm, sizeof(nm), "%sq_a_proj.bias", ap);
      L->q_a_b = get_opt_1d_(m->st, nm, m->q_lora);
      if (!L->q_a_w.data || !L->q_a_norm.data || !L->q_b_w.data) {
        sgl_deepseek_free(m);
        return NULL;
      }
    } else {
      snprintf(nm, sizeof(nm), "%sq_proj.weight", ap);
      L->q_b_w = get_2d_(m->st, nm, qdim, E);
      snprintf(nm, sizeof(nm), "%sq_proj.bias", ap);
      L->q_a_b = get_opt_1d_(m->st, nm, qdim);
      if (!L->q_b_w.data) { sgl_deepseek_free(m); return NULL; }
    }
    snprintf(nm, sizeof(nm), "%skv_a_proj_with_mqa.weight", ap);
    L->kv_a_w = get_2d_(m->st, nm, m->stride, E);
    snprintf(nm, sizeof(nm), "%skv_a_layernorm.weight", ap);
    L->kv_a_norm = get_1d_(m->st, nm, m->kv_lora);
    snprintf(nm, sizeof(nm), "%skv_b_proj.weight", ap);
    L->kv_b_w = get_2d_(m->st, nm, kvb_rows, m->kv_lora);
    snprintf(nm, sizeof(nm), "%so_proj.weight", ap);
    L->o_w = get_2d_(m->st, nm, E, (int64_t)m->n_head * m->v_head);
    snprintf(nm, sizeof(nm), "%skv_a_proj_with_mqa.bias", ap);
    L->kv_a_b = get_opt_1d_(m->st, nm, m->stride);
    snprintf(nm, sizeof(nm), "%so_proj.bias", ap);
    L->o_b = get_opt_1d_(m->st, nm, E);
    if (!L->kv_a_w.data || !L->kv_a_norm.data || !L->kv_b_w.data || !L->o_w.data) {
      sgl_deepseek_free(m);
      return NULL;
    }

    /* ---- MLP ---- */
    int is_moe = m->n_experts > 0 && l >= m->first_k_dense && (l % m->moe_freq) == 0;
    L->is_moe = is_moe;
    if (is_moe) {
      if (load_moe_(m, l, L) != 0) { sgl_deepseek_free(m); return NULL; }
      m->n_moe++;
    } else {
      snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate_proj.weight", l);
      L->gate_w = get_2d_(m->st, nm, m->n_inter, E);
      snprintf(nm, sizeof(nm), "model.layers.%d.mlp.up_proj.weight", l);
      L->up_w = get_2d_(m->st, nm, m->n_inter, E);
      snprintf(nm, sizeof(nm), "model.layers.%d.mlp.down_proj.weight", l);
      L->down_w = get_2d_(m->st, nm, E, m->n_inter);
      if (!L->gate_w.data || !L->up_w.data || !L->down_w.data) {
        sgl_deepseek_free(m);
        return NULL;
      }
    }
  }

  /* ---- scratch（每行宽：dense I / MoE gate_up 2*mi / 共享专家 sh 取最大）---- */
  size_t ctx = (size_t)m->n_ctx;
  int rw = m->n_inter;
  if (2 * m->moe_inter > rw) rw = 2 * m->moe_inter;
  if (m->shared_inter > rw) rw = m->shared_inter;
  if (rw <= 0) rw = 1;
  m->x = (float *)sgl_xmalloc(ctx * (size_t)E * sizeof(float));
  m->h = (float *)sgl_xmalloc(ctx * (size_t)E * sizeof(float));
  m->q = (float *)sgl_xmalloc(ctx * (size_t)qdim * sizeof(float));
  if (m->q_lora > 0)
    m->qa = (float *)sgl_xmalloc(ctx * (size_t)m->q_lora * sizeof(float));
  m->kva = (float *)sgl_xmalloc(ctx * (size_t)m->stride * sizeof(float));
  m->lat = (float *)sgl_xmalloc(ctx * (size_t)m->kv_lora * sizeof(float));
  m->att = (float *)sgl_xmalloc(ctx * (size_t)m->n_head * (size_t)m->v_head * sizeof(float));
  m->gu = (float *)sgl_xmalloc(ctx * (size_t)rw * sizeof(float));
  m->upd = (float *)sgl_xmalloc(ctx * (size_t)rw * sizeof(float));
  m->mlp_out = (float *)sgl_xmalloc(ctx * (size_t)E * sizeof(float));
  m->rt_out = (float *)sgl_xmalloc((size_t)E * sizeof(float));
  m->scores = (float *)sgl_xmalloc((size_t)m->n_score_rows * ctx * sizeof(float));
  m->tmp_abs = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->nthr *
                                    ((size_t)m->n_head * m->kv_lora + m->kv_lora));
  m->bpos = (int *)sgl_xmalloc(sizeof(int) * ctx);
  m->btabs = (int **)sgl_xmalloc(sizeof(int *) * ctx);
  if (m->n_experts > 0) {
    m->rt_sc = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->n_experts);
    m->rt_ch = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->n_experts);
    m->rt_sel = (int *)sgl_xmalloc(sizeof(int) * 64);
    m->rt_w = (float *)sgl_xmalloc(sizeof(float) * 64);
    m->rt_gs = (float *)sgl_xmalloc(sizeof(float) * (size_t)(m->n_group > 0 ? m->n_group : 1));
    m->rt_gsel = (int *)sgl_xmalloc(sizeof(int) * (size_t)(m->n_group > 0 ? m->n_group : 1));
  }

  /* ---- 分页 KV 池 ---- */
  m->n_pool = SGL_DEEPSEEK_POOL_SLOTS;
  {
    const char *env = getenv("SGL_POOL_SLOTS");
    if (env && atoi(env) > 0) m->n_pool = atoi(env);
  }
  m->kc = (float *)sgl_xmalloc((size_t)m->n_layer * m->n_pool * m->stride * sizeof(float));
  m->free_pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)m->n_pool);
  m->page_refs = (int *)sgl_xcalloc((size_t)m->n_pool, sizeof(int));
  for (int i = 0; i < m->n_pool; i++) m->free_pages[i] = m->n_pool - 1 - i;
  m->n_free = m->n_pool;

  SGL_INFO("DeepSeek loaded: %d layers (%d MoE, %d dense), MoE: experts=%d top-%d moe_inter=%d "
           "shared=%d(%d) score=%s topk=%s norm_topk=%d scale=%g; KV pool %d pages (%.1f MB)",
           m->n_layer, m->n_moe, m->n_layer - m->n_moe, m->n_experts, m->n_topk, m->moe_inter,
           m->n_shared, m->shared_inter, m->sigmoid_score ? "sigmoid" : "softmax",
           m->group_limited ? "group_limited_greedy" : "greedy", m->norm_topk,
           (double)m->routed_scale, m->n_pool,
           ((double)m->n_layer * m->n_pool * m->stride * sizeof(float)) / 1048576.0);
  return m;
}

void sgl_deepseek_free(sgl_deepseek *m) {
  if (!m) return;
  for (int l = 0; l < m->n_layer && m->layers; l++) {
    sgl_deepseek_layer *L = &m->layers[l];
    sgl_xfree(L->ec_bias);
    if (L->exp_gate) sgl_xfree(L->exp_gate);
    if (L->exp_up) sgl_xfree(L->exp_up);
    if (L->exp_down) sgl_xfree(L->exp_down);
  }
  sgl_xfree(m->layers);
  sgl_xfree(m->invf);
  sgl_xfree(m->x); sgl_xfree(m->h); sgl_xfree(m->qa); sgl_xfree(m->q);
  sgl_xfree(m->kva); sgl_xfree(m->lat); sgl_xfree(m->att);
  sgl_xfree(m->gu); sgl_xfree(m->upd); sgl_xfree(m->mlp_out); sgl_xfree(m->rt_out);
  sgl_xfree(m->scores); sgl_xfree(m->tmp_abs);
  sgl_xfree(m->rt_sc); sgl_xfree(m->rt_ch); sgl_xfree(m->rt_gs); sgl_xfree(m->rt_gsel);
  sgl_xfree(m->rt_sel); sgl_xfree(m->rt_w);
  sgl_xfree(m->bpos); sgl_xfree(m->btabs);
  sgl_xfree(m->kc); sgl_xfree(m->free_pages); sgl_xfree(m->page_refs);
  if (m->st) sgl_st_close(m->st);
  sgl_xfree(m);
}

int sgl_deepseek_vocab(const sgl_deepseek *m) { return m ? m->vocab : 0; }
int sgl_deepseek_ctx(const sgl_deepseek *m) { return m ? m->n_ctx : 0; }
int sgl_deepseek_kv_stride(const sgl_deepseek *m) { return m ? m->stride : 0; }
int sgl_deepseek_heads(const sgl_deepseek *m) { return m ? m->n_head : 0; }
int sgl_deepseek_hidden(const sgl_deepseek *m) { return m ? m->n_embd : 0; }

/* ---------------- 分页 KV 池（语义同 llama：LIFO 空闲栈 + 引用计数）---------------- */

int sgl_deepseek_pages_alloc(sgl_deepseek *m, int *idx, int n) {
  if (!m || n <= 0) return 0;
  if (n > m->n_free) return -1;
  m->n_free -= n;
  for (int i = 0; i < n; i++) {
    idx[i] = m->free_pages[m->n_free + i];
    m->page_refs[idx[i]] = 1;
  }
  return 0;
}

void sgl_deepseek_page_ref(sgl_deepseek *m, int page, int delta) {
  if (!m || page < 0 || page >= m->n_pool) return;
  m->page_refs[page] += delta;
  if (m->page_refs[page] <= 0) {
    m->page_refs[page] = 0;
    m->free_pages[m->n_free++] = page;
  }
}

void sgl_deepseek_pages_unref(sgl_deepseek *m, const int *idx, int n) {
  for (int i = 0; i < n; i++) sgl_deepseek_page_ref(m, idx[i], -1);
}

int sgl_deepseek_nfree(const sgl_deepseek *m) { return m ? m->n_free : 0; }
int sgl_deepseek_pool_pages(const sgl_deepseek *m) { return m ? m->n_pool : 0; }

int sgl_deepseek_page_refs(const sgl_deepseek *m, int page) {
  if (!m || page < 0 || page >= m->n_pool) return 0;
  return m->page_refs[page];
}

/* ---------------- RoPE ---------------- */

/* 单段 RoPE（长度 dim = qk_rope_head_dim、同一位置 pos）。
 * 【交错】配对 (2i, 2i+1)：out[2i] = x[2i]*c - x[2i+1]*s，out[2i+1] = x[2i+1]*c + x[2i]*s；
 * 与 HF DeepseekV2 的 apply_rotary_emb（view_as_complex 复乘）一致。
 * rope_interleave = 0 时改走非交错 rotate_half（配对 (i, i+dim/2)）——仅为旧测试模型兼容。*/
static void ds_rope_seg_(const sgl_deepseek *m, float *x, int pos) {
  int half = m->qk_rope / 2;
  /* P1: YaRN 的 attention_factor 乘在 cos/sin 上（HF 同样做法），
   * 对旋转结果等价于整体乘 s（s * (x*cos + rot(x)*sin)）*/
  const float s_scale = m->yarn ? m->rope_attn_factor : 1.0f;
  if (!m->rope_interleave) {
    for (int i = 0; i < half; i++) {
      double ang = (double)pos * m->invf[i];
      float c = (float)cos(ang) * s_scale, s = (float)sin(ang) * s_scale;
      float a = x[i], b = x[i + half];
      x[i] = a * c - b * s;
      x[i + half] = a * s + b * c;
    }
  } else {
    for (int i = 0; i < half; i++) {
      double ang = (double)pos * m->invf[i];
      float c = (float)cos(ang) * s_scale, s = (float)sin(ang) * s_scale;
      float a = x[2 * i], b = x[2 * i + 1];
      x[2 * i] = a * c - b * s;
      x[2 * i + 1] = a * s + b * c;
    }
  }
}

/* ---------------- MLP / MoE ---------------- */

/* MoE 单行（HF DeepseekV2MoE.forward + DeepseekV2TopkRouter/DeepseekV3TopkRouter）：
 *   1) router 打分：softmax（V2）或 sigmoid（V3）；
 *   2) 选择用分数 = 打分 + e_score_correction_bias（有的话）；
 *   3) topk_method=group_limited_greedy（V3）：每 group 取 top-2 求和当组分数，
 *      选 topk_group 个组，组外置 -inf；
 *   4) 在（可能被组过滤的）选择分数上取 top-k；
 *   5) 权重取【未修正】的打分值（HF scores.gather(1, topk_indices)，同 DeepSeek 官方
 *      inference），norm_topk_prob 时归一化（分母 +1e-20），再乘 routed_scaling_factor；
 *   6) 专家 gate/up/down 按权重累加，再加共享专家（宽度 sh，无门控）。
 * 公式照 HF，随机模型 numpy 参考同式（见 tests/gen_random_deepseek.py）。*/
static int ds_moe_row_(sgl_deepseek *m, sgl_deepseek_layer *L, const float *hrow, float *orow) {
  int E = m->n_embd, NE = m->n_experts, mi = m->moe_inter, sh = m->shared_inter;
  int k = m->n_topk;
  float *s = m->rt_sc;  /* 打分（未修正）*/
  float *ch = m->rt_ch; /* 选择用（+bias，组过滤后组外 -inf）*/
  sgl_matmul_nt(hrow, L->router_w.data, NULL, s, 1, NE, E);
  if (m->sigmoid_score) {
    for (int i = 0; i < NE; i++) s[i] = 1.0f / (1.0f + expf(-s[i]));
  } else {
    sgl_softmax_inplace(s, NE);
  }
  for (int i = 0; i < NE; i++) ch[i] = s[i] + (L->ec_bias ? L->ec_bias[i] : 0.0f);
  if (m->group_limited) {
    int ng = m->n_group, epg = NE / ng;
    float *gs = m->rt_gs;
    int *gsel = m->rt_gsel;
    for (int g = 0; g < ng; g++) {
      const float *base = ch + (size_t)g * epg;
      float t1 = -1e30f, t2 = -1e30f;
      for (int i = 0; i < epg; i++) {
        float v = base[i];
        if (v > t1) { t2 = t1; t1 = v; }
        else if (v > t2) { t2 = v; }
      }
      gs[g] = (epg > 1 ? t1 + t2 : t1);
    }
    for (int g = 0; g < ng; g++) {
      int best = -1;
      float bv = -1e30f;
      for (int i = 0; i < ng; i++) {
        int used = 0;
        for (int j = 0; j < g; j++) if (gsel[j] == i) used = 1;
        if (!used && gs[i] > bv) { bv = gs[i]; best = i; }
      }
      gsel[g] = best;
    }
    for (int i = 0; i < NE; i++) {
      int g = i / epg, keep = 0;
      for (int j = 0; j < m->topk_group; j++) if (gsel[j] == g) keep = 1;
      if (!keep) ch[i] = -1e30f;
    }
  }
  /* top-k（NE 小，直接选；并列时取小号，与 torch.topk 的稳定行为一致）*/
  float *sw = m->rt_w;
  int *sel = m->rt_sel;
  for (int j = 0; j < k; j++) {
    int best = -1;
    float bv = -1e30f;
    for (int i = 0; i < NE; i++) {
      int used = 0;
      for (int q = 0; q < j; q++) if (sel[q] == i) used = 1;
      if (!used && ch[i] > bv) { bv = ch[i]; best = i; }
    }
    sel[j] = best;
    /* 权重取未修正打分（HF：topk_weights = scores.gather(1, topk_indices)）*/
    sw[j] = best >= 0 ? s[best] : 0.0f;
  }
  if (k > 1 && m->norm_topk) {
    double t = 0;
    for (int j = 0; j < k; j++) t += sw[j];
    t += 1e-20; /* HF: denominator = sum + 1e-20 */
    if (t != 0)
      for (int j = 0; j < k; j++) sw[j] = (float)(sw[j] / t);
  }
  for (int j = 0; j < k; j++) sw[j] *= m->routed_scale;

  memset(orow, 0, sizeof(float) * (size_t)E);
  for (int j = 0; j < k; j++) {
    int e = sel[j];
    if (e < 0) continue;
    if (L->exp_stacked) {
      sgl_matmul_nt(hrow, L->exp_gu3.data + (size_t)e * 2 * mi * E, NULL, m->gu, 1, 2 * mi, E);
      for (int i = 0; i < mi; i++) {
        float g = m->gu[i];
        m->gu[i] = (g / (1.0f + expf(-g))) * m->gu[mi + i];
      }
      sgl_matmul_nt(m->gu, L->exp_dn3.data + (size_t)e * E * mi, NULL, m->rt_out, 1, E, mi);
    } else {
      sgl_matmul_nt(hrow, L->exp_gate[e].data, NULL, m->gu, 1, mi, E);
      sgl_matmul_nt(hrow, L->exp_up[e].data, NULL, m->upd, 1, mi, E);
      for (int i = 0; i < mi; i++) {
        float g = m->gu[i];
        m->gu[i] = (g / (1.0f + expf(-g))) * m->upd[i];
      }
      sgl_matmul_nt(m->gu, L->exp_down[e].data, NULL, m->rt_out, 1, E, mi);
    }
    float w = sw[j];
    for (int i = 0; i < E; i++) orow[i] += w * m->rt_out[i];
  }
  /* 共享专家：直接相加（DeepSeek 无 sigmoid 门控，与 qwen3-next 不同）*/
  if (sh > 0 && L->sh_gate_w.data) {
    sgl_matmul_nt(hrow, L->sh_gate_w.data, NULL, m->gu, 1, sh, E);
    sgl_matmul_nt(hrow, L->sh_up_w.data, NULL, m->upd, 1, sh, E);
    for (int i = 0; i < sh; i++) {
      float g = m->gu[i];
      m->gu[i] = (g / (1.0f + expf(-g))) * m->upd[i];
    }
    sgl_matmul_nt(m->gu, L->sh_down_w.data, NULL, m->rt_out, 1, E, sh);
    for (int i = 0; i < E; i++) orow[i] += m->rt_out[i];
  }
  return 0;
}

/* MLP 块（行级）：dense SwiGLU 或 MoE，结果写 out（[rows][E]）*/
static int ds_mlp_rows_(sgl_deepseek *m, sgl_deepseek_layer *L, const float *h, int rows,
                        float *out) {
  int E = m->n_embd;
  if (L->is_moe) {
    for (int t = 0; t < rows; t++)
      if (ds_moe_row_(m, L, h + (size_t)t * E, out + (size_t)t * E) != 0) return -1;
    return 0;
  }
  int I = m->n_inter;
  sgl_matmul_nt(h, L->gate_w.data, NULL, m->gu, rows, I, E);
  sgl_matmul_nt(h, L->up_w.data, NULL, m->upd, rows, I, E);
  sgl_silu_inplace(m->gu, (size_t)rows * I);
  for (size_t i = 0; i < (size_t)rows * I; i++) m->gu[i] *= m->upd[i];
  sgl_matmul_nt(m->gu, L->down_w.data, NULL, out, rows, E, I);
  return 0;
}

/* ---------------- 前向 ---------------- */

static void ds_logits_(const sgl_deepseek *m, const float *hrow, float *out) {
  sgl_matmul_nt(hrow, m->lm_head.data, NULL, out, 1, m->vocab, m->n_embd);
}

/* 批量描述：第 t 行的绝对位置 b->pos[t]（causal 可见 0..pos[t]），页表 b->tabs[t]。
 * prefill 时 n 行共用一个页表（tabs 全指向同一个 idx），decode_batch 时每行各自一份，
 * 所以两条路径共用同一个实现（与 llama 的两个函数等价）。*/
typedef struct {
  const int *tokens;   /* [n]（emb 模式为 NULL）*/
  const float *emb;    /* [n][E]（token 模式为 NULL）*/
  int n;
  const int *pos;      /* [n] 绝对位置 */
  int *const *tabs;    /* [n] 页表指针 */
  int logits_rows;     /* 0 = 只末位；n = 全部 n 行 */
  float *logits;       /* 见上；可 NULL */
} ds_batch;

static int ds_forward_impl_(sgl_deepseek *m, const ds_batch *b) {
  int E = m->n_embd, H = m->n_head, Dn = m->qk_nope, Dr = m->qk_rope, Dc = m->kv_lora;
  int Dv = m->v_head, qhd = m->q_head_dim, stride = m->stride, n = b->n;
  if (n <= 0 || n > m->n_ctx) {
    SGL_ERROR("deepseek forward: n=%d 超出 ctx=%d", n, m->n_ctx);
    return -1;
  }
  for (int t = 0; t < n; t++) {
    if (b->pos[t] < 0 || b->pos[t] >= m->n_ctx) {
      SGL_ERROR("deepseek forward: pos=%d 超出 ctx=%d", b->pos[t], m->n_ctx);
      return -1;
    }
    /* 页表检查：prefill 时各行共用一个表，指针相同只查一次 */
    int same = (t > 0 && b->tabs[t] == b->tabs[t - 1]);
    for (int p = same ? b->pos[t] : 0; p <= b->pos[t]; p++) {
      int pg = b->tabs[t][p];
      if (pg < 0 || pg >= m->n_pool) {
        SGL_ERROR("deepseek forward: page %d out of pool at pos %d", pg, p);
        return -1;
      }
    }
  }

  if (b->emb)
    memcpy(m->x, b->emb, sizeof(float) * (size_t)n * E);
  else
    sgl_embedding_get(m->wte.data, b->tokens, n, E, m->x);

  const int per_head = (Dn + Dv) * Dc;
  int trace = getenv("SGL_DS_TRACE") != NULL;
  for (int l = 0; l < m->n_layer; l++) {
    sgl_deepseek_layer *L = &m->layers[l];
    /* ---- MLA ---- */
    sgl_rmsnorm(m->x, L->inp_norm.data, m->h, n, E, m->rms_eps);
    if (m->q_lora > 0) {
      sgl_matmul_nt(m->h, L->q_a_w.data, L->q_a_b.data, m->qa, n, m->q_lora, E);
      sgl_rmsnorm(m->qa, L->q_a_norm.data, m->qa, n, m->q_lora, m->rms_eps);
      sgl_matmul_nt(m->qa, L->q_b_w.data, NULL, m->q, n, H * qhd, m->q_lora);
    } else {
      sgl_matmul_nt(m->h, L->q_b_w.data, L->q_a_b.data, m->q, n, H * qhd, E);
    }
    sgl_matmul_nt(m->h, L->kv_a_w.data, L->kv_a_b.data, m->kva, n, stride, E);

    /* 逐 token：q_pe（每头尾部 Dr 维）与 k_pe（kv_a 尾部）过 RoPE；抽 latent 行 */
    for (int t = 0; t < n; t++) {
      int p = b->pos[t];
      float *qrow = m->q + (size_t)t * H * qhd;
      for (int h = 0; h < H; h++) ds_rope_seg_(m, qrow + (size_t)h * qhd + Dn, p);
      ds_rope_seg_(m, m->kva + (size_t)t * stride + Dc, p);
      memcpy(m->lat + (size_t)t * Dc, m->kva + (size_t)t * stride, sizeof(float) * Dc);
    }
    sgl_rmsnorm(m->lat, L->kv_a_norm.data, m->lat, n, Dc, m->rms_eps);

    /* 写压缩 KV：行 = [latent | k_pe]，按页表落到页 slot_idx[pos]（前缀共享靠页引用）*/
    for (int t = 0; t < n; t++) {
      int pg = b->tabs[t][b->pos[t]];
      float *row = m->kc + (size_t)l * m->n_pool * stride + (size_t)pg * stride;
      memcpy(row, m->lat + (size_t)t * Dc, sizeof(float) * Dc);
      memcpy(row + Dc, m->kva + (size_t)t * stride + Dc, sizeof(float) * Dr);
    }

    /* 吸收式 causal attention：q_abs[h] = W_UK[h]·q_nope[h]（[Dc]）；
     * score = q_abs·latent + q_pe·k_pe；o_lat = Σ a·latent；o[h] = W_UV[h]·o_lat。*/
    {
      const float *kvb = L->kv_b_w.data;
#pragma omp parallel for
      for (int t = 0; t < n; t++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float *sc = m->scores + (size_t)tid * m->n_ctx;
        float *qabs = m->tmp_abs + (size_t)tid * ((size_t)H * Dc + Dc);
        float *olat = qabs + (size_t)H * Dc;
        const float *qrow = m->q + (size_t)t * H * qhd;
        const int np = b->pos[t] + 1;
        const int *tab = b->tabs[t];
        const float *kcl = m->kc + (size_t)l * m->n_pool * stride;
        for (int h = 0; h < H; h++) {
          const float *q_nope = qrow + (size_t)h * qhd;
          const float *q_pe = q_nope + Dn;
          const float *uk = kvb + (size_t)h * per_head; /* [Dn][Dc] */
          const float *uv = uk + (size_t)Dn * Dc;       /* [Dv][Dc] */
          float *qa_h = qabs + (size_t)h * Dc;
          sgl_matmul(q_nope, uk, NULL, qa_h, 1, Dc, Dn);
          float maxv = -1e30f;
          for (int j = 0; j < np; j++) {
            const float *row = kcl + (size_t)tab[j] * stride;
            float s1 = 0, s2 = 0;
            sgl_dot_f32(qa_h, row, Dc, &s1);
            sgl_dot_f32(q_pe, row + Dc, Dr, &s2);
            float sv = (s1 + s2) * m->softmax_scale;
            sc[j] = sv;
            if (sv > maxv) maxv = sv;
          }
          double sum = 0;
          for (int j = 0; j < np; j++) { sc[j] = expf(sc[j] - maxv); sum += sc[j]; }
          float inv = (float)(1.0 / sum);
          memset(olat, 0, sizeof(float) * (size_t)Dc);
          for (int j = 0; j < np; j++) {
            float a = sc[j] * inv;
            if (a == 0.0f) continue;
            const float *row = kcl + (size_t)tab[j] * stride;
            for (int d = 0; d < Dc; d++) olat[d] += a * row[d];
          }
          sgl_matmul_nt(olat, uv, NULL, m->att + (size_t)t * H * Dv + (size_t)h * Dv, 1, Dv, Dc);
        }
      }
    }
    sgl_matmul_nt(m->att, L->o_w.data, L->o_b.data, m->h, n, E, H * Dv);
    for (int t = 0; t < n; t++)
      for (int k = 0; k < E; k++) m->x[(size_t)t * E + k] += m->h[(size_t)t * E + k];
    if (trace) {
      double ss = 0;
      for (int k = 0; k < E; k++) ss += (double)m->x[k] * m->x[k];
      fprintf(stderr, "ds l=%d attn: x0=%.7f x1=%.7f rms=%.7f\n", l, (double)m->x[0],
              (double)m->x[1], sqrt(ss / E));
    }

    /* ---- MLP（dense / MoE）---- */
    sgl_rmsnorm(m->x, L->post_norm.data, m->h, n, E, m->rms_eps);
    if (ds_mlp_rows_(m, L, m->h, n, m->mlp_out) != 0) return -1;
    for (int t = 0; t < n; t++)
      for (int k = 0; k < E; k++) m->x[(size_t)t * E + k] += m->mlp_out[(size_t)t * E + k];
    if (trace) {
      double ss = 0;
      for (int k = 0; k < E; k++) ss += (double)m->x[k] * m->x[k];
      fprintf(stderr, "ds l=%d mlp : x0=%.7f x1=%.7f rms=%.7f\n", l, (double)m->x[0],
              (double)m->x[1], sqrt(ss / E));
    }
  }

  sgl_rmsnorm(m->x, m->final_norm.data, m->h, n, E, m->rms_eps);
  if (b->logits) {
    if (b->logits_rows == n) {
      for (int t = 0; t < n; t++)
        ds_logits_(m, m->h + (size_t)t * E, b->logits + (size_t)t * m->vocab);
    } else {
      ds_logits_(m, m->h + (size_t)(n - 1) * E, b->logits);
    }
  }
  return 0;
}

/* prefill：n 行连续位置 [pos0, pos0+n)，共用页表 idx */
static int ds_prefill_common_(sgl_deepseek *m, const int *tokens, const float *emb, int n,
                              int pos0, const int *idx, float *logits, int logits_rows) {
  if (!m || n <= 0) return -1;
  if (pos0 < 0 || pos0 + n > m->n_ctx) {
    SGL_ERROR("deepseek prefill: pos0=%d n=%d 超出 ctx=%d", pos0, n, m->n_ctx);
    return -1;
  }
  for (int t = 0; t < n; t++) {
    m->bpos[t] = pos0 + t;
    m->btabs[t] = (int *)idx;
  }
  ds_batch b;
  b.tokens = tokens;
  b.emb = emb;
  b.n = n;
  b.pos = m->bpos;
  b.tabs = m->btabs;
  b.logits_rows = logits_rows;
  b.logits = logits;
  return ds_forward_impl_(m, &b);
}

int sgl_deepseek_prefill(sgl_deepseek *m, const int *tokens, int n, int pos0, const int *idx,
                         float *logits_out) {
  return ds_prefill_common_(m, tokens, NULL, n, pos0, idx, logits_out, 0);
}

int sgl_deepseek_forward(sgl_deepseek *m, const int *tokens, int n, int pos0, const int *idx,
                         float *logits_out) {
  return ds_prefill_common_(m, tokens, NULL, n, pos0, idx, logits_out, 0);
}

int sgl_deepseek_forward_n(sgl_deepseek *m, const int *tokens, int n, int pos0, const int *idx,
                           float *logits_all) {
  return ds_prefill_common_(m, tokens, NULL, n, pos0, idx, logits_all, logits_all ? n : 0);
}

int sgl_deepseek_prefill_emb(sgl_deepseek *m, const float *emb, int n, int pos0, const int *idx,
                             float *logits_out) {
  if (!emb) return -1;
  return ds_prefill_common_(m, NULL, emb, n, pos0, idx, logits_out, 0);
}

int sgl_deepseek_decode_batch(sgl_deepseek *m, const int *tokens, int *const *idx,
                              const int *seq_lens, int B, float *logits_out) {
  if (!m || B <= 0 || B > 64) {
    SGL_ERROR("deepseek decode_batch: B=%d", B);
    return -1;
  }
  for (int b = 0; b < B; b++) {
    if (seq_lens[b] < 0 || seq_lens[b] + 1 > m->n_ctx) {
      SGL_ERROR("deepseek decode: req %d pos>ctx", b);
      return -1;
    }
    m->bpos[b] = seq_lens[b];
    m->btabs[b] = idx[b];
  }
  ds_batch bt;
  bt.tokens = tokens;
  bt.emb = NULL;
  bt.n = B;
  bt.pos = m->bpos;
  bt.tabs = m->btabs;
  bt.logits_rows = B;
  bt.logits = logits_out;
  return ds_forward_impl_(m, &bt);
}