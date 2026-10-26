/* sgl clip.c — CLIP ViT vision encoder (t10 LLaVA 第一步; G8 真实权重对齐)
 * 对应 HF transformers modeling_clip.py 的 CLIPVisionModel（sglang srt/models/llava.py
 * 的视觉塔同源）：
 *   conv patch embed（等效 GEMM：patch [3*P*P] × W^T）+ class emb + learnable pos emb
 *   + pre_layrnorm（带仿射）+ N 层 CLIPEncoderLayer（双向 attention + hidden_act MLP）
 *   → last_hidden_state（不过 post_layernorm）；pooled = post_layernorm(cls 行)；
 *     image_embeds = visual_projection(pooled)。
 * LLaVA 取 last_hidden_state[:, 1:, :] 作视觉 token。
 * 与 t10 旧实现的三处结构差异（G8 修正）：
 *   1) MLP 激活：真实 CLIP 是 quick_gelu（x·sigmoid(1.702x)），不是 erf gelu；
 *   2) post_layernorm 只作用于 class 行（pooled），patch 行不过 LN；
 *   3) visual_projection 只作用于 pooled，不作用于 patch 行。
 * 计算全程 fp32；数值与 torch 真实模型分层对照（test_clip_real）、
 * 与 numpy 参考逐位对齐（test_llava）。 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/clip.h"

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

static void ln_f32(const float *x, const float *w, const float *b, float *y, int E, float eps) {
  double mean = 0, var = 0;
  for (int i = 0; i < E; i++) mean += x[i];
  mean /= E;
  for (int i = 0; i < E; i++) {
    double d = x[i] - mean;
    var += d * d;
  }
  var /= E;
  float inv = (float)(1.0 / sqrt(var + eps));
  for (int i = 0; i < E; i++) y[i] = (float)((x[i] - mean) * inv) * w[i] + b[i];
}

/* 三平面布局：q/k/v 各自 [n][E] 连续（与 sgl_matmul_nt 的输出一致） */
static void attention_rows(const float *q, const float *k, const float *v,
                           float *out, int n, int E, int H) {
  int D = E / H;
  float *sc = (float *)sgl_xmalloc(sizeof(float) * (size_t)n);
  for (int t = 0; t < n; t++) {
    const float *qt = q + (size_t)t * E;
    float *orow = out + (size_t)t * E;
    for (int h = 0; h < H; h++) {
      const float *qh = qt + h * D;
      float maxv = -1e30f;
      for (int p = 0; p < n; p++) {
        const float *kh = k + (size_t)p * E + h * D;
        float s = 0;
        for (int d = 0; d < D; d++) s += qh[d] * kh[d];
        s *= 1.0f / sqrtf((float)D);
        sc[p] = s;
        if (s > maxv) maxv = s;
      }
      double sum = 0;
      for (int p = 0; p < n; p++) { sc[p] = expf(sc[p] - maxv); sum += sc[p]; }
      float *o = orow + h * D;
      memset(o, 0, sizeof(float) * D);
      for (int p = 0; p < n; p++) {
        const float *vh = v + (size_t)p * E + h * D;
        float w = sc[p] / (float)sum;
        for (int d = 0; d < D; d++) o[d] += w * vh[d];
      }
    }
  }
  sgl_xfree(sc);
}

static float sgl_clip_act(float u, int kind) {
  switch (kind) {
  case SGL_CLIP_ACT_GELU_ERF:
    return 0.5f * u * (1.0f + erff(u * 0.7071067811865476f));
  case SGL_CLIP_ACT_GELU_TANH: {
    double x = u;
    return (float)(0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))));
  }
  case SGL_CLIP_ACT_QUICK:
  default:
    return u / (1.0f + expf(-1.702f * u)); /* quick_gelu */
  }
}

/* 读视觉塔 config：优先 <dir>/clip_config.json（扁平，随机小模型），
 * 没有则 <dir>/config.json 里的 vision_config（真实 HF CLIP）。
 * 返回可直接取字段的对象（扁平或 vision_config 子对象），*root_out 是整棵树，
 * 调用方用 sgl_json_free(root_out) 释放。 */
static sgl_json *clip_read_cfg(const char *dir, sgl_json **root_out, int *flat_out) {
  char path[1024];
  FILE *f = NULL;
  snprintf(path, sizeof(path), "%s/clip_config.json", dir);
  fopen_s(&f, path, "rb");
  if (f) *flat_out = 1;
  if (!f) {
    snprintf(path, sizeof(path), "%s/config.json", dir);
    fopen_s(&f, path, "rb");
    *flat_out = 0;
  }
  if (!f) { SGL_ERROR("clip_config.json / config.json missing in %s", dir); return NULL; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); sgl_xfree(buf); return NULL; }
  buf[sz] = 0;
  fclose(f);
  sgl_json *j = sgl_json_parse(buf, (size_t)sz);
  sgl_xfree(buf);
  if (!j) return NULL;
  *root_out = j;
  const sgl_json *vc = sgl_json_get(j, "vision_config");
  if (vc && vc->type == SGL_JSON_OBJECT) return (sgl_json *)vc;
  return j;
}

/* 可选张量：名字不存在时不刷 ERROR（真实 CLIP 没有 patch bias；位置/pre_ln 名有两套写法）*/
static int st_has(sgl_st_ctx *st, const char *name) {
  const char **ns = NULL;
  int n = sgl_st_names(st, &ns);
  for (int i = 0; i < n; i++)
    if (ns[i] && !strcmp(ns[i], name)) return 1;
  return 0;
}
static sgl_tensor st_opt(sgl_st_ctx *st, const char *name, int ndim, const int64_t *shape) {
  if (!st_has(st, name)) return sgl_tensor_null();
  return sgl_st_get(st, name, ndim, shape);
}

sgl_clip *sgl_clip_load(const char *dir) {
  char path[1024];
  sgl_json *root = NULL;
  int flat = 1; /* 用了 clip_config.json（随机小模型布局）*/
  sgl_json *cfg = clip_read_cfg(dir, &root, &flat);
  if (!cfg) return NULL;

  sgl_clip *c = (sgl_clip *)sgl_xcalloc(1, sizeof(*c));
  c->E = sgl_json_get_int(cfg, "hidden_size", 768);
  c->H = sgl_json_get_int(cfg, "num_heads", sgl_json_get_int(cfg, "num_attention_heads", 12));
  c->L = sgl_json_get_int(cfg, "num_hidden_layers", 12);
  c->P = sgl_json_get_int(cfg, "patch_size", 32);
  c->img = sgl_json_get_int(cfg, "image_size", 336);
  c->I = sgl_json_get_int(cfg, "intermediate_size", 4 * c->E);
  c->eps = (float)sgl_json_get_num(cfg, "layer_norm_eps", 1e-5);
  /* projection_dim：真实 CLIP 是 pooled→image_embeds 的维（不影响 encode 宽）*/
  c->img_proj_dim = sgl_json_get_int(cfg, "projection_dim", 0);
  c->proj_dim = c->E; /* encode 每行宽 = hidden_size（字段保留给已有调用方算 buffer）*/
  {
    const char *act = sgl_json_get_str(cfg, "hidden_act", "");
    if (!act[0]) act = sgl_json_get_str(cfg, "activation_function", "quick_gelu");
    if (!strcmp(act, "quick_gelu")) c->act = SGL_CLIP_ACT_QUICK;
    else if (!strcmp(act, "gelu")) c->act = SGL_CLIP_ACT_GELU_ERF;
    else if (!strcmp(act, "gelu_new") || !strcmp(act, "gelu_python")) c->act = SGL_CLIP_ACT_GELU_TANH;
    else { SGL_ERROR("clip hidden_act '%s' unsupported", act); sgl_xfree(c); sgl_json_free(root); return NULL; }
  }
  c->n_patch = (c->img / c->P) * (c->img / c->P);
  sgl_json_free(root);
  if (c->E <= 0 || c->H <= 0 || c->L <= 0 || c->P <= 0 || c->img % c->P != 0 || c->E % c->H) {
    SGL_ERROR("clip config invalid");
    sgl_xfree(c);
    return NULL;
  }

  /* 权重文件：随机小模型用 clip_model.safetensors，HF 仓库用 model.safetensors
   * （按 config 来源先试对应那个，另一个当回退，避免刷 open failed 日志）*/
  {
    const char *first = flat ? "clip_model.safetensors" : "model.safetensors";
    const char *second = flat ? "model.safetensors" : "clip_model.safetensors";
    snprintf(path, sizeof(path), "%s/%s", dir, first);
    c->st = sgl_st_open(path);
    if (!c->st) {
      snprintf(path, sizeof(path), "%s/%s", dir, second);
      c->st = sgl_st_open(path);
    }
  }
  if (!c->st) { SGL_ERROR("clip safetensors open failed in %s", dir); sgl_xfree(c); return NULL; }

  int E = c->E, NP = c->n_patch;
  c->cls_emb = sgl_st_get(c->st, "vision_model.embeddings.class_embedding", 1, (int64_t[]){E});
  /* HF: position_embedding.weight；旧随机布局：position_embedding（无 .weight）*/
  c->pos_emb = st_opt(c->st, "vision_model.embeddings.position_embedding.weight",
                      2, (int64_t[]){NP + 1, E});
  if (!c->pos_emb.data)
    c->pos_emb = st_opt(c->st, "vision_model.embeddings.position_embedding",
                        2, (int64_t[]){NP + 1, E});
  {
    /* conv weight [E,3,P,P] row-major flattens to [E,3*P*P]; take it unvalidated
     * (st_get expects exact ndim) then check numel by hand. */
    c->patch_w = sgl_st_get(c->st, "vision_model.embeddings.patch_embedding.weight", -1, NULL);
    if (c->patch_w.data &&
        sgl_tensor_numel(&c->patch_w) != (int64_t)E * 3 * c->P * c->P)
      c->patch_w = sgl_tensor_null();
  }
  /* 真实 CLIP 的 patch_embedding 是 bias=False（checkpoint 里没这个张量），
   * 缺省当零处理；旧随机模型有 bias，照加。 */
  c->patch_b = st_opt(c->st, "vision_model.embeddings.patch_embedding.bias", 1, (int64_t[]){E});
  /* pre_ln：HF 名字 vision_model.pre_layrnorm.[wb]；旧随机名字 vision_model.embeddings.pre_layernorm.[wb] */
  c->pre_ln_w = st_opt(c->st, "vision_model.pre_layrnorm.weight", 1, (int64_t[]){E});
  c->pre_ln_b = st_opt(c->st, "vision_model.pre_layrnorm.bias", 1, (int64_t[]){E});
  if (!c->pre_ln_w.data) {
    c->pre_ln_w = st_opt(c->st, "vision_model.embeddings.pre_layernorm.weight", 1, (int64_t[]){E});
    c->pre_ln_b = st_opt(c->st, "vision_model.embeddings.pre_layernorm.bias", 1, (int64_t[]){E});
  }
  c->post_ln_w = sgl_st_get(c->st, "vision_model.post_layernorm.weight", 1, (int64_t[]){E});
  c->post_ln_b = st_opt(c->st, "vision_model.post_layernorm.bias", 1, (int64_t[]){E});
  if (c->img_proj_dim > 0)
    c->vproj = st_opt(c->st, "visual_projection.weight",
                      2, (int64_t[]){c->img_proj_dim, E});
  if (!c->cls_emb.data || !c->pos_emb.data || !c->patch_w.data ||
      !c->post_ln_w.data) {
    SGL_ERROR("clip tensors missing in %s", path);
    sgl_clip_free(c);
    return NULL;
  }
  /* pre_ln / patch_b / post_ln bias 缺权重时合成全 1 / 全 0（普通 LN）：真实 CLIP 的
   * pre_layrnorm 带仿射、旧随机模型也有；这里只防「真的没给」的布局。
   * 合成的是自有内存，由 sgl_clip_free 按 synth 标志释放（mmap 权重不释放）。*/
  if (!c->pre_ln_w.data) {
    int64_t sh[1] = {E};
    c->pre_ln_w = sgl_tensor_zeros(1, sh);
    for (int i = 0; i < E; i++) c->pre_ln_w.data[i] = 1.0f;
    c->pre_ln_b = sgl_tensor_zeros(1, sh);
    c->synth = 1;
  }
  if (!c->patch_b.data) {
    int64_t sh[1] = {E};
    c->patch_b = sgl_tensor_zeros(1, sh);
    c->synth = 1;
  }
  if (!c->post_ln_b.data) {
    int64_t sh[1] = {E};
    c->post_ln_b = sgl_tensor_zeros(1, sh);
    c->synth = 1;
  }

  c->layers = (sgl_clip_layer *)sgl_xcalloc((size_t)c->L, sizeof(*c->layers));
  for (int l = 0; l < c->L; l++) {
    char n[160];
    sgl_clip_layer *L = &c->layers[l];
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.layer_norm1.weight", l);
    L->ln1_w = sgl_st_get(c->st, n, 1, (int64_t[]){E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.layer_norm1.bias", l);
    L->ln1_b = sgl_st_get(c->st, n, 1, (int64_t[]){E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.self_attn.q_proj.weight", l);
    L->qw = sgl_st_get(c->st, n, 2, (int64_t[]){E, E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.self_attn.q_proj.bias", l);
    L->qb = sgl_st_get(c->st, n, 1, (int64_t[]){E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.self_attn.k_proj.weight", l);
    L->kw = sgl_st_get(c->st, n, 2, (int64_t[]){E, E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.self_attn.k_proj.bias", l);
    L->kb = sgl_st_get(c->st, n, 1, (int64_t[]){E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.self_attn.v_proj.weight", l);
    L->vw = sgl_st_get(c->st, n, 2, (int64_t[]){E, E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.self_attn.v_proj.bias", l);
    L->vb = sgl_st_get(c->st, n, 1, (int64_t[]){E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.self_attn.out_proj.weight", l);
    L->ow = sgl_st_get(c->st, n, 2, (int64_t[]){E, E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.self_attn.out_proj.bias", l);
    L->ob = sgl_st_get(c->st, n, 1, (int64_t[]){E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.layer_norm2.weight", l);
    L->ln2_w = sgl_st_get(c->st, n, 1, (int64_t[]){E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.layer_norm2.bias", l);
    L->ln2_b = sgl_st_get(c->st, n, 1, (int64_t[]){E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.mlp.fc1.weight", l);
    L->fc1w = sgl_st_get(c->st, n, 2, (int64_t[]){c->I, E});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.mlp.fc1.bias", l);
    L->fc1b = sgl_st_get(c->st, n, 1, (int64_t[]){c->I});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.mlp.fc2.weight", l);
    L->fc2w = sgl_st_get(c->st, n, 2, (int64_t[]){E, c->I});
    snprintf(n, sizeof(n), "vision_model.encoder.layers.%d.mlp.fc2.bias", l);
    L->fc2b = sgl_st_get(c->st, n, 1, (int64_t[]){E});
    if (!L->qw.data || !L->ow.data || !L->fc1w.data) {
      SGL_ERROR("clip layer %d missing", l);
      sgl_clip_free(c);
      return NULL;
    }
  }

  int rows = NP + 1;
  c->x = (float *)sgl_xmalloc(sizeof(float) * (size_t)rows * E);
  c->h = (float *)sgl_xmalloc(sizeof(float) * (size_t)rows * E);
  c->qkv = (float *)sgl_xmalloc(sizeof(float) * (size_t)rows * 3 * E);
  c->attn = (float *)sgl_xmalloc(sizeof(float) * (size_t)rows * E);
  c->mid = (float *)sgl_xmalloc(sizeof(float) * (size_t)rows * c->I);
  return c;
}

/* image: [img*img*3] CHW float（RGB, 已按 CLIP 均值方差归一化） */
int sgl_clip_encode_full(const sgl_clip *c, const float *image, float *patch_out,
                         float *pooled_out, float *embeds_out, float *layer_dump) {
  int E = c->E, P = c->P, NP = c->n_patch, side = c->img / P, K = 3 * P * P;
  int rows = NP + 1;
  /* patch embed: conv == per-patch GEMM. W[E,3,P,P] row-major is [E,K]. */
  for (int py = 0; py < side; py++)
    for (int px = 0; px < side; px++) {
      int pi = py * side + px;
      float *pin = (float *)sgl_xmalloc(sizeof(float) * (size_t)K);
      for (int ch = 0; ch < 3; ch++)
        for (int yy = 0; yy < P; yy++)
          for (int xx = 0; xx < P; xx++)
            pin[ch * P * P + yy * P + xx] =
                image[(size_t)ch * c->img * c->img + (size_t)(py * P + yy) * c->img + (px * P + xx)];
      sgl_matmul_nt(pin, c->patch_w.data, c->patch_b.data, c->x + (size_t)(pi + 1) * E, 1, E, K);
      sgl_xfree(pin);
    }
  /* class token at row 0 + position embedding + pre_layrnorm */
  memcpy(c->x, c->cls_emb.data, sizeof(float) * (size_t)E);
  for (int t = 0; t < NP + 1; t++)
    for (int i = 0; i < E; i++) c->x[(size_t)t * E + i] += c->pos_emb.data[(size_t)t * E + i];
  for (int t = 0; t < NP + 1; t++)
    ln_f32(c->x + (size_t)t * E, c->pre_ln_w.data, c->pre_ln_b.data, c->x + (size_t)t * E, E, c->eps);
  if (layer_dump) memcpy(layer_dump, c->x, sizeof(float) * (size_t)rows * E);

  float *q = c->qkv, *k = c->qkv + (size_t)rows * E, *v = c->qkv + 2 * (size_t)rows * E;
  for (int l = 0; l < c->L; l++) {
    const sgl_clip_layer *L = &c->layers[l];
    for (int t = 0; t < rows; t++)
      ln_f32(c->x + (size_t)t * E, L->ln1_w.data, L->ln1_b.data, c->h + (size_t)t * E, E, c->eps);
    sgl_matmul_nt(c->h, L->qw.data, L->qb.data, q, rows, E, E);
    sgl_matmul_nt(c->h, L->kw.data, L->kb.data, k, rows, E, E);
    sgl_matmul_nt(c->h, L->vw.data, L->vb.data, v, rows, E, E);
    attention_rows(q, k, v, c->attn, rows, E, c->H);
    sgl_matmul_nt(c->attn, L->ow.data, L->ob.data, c->h, rows, E, E);
    for (size_t i = 0; i < (size_t)rows * E; i++) c->x[i] += c->h[i];
    for (int t = 0; t < rows; t++)
      ln_f32(c->x + (size_t)t * E, L->ln2_w.data, L->ln2_b.data, c->h + (size_t)t * E, E, c->eps);
    sgl_matmul_nt(c->h, L->fc1w.data, L->fc1b.data, c->mid, rows, c->I, E);
    /* hidden_act：真实 CLIP = quick_gelu；旧随机模型用 erf gelu（ref_llava.py 同步） */
    for (size_t i = 0; i < (size_t)rows * c->I; i++)
      c->mid[i] = sgl_clip_act(c->mid[i], c->act);
    sgl_matmul_nt(c->mid, L->fc2w.data, L->fc2b.data, c->h, rows, E, c->I);
    for (size_t i = 0; i < (size_t)rows * E; i++) c->x[i] += c->h[i];
    if (layer_dump)
      memcpy(layer_dump + (size_t)(l + 1) * rows * E, c->x, sizeof(float) * (size_t)rows * E);
  }
  /* last_hidden_state = c->x（★ 不过 post_layernorm）：LLaVA 取 patch 行 1..NP */
  if (patch_out) memcpy(patch_out, c->x + (size_t)E, sizeof(float) * (size_t)NP * E);
  /* pooled_output = post_layernorm(x[:, 0, :])；image_embeds = visual_projection(pooled) */
  float *pl = (float *)sgl_xmalloc(sizeof(float) * (size_t)E);
  ln_f32(c->x, c->post_ln_w.data, c->post_ln_b.data, pl, E, c->eps);
  if (pooled_out) memcpy(pooled_out, pl, sizeof(float) * (size_t)E);
  if (embeds_out && c->img_proj_dim > 0)
    sgl_matmul_nt(pl, c->vproj.data, NULL, embeds_out, 1, c->img_proj_dim, E);
  sgl_xfree(pl);
  return 0;
}

int sgl_clip_encode(const sgl_clip *c, const float *image, float *out /* [n_patch][E] */) {
  return sgl_clip_encode_full(c, image, out, NULL, NULL, NULL);
}

void sgl_clip_free(sgl_clip *c) {
  if (!c) return;
  if (c->synth) { /* 合成权重（自有内存）*/
    if (c->pre_ln_w.data) sgl_tensor_free(&c->pre_ln_w);
    if (c->pre_ln_b.data) sgl_tensor_free(&c->pre_ln_b);
    if (c->patch_b.data) sgl_tensor_free(&c->patch_b);
    if (c->post_ln_b.data) sgl_tensor_free(&c->post_ln_b);
  }
  if (c->st) sgl_st_close(c->st);
  sgl_xfree(c->layers);
  sgl_xfree(c->x); sgl_xfree(c->h); sgl_xfree(c->qkv); sgl_xfree(c->attn); sgl_xfree(c->mid);
  sgl_xfree(c);
}
