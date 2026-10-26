/* sgl clip.c — CLIP ViT vision encoder (t10 LLaVA 第一步)
 * 对应 sglang srt/models/llava.py 的 CLIPVisionModel 部分：
 *   conv patch embed（等效 GEMM：patch [3*P*P] × W^T）+ class emb + pos emb
 *   + ln_pre + N 层 transformer（全可见 attention，GELU MLP）
 *   + post_layernorm（drop class token）+ 可选 visual_projection。
 * 权重来自 HF openai/clip-vit-large-patch14-336 布局（llava-1.5 同源）；
 * 计算全程 fp32，数值与 numpy 参考逐位对齐（test_llava 断言）。 */
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

sgl_clip *sgl_clip_load(const char *dir) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/clip_config.json", dir);
  FILE *f = NULL;
  fopen_s(&f, path, "rb");
  if (!f) { SGL_ERROR("clip_config.json missing in %s", dir); return NULL; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); sgl_xfree(buf); return NULL; }
  buf[sz] = 0;
  fclose(f);
  sgl_json *cfg = sgl_json_parse(buf, (size_t)sz);
  sgl_xfree(buf);
  if (!cfg) return NULL;

  sgl_clip *c = (sgl_clip *)sgl_xcalloc(1, sizeof(*c));
  c->E = sgl_json_get_int(cfg, "hidden_size", 768);
  c->H = sgl_json_get_int(cfg, "num_heads", 12);
  c->L = sgl_json_get_int(cfg, "num_hidden_layers", 12);
  c->P = sgl_json_get_int(cfg, "patch_size", 32);
  c->img = sgl_json_get_int(cfg, "image_size", 336);
  c->I = sgl_json_get_int(cfg, "intermediate_size", 4 * c->E);
  c->proj_dim = sgl_json_get_int(cfg, "projection_dim", 0); /* 0 = no projection */
  c->eps = 1e-5f;
  c->n_patch = (c->img / c->P) * (c->img / c->P);
  sgl_json_free(cfg);
  if (c->E <= 0 || c->H <= 0 || c->L <= 0 || c->P <= 0 || c->img % c->P != 0) {
    SGL_ERROR("clip config invalid");
    sgl_xfree(c);
    return NULL;
  }

  snprintf(path, sizeof(path), "%s/clip_model.safetensors", dir);
  c->st = sgl_st_open(path);
  if (!c->st) { SGL_ERROR("clip_model.safetensors open failed: %s", path); sgl_xfree(c); return NULL; }

  int E = c->E, NP = c->n_patch;
  c->cls_emb = sgl_st_get(c->st, "vision_model.embeddings.class_embedding", 1, (int64_t[]){E});
  c->pos_emb = sgl_st_get(c->st, "vision_model.embeddings.position_embedding", 2, (int64_t[]){NP + 1, E});
  {
    /* conv weight [E,3,P,P] row-major flattens to [E,3*P*P]; take it unvalidated
     * (st_get expects exact ndim) then check numel by hand. */
    c->patch_w = sgl_st_get(c->st, "vision_model.embeddings.patch_embedding.weight", -1, NULL);
    if (c->patch_w.data &&
        sgl_tensor_numel(&c->patch_w) != (int64_t)E * 3 * c->P * c->P)
      c->patch_w = sgl_tensor_null();
  }
  c->patch_b = sgl_st_get(c->st, "vision_model.embeddings.patch_embedding.bias", 1, (int64_t[]){E});
  c->pre_ln_w = sgl_st_get(c->st, "vision_model.embeddings.pre_layernorm.weight", 1, (int64_t[]){E});
  c->pre_ln_b = sgl_st_get(c->st, "vision_model.embeddings.pre_layernorm.bias", 1, (int64_t[]){E});
  c->post_ln_w = sgl_st_get(c->st, "vision_model.post_layernorm.weight", 1, (int64_t[]){E});
  c->post_ln_b = sgl_st_get(c->st, "vision_model.post_layernorm.bias", 1, (int64_t[]){E});
  if (c->proj_dim > 0)
    c->vproj = sgl_st_get(c->st, "visual_projection.weight", 2, (int64_t[]){c->proj_dim, E});
  if (!c->cls_emb.data || !c->pos_emb.data || !c->patch_w.data || !c->patch_b.data ||
      !c->pre_ln_w.data || !c->post_ln_w.data) {
    SGL_ERROR("clip tensors missing");
    sgl_clip_free(c);
    return NULL;
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

/* image: [img*img*3] CHW float（RGB, 已归一化）——与 HF preprocess 一致的布局 */
int sgl_clip_encode(const sgl_clip *c, const float *image, float *out /* [n_patch][E or proj] */) {
  int E = c->E, P = c->P, NP = c->n_patch, side = c->img / P, K = 3 * P * P;
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
  /* class token at row 0 + position embedding + pre LN */
  memcpy(c->x, c->cls_emb.data, sizeof(float) * (size_t)E);
  for (int t = 0; t < NP + 1; t++)
    for (int i = 0; i < E; i++) c->x[(size_t)t * E + i] += c->pos_emb.data[(size_t)t * E + i];
  for (int t = 0; t < NP + 1; t++)
    ln_f32(c->x + (size_t)t * E, c->pre_ln_w.data, c->pre_ln_b.data, c->x + (size_t)t * E, E, c->eps);

  int rows = NP + 1;
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
    /* GELU exact (erf) — HF CLIP uses nn.GELU() default (erf) */
    for (size_t i = 0; i < (size_t)rows * c->I; i++) {
      float u = c->mid[i];
      c->mid[i] = 0.5f * u * (1.0f + erff(u * 0.70710678118654752440f));
    }
    sgl_matmul_nt(c->mid, L->fc2w.data, L->fc2b.data, c->h, rows, E, c->I);
    for (size_t i = 0; i < (size_t)rows * E; i++) c->x[i] += c->h[i];
  }
  /* post LN on all rows; drop the class row (row 0) in the output */
  for (int t = 0; t < rows; t++)
    ln_f32(c->x + (size_t)t * E, c->post_ln_w.data, c->post_ln_b.data, c->h + (size_t)t * E, E, c->eps);
  if (c->proj_dim > 0)
    sgl_matmul_nt(c->h + E /* skip class row */, c->vproj.data, NULL, out, NP, c->proj_dim, E);
  else
    memcpy(out, c->h + E, sizeof(float) * (size_t)NP * E);
  return 0;
}

void sgl_clip_free(sgl_clip *c) {
  if (!c) return;
  if (c->st) sgl_st_close(c->st);
  sgl_xfree(c->layers);
  sgl_xfree(c->x); sgl_xfree(c->h); sgl_xfree(c->qkv); sgl_xfree(c->attn); sgl_xfree(c->mid);
  sgl_xfree(c);
}
