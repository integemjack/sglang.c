/* sgl/clip.h — CLIP ViT encoder (LLaVA 视觉塔, t10) */
#ifndef SGL_CLIP_H
#define SGL_CLIP_H

#include "sgl/safetensors.h"
#include "sgl/tensor.h"

typedef struct sgl_clip_layer {
  sgl_tensor ln1_w, ln1_b, qw, qb, kw, kb, vw, vb, ow, ob;
  sgl_tensor ln2_w, ln2_b, fc1w, fc1b, fc2w, fc2b;
} sgl_clip_layer;

typedef struct sgl_clip {
  int E, H, L, P, img, I, proj_dim, n_patch;
  float eps;
  sgl_st_ctx *st;
  sgl_tensor cls_emb, pos_emb, patch_w, patch_b, pre_ln_w, pre_ln_b;
  sgl_tensor post_ln_w, post_ln_b, vproj;
  sgl_clip_layer *layers;
  float *x, *h, *qkv, *attn, *mid;
} sgl_clip;

sgl_clip *sgl_clip_load(const char *dir);
/* image: [img*img*3] CHW；out: [n_patch][proj_dim>0 ? proj_dim : E]（patch tokens，无 class） */
int sgl_clip_encode(const sgl_clip *c, const float *image, float *out);
void sgl_clip_free(sgl_clip *c);

#endif
