/* sgl/clip.h — CLIP ViT encoder (LLaVA 视觉塔, t10; 真实权重对齐 G8) */
#ifndef SGL_CLIP_H
#define SGL_CLIP_H

#include "sgl/safetensors.h"
#include "sgl/tensor.h"

/* MLP 激活：真实 HF CLIP（hidden_act=quick_gelu）为 QUICK；
 * GELU_ERF = 0.5x(1+erf(x/√2))，GELU_TANH = gelu_new。 */
enum {
  SGL_CLIP_ACT_QUICK = 0,
  SGL_CLIP_ACT_GELU_ERF = 1,
  SGL_CLIP_ACT_GELU_TANH = 2
};

typedef struct sgl_clip_layer {
  sgl_tensor ln1_w, ln1_b, qw, qb, kw, kb, vw, vb, ow, ob;
  sgl_tensor ln2_w, ln2_b, fc1w, fc1b, fc2w, fc2b;
} sgl_clip_layer;

typedef struct sgl_clip {
  int E, H, L, P, img, I, proj_dim, n_patch;
  /* proj_dim = sgl_clip_encode 每行的宽度（真实语义恒等于 E，保留该字段
   * 是给已有调用方（cuda_http/cuda_test）算 buffer 用）；
   * img_proj_dim = CLIP visual_projection 的输出维（0 = 权重里没有）*/
  int img_proj_dim;
  int act;   /* SGL_CLIP_ACT_* */
  float eps; /* layer_norm_eps（真实 CLIP = 1e-5）*/
  sgl_st_ctx *st;
  sgl_tensor cls_emb, pos_emb, patch_w, patch_b, pre_ln_w, pre_ln_b;
  sgl_tensor post_ln_w, post_ln_b, vproj;
  sgl_clip_layer *layers;
  int synth; /* 1 = 有合成权重（pre_ln/patch_b/post_ln bias 缺省），释放时回收 */
  float *x, *h, *qkv, *attn, *mid;
} sgl_clip;

/* 打开 <dir>/clip_config.json（随机小模型）或 <dir>/config.json（真实 HF CLIP，
 * 读 vision_config 子对象）；权重 <dir>/clip_model.safetensors 或 <dir>/model.safetensors。
 * 张量名兼容 HF：position_embedding[.weight]、pre_layrnorm / embeddings.pre_layernorm。 */
sgl_clip *sgl_clip_load(const char *dir);

/* 真实 CLIPVisionModel 语义（transformers modeling_clip.py）：
 *   emb = conv(patch) 拼 class 再加 position_embedding
 *   x   = pre_layrnorm(emb)                       （带仿射，权重缺省则退化为普通 LN）
 *   x   = encoder layers 0..L-1                   （双向 attention + hidden_act MLP）
 *   last_hidden_state = x                          （★ 不过 post_layernorm）
 *   pooled_output     = post_layernorm(x[:, 0, :])
 *   image_embeds      = visual_projection(pooled)
 * LLaVA 取的视觉 token 是 last_hidden_state[:, 1:, :]，即 patch_rows。
 *
 * image: [img*img*3] CHW float（RGB，已按 CLIP 均值方差归一化）
 * patch_out: [n_patch][E]（可为 NULL）
 * pooled_out: [E]（可为 NULL）  embeds_out: [img_proj_dim]（可为 NULL）
 * layer_dump: 可为 NULL；非 NULL 时需 (L+1)*n_patch+1 行 × E 列，
 *   第 0 行 = encoder 输入（pre_ln 之后，含 class 共 n_patch+1 行），
 *   第 1..L 行 = 每个 encoder layer 输出（各 n_patch+1 行）。 */
int sgl_clip_encode_full(const sgl_clip *c, const float *image, float *patch_out,
                         float *pooled_out, float *embeds_out, float *layer_dump);

/* 兼容旧接口：out: [n_patch][proj_dim]（= [n_patch][E]，patch tokens，无 class）*/
int sgl_clip_encode(const sgl_clip *c, const float *image, float *out);
void sgl_clip_free(sgl_clip *c);

#endif
