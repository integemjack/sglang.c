/* sgl/image.h — 图片解码 + CLIP 预处理 + base64（G8 多模态入口）
 * 对应 sglang srt/multimodal/processors 的「字节 → pixel_values」那一段：
 *   base64 解码 → stb_image 解 JPEG/PNG → RGB → 双三次 resize → center crop
 *   → /255 → (x-mean)/std → CHW fp32（直接喂 sgl_clip_encode）。
 * resize 按 PIL Image.BICUBIC（Catmull-Rom a=-0.5，缩小时按倍率放大支撑窗=预滤波）
 * 实现，与 python 侧逐像素对照见 test_llava（阈值 2/255 归一化后 ~0.03）。 */
#ifndef SGL_IMAGE_H
#define SGL_IMAGE_H

#include <stddef.h>

#include "sgl/clip.h"

/* CLIP 默认均值/方差（HF preprocessor_config.json 的 image_mean/std）*/
extern const float sgl_clip_mean[3];
extern const float sgl_clip_std[3];

/* base64 解码：返回 sgl_xmalloc 的字节（调用方 sgl_xfree），失败 -1。
 * 允许换行/空白；padding 可选。*/
int sgl_base64_decode(const char *in, size_t n, unsigned char **out, size_t *out_len);

/* 解码 JPEG/PNG 字节 → RGB [h][w][3]（sgl_xmalloc，调用方 sgl_xfree）；失败 -1 */
int sgl_image_decode_rgb(const unsigned char *bytes, size_t n,
                         unsigned char **rgb, int *w, int *h);

/* 双三次（PIL BICUBIC 语义）把 [sh][sw][3] 缩到 [dh][dw][3]；dst 需 dh*dw*3 字节 */
void sgl_image_resize_bicubic(const unsigned char *src, int sw, int sh,
                              unsigned char *dst, int dw, int dh);

/* center crop：从 [sh][sw][3] 取 ((sw-cw)/2,(sh-ch)/2) 起的 [ch][cw][3] */
void sgl_image_center_crop(const unsigned char *src, int sw, int sh,
                           unsigned char *dst, int cw, int ch);

/* 一步到位：RGB → resize 到 size×size（保持 PIL 语义）→ center crop size×size
 * → /255 → (x-mean)/std → CHW fp32 [3][size][size]（chw 需 3*size*size float）。
 * mean/std 为 NULL 时用 sgl_clip_mean/std。 */
void sgl_clip_preprocess(const unsigned char *rgb, int w, int h, int size,
                         const float *mean, const float *stdv, float *chw);

/* G8 一站式（CPU）：图片字节（JPEG/PNG）→ 解码 → resize/crop → 归一化
 * → CLIP 视觉塔（last_hidden_state 的 patch 行）→ mm_projector → LM 空间特征。
 * proj_w: [e_lm][clip_E] 行主序（llava-1.5 的 linear_1，单层简化）；proj_b 可为 NULL。
 * out: sgl_xmalloc 的 [n_img][e_lm]（调用方 sgl_xfree），*n_out = n_img * e_lm。
 * 返回 0 成功。 */
int sgl_clip_project_bytes(const unsigned char *bytes, size_t n, const sgl_clip *c,
                           const float *proj_w, const float *proj_b, int e_lm,
                           float **out, int *n_out);

#endif
