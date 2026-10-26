/* sgl image.c — 图片解码（stb_image）+ CLIP 预处理 + base64（G8）
 * 见 include/sgl/image.h。resize 复刻 PIL 的 BICUBIC（Catmull-Rom a=-0.5，
 * 缩小时按倍率放大支撑窗做预滤波），与 python PIL 逐像素对照见 test_llava。 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/log.h"
#include "sgl/ops.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO /* 只从内存解，不读文件 */
#include "stb_image.h"

const float sgl_clip_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
const float sgl_clip_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};

/* ---------------- base64 ---------------- */

static int b64_val(int c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

int sgl_base64_decode(const char *in, size_t n, unsigned char **out, size_t *out_len) {
  unsigned char *buf = (unsigned char *)sgl_xmalloc(n / 4 * 3 + 4);
  size_t o = 0;
  int quad[4], qn = 0;
  for (size_t i = 0; i < n; i++) {
    char c = in[i];
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    int v = b64_val((unsigned char)c);
    if (v < 0) { sgl_xfree(buf); return -1; }
    quad[qn++] = v;
    if (qn == 4) {
      buf[o++] = (unsigned char)((quad[0] << 2) | (quad[1] >> 4));
      buf[o++] = (unsigned char)(((quad[1] & 15) << 4) | (quad[2] >> 2));
      buf[o++] = (unsigned char)(((quad[2] & 3) << 6) | quad[3]);
      qn = 0;
    }
  }
  if (qn == 2) {
    buf[o++] = (unsigned char)((quad[0] << 2) | (quad[1] >> 4));
  } else if (qn == 3) {
    buf[o++] = (unsigned char)((quad[0] << 2) | (quad[1] >> 4));
    buf[o++] = (unsigned char)(((quad[1] & 15) << 4) | (quad[2] >> 2));
  } else if (qn == 1) {
    sgl_xfree(buf);
    return -1;
  }
  *out = buf;
  *out_len = o;
  return 0;
}

/* ---------------- 解码 ---------------- */

int sgl_image_decode_rgb(const unsigned char *bytes, size_t n,
                         unsigned char **rgb, int *w, int *h) {
  int cw = 0, chh = 0, cn = 0;
  unsigned char *src = stbi_load_from_memory(bytes, (int)n, &cw, &chh, &cn, 3);
  if (!src) {
    SGL_ERROR("image decode failed: %s", stbi_failure_reason());
    return -1;
  }
  size_t sz = (size_t)cw * (size_t)chh * 3;
  unsigned char *dst = (unsigned char *)sgl_xmalloc(sz);
  memcpy(dst, src, sz);
  stbi_image_free(src);
  *rgb = dst;
  *w = cw;
  *h = chh;
  return 0;
}

/* ---------------- resize（PIL BICUBIC）---------------- */

/* PIL 的 cubic filter：a = -0.5（Catmull-Rom） */
static float cubic_w(float x) {
  const float a = -0.5f;
  if (x < 0) x = -x;
  if (x < 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
  if (x < 2.0f) return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
  return 0.0f;
}

/* 一个轴的采样表：每个输出下标对应 [lo, hi) 的输入下标与权重（权重已归一化）。
 * PIL 语义：filterscale = max(1, in/out)（缩小时放大支撑窗 = 预滤波），
 * center = (o + 0.5) * in/out - 0.5，支撑半径 2*filterscale。 */
typedef struct {
  int *lo, *hi;
  float **wt;
} axis_map;

static void axis_map_build(axis_map *m, int in, int out) {
  m->lo = (int *)sgl_xmalloc(sizeof(int) * (size_t)out);
  m->hi = (int *)sgl_xmalloc(sizeof(int) * (size_t)out);
  m->wt = (float **)sgl_xmalloc(sizeof(float *) * (size_t)out);
  double scale = in > out ? (double)in / out : 1.0;
  double support = 2.0 * scale;
  for (int o = 0; o < out; o++) {
    double center = (o + 0.5) * in / out - 0.5;
    int lo = (int)floor(center - support);
    int hi = (int)ceil(center + support);
    if (lo < 0) lo = 0;
    if (hi > in) hi = in;
    if (hi <= lo) { /* 极端上采样时的兜底：取最近像素 */
      lo = (int)floor(center + 0.5);
      if (lo < 0) lo = 0;
      if (lo >= in) lo = in - 1;
      hi = lo + 1;
    }
    int cnt = hi - lo;
    float *w = (float *)sgl_xmalloc(sizeof(float) * (size_t)cnt);
    double sum = 0;
    for (int j = 0; j < cnt; j++) {
      double x = ((lo + j) - center) / scale;
      double v = cubic_w((float)x); /* 负瓣（ringing）要保留，PIL 不夹 */
      w[j] = (float)v;
      sum += v;
    }
    if (sum <= 0) {
      for (int j = 0; j < cnt; j++) w[j] = 0;
      w[cnt / 2] = 1.0f;
      sum = 1.0;
    }
    for (int j = 0; j < cnt; j++) w[j] = (float)(w[j] / sum);
    m->lo[o] = lo;
    m->hi[o] = hi;
    m->wt[o] = w;
  }
}

static void axis_map_free(axis_map *m) {
  /* 每个输出下标的权重表由调用方先逐个 sgl_xfree */
  sgl_xfree(m->lo);
  sgl_xfree(m->hi);
  sgl_xfree(m->wt);
}

void sgl_image_resize_bicubic(const unsigned char *src, int sw, int sh,
                              unsigned char *dst, int dw, int dh) {
  if (sw == dw && sh == dh) {
    memcpy(dst, src, (size_t)sw * (size_t)sh * 3);
    return;
  }
  axis_map mx, my;
  axis_map_build(&mx, sw, dw);
  axis_map_build(&my, sh, dh);
  /* 横向：src[sh][sw][3] → tmp[sh][dw][3]（uint8）。
   * PIL 的两 pass 中间结果也是 8bit（ImagingResampleHorizontal_8bit 写回 UINT8），
   * 这里跟着舍入，否则与 PIL 在强边缘处差到 ~9 个像素级。*/
  unsigned char *tmp = (unsigned char *)sgl_xmalloc((size_t)sh * dw * 3);
  for (int y = 0; y < sh; y++) {
    const unsigned char *row = src + (size_t)y * sw * 3;
    unsigned char *orow = tmp + (size_t)y * dw * 3;
    for (int x = 0; x < dw; x++) {
      const float *w = mx.wt[x];
      int lo = mx.lo[x], hi = mx.hi[x];
      float acc[3] = {0, 0, 0};
      for (int j = lo; j < hi; j++) {
        float ww = w[j - lo];
        const unsigned char *p = row + (size_t)j * 3;
        acc[0] += ww * p[0];
        acc[1] += ww * p[1];
        acc[2] += ww * p[2];
      }
      for (int c = 0; c < 3; c++) {
        int v = (int)lroundf(acc[c]);
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        orow[x * 3 + c] = (unsigned char)v;
      }
    }
  }
  /* 纵向：tmp → dst（四舍五入并夹到 [0,255]，与 PIL 一致）*/
  for (int y = 0; y < dh; y++) {
    const float *w = my.wt[y];
    int lo = my.lo[y], hi = my.hi[y];
    unsigned char *orow = dst + (size_t)y * dw * 3;
    for (int x = 0; x < dw; x++) {
      float acc[3] = {0, 0, 0};
      for (int j = lo; j < hi; j++) {
        float ww = w[j - lo];
        const unsigned char *p = tmp + ((size_t)j * dw + x) * 3;
        acc[0] += ww * p[0];
        acc[1] += ww * p[1];
        acc[2] += ww * p[2];
      }
      for (int c = 0; c < 3; c++) {
        int v = (int)lroundf(acc[c]);
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        orow[x * 3 + c] = (unsigned char)v;
      }
    }
  }
  sgl_xfree(tmp);
  for (int x = 0; x < dw; x++) sgl_xfree(mx.wt[x]);
  for (int y = 0; y < dh; y++) sgl_xfree(my.wt[y]);
  axis_map_free(&mx);
  axis_map_free(&my);
}

void sgl_image_center_crop(const unsigned char *src, int sw, int sh,
                           unsigned char *dst, int cw, int ch) {
  int left = (sw - cw) / 2;
  int top = (sh - ch) / 2;
  if (left < 0) left = 0;
  if (top < 0) top = 0;
  for (int y = 0; y < ch; y++)
    memcpy(dst + (size_t)y * cw * 3,
           src + (size_t)(top + y) * sw * 3 + (size_t)left * 3,
           (size_t)cw * 3);
}

int sgl_clip_project_bytes(const unsigned char *bytes, size_t n, const sgl_clip *c,
                           const float *proj_w, const float *proj_b, int e_lm,
                           float **out, int *n_out) {
  unsigned char *rgb = NULL;
  int w = 0, h = 0;
  if (sgl_image_decode_rgb(bytes, n, &rgb, &w, &h) != 0) return -1;
  int sz = c->img;
  float *chw = (float *)sgl_xmalloc(sizeof(float) * 3 * (size_t)sz * (size_t)sz);
  sgl_clip_preprocess(rgb, w, h, sz, NULL, NULL, chw);
  sgl_xfree(rgb);
  int NP = c->n_patch, E = c->E;
  float *feats = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * E);
  if (sgl_clip_encode(c, chw, feats) != 0) {
    sgl_xfree(chw); sgl_xfree(feats);
    return -1;
  }
  sgl_xfree(chw);
  float *lm = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * e_lm);
  sgl_matmul_nt(feats, proj_w, proj_b, lm, NP, e_lm, E);
  sgl_xfree(feats);
  *out = lm;
  *n_out = NP * e_lm;
  return 0;
}

void sgl_clip_preprocess(const unsigned char *rgb, int w, int h, int size,
                         const float *mean, const float *stdv, float *chw) {
  if (!mean) mean = sgl_clip_mean;
  if (!stdv) stdv = sgl_clip_std;
  unsigned char *sq = (unsigned char *)sgl_xmalloc((size_t)size * size * 3);
  /* CLIP：短边缩到 size 后 center crop（方形输入时等价于直接缩到 size×size）*/
  int sw = size, sh = size;
  if (w != size || h != size) {
    /* 先按 PIL 语义缩到 size×size（LLaVA/CLIP 的 resize(size,size)），再 crop（恒等）*/
    sgl_image_resize_bicubic(rgb, w, h, sq, sw, sh);
  } else {
    memcpy(sq, rgb, (size_t)size * size * 3);
  }
  unsigned char *cr = (unsigned char *)sgl_xmalloc((size_t)size * size * 3);
  sgl_image_center_crop(sq, sw, sh, cr, size, size);
  for (int c = 0; c < 3; c++)
    for (int y = 0; y < size; y++)
      for (int x = 0; x < size; x++) {
        float v = cr[((size_t)y * size + x) * 3 + c] / 255.0f;
        chw[(size_t)c * size * size + (size_t)y * size + x] = (v - mean[c]) / stdv[c];
      }
  sgl_xfree(sq);
  sgl_xfree(cr);
}
