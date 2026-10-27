/* sgl gguf 实现：GGUF 容器解析 + ggml 量化反解 + GGUF↔HF 名字映射
 *
 * 文件布局（v2/v3，都是小端）：
 *   u32 magic "GGUF" | u32 version | u64 n_tensors | u64 n_kv
 *   kv[n_kv]      = { str key | u32 type | value }
 *   info[n_tensors] = { str name | u32 n_dims | u64 ne[n_dims] | u32 type | u64 offset }
 *   <padding 到 alignment（general.alignment，默认 32）>
 *   data 区（各张量按 offset 定位）
 * 字符串一律 { u64 len | bytes }。
 *
 * 反量化的位布局逐项对应 llama.cpp 的 ggml-quants.c（dequantize_row_*）。
 * 每个 block 的最内维长度固定（Q4_0 系列 32、K 系列 256），且 ne[0] 必须能被整除。
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/gguf.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <dirent.h>
#endif

#include "sgl/alloc.h"
#include "sgl/log.h"
#include "sgl/port.h"

/* ---- ggml_type 编号（与 ggml.h 保持一致）---- */
enum {
  GGML_T_F32 = 0,
  GGML_T_F16 = 1,
  GGML_T_Q4_0 = 2,
  GGML_T_Q4_1 = 3,
  GGML_T_Q5_0 = 6,
  GGML_T_Q5_1 = 7,
  GGML_T_Q8_0 = 8,
  GGML_T_Q8_1 = 9,
  GGML_T_IQ4_NL = 20, /* 4bit 非线性码本（kvalues_iq4nl），bartowski 的小体积远离体型里常见 */
  GGML_T_Q2_K = 10,
  GGML_T_Q3_K = 11,
  GGML_T_Q4_K = 12,
  GGML_T_Q5_K = 13,
  GGML_T_Q6_K = 14,
  GGML_T_Q8_K = 15,
  GGML_T_I8 = 24,
  GGML_T_I16 = 25,
  GGML_T_I32 = 26,
  GGML_T_I64 = 27,
  GGML_T_F64 = 28,
  GGML_T_BF16 = 30
};

/* GGUF 元数据类型 */
enum { GK_U8 = 0, GK_I8 = 1, GK_U16 = 2, GK_I16 = 3, GK_U32 = 4, GK_I32 = 5,
       GK_F32 = 6, GK_BOOL = 7, GK_STR = 8, GK_ARR = 9, GK_U64 = 10, GK_I64 = 11,
       GK_F64 = 12 };

typedef struct {
  const char *name; /* 指向 mmap 内（无 NUL）*/
  char *name_owned; /* NUL 结尾副本（strcmp 用）*/
  int n_dims;
  int64_t ne[4];
  int type;
  uint64_t offset;
  int64_t numel;
} gg_tensor;

typedef struct {
  char *key_owned;     /* NUL 结尾副本 */
  const char *key;
  uint32_t type;
  uint64_t n;        /* 数组元素数；标量为 1 */
  uint32_t elem_type; /* 数组元素类型 */
  const uint8_t *vals; /* 值区起点 */
  char *str_owned;     /* 字符串值的 NUL 结尾副本 */
  const char *str;     /* 字符串值（type==GK_STR）*/
  uint64_t str_len;
} gg_kv;

struct sgl_gguf {
  sgl_mmap map;
  uint32_t version;
  gg_tensor *tensors;
  int n_tensors;
  gg_kv *kv;
  int n_kv;
  const char *data;
  size_t data_len;
  char *path;
  /* 名字映射缓存（HF 名），sgl_gguf_names 用 */
  char **hf_names;
  int n_hf_names;
  int hf_names_built;
};

/* ---------- 小工具 ---------- */

static char *xstrdup_(const char *s) {
  size_t n = strlen(s);
  char *d = (char *)sgl_xmalloc(n + 1);
  memcpy(d, s, n + 1);
  return d;
}

/* GGUF 的字符串是 {len, bytes}，没有 NUL：比较/打印前必须复制补 NUL */
static char *xstrndup_(const char *s, size_t n) {
  char *d = (char *)sgl_xmalloc(n + 1);
  if (n) memcpy(d, s, n);
  d[n] = 0;
  return d;
}

static float f16_to_f32_(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  int32_t e = (int32_t)((h >> 10) & 0x1F);
  uint32_t m = (uint32_t)(h & 0x3FFu);
  uint32_t f;
  if (e == 0) {
    if (m == 0) {
      f = s; /* ±0 */
    } else {
      /* 次正规数：值 = m * 2^-24，必须归一化到 f32 的正常数指数，
       * 直接放 m<<13 会变成 f32 次正规（~1e-38）—— 量化块 d 落在
       * 5e-5 量级时就会撞上这个分支。*/
      e = 1;
      while (!(m & 0x400u)) {
        m <<= 1;
        e--;
      }
      m &= 0x3FFu;
      f = s | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
    }
  } else if (e == 31) {
    f = s | 0x7F800000u | (m << 13);
  } else {
    f = s | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
  }
  float out;
  memcpy(&out, &f, 4);
  return out;
}

static float bf16_to_f32_(uint16_t h) {
  uint32_t f = (uint32_t)h << 16;
  float out;
  memcpy(&out, &f, 4);
  return out;
}

/* 读一个 ggml 半精度（小端 u16）*/
static float rd_half(const uint8_t *p) {
  uint16_t h;
  memcpy(&h, p, 2);
  return f16_to_f32_(h);
}

/* ---------- 反量化 ---------- */

/* 每个 block 的元素数与字节数（0 表示未知/不支持）*/
static void block_layout(int type, int *elems, int *bytes) {
  switch (type) {
    case GGML_T_F32: *elems = 1; *bytes = 4; return;
    case GGML_T_F16: *elems = 1; *bytes = 2; return;
    case GGML_T_BF16: *elems = 1; *bytes = 2; return;
    case GGML_T_I8: *elems = 1; *bytes = 1; return;
    case GGML_T_I16: *elems = 1; *bytes = 2; return;
    case GGML_T_I32: *elems = 1; *bytes = 4; return;
    case GGML_T_I64: *elems = 1; *bytes = 8; return;
    case GGML_T_F64: *elems = 1; *bytes = 8; return;
    case GGML_T_Q4_0: *elems = 32; *bytes = 18; return;
    case GGML_T_IQ4_NL: *elems = 32; *bytes = 18; return;
    case GGML_T_Q4_1: *elems = 32; *bytes = 20; return;
    case GGML_T_Q5_0: *elems = 32; *bytes = 22; return;
    case GGML_T_Q5_1: *elems = 32; *bytes = 24; return;
    case GGML_T_Q8_0: *elems = 32; *bytes = 34; return;
    case GGML_T_Q8_1: *elems = 32; *bytes = 36; return;
    case GGML_T_Q2_K: *elems = 256; *bytes = 84; return;
    case GGML_T_Q3_K: *elems = 256; *bytes = 110; return;
    case GGML_T_Q4_K: *elems = 256; *bytes = 144; return;
    case GGML_T_Q5_K: *elems = 256; *bytes = 176; return;
    case GGML_T_Q6_K: *elems = 256; *bytes = 210; return;
    case GGML_T_Q8_K: *elems = 256; *bytes = 292; return;
    default: *elems = 0; *bytes = 0; return;
  }
}

const char *sgl_gguf_type_name(int type) {
  switch (type) {
    case GGML_T_F32: return "F32";
    case GGML_T_F16: return "F16";
    case GGML_T_BF16: return "BF16";
    case GGML_T_Q4_0: return "Q4_0";
    case GGML_T_Q4_1: return "Q4_1";
    case GGML_T_Q5_0: return "Q5_0";
    case GGML_T_Q5_1: return "Q5_1";
    case GGML_T_Q8_0: return "Q8_0";
    case GGML_T_Q8_1: return "Q8_1";
    case GGML_T_IQ4_NL: return "IQ4_NL";
    case GGML_T_Q2_K: return "Q2_K";
    case GGML_T_Q3_K: return "Q3_K";
    case GGML_T_Q4_K: return "Q4_K";
    case GGML_T_Q5_K: return "Q5_K";
    case GGML_T_Q6_K: return "Q6_K";
    case GGML_T_Q8_K: return "Q8_K";
    case GGML_T_I8: return "I8";
    case GGML_T_I16: return "I16";
    case GGML_T_I32: return "I32";
    case GGML_T_I64: return "I64";
    case GGML_T_F64: return "F64";
    default: return "unknown";
  }
}

/* Q4_K/Q5_K 的 6-bit scale/min 打包（ggml 的 get_scale_min_k4）*/
static void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
    *m = (uint8_t)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
  }
}

/* 反量化一个张量的全部元素到 out（numel 个）。
 * 返回 0 成功；-1 表示类型不支持。 */
static int dequant_tensor(int type, const uint8_t *src, int64_t numel, float *out) {
  int be = 0, bb = 0;
  block_layout(type, &be, &bb);
  if (be == 0) return -1;
  if (be == 1) { /* 非量化：逐元素 */
    for (int64_t i = 0; i < numel; i++) {
      if (type == GGML_T_F32) {
        memcpy(&out[i], src + (size_t)i * 4, 4);
      } else if (type == GGML_T_F16) {
        uint16_t h;
        memcpy(&h, src + (size_t)i * 2, 2);
        out[i] = f16_to_f32_(h);
      } else if (type == GGML_T_BF16) {
        uint16_t h;
        memcpy(&h, src + (size_t)i * 2, 2);
        out[i] = bf16_to_f32_(h);
      } else if (type == GGML_T_I8) {
        out[i] = (float)(int8_t)src[i];
      } else if (type == GGML_T_I16) {
        int16_t v;
        memcpy(&v, src + (size_t)i * 2, 2);
        out[i] = (float)v;
      } else if (type == GGML_T_I32) {
        int32_t v;
        memcpy(&v, src + (size_t)i * 4, 4);
        out[i] = (float)v;
      } else if (type == GGML_T_I64) {
        int64_t v;
        memcpy(&v, src + (size_t)i * 8, 8);
        out[i] = (float)v;
      } else if (type == GGML_T_F64) {
        double v;
        memcpy(&v, src + (size_t)i * 8, 8);
        out[i] = (float)v;
      } else {
        return -1;
      }
    }
    return 0;
  }

  if (numel % be != 0) return -1;
  int64_t nb = numel / be;
  for (int64_t b = 0; b < nb; b++) {
    const uint8_t *p = src + (size_t)b * bb;
    float *y = out + (size_t)b * be;
    switch (type) {
      case GGML_T_IQ4_NL: {
        /* block_iq4_nl = { ggml_half d; uint8_t qs[16]; }，4bit 索引查 kvalues 表 */
        static const float kvalues[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f,
                                          -22.f, -10.f, 1.f, 13.f, 25.f, 38.f,
                                          53.f, 69.f, 89.f, 113.f};
        float d = rd_half(p); /* 注意：rd_half 已经转成 float，不要再套 f16_to_f32_ */
        const uint8_t *qs = p + 2;
        for (int j = 0; j < 16; j++) {
          y[j] = d * kvalues[qs[j] & 0xF];
          y[j + 16] = d * kvalues[qs[j] >> 4];
        }
        break;
      }
      case GGML_T_Q4_0: {
        const float d = rd_half(p);
        const uint8_t *qs = p + 2;
        for (int j = 0; j < 16; j++) {
          y[j] = (float)((int)(qs[j] & 0x0F) - 8) * d;
          y[j + 16] = (float)((int)(qs[j] >> 4) - 8) * d;
        }
        break;
      }
      case GGML_T_Q4_1: {
        const float d = rd_half(p), m = rd_half(p + 2);
        const uint8_t *qs = p + 4;
        for (int j = 0; j < 16; j++) {
          y[j] = (float)(qs[j] & 0x0F) * d + m;
          y[j + 16] = (float)(qs[j] >> 4) * d + m;
        }
        break;
      }
      case GGML_T_Q5_0: {
        const float d = rd_half(p);
        uint32_t qh;
        memcpy(&qh, p + 2, 4);
        const uint8_t *qs = p + 6;
        for (int j = 0; j < 16; j++) {
          int xh0 = (int)((qh >> j) & 1);
          int xh1 = (int)((qh >> (j + 16)) & 1);
          int v0 = (int)(qs[j] & 0x0F) | (xh0 << 4);
          int v1 = (int)(qs[j] >> 4) | (xh1 << 4);
          y[j] = (float)(v0 - 16) * d;
          y[j + 16] = (float)(v1 - 16) * d;
        }
        break;
      }
      case GGML_T_Q5_1: {
        const float d = rd_half(p), m = rd_half(p + 2);
        uint32_t qh;
        memcpy(&qh, p + 4, 4);
        const uint8_t *qs = p + 8;
        for (int j = 0; j < 16; j++) {
          int v0 = (int)(qs[j] & 0x0F) | (int)(((qh >> j) & 1) << 4);
          int v1 = (int)(qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4);
          y[j] = (float)v0 * d + m;
          y[j + 16] = (float)v1 * d + m;
        }
        break;
      }
      case GGML_T_Q8_0: {
        const float d = rd_half(p);
        const int8_t *qs = (const int8_t *)(p + 2);
        for (int j = 0; j < 32; j++) y[j] = (float)qs[j] * d;
        break;
      }
      case GGML_T_Q8_1: {
        const float d = rd_half(p);
        const int8_t *qs = (const int8_t *)(p + 4); /* d + s(2) 之后 */
        for (int j = 0; j < 32; j++) y[j] = (float)qs[j] * d;
        break;
      }
      case GGML_T_Q2_K: {
        const float d = rd_half(p + 80), min = rd_half(p + 82);
        const uint8_t *scales = p;
        const uint8_t *q = p + 16;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
          int shift = 0;
          for (int j = 0; j < 4; j++) {
            uint8_t sc = scales[is++];
            float dl = d * (float)(sc & 0x0F), ml = min * (float)(sc >> 4);
            for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l] >> shift) & 3) - ml;
            sc = scales[is++];
            dl = d * (float)(sc & 0x0F);
            ml = min * (float)(sc >> 4);
            for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l + 16] >> shift) & 3) - ml;
            shift += 2;
          }
          q += 32;
        }
        break;
      }
      case GGML_T_Q3_K: {
        const float d_all = rd_half(p + 108);
        const uint8_t *hm = p;
        const uint8_t *q = p + 32;
        uint32_t aux[4];
        memcpy(aux, p + 96, 12); /* scales[12] */
        {
          const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0F0F0F0Fu;
          uint32_t tmp = aux[2];
          aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
          aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
          aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
          aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        }
        const int8_t *scales = (const int8_t *)aux;
        int is = 0;
        uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
          int shift = 0;
          for (int j = 0; j < 4; j++) {
            float dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++) {
              int v = (int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
              *y++ = dl * (float)v;
            }
            dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++) {
              int v = (int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
              *y++ = dl * (float)v;
            }
            shift += 2;
            m = (uint8_t)(m << 1);
          }
          q += 32;
        }
        break;
      }
      case GGML_T_Q4_K: {
        const float d = rd_half(p + 0), min = rd_half(p + 2);
        const uint8_t *scales = p + 4; /* 12 bytes */
        const uint8_t *q = p + 16;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
          uint8_t sc = 0, m = 0;
          scale_min_k4(is + 0, scales, &sc, &m);
          const float d1 = d * (float)sc, m1 = min * (float)m;
          scale_min_k4(is + 1, scales, &sc, &m);
          const float d2 = d * (float)sc, m2 = min * (float)m;
          for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0x0F) - m1;
          for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >> 4) - m2;
          q += 32;
          is += 2;
        }
        break;
      }
      case GGML_T_Q5_K: {
        const float d = rd_half(p + 0), min = rd_half(p + 2);
        const uint8_t *scales = p + 4; /* 12 */
        const uint8_t *qh = p + 16;    /* 32 */
        const uint8_t *ql = p + 48;    /* 128 */
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
          uint8_t sc = 0, m = 0;
          scale_min_k4(is + 0, scales, &sc, &m);
          const float d1 = d * (float)sc, m1 = min * (float)m;
          scale_min_k4(is + 1, scales, &sc, &m);
          const float d2 = d * (float)sc, m2 = min * (float)m;
          for (int l = 0; l < 32; l++) {
            int v = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
            *y++ = d1 * (float)v - m1;
          }
          for (int l = 0; l < 32; l++) {
            int v = (int)(ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            *y++ = d2 * (float)v - m2;
          }
          ql += 32;
          is += 2;
          u1 = (uint8_t)(u1 << 2);
          u2 = (uint8_t)(u2 << 2);
        }
        break;
      }
      case GGML_T_Q6_K: {
        const float d = rd_half(p + 208);
        const uint8_t *ql = p;       /* 128 */
        const uint8_t *qh = p + 128; /* 64 */
        const int8_t *sc = (const int8_t *)(p + 192); /* 16 */
        for (int n = 0; n < 256; n += 128) {
          for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
          }
          y += 128;
          ql += 64;
          qh += 32;
          sc += 8;
        }
        break;
      }
      case GGML_T_Q8_K: {
        /* block_q8_K = { float d; int8_t qs[256]; int16_t bsums[16]; } */
        float d;
        memcpy(&d, p, 4);
        const int8_t *qs = (const int8_t *)(p + 4);
        for (int j = 0; j < 256; j++) y[j] = d * (float)qs[j];
        break;
      }
      default:
        return -1;
    }
  }
  return 0;
}

/* ---------- 解析 ---------- */

typedef struct {
  const uint8_t *p;
  const uint8_t *end;
  int err;
} cur_t;

static uint32_t rd_u32(cur_t *c) {
  uint32_t v = 0;
  if (c->err || c->p + 4 > c->end) { c->err = 1; return 0; }
  memcpy(&v, c->p, 4);
  c->p += 4;
  return v;
}

static uint64_t rd_u64(cur_t *c) {
  uint64_t v = 0;
  if (c->err || c->p + 8 > c->end) { c->err = 1; return 0; }
  memcpy(&v, c->p, 8);
  c->p += 8;
  return v;
}

static const char *rd_str(cur_t *c, uint64_t *out_len) {
  uint64_t n = rd_u64(c);
  if (c->err || n > (uint64_t)(c->end - c->p)) {
    c->err = 1;
    return NULL;
  }
  const char *s = (const char *)c->p;
  c->p += n;
  if (out_len) *out_len = n;
  return s;
}

/* 跳过（或记录）一个值；返回值的起始指针。depth 用于数组嵌套防护。*/
static const uint8_t *rd_value(cur_t *c, uint32_t type, uint64_t *count,
                               uint32_t *elem_type, int depth);

static const uint8_t *rd_value(cur_t *c, uint32_t type, uint64_t *count,
                               uint32_t *elem_type, int depth) {
  const uint8_t *start = c->p;
  if (depth > 4) { c->err = 1; return NULL; }
  if (count) *count = 1;
  if (elem_type) *elem_type = type;
  switch (type) {
    case GK_U8: case GK_I8: case GK_BOOL: c->p += 1; break;
    case GK_U16: case GK_I16: c->p += 2; break;
    case GK_U32: case GK_I32: case GK_F32: c->p += 4; break;
    case GK_U64: case GK_I64: case GK_F64: c->p += 8; break;
    case GK_STR: rd_str(c, NULL); break;
    case GK_ARR: {
      uint32_t et = rd_u32(c);
      uint64_t n = rd_u64(c);
      if (count) *count = n;
      if (elem_type) *elem_type = et;
      if (c->err) return NULL;
      for (uint64_t i = 0; i < n && !c->err; i++) rd_value(c, et, NULL, NULL, depth + 1);
      break;
    }
    default: c->err = 1; break;
  }
  if (c->err || c->p > c->end) { c->err = 1; return NULL; }
  return start;
}

/* 元数据取值 */
static const gg_kv *kv_find(const sgl_gguf *g, const char *key) {
  for (int i = 0; i < g->n_kv; i++)
    if (!strcmp(g->kv[i].key, key)) return &g->kv[i];
  return NULL;
}

static int kv_get_i64(const sgl_gguf *g, const char *key, int64_t *out) {
  const gg_kv *k = kv_find(g, key);
  if (!k) return 0;
  switch (k->type) {
    case GK_U8: *out = (int64_t)k->vals[0]; return 1;
    case GK_I8: *out = (int64_t)(int8_t)k->vals[0]; return 1;
    case GK_U16: { uint16_t v; memcpy(&v, k->vals, 2); *out = v; return 1; }
    case GK_I16: { int16_t v; memcpy(&v, k->vals, 2); *out = v; return 1; }
    case GK_U32: { uint32_t v; memcpy(&v, k->vals, 4); *out = v; return 1; }
    case GK_I32: { int32_t v; memcpy(&v, k->vals, 4); *out = v; return 1; }
    case GK_U64: { uint64_t v; memcpy(&v, k->vals, 8); *out = (int64_t)v; return 1; }
    case GK_I64: { int64_t v; memcpy(&v, k->vals, 8); *out = v; return 1; }
    case GK_BOOL: *out = k->vals[0] ? 1 : 0; return 1;
    default: return 0;
  }
}

static double kv_get_f64(const sgl_gguf *g, const char *key, double dflt) {
  const gg_kv *k = kv_find(g, key);
  if (!k) return dflt;
  if (k->type == GK_F32) {
    float v;
    memcpy(&v, k->vals, 4);
    return (double)v;
  }
  if (k->type == GK_F64) {
    double v;
    memcpy(&v, k->vals, 8);
    return v;
  }
  int64_t iv = 0;
  if (kv_get_i64(g, key, &iv)) return (double)iv;
  return dflt;
}

static const char *kv_get_str(const sgl_gguf *g, const char *key) {
  const gg_kv *k = kv_find(g, key);
  if (!k || k->type != GK_STR || !k->str) return NULL;
  return k->str;
}

static int kv_get_int(const sgl_gguf *g, const char *key, int dflt) {
  int64_t v = 0;
  if (!kv_get_i64(g, key, &v)) return dflt;
  return (int)v;
}

/* ---------- 名字映射 ---------- */

/* GGUF: blk.N.attn_q.weight / token_embd.weight ...  HF: model.layers.N.self_attn.q_proj.weight ... */
int sgl_gguf_name_to_hf(const char *gname, char *out, size_t cap) {
  if (!gname || !out || cap == 0) return 0;
  if (!strcmp(gname, "token_embd.weight")) { snprintf(out, cap, "model.embed_tokens.weight"); return 1; }
  if (!strcmp(gname, "output_norm.weight")) { snprintf(out, cap, "model.norm.weight"); return 1; }
  if (!strcmp(gname, "output.weight")) { snprintf(out, cap, "lm_head.weight"); return 1; }

  /* blk.%d.<rest> */
  if (!strncmp(gname, "blk.", 4)) {
    char *endp = NULL;
    long idx = strtol(gname + 4, &endp, 10);
    if (!endp || *endp != '.') return 0;
    const char *rest = endp + 1;
    static const struct {
      const char *gguf;
      const char *hf;
    } map[] = {
        {"attn_norm.weight", "input_layernorm.weight"},
        {"attn_norm.bias", "input_layernorm.bias"},
        {"attn_q.weight", "self_attn.q_proj.weight"},
        {"attn_q.bias", "self_attn.q_proj.bias"},
        {"attn_k.weight", "self_attn.k_proj.weight"},
        {"attn_k.bias", "self_attn.k_proj.bias"},
        {"attn_v.weight", "self_attn.v_proj.weight"},
        {"attn_v.bias", "self_attn.v_proj.bias"},
        {"attn_output.weight", "self_attn.o_proj.weight"},
        {"attn_output.bias", "self_attn.o_proj.bias"},
        {"attn_q_norm.weight", "self_attn.q_norm.weight"},
        {"attn_k_norm.weight", "self_attn.k_norm.weight"},
        {"ffn_norm.weight", "post_attention_layernorm.weight"},
        {"ffn_norm.bias", "post_attention_layernorm.bias"},
        {"ffn_gate.weight", "mlp.gate_proj.weight"},
        {"ffn_up.weight", "mlp.up_proj.weight"},
        {"ffn_down.weight", "mlp.down_proj.weight"},
        {"ffn_gate_inp.weight", "mlp.gate.weight"},
        {"ffn_gate_exps.weight", "mlp.experts.gate_up_proj"},
        {"ffn_down_exps.weight", "mlp.experts.down_proj"},
    };
    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
      if (!strcmp(rest, map[i].gguf)) {
        snprintf(out, cap, "model.layers.%ld.%s", idx, map[i].hf);
        return 1;
      }
    }
    return 0;
  }
  /* 其余（如 rope_freqs.weight）不映射 */
  return 0;
}

int sgl_gguf_name_from_hf(const char *hf_name, char *out, size_t cap) {
  if (!hf_name || !out || cap == 0) return 0;
  if (!strcmp(hf_name, "model.embed_tokens.weight")) { snprintf(out, cap, "token_embd.weight"); return 1; }
  if (!strcmp(hf_name, "model.norm.weight")) { snprintf(out, cap, "output_norm.weight"); return 1; }
  if (!strcmp(hf_name, "lm_head.weight")) { snprintf(out, cap, "output.weight"); return 1; }
  if (!strncmp(hf_name, "model.layers.", 13)) {
    const char *p = hf_name + 13;
    char *endp = NULL;
    long idx = strtol(p, &endp, 10);
    if (!endp || *endp != '.') return 0;
    const char *rest = endp + 1;
    static const struct {
      const char *hf;
      const char *gguf;
    } map[] = {
        {"input_layernorm.weight", "attn_norm.weight"},
        {"input_layernorm.bias", "attn_norm.bias"},
        {"self_attn.q_proj.weight", "attn_q.weight"},
        {"self_attn.q_proj.bias", "attn_q.bias"},
        {"self_attn.k_proj.weight", "attn_k.weight"},
        {"self_attn.k_proj.bias", "attn_k.bias"},
        {"self_attn.v_proj.weight", "attn_v.weight"},
        {"self_attn.v_proj.bias", "attn_v.bias"},
        {"self_attn.o_proj.weight", "attn_output.weight"},
        {"self_attn.o_proj.bias", "attn_output.bias"},
        {"self_attn.q_norm.weight", "attn_q_norm.weight"},
        {"self_attn.k_norm.weight", "attn_k_norm.weight"},
        {"post_attention_layernorm.weight", "ffn_norm.weight"},
        {"post_attention_layernorm.bias", "ffn_norm.bias"},
        {"mlp.gate_proj.weight", "ffn_gate.weight"},
        {"mlp.up_proj.weight", "ffn_up.weight"},
        {"mlp.down_proj.weight", "ffn_down.weight"},
        {"mlp.gate.weight", "ffn_gate_inp.weight"},
    };
    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
      if (!strcmp(rest, map[i].hf)) {
        snprintf(out, cap, "blk.%ld.%s", idx, map[i].gguf);
        return 1;
      }
    }
    return 0;
  }
  return 0;
}

/* ---------- 打开/关闭 ---------- */

static void gg_free(sgl_gguf *g) {
  if (!g) return;
  for (int i = 0; i < g->n_hf_names; i++) sgl_xfree(g->hf_names[i]);
  sgl_xfree(g->hf_names);
  for (int i = 0; i < g->n_tensors; i++) sgl_xfree(g->tensors[i].name_owned);
  sgl_xfree(g->tensors);
  for (int i = 0; i < g->n_kv; i++) {
    sgl_xfree(g->kv[i].key_owned);
    sgl_xfree(g->kv[i].str_owned);
  }
  sgl_xfree(g->kv);
  sgl_xfree(g->path);
  if (g->map.data) sgl_mmap_close(&g->map);
  sgl_xfree(g);
}

sgl_gguf *sgl_gguf_open(const char *path) {
  sgl_mmap map;
  memset(&map, 0, sizeof(map));
  if (sgl_mmap_open(&map, path) != 0) {
    SGL_ERROR("gguf: open %s failed", path);
    return NULL;
  }
  if (map.size < 24) {
    SGL_ERROR("gguf: %s too small", path);
    sgl_mmap_close(&map);
    return NULL;
  }
  cur_t c = {(const uint8_t *)map.data, (const uint8_t *)map.data + map.size, 0};
  uint32_t magic = rd_u32(&c);
  if (magic != 0x46554747u) { /* "GGUF" 小端 */
    SGL_ERROR("gguf: %s: bad magic 0x%08x (not a GGUF file)", path, magic);
    sgl_mmap_close(&map);
    return NULL;
  }
  sgl_gguf *g = (sgl_gguf *)sgl_xcalloc(1, sizeof(*g));
  g->map = map;
  g->version = rd_u32(&c);
  if (g->version < 2 || g->version > 3) {
    SGL_ERROR("gguf: %s: version %u unsupported (need 2 or 3)", path, g->version);
    gg_free(g);
    /* map 已被 g 接管，gg_free 会关 */
    return NULL;
  }
  uint64_t n_tensors = rd_u64(&c);
  uint64_t n_kv = rd_u64(&c);
  if (c.err || n_kv > 1000000u || n_tensors > 1000000u) {
    SGL_ERROR("gguf: %s: header counts insane (kv=%llu tensors=%llu)", path,
              (unsigned long long)n_kv, (unsigned long long)n_tensors);
    gg_free(g);
    return NULL;
  }
  g->kv = (gg_kv *)sgl_xcalloc(n_kv ? (size_t)n_kv : 1, sizeof(gg_kv));
  g->n_kv = 0;
  for (uint64_t i = 0; i < n_kv; i++) {
    uint64_t klen = 0;
    const char *key = rd_str(&c, &klen);
    if (c.err || !key) break;
    uint32_t type = rd_u32(&c);
    uint64_t cnt = 1;
    uint32_t et = 0;
    const uint8_t *vals = rd_value(&c, type, &cnt, &et, 0);
    if (c.err || !vals) break;
    if (type == GK_ARR) vals += 12; /* 跳过 elem_type(4) + count(8)，指向首元素 */
    gg_kv *k = &g->kv[g->n_kv++];
    k->key_owned = xstrndup_(key, (size_t)klen);
    k->key = k->key_owned;
    k->type = type;
    k->n = cnt;
    k->elem_type = et;
    k->vals = vals;
    if (type == GK_STR) {
      /* 字符串值：vals 指向 {u64 len | bytes} */
      uint64_t sl = 0;
      memcpy(&sl, vals, 8);
      k->str_owned = xstrndup_((const char *)(vals + 8), (size_t)sl);
      k->str = k->str_owned;
      k->str_len = sl;
    }
  }
  if (c.err || g->n_kv != (int)n_kv) {
    SGL_ERROR("gguf: %s: metadata truncated (%d/%llu read)", path, g->n_kv,
              (unsigned long long)n_kv);
    gg_free(g);
    return NULL;
  }
  g->tensors = (gg_tensor *)sgl_xcalloc(n_tensors ? (size_t)n_tensors : 1, sizeof(gg_tensor));
  g->n_tensors = 0;
  for (uint64_t i = 0; i < n_tensors; i++) {
    uint64_t nlen = 0;
    const char *name = rd_str(&c, &nlen);
    if (c.err || !name) break;
    uint32_t nd = rd_u32(&c);
    if (c.err || nd < 1 || nd > 4) {
      SGL_ERROR("gguf: %s: tensor '%s' n_dims=%u unsupported", path, name, nd);
      c.err = 1;
      break;
    }
    gg_tensor *t = &g->tensors[g->n_tensors++];
    t->name_owned = xstrndup_(name, (size_t)nlen);
    t->name = t->name_owned;
    t->n_dims = (int)nd;
    t->numel = 1;
    for (uint32_t d = 0; d < nd; d++) {
      t->ne[d] = (int64_t)rd_u64(&c);
      t->numel *= t->ne[d];
    }
    t->type = (int)rd_u32(&c);
    t->offset = rd_u64(&c);
    if (c.err) break;
  }
  if (c.err || g->n_tensors != (int)n_tensors) {
    SGL_ERROR("gguf: %s: tensor info truncated", path);
    gg_free(g);
    return NULL;
  }
  /* 对齐到 general.alignment（默认 32）*/
  int align = kv_get_int(g, "general.alignment", 32);
  if (align <= 0 || (align & (align - 1))) align = 32;
  size_t off = (size_t)(c.p - (const uint8_t *)g->map.data);
  size_t pad = (size_t)align - (off % (size_t)align);
  if (pad == (size_t)align) pad = 0;
  if (off + pad > g->map.size) {
    SGL_ERROR("gguf: %s: data offset past EOF", path);
    gg_free(g);
    return NULL;
  }
  g->data = (const char *)g->map.data + off + pad;
  g->data_len = g->map.size - off - pad;
  g->path = xstrdup_(path);

  const char *arch = kv_get_str(g, "general.architecture");
  SGL_INFO("gguf: %s v%u, arch=%s, %d tensors, %d kv, data %.1f MB", path, g->version,
           arch ? arch : "?", g->n_tensors, g->n_kv, g->data_len / 1048576.0);
  return g;
}

sgl_gguf *sgl_gguf_open_dir(const char *dir) {
  /* 目录列举：优先不带 "-of-" 的单个文件（split 分片时取第一片）*/
  char best[1400] = {0};
  char any[1400] = {0};
#ifdef _WIN32
  {
    WIN32_FIND_DATAA fd;
    char pat[1500];
    snprintf(pat, sizeof(pat), "%s\\*.gguf", dir);
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h != INVALID_HANDLE_VALUE) {
      do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        char cand[1600];
        snprintf(cand, sizeof(cand), "%s/%s", dir, fd.cFileName);
        if (!any[0]) snprintf(any, sizeof(any), "%s", cand);
        if (!strstr(fd.cFileName, "-of-") &&
            (!best[0] || strcmp(cand, best) < 0))
          snprintf(best, sizeof(best), "%s", cand);
      } while (FindNextFileA(h, &fd));
      FindClose(h);
    }
  }
#else
  {
    DIR *dp = opendir(dir);
    if (dp) {
      struct dirent *de;
      while ((de = readdir(dp)) != NULL) {
        size_t l = strlen(de->d_name);
        if (l < 6 || strcmp(de->d_name + l - 5, ".gguf") != 0) continue;
        char cand[1600];
        snprintf(cand, sizeof(cand), "%s/%s", dir, de->d_name);
        if (!any[0]) snprintf(any, sizeof(any), "%s", cand);
        if (!strstr(de->d_name, "-of-") && (!best[0] || strcmp(cand, best) < 0))
          snprintf(best, sizeof(best), "%s", cand);
      }
      closedir(dp);
    }
  }
#endif
  if (!best[0]) snprintf(best, sizeof(best), "%s", any);
  if (!best[0]) {
    char cand[1600];
    snprintf(cand, sizeof(cand), "%s/model.gguf", dir);
    FILE *f = NULL;
    if (fopen_s(&f, cand, "rb") == 0 && f) {
      fclose(f);
      snprintf(best, sizeof(best), "%s", cand);
    }
  }
  if (!best[0]) return NULL;
  return sgl_gguf_open(best);
}

void sgl_gguf_close(sgl_gguf *g) { gg_free(g); }

/* ---------- 取张量 ---------- */

static const gg_tensor *find_tensor(const sgl_gguf *g, const char *name) {
  for (int i = 0; i < g->n_tensors; i++)
    if (!strcmp(g->tensors[i].name, name)) return &g->tensors[i];
  return NULL;
}

/* 按 GGUF 名取；找不到返回 NULL */
static const gg_tensor *find_by_hf(const sgl_gguf *g, const char *hf_name, char *ggname,
                                   size_t cap) {
  if (!sgl_gguf_name_from_hf(hf_name, ggname, cap)) return NULL;
  return find_tensor(g, ggname);
}

int sgl_gguf_has(sgl_gguf *g, const char *hf_name) {
  char nm[256];
  if (!g || !hf_name) return 0;
  return find_by_hf(g, hf_name, nm, sizeof(nm)) != NULL;
}

const char *sgl_gguf_path(const sgl_gguf *g) { return g ? g->path : NULL; }

static int ends_with_(const char *s, const char *suf) {
  size_t a = strlen(s), b = strlen(suf);
  return a >= b && !strcmp(s + a - b, suf);
}

/* q_proj/k_proj 的 head 置换还原（详见头文件注释）。
 * 置换是每行独立重排（行内数据不动），所以原地加一个临时缓冲就行。*/
int sgl_gguf_unpermute_qk(sgl_gguf *g, const char *hf_name, float *buf, int ndim,
                          const int64_t *shape) {
  if (!g || !hf_name || !buf || ndim != 2 || !shape) return 0;
  if (getenv("SGL_GGUF_NO_PERMUTE")) return 0;
  int is_q = ends_with_(hf_name, "self_attn.q_proj.weight");
  int is_k = ends_with_(hf_name, "self_attn.k_proj.weight");
  if (!is_q && !is_k) return 0;
  const char *arch = sgl_gguf_meta_str(g, "general.architecture", "llama");
  char kc[128];
  snprintf(kc, sizeof(kc), "%s.attention.head_count%s", arch, is_k ? "_kv" : "");
  int n_head = sgl_gguf_meta_i32(g, kc, 0);
  if (n_head <= 0) return 0;
  int64_t out = shape[0], in = shape[1];
  if (out <= 0 || in <= 0 || out % n_head != 0) return 0;
  int64_t hd = out / n_head;
  if (hd % 2 != 0) return 0;
  int64_t half = hd / 2;
  /* 校验：真的像被置换过吗？（否则别改）
   * 判据：permute 后行 r'=2k+half 的内容来自原行 half*half+k。无法从数值判断，
   * 所以只能靠约定；但可以检查尺寸一致性。*/
  float *tmp = (float *)sgl_xmalloc((size_t)(out * in) * sizeof(float));
  for (int64_t h = 0; h < n_head; h++) {
    for (int64_t rem = 0; rem < hd; rem++) {
      int64_t kk = rem / 2;          /* 原行里的 k */
      int64_t hf = rem % 2;          /* 原行里的 half */
      int64_t r_src = h * hd + rem;               /* gguf 行 */
      int64_t r_dst = h * hd + hf * half + kk;    /* HF 行 */
      memcpy(tmp + (size_t)(r_dst * in), buf + (size_t)(r_src * in),
             (size_t)in * sizeof(float));
    }
  }
  memcpy(buf, tmp, (size_t)(out * in) * sizeof(float));
  sgl_xfree(tmp);
  return 1;
}

/* ---------- 元数据访问 ---------- */
int sgl_gguf_meta_i32(sgl_gguf *g, const char *key, int def) {
  const gg_kv *k = g ? kv_find(g, key) : NULL;
  if (!k || k->type == GK_ARR) return def;
  int32_t v = 0;
  switch (k->type) {
    case GK_U8: v = (int32_t)(*(const uint8_t *)k->vals); break;
    case GK_I8: v = (int32_t)(*(const int8_t *)k->vals); break;
    case GK_U16: { uint16_t x; memcpy(&x, k->vals, 2); v = (int32_t)x; break; }
    case GK_I16: { int16_t x; memcpy(&x, k->vals, 2); v = (int32_t)x; break; }
    case GK_U32: { uint32_t x; memcpy(&x, k->vals, 4); v = (int32_t)x; break; }
    case GK_I32: memcpy(&v, k->vals, 4); break;
    case GK_U64: { uint64_t x; memcpy(&x, k->vals, 8); v = (int32_t)x; break; }
    case GK_I64: { int64_t x; memcpy(&x, k->vals, 8); v = (int32_t)x; break; }
    case GK_BOOL: v = (*(const uint8_t *)k->vals) ? 1 : 0; break;
    case GK_F32: { float f; memcpy(&f, k->vals, 4); v = (int32_t)f; break; }
    default: return def;
  }
  return (int)v;
}

float sgl_gguf_meta_f32(sgl_gguf *g, const char *key, float def) {
  const gg_kv *k = g ? kv_find(g, key) : NULL;
  if (!k || k->type == GK_ARR) return def;
  if (k->type == GK_F32) { float f; memcpy(&f, k->vals, 4); return f; }
  if (k->type == GK_F64) { double d; memcpy(&d, k->vals, 8); return (float)d; }
  return (float)sgl_gguf_meta_i32(g, key, (int)def);
}

const char *sgl_gguf_meta_str(sgl_gguf *g, const char *key, const char *def) {
  const gg_kv *k = g ? kv_find(g, key) : NULL;
  if (!k || k->type != GK_STR || !k->str) return def;
  return k->str;
}

int sgl_gguf_meta_str_array_n(sgl_gguf *g, const char *key) {
  const gg_kv *k = g ? kv_find(g, key) : NULL;
  if (!k || k->type != GK_ARR || k->elem_type != GK_STR) return -1;
  return (int)k->n;
}

int sgl_gguf_meta_str_array_at(sgl_gguf *g, const char *key, int i, const char **s,
                               size_t *len) {
  const gg_kv *k = g ? kv_find(g, key) : NULL;
  if (!k || k->type != GK_ARR || k->elem_type != GK_STR || i < 0) return -1;
  if ((uint64_t)i >= k->n) return -1;
  /* 字符串数组是变长排列：只能从前往后走 */
  const uint8_t *p = k->vals;
  const uint8_t *end = g->map.data ? (const uint8_t *)g->map.data + g->map.size : NULL;
  for (int j = 0; j <= i; j++) {
    uint64_t l = 0;
    if (!end || p + 8 > end) return -1;
    memcpy(&l, p, 8);
    p += 8;
    if (!end || p + l > end) return -1;
    if (j == i) {
      if (s) *s = (const char *)p;
      if (len) *len = (size_t)l;
      return 0;
    }
    p += l;
  }
  return -1;
}

int sgl_gguf_meta_i32_array_n(sgl_gguf *g, const char *key) {
  const gg_kv *k = g ? kv_find(g, key) : NULL;
  if (!k || k->type != GK_ARR) return -1;
  if (k->elem_type != GK_I32 && k->elem_type != GK_U32) return -1;
  return (int)k->n;
}

int sgl_gguf_meta_i32_array_at(sgl_gguf *g, const char *key, int i) {
  const gg_kv *k = g ? kv_find(g, key) : NULL;
  if (!k || k->type != GK_ARR || i < 0 || (uint64_t)i >= k->n) return -1;
  if (k->elem_type == GK_I32) {
    int32_t v;
    memcpy(&v, k->vals + (size_t)i * 4, 4);
    return (int)v;
  }
  if (k->elem_type == GK_U32) {
    uint32_t v;
    memcpy(&v, k->vals + (size_t)i * 4, 4);
    return (int)v;
  }
  return -1;
}

int sgl_gguf_tensor_type(sgl_gguf *g, const char *hf_name, const char **type_name) {
  char nm[256];
  if (type_name) *type_name = NULL;
  if (!g || !hf_name) return -1;
  const gg_tensor *t = find_by_hf(g, hf_name, nm, sizeof(nm));
  if (!t) return -1;
  if (type_name) *type_name = sgl_gguf_type_name(t->type);
  return t->type;
}

float *sgl_gguf_get_f32(sgl_gguf *g, const char *hf_name, int *out_ndim,
                        int64_t out_shape[4]) {
  if (!g || !hf_name) return NULL;
  char nm[256];
  const gg_tensor *t = find_by_hf(g, hf_name, nm, sizeof(nm));
  if (!t) return NULL;
  int be = 0, bb = 0;
  block_layout(t->type, &be, &bb);
  if (be == 0) {
    SGL_ERROR("gguf: tensor '%s' (%s) type %d (%s) unsupported by this build",
              t->name, hf_name, t->type, sgl_gguf_type_name(t->type));
    return NULL;
  }
  if (t->ne[0] % be != 0) {
    SGL_ERROR("gguf: tensor '%s' ne[0]=%lld not a multiple of block %d", t->name,
              (long long)t->ne[0], be);
    return NULL;
  }
  size_t need = (size_t)t->numel / (size_t)be * (size_t)bb;
  if (t->offset + need > g->data_len) {
    SGL_ERROR("gguf: tensor '%s' data past EOF (%llu+%llu > %llu)", t->name,
              (unsigned long long)t->offset, (unsigned long long)need,
              (unsigned long long)g->data_len);
    return NULL;
  }
  float *buf = (float *)sgl_xmalloc((size_t)t->numel * sizeof(float));
  if (dequant_tensor(t->type, (const uint8_t *)g->data + t->offset, t->numel, buf) != 0) {
    SGL_ERROR("gguf: tensor '%s' dequant failed", t->name);
    sgl_xfree(buf);
    return NULL;
  }
  if (out_ndim) *out_ndim = t->n_dims;
  if (out_shape) {
    /* GGML 的 ne[0] 是最内维（连续），HF 是行主序 [rows, cols]：
     * 两者数据布局一致（偏移 = ne0*i1 + i0 == col + cols*row），
     * 但形状要逆序报告。2D: GGUF ne=[in,out] → HF [out,in]。*/
    for (int i = 0; i < 4; i++)
      out_shape[i] = i < t->n_dims ? t->ne[t->n_dims - 1 - i] : 1;
  }
  return buf;
}

int sgl_gguf_names(sgl_gguf *g, const char ***names) {
  if (!g) {
    *names = NULL;
    return 0;
  }
  if (!g->hf_names_built) {
    g->hf_names = (char **)sgl_xcalloc((size_t)g->n_tensors, sizeof(char *));
    g->hf_names_built = 1;
    int n = 0;
    for (int i = 0; i < g->n_tensors; i++) {
      char hf[256];
      if (sgl_gguf_name_to_hf(g->tensors[i].name, hf, sizeof(hf))) {
        size_t l = strlen(hf);
        g->hf_names[n] = (char *)sgl_xmalloc(l + 1);
        memcpy(g->hf_names[n], hf, l + 1);
        n++;
      }
    }
    g->n_hf_names = n;
  }
  *names = (const char **)g->hf_names;
  return g->n_hf_names;
}

/* ---------- 元数据 → HF config.json ---------- */

typedef struct {
  char *p;
  size_t n, cap;
  int comma; /* 当前对象里是否已有元素（写分隔符用）*/
} sbuf;

static void sb_init(sbuf *b) {
  b->cap = 512;
  b->n = 0;
  b->comma = 0;
  b->p = (char *)sgl_xmalloc(b->cap);
  b->p[0] = 0;
}

static void sb_add(sbuf *b, const char *s) {
  size_t l = strlen(s);
  if (b->n + l + 1 > b->cap) {
    while (b->n + l + 1 > b->cap) b->cap *= 2;
    b->p = (char *)sgl_xrealloc(b->p, b->cap);
  }
  memcpy(b->p + b->n, s, l + 1);
  b->n += l;
}

static void sb_printf(sbuf *b, const char *fmt, ...) {
  char tmp[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(tmp, sizeof(tmp), fmt, ap);
  va_end(ap);
  sb_add(b, tmp);
}

/* 分隔符 + 一个 kv 对（逗号由 b->comma 决定，嵌套对象里也不会多）*/
static void sb_sep(sbuf *b) {
  if (b->comma) sb_add(b, ", ");
  b->comma = 1;
}

/* 追加 "key": value，自动补逗号 */
static void sb_kv_i(sbuf *b, const char *key, long long v) {
  sb_sep(b);
  sb_printf(b, "\"%s\": %lld", key, v);
}

static void sb_kv_f(sbuf *b, const char *key, double v) {
  sb_sep(b);
  sb_printf(b, "\"%s\": %.9g", key, v);
}

static void sb_kv_s(sbuf *b, const char *key, const char *v) {
  sb_sep(b);
  sb_printf(b, "\"%s\": \"%s\"", key, v);
}

/* 架构名 → HF model_type / architectures */
static void arch_to_hf(const char *arch, char *model_type, size_t mt_cap, char *archs,
                       size_t ar_cap) {
  static const struct {
    const char *gguf;
    const char *mt;
    const char *cls;
  } map[] = {
      {"llama", "llama", "LlamaForCausalLM"},
      {"qwen2", "qwen2", "Qwen2ForCausalLM"},
      {"qwen2moe", "qwen2_moe", "Qwen2MoeForCausalLM"},
      {"qwen3", "qwen3", "Qwen3ForCausalLM"},
      {"qwen3next", "qwen3_next", "Qwen3NextForCausalLM"},
      {"mistral", "mistral", "MistralForCausalLM"},
      {"gemma", "gemma", "GemmaForCausalLM"},
      {"gemma2", "gemma2", "Gemma2ForCausalLM"},
      {"deepseek2", "deepseek_v2", "DeepseekV2ForCausalLM"},
      {"phi3", "phi3", "Phi3ForCausalLM"},
  };
  for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
    if (!strcmp(arch, map[i].gguf)) {
      snprintf(model_type, mt_cap, "%s", map[i].mt);
      snprintf(archs, ar_cap, "%s", map[i].cls);
      return;
    }
  }
  snprintf(model_type, mt_cap, "%s", arch);
  snprintf(archs, ar_cap, "%sForCausalLM", arch);
}

char *sgl_gguf_config_json(sgl_gguf *g, size_t *out_len) {
  if (!g) return NULL;
  const char *arch = kv_get_str(g, "general.architecture");
  if (!arch) arch = "llama";
  char model_type[64], archs[96];
  arch_to_hf(arch, model_type, sizeof(model_type), archs, sizeof(archs));

  char k[128];
  sbuf b;
  sb_init(&b);
  sb_add(&b, "{");

#define KEY(name) (snprintf(k, sizeof(k), "%s.%s", arch, name), k)
  int n_layer = kv_get_int(g, KEY("block_count"), 0);
  int n_embd = kv_get_int(g, KEY("embedding_length"), 0);
  int n_ff = kv_get_int(g, KEY("feed_forward_length"), 0);
  int n_head = kv_get_int(g, KEY("attention.head_count"), 0);
  int n_kv = kv_get_int(g, KEY("attention.head_count_kv"), n_head);
  int n_ctx = kv_get_int(g, KEY("context_length"), 2048);
  int hd = kv_get_int(g, KEY("attention.key_length"), 0);
  int sw = kv_get_int(g, KEY("attention.sliding_window"), 0);
  double eps = kv_get_f64(g, KEY("attention.layer_norm_rms_epsilon"), -1.0);
  double eps_ln = kv_get_f64(g, KEY("attention.layer_norm_epsilon"), -1.0);
  double rope_base = kv_get_f64(g, KEY("rope.freq_base"), 10000.0);
  double softcap = kv_get_f64(g, KEY("final_logit_softcapping"), 0.0);
  int n_experts = kv_get_int(g, KEY("expert_count"), 0);
  int n_experts_used = kv_get_int(g, KEY("expert_used_count"), 0);
  int n_ff_exp = kv_get_int(g, KEY("expert_feed_forward_length"), 0);
#undef KEY

  /* vocab：优先 tokenizer 的 tokens 数组长度，其次 embed 张量 */
  int vocab = 0;
  {
    const gg_kv *tk = kv_find(g, "tokenizer.ggml.tokens");
    if (tk && tk->type == GK_ARR) vocab = (int)tk->n;
    if (vocab <= 0) {
      char nm[256];
      const gg_tensor *t = find_by_hf(g, "model.embed_tokens.weight", nm, sizeof(nm));
      if (t && t->n_dims == 2) vocab = (int)t->ne[1]; /* GGUF: ne=[hidden, vocab] */
    }
  }
  int tie = 1;
  {
    char nm[256];
    if (find_by_hf(g, "lm_head.weight", nm, sizeof(nm))) tie = 0;
  }

  sb_kv_s(&b, "model_type", model_type);
  sb_sep(&b);
  sb_printf(&b, "\"architectures\": [\"%s\"]", archs);
  sb_kv_i(&b, "num_hidden_layers", n_layer);
  sb_kv_i(&b, "hidden_size", n_embd);
  if (n_ff > 0) sb_kv_i(&b, "intermediate_size", n_ff);
  sb_kv_i(&b, "num_attention_heads", n_head);
  sb_kv_i(&b, "num_key_value_heads", n_kv);
  sb_kv_i(&b, "max_position_embeddings", n_ctx);
  if (vocab > 0) sb_kv_i(&b, "vocab_size", vocab);
  if (hd > 0) sb_kv_i(&b, "head_dim", hd);
  if (sw > 0) sb_kv_i(&b, "sliding_window", sw);
  if (eps > 0) sb_kv_f(&b, "rms_norm_eps", eps);
  else if (eps_ln > 0) sb_kv_f(&b, "layer_norm_eps", eps_ln);
  sb_kv_f(&b, "rope_theta", rope_base);
  if (softcap != 0.0) sb_kv_f(&b, "hidden_softcap", softcap);
  if (n_experts > 0) {
    sb_kv_i(&b, "num_experts", n_experts);
    if (n_experts_used > 0) sb_kv_i(&b, "num_experts_per_tok", n_experts_used);
    if (n_ff_exp > 0) sb_kv_i(&b, "moe_intermediate_size", n_ff_exp);
  }
  if (!tie) sb_kv_i(&b, "tie_word_embeddings", 0);
  else sb_kv_i(&b, "tie_word_embeddings", 1);

  /* rope scaling（linear / yarn）*/
  {
    char ks[128];
    snprintf(ks, sizeof(ks), "%s.rope.scaling.type", arch);
    const char *rt = kv_get_str(g, ks);
    if (rt && strcmp(rt, "none")) {
      char kf[128], ko[128];
      snprintf(kf, sizeof(kf), "%s.rope.scaling.factor", arch);
      snprintf(ko, sizeof(ko), "%s.rope.scaling.original_context_length", arch);
      double factor = kv_get_f64(g, kf, 1.0);
      int orig = kv_get_int(g, ko, 0);
      sb_sep(&b);
      sb_add(&b, "\"rope_scaling\": {");
      b.comma = 0;
      sb_kv_s(&b, "rope_type", rt);
      sb_kv_f(&b, "factor", factor);
      if (orig > 0) sb_kv_i(&b, "original_max_position_embeddings", orig);
      sb_add(&b, "}");
      b.comma = 1;
    }
  }
  sb_add(&b, "}");
  if (out_len) *out_len = b.n;
  return b.p;
}

char *sgl_config_read_text(const char *dir, size_t *out_len) {
  char p[1400];
  snprintf(p, sizeof(p), "%s/config.json", dir);
  FILE *f = NULL;
  if (fopen_s(&f, p, "rb") == 0 && f) {
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
      fclose(f);
      return NULL;
    }
    char *buf = (char *)sgl_xmalloc((size_t)n + 1);
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = 0;
    if (out_len) *out_len = got;
    return buf;
  }
  /* 没有 config.json：从同目录的 .gguf 元数据合成 */
  sgl_gguf *g = sgl_gguf_open_dir(dir);
  if (!g) return NULL;
  char *js = sgl_gguf_config_json(g, out_len);
  sgl_gguf_close(g);
  if (js) SGL_INFO("config: synthesized from GGUF metadata (%s)", dir);
  return js;
}