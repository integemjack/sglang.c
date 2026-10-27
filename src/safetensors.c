/* sgl safetensors 实现
 * 文件格式：<u64 header_len><header JSON><数据区>
 * header: { "tensor": {"dtype":"F32","shape":[..],"data_offsets":[s,e]}, ... }
 *
 * P0-1（2026-10-27）：支持 HF 分片权重
 *   - sgl_st_open(path)      单文件（原语义不变）
 *   - sgl_st_open_dir(dir)   目录：优先 model.safetensors.index.json（多分片），
 *                            否则 model.safetensors，再否则 HF 的其他约定名
 *   index.json 的 weight_map 给出「张量名 -> 分片文件名」，按需在对应分片里查，
 *   不再要求整个模型塞进一个 model.safetensors。
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/gguf.h"
#include "sgl/json.h"
#include "sgl/log.h"
#include "sgl/port.h"
#include "sgl/qweight.h"

#define SGL_ST_MAX_SHARDS 128

static char *xstrdup_(const char *s) {
  size_t n = strlen(s);
  char *d = (char *)sgl_xmalloc(n + 1);
  memcpy(d, s, n + 1);
  return d;
}

typedef struct {
  sgl_mmap map;
  sgl_json *header; /* 解析后的 header */
  const char *data; /* 数据区起始 */
  size_t data_len;
  void **owned; /* F16/BF16->F32 转换缓冲（close 时释放）*/
  int n_owned, cap_owned;
  char *name; /* 分片文件名（单文件模式为 NULL）*/
} st_shard;

struct sgl_st_ctx {
  st_shard sh[SGL_ST_MAX_SHARDS];
  int n_sh;
  sgl_json *root; /* index.json 解析结果（整个文档）*/
  sgl_json *wmap; /* root.weight_map；NULL = 单文件模式 */
  char *dir;
  int from_index;
  /* GGUF 权重源（目录里只有 .gguf 时）：sgl_st_get 走反量化路径。
   * 上层 llama/deepseek 不用区分权重格式。*/
  sgl_gguf *gg;
  /* GGUF 反量化得到的 f32 缓冲，归 ctx 生命周期（f32 已经是 owned）*/
  void **gowned;
  int n_gowned, cap_gowned;
  /* AWQ/GPTQ：config.json 的 quantization_config；method==SGL_Q_NONE 表示无 */
  sgl_qcfg q;
};

/* IEEE 754 half → float（位运算，无依赖）
 * 次正规项必须归一化：值 = frac * 2^-24，直接 frac<<13 会变成 f32 的
 * 次正规（~1e-38），差 2^-125 倍。*/ 
static float f16_to_f32(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  int32_t e = (int32_t)((h >> 10) & 0x1F);
  uint32_t m = (uint32_t)(h & 0x3FFu);
  uint32_t f;
  if (e == 0) {
    if (m == 0) {
      f = s;
    } else {
      e = 1;
      while (!(m & 0x400u)) {
        m <<= 1;
        e--;
      }
      m &= 0x3FFu;
      f = s | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
    }
  } else if (e == 31) {
    f = s | 0x7F800000u | (m << 13); /* inf/nan */
  } else {
    f = s | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
  }
  float out;
  memcpy(&out, &f, 4);
  return out;
}

/* 无常量除零的 NaN / ±inf（MSVC 把 0.0f/0.0f 当编译错误 C2124）*/
static float f32_from_bits(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static float f32_nan(void) { return f32_from_bits(0x7FC00000u); }
static float f32_inf(int sign) { return f32_from_bits(sign ? 0xFF800000u : 0x7F800000u); }

/* FP8 → f32：E4M3FN（DeepSeek/Llama fp8 权重）与 E5M2。
 * E4M3FN：1 符号 + 4 指数（偏置 7）+ 3 尾数；无 inf，0x7F/0xFF 是 NaN。
 * E5M2：1 + 5（偏置 15）+ 2 尾数；有 inf。 */
static float fp8_to_f32(unsigned char b, int e5) {
  int sign = (b >> 7) & 1;
  float v;
  if (!e5) {
    int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0xF && m == 0x7) return f32_nan(); /* NaN（0x7F/0xFF）；E4M3FN 无 inf */
    if (e == 0)
      v = ((float)m * (1.0f / 8.0f)) * ldexpf(1.0f, -6);  /* subnormal: m/8 * 2^-6 */
    else
      v = (1.0f + (float)m * (1.0f / 8.0f)) * ldexpf(1.0f, e - 7);
  } else {
    int e = (b >> 2) & 0x1F, m = b & 0x3;
    if (e == 0x1F) return m ? f32_nan() : f32_inf(sign);
    if (e == 0)
      v = ((float)m * (1.0f / 4.0f)) * ldexpf(1.0f, -14); /* subnormal: m/4 * 2^-14 */
    else
      v = (1.0f + (float)m * (1.0f / 4.0f)) * ldexpf(1.0f, e - 15);
  }
  return sign ? -v : v;
}

/* bfloat16 → float：即 f32 的高 16 位，低位补零 */
static float bf16_to_f32(uint16_t h) {
  uint32_t f = (uint32_t)h << 16;
  float out;
  memcpy(&out, &f, 4);
  return out;
}

/* ---- 单个分片 ---- */

static int shard_open(st_shard *s, const char *path) {
  memset(s, 0, sizeof(*s));
  if (sgl_mmap_open(&s->map, path) != 0) {
    SGL_ERROR("open safetensors %s failed", path);
    return -1;
  }
  if (s->map.size < 8) {
    SGL_ERROR("safetensors too small: %s", path);
    goto fail;
  }
  uint64_t hlen;
  memcpy(&hlen, s->map.data, 8);
  if ((uint64_t)8 + hlen > s->map.size) {
    SGL_ERROR("safetensors header overflow: %s", path);
    goto fail;
  }
  s->header = sgl_json_parse((const char *)s->map.data + 8, (size_t)hlen);
  if (!s->header || s->header->type != SGL_JSON_OBJECT) {
    SGL_ERROR("safetensors header JSON invalid: %s", path);
    goto fail;
  }
  s->data = (const char *)s->map.data + 8 + hlen;
  s->data_len = s->map.size - 8 - hlen;
  SGL_INFO("safetensors shard: %s, %d tensors, data %.1f MB", path, s->header->count,
           s->data_len / 1048576.0);
  return 0;

fail:
  if (s->header) sgl_json_free(s->header);
  s->header = NULL;
  sgl_mmap_close(&s->map);
  return -1;
}

static void shard_close(st_shard *s) {
  for (int i = 0; i < s->n_owned; i++) sgl_xfree(s->owned[i]);
  sgl_xfree(s->owned);
  s->owned = NULL;
  s->n_owned = s->cap_owned = 0;
  if (s->header) sgl_json_free(s->header);
  s->header = NULL;
  if (s->map.data) sgl_mmap_close(&s->map);
  sgl_xfree(s->name);
  s->name = NULL;
}

/* ---- 裸张量访问（AWQ/GPTQ 的 qweight 是 int32，shard_get 只认浮点）---- */

typedef struct {
  const char *dtype;
  int ndim;
  int64_t dims[4];
  const char *bytes; /* mmap 内的数据区指针 */
  size_t nbytes;
} st_raw;

/* 取张量原始字节；不存在或头部异常返回 0 */
static int shard_raw(const st_shard *c, const char *name, st_raw *out) {
  const sgl_json *e = sgl_json_get(c->header, name);
  if (!e) return 0;
  const sgl_json *shape = sgl_json_get(e, "shape");
  const sgl_json *offs = sgl_json_get(e, "data_offsets");
  if (!shape || shape->type != SGL_JSON_ARRAY || !offs || offs->type != SGL_JSON_ARRAY ||
      offs->count != 2 || shape->count < 1 || shape->count > 4)
    return 0;
  out->dtype = sgl_json_get_str(e, "dtype", "");
  out->ndim = shape->count;
  for (int i = 0; i < 4; i++) out->dims[i] = 1;
  for (int i = 0; i < shape->count; i++) out->dims[i] = (int64_t)shape->items[i]->num;
  size_t s = (size_t)offs->items[0]->num, t = (size_t)offs->items[1]->num;
  if (s > t || t > c->data_len) return 0;
  out->bytes = c->data + s;
  out->nbytes = t - s;
  return 1;
}

/* 按名字定位所在分片（index 模式靠 weight_map）*/
static const st_shard *shard_for(const sgl_st_ctx *c, const char *name) {
  if (!c || c->gg) return NULL;
  if (c->wmap) {
    const sgl_json *v = sgl_json_get(c->wmap, name);
    if (!v || v->type != SGL_JSON_STRING || !v->str) return NULL;
    for (int i = 0; i < c->n_sh; i++)
      if (c->sh[i].name && !strcmp(c->sh[i].name, v->str)) return &c->sh[i];
    return NULL;
  }
  return c->n_sh > 0 ? &c->sh[0] : NULL;
}

/* 把 int32 little-endian 字节复制成对齐的 int32 缓冲（拷贝是为了避开非对齐访问）*/
static int32_t *bytes_to_i32(const char *bytes, size_t nbytes, int64_t *n) {
  if (nbytes % 4) return NULL;
  int64_t cnt = (int64_t)(nbytes / 4);
  int32_t *p = (int32_t *)sgl_xmalloc((size_t)cnt * sizeof(int32_t));
  memcpy(p, bytes, nbytes);
  if (n) *n = cnt;
  return p;
}

/* f16/bf16/f32 → 对齐 f32 缓冲 */
static float *bytes_to_f32(const char *bytes, size_t nbytes, const char *dtype, int64_t *n) {
  if (!strcmp(dtype, "F32")) {
    if (nbytes % 4) return NULL;
    int64_t cnt = (int64_t)(nbytes / 4);
    float *p = (float *)sgl_xmalloc((size_t)cnt * sizeof(float));
    memcpy(p, bytes, nbytes);
    if (n) *n = cnt;
    return p;
  }
  if (!strcmp(dtype, "F16") || !strcmp(dtype, "BF16")) {
    if (nbytes % 2) return NULL;
    int64_t cnt = (int64_t)(nbytes / 2);
    float *p = (float *)sgl_xmalloc((size_t)cnt * sizeof(float));
    const uint16_t *h = (const uint16_t *)bytes;
    int bf = !strcmp(dtype, "BF16");
    for (int64_t i = 0; i < cnt; i++) p[i] = bf ? bf16_to_f32(h[i]) : f16_to_f32(h[i]);
    if (n) *n = cnt;
    return p;
  }
  return NULL;
}

/* 把 qweight/qzeros/scales(/g_idx) 解成 [N,K] 的 f32。
 * 成功时返回 malloc 的缓冲（调用方负责登记归属），失败返回 NULL。*/
static float *quant_get(sgl_st_ctx *c, const char *base, int64_t *out_N, int64_t *out_K,
                        int *out_ndim) {
  char nm[512];
  st_raw qw, qz, sc, gi;
  memset(&qz, 0, sizeof(qz));
  memset(&gi, 0, sizeof(gi));
  int has_z = 0, has_g = 0;

  snprintf(nm, sizeof(nm), "%s.qweight", base);
  const st_shard *sh = shard_for(c, nm);
  if (!sh || !shard_raw(sh, nm, &qw)) return NULL;
  if (strcmp(qw.dtype, "I32")) {
    SGL_ERROR("quant: %s dtype=%s，期望 I32", nm, qw.dtype);
    return NULL;
  }
  snprintf(nm, sizeof(nm), "%s.scales", base);
  sh = shard_for(c, nm);
  if (!sh || !shard_raw(sh, nm, &sc)) {
    SGL_ERROR("quant: 缺 %s.scales", base);
    return NULL;
  }
  snprintf(nm, sizeof(nm), "%s.qzeros", base);
  sh = shard_for(c, nm);
  if (sh && shard_raw(sh, nm, &qz)) has_z = 1;
  else if (c->q.zero_point) {
    SGL_ERROR("quant: zero_point 开启但缺 %s.qzeros", base);
    return NULL;
  }
  snprintf(nm, sizeof(nm), "%s.g_idx", base);
  sh = shard_for(c, nm);
  if (sh && shard_raw(sh, nm, &gi)) has_g = 1;

  /* 形状推导：scales 是 [K/G, N]，N 从最后一维拿 */
  int64_t sc_n = 0;
  float *scl = bytes_to_f32(sc.bytes, sc.nbytes, sc.dtype, &sc_n);
  if (!scl) {
    SGL_ERROR("quant: %s.scales dtype=%s 不支持", base, sc.dtype);
    return NULL;
  }
  int64_t N = sc.dims[sc.ndim - 1];
  int64_t sc_rows = (int64_t)(sc_n / (N > 0 ? N : 1));
  int64_t K = 0;
  const int pack = sgl_q_pack(c->q.bits);
  if (c->q.method == SGL_Q_AWQ) {
    /* gemm: qweight [K, N/pack]；gemv: [N/pack, K]。方形权重（K==N）两者都满足，
     * 所以先信 config 的 version，对不上再按形状修正。*/
    const int gemm_ok = qw.dims[1] * pack == N;
    const int gemv_ok = qw.dims[0] * pack == N;
    if (!c->q.transposed && gemm_ok) {
      K = qw.dims[0];
    } else if ((c->q.transposed || !gemm_ok) && gemv_ok) {
      c->q.transposed = 1;
      K = qw.dims[1];
    } else {
      SGL_ERROR("quant(awq): %s qweight [%lld,%lld] 与 N=%lld（pack=%d）对不上", base,
                (long long)qw.dims[0], (long long)qw.dims[1], (long long)N, pack);
      sgl_xfree(scl);
      return NULL;
    }
  } else {
    K = qw.dims[0] * pack;
    N = qw.dims[1];
  }
  const int64_t G = c->q.group_size > 0 ? c->q.group_size : K;
  if (N <= 0 || K <= 0 || sc_rows < 1) {
    SGL_ERROR("quant: %s 形状推不出来（qweight [%lld,%lld], scales [%lld,%lld]）", base,
              (long long)qw.dims[0], (long long)qw.dims[1], (long long)sc_rows, (long long)N);
    sgl_xfree(scl);
    return NULL;
  }
  /* 没 g_idx 时 group 必须是连续的 k/G，行数要对得上 */
  if (!has_g && sc_rows != (K + G - 1) / G) {
    SGL_ERROR("quant: %s scales 行数 %lld 与 K=%lld / group=%lld 不符", base,
              (long long)sc_rows, (long long)K, (long long)G);
    sgl_xfree(scl);
    return NULL;
  }

  int64_t qw_n = 0, qz_n = 0, gi_n = 0;
  int32_t *qwv = bytes_to_i32(qw.bytes, qw.nbytes, &qw_n);
  int32_t *qzv = has_z ? bytes_to_i32(qz.bytes, qz.nbytes, &qz_n) : NULL;
  int32_t *giv = has_g ? bytes_to_i32(gi.bytes, gi.nbytes, &gi_n) : NULL;
  float *out = (float *)sgl_xmalloc((size_t)(N * K) * sizeof(float));
  int rc = sgl_qweight_dequant(&c->q, qwv, qw.dims[0], qw.dims[1], qzv, qz.dims[0], qz.dims[1],
                              scl, sc_rows, N, giv, gi_n, N, K, out);
  sgl_xfree(scl);
  sgl_xfree(qwv);
  sgl_xfree(qzv);
  sgl_xfree(giv);
  if (rc != 0) {
    sgl_xfree(out);
    return NULL;
  }
  if (out_N) *out_N = N;
  if (out_K) *out_K = K;
  if (out_ndim) *out_ndim = 2;
  return out;
}

static sgl_tensor shard_get(st_shard *c, const char *name, int expect_ndim,
                            const int64_t *expect_shape) {
  sgl_tensor bad = sgl_tensor_null();
  const sgl_json *e = sgl_json_get(c->header, name);
  if (!e) return bad; /* 缺失交给调用方判断（可选张量很常见）*/
  const char *dtype = sgl_json_get_str(e, "dtype", "");
  int is_f32 = strcmp(dtype, "F32") == 0;
  int is_f16 = strcmp(dtype, "F16") == 0;
  int is_bf16 = strcmp(dtype, "BF16") == 0;
  /* P1: FP8（DeepSeek V3 / Llama-3.1 fp8 权重的默认格式）：e4m3 与 e5m2 都支持，
   * 加载时解成 f32（CPU 路径上 fp8 只是存储格式；计算仍走 f32）*/
  int is_fp8 = strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E5M2") == 0;
  int is_fp8_e5 = strcmp(dtype, "F8_E5M2") == 0;
  if (!is_f32 && !is_f16 && !is_bf16 && !is_fp8) {
    SGL_ERROR("tensor '%s' dtype=%s, only F32/F16/BF16/F8_E4M3/F8_E5M2 supported now", name, dtype);
    return bad;
  }
  const sgl_json *shape = sgl_json_get(e, "shape");
  const sgl_json *offs = sgl_json_get(e, "data_offsets");
  if (!shape || !offs || shape->type != SGL_JSON_ARRAY || offs->type != SGL_JSON_ARRAY ||
      offs->count != 2)
    return bad;

  int ndim = shape->count;
  if (ndim < 1 || ndim > 4) {
    SGL_ERROR("tensor '%s' ndim=%d unsupported", name, ndim);
    return bad;
  }
  int64_t dims[4] = {1, 1, 1, 1};
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) {
    dims[i] = (int64_t)shape->items[i]->num;
    numel *= dims[i];
  }
  if (expect_ndim >= 0) {
    if (expect_ndim != ndim) {
      SGL_ERROR("tensor '%s' ndim %d != expected %d", name, ndim, expect_ndim);
      return bad;
    }
    for (int i = 0; i < ndim; i++)
      if (expect_shape && expect_shape[i] >= 0 && expect_shape[i] != dims[i]) {
        SGL_ERROR("tensor '%s' shape[%d]=%lld != expected %lld", name, i, (long long)dims[i],
                  (long long)expect_shape[i]);
        return bad;
      }
  }
  size_t s = (size_t)offs->items[0]->num, t = (size_t)offs->items[1]->num;
  size_t esz = is_f32 ? sizeof(float) : (is_fp8 ? 1u : sizeof(uint16_t));
  size_t want = (size_t)numel * esz;
  if (s > t || t > c->data_len || (t - s) != want) {
    SGL_ERROR("tensor '%s' data_offsets invalid (%zu..%zu, numel=%lld, data=%zu)", name, s, t,
              (long long)numel, c->data_len);
    return bad;
  }
  if (is_f32) return sgl_tensor_from_data(ndim, dims, (float *)(c->data + s));

  /* F16/BF16：转换到 owned 缓冲（fp32），归分片生命周期 */
  if (c->n_owned == c->cap_owned) {
    c->cap_owned = c->cap_owned ? c->cap_owned * 2 : 64;
    c->owned = (void **)sgl_xrealloc(c->owned, sizeof(void *) * c->cap_owned);
  }
  float *buf = (float *)sgl_xmalloc((size_t)numel * sizeof(float));
  const uint16_t *src = (const uint16_t *)(c->data + s);
  if (is_fp8) {
    const unsigned char *p8 = (const unsigned char *)(c->data + s);
    for (int64_t i = 0; i < numel; i++) buf[i] = fp8_to_f32(p8[i], is_fp8_e5);
  } else if (is_bf16)
    for (int64_t i = 0; i < numel; i++) buf[i] = bf16_to_f32(src[i]);
  else
    for (int64_t i = 0; i < numel; i++) buf[i] = f16_to_f32(src[i]);
  c->owned[c->n_owned++] = buf;
  return sgl_tensor_from_data(ndim, dims, buf);
}

/* ---- 公共 ---- */

static sgl_st_ctx *ctx_new(void) { return (sgl_st_ctx *)sgl_xcalloc(1, sizeof(sgl_st_ctx)); }

static void ctx_free(sgl_st_ctx *c) {
  if (!c) return;
  for (int i = 0; i < c->n_sh; i++) shard_close(&c->sh[i]);
  for (int i = 0; i < c->n_gowned; i++) sgl_xfree(c->gowned[i]);
  sgl_xfree(c->gowned);
  if (c->gg) sgl_gguf_close(c->gg);
  if (c->root) sgl_json_free(c->root);
  sgl_xfree(c->dir);
  sgl_xfree(c);
}

/* GGUF 反量化缓冲登记（close 时统一释放）*/
static void ctx_track(sgl_st_ctx *c, void *p) {
  if (c->n_gowned == c->cap_gowned) {
    c->cap_gowned = c->cap_gowned ? c->cap_gowned * 2 : 64;
    c->gowned = (void **)sgl_xrealloc(c->gowned, sizeof(void *) * (size_t)c->cap_gowned);
  }
  c->gowned[c->n_gowned++] = p;
}

static int file_exists_(const char *p) {
  FILE *f = NULL;
  if (fopen_s(&f, p, "rb") != 0 || !f) return 0;
  fclose(f);
  return 1;
}

sgl_st_ctx *sgl_st_open(const char *path) {
  sgl_st_ctx *c = ctx_new();
  if (shard_open(&c->sh[0], path) != 0) {
    ctx_free(c);
    return NULL;
  }
  c->n_sh = 1;
  return c;
}

/* 读 config.json 的 quantization_config（AWQ/GPTQ）。没文件/没量化信息时 q 保持 NONE。*/
static void load_quant_cfg(sgl_st_ctx *c, const char *dir);

/* 读整个文本文件（index.json），返回可 free 的 NUL 结尾缓冲 */
static char *read_text_(const char *path, size_t *out_len) {
  FILE *f = NULL;
  if (fopen_s(&f, path, "rb") != 0 || !f) return NULL;
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

sgl_st_ctx *sgl_st_open_dir(const char *dir) {
  char p[1400];
  /* 1) 分片 index */
  snprintf(p, sizeof(p), "%s/model.safetensors.index.json", dir);
  if (file_exists_(p)) {
    size_t len = 0;
    char *txt = read_text_(p, &len);
    if (!txt) {
      SGL_ERROR("read %s failed", p);
      return NULL;
    }
    sgl_json *idx = sgl_json_parse(txt, len);
    sgl_xfree(txt);
    if (!idx) return NULL;
    const sgl_json *wm = sgl_json_get(idx, "weight_map");
    if (!wm || wm->type != SGL_JSON_OBJECT) {
      SGL_ERROR("%s: weight_map missing", p);
      sgl_json_free(idx);
      return NULL;
    }
    sgl_st_ctx *c = ctx_new();
    c->root = idx;
    c->wmap = (sgl_json *)wm;
    c->dir = xstrdup_(dir);
    /* 收集唯一分片名（保持出现顺序，便于日志对照）*/
    for (int i = 0; i < wm->count; i++) {
      const sgl_json *v = wm->items[i];
      if (!v || v->type != SGL_JSON_STRING || !v->str) continue;
      if (c->n_sh >= SGL_ST_MAX_SHARDS) {
        SGL_ERROR("too many shards (>%d)", SGL_ST_MAX_SHARDS);
        ctx_free(c);
        return NULL;
      }
      int dup = 0;
      for (int k = 0; k < c->n_sh; k++)
        if (!strcmp(c->sh[k].name, v->str)) {
          dup = 1;
          break;
        }
      if (dup) continue;
      char sp[1600];
      snprintf(sp, sizeof(sp), "%s/%s", dir, v->str);
      if (shard_open(&c->sh[c->n_sh], sp) != 0) {
        SGL_ERROR("shard %s listed in index not loadable", sp);
        ctx_free(c);
        return NULL;
      }
      c->sh[c->n_sh].name = xstrdup_(v->str);
      c->n_sh++;
    }
    if (c->n_sh == 0) {
      SGL_ERROR("%s: weight_map empty", p);
      ctx_free(c);
      return NULL;
    }
    c->from_index = 1;
    SGL_INFO("safetensors: %d shard(s) from index (%s)", c->n_sh, dir);
    load_quant_cfg(c, dir);
    return c;
  }

  /* 2) 单文件 model.safetensors */
  snprintf(p, sizeof(p), "%s/model.safetensors", dir);
  if (file_exists_(p)) {
    sgl_st_ctx *c = sgl_st_open(p);
    if (c) load_quant_cfg(c, dir);
    return c;
  }

  /* 3) HF 的其他约定名 */
  static const char *cand[] = {"pytorch_model.safetensors", "consolidated.safetensors"};
  for (int i = 0; i < 2; i++) {
    snprintf(p, sizeof(p), "%s/%s", dir, cand[i]);
    if (file_exists_(p)) {
      sgl_st_ctx *c = sgl_st_open(p);
      if (c) load_quant_cfg(c, dir);
      return c;
    }
  }

  /* 4) 没有 safetensors：退到 GGUF（llama.cpp 的量化格式）*/
  {
    sgl_gguf *gg = sgl_gguf_open_dir(dir);
    if (gg) {
      sgl_st_ctx *c = ctx_new();
      c->gg = gg;
      c->dir = xstrdup_(dir);
      SGL_INFO("weights: GGUF source in %s", dir);
      return c;
    }
  }
  SGL_ERROR("no weights found in %s (need *.safetensors, model.safetensors.index.json, or *.gguf)",
            dir);
  return NULL;
}

void sgl_st_close(sgl_st_ctx *c) { ctx_free(c); }

/* 直接按名字取（safetensors 原生张量，不涉及 AWQ/GPTQ 解包）。
 * index 模式下张量不在 weight_map 里时返回 null（交给 sgl_st_get 去试量化路径）。*/
static sgl_tensor st_get_plain(sgl_st_ctx *c, const char *name, int expect_ndim,
                              const int64_t *expect_shape) {
  if (c->wmap) {
    const sgl_json *v = sgl_json_get(c->wmap, name);
    if (!v || v->type != SGL_JSON_STRING || !v->str) return sgl_tensor_null();
    for (int i = 0; i < c->n_sh; i++)
      if (c->sh[i].name && !strcmp(c->sh[i].name, v->str))
        return shard_get(&c->sh[i], name, expect_ndim, expect_shape);
    return sgl_tensor_null();
  }
  if (c->n_sh == 0) return sgl_tensor_null();
  return shard_get(&c->sh[0], name, expect_ndim, expect_shape);
}

/* 读 config.json 的 quantization_config（AWQ/GPTQ）。没有文件/没有量化信息时 q 保持 NONE。*/
static void load_quant_cfg(sgl_st_ctx *c, const char *dir) {
  char p[1400];
  snprintf(p, sizeof(p), "%s/config.json", dir);
  size_t len = 0;
  char *txt = read_text_(p, &len);
  if (!txt) return;
  sgl_qcfg_from_json(txt, len, &c->q);
  sgl_xfree(txt);
}

sgl_tensor sgl_st_get(sgl_st_ctx *c, const char *name, int expect_ndim,
                      const int64_t *expect_shape) {
  if (!c || (c->n_sh == 0 && !c->gg)) return sgl_tensor_null();
  /* GGUF 权重源：按 HF 名映射 + 反量化成 f32 */
  if (c->gg) {
    int nd = 0;
    int64_t sh[4] = {1, 1, 1, 1};
    float *buf = sgl_gguf_get_f32(c->gg, name, &nd, sh);
    if (!buf) return sgl_tensor_null();
    if (expect_ndim >= 0) {
      int bad = expect_ndim != nd;
      for (int i = 0; !bad && expect_shape && i < nd; i++)
        if (expect_shape[i] >= 0 && expect_shape[i] != sh[i]) bad = 1;
      if (bad) {
        SGL_ERROR("gguf tensor '%s' shape %lldx%lld != expected %d-dim(%lld,%lld)", name,
                  (long long)sh[0], (long long)sh[1], expect_ndim,
                  (long long)(expect_shape ? expect_shape[0] : -1),
                  (long long)(expect_shape && expect_ndim > 1 ? expect_shape[1] : -1));
        sgl_xfree(buf);
        return sgl_tensor_null();
      }
    }
    ctx_track(c, buf);
    /* llama.cpp 的 GGUF 把 q/k_proj 按 head 交错重排了，这里的 forward 是 HF 布局 → 还原 */
    sgl_gguf_unpermute_qk(c->gg, name, buf, nd, sh);
    return sgl_tensor_from_data(nd, sh, buf);
  }
  sgl_tensor t = st_get_plain(c, name, expect_ndim, expect_shape);
  if (!sgl_tensor_is_null(t)) return t;
  /* AWQ/GPTQ：<prefix>.weight 由 <prefix>.qweight/.qzeros/.scales(/g_idx) 现场解开 */
  if (c->q.method != SGL_Q_NONE && expect_ndim <= 2) {
    size_t nl = strlen(name);
    if (nl > 7 && !strcmp(name + nl - 7, ".weight")) {
      char base[512];
      if (nl - 7 < sizeof(base)) {
        memcpy(base, name, nl - 7);
        base[nl - 7] = 0;
        int64_t N = 0, K = 0;
        int nd = 2;
        float *buf = quant_get(c, base, &N, &K, &nd);
        if (buf) {
          int bad = 0;
          if (expect_ndim >= 0) {
            bad = expect_ndim != 2;
            if (!bad && expect_shape) {
              if (expect_shape[0] >= 0 && expect_shape[0] != N) bad = 1;
              if (expect_shape[1] >= 0 && expect_shape[1] != K) bad = 1;
            }
          }
          if (bad) {
            SGL_ERROR("quant: %s 解出 [%lld,%lld]，调用方期望 [%lld,%lld]", name, (long long)N,
                      (long long)K, (long long)(expect_shape ? expect_shape[0] : -1),
                      (long long)(expect_shape && expect_ndim > 1 ? expect_shape[1] : -1));
            sgl_xfree(buf);
            return sgl_tensor_null();
          }
          ctx_track(c, buf);
          return sgl_tensor_from_data(2, (int64_t[]){N, K}, buf);
        }
      }
    }
  }
  return t;
}

int sgl_st_names(sgl_st_ctx *c, const char ***names) {
  if (!c) {
    *names = NULL;
    return 0;
  }
  if (c->gg) return sgl_gguf_names(c->gg, names);
  if (c->n_sh == 0) {
    *names = NULL;
    return 0;
  }
  if (c->wmap) { /* 分片模式：列 weight_map 的键 */
    *names = (const char **)c->wmap->keys;
    return c->wmap->count;
  }
  *names = (const char **)c->sh[0].header->keys;
  return c->sh[0].header->count;
}

int sgl_st_shard_count(const sgl_st_ctx *c) {
  if (!c) return 0;
  return c->gg ? 1 : c->n_sh;
}