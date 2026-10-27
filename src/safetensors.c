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
#include "sgl/json.h"
#include "sgl/log.h"
#include "sgl/port.h"

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
};

/* IEEE 754 half → float（位运算，无依赖）*/
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h >> 15) & 1;
  uint32_t exp = (uint32_t)(h >> 10) & 0x1F;
  uint32_t frac = (uint32_t)h & 0x3FF;
  uint32_t f;
  if (exp == 0) f = sign << 31 | frac << 13;                     /* 次正规 */
  else if (exp == 31) f = sign << 31 | 0x7F800000u | frac << 13; /* inf/nan */
  else f = sign << 31 | (exp - 15 + 127) << 23 | frac << 13;
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
  if (c->root) sgl_json_free(c->root);
  sgl_xfree(c->dir);
  sgl_xfree(c);
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
    return c;
  }

  /* 2) 单文件 model.safetensors */
  snprintf(p, sizeof(p), "%s/model.safetensors", dir);
  if (file_exists_(p)) return sgl_st_open(p);

  /* 3) HF 的其他约定名 */
  static const char *cand[] = {"pytorch_model.safetensors", "consolidated.safetensors"};
  for (int i = 0; i < 2; i++) {
    snprintf(p, sizeof(p), "%s/%s", dir, cand[i]);
    if (file_exists_(p)) return sgl_st_open(p);
  }
  SGL_ERROR("no weights found in %s (need model.safetensors or model.safetensors.index.json)", dir);
  return NULL;
}

void sgl_st_close(sgl_st_ctx *c) { ctx_free(c); }

sgl_tensor sgl_st_get(sgl_st_ctx *c, const char *name, int expect_ndim,
                      const int64_t *expect_shape) {
  if (!c || c->n_sh == 0) return sgl_tensor_null();
  if (c->wmap) {
    const sgl_json *v = sgl_json_get(c->wmap, name);
    if (!v || v->type != SGL_JSON_STRING || !v->str) {
      SGL_ERROR("tensor '%s' not found in weight_map", name);
      return sgl_tensor_null();
    }
    for (int i = 0; i < c->n_sh; i++)
      if (c->sh[i].name && !strcmp(c->sh[i].name, v->str))
        return shard_get(&c->sh[i], name, expect_ndim, expect_shape);
    SGL_ERROR("tensor '%s' maps to unknown shard '%s'", name, v->str);
    return sgl_tensor_null();
  }
  sgl_tensor t = shard_get(&c->sh[0], name, expect_ndim, expect_shape);
  if (sgl_tensor_is_null(t) && !sgl_json_get(c->sh[0].header, name))
    SGL_ERROR("tensor '%s' not found", name);
  return t;
}

int sgl_st_names(sgl_st_ctx *c, const char ***names) {
  if (!c || c->n_sh == 0) {
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

int sgl_st_shard_count(const sgl_st_ctx *c) { return c ? c->n_sh : 0; }