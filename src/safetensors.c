/* sgl safetensors 实现
 * 文件格式：<u64 header_len><header JSON><数据区>
 * header: { "tensor": {"dtype":"F32","shape":[..],"data_offsets":[s,e]}, ... } */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/log.h"

struct sgl_st_ctx {
  sgl_mmap map;
  sgl_json *header;   /* 解析后的 header */
  const char *data;   /* 数据区起始 */
  size_t data_len;
  void **owned;       /* F16→F32 转换缓冲（close 时释放）*/
  int n_owned, cap_owned;
};

/* IEEE 754 half → float（位运算，无依赖）*/
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h >> 15) & 1;
  uint32_t exp = (uint32_t)(h >> 10) & 0x1F;
  uint32_t frac = (uint32_t)h & 0x3FF;
  uint32_t f;
  if (exp == 0) f = sign << 31 | frac << 13;          /* 次正规 */
  else if (exp == 31) f = sign << 31 | 0x7F800000u | frac << 13; /* inf/nan */
  else f = sign << 31 | (exp - 15 + 127) << 23 | frac << 13;
  float out;
  memcpy(&out, &f, 4);
  return out;
}

/* bfloat16 → float：即 f32 的高 16 位，低位补零 */
static float bf16_to_f32(uint16_t h) {
  uint32_t f = (uint32_t)h << 16;
  float out;
  memcpy(&out, &f, 4);
  return out;
}

sgl_st_ctx *sgl_st_open(const char *path) {
  sgl_mmap map;
  if (sgl_mmap_open(&map, path) != 0) {
    SGL_ERROR("open safetensors %s failed", path);
    return NULL;
  }
  if (map.size < 8) { SGL_ERROR("safetensors too small"); goto fail; }

  uint64_t hlen;
  memcpy(&hlen, map.data, 8);
  if ((uint64_t)8 + hlen > map.size) { SGL_ERROR("safetensors header overflow"); goto fail; }

  sgl_json *header = sgl_json_parse((const char *)map.data + 8, (size_t)hlen);
  if (!header || header->type != SGL_JSON_OBJECT) {
    SGL_ERROR("safetensors header JSON invalid");
    if (header) sgl_json_free(header);
    goto fail;
  }

  sgl_st_ctx *c = (sgl_st_ctx *)sgl_xcalloc(1, sizeof(sgl_st_ctx));
  c->map = map;
  c->header = header;
  c->data = (const char *)map.data + 8 + hlen;
  c->data_len = map.size - 8 - hlen;
  SGL_INFO("safetensors opened: %s, %lld tensors, data %.1f MB", path,
           (long long)header->count, c->data_len / 1048576.0);
  return c;

fail:
  sgl_mmap_close(&map);
  return NULL;
}

void sgl_st_close(sgl_st_ctx *c) {
  if (!c) return;
  for (int i = 0; i < c->n_owned; i++) sgl_xfree(c->owned[i]);
  sgl_xfree(c->owned);
  sgl_json_free(c->header);
  sgl_mmap_close(&c->map);
  sgl_xfree(c);
}

sgl_tensor sgl_st_get(sgl_st_ctx *c, const char *name, int expect_ndim,
                      const int64_t *expect_shape) {
  sgl_tensor bad = sgl_tensor_null();
  const sgl_json *e = sgl_json_get(c->header, name);
  if (!e) { SGL_ERROR("tensor '%s' not found", name); return bad; }
  const char *dtype = sgl_json_get_str(e, "dtype", "");
  int is_f32 = strcmp(dtype, "F32") == 0;
  int is_f16 = strcmp(dtype, "F16") == 0;
  int is_bf16 = strcmp(dtype, "BF16") == 0;
  if (!is_f32 && !is_f16 && !is_bf16) {
    SGL_ERROR("tensor '%s' dtype=%s, only F32/F16/BF16 supported now", name, dtype);
    return bad;
  }
  const sgl_json *shape = sgl_json_get(e, "shape");
  const sgl_json *offs = sgl_json_get(e, "data_offsets");
  if (!shape || !offs || shape->type != SGL_JSON_ARRAY ||
      offs->type != SGL_JSON_ARRAY || offs->count != 2)
    return bad;

  int ndim = shape->count;
  if (ndim < 1 || ndim > 3) { SGL_ERROR("tensor '%s' ndim=%d unsupported", name, ndim); return bad; }
  int64_t dims[3] = {1, 1, 1};
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
        SGL_ERROR("tensor '%s' shape[%d]=%lld != expected %lld", name, i,
                  (long long)dims[i], (long long)expect_shape[i]);
        return bad;
      }
  }
  size_t s = (size_t)offs->items[0]->num, t = (size_t)offs->items[1]->num;
  size_t want = (size_t)numel * (is_f32 ? sizeof(float) : sizeof(uint16_t));
  if (s > t || t > c->data_len || (t - s) != want) {
    SGL_ERROR("tensor '%s' data_offsets invalid (%zu..%zu, numel=%lld, data=%zu)",
              name, s, t, (long long)numel, c->data_len);
    return bad;
  }
  if (is_f32) {
    return sgl_tensor_from_data(ndim, dims, (float *)(c->data + s));
  }
  /* F16/BF16：转换到 owned 缓冲（fp32），归 ctx 生命周期 */
  if (c->n_owned == c->cap_owned) {
    c->cap_owned = c->cap_owned ? c->cap_owned * 2 : 64;
    c->owned = (void **)sgl_xrealloc(c->owned, sizeof(void *) * c->cap_owned);
  }
  float *buf = (float *)sgl_xmalloc((size_t)numel * sizeof(float));
  const uint16_t *src = (const uint16_t *)(c->data + s);
  if (is_bf16)
    for (int64_t i = 0; i < numel; i++) buf[i] = bf16_to_f32(src[i]);
  else
    for (int64_t i = 0; i < numel; i++) buf[i] = f16_to_f32(src[i]);
  c->owned[c->n_owned++] = buf;
  return sgl_tensor_from_data(ndim, dims, buf);
}

int sgl_st_names(sgl_st_ctx *c, const char ***names) {
  *names = (const char **)c->header->keys;
  return c->header->count;
}
