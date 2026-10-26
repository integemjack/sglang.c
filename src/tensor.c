/* sgl tensor 实现 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"

sgl_tensor sgl_tensor_null(void) {
  sgl_tensor t;
  memset(&t, 0, sizeof(t));
  t.ndim = -1;
  return t;
}

int sgl_tensor_is_null(sgl_tensor t) { return t.ndim < 0; }

static void fill_meta(sgl_tensor *t, int ndim, const int64_t *shape) {
  t->ndim = ndim;
  int64_t stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    t->shape[i] = shape[i];
    t->strides[i] = stride;
    stride *= shape[i];
  }
  t->device = SGL_CPU;
}

sgl_tensor sgl_tensor_zeros(int ndim, const int64_t *shape) {
  sgl_tensor t = sgl_tensor_null();
  fill_meta(&t, ndim, shape);
  size_t bytes = (size_t)sgl_tensor_numel(&t) * sizeof(float);
  t.data = (float *)sgl_xmalloc(bytes);
  t._owner = t.data;
  memset(t.data, 0, bytes);
  return t;
}

sgl_tensor sgl_tensor_empty(int ndim, const int64_t *shape) {
  sgl_tensor t = sgl_tensor_zeros(ndim, shape);
  return t; /* 简化：直接给零初始化，语义安全 */
}

sgl_tensor sgl_tensor_from_data(int ndim, const int64_t *shape, float *data) {
  sgl_tensor t = sgl_tensor_null();
  fill_meta(&t, ndim, shape);
  t.data = data;
  t._owner = NULL;
  return t;
}

sgl_tensor sgl_tensor_copy_from(int ndim, const int64_t *shape, const float *src) {
  sgl_tensor t = sgl_tensor_zeros(ndim, shape);
  memcpy(t.data, src, (size_t)sgl_tensor_numel(&t) * sizeof(float));
  return t;
}

int64_t sgl_tensor_numel(const sgl_tensor *t) {
  if (t->ndim <= 0) return 0;
  int64_t n = 1;
  for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
  return n;
}

void sgl_tensor_free(sgl_tensor *t) {
  if (t->_owner) sgl_xfree(t->_owner);
  t->data = NULL;
  t->_owner = NULL;
  t->ndim = -1;
}

float sgl_tensor_sum(const sgl_tensor *t) {
  int64_t n = sgl_tensor_numel(t);
  double s = 0;
  for (int64_t i = 0; i < n; i++) s += t->data[i];
  return (float)s;
}

float sgl_tensor_max_abs(const sgl_tensor *t) {
  int64_t n = sgl_tensor_numel(t);
  float m = 0;
  for (int64_t i = 0; i < n; i++) {
    float a = fabsf(t->data[i]);
    if (a > m) m = a;
  }
  return m;
}
