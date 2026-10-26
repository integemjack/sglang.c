/* test_tensor.c — 张量与算子测试 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "sgl/ops.h"
#include "sgl/tensor.h"

static int fails = 0;
#define CHECK(cond, ...)                       \
  do {                                         \
    if (!(cond)) {                             \
      printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      printf(__VA_ARGS__);                     \
      printf("\n");                            \
      fails++;                                 \
    }                                          \
  } while (0)

static void test_tensor_basic(void) {
  int64_t shape[2] = {3, 4};
  sgl_tensor t = sgl_tensor_zeros(2, shape);
  CHECK(sgl_tensor_numel(&t) == 12, "numel=%lld", (long long)sgl_tensor_numel(&t));
  t.data[0] = 1.5f;
  CHECK(fabs(sgl_tensor_sum(&t) - 1.5) < 1e-6, "sum");
  t.data[1] = -3.0f;
  CHECK(fabs(sgl_tensor_max_abs(&t) - 3.0) < 1e-6, "max_abs");
  sgl_tensor_free(&t);
  CHECK(sgl_tensor_is_null(t), "null after free");
}

static void test_matmul(void) {
  /* 2x3 @ 3x2，手算对照 */
  float A[6] = {1, 2, 3, 4, 5, 6};
  float B[6] = {7, 8, 9, 10, 11, 12};
  float bias[2] = {0.5f, -0.5f};
  float C[4];
  sgl_matmul(A, B, bias, C, 2, 2, 3);
  CHECK(fabs(C[0] - (58.5)) < 1e-4, "C00=%f", C[0]);      /* 1*7+2*9+3*11 +0.5 */
  CHECK(fabs(C[1] - (63.5)) < 1e-4, "C01=%f", C[1]);      /* 1*8+2*10+3*12 -0.5 */
  CHECK(fabs(C[2] - (139.5)) < 1e-4, "C10=%f", C[2]);
  CHECK(fabs(C[3] - (153.5)) < 1e-4, "C11=%f", C[3]);
}

static void test_matmul_random(void) {
  /* 与朴素三重循环对照（含 M=1 的 decode 形状）*/
  int M = 5, N = 7, K = 11;
  float *A = malloc(sizeof(float) * M * K);
  float *B = malloc(sizeof(float) * K * N);
  float *C = malloc(sizeof(float) * M * N);
  float *R = malloc(sizeof(float) * M * N);
  srand(42);
  for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 1000) / 100.f - 5.f;
  for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 1000) / 100.f - 5.f;
  sgl_matmul(A, B, NULL, C, M, N, K);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      double s = 0;
      for (int k = 0; k < K; k++) s += (double)A[i * K + k] * B[k * N + j];
      R[i * N + j] = (float)s;
    }
  double maxerr = 0;
  for (int i = 0; i < M * N; i++) {
    double e = fabs(C[i] - R[i]);
    if (e > maxerr) maxerr = e;
  }
  CHECK(maxerr < 1e-3, "maxerr=%g", maxerr);
  free(A); free(B); free(C); free(R);
}

static void test_layernorm(void) {
  float x[4] = {1, 2, 3, 4};
  float g[4] = {1, 1, 1, 1};
  float b[4] = {0, 0, 0, 0};
  float y[4];
  sgl_layernorm(x, g, b, y, 1, 4, 1e-5f);
  double mean = (y[0] + y[1] + y[2] + y[3]) / 4;
  CHECK(fabs(mean) < 1e-5, "mean=%g", mean);
  double var = 0;
  for (int i = 0; i < 4; i++) var += (y[i] - mean) * (y[i] - mean);
  var /= 4;
  CHECK(fabs(var - 1.0) < 1e-3, "var=%g", var);
}

static void test_softmax_gelu(void) {
  float r[3] = {1000.0f, 1000.0f, 1000.0f}; /* 数值稳定 */
  sgl_softmax_inplace(r, 3);
  CHECK(fabs(r[0] - 1.0 / 3) < 1e-6 && fabs(r[2] - 1.0 / 3) < 1e-6, "r0=%f r2=%f", r[0], r[2]);
  float g[2] = {0.0f, 1.0f};
  sgl_gelu_inplace(g, 2);
  CHECK(fabs(g[0]) < 1e-6, "gelu(0)=%f", g[0]);
  CHECK(fabs(g[1] - 0.8412f) < 1e-3, "gelu(1)=%f", g[1]); /* tanh 近似值 */
}

int main(void) {
#ifndef SGL_SKIP_BASIC
  test_tensor_basic();
#endif
#ifndef SGL_SKIP_MM
  test_matmul();
#endif
#ifndef SGL_SKIP_MMR
  test_matmul_random();
#endif
#ifndef SGL_SKIP_LN
  test_layernorm();
#endif
#ifndef SGL_SKIP_SF
  test_softmax_gelu();
#endif
  printf(fails ? "test_tensor: %d FAILURES\n" : "test_tensor: all OK\n", fails);
  return fails ? 1 : 0;
}
