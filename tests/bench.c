/* bench.c — GEMM/点积基准（M14 SIMD 前后对比用；不进 ctest 断言，只打印）*/
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "sgl/alloc.h"
#include "sgl/ops.h"
#include "sgl/quant.h"

static double now(void) {
  return (double)clock() / CLOCKS_PER_SEC;
}

int main(void) {
  /* 实测形状：TinyLlama qkv 投影 [1,2048]×[2048,2048] 与 down [1,5632]×[2048,5632] */
  enum { E = 2048, I = 5632 };
  float *A = (float *)sgl_xmalloc(sizeof(float) * E);
  float *W1 = (float *)sgl_xmalloc(sizeof(float) * E * E);
  float *W2 = (float *)sgl_xmalloc(sizeof(float) * E * I);
  float *C = (float *)sgl_xmalloc(sizeof(float) * E);
  for (int i = 0; i < E; i++) A[i] = (float)sin(i * 0.3);
  for (size_t i = 0; i < (size_t)E * E; i++) W1[i] = (float)cos(i * 0.13) * 0.05f;
  for (size_t i = 0; i < (size_t)E * I; i++) W2[i] = (float)cos(i * 0.17) * 0.05f;

  int8_t *W8 = (int8_t *)sgl_xmalloc((size_t)E * I);
  float *s8 = (float *)sgl_xmalloc(sizeof(float) * E);
  sgl_quant_rows_i8(W2, W8, s8, E, I);

  /* 预热 + 3 次取最小 */
#define BEST3(fn) ({ double best = 1e9; for (int r = 0; r < 3; r++) { double s0 = now(); fn; double d = now() - s0; if (d < best) best = d; } best; })
  double d1 = BEST3(sgl_matmul_nt(A, W1, NULL, C, 1, E, E));
  double d2 = BEST3(sgl_matmul_nt(A, W2, NULL, C, 1, E, I));
  double d3 = BEST3(sgl_matmul_w8a16(A, W8, s8, NULL, C, 1, E, I));

  printf("matmul_nt  [1,2048]x[2048,2048]: %.1f ms (%.2f GFLOPs)\n",
         d1 * 1e3, 2.0 * E * E / d1 / 1e9);
  printf("matmul_nt  [1,2048]x[2048,5632]: %.1f ms (%.2f GFLOPs)\n",
         d2 * 1e3, 2.0 * E * I / d2 / 1e9);
  printf("w8a16      [1,2048]x[2048,5632]: %.1f ms\n", d3 * 1e3);
  return 0;
}
