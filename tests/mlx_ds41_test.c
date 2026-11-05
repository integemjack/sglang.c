/* mlx_ds41_test.c — dsv41-mini 金标准对拍。
 * 用法：mlx_ds41_test <dir>   （dir 下有 mini_ref.st / mini_ref_io.bin / mini_ref_tok.bin）
 */
#include "sgl/ds41.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define B 2
#define V 256
#define N_DEC 4

static double max_abs_diff(const float* a, const float* b, size_t n) {
  double m = 0;
  for (size_t i = 0; i < n; i++) {
    double d = fabs((double)a[i] - (double)b[i]);
    if (d > m) m = d;
  }
  return m;
}

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  char path[512];
  if (sgl_ds41_load(dir) != 0) return 1;

  snprintf(path, sizeof path, "%s/mini_ref_io.bin", dir);
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); return 1; }
  size_t nf = (size_t)(1 + N_DEC) * B * V;
  float* ref = (float*)malloc(nf * 4);
  if (fread(ref, 4, nf, f) != nf) { fprintf(stderr, "io.bin short\n"); return 1; }
  fclose(f);
  snprintf(path, sizeof path, "%s/mini_ref_tok.bin", dir);
  f = fopen(path, "rb");
  if (!f) { perror(path); return 1; }
  uint32_t* reft = (uint32_t*)malloc((size_t)N_DEC * B * 4);
  if (fread(reft, 4, (size_t)N_DEC * B, f) != (size_t)N_DEC * B) { fprintf(stderr, "tok.bin short\n"); return 1; }
  fclose(f);

  float* lp = (float*)malloc(B * V * 4);
  float* ld = (float*)malloc((size_t)N_DEC * B * V * 4);
  uint32_t* tk = (uint32_t*)malloc((size_t)N_DEC * B * 4);
  if (sgl_ds41_run(N_DEC, lp, tk, ld) != 0) return 1;

  int fail = 0;
  double m = max_abs_diff(lp, ref, B * V);
  printf("prefill  maxerr=%.6g\n", m);
  if (!(m < 1e-3)) fail = 1;
  for (int i = 0; i < N_DEC; i++) {
    m = max_abs_diff(ld + (size_t)i * B * V, ref + (size_t)(1 + i) * B * V, B * V);
    printf("decode%2d maxerr=%.6g", i, m);
    int tok_ok = tk[2 * i] == reft[2 * i] && tk[2 * i + 1] == reft[2 * i + 1];
    printf("  tok[%u,%u] vs [%u,%u]%s\n", tk[2 * i], tk[2 * i + 1], reft[2 * i], reft[2 * i + 1],
           tok_ok ? "" : "  MISMATCH");
    if (!(m < 1e-3) || !tok_ok) fail = 1;
  }
  printf(fail ? "DS41_PARITY_FAIL\n" : "DS41_PARITY_PASS\n");
  return fail;
}
