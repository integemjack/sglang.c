/* tests/mlx_q4_test.c — 4-bit 检查点的 logits 对拍工具
 *
 * 用法: mlx_q4_test <model_dir> [token_id ...]
 *   加载（走 sgl_mlx_from_dir，不经过 CPU 权重加载），prefill 给定 token，
 *   打印最后一行 logits 的 top-10。给 Python 侧（mlx_lm）对照用。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/mlx.h"

typedef struct { int id; float v; } Cand;

static int cmp_desc(const void *a, const void *b) {
  float x = ((const Cand *)a)->v, y = ((const Cand *)b)->v;
  return (x < y) - (x > y);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model_dir> [token_id ...]\n", argv[0]);
    return 1;
  }
  int ids[64], n = 0;
  for (int i = 2; i < argc && n < 64; i++) ids[n++] = atoi(argv[i]);
  if (n == 0) ids[n++] = 9707; /* "Hello" in Qwen */

  sgl_mlx_model *g = sgl_mlx_from_dir(argv[1]);
  if (!g) {
    fprintf(stderr, "load failed\n");
    return 1;
  }
  const int V = sgl_mlx_vocab(g);
  printf("loaded: V=%d ctx=%d\n", V, sgl_mlx_ctx(g));
  float *lg = (float *)malloc(sizeof(float) * (size_t)V);
  if (sgl_mlx_forward_seg(g, ids, n, 0, 0, lg) != 0) {
    fprintf(stderr, "forward failed\n");
    return 1;
  }
  Cand *c = (Cand *)malloc(sizeof(Cand) * (size_t)V);
  for (int i = 0; i < V; i++) {
    c[i].id = i;
    c[i].v = lg[i];
  }
  qsort(c, (size_t)V, sizeof(Cand), cmp_desc);
  printf("top10:");
  for (int i = 0; i < 10; i++) printf(" %d:%.4f", c[i].id, c[i].v);
  printf("\n");
  /* 也打几个统计量，方便定位是不是整体尺度不对 */
  double s = 0, sq = 0;
  float mn = 1e30f, mxv = -1e30f;
  for (int i = 0; i < V; i++) {
    s += lg[i];
    sq += (double)lg[i] * lg[i];
    if (lg[i] < mn) mn = lg[i];
    if (lg[i] > mxv) mxv = lg[i];
  }
  double mean = s / V;
  printf("logits: mean=%+.5f std=%.5f min=%+.3f max=%+.3f\n", mean, sqrt(sq / V - mean * mean),
         mn, mxv);

  /* decode bench：与 mlx_test 同尺子 —— greedy 64 步，墙钟，报 ms/步。*/
  {
    const int STEPS = 64;
    int *toks = (int *)malloc(sizeof(int) * (size_t)(STEPS + 8));
    int nt = n;
    memcpy(toks, ids, sizeof(int) * (size_t)n);
    float *lg = (float *)malloc(sizeof(float) * (size_t)V);
    sgl_mlx_forward_seg(g, toks, nt, 0, 0, lg);
    int tok = 0;
    for (int i = 0; i < V; i++)
      if (lg[i] > lg[tok]) tok = i;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int st = 0; st < STEPS; st++) {
      toks[nt++] = tok;
      if (sgl_mlx_forward_seg(g, &tok, 1, nt - 1, 0, lg) != 0) break;
      tok = 0;
      for (int i = 0; i < V; i++)
        if (lg[i] > lg[tok]) tok = i;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("BENCH q_decode_ms_per_step=%.2f tok_s=%.1f (%d steps)\n", ms / STEPS,
           STEPS * 1000.0 / ms, STEPS);
    free(toks);
    free(lg);
  }

  free(c);
  free(lg);
  sgl_mlx_free(g);
  return 0;
}