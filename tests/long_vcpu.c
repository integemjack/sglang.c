/* _long_vcpu.c — t2 验收：8K-token 重复文本 prompt，GPU forward_seg prefill
 * logits vs CPU sgl_llama_forward 同 ids 的 maxerr（要求 < 1e-3）。
 * 一次性验收工具（不进 ctest）：CPU 8K prefill 需数分钟。 */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/cuda.h"
#include "sgl/llama.h"

static double now_sec(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : "models/qwen05";
  int NP = argc > 2 ? atoi(argv[2]) : 8192;

  char tpath[512];
  snprintf(tpath, sizeof(tpath), "%s/tokenizer.json", dir);
  sgl_tokenizer *tok = sgl_tokenizer_load(tpath);
  /* 重复文本铺到 NP 个 token（重复单元 = 编码 "The quick brown fox jumps
   * over the lazy dog. " 后循环） */
  const char *unit = "The quick brown fox jumps over the lazy dog. ";
  int un = 0;
  int *uids = sgl_tokenizer_encode(tok, unit, strlen(unit), &un);
  int *ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)NP);
  for (int i = 0; i < NP; i++) ids[i] = uids[i % un];
  printf("prompt: %d tokens (repeat unit %d)\n", NP, un);

  sgl_llama *cpu = sgl_llama_load(dir);
  int V = cpu->vocab;
  sgl_cuda_model *gpu = sgl_cuda_from_llama(cpu);

  float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  float *lc = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);

  double t0 = now_sec();
  CHECK:;
  if (sgl_cuda_forward_seg(gpu, ids, NP, 0, 0, lg) != 0) { printf("gpu fail\n"); return 1; }
  printf("GPU prefill %d tok: %.1fs\n", NP, now_sec() - t0);

  if (argc > 3 && !strcmp(argv[3], "gpu")) {
    printf("gpu-only mode, skip CPU compare\n");
    return 0;
  }

  t0 = now_sec();
  int *pg = (int *)sgl_xmalloc(sizeof(int) * (size_t)(NP + 8));
  if (sgl_llama_pages_alloc(cpu, pg, NP + 8) != 0) { printf("cpu pages fail (need SGL_POOL_SLOTS)\n"); return 1; }
  if (sgl_llama_forward(cpu, ids, NP, 0, pg, lc) != 0) { printf("cpu fail\n"); return 1; }
  printf("CPU prefill %d tok: %.1fs\n", NP, now_sec() - t0);

  double maxerr = 0;
  for (int i = 0; i < V; i++) {
    double e = fabs((double)lc[i] - lg[i]);
    if (e > maxerr) maxerr = e;
  }
  int gc = 0, gg = 0;
  for (int i = 1; i < V; i++) { if (lc[i] > lc[gc]) gc = i; if (lg[i] > lg[gg]) gg = i; }
  printf("maxerr = %g  (greedy cpu=%d gpu=%d %s)\n", maxerr, gc, gg, gc == gg ? "SAME" : "DIFF");
  printf("verdict: %s\n", maxerr < 1e-3 ? "PASS (<1e-3)" : "FAIL");
  return maxerr < 1e-3 ? 0 : 1;
}
