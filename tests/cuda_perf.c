/* cuda_perf.c — GPU 性能基准（G1 prefill / G2 W8 / G3 graph 的 A/B 工具）
 *
 * 用法: cuda_perf <model_dir> [--prompt N[,N..]] [--decode M] [--batch B[,B..]]
 *                   [--rounds R] [--seg S]
 * 档位由环境变量决定（这样同一份二进制可以 A/B 对照）:
 *   SGL_CUDA_FP16 / SGL_CUDA_W8 / SGL_W8_DP4A / SGL_CUDA_GRAPH / SGL_CUDA_CTX
 * 输出每行一个 [perf] 记录，best-of-rounds，单位 ms / tok/s。
 * 不进 ctest：纯测量工具，不做断言。
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/alloc.h"
#include "sgl/cuda.h"
#include "sgl/llama.h"

static double now_sec(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int parse_list(const char *s, int *out, int max) {
  int n = 0;
  const char *p = s;
  while (*p && n < max) {
    out[n++] = atoi(p);
    const char *c = strchr(p, ',');
    if (!c) break;
    p = c + 1;
  }
  return n;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = "models/qwen05";
  int pn[8] = {512, 2048, 8192};
  int npn = 3;
  int bs[8] = {1, 4, 8};
  int nbs = 3;
  int decode_steps = 32;
  int rounds = 3;
  int ab = 0;   /* G1/G2: same-process A/B (toggle the path per round) */
  int seg = 0; /* 0 => ctx/8 per sequence */

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
    else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) npn = parse_list(argv[++i], pn, 8);
    else if (!strcmp(argv[i], "--batch") && i + 1 < argc) nbs = parse_list(argv[++i], bs, 8);
    else if (!strcmp(argv[i], "--decode") && i + 1 < argc) decode_steps = atoi(argv[i + 1]), i++;
    else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[i + 1]), i++;
    else if (!strcmp(argv[i], "--seg") && i + 1 < argc) seg = atoi(argv[i + 1]), i++;
    else if (!strcmp(argv[i], "--ab")) ab = 1;
    else dir = argv[i]; /* positional model dir */
  }

  sgl_llama *cpu = sgl_llama_load(dir);
  if (!cpu) { fprintf(stderr, "cuda_perf: cannot load %s\n", dir); return 1; }
  int V = cpu->vocab;
  sgl_cuda_model *gpu = sgl_cuda_from_llama(cpu);
  if (!gpu) { fprintf(stderr, "cuda_perf: cannot upload %s to GPU\n", dir); return 1; }
  int ctx = sgl_cuda_ctx(gpu);
  if (!seg) seg = ctx / 8;
  printf("[cfg] model=%s ctx=%d seg=%d rounds=%d free_vram=%ldMB\n", dir, ctx, seg, rounds,
         sgl_cuda_free_mem_mb());

  /* synthetic prompt: cycling ids in a safe range (timing is content-free) */
  int maxn = pn[0];
  for (int i = 1; i < npn; i++) if (pn[i] > maxn) maxn = pn[i];
  int *ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)maxn);
  for (int i = 0; i < maxn; i++) ids[i] = 100 + (i % 900) % V;
  float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * 8);

  /* ---- prefill ----
   * --ab: alternate the two code paths inside one process so both see the
   * identical allocation state. WDDM spills to system memory once the working
   * set exceeds free VRAM, which inflates absolute numbers by 3-4x (this card
   * normally has ~2.7GB free with the desktop running), so the same-process
   * ratio is the number to trust. */
  for (int i = 0; i < npn; i++) {
    int n = pn[i];
    if (n > ctx) { printf("[perf] prefill n=%-6d skipped (> ctx %d)\n", n, ctx); continue; }
    if (ab) {
      double b_old = 1e30, b_new = 1e30;
      int rc = 0;
      for (int r = 0; r < rounds; r++) {
        _putenv("SGL_ATTN_TWOPASS=1");
        double t0 = now_sec();
        rc = sgl_cuda_forward_seg(gpu, ids, n, 0, 0, lg);
        double dt = now_sec() - t0;
        _putenv("SGL_ATTN_TWOPASS=0");
        if (rc != 0) break;
        if (dt < b_old) b_old = dt;
        t0 = now_sec();
        rc = sgl_cuda_forward_seg(gpu, ids, n, 0, 0, lg);
        dt = now_sec() - t0;
        if (rc != 0) break;
        if (dt < b_new) b_new = dt;
      }
      if (rc != 0) { printf("[perf] prefill n=%-6d A/B FAILED rc=%d\n", n, rc); continue; }
      printf("[perf] prefill n=%-6d two-pass %10.1f ms | flash %8.1f ms | %.1fx\n",
             n, b_old * 1e3, b_new * 1e3, b_old / b_new);
      /* W8 A/B: same trick for the int8 projections (dp4a vs IMMA) */
      if (getenv("SGL_CUDA_W8")) {
        double w_old = 1e30, w_new = 1e30;
        for (int r = 0; r < rounds; r++) {
          _putenv("SGL_W8_DP4A=1");
          double t0 = now_sec();
          rc = sgl_cuda_forward_seg(gpu, ids, n, 0, 0, lg);
          double dt = now_sec() - t0;
          _putenv("SGL_W8_DP4A=0");
          if (rc != 0) break;
          if (dt < w_old) w_old = dt;
          t0 = now_sec();
          rc = sgl_cuda_forward_seg(gpu, ids, n, 0, 0, lg);
          dt = now_sec() - t0;
          if (rc != 0) break;
          if (dt < w_new) w_new = dt;
        }
        if (rc == 0)
          printf("[perf] prefill n=%-6d w8-dp4a %10.1f ms | w8-imma %8.1f ms | %.1fx\n",
                 n, w_old * 1e3, w_new * 1e3, w_old / w_new);
        else printf("[perf] prefill n=%-6d w8 A/B FAILED rc=%d\n", n, rc);
      }
      continue;
    }
    double best = 1e30;
    int rc = 0;
    for (int r = 0; r < rounds; r++) {
      double t0 = now_sec();
      rc = sgl_cuda_forward_seg(gpu, ids, n, 0, 0, lg);
      double dt = now_sec() - t0;
      if (rc != 0) break;
      if (dt < best) best = dt;
    }
    if (rc != 0) { printf("[perf] prefill n=%-6d FAILED rc=%d\n", n, rc); continue; }
    printf("[perf] prefill n=%-6d %10.1f ms  %8.1f tok/s\n", n, best * 1e3, n / best);
  }

  /* ---- decode (B sequences, one step each) ---- */
  for (int i = 0; i < nbs; i++) {
    int B = bs[i];
    if (B < 1 || B > 32) continue;
    int *toks = (int *)sgl_xmalloc(sizeof(int) * (size_t)B * 3);
    int *base = toks + B, *lens = toks + 2 * B;
    for (int b = 0; b < B; b++) { toks[b] = 1 + (b * 7) % (V - 2); base[b] = b * seg; lens[b] = 16; }
    if (base[B - 1] + lens[B - 1] + decode_steps + 1 > ctx) {
      printf("[perf] decode B=%-3d skipped (seg %d x B > ctx %d)\n", B, seg, ctx);
      sgl_xfree(toks); continue;
    }
    double best = 1e30;
    int rc = 0, steps_done = 0;
    for (int r = 0; r < rounds; r++) {
      /* warm the segment (also warms graph capture / workspace growth) */
      for (int b = 0; b < B; b++) lens[b] = 16;
      double t0 = now_sec();
      for (int s = 0; s < decode_steps; s++) {
        rc = sgl_cuda_decode_batch(gpu, toks, base, lens, B, lg);
        if (rc != 0) break;
        for (int b = 0; b < B; b++) {
          int top = 0;
          for (int j = 1; j < V; j++) if (lg[(size_t)b * V + j] > lg[(size_t)b * V + top]) top = j;
          toks[b] = top;
          lens[b]++;
        }
      }
      double dt = now_sec() - t0;
      if (rc != 0) break;
      steps_done = decode_steps;
      if (dt < best) best = dt;
    }
    if (rc != 0) { printf("[perf] decode B=%-3d FAILED rc=%d\n", B, rc); sgl_xfree(toks); continue; }
    double tps = (double)steps_done * B / best;
    printf("[perf] decode B=%-3d %10.1f ms/step  %8.1f tok/s aggregate\n", B,
           best * 1e3 / steps_done, tps);
    sgl_xfree(toks);
  }

  sgl_xfree(ids); sgl_xfree(lg);
  sgl_cuda_free(gpu);
  sgl_llama_free(cpu);
  return 0;
}
