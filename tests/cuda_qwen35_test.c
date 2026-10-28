/* 迷你 qwen3_5 的 CPU vs GPU logits 对比（定位移植数值 bug 用）。
 * 用法: cuda_qwen35_test <model_dir> [n_steps]
 * 编译进 CMake 或用 gcc 直接连 libsgl.a（见 CMakeLists）。*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "sgl/cuda.h"
#include "sgl/llama.h"
#include "sgl/sampling.h"
#include "sgl/bpe.h"

static double now_sec(void) {
#ifdef _WIN32
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart / (double)f.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : "models/qwen35-mini";
  int nsteps = argc > 2 ? atoi(argv[2]) : 8;

  sgl_llama *cpu = sgl_llama_load(dir);
  if (!cpu) { printf("load failed\n"); return 1; }
  printf("CPU loaded: L=%d E=%d V=%d q_dim=%d rot_dim=%d q_gate=%d n_lin=%d\n",
         cpu->n_layer, cpu->n_embd, cpu->vocab, cpu->q_dim, cpu->rot_dim,
         cpu->q_gate, cpu->n_lin);
  sgl_cuda_model *gpu = sgl_cuda_from_llama(cpu);
  if (!gpu) { printf("cuda upload failed\n"); return 1; }

  char p[512];
  snprintf(p, sizeof(p), "%s/tokenizer.json", dir);
  sgl_tokenizer *tok = sgl_tokenizer_load(p);
  const char *prompt = "hello world";
  int np = 0;
  int *ids = sgl_tokenizer_encode(tok, prompt, strlen(prompt), &np);
  int V = cpu->vocab;
  for (int i = 0; i < np; i++) ids[i] %= V; /* 与 hf_ref.py 一致的词表折叠 */
  printf("prompt=%d tokens:", np);
  for (int i = 0; i < np; i++) printf(" %d", ids[i]);
  printf("\n");
  float *lc = malloc(sizeof(float) * (size_t)V);
  float *lg = malloc(sizeof(float) * (size_t)V);
  int need = np + nsteps + 8;
  int *pg = malloc(sizeof(int) * (size_t)need);
  if (sgl_llama_pages_alloc(cpu, pg, need) != 0) { printf("pages\n"); return 1; }

  if (sgl_llama_forward(cpu, ids, np, 0, pg, lc) != 0) { printf("cpu fwd\n"); return 1; }
  if (sgl_cuda_forward(gpu, ids, np, 0, lg) != 0) { printf("gpu fwd\n"); return 1; }
  double me = 0;
  int am = 0;
  for (int i = 0; i < V; i++) {
    double e = fabs((double)lc[i] - lg[i]);
    if (e > me) { me = e; am = i; }
  }
  int g1c = sgl_sample_greedy(lc, V), g1g = sgl_sample_greedy(lg, V);
  printf("PREFILL logits maxerr=%.6g (at %d: cpu=%.6f gpu=%.6f)\n", me, am, lc[am], lg[am]);
  printf("PREFILL greedy: cpu=%d gpu=%d %s\n", g1c, g1g, g1c == g1g ? "SAME" : "DIFF");

  if (argc > 3) { /* dump logits 供 Python 侧对比 */
    FILE *fg = fopen(argv[3], "w");
    if (fg) {
      for (int i = 0; i < V; i++) fprintf(fg, "%.9g\n", lg[i]);
      fclose(fg);
      printf("wrote gpu logits -> %s\n", argv[3]);
    }
  }
  if (argc > 4) { /* 同时 dump CPU logits（三方对比：HF / CPU f32 / GPU int8）*/
    FILE *fc2 = fopen(argv[4], "w");
    if (fc2) {
      for (int i = 0; i < V; i++) fprintf(fc2, "%.9g\n", lc[i]);
      fclose(fc2);
      printf("wrote cpu logits -> %s\n", argv[4]);
    }
  }
  int gc = g1c, len = np;
  for (int s = 0; s < nsteps; s++) {
    if (sgl_llama_forward(cpu, &gc, 1, len, pg, lc) != 0) { printf("cpu step\n"); return 1; }
    if (sgl_cuda_forward(gpu, &gc, 1, len, lg) != 0) { printf("gpu step\n"); return 1; }
    me = 0; am = 0;
    for (int i = 0; i < V; i++) {
      double e = fabs((double)lc[i] - lg[i]);
      if (e > me) { me = e; am = i; }
    }
    int a = sgl_sample_greedy(lc, V), b = sgl_sample_greedy(lg, V);
    printf("step %d (in=%d len=%d): maxerr=%.6g greedy cpu=%d gpu=%d %s\n", s, gc, len, me, a, b,
           a == b ? "SAME" : "DIFF");
    gc = a;
    len++;
  }
  return 0;
}