/* sgl cuda_cli.c - standalone CUDA inference CLI (MSVC build, links sgl_cuda)
 * usage: cuda_cli.exe <model_dir> "<prompt>" [max_new] [temperature] */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/cuda.h"
#include "sgl/llama.h"
#include "sgl/sampling.h"

static double now_sec(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <model_dir> \"<prompt>\" [max_new=48] [temp=0]\n", argv[0]);
    return 1;
  }
  const char *dir = argv[1];
  const char *prompt = argv[2];
  char *prompt_load = NULL;
  if (prompt[0] == '@') {
    /* @file: read the prompt from a file (long prompts exceed argv limits) */
    FILE *pf = fopen(prompt + 1, "rb");
    if (!pf) { fprintf(stderr, "cannot open %s\n", prompt + 1); return 1; }
    fseek(pf, 0, SEEK_END);
    long sz = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    prompt_load = (char *)malloc((size_t)sz + 1);
    if (!prompt_load) return 1;
    if (fread(prompt_load, 1, (size_t)sz, pf) != (size_t)sz) { return 1; }
    prompt_load[sz] = 0;
    fclose(pf);
    prompt = prompt_load;
  }
  int max_new = argc > 3 ? atoi(argv[3]) : 48;
  float temp = argc > 4 ? (float)atof(argv[4]) : 0.0f;

  double t0 = now_sec();
  sgl_llama *cpu = sgl_llama_load(dir);
  if (!cpu) return 2;
  printf("[load] cpu model %.2fs (L=%d V=%d)\n", now_sec() - t0, cpu->n_layer, cpu->vocab);

  t0 = now_sec();
  sgl_cuda_model *gpu = sgl_cuda_from_llama(cpu);
  if (!gpu) return 2;
  printf("[upload] %.2fs, free VRAM %ld MB\n", now_sec() - t0, sgl_cuda_free_mem_mb());

  char tpath[1024];
  snprintf(tpath, sizeof(tpath), "%s/tokenizer.json", dir);
  sgl_tokenizer *tok = sgl_tokenizer_load(tpath);
  if (!tok) return 2;

  int np = 0;
  int *ids = sgl_tokenizer_encode(tok, prompt, strlen(prompt), &np);
  if (getenv("SGL_DEBUG_IDS")) {
    printf("[ids]", np);
    for (int i = 0; i < np; i++) printf(" %d", ids[i]);
    printf("\n");
  }
  printf("[prompt] %d tokens\n", np);
  if (np <= 0) return 2;

  float *lg = (float *)malloc(sizeof(float) * (size_t)cpu->vocab);
  int dbg_gap = getenv("SGL_DEBUG_GAP") != NULL; /* t1: print top1/top2 gap per step */
  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp));
  sp.temperature = temp;
  sp.seed = 12345;
  uint64_t rng = 0x9E3779B97F4A7C15ULL;

  fputs(prompt, stdout);
  fflush(stdout);
  t0 = now_sec();
  if (sgl_cuda_forward(gpu, ids, np, 0, lg) != 0) return 2;
  int produced = 0;
  int eos = sgl_tokenizer_eos_id(tok);
  int full_recomp = getenv("SGL_FULL_RECOMP") != NULL;
  for (int s = 0; s < max_new; s++) {
    if (dbg_gap) {
      int t1 = 0, t2 = -1;
      for (int i = 1; i < (int)cpu->vocab; i++) {
        if (lg[i] > lg[t1]) { t2 = t1; t1 = i; }
        else if (t2 < 0 || lg[i] > lg[t2]) { t2 = i; }
      }
      printf("[gap] step %d: top1=%d gap=%.4g\n", s, t1,
             t2 >= 0 ? lg[t1] - lg[t2] : 0.0f);
    }
    int t = sp.temperature > 0 ? sgl_sample(lg, cpu->vocab, &sp, &rng)
                               : sgl_sample_greedy(lg, cpu->vocab);
    if (t == eos) break;
    if (getenv("SGL_DEBUG_IDS")) printf("%d ", t);
    char *piece = sgl_tokenizer_decode(tok, &t, 1);
    fputs(piece ? piece : "", stdout);
    fflush(stdout);
    sgl_xfree(piece);
    produced++;
    if (full_recomp) {
      ids[np + produced - 1] = t;
      if (sgl_cuda_forward(gpu, ids, np + produced, 0, lg) != 0) break;
    } else {
      /* token t goes at position np+produced-1 (produced already counted it);
       * the OLD code used np+produced — skipping position np entirely, which
       * fed garbage into attention from the first decode step on. */
      if (sgl_cuda_forward(gpu, &t, 1, np + produced - 1, lg) != 0) break;
    }
  }
  double dt = now_sec() - t0;
  printf("\n[done] %d tokens in %.2fs = %.1f tok/s (GPU)\n", produced, dt, produced / dt);

  free(lg);
  free(prompt_load);
  sgl_xfree(ids);
  sgl_tokenizer_free(tok);
  sgl_cuda_free(gpu);
  sgl_llama_free(cpu);
  return 0;
}
