/* sg_probe.c - MinGW single-stream forward, print greedy token ids
 * usage: sg_probe.exe <model_dir> "<prompt>" [n] */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/llama.h"
#include "sgl/sampling.h"

int main(int argc, char **argv) {
  if (argc < 3) return 1;
  const char *dir = argv[1];
  const char *prompt = argv[2];
  int nnew = argc > 3 ? atoi(argv[3]) : 4;
  char p[512];
  snprintf(p, sizeof(p), "%s/tokenizer.json", dir);
  sgl_tokenizer *tok = sgl_tokenizer_load_dir(dir);
  sgl_llama *m = sgl_llama_load(dir);
  if (!tok || !m) return 2;
  int np = 0;
  int *ids = sgl_tokenizer_encode(tok, prompt, strlen(prompt), &np);
  printf("[ids]", np);
  for (int i = 0; i < np; i++) printf(" %d", ids[i]);
  printf("\n");
  int need = np + nnew + 4;
  int *pg = (int *)sgl_xmalloc(sizeof(int) * (size_t)need);
  if (sgl_llama_pages_alloc(m, pg, need) != 0) return 3;
  float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->vocab);
  if (sgl_llama_forward(m, ids, np, 0, pg, lg) != 0) return 4;
  int len = np;
  printf("[greedy]", np);
  for (int s = 0; s < nnew; s++) {
    int t = sgl_sample_greedy(lg, m->vocab);
    printf(" %d", t);
    if (s < nnew - 1 && sgl_llama_forward(m, &t, 1, len, pg, lg) != 0) break;
    len++;
  }
  printf("\n");
  return 0;
}
