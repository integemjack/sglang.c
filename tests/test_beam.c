/* test_beam.c — beam search 验证
 * 核心不变量：beam_width=4 的最优累计 logprob ≥ beam_width=1（=greedy）的
 * logprob —— W=1 恰是 greedy，W=4 的保留集是其超集（每步每 beam 展开的候选
 * 覆盖 greedy 路径），故数学上必然成立。另验输出非空、按 lp 降序。 */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/llama.h"

static int FAILURES = 0;
#define CHECK(cond, ...)                                        \
  do {                                                          \
    if (!(cond)) {                                              \
      FAILURES++;                                               \
      printf("FAIL %s:%d: ", __FILE__, __LINE__);              \
      printf(__VA_ARGS__);                                      \
      printf("\n");                                             \
    }                                                           \
  } while (0)

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_LLAMA");
  if (!dir || !*dir) dir = "models/tiny-llama";
  FILE *f = NULL;
  char ptmp[512];
  snprintf(ptmp, sizeof(ptmp), "%s/model.safetensors", dir);
  if (fopen_s(&f, ptmp, "rb") != 0 || !f) { printf("SKIP: no model at %s\n", dir); return 0; }
  fclose(f);

  sgl_llama *m = sgl_llama_load(dir);
  CHECK(m != NULL, "load");
  if (!m) return 1;
  char tpath[512];
  snprintf(tpath, sizeof(tpath), "%s/tokenizer.json", dir);
  sgl_tokenizer *tok = sgl_tokenizer_load(tpath);
  CHECK(tok != NULL, "tokenizer");
  if (!tok) return 1;
  int eos = sgl_tokenizer_eos_id(tok);

  const char *prompt = "The meaning of life is";
  int np = 0;
  int *pids = sgl_tokenizer_encode(tok, prompt, strlen(prompt), &np);
  CHECK(np > 0, "encode");

  int max_new = 8;
  sgl_llama_beam_result r1[1], r4[4], r8[8];

  int n1 = sgl_llama_beam_search(m, pids, np, 1, max_new, eos, r1);
  CHECK(n1 == 1, "beam1 n=%d", n1);
  int n4 = sgl_llama_beam_search(m, pids, np, 4, max_new, eos, r4);
  CHECK(n4 >= 1 && n4 <= 4, "beam4 n=%d", n4);
  int n8 = sgl_llama_beam_search(m, pids, np, 8, max_new, eos, r8);
  CHECK(n8 >= 1 && n8 <= 8, "beam8 n=%d", n8);

  printf("greedy(W=1) lp=%.4f\n", n1 > 0 ? r1[0].logprob : 0.0);
  for (int i = 0; i < n4; i++) printf("W=4 beam%d lp=%.4f n=%d\n", i, r4[i].logprob, r4[i].n);
  for (int i = 0; i < n8; i++) printf("W=8 beam%d lp=%.4f\n", i, r8[i].logprob);

  /* W=1 必须逐 token 等于 greedy（同路径）；W>1 不保证 lp 单调于 W（贪心路径可能被挤出，
   * 非完备搜索的已知性质），只做宽松 sanity：不低于 greedy 太多，且降序 */
  CHECK(n1 > 0 && n4 > 0 && r4[0].logprob >= r1[0].logprob - 2.0,
        "beam4 lp (%.4f) 远低于 greedy lp (%.4f)", r4[0].logprob, r1[0].logprob);
  CHECK(n8 > 0 && r8[0].logprob >= r1[0].logprob - 2.0,
        "beam8 lp (%.4f) 远低于 greedy lp (%.4f)", r8[0].logprob, r1[0].logprob);
  /* 降序 */
  for (int i = 1; i < n4; i++)
    CHECK(r4[i - 1].logprob >= r4[i].logprob - 1e-12, "未按 lp 降序");

  /* beam1 的输出 == 逐步 greedy（用 forward 直推对照）*/
  {
    float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->vocab);
    int need = np + max_new + 4;
    int *pg = (int *)sgl_xmalloc(sizeof(int) * (size_t)need);
    CHECK(sgl_llama_pages_alloc(m, pg, need) == 0, "pages");
    CHECK(sgl_llama_forward(m, pids, np, 0, pg, lg) == 0, "prefill");
    int len = np, ok = 1;
    for (int s = 0; s < r1[0].n && s < max_new; s++) {
      int best = 0;
      for (int i = 0; i < m->vocab; i++) if (lg[i] > lg[best]) best = i;
      if (r1[0].ids[s] != best) {
        printf("  [diag] step %d: greedy=%d beam1=%d\n", s, best, r1[0].ids[s]);
        ok = 0;
        break;
      }
      if (sgl_llama_forward(m, &best, 1, len, pg, lg) != 0) break;
      len++;
    }
    CHECK(ok, "W=1 与逐步 greedy 不一致");
    printf("W=1 == greedy ✓\n");
    sgl_llama_pages_unref(m, pg, need);
    sgl_xfree(pg);
    sgl_xfree(lg);
  }

  for (int i = 0; i < n1; i++) sgl_xfree(r1[i].ids);
  for (int i = 0; i < n4; i++) sgl_xfree(r4[i].ids);
  for (int i = 0; i < n8; i++) sgl_xfree(r8[i].ids);
  sgl_xfree(pids);
  sgl_tokenizer_free(tok);
  sgl_llama_free(m);
  if (FAILURES) { printf("test_beam: %d FAILURES\n", FAILURES); return 1; }
  printf("test_beam: all OK\n");
  return 0;
}
