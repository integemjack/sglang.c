/* test_spec.c — speculative decoding 验证
 * 不变量：greedy 下 spec（draft+target）输出与 target-only greedy 逐 token 全等
 * （每步最终 token 都是 target argmax 的数学保证）。另报接受率与加速。 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/port.h"
#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/llama.h"
#include "sgl/spec.h"

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

static double now_sec(void) {
  return (double)clock() / CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_TINYLLAMA");
  if (!dir || !*dir) dir = "models/tinyllama";
  FILE *f = NULL;
  char ptmp[512];
  snprintf(ptmp, sizeof(ptmp), "%s/model.safetensors", dir);
  if (fopen_s(&f, ptmp, "rb") != 0 || !f) { printf("SKIP: no model at %s\n", dir); return 0; }
  fclose(f);

  sgl_llama *target = sgl_llama_load(dir);
  CHECK(target, "target load");
  if (!target) return 1;
  /* draft：优先 SGL_DRAFT_DIR（独立小模型，如 JackFram/llama-160m，需同词表）；
   * 否则用截层自草稿（SGL_DRAFT_LAYERS，接受率≈0，仅验证正确性） */
  const char *ddir = getenv("SGL_DRAFT_DIR");
  sgl_llama *draft;
  if (ddir && *ddir) {
    draft = sgl_llama_load(ddir);
    CHECK(draft, "draft load %s", ddir);
    CHECK(draft && draft->vocab == target->vocab,
          "draft vocab=%d != target %d（需同词表）",
          draft ? draft->vocab : -1, target->vocab);
  } else {
    _putenv_s("SGL_DRAFT_LAYERS", "4");
    draft = sgl_llama_load(dir);
    _putenv_s("SGL_DRAFT_LAYERS", "");
    CHECK(draft && draft->n_layer == 4, "draft layers=%d（应 4）", draft ? draft->n_layer : -1);
  }
  if (!draft) return 1;

  char tpath[512];
  snprintf(tpath, sizeof(tpath), "%s/tokenizer.json", dir);
  sgl_tokenizer *tok = sgl_tokenizer_load(tpath);
  CHECK(tok, "tokenizer");
  int eos = sgl_tokenizer_eos_id(tok);

  const char *prompt = "The meaning of life is";
  int np = 0;
  int *pids = sgl_tokenizer_encode(tok, prompt, strlen(prompt), &np);
  int max_new = 24;

  /* 1) spec 生成 */
  int *spec_ids = NULL;
  double t0 = now_sec();
  int ns = sgl_spec_generate(target, draft, pids, np, max_new, 4, eos, &spec_ids);
  double dt_spec = now_sec() - t0;
  CHECK(ns > 0, "spec n=%d", ns);

  /* 2) target-only greedy 对照 */
  int *ref_ids = NULL;
  int nr = 0;
  {
    float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)target->vocab);
    int need = np + max_new + 4;
    int *pg = (int *)sgl_xmalloc(sizeof(int) * (size_t)need);
    CHECK(sgl_llama_pages_alloc(target, pg, need) == 0, "pages");
    double t1 = now_sec();
    CHECK(sgl_llama_forward(target, pids, np, 0, pg, lg) == 0, "prefill");
    int len = np;
    ref_ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)max_new);
    int stop_early = 0;
    for (int s = 0; s < max_new; s++) {
      int best = 0;
      for (int i = 0; i < target->vocab; i++) if (lg[i] > lg[best]) best = i;
      if (best == eos) { stop_early = 1; break; }
      ref_ids[nr++] = best;
      if (s + 1 < max_new) {
        if (sgl_llama_forward(target, &best, 1, len, pg, lg) != 0) break;
        len++;
      }
    }
    double dt_ref = now_sec() - t1;
    printf("target-only: %d tok / %.2fs（%.1f tok/s）%s\n", nr, dt_ref,
           nr / dt_ref, stop_early ? "（eos 提前停）" : "");
    sgl_llama_pages_unref(target, pg, need);
    sgl_xfree(pg);
    sgl_xfree(lg);
  }

  printf("spec        : %d tok / %.2fs（%.1f tok/s）\n", ns, dt_spec, ns / dt_spec);
  CHECK(ns == nr, "长度不一致 spec=%d ref=%d", ns, nr);
  int same = ns == nr;
  for (int i = 0; same && i < nr; i++)
    if (spec_ids[i] != ref_ids[i]) {
      printf("  首个分歧 @%d: spec=%d ref=%d\n", i, spec_ids[i], ref_ids[i]);
      same = 0;
    }
  CHECK(same, "spec 输出与 target-only greedy 不一致");
  if (same && ns > 0) {
    char *txt = sgl_tokenizer_decode(tok, spec_ids, ns);
    printf("[spec==greedy ✓] %.60s\n", txt);
    sgl_xfree(txt);
  }

  sgl_xfree(spec_ids); sgl_xfree(ref_ids); sgl_xfree(pids);
  sgl_tokenizer_free(tok);
  sgl_llama_free(draft);
  sgl_llama_free(target);
  if (FAILURES) { printf("test_spec: %d FAILURES\n", FAILURES); return 1; }
  printf("test_spec: all OK\n");
  return 0;
}
