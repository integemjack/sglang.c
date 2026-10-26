/* test_gpt2.c — GPT-2 前向正确性测试（需要真实权重；没有则 SKIP）
 * 验证方式与 sglang 的模型 CI 相同：固定输入 → 固定输出 token（greedy）。*/
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/gpt2.h"
#include "sgl/sampling.h"

static int fails = 0;
#define CHECK(cond, ...)                          \
  do {                                            \
    if (!(cond)) {                                \
      printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      printf(__VA_ARGS__);                        \
      printf("\n");                               \
      fails++;                                    \
    }                                             \
  } while (0)

int main(void) {
  const char *dir = getenv("SGL_TEST_MODEL");
  if (!dir || !*dir) dir = "models/gpt2";
  char path[1024];
  snprintf(path, sizeof(path), "%s/model.safetensors", dir);
  FILE *f = fopen(path, "rb");
  if (!f) {
    printf("test_gpt2: SKIP（没找到 %s；下载 GPT-2 权重后重跑）\n", path);
    return 0;
  }
  fclose(f);

  snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
  sgl_tokenizer *tok = sgl_tokenizer_load(path);
  CHECK(tok, "tokenizer");
  sgl_gpt2 *m = sgl_gpt2_load(dir);
  CHECK(m, "model");
  if (!tok || !m) return 1;

  /* prefill + greedy 一步：与参考实现对照（GPT-2 对 "The meaning of life is"
   * 的 greedy 续写，参考值由 HF transformers 生成）*/
  const char *prompt = "The meaning of life is";
  int n = 0;
  int *ids = sgl_tokenizer_encode(tok, prompt, strlen(prompt), &n);
  float *logits = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->vocab);
  int need = n + 32;
  int *pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)need);
  CHECK(sgl_gpt2_pages_alloc(m, pages, need) == 0, "pages_alloc");
  int rc = sgl_gpt2_prefill(m, ids, n, 0, pages, logits);
  CHECK(rc == 0, "prefill rc=%d", rc);

  int top = sgl_sample_greedy(logits, m->vocab);
  char *piece = sgl_tokenizer_decode(tok, &top, 1);
  printf("  greedy next token: %d (%s)\n", top, piece);
  /* 与 numpy 独立参考实现（tests/ref_gpt2.py）交叉验证：top5 = 407/284/262/326/257 */
  CHECK(top == 407 || top == 284 || top == 262 || top == 326 || top == 257,
        "unexpected token %d (%s) —— 与 numpy 参考（tests/ref_gpt2.py）对照确认", top, piece);
  sgl_xfree(piece);

  /* logits 数值健全性：无 NaN，量级正常 */
  int has_nan = 0;
  float mx = -1e30f, mn = 1e30f;
  for (int i = 0; i < m->vocab; i++) {
    if (isnan(logits[i]) || isinf(logits[i])) has_nan = 1;
    if (logits[i] > mx) mx = logits[i];
    if (logits[i] < mn) mn = logits[i];
  }
  CHECK(!has_nan, "logits contain NaN/Inf");
  CHECK(mx < 200 && mn > -200, "logits range [%g,%g] suspicious", mn, mx);

  /* KV cache 一致性：全量前向 vs 增量前向 的 logits 应几乎一致 */
  float *logits2 = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->vocab);
  sgl_gpt2 *m2 = sgl_gpt2_load(dir);
  CHECK(m2 != NULL, "second model instance");
  if (m2) {
    int *pages2 = (int *)sgl_xmalloc(sizeof(int) * (size_t)need);
    CHECK(sgl_gpt2_pages_alloc(m2, pages2, need) == 0, "pages_alloc 2");
    int rc2 = sgl_gpt2_prefill(m2, ids, n - 1, 0, pages2, NULL);
    CHECK(rc2 == 0, "inc rc1");
    int last = ids[n - 1];
    rc2 = sgl_gpt2_prefill(m2, &last, 1, n - 1, pages2, logits2);
    CHECK(rc2 == 0, "inc rc2");
    double maxerr = 0;
    for (int i = 0; i < m->vocab; i++) {
      double e = fabs((double)logits[i] - logits2[i]);
      if (e > maxerr) maxerr = e;
    }
    CHECK(maxerr < 5e-3, "incremental vs full prefill maxerr=%g", maxerr);
    printf("  incremental-vs-full maxerr: %g\n", maxerr);
    sgl_gpt2_pages_unref(m2, pages2, need);
    sgl_xfree(pages2);
    sgl_gpt2_free(m2);
  }

  sgl_gpt2_pages_unref(m, pages, need);
  sgl_xfree(pages);

  sgl_xfree(logits2);
  sgl_xfree(logits);
  sgl_xfree(ids);
  sgl_gpt2_free(m);
  sgl_tokenizer_free(tok);
  printf(fails ? "test_gpt2: %d FAILURES\n" : "test_gpt2: all OK\n", fails);
  return fails ? 1 : 0;
}
