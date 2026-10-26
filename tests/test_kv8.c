/* test_kv8.c — int8 KV cache 验证（SGL_KV8 开关）
 * 方法：f32 KV vs int8 KV 同输入对照 logits/greedy；KV 是后置缓存
 * （写入的 k/v 是精确 f32 投影结果），per-page 对称量化往返误差 ~0.4%。 */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/port.h"
#include "sgl/alloc.h"
#include "sgl/llama.h"
#include "sgl/sampling.h"

static int FAILURES = 0;
#define CHECK(cond, ...) do { if (!(cond)) { FAILURES++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)
static void set_env(const char *k, const char *v) { _putenv_s(k, v); }

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_TINYLLAMA");
  if (!dir || !*dir) dir = "models/tinyllama";
  char p[1024];
  FILE *f = NULL;
  fopen_s(&f, p, "rb");
  snprintf(p, sizeof(p), "%s/model.safetensors", dir);
  fopen_s(&f, p, "rb");
  if (!f) { printf("test_kv8: SKIP（无 %s）\n", dir); return 0; }
  fclose(f);

  set_env("SGL_KV8", "0");
  sgl_llama *mf = sgl_llama_load(dir);
  CHECK(mf != NULL, "f32 load");
  set_env("SGL_KV8", "1");
  sgl_llama *m8 = sgl_llama_load(dir);
  CHECK(m8 != NULL, "kv8 load");
  if (!mf || !m8) return 1;
  CHECK(m8->kv8 == 1 && m8->kc8 != NULL, "kv8 storage");

  /* 长 prompt 拉开 KV 误差积累面 */
  int ids[] = {1, 450, 6593, 310, 2272, 374, 304, 11, 767, 290, 318, 13779, 13, 198,
               464, 3616, 286, 262, 11571, 286, 7031, 318, 407, 262, 976, 355};
  int n = sizeof(ids) / sizeof(ids[0]);
  int pg[128], pg8[128];
  sgl_llama_pages_alloc(mf, pg, 128);
  sgl_llama_pages_alloc(m8, pg8, 128);
  float *lf = (float *)sgl_xmalloc(sizeof(float) * (size_t)mf->vocab);
  float *l8 = (float *)sgl_xmalloc(sizeof(float) * (size_t)m8->vocab);
  sgl_llama_forward(mf, ids, n, 0, pg, lf);
  sgl_llama_forward(m8, ids, n, 0, pg8, l8);
  double maxerr = 0, refn = 0;
  for (int i = 0; i < mf->vocab; i++) {
    double e = fabs((double)lf[i] - l8[i]);
    if (e > maxerr) maxerr = e;
    refn += fabs(lf[i]);
  }
  printf("f32 vs KV8 logits（prompt %d tok）: maxerr=%.4f, mean|logit|=%.4f\n",
         n, maxerr, refn / mf->vocab);
  int topf = sgl_sample_greedy(lf, mf->vocab);
  int top8 = sgl_sample_greedy(l8, m8->vocab);
  printf("greedy: f32=%d kv8=%d %s\n", topf, top8, topf == top8 ? "一致" : "不同");
  CHECK(topf == top8, "kv8 greedy 不一致 f32=%d kv8=%d", topf, top8);

  /* 增量 decode 8 步（继续对照，覆盖 kv8 的增量写路径）*/
  int cf = topf, c8 = top8, kv = n, agree = 0;
  for (int s = 0; s < 8; s++) {
    sgl_llama_forward(mf, &cf, 1, kv, pg, lf);
    sgl_llama_forward(m8, &c8, 1, kv, pg8, l8);
    cf = sgl_sample_greedy(lf, mf->vocab);
    c8 = sgl_sample_greedy(l8, m8->vocab);
    if (cf == c8) agree++;
    kv++;
  }
  printf("增量 8 步 greedy 一致 %d/8\n", agree);
  CHECK(agree >= 7, "kv8 增量贪心一致率 %d/8", agree);

  sgl_xfree(lf); sgl_xfree(l8);
  sgl_llama_free(mf);
  sgl_llama_free(m8);
  if (FAILURES) { printf("test_kv8: %d FAILURES\n", FAILURES); return 1; }
  printf("test_kv8: all OK\n");
  return 0;
}
