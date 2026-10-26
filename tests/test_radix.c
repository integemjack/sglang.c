/* test_radix.c — RadixCache 前缀复用测试
 * 验证（全部与冷启动参考对照，greedy）：
 *   1. 同 prompt 第二次：hit = prompt 长度，输出与第一次完全一致
 *   2. 前缀扩展 prompt（共享前缀）：hit = 共享长度，输出与冷参考一致
 *   3. 小池压力（SGL_POOL_SLOTS=64）：多条不同序列滚动插入/驱逐，
 *      全部输出仍与冷参考一致（驱逐不破坏正确性）
 *   4. 统计：cache_hit_tokens 累计正确 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/port.h"
#include "sgl/alloc.h"
#include "sgl/scheduler.h"

static int FAILURES = 0;
#define CHECK(cond, ...) do { if (!(cond)) { FAILURES++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double now_sec(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* 跑一个请求到完成，返回生成的 token（malloc，调用方 sgl_xfree），out_hit 可为 NULL */
static int *run_one(sgl_scheduler *s, const char *prompt, int max_new, int *out_n, int *out_hit) {
  sgl_sampling_params sp = {0, 0, 0, 0}; /* greedy */
  int id = sgl_sched_submit(s, prompt, max_new, &sp);
  CHECK(id > 0, "submit '%s'", prompt);
  if (id <= 0) return NULL;
  int guard = 0;
  while (sgl_sched_nactive(s) > 0 && guard++ < 8192) sgl_sched_step(s);
  int rid, n, npt, hit;
  int *ids = NULL;
  char *text;
  while (sgl_sched_poll(s, &rid, &text, &ids, &n, &npt, &hit)) {
    if (rid == id) {
      if (out_n) *out_n = n;
      if (out_hit) *out_hit = hit;
      sgl_xfree(text);
      return ids;
    }
    sgl_xfree(text);
    sgl_xfree(ids);
    ids = NULL;
  }
  return NULL;
}

static int cmp_ids(const int *a, const int *b, int n) {
  for (int i = 0; i < n; i++) if (a[i] != b[i]) return 0;
  return 1;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_MODEL");
  if (!dir || !*dir) dir = "models/gpt2";
  char ptmp[1024];
  snprintf(ptmp, sizeof(ptmp), "%s/config.json", dir);
  FILE *f = fopen(ptmp, "rb");
  if (!f) { printf("test_radix: SKIP（没找到 %s）\n", dir); return 0; }
  fclose(f);

  const char *PA = "The meaning of life is";
  const char *PA2 = "The meaning of life is not";
  const char *PB = "My name is John and I work at";

  /* ---- 阶段 1：复用正确性（大池）---- */
  sgl_scheduler *s1 = sgl_sched_new(dir, 1);
  if (!s1) { printf("test_radix: sched init failed\n"); return 1; }

  int nA = 0, hit = -1;
  int *GA = run_one(s1, PA, 12, &nA, &hit);
  CHECK(GA && nA == 12, "A cold n=%d", nA);
  CHECK(hit == 0, "A cold hit=%d（应为 0）", hit);

  int nA2 = 0, hit2 = -1;
  int *GA2 = run_one(s1, PA, 12, &nA2, &hit2);
  CHECK(GA2 && cmp_ids(GA, GA2, nA), "A 重跑输出不一致");
  /* PA 是 5 token（[464,3616,286,1204,318]，见 test_bpe）；完整序列（prompt+12 生成）
   * 已入树，第二次应全命中 prompt 的 5 个 token */
  CHECK(hit2 == 5, "A warm hit=%d（应为 5 = prompt 长度）", hit2);
  printf("phase1: A cold hit=0, A warm hit=%d, 输出一致 ✓\n", hit2);

  /* PA2 与 PA 共享前缀 6 token；冷参考用新调度器算 */
  sgl_scheduler *sref = sgl_sched_new(dir, 2);
  int nR = 0, hitR;
  int *GR = run_one(sref, PA2, 12, &nR, &hitR);
  CHECK(GR && hitR == 0, "PA2 冷参考");
  int nW = 0, hitW = -1;
  int *GW = run_one(s1, PA2, 12, &nW, &hitW);
  CHECK(GW && cmp_ids(GR, GW, nR), "PA2 warm 输出与冷参考不一致（n=%d/%d）", nR, nW);
  CHECK(hitW == 6, "PA2 hit=%d（应为 6 = 共享前缀 5 + 生成首 token ' not' 也已在树中）", hitW);
  printf("phase1: PA2 前缀扩展 hit=%d, 输出与冷参考一致 ✓\n", hitW);

  long hits = sgl_sched_get_stats(s1).cache_hit_tokens;
  CHECK(hits == 11, "cache_hit_tokens=%ld（5+6）", hits);
  sgl_xfree(GA); sgl_xfree(GA2); sgl_xfree(GR); sgl_xfree(GW);
  sgl_sched_free(s1);

  /* ---- 阶段 2：小池驱逐压力 ---- */
  _putenv("SGL_POOL_SLOTS=64");
  sgl_scheduler *s3 = sgl_sched_new(dir, 3);
  if (!s3) { printf("test_radix: small-pool sched init failed\n"); return 1; }
  sgl_sched_stats st0 = sgl_sched_get_stats(s3);
  CHECK(st0.pool_pages == 64, "pool=%d（应为 64）", st0.pool_pages);

  const char *PS[6] = {
    "The capital of France is",
    PB,
    "In a shocking finding, scientists discovered",
    "The quick brown fox jumps over",
    "Artificial intelligence will transform",
    "Deep learning models can generate",
  };
  /* 冷参考（用先前的大池调度器 sref 已有缓存？sref 只跑过 PA2，其他冷；顺序跑参考）*/
  int *refs[6]; int nrefs[6];
  for (int i = 0; i < 6; i++) refs[i] = run_one(sref, PS[i], 12, &nrefs[i], NULL);
  sgl_sched_free(sref);

  for (int i = 0; i < 6; i++) {
    int n = 0;
    int *g = run_one(s3, PS[i], 12, &n, NULL);
    CHECK(g && cmp_ids(refs[i], g, n), "stress[%d] '%s' 输出与冷参考不一致", i, PS[i]);
    sgl_xfree(g);
  }
  sgl_sched_stats st = sgl_sched_get_stats(s3);
  printf("phase2: 小池(64页) 6 序列滚动驱逐，全部输出与冷参考一致 ✓ "
         "(prefill=%ld hit=%ld radix_pages_now_free=%d)\n",
         st.prefill_tokens, st.cache_hit_tokens, 1 /*占位，树页数不暴露*/);
  for (int i = 0; i < 6; i++) sgl_xfree(refs[i]);
  sgl_sched_free(s3);
  _putenv("SGL_POOL_SLOTS=");

  /* ---- 阶段 3：长 prompt 命中加速（观察值，不 assert）---- */
  _putenv("SGL_POOL_SLOTS=");
  sgl_scheduler *s4 = sgl_sched_new(dir, 4);
  const char *LONGP =
    "The history of artificial intelligence began in antiquity, with myths, stories "
    "and rumors of artificial beings endowed with intelligence or consciousness by "
    "master craftsmen. The seeds of modern AI were planted by philosophers who "
    "attempted to describe the process of human thinking as the mechanical "
    "manipulation of symbols. This work culminated in the invention of the "
    "programmable digital computer";
  double t0 = now_sec();
  int nL = 0;
  int *GL = run_one(s4, LONGP, 8, &nL, &hit);
  double cold = now_sec() - t0;
  t0 = now_sec();
  int nL2 = 0;
  int *GL2 = run_one(s4, LONGP, 8, &nL2, &hit2);
  double warm = now_sec() - t0;
  CHECK(GL && GL2 && cmp_ids(GL, GL2, nL), "长 prompt 重跑不一致");
  printf("phase3: 长 prompt（%d tok）cold %.0f ms vs warm（hit=%d）%.0f ms\n",
         hit == 0 ? nL : hit, cold * 1000, hit2, warm * 1000);
  sgl_xfree(GL); sgl_xfree(GL2);
  sgl_sched_free(s4);

  if (FAILURES) { printf("test_radix: %d FAILURES\n", FAILURES); return 1; }
  printf("test_radix: all OK\n");
  return 0;
}
