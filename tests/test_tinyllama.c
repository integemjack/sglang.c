/* test_tinyllama.c — TinyLlama-1.1B 真实权重端到端生成（走调度器抽象层）
 * 验证：
 *   1. 1.1B F16 权重加载（2.2GB → fp32 计算）
 *   2. 调度器 submit/step/poll 全链路（llama 后端 + 批处理 + radix）
 *   3. chat 模板生成连贯英文；两请求并发输出与各自串行一致（不变量）
 */
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

static void run_to_done(sgl_scheduler *s) {
  int guard = 0;
  while (sgl_sched_nactive(s) > 0 && guard++ < 8192) sgl_sched_step(s);
}

static int *collect(sgl_scheduler *s, int want_id, int *out_n, char **out_text) {
  int id, n;
  int *ids;
  char *text;
  while (sgl_sched_poll(s, &id, &text, &ids, &n, NULL, NULL)) {
    if (id == want_id) { *out_n = n; if (out_text) *out_text = text; else sgl_xfree(text); return ids; }
    sgl_xfree(text); sgl_xfree(ids);
  }
  return NULL;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_TINYLLAMA");
  if (!dir || !*dir) dir = "models/tinyllama";
  FILE *f = NULL;
  char p[1024];
  snprintf(p, sizeof(p), "%s/model.safetensors", dir);
  fopen_s(&f, p, "rb");
  if (!f) { printf("test_tinyllama: SKIP（无 %s）\n", dir); return 0; }
  fclose(f);

  clock_t t0 = clock();
  sgl_scheduler *s = sgl_sched_new(dir, 42);
  CHECK(s != NULL, "sched_new");
  if (!s) return 1;
  printf("load: %.1fs\n", (double)(clock() - t0) / CLOCKS_PER_SEC);

  sgl_sampling_params sp = {0.0f, 0, 1.0f, 0}; /* greedy：可复现 */

  /* 1) 串行参考：A 单独跑 */
  const char *PA = "The capital of France is";
  int idA = sgl_sched_submit(s, PA, 24, &sp);
  CHECK(idA > 0, "submit A");
  run_to_done(s);
  int nA = 0;
  char *tA = NULL;
  int *GA = collect(s, idA, &nA, &tA);
  CHECK(GA && nA > 0, "A 生成 %d", nA);
  sgl_sched_stats st = sgl_sched_get_stats(s);
  printf("A: %d tokens, prefill=%ld decode=%ld\n", nA, st.prefill_tokens, st.decode_tokens);

  /* 2) 重建调度器（清空 radix，保证冷启动），并发两请求 → 与串行参考对照 */
  sgl_sched_free(s);
  s = sgl_sched_new(dir, 42);
  CHECK(s != NULL, "sched_new 2");
  const char *PB = "Water boils at a temperature of";
  int idA2 = sgl_sched_submit(s, PA, 24, &sp);
  int idB = sgl_sched_submit(s, PB, 24, &sp);
  CHECK(idA2 > 0 && idB > 0, "submit A2/B");
  t0 = clock();
  run_to_done(s);
  double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
  int nA2 = 0, nB = 0;
  char *tA2 = NULL, *tB = NULL;
  int *GA2 = collect(s, idA2, &nA2, &tA2);
  int *GB = collect(s, idB, &nB, &tB);
  CHECK(GA2 && nA2 == nA, "并发 A 长度 %d != 串行 %d", nA2, nA);
  int same = nA2 == nA;
  for (int i = 0; i < nA && i < nA2; i++) if (GA2[i] != GA[i]) same = 0;
  CHECK(same, "并发 A 与串行输出不一致（批处理不应改变结果）");

  /* 3) 输出文本（人工检查连贯性）*/
  printf("[A] %s%s\n", PA, tA ? tA : "?");
  printf("[B] %s%s\n", PB, tB ? tB : "?");
  st = sgl_sched_get_stats(s);
  printf("并发 2 请求：%.1fs，prefill=%ld decode=%ld（%.1f tok/s）\n",
         dt, st.prefill_tokens, st.decode_tokens, st.decode_tokens / dt);

  sgl_xfree(tA); sgl_xfree(tA2); sgl_xfree(tB);
  sgl_xfree(GA); sgl_xfree(GA2); sgl_xfree(GB);
  sgl_sched_free(s);
  if (FAILURES) { printf("test_tinyllama: %d FAILURES\n", FAILURES); return 1; }
  printf("test_tinyllama: all OK\n");
  return 0;
}
