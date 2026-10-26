/* test_scheduler.c — 连续批处理调度器正确性测试
 * 核心不变量：贪心（temperature=0）下，多请求交错调度的输出 token 序列
 * 必须与单请求串行生成完全一致（逐 token 相等）——批处理只是并行分组，
 * 不改变每条序列的数学路径。另测：中途提交的新请求混入运行中的批。*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/generate.h"
#include "sgl/gpt2.h"
#include "sgl/log.h"
#include "sgl/sampling.h"
#include "sgl/scheduler.h"

#define N_PROMPT 3
static const char *PROMPTS[N_PROMPT] = {
  "The meaning of life is",
  "My name is John and I work at",
  "In the beginning",
};
static const int MAX_NEW[N_PROMPT] = { 12, 10, 8 };

static int FAILURES = 0;
#define CHECK(cond, ...) do { if (!(cond)) { FAILURES++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* 收集单请求参考输出的回调 */
typedef struct { int ids[256]; int n; } collector;
static int collect(void *user, int tok, const char *piece) {
  (void)piece;
  collector *c = (collector *)user;
  if (c->n < 256) c->ids[c->n++] = tok;
  return 0;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_MODEL");
  if (!dir || !*dir) dir = "models/gpt2";

  /* ---- 参考输出：独立加载一份模型（mmap 同一文件，页共享）---- */
  sgl_gpt2 *ref_m = sgl_gpt2_load(dir);
  if (!ref_m) { printf("SKIP: no model at %s\n", dir); return 0; }
  char tpath[512];
  snprintf(tpath, sizeof(tpath), "%s/tokenizer.json", dir);
  sgl_tokenizer *ref_tok = sgl_tokenizer_load(tpath);
  if (!ref_tok) { sgl_gpt2_free(ref_m); printf("SKIP: no tokenizer\n"); return 0; }

  collector ref[N_PROMPT];
  sgl_sampling_params sp = {0}; /* greedy */
  for (int i = 0; i < N_PROMPT; i++) {
    int n = 0;
    int *ids = sgl_tokenizer_encode(ref_tok, PROMPTS[i], strlen(PROMPTS[i]), &n);
    ref[i].n = 0;
    int got = sgl_generate(ref_m, ref_tok, ids, n, &sp, MAX_NEW[i],
                           collect, &ref[i]);
    CHECK(got == ref[i].n, "ref %d: got=%d collected=%d", i, got, ref[i].n);
    sgl_xfree(ids);
    printf("ref[%d] %-32s -> %d tokens: ", i, PROMPTS[i], ref[i].n);
    for (int k = 0; k < ref[i].n; k++) printf("%d ", ref[i].ids[k]);
    printf("\n");
  }
  sgl_gpt2_free(ref_m);
  sgl_tokenizer_free(ref_tok);

  /* ---- 调度器：A、B 先提交并推两步（各 prefill 一个），
   *    再中途提交 C，连续 step 到全部完成 ---- */
  sgl_scheduler *s = sgl_sched_new(dir, 42);
  if (!s) { printf("SKIP: scheduler init failed\n"); return 0; }

  int idA = sgl_sched_submit(s, PROMPTS[0], MAX_NEW[0], &sp);
  int idB = sgl_sched_submit(s, PROMPTS[1], MAX_NEW[1], &sp);
  CHECK(idA > 0 && idB > 0, "submit A=%d B=%d", idA, idB);
  sgl_sched_step(s); /* prefill A */
  sgl_sched_step(s); /* prefill B（A 已在运行中）*/
  int idC = sgl_sched_submit(s, PROMPTS[2], MAX_NEW[2], &sp);
  CHECK(idC > 0, "submit C=%d", idC);

  clock_t t0 = clock();
  int guard = 0;
  while (sgl_sched_nactive(s) > 0 && guard++ < 4096) sgl_sched_step(s);
  double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;

  /* ---- 取回全部结果并对号比对 ---- */
  int seen[N_PROMPT] = {0};
  int req_id, n_ids;
  int *ids;
  char *text;
  while (sgl_sched_poll(s, &req_id, &text, &ids, &n_ids, NULL, NULL)) {
    int which = req_id == idA ? 0 : req_id == idB ? 1 : req_id == idC ? 2 : -1;
    CHECK(which >= 0, "unknown req id %d", req_id);
    if (which >= 0) {
      seen[which]++;
      CHECK(n_ids == ref[which].n, "req %d: n_ids=%d want %d", req_id, n_ids, ref[which].n);
      int same = n_ids == ref[which].n;
      for (int k = 0; same && k < n_ids; k++)
        if (ids[k] != ref[which].ids[k]) same = 0;
      CHECK(same, "req %d (%s): token sequence differs from serial reference", req_id, PROMPTS[which]);
      printf("req %d (%s) -> \"%s\" [%s]\n", req_id, PROMPTS[which], text,
             same ? "MATCH" : "DIFF");
    }
    sgl_xfree(text);
    sgl_xfree(ids);
  }
  for (int i = 0; i < N_PROMPT; i++)
    CHECK(seen[i] == 1, "prompt %d seen %d times", i, seen[i]);

  sgl_sched_stats st = sgl_sched_get_stats(s);
  printf("stats: prefill=%ld decode=%ld finished=%d (%.2fs, %.1f decode tok/s)\n",
         st.prefill_tokens, st.decode_tokens, st.n_finished, secs,
         secs > 0 ? st.decode_tokens / secs : 0.0);
  sgl_sched_free(s);

  if (FAILURES) { printf("test_scheduler: %d FAILURES\n", FAILURES); return 1; }
  printf("test_scheduler: all OK\n");
  return 0;
}
