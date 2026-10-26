/* test_lora_multi.c - 多 LoRA 运行时热切换（方案 B：多实例池）验证
 * invariant: 同一调度器内多个 LoRA 实例（base+N 份合并副本）按请求名路由，
 * 各实例的 greedy 输出必须与「单独加载该 adapter 的 numpy 参考」全等；
 * 并发混跑（不同实例请求交错进同一连续批处理）结果与串行一致。
 * adapter 由 tests/gen_lora.py --seed 1/2 生成，参考由 tests/ref_lora.py 生成。
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/port.h"
#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/scheduler.h"

static int FAILURES = 0;
#define CHECK(cond, ...)                                                  \
  do {                                                                    \
    if (!(cond)) {                                                        \
      FAILURES++;                                                         \
      printf("FAIL %s:%d: ", __FILE__, __LINE__);                         \
      printf(__VA_ARGS__);                                                \
      printf("\n");                                                       \
    }                                                                     \
  } while (0)

static int file_exists(const char *p) {
  FILE *f = fopen(p, "rb");
  if (!f) return 0;
  fclose(f);
  return 1;
}

/* 读参考 json 的 greedy6（6 个 token id），失败返回 NULL */
static int *load_greedy6(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); sgl_xfree(buf); return NULL; }
  fclose(f);
  buf[sz] = 0;
  sgl_json *j = sgl_json_parse(buf, (size_t)sz);
  sgl_xfree(buf);
  if (!j) return NULL;
  const sgl_json *g6 = sgl_json_get(j, "greedy6");
  int *out = NULL;
  if (g6 && g6->type == SGL_JSON_ARRAY && g6->count >= 6) {
    out = (int *)sgl_xmalloc(sizeof(int) * 6);
    for (int i = 0; i < 6; i++) out[i] = (int)g6->items[i]->num;
  }
  sgl_json_free(j);
  return out;
}

/* 跑完调度器并按请求 id 收集生成 token（调用方 sgl_xfree）*/
static int *collect(sgl_scheduler *s, int req_id, int *n_out) {
  int guard = 0;
  while (sgl_sched_nactive(s) > 0 && guard++ < 100000) sgl_sched_step(s);
  for (;;) {
    int rid, n, npt, nhit;
    char *text;
    int *ids;
    if (!sgl_sched_poll(s, &rid, &text, &ids, &n, &npt, &nhit)) return NULL;
    if (rid == req_id) {
      sgl_xfree(text);
      *n_out = n;
      return ids;
    }
    sgl_xfree(text);
    sgl_xfree(ids);
  }
}

static int check_ref(const int *got, int n, const int *ref, const char *tag) {
  CHECK(n >= 6, "%s: only %d tokens generated", tag, n);
  if (n < 6) return 1;
  int ok = 1;
  for (int i = 0; i < 6; i++)
    if (got[i] != ref[i]) { ok = 0; break; }
  CHECK(ok, "%s: greedy6 mismatch: got %d,%d,%d,%d,%d,%d want %d,%d,%d,%d,%d,%d",
        tag, got[0], got[1], got[2], got[3], got[4], got[5],
        ref[0], ref[1], ref[2], ref[3], ref[4], ref[5]);
  return ok ? 0 : 1;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_LLAMA");
  const char *lora1 = argc > 2 ? argv[2] : getenv("SGL_TEST_LORA_S1");
  const char *lora2 = argc > 3 ? argv[3] : getenv("SGL_TEST_LORA_S2");
  const char *ref1 = argc > 4 ? argv[4] : getenv("SGL_TEST_LORA_REF_S1");
  const char *ref2 = argc > 5 ? argv[5] : getenv("SGL_TEST_LORA_REF_S2");
  if (!dir || !*dir) dir = "models/tiny-llama";
  if (!lora1 || !*lora1) lora1 = "models/test-lora-s1";
  if (!lora2 || !*lora2) lora2 = "models/test-lora-s2";
  if (!ref1 || !*ref1) ref1 = "sglang-c/tests/ref_lora_s1.json";
  if (!ref2 || !*ref2) ref2 = "sglang-c/tests/ref_lora_s2.json";

  char p[512];
  snprintf(p, sizeof(p), "%s/adapter.safetensors", lora1);
  if (!file_exists(p)) { printf("SKIP: no adapter at %s (run gen_lora.py --seed 1)\n", lora1); return 0; }
  snprintf(p, sizeof(p), "%s/adapter.safetensors", lora2);
  if (!file_exists(p)) { printf("SKIP: no adapter at %s (run gen_lora.py --seed 2)\n", lora2); return 0; }
  if (!file_exists(ref1)) { printf("SKIP: no ref %s (run ref_lora.py)\n", ref1); return 0; }
  if (!file_exists(ref2)) { printf("SKIP: no ref %s (run ref_lora.py)\n", ref2); return 0; }

  int *want1 = load_greedy6(ref1);
  int *want2 = load_greedy6(ref2);
  CHECK(want1 && want2, "load greedy6 refs");
  if (!want1 || !want2) return 1;

  /* base 实例必须加载干净权重（无 SGL_LORA 合并）*/
  _putenv_s("SGL_LORA", "");
  sgl_scheduler *s = sgl_sched_new(dir, 0);
  CHECK(s, "scheduler (base %s)", dir);
  if (!s) return 1;
  CHECK(sgl_sched_find_lora(s, "s1") < 0, "s1 not present before add");
  CHECK(sgl_sched_add_lora(s, "s1", dir, lora1) >= 1, "add_lora s1");
  CHECK(sgl_sched_add_lora(s, "s2", dir, lora2) >= 1, "add_lora s2");
  CHECK(sgl_sched_find_lora(s, "s1") == 1, "find s1");
  CHECK(sgl_sched_find_lora(s, "s2") == 2, "find s2");
  CHECK(sgl_sched_find_lora(s, NULL) == 0, "find NULL = base");
  CHECK(sgl_sched_find_lora(s, "nope") < 0, "find unknown");
  CHECK(sgl_sched_add_lora(s, "s1", dir, lora1) < 0, "duplicate name rejected");
  CHECK(sgl_sched_submit_lora(s, "Hello world", 8, NULL, "nope") < 0, "unknown lora submit -1");

  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp)); /* temperature=0 → greedy */

  /* ---- 串行：两请求各带不同 lora 名，各自 == 单独加载参考 ---- */
  int id1 = sgl_sched_submit_lora(s, "Hello world", 8, &sp, "s1");
  int id2 = sgl_sched_submit_lora(s, "Hello world", 8, &sp, "s2");
  CHECK(id1 > 0 && id2 > 0, "serial submit");
  int n1 = 0, n2 = 0;
  int *g1 = collect(s, id1, &n1);
  int *g2 = collect(s, id2, &n2);
  CHECK(g1 && g2, "serial collect");
  int r1 = check_ref(g1, n1, want1, "serial s1");
  int r2 = check_ref(g2, n2, want2, "serial s2");
  /* 两 adapter 参考本身不同（不同 seed），且路由真的发生了 */
  CHECK(memcmp(want1, want2, sizeof(int) * 6) != 0, "refs differ across seeds");
  printf("[lora_multi] serial s1 %s, s2 %s\n", r1 ? "MISMATCH" : "OK", r2 ? "MISMATCH" : "OK");

  /* base 实例仍可用（回归）*/
  int idb = sgl_sched_submit(s, "Hello world", 8, &sp);
  CHECK(idb > 0, "base submit");
  int nb = 0;
  int *gb = collect(s, idb, &nb);
  CHECK(gb && nb > 0, "base collect");
  sgl_xfree(gb);

  /* ---- 并发混跑：2×2 不同实例请求交错提交，结果与串行参考一致 ---- */
  int idc[4];
  const char *cname[4] = {"s1", "s2", "s1", "s2"};
  const int *cwant[4] = {want1, want2, want1, want2};
  for (int i = 0; i < 4; i++) {
    idc[i] = sgl_sched_submit_lora(s, "Hello world", 8, &sp, cname[i]);
    CHECK(idc[i] > 0, "mixed submit %d (%s)", i, cname[i]);
  }
  int guard = 0;
  while (sgl_sched_nactive(s) > 0 && guard++ < 100000) sgl_sched_step(s);
  int got_flags = 0;
  for (;;) {
    int rid, n, npt, nhit;
    char *text;
    int *ids;
    if (!sgl_sched_poll(s, &rid, &text, &ids, &n, &npt, &nhit)) break;
    sgl_xfree(text);
    for (int i = 0; i < 4; i++) {
      if (rid == idc[i]) {
        check_ref(ids, n, cwant[i], "mixed");
        got_flags |= (1 << i);
      }
    }
    sgl_xfree(ids);
  }
  CHECK(got_flags == 0xF, "mixed: all 4 collected (got 0x%x)", got_flags);
  printf("[lora_multi] mixed 2x2 %s\n", got_flags == 0xF ? "OK" : "INCOMPLETE");

  sgl_sched_free(s);
  sgl_xfree(g1);
  sgl_xfree(g2);
  sgl_xfree(want1);
  sgl_xfree(want2);
  if (FAILURES) { printf("test_lora_multi: %d FAILURES\n", FAILURES); return 1; }
  printf("test_lora_multi: all OK\n");
  return 0;
}
