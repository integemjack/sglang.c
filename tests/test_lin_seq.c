/* G13 验收：linear_attention 的递归状态按「序列」归属，而不是按首个 KV 页。
 *
 * 场景（radix 前缀共享）：两条序列的页表前 k 个位置指向**【同一批物理页】**（共享
 * 已算好的前缀 KV），只有后缀页各归各的。旧实现用 slot_idx[0] 当状态归属键，两条
 * 序列会抢同一个 slot → 递归状态互踩，两条流各自的输出都错。现在用 64 位序列 key：
 *   - sgl_llama_forward_seq / sgl_llama_decode_batch_seq 显式传 seq_id；
 *   - 不传时退化成页表指针（每个序列一份页表，实际调用里天然唯一）。
 *
 * 断言：
 *  1) 共享前缀 + 不同后缀，并发 decode 每一步的 logits 与「各自单独跑」逐位相同；
 *  2) 运行期间两个序列各占一个状态槽（sgl_llama_lin_slots_in_use == 2），
 *     释放后归零；
 *  3) 不共享页（首 url 不同）时结果与共享时一致（说明不是靠运气避开冲突）。
 *
 * 用法: test_lin_seq <qwen3next-model-dir>   （缺模型时打印 SKIP 并返回 0）
 */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/llama.h"
#include "sgl/port.h" /* fopen_s on POSIX（MSVC 有原生实现）*/

static int FAILURES = 0;
#define NP 8 /* 共享前缀长度（= 共享页数） */
#define ND 4 /* 各自 decode 步数 */
#define CHECK(cond, ...) do { if (!(cond)) { FAILURES++; printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                              printf(__VA_ARGS__); printf("\n"); } } while (0)

static int same_logits(const float *a, const float *b, int n, double *maxerr) {
  double e = 0;
  for (int i = 0; i < n; i++) {
    double d = fabs((double)a[i] - (double)b[i]);
    if (d > e) e = d;
  }
  if (maxerr) *maxerr = e;
  return e == 0.0;
}

/* 一条序列的全部页（每 token 一页）：返回 0 成功 */
static int alloc_pages(sgl_llama *m, int *idx, int n) {
  return sgl_llama_pages_alloc(m, idx, n);
}
static void free_pages(sgl_llama *m, int *idx, int n) {
  sgl_llama_pages_unref(m, idx, n);
}

int main(int argc, char **argv) {
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_QNEXT");
  if (!dir || !*dir) dir = "models/qwen3next-rand";
  char p[512];
  snprintf(p, sizeof(p), "%s/model.safetensors", dir);
  FILE *f = NULL;
#ifdef _MSC_VER
  if (fopen_s(&f, p, "rb") != 0) f = NULL;
#else
  f = fopen(p, "rb");
#endif
  if (!f) { printf("SKIP: no model at %s\n", dir); return 0; }
  fclose(f);

  sgl_llama *m = sgl_llama_load(dir);
  CHECK(m != NULL, "load %s", dir);
  if (!m) return 1;
  if (m->n_lin <= 0) { printf("SKIP: model has no linear_attention layers\n"); sgl_llama_free(m); return 0; }

  int V = m->vocab;
  int prompt[NP] = {9707, 1879, 785, 7290, 315, 2272, 374, 59604}; /* 任意合法 token id */
  for (int i = 0; i < NP; i++) if (prompt[i] >= V) prompt[i] = prompt[i] % V;
  int sufA[ND] = {1125, 315, 30463, 19484};
  int sufB[ND] = {36966, 59604, 374, 279};
  for (int i = 0; i < ND; i++) { if (sufA[i] >= V) sufA[i] %= V; if (sufB[i] >= V) sufB[i] %= V; }

  float *lA = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * (ND + 1));
  float *lB = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * (ND + 1));
  float *lC = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  float *lD = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  float *lb = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * 2);

  /* ---------- 1) 单独跑 A / B（各自的页，状态各一份）---------- */
  int pgA[NP + ND], pgB[NP + ND];
  CHECK(alloc_pages(m, pgA, NP + ND) == 0, "alloc A pages");
  CHECK(sgl_llama_forward_seq(m, prompt, NP, 0, pgA, 1001, lA) == 0, "A prefill");
  for (int t = 0; t < ND; t++)
    CHECK(sgl_llama_forward_seq(m, &sufA[t], 1, NP + t, pgA, 1001,
                                lA + (size_t)(t + 1) * V) == 0, "A decode %d", t);
  int inuseA = sgl_llama_lin_slots_in_use(m);
  CHECK(alloc_pages(m, pgB, NP + ND) == 0, "alloc B pages");
  CHECK(sgl_llama_forward_seq(m, prompt, NP, 0, pgB, 2002, lB) == 0, "B prefill");
  for (int t = 0; t < ND; t++)
    CHECK(sgl_llama_forward_seq(m, &sufB[t], 1, NP + t, pgB, 2002,
                                lB + (size_t)(t + 1) * V) == 0, "B decode %d", t);
  int inuseBoth = sgl_llama_lin_slots_in_use(m);
  printf("[lin] 单独跑：槽位占用 %d -> %d（两条序列各一个）\n", inuseA, inuseBoth);
  CHECK(inuseBoth >= 2, "两条序列应各占一个递归状态槽（实际 %d）", inuseBoth);
  sgl_llama_lin_release(m, 1001);
  sgl_llama_lin_release(m, 2002);
  int inuseAfter = sgl_llama_lin_slots_in_use(m);
  printf("[lin] 释放后槽位占用 = %d\n", inuseAfter);
  CHECK(inuseAfter == 0, "release 后槽位应归零（实际 %d）", inuseAfter);

  /* ---------- 2) 共享前缀页 + 不同后缀，并发 decode ---------- */
  int pgS[NP];      /* 共享的前缀物理页 */
  int pg1[ND], pg2[ND];
  CHECK(alloc_pages(m, pgS, NP) == 0, "alloc shared pages");
  CHECK(alloc_pages(m, pg1, ND) == 0 && alloc_pages(m, pg2, ND) == 0, "alloc suffix pages");
  for (int i = 0; i < NP; i++) sgl_llama_page_ref(m, pgS[i], 1); /* 第二条序列也引用这批页 */
  int t1[NP + ND], t2[NP + ND];
  memcpy(t1, pgS, sizeof(int) * NP);
  memcpy(t2, pgS, sizeof(int) * NP);
  memcpy(t1 + NP, pg1, sizeof(int) * ND);
  memcpy(t2 + NP, pg2, sizeof(int) * ND);
  printf("[lin] 共享前 %d 页：seq1[0]=%d seq2[0]=%d（旧实现会因此抢同一个槽）\n",
         NP, t1[0], t2[0]);
  CHECK(t1[0] == t2[0], "两条序列的首页必须相同（测试前提）");

  CHECK(sgl_llama_forward_seq(m, prompt, NP, 0, t1, 11, lC) == 0, "seq1 prefill");
  CHECK(sgl_llama_forward_seq(m, prompt, NP, 0, t2, 22, lD) == 0, "seq2 prefill");
  int inuseShared = sgl_llama_lin_slots_in_use(m);
  printf("[lin] 共享前缀后槽位占用 = %d（若为 1 就是踩状态了）\n", inuseShared);
  CHECK(inuseShared == 2, "共享前缀的两条序列也要各占一个槽（实际 %d）", inuseShared);
  CHECK(same_logits(lC, lA, V, NULL), "seq1 prefill logits != 单独跑");
  CHECK(same_logits(lD, lB, V, NULL), "seq2 prefill logits != 单独跑");

  int lens[2] = {NP, NP};
  int *tabs[2] = {t1, t2};
  long long ids[2] = {11, 22};
  double worst = 0;
  for (int t = 0; t < ND; t++) {
    int toks[2] = {sufA[t], sufB[t]};
    CHECK(sgl_llama_decode_batch_seq(m, toks, tabs, ids, lens, 2, lb) == 0,
          "batched decode seq %d", t);
    double e1, e2;
    int ok1 = same_logits(lb, lA + (size_t)(t + 1) * V, V, &e1);
    int ok2 = same_logits(lb + V, lB + (size_t)(t + 1) * V, V, &e2);
    if (e1 > worst) worst = e1;
    if (e2 > worst) worst = e2;
    CHECK(ok1, "step %d: seq1 batched logits != 单独跑（maxerr %g）", t, e1);
    CHECK(ok2, "step %d: seq2 batched logits != 单独跑（maxerr %g）", t, e2);
    lens[0]++; lens[1]++;
  }
  printf("[lin] 并发 decode %d 步 vs 单独跑：最大 logits 差 = %g（逐位相同=%s）\n",
         ND, worst, worst == 0.0 ? "是" : "否");
  sgl_llama_lin_release(m, 11);
  sgl_llama_lin_release(m, 22);

  /* ---------- 2b) 反向对照：故意给两条序列**同一个** seq key（= 旧实现的归属键），
   * 结果必须错 —— 证明本测试真能抓住“状态互踩”。---------- */
  {
    int n1[NP + ND], n2[NP + ND];
    CHECK(alloc_pages(m, n1, NP + ND) == 0 && alloc_pages(m, n2, NP + ND) == 0, "alloc neg");
    long long lsame[2] = {77, 77};
    int *ntabs[2] = {n1, n2};
    int nlens[2] = {NP, NP};
    CHECK(sgl_llama_forward_seq(m, prompt, NP, 0, n1, 77, lC) == 0, "neg prefill 1");
    CHECK(sgl_llama_forward_seq(m, prompt, NP, 0, n2, 77, lD) == 0, "neg prefill 2");
    int bad = 0;
    for (int t = 0; t < ND; t++) {
      int toks[2] = {sufA[t], sufB[t]};
      CHECK(sgl_llama_decode_batch_seq(m, toks, ntabs, lsame, nlens, 2, lb) == 0, "neg decode");
      if (!same_logits(lb, lA + (size_t)(t + 1) * V, V, NULL)) bad = 1;
      if (!same_logits(lb + V, lB + (size_t)(t + 1) * V, V, NULL)) bad = 1;
      nlens[0]++; nlens[1]++;
    }
    printf("[lin] 反向对照（两条序列故意共用 seq key）：结果%s（应为不同）\n",
           bad ? "不同 ✓" : "相同 ✗");
    CHECK(bad, "共用 seq key 时结果居然一致 —— 测试失去鉴别力");
    sgl_llama_lin_release(m, 77);
    free_pages(m, n1, NP + ND);
    free_pages(m, n2, NP + ND);
  }

  /* ---------- 3) 不共享页时（首页不同）结果一致，说明不是靠运气 ---------- */
  int u1[NP + ND], u2[NP + ND];
  CHECK(alloc_pages(m, u1, NP + ND) == 0 && alloc_pages(m, u2, NP + ND) == 0, "alloc unshared");
  CHECK(u1[0] != u2[0], "两条序列页表应不同");
  CHECK(sgl_llama_forward_seq(m, prompt, NP, 0, u1, 33, lC) == 0, "u1 prefill");
  CHECK(sgl_llama_forward_seq(m, prompt, NP, 0, u2, 44, lD) == 0, "u2 prefill");
  lens[0] = lens[1] = NP;
  int *utabs[2] = {u1, u2};
  long long uids[2] = {33, 44};
  for (int t = 0; t < ND; t++) {
    int toks[2] = {sufA[t], sufB[t]};
    CHECK(sgl_llama_decode_batch_seq(m, toks, utabs, uids, lens, 2, lb) == 0, "unshared decode");
    CHECK(same_logits(lb, lA + (size_t)(t + 1) * V, V, NULL), "不共享页: seq1 不一致");
    CHECK(same_logits(lb + V, lB + (size_t)(t + 1) * V, V, NULL), "不共享页: seq2 不一致");
    lens[0]++; lens[1]++;
  }
  sgl_llama_lin_release(m, 33);
  sgl_llama_lin_release(m, 44);

  free_pages(m, pgA, NP + ND);
  free_pages(m, pgB, NP + ND);
  free_pages(m, pgS, NP);
  free_pages(m, pg1, ND);
  free_pages(m, pg2, ND);
  free_pages(m, u1, NP + ND);
  free_pages(m, u2, NP + ND);
  sgl_xfree(lA); sgl_xfree(lB); sgl_xfree(lC); sgl_xfree(lD); sgl_xfree(lb);
  sgl_llama_free(m);

  if (FAILURES) { printf("test_lin_seq: %d FAILURES\n", FAILURES); return 1; }
  printf("test_lin_seq: all OK\n");
  return 0;
}