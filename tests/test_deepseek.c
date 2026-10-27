/* test_deepseek.c — DeepSeek-V2/V3 完整前向（MLA + MoE）数值正确性，对照 numpy 参考
 * 参考由 tests/gen_random_deepseek.py 生成（模型目录 ../models/deepseek-rand{,-v3,-ql0} 下的
 * deepseek_ref.json，全程 float64，按 HF modeling_deepseek_v2/v3.py 的非吸收式公式；C 侧是吸收式 MLA，
 * 两者数学等价，差异只剩 f32 舍入）。
 *
 * 覆盖：
 *   1) 维度 / detect / 分页池语义（alloc/ref/unref/nfree，同 llama）
 *   2) 全序列 prefill（forward_n）：n 个位置的 logits 逐元素比对
 *   3) 压缩 KV cache 行（[latent | k_pe]，layer 0）逐元素比对
 *   4) 增量 decode：prefill 1 token 后逐 token 前向，每步 logits 对比参考对应行
 *      （= prefill 与增量两条路径走过的 KV cache 一致）
 *   5) 分块 prefill（2+3+4）与一次 prefill 的等价性
 *   6) decode_batch：两条不同前缀长度的序列各 1 token，结果与参考对应行一致
 *
 * 用法：test_deepseek [model_dir]
 *   不给参数时依次找 ../models/deepseek-rand{,-v3,-ql0,-3d,-il}（环境变量 SGL_MODELS_DIR 也认）；
 *   一个都找不到就打印 SKIP 并返回 0（干净 clone 里不红）。
 */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/deepseek.h"
#include "sgl/json.h"
#include "sgl/log.h"
#include "sgl/port.h" /* fopen_s on POSIX（MSVC 有原生实现）*/

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                    \
  do {                                      \
    g_checks++;                             \
    if (!(cond)) {                          \
      g_fail++;                             \
      printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
      printf(__VA_ARGS__);                  \
      printf("\n");                         \
    }                                       \
  } while (0)

/* 最大绝对误差 / 参考最大绝对值（相对误差），同时返回最大绝对误差 */
static double max_rel_(const float *a, const double *b, int n, double *maxabs) {
  double m = 0, refmax = 0;
  for (int i = 0; i < n; i++) {
    double d = fabs((double)a[i] - b[i]);
    if (d > m) m = d;
    if (fabs(b[i]) > refmax) refmax = fabs(b[i]);
  }
  if (maxabs) *maxabs = m;
  if (refmax < 1e-12) refmax = 1e-12;
  return m / refmax;
}

static double max_rel_f_(const float *a, const float *b, int n, double *maxabs) {
  double m = 0, refmax = 0;
  for (int i = 0; i < n; i++) {
    double d = fabs((double)a[i] - (double)b[i]);
    if (d > m) m = d;
    if (fabs((double)b[i]) > refmax) refmax = fabs((double)b[i]);
  }
  if (maxabs) *maxabs = m;
  if (refmax < 1e-12) refmax = 1e-12;
  return m / refmax;
}

/* ref JSON 里 logits_all 的第 t 行 / cache_l0 的第 t 行（转 double 到 buf） */
static void ref_row_(const sgl_json *arr, int row, int cols, double *buf) {
  const sgl_json *r = arr->items[row];
  for (int i = 0; i < cols; i++) buf[i] = r->items[i]->num;
}

static char *join_(const char *a, const char *b) {
  size_t la = strlen(a), lb = strlen(b);
  char *s = (char *)sgl_xmalloc(la + lb + 2);
  memcpy(s, a, la);
  s[la] = '/';
  memcpy(s + la + 1, b, lb + 1);
  return s;
}

static char *read_file_(const char *path, size_t *len) {
  FILE *f = NULL;
  if (fopen_s(&f, path, "rb") != 0 || !f) f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0) { fclose(f); return NULL; }
  char *b = (char *)sgl_xmalloc((size_t)sz + 1);
  size_t got = fread(b, 1, (size_t)sz, f);
  fclose(f);
  b[got] = 0;
  if (len) *len = got;
  return b;
}

static int run_model(const char *dir) {
  printf("[deepseek] %s\n", dir);
  char *refpath = join_(dir, "deepseek_ref.json");
  size_t rlen = 0;
  char *rtxt = read_file_(refpath, &rlen);
  sgl_xfree(refpath);
  if (!rtxt) {
    printf("  SKIP: 缺 deepseek_ref.json（先跑 tests/gen_random_deepseek.py）\n");
    return 0;
  }
  sgl_json *ref = sgl_json_parse(rtxt, rlen);
  sgl_xfree(rtxt);
  if (!ref) {
    printf("  FAIL: deepseek_ref.json 解析失败\n");
    g_fail++;
    return 1;
  }
  const sgl_json *jtokens = sgl_json_get(ref, "tokens");
  const sgl_json *jlogits = sgl_json_get(ref, "logits_all");
  const sgl_json *jcache = sgl_json_get(ref, "cache_l0");
  if (!jtokens || !jlogits || !jcache) {
    printf("  FAIL: 参考缺 tokens/logits_all/cache_l0\n");
    g_fail++;
    sgl_json_free(ref);
    return 1;
  }
  int n = jtokens->count;
  int tokens[256];
  if (n > 256) { printf("  FAIL: n_pos>256\n"); g_fail++; sgl_json_free(ref); return 1; }
  for (int i = 0; i < n; i++) tokens[i] = (int)jtokens->items[i]->num;

  /* ---- 1) 加载 + 维度 ---- */
  CHECK(sgl_deepseek_detect(dir) == 1, "detect() 应识别为 DeepSeek");
  sgl_deepseek *m = sgl_deepseek_load(dir);
  if (!m) {
    printf("  FAIL: sgl_deepseek_load 失败\n");
    g_fail++;
    sgl_json_free(ref);
    return 1;
  }
  int V = sgl_deepseek_vocab(m), C = sgl_deepseek_ctx(m);
  int stride = sgl_deepseek_kv_stride(m), E = sgl_deepseek_hidden(m);
  const sgl_json *meta = sgl_json_get(ref, "meta");
  int refV = meta ? (int)sgl_json_get_num(meta, "vocab", 0) : 0;
  int refStride = meta ? (int)sgl_json_get_num(meta, "stride", 0) : 0;
  CHECK(V == refV, "vocab %d != 参考 %d", V, refV);
  CHECK(stride == refStride, "stride %d != 参考 %d", stride, refStride);
  CHECK(n <= C, "n_pos %d > ctx %d", n, C);
  printf("  dims: V=%d ctx=%d stride=%d hidden=%d layers(moe)=%d/%d n_pos=%d\n", V, C, stride, E,
         m->n_moe, m->n_layer, n);

  /* ---- 2) 分页池语义（同 llama）---- */
  {
    int nf0 = sgl_deepseek_nfree(m), pool = sgl_deepseek_pool_pages(m);
    int idx[8];
    CHECK(pool >= n + 8, "池太小 %d", pool);
    CHECK(sgl_deepseek_pages_alloc(m, idx, 4) == 0, "pages_alloc 失败");
    CHECK(sgl_deepseek_nfree(m) == nf0 - 4, "alloc 后 nfree %d != %d", sgl_deepseek_nfree(m), nf0 - 4);
    CHECK(sgl_deepseek_page_refs(m, idx[0]) == 1, "新页 refs 应为 1");
    sgl_deepseek_page_ref(m, idx[0], 1);
    CHECK(sgl_deepseek_page_refs(m, idx[0]) == 2, "page_ref +1 后应 2");
    sgl_deepseek_pages_unref(m, idx, 4);
    CHECK(sgl_deepseek_page_refs(m, idx[0]) == 1, "共享页 unref 后应仍为 1");
    CHECK(sgl_deepseek_nfree(m) == nf0 - 1, "unref 后 nfree %d != %d", sgl_deepseek_nfree(m), nf0 - 1);
    sgl_deepseek_pages_unref(m, idx, 1);
    CHECK(sgl_deepseek_page_refs(m, idx[0]) == 0, "最后一引用 unref 后应 0");
    CHECK(sgl_deepseek_nfree(m) == nf0, "全部归还后 nfree %d != %d", sgl_deepseek_nfree(m), nf0);
    CHECK(sgl_deepseek_pages_alloc(m, idx, pool + 1) != 0, "超池申请应失败");
  }

  float *lg_all = (float *)sgl_xmalloc(sizeof(float) * (size_t)n * V);
  int *idx = (int *)sgl_xmalloc(sizeof(int) * (size_t)n);
  double *dbuf = (double *)sgl_xmalloc(sizeof(double) * (size_t)(V > stride ? V : stride));
  double worst = 0, worst_abs = 0;

  /* ---- 3) 全序列 prefill（forward_n）---- */
  CHECK(sgl_deepseek_pages_alloc(m, idx, n) == 0, "prefill pages_alloc 失败");
  CHECK(sgl_deepseek_forward_n(m, tokens, n, 0, idx, lg_all) == 0, "forward_n 失败");
  for (int t = 0; t < n; t++) {
    ref_row_(jlogits, t, V, dbuf);
    double ma;
    double r = max_rel_(lg_all + (size_t)t * V, dbuf, V, &ma);
    if (r > worst) { worst = r; worst_abs = ma; }
    CHECK(r < 1e-4, "prefill logits[%d] 相对误差 %.3e >= 1e-4（abs %.3e）", t, r, ma);
  }
  printf("  prefill logits: 最大相对误差 %.3e（abs %.3e）\n", worst, worst_abs);

  /* ---- 4) 压缩 KV 行（layer 0：latent | k_pe）---- */
  {
    double w = 0, wa = 0;
    for (int t = 0; t < n; t++) {
      const float *row = m->kc + (size_t)idx[t] * stride; /* layer 0 */
      ref_row_(jcache, t, stride, dbuf);
      double ma;
      double r = max_rel_(row, dbuf, stride, &ma);
      if (r > w) { w = r; wa = ma; }
      CHECK(r < 1e-4, "cache_l0[%d] 相对误差 %.3e >= 1e-4（abs %.3e）", t, r, ma);
    }
    printf("  KV cache 行: 最大相对误差 %.3e（abs %.3e）\n", w, wa);
  }
  sgl_deepseek_pages_unref(m, idx, n);

  /* ---- 5) 增量 decode：1 token prefill + 逐 token 前向 ---- */
  CHECK(sgl_deepseek_pages_alloc(m, idx, n) == 0, "decode pages_alloc 失败");
  {
    float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
    double w = 0, wa = 0;
    for (int t = 0; t < n; t++) {
      int rc = sgl_deepseek_prefill(m, tokens + t, 1, t, idx, lg);
      CHECK(rc == 0, "增量 prefill t=%d 返回 %d", t, rc);
      float *dst = lg_all + (size_t)t * V;
      memcpy(dst, lg, sizeof(float) * (size_t)V); /* 复用缓冲做下一步的自洽检查 */
      ref_row_(jlogits, t, V, dbuf);
      double ma;
      double r = max_rel_(lg, dbuf, V, &ma);
      if (r > w) { w = r; wa = ma; }
      CHECK(r < 1e-4, "增量 decode logits[%d] 相对误差 %.3e >= 1e-4", t, r);
    }
    printf("  增量 decode: 最大相对误差 %.3e（abs %.3e）\n", w, wa);
    sgl_xfree(lg);
  }
  sgl_deepseek_pages_unref(m, idx, n);

  /* ---- 6) 分块 prefill 与一次 prefill 等价 ---- */
  {
    float *lg_part = (float *)sgl_xmalloc(sizeof(float) * (size_t)n * V);
    int *idx2 = (int *)sgl_xmalloc(sizeof(int) * (size_t)n);
    CHECK(sgl_deepseek_pages_alloc(m, idx2, n) == 0, "chunk pages_alloc 失败");
    int pos = 0, ok = 1;
    const int chunks[3] = {2, 3, 4};
    for (int c = 0; c < 3 && pos < n; c++) {
      int cn = chunks[c];
      if (pos + cn > n) cn = n - pos;
      ok &= (sgl_deepseek_forward_n(m, tokens + pos, cn, pos, idx2, lg_part + (size_t)pos * V) == 0);
      pos += cn;
    }
    if (pos < n) ok &= (sgl_deepseek_forward_n(m, tokens + pos, n - pos, pos, idx2,
                                               lg_part + (size_t)pos * V) == 0);
    CHECK(ok, "分块 prefill 失败");
    /* 与一次 prefill 比较：先重跑一次一次 prefill（idx 已归还，用 idx2 之前的结果已覆盖）*/
    double w = 0, wa = 0;
    for (int t = 0; t < n; t++) {
      ref_row_(jlogits, t, V, dbuf);
      double ma;
      double r = max_rel_(lg_part + (size_t)t * V, dbuf, V, &ma);
      if (r > w) { w = r; wa = ma; }
    }
    printf("  分块 prefill(2+3+4): 最大相对误差 %.3e（abs %.3e）\n", w, wa);
    CHECK(w < 1e-4, "分块 prefill 与参考相对误差 %.3e >= 1e-4", w);
    /* 分块 vs 逐 token 增量的自洽（同一个 token 序列、不同片宽 → 只允许极小舍入差）*/
    double mabs;
    double r2 = max_rel_f_(lg_part, lg_all, n * V, &mabs);
    printf("  分块 prefill vs 逐 token 增量: 相对 %.3e（abs %.3e）\n", r2, mabs);
    CHECK(r2 < 1e-5, "分块与逐 token 增量不一致：%.3e", r2);
    sgl_deepseek_pages_unref(m, idx2, n);
    sgl_xfree(idx2);
    sgl_xfree(lg_part);
  }

  /* ---- 7) decode_batch：两条不同前缀长度的序列各 1 token ---- */
  {
    int mlen = n > 4 ? 3 : n - 1; /* 第二条序列的前缀长度（严格小于 n）*/
    int *ia = (int *)sgl_xmalloc(sizeof(int) * (size_t)n);
    int *ib = (int *)sgl_xmalloc(sizeof(int) * (size_t)(mlen + 1));
    CHECK(sgl_deepseek_pages_alloc(m, ia, n) == 0, "batch A alloc 失败");
    CHECK(sgl_deepseek_pages_alloc(m, ib, mlen + 1) == 0, "batch B alloc 失败");
    float *dummy = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
    CHECK(sgl_deepseek_prefill(m, tokens, n - 1, 0, ia, NULL) == 0, "batch A prompt prefill 失败");
    CHECK(sgl_deepseek_prefill(m, tokens, mlen, 0, ib, dummy) == 0, "batch B prompt prefill 失败");
    int btok[2] = {tokens[n - 1], tokens[mlen]};
    int slen[2] = {n - 1, mlen};
    int *tabs[2] = {ia, ib};
    float *lgb = (float *)sgl_xmalloc(sizeof(float) * 2 * (size_t)V);
    CHECK(sgl_deepseek_decode_batch(m, btok, tabs, slen, 2, lgb) == 0, "decode_batch 失败");
    ref_row_(jlogits, n - 1, V, dbuf);
    double ma1, r1 = max_rel_(lgb, dbuf, V, &ma1);
    ref_row_(jlogits, mlen, V, dbuf);
    double ma2, r2 = max_rel_(lgb + V, dbuf, V, &ma2);
    printf("  decode_batch: [长前缀] 相对 %.3e / [短前缀] 相对 %.3e\n", r1, r2);
    CHECK(r1 < 1e-4, "decode_batch 序列 A 相对误差 %.3e >= 1e-4", r1);
    CHECK(r2 < 1e-4, "decode_batch 序列 B 相对误差 %.3e >= 1e-4", r2);
    sgl_deepseek_pages_unref(m, ia, n);
    sgl_deepseek_pages_unref(m, ib, mlen + 1);
    sgl_xfree(ia); sgl_xfree(ib); sgl_xfree(dummy); sgl_xfree(lgb);
  }

  /* ---- 7) prefill_emb：直接吃 embedding 序列，与 token prefill 等价 ---- */
  {
    int *ie = (int *)sgl_xmalloc(sizeof(int) * (size_t)n);
    float *emb = (float *)sgl_xmalloc(sizeof(float) * (size_t)n * E);
    float *lge = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
    for (int t = 0; t < n; t++)
      memcpy(emb + (size_t)t * E, m->wte.data + (size_t)tokens[t] * E, sizeof(float) * E);
    CHECK(sgl_deepseek_pages_alloc(m, ie, n) == 0, "emb pages_alloc 失败");
    CHECK(sgl_deepseek_prefill_emb(m, emb, n, 0, ie, lge) == 0, "prefill_emb 失败");
    ref_row_(jlogits, n - 1, V, dbuf);
    double mae, re = max_rel_(lge, dbuf, V, &mae);
    printf("  prefill_emb（末位）: 相对 %.3e（abs %.3e）\n", re, mae);
    CHECK(re < 1e-4, "prefill_emb 末位相对误差 %.3e >= 1e-4", re);
    sgl_deepseek_pages_unref(m, ie, n);
    sgl_xfree(ie); sgl_xfree(emb); sgl_xfree(lge);
  }

  sgl_deepseek_free(m);
  sgl_xfree(lg_all); sgl_xfree(idx); sgl_xfree(dbuf);
  sgl_json_free(ref);
  return 0;
}

int main(int argc, char **argv) {
  sgl_log_set_level(SGL_LOG_WARN);
  if (argc > 1) {
    int rc = run_model(argv[1]);
    printf("%s: %d 项检查，%d 项失败\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : rc;
  }
  /* 不给参数：按 SGL_MODELS_DIR 起的环境变量 / 常见相对路径找三个随机测试模型 */
  const char *names[6] = {"deepseek-rand", "deepseek-rand-v3", "deepseek-rand-ql0",
                         "deepseek-rand-3d", "deepseek-rand-il", "deepseek-rand-yarn"};
  const char *md = getenv("SGL_MODELS_DIR");
  const char *pfx[4];
  int np = 0;
  if (md && *md) pfx[np++] = md;
  pfx[np++] = "../models";
  pfx[np++] = "../../models";
  pfx[np++] = "models";
  int found = 0;
  for (int i = 0; i < 6; i++) {
    for (int k = 0; k < np; k++) {
      char *dir = join_(pfx[k], names[i]);
      char *cf = join_(dir, "config.json");
      FILE *f = NULL;
      if (fopen_s(&f, cf, "rb") != 0 || !f) f = fopen(cf, "rb");
      sgl_xfree(cf);
      if (f) {
        fclose(f);
        run_model(dir);
        found++;
        sgl_xfree(dir);
        break;
      }
      sgl_xfree(dir);
    }
  }
  if (!found) {
    printf("SKIP test_deepseek：没找到 DeepSeek 测试模型（跑 tests/gen_random_deepseek.py 生成）\n");
    return 0;
  }
  printf("%s: %d 项检查，%d 项失败\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
  return g_fail ? 1 : 0;
}