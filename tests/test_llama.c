/* test_llama.c — Llama 前向数值对照（tiny-random 权重）
 * 参考：tests/ref_llama.py 用 numpy + HF tokenizers(Rust) 独立生成
 * tests/ref_tinyllama.json（含编码 ids / top5 logits / greedy 续写）。 */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/json.h"
#include "sgl/llama.h"
#include "sgl/ops.h"
#include "sgl/sampling.h"

static int FAILURES = 0;
#define CHECK(cond, ...) do { if (!(cond)) { FAILURES++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_LLAMA");
  if (!dir || !*dir) dir = "models/tiny-llama";
  char ptmp[1024];
  snprintf(ptmp, sizeof(ptmp), "%s/model.safetensors", dir);
  FILE *f = fopen(ptmp, "rb");
  if (!f) { printf("test_llama: SKIP（没找到 %s）\n", dir); return 0; }
  fclose(f);
  /* ref 文件：argv[2] 或 SGL_TEST_LLAMA_REF（qwen 等其他模型可指向对应 ref）*/
  const char *refname = argc > 2 ? argv[2] : getenv("SGL_TEST_LLAMA_REF");
  if (!refname || !*refname) refname = "sglang-c/tests/ref_tinyllama.json"; /* 仓库根下跑 */
  snprintf(ptmp, sizeof(ptmp), "%s", refname);
  FILE *rf = fopen(ptmp, "rb");
  if (!rf) { printf("test_llama: SKIP（没找到参考 %s，先跑 ref_llama.py）\n", ptmp); return 0; }
  fclose(rf);

  sgl_llama *m = sgl_llama_load(dir);
  CHECK(m != NULL, "model load");
  if (!m) return 1;

  char tpath[1024];
  snprintf(tpath, sizeof(tpath), "%s/tokenizer.json", dir);
  sgl_tokenizer *tok = sgl_tokenizer_load(tpath);
  CHECK(tok != NULL, "tokenizer");
  if (!tok) return 1;

  /* 解析参考 JSON */
  size_t rlen;
  FILE *jf = fopen(ptmp, "rb");
  fseek(jf, 0, SEEK_END);
  long jsz = ftell(jf);
  fseek(jf, 0, SEEK_SET);
  char *jtext = (char *)sgl_xmalloc((size_t)jsz + 1);
  if (fread(jtext, 1, (size_t)jsz, jf) != (size_t)jsz) { printf("read ref failed\n"); return 1; }
  jtext[jsz] = 0;
  fclose(jf);
  sgl_json *ref = sgl_json_parse(jtext, (size_t)jsz);
  sgl_xfree(jtext);
  CHECK(ref != NULL, "ref json parse");
  if (!ref) return 1;
  const sgl_json *cases = sgl_json_get(ref, "cases");
  CHECK(cases != NULL, "ref cases");

  float *logits = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->vocab);
  int ncase = cases ? cases->count : 0;
  for (int c = 0; c < ncase; c++) {
    const sgl_json *cs = cases->items[c];
    const char *text = sgl_json_get_str(cs, "text", "");
    const sgl_json *jids = sgl_json_get(cs, "ids");
    const sgl_json *jtop = sgl_json_get(cs, "top5");
    const sgl_json *jtlog = sgl_json_get(cs, "top5_logits");
    const sgl_json *jgreedy = sgl_json_get(cs, "greedy6");

    /* 1) tokenizer 编码对照（HF tokenizers Rust 实现）*/
    int n = 0;
    int *ids = sgl_tokenizer_encode(tok, text, strlen(text), &n);
    CHECK(n == jids->count, "[%s] 编码长度 %d != %d", text, n, jids->count);
    int ids_ok = n == jids->count;
    for (int i = 0; i < n && i < jids->count; i++)
      if (ids[i] != (int)jids->items[i]->num) ids_ok = 0;
    CHECK(ids_ok, "[%s] 编码不一致（C vs HF tokenizers）", text);
    printf("[%s] ids=%s（与 HF tokenizers %s）\n", text, ids_ok ? "一致" : "不一致", ids_ok ? "✓" : "✗");

    /* 2) prefill logits top5 数值对照（numpy 独立前向）*/
    int need = n + 32;
    int *pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)need);
    CHECK(sgl_llama_pages_alloc(m, pages, need) == 0, "pages_alloc");
    CHECK(sgl_llama_forward(m, ids, n, 0, pages, logits) == 0, "forward");
    int top = sgl_sample_greedy(logits, m->vocab);
    CHECK(top == (int)jtop->items[0]->num,
          "[%s] greedy top %d != 参考 %d", text, top, (int)jtop->items[0]->num);
    /* top5 logits 的相对误差（随机权重，logits 量级 ~1e-3..1）*/
    double maxerr = 0;
    for (int i = 0; i < jtop->count && jtlog; i++) {
      int t = (int)jtop->items[i]->num;
      double want = jtlog->items[i]->num;
      double err = fabs((double)logits[t] - want);
      if (err > maxerr) maxerr = err;
    }
    CHECK(maxerr < 5e-3, "[%s] top5 logits maxerr=%g", text, maxerr);
    printf("[%s] logits top5 maxerr=%g（vs numpy）\n", text, maxerr);

    /* 3) greedy 续写对照（增量 decode 路径）+ 全量参照诊断 */
    int cur = ids[n - 1], kv = n;
    int gok = 1;
    sgl_llama *dbg = NULL; /* 增量 vs 全量诊断（首个分歧处打印） */
    for (int g = 0; g < jgreedy->count; g++) {
      int want = (int)jgreedy->items[g]->num;
      if (g > 0) {
        CHECK(sgl_llama_forward(m, &cur, 1, kv, pages, logits) == 0, "inc forward");
        kv++;
        int inc_top = sgl_sample_greedy(logits, m->vocab);
        /* 全量重算（独立实例）对照增量 */
        if (!dbg) dbg = sgl_llama_load(dir);
        if (dbg) {
          int full_len = n + g;
          int *full = (int *)sgl_xmalloc(sizeof(int) * (size_t)(full_len + 1));
          memcpy(full, ids, sizeof(int) * (size_t)n);
          for (int q = 0; q < g; q++) full[n + q] = (int)jgreedy->items[q]->num;
          int need2 = full_len + 32;
          int *pg2 = (int *)sgl_xmalloc(sizeof(int) * (size_t)need2);
          sgl_llama_pages_alloc(dbg, pg2, need2);
          float *lg2 = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->vocab);
          sgl_llama_forward(dbg, full, full_len, 0, pg2, lg2);
          int full_top = sgl_sample_greedy(lg2, m->vocab);
          double derr = 0;
          for (int v2 = 0; v2 < m->vocab; v2++) {
            double e2 = fabs((double)logits[v2] - lg2[v2]);
            if (e2 > derr) derr = e2;
          }
          if (inc_top != full_top || derr > 1e-2)
            printf("  [diag g=%d] inc_top=%d full_top=%d maxerr=%g\n", g, inc_top, full_top, derr);
          sgl_llama_pages_unref(dbg, pg2, need2); sgl_xfree(pg2); sgl_xfree(lg2); sgl_xfree(full);
        }
        cur = inc_top;
      } /* g==0：用 prefill 的 top 作为 cur（下一步的输入）*/
      if (g == 0) cur = top;
      if (cur != want) gok = 0;
    }
    if (dbg) sgl_llama_free(dbg);
    CHECK(gok, "[%s] greedy 续写不一致", text);
    printf("[%s] greedy6=%s（vs numpy 增量）\n", text, gok ? "一致 ✓" : "不一致 ✗");

    sgl_llama_pages_unref(m, pages, need);
    sgl_xfree(pages);
    sgl_xfree(ids);
  }

  /* 4) decode_batch == 逐 token forward（严格数值不变量；曾因 rope 只旋转头 0
   *    使 decode_batch 自第 2 token 起偏离——真实模型仍答对 Paris 的假绿掩盖了它，
   *    本断言用首个 case 的 ids 直接对照 B=1/B=2 两路）*/
  {
    const sgl_json *cs = cases->items[0];
    const char *text = sgl_json_get_str(cs, "text", "");
    int n = 0;
    int *ids = sgl_tokenizer_encode(tok, text, strlen(text), &n);
    int steps = 6;
    /* 参考序列：单流 forward 逐 token */
    int want[64];
    int need = n + steps + 4;
    int *pg = (int *)sgl_xmalloc(sizeof(int) * (size_t)need);
    CHECK(sgl_llama_pages_alloc(m, pg, need) == 0, "db pages");
    CHECK(sgl_llama_forward(m, ids, n, 0, pg, logits) == 0, "db prefill");
    int len = n, nw = 0;
    for (int s = 0; s < steps; s++) {
      int best = 0;
      for (int i = 0; i < m->vocab; i++) if (logits[i] > logits[best]) best = i;
      want[nw++] = best;
      if (s + 1 < steps) {
        CHECK(sgl_llama_forward(m, &best, 1, len, pg, logits) == 0, "db ref fwd");
        len++;
      }
    }
    /* decode_batch B=1 */
    {
      int got[64], ng = 0, len2 = n;
      float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->vocab);
      CHECK(sgl_llama_forward(m, ids, n, 0, pg, lg) == 0, "db prefill 1");
      for (int s = 0; s < steps; s++) {
        int best = 0;
        for (int i = 0; i < m->vocab; i++) if (lg[i] > lg[best]) best = i;
        got[ng++] = best;
        if (s + 1 < steps) {
          CHECK(sgl_llama_decode_batch(m, &best, &pg, &len2, 1, lg) == 0, "db fwd 1");
          len2++;
        }
      }
      int ok = ng == nw;
      for (int i = 0; i < ng && i < nw; i++) if (got[i] != want[i]) ok = 0;
      CHECK(ok, "decode_batch B=1 与单流不一致");
      printf("[decode_batch B=1] %s\n", ok ? "一致 ✓" : "不一致 ✗");
      sgl_xfree(lg);
    }
    /* decode_batch B=2：两条相同序列并行，各自应等于参考 */
    {
      sgl_llama_pages_unref(m, pg, need);
      CHECK(sgl_llama_pages_alloc(m, pg, need) == 0, "db pages 2");
      int *pg2 = (int *)sgl_xmalloc(sizeof(int) * (size_t)need);
      CHECK(sgl_llama_pages_alloc(m, pg2, need) == 0, "db pages 2b");
      float *lg = (float *)sgl_xmalloc(sizeof(float) * 2 * (size_t)m->vocab);
      int toks[2]; int *pgs[2] = { pg, pg2 }; int lens[2];
      int len2 = n;
      CHECK(sgl_llama_forward(m, ids, n, 0, pg, logits) == 0, "db prefill 2a");
      CHECK(sgl_llama_forward(m, ids, n, 0, pg2, logits) == 0, "db prefill 2b");
      int ok = 1;
      for (int s = 0; s < steps; s++) {
        toks[0] = want[s]; toks[1] = want[s];
        lens[0] = lens[1] = len2;
        if (s + 1 < steps) {
          CHECK(sgl_llama_decode_batch(m, toks, pgs, lens, 2, lg) == 0, "db fwd 2");
          len2++;
          /* 下一 token 的输入用各自行 logits 的 argmax（应 == want[s+1]）*/
          for (int b = 0; b < 2; b++) {
            const float *row = lg + (size_t)b * m->vocab;
            int best = 0;
            for (int i = 0; i < m->vocab; i++) if (row[i] > row[best]) best = i;
            if (best != want[s + 1]) ok = 0;
          }
        }
      }
      CHECK(ok, "decode_batch B=2 与单流不一致");
      printf("[decode_batch B=2] %s\n", ok ? "一致 ✓" : "不一致 ✗");
      sgl_llama_pages_unref(m, pg2, need);
      sgl_xfree(pg2); sgl_xfree(lg);
    }
    sgl_llama_pages_unref(m, pg, need);
    sgl_xfree(pg); sgl_xfree(ids);
  }

  sgl_json_free(ref);
  sgl_xfree(logits);
  sgl_llama_free(m);
  sgl_tokenizer_free(tok);
  if (FAILURES) { printf("test_llama: %d FAILURES\n", FAILURES); return 1; }
  printf("test_llama: all OK\n");
  return 0;
}
