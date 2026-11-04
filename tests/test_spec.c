/* test_spec.c — speculative decoding 验证
 * 不变量：greedy 下 spec（draft+target）输出与 target-only greedy 逐 token 全等
 * （每步最终 token 都是 target argmax 的数学保证）。另报接受率与加速。 */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/port.h"
#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/llama.h"
#include "sgl/sampling.h"
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
  sgl_xfree(spec_ids);

  /* ================= temperature>0：修正采样（speculative sampling） =================
   *
   * A. T=0 退化：sgl_spec_sample_generate(temperature=0) 必须与 greedy 版
   *    逐 token 相同（走同一条退化路径）。
   * B. draft==target：q̃==p̃ ⇒ 接受率必须 ≈100%（数学不变量）。
   * C. 分布对拍：弱草稿（截层）+ T=1，一号 token 的经验分布 vs 直接从 target
   *    抽样的经验分布，TV 距离 ≤ 自校准噪声底（两组独立同分布抽样之间的
   *    TV）× 1.6 + 0.02。若修正数学有错（拒绝重采样不按 (p̃-q̃)₊），分布
   *    会系统性偏离，远超噪声底。SGL_TEST_SPEC_TV_N 可调样本数（默认 96）。
   */
  {
    /* A. T=0 退化与 greedy 版逐 token 相同 */
    sgl_sampling_params sp0;
    memset(&sp0, 0, sizeof(sp0));
    sp0.temperature = 0.0f;
    uint64_t rng0 = 12345;
    int *s0 = NULL;
    int n0 = sgl_spec_sample_generate(target, draft, pids, np, max_new, 4, eos,
                                      &sp0, &rng0, &s0, NULL, NULL);
    CHECK(n0 == nr, "A: T=0 长度 n0=%d ref=%d", n0, nr);
    int same0 = n0 == nr;
    for (int i = 0; same0 && i < nr && i < n0; i++)
      if (s0[i] != ref_ids[i]) {
        printf("  A 首个分歧 @%d: %d vs %d\n", i, s0[i], ref_ids[i]);
        same0 = 0;
      }
    CHECK(same0, "A: T=0 退化输出 != greedy spec");
    if (same0) printf("[A T=0 退化 == greedy spec ✓]\n");
    sgl_xfree(s0);

    /* B. draft==target：接受率 ≈100%（CPU 批量/单 token 前向数值逐位一致，
     *    理论上严格 100%；留 99% 余量容 fp 实现） */
    sgl_sampling_params sp1;
    memset(&sp1, 0, sizeof(sp1));
    sp1.temperature = 1.0f;
    uint64_t rng1 = 777;
    int *s1 = NULL, dr1 = 0, ac1 = 0;
    int n1 = sgl_spec_sample_generate(target, target, pids, np, max_new, 4, eos,
                                      &sp1, &rng1, &s1, &dr1, &ac1);
    CHECK(n1 > 0, "B: draft==target n=%d", n1);
    CHECK(dr1 > 0 && ac1 >= dr1 * 99 / 100,
          "B: 接受率应 ≈100%%（%d/%d）", ac1, dr1);
    if (n1 > 0) {
      int ok1 = 1;
      for (int i = 0; i < n1; i++)
        if (s1[i] < 0 || s1[i] >= target->vocab) ok1 = 0;
      CHECK(ok1, "B: 输出含非法 token id");
    }
    printf("[B draft==target 接受率 %.1f%% (%d/%d) ✓]\n",
           dr1 ? 100.0 * ac1 / dr1 : 0.0, ac1, dr1);
    sgl_xfree(s1);

    /* C. 分布对拍（TV 自校准）*/
    {
      int VN = target->vocab;
      int *ha = (int *)calloc((size_t)VN, sizeof(int)); /* 直接抽样 A */
      int *hb = (int *)calloc((size_t)VN, sizeof(int)); /* 直接抽样 B（噪声底） */
      int *hs = (int *)calloc((size_t)VN, sizeof(int)); /* spec 采样 */
      int n_tv = getenv("SGL_TEST_SPEC_TV_N") ? atoi(getenv("SGL_TEST_SPEC_TV_N")) : 96;
      if (n_tv < 32) n_tv = 32;
      /* 直接抽样：一次 prefill 后反复抽（一号 token 分布 == p̃）*/
      float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)VN);
      int *pg = (int *)sgl_xmalloc(sizeof(int) * (size_t)(np + 4));
      CHECK(sgl_llama_pages_alloc(target, pg, np + 4) == 0, "C: pages");
      CHECK(sgl_llama_forward(target, pids, np, 0, pg, lg) == 0, "C: prefill");
      for (int s = 0; s < n_tv; s++) {
        uint64_t ra = 0xABCD0000u + (uint64_t)s, rb = 0x12340000u + (uint64_t)s;
        ha[sgl_sample_ex(lg, VN, &sp1, &ra, NULL, 0, NULL)]++;
        hb[sgl_sample_ex(lg, VN, &sp1, &rb, NULL, 0, NULL)]++;
      }
      sgl_llama_pages_unref(target, pg, np + 4);
      sgl_xfree(pg);
      sgl_xfree(lg);
      /* spec：每 seed 一轮（max_new=1 → 一号 token）*/
      for (int s = 0; s < n_tv; s++) {
        uint64_t rs = 0x5EED0000u + (uint64_t)s;
        int *so = NULL;
        int no = sgl_spec_sample_generate(target, draft, pids, np, 1, 4, eos,
                                          &sp1, &rs, &so, NULL, NULL);
        if (no > 0) hs[so[0]]++;
        else CHECK(0, "C: spec run %d 失败", s);
        sgl_xfree(so);
      }
      double tv_ab = 0, tv_sa = 0;
      for (int i = 0; i < VN; i++) {
        tv_ab += fabs((double)ha[i] - hb[i]);
        tv_sa += fabs((double)ha[i] - hs[i]);
      }
      tv_ab /= 2.0 * n_tv;
      tv_sa /= 2.0 * n_tv;
      double lim = 1.6 * tv_ab + 0.02;
      printf("[C 分布对拍] N=%d TV(spec,plain)=%.3f ≤ 1.6×TV(plainA,plainB)+0.02=%.3f %s\n",
             n_tv, tv_sa, lim, tv_sa <= lim ? "✓" : "✗");
      CHECK(tv_sa <= lim, "C: TV(spec,plain)=%.3f 超噪声底 %.3f", tv_sa, lim);
      free(ha); free(hb); free(hs);
    }
  }

  sgl_xfree(ref_ids); sgl_xfree(pids);
  sgl_tokenizer_free(tok);
  sgl_llama_free(draft);
  sgl_llama_free(target);
  if (FAILURES) { printf("test_spec: %d FAILURES\n", FAILURES); return 1; }
  printf("test_spec: all OK\n");
  return 0;
}
