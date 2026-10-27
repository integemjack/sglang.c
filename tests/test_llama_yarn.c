/* test_llama_yarn — llama 后端的 YaRN 路经，对着 transformers 的 LlamaForCausalLM 验证。
 *
 * 为什么单独一个测试：llama 家族的 yarn 在 HF 里是真的生效（不像 deepseek_v2 会静默忽略
 * rope_scaling），所以这里能拿到整模金标准；模型 + 参考值由
 * tests/gen_hf_llama_yarn.py 生成（rope_type=yarn, factor=4, orig=16, mscale=1/1）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/llama.h"
#include "sgl/port.h"

static int FAILURES = 0;
static void run_one(const char *dir);
#define CHECK(cond, ...)                          \
  do {                                            \
    if (!(cond)) {                                \
      FAILURES++;                                 \
      printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      printf(__VA_ARGS__);                        \
      printf("\n");                               \
    }                                             \
  } while (0)

static char *read_all(const char *p, size_t *n) {
  FILE *f = NULL;
  if (fopen_s(&f, p, "rb") != 0 || !f) f = fopen(p, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *b = (char *)malloc((size_t)sz + 1);
  if (b && fread(b, 1, (size_t)sz, f) == (size_t)sz) {
    b[sz] = 0;
    if (n) *n = (size_t)sz;
  } else {
    free(b);
    b = NULL;
  }
  fclose(f);
  return b;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dirs[2];
  const char *def[2] = {"../models/hf-llama-yarn", "../models/hf-llama-yarn-ms"};
  if (argc > 1) {
    dirs[0] = argv[1];
    dirs[1] = argc > 2 ? argv[2] : NULL;
  } else {
    dirs[0] = def[0];
    dirs[1] = def[1];
  }
  for (int k = 0; k < 2; k++) {
    if (!dirs[k]) continue;
    run_one(dirs[k]);
  }
  if (FAILURES) {
    printf("test_llama_yarn: %d FAILURES\n", FAILURES);
    return 1;
  }
  printf("test_llama_yarn: all OK\n");
  return 0;
}

/* 单个模型：读 ref_hf_llama.json → sgl_llama_forward → 比对最后一个位置的 logits */
static void run_one(const char *dir) {  char p[1200];
  snprintf(p, sizeof(p), "%s/ref_hf_llama.json", dir);
  size_t n = 0;
  char *txt = read_all(p, &n);
  if (!txt) {
    printf("SKIP test_llama_yarn（无 %s，先跑 python tests/gen_hf_llama_yarn.py）\n", p);
    return;
  }
  sgl_json *ref = sgl_json_parse(txt, n);
  free(txt);
  CHECK(ref != NULL, "参考 JSON 解析失败");
  if (!ref) return;

  const sgl_json *toks = sgl_json_get(ref, "tokens");
  const sgl_json *lp = sgl_json_get(ref, "logits_last");
  CHECK(toks && lp, "参考字段缺失");
  if (!toks || !lp) { sgl_json_free(ref); return; }

  sgl_llama *m = sgl_llama_load(dir);
  CHECK(m != NULL, "sgl_llama_load(%s) 失败", dir);
  if (!m) { sgl_json_free(ref); return; }

  /* 确认 config 里确实是 yarn：C 的逆频率表与 attention_factor 要跟 HF 的 rotay 模块一致 */
  double ref_attn = sgl_json_get_num(ref, "attention_factor", 1.0);
  printf("  attention_factor: c=%.6f ref=%.6f\n", m->rope_attn_factor, ref_attn);
  CHECK(fabs(m->rope_attn_factor - ref_attn) < 1e-6, "attention_factor %.6f != %.6f",
        m->rope_attn_factor, ref_attn);
  {
    const sgl_json *rinv = sgl_json_get(ref, "inv_freq");
    int half = m->head_dim / 2;
    const double *cinv = m->layer_invf; /* 第 0 层即可（各层基频相同）*/
    if (rinv && rinv->count > 0) {
      double max_rel = 0;
      int worst = 0;
      for (int i = 0; i < half && i < rinv->count; i++) {
        double want = rinv->items[i]->num;
        double d = fabs(cinv[i] - want) / (fabs(want) > 1e-12 ? fabs(want) : 1);
        if (d > max_rel) { max_rel = d; worst = i; }
      }
      printf("  inv_freq: max rel=%.3e (worst=%d)\n", max_rel, worst);
      CHECK(max_rel < 1e-6, "inv_freq[%d] 相对误差 %.3e", worst, max_rel);
    }
  }

  int V = m->vocab;
  int ntok = toks->count;
  int *ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)ntok);
  for (int i = 0; i < ntok; i++) ids[i] = (int)toks->items[i]->num;

  int npages = (m->n_ctx + 3) / 4 + 8;
  int *pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)npages);
  sgl_llama_pages_alloc(m, pages, npages);

  float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  int rc = sgl_llama_forward(m, ids, ntok, 0, pages, lg);
  CHECK(rc == 0, "forward rc=%d", rc);

  double max_abs = 0, dot = 0, ng = 0, nw = 0;
  for (int i = 0; i < V && i < lp->count; i++) {
    double g = lg[i], w = lp->items[i]->num;
    double d = fabs(g - w);
    if (d > max_abs) max_abs = d;
    dot += g * w;
    ng += g * g;
    nw += w * w;
  }
  double cos = (ng > 0 && nw > 0) ? dot / (sqrt(ng) * sqrt(nw)) : 0.0;
  double rel_norm = fabs(sqrt(ng) - sqrt(nw)) / (sqrt(nw) > 1e-12 ? sqrt(nw) : 1);
  printf("  yarn llama: max|d|=%.3e cos=%.9f norm rel=%.3e\n", max_abs, cos, rel_norm);
  /* 注：transformers 5.20 的 llama 前向实测不受 rope yarn 影响（同权重 yarn/plain 差 7.5e-08），
   * 所以这里能对上的只是「非 yarn 部分（模型/MLA/MLP）一致」；yarn 本身的正确性由上面的
   * inv_freq/attention_factor 逐值比对 + test_yarn 保证。*/
  CHECK(cos > 0.99999, "cos=%.9f 太低", cos);
  CHECK(rel_norm < 1e-4, "向量范数相对差 %.3e", rel_norm);

  /* 变异检查：把 yarn 关掉（rope_attn_factor=1 + 普通频率）后输出必须变——
   * 证明 yarn 真的生效了，而不是被静默忽略。*/
  {
    sgl_llama *m2 = sgl_llama_load(dir);
    if (m2) {
      m2->rope_attn_factor = 1.0;
      int half = m2->head_dim / 2;
      for (int l = 0; l < m2->n_layer; l++)
        for (int i = 0; i < half; i++)
          m2->layer_invf[(size_t)l * half + i] =
              pow(m2->rope_theta, -(double)(2 * i) / (double)m2->head_dim);
      int np2 = (m2->n_ctx + 3) / 4 + 8;
      int *pg2 = (int *)sgl_xmalloc(sizeof(int) * (size_t)np2);
      sgl_llama_pages_alloc(m2, pg2, np2);
      float *lg2 = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
      if (sgl_llama_forward(m2, ids, ntok, 0, pg2, lg2) == 0) {
        double dmax = 0;
        for (int i = 0; i < V; i++) {
          double d = fabs((double)lg2[i] - (double)lg[i]);
          if (d > dmax) dmax = d;
        }
        printf("  yarn vs plain（同一 C 引擎）: max|d|=%.3e\n", dmax);
        CHECK(dmax > 1e-4, "关掉 yarn 后输出几乎不变（%.3e）—— yarn 没生效", dmax);
      }
      sgl_xfree(lg2);
      sgl_xfree(pg2);
      sgl_llama_free(m2);
    }
  }

  sgl_xfree(lg);
  sgl_xfree(pages);
  sgl_xfree(ids);
  sgl_llama_free(m);
  sgl_json_free(ref);
}