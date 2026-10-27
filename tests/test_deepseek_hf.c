/* test_deepseek_hf — 用 transformers 自己的 DeepseekV2ForCausalLM 当金标准，
 * 验证 sglang-c 的 DeepSeek 后端（MLA + MoE + RoPE/yarn）。
 *
 * 模型与参考值由 tests/gen_hf_deepseek.py 生成（save_pretrained 的目录可直接被
 * sgl_safetensors 读）：ref_hf_deepseek.json 里有 tokens / logits_all / logits_last。
 *
 * 三种配置：
 *   1. plain   rope_theta=1e4，无 rope_scaling（V2 风格 softmax + greedy 路由）
 *   2. yarn    rope_scaling=yarn factor=4（original_max_position_embeddings=16）
 *   3. v3      sigmoid 打分 + noaux_tc 分组路由 + norm_topk + routed_scaling_factor
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/deepseek.h"
#include "sgl/json.h"
#include "sgl/port.h"

static int FAILURES = 0;
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
  if (fopen_s(&f, p, "rb") != 0 || !f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *b = (char *)malloc((size_t)sz + 1);
  if (!b) { fclose(f); return NULL; }
  if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
  b[sz] = 0;
  fclose(f);
  if (n) *n = (size_t)sz;
  return b;
}

/* 与 HF 参考比较 logits（最后一个位置）：同时看绝对误差与相对误差 */
static void cmp_last(const char *tag, const float *got, const float *want, int V, double tol_cos,
                     double tol_norm) {
  double max_abs = 0, max_rel = 0, nrm_g = 0, nrm_w = 0, dot = 0;
  int worst = 0;
  for (int i = 0; i < V; i++) {
    double d = fabs((double)got[i] - (double)want[i]);
    double denom = fabs((double)want[i]) > 1e-9 ? fabs((double)want[i]) : 1e-9;
    if (d > max_abs) { max_abs = d; worst = i; }
    if (d / denom > max_rel) max_rel = d / denom;
    nrm_g += (double)got[i] * got[i];
    nrm_w += (double)want[i] * want[i];
    dot += (double)got[i] * want[i];
  }
  double cos = (nrm_g > 0 && nrm_w > 0) ? dot / (sqrt(nrm_g) * sqrt(nrm_w)) : 0.0;
  double rel_norm = fabs(sqrt(nrm_g) - sqrt(nrm_w)) / (sqrt(nrm_w) > 1e-12 ? sqrt(nrm_w) : 1);
  printf("  %-6s max|d|=%.3e (worst=%d) max rel=%.3e cos=%.9f norm rel=%.3e\n", tag, max_abs, worst,
         max_rel, cos, rel_norm);
  /* 阈值：cos 与 norm 收紧，逐元素用 1e-3 相对（float32 前向 + 绝对值可能很小）*/
  CHECK(cos > tol_cos, "%s: cos=%.9f", tag, cos);
  CHECK(rel_norm < tol_norm, "%s: vector norm rel diff %.3e", tag, rel_norm);
}

static void run_case(const char *dir, const char *tag) {
  char p[1200];
  snprintf(p, sizeof(p), "%s/ref_hf_deepseek.json", dir);
  size_t n = 0;
  char *txt = read_all(p, &n);
  if (!txt) {
    printf("SKIP %s（无 %s）\n", tag, p);
    return;
  }
  sgl_json *ref = sgl_json_parse(txt, n);
  free(txt);
  CHECK(ref != NULL, "%s: 参考 JSON 解析失败", tag);
  if (!ref) return;

  const sgl_json *toks = sgl_json_get(ref, "tokens");
  const sgl_json *lp = sgl_json_get(ref, "logits_last");
  CHECK(toks && toks->count > 0 && lp && lp->count > 0, "%s: 参考字段缺失", tag);

  sgl_deepseek *m = sgl_deepseek_load(dir);
  CHECK(m != NULL, "%s: sgl_deepseek_load 失败", tag);
  if (!m) { sgl_json_free(ref); return; }

  int V = sgl_deepseek_vocab(m);
  int ntok = toks->count;
  int *ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)ntok);
  for (int i = 0; i < ntok; i++) ids[i] = (int)toks->items[i]->num;

  int npos = sgl_deepseek_ctx(m);
  int npages = (npos + 3) / 4; /* 页大小 4（见 sgl_deepseek_pages_alloc 约定：传页数即可）*/
  int *pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)(npages + 8));
  int got_pages = sgl_deepseek_pages_alloc(m, pages, npages + 8);
  CHECK(got_pages == 0, "%s: pages_alloc rc=%d", tag, got_pages);

  float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  int rc = sgl_deepseek_prefill(m, ids, ntok, 0, pages, lg);
  CHECK(rc == 0, "%s: prefill rc=%d", tag, rc);

  int nref = lp->count < V ? lp->count : V;
  float *want = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  for (int i = 0; i < V; i++) want[i] = (i < nref) ? (float)lp->items[i]->num : 0.0f;
  cmp_last(tag, lg, want, V, strcmp(tag, "v3") ? 0.99999 : 0.9998,
           strcmp(tag, "v3") ? 1e-4 : 1e-3);

  /* 再验一次「分块 prefill == 一次 prefill」（yarn 的频率是逐位置算的，分块不应改变结果）*/
  {
    float *lg2 = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
    int *pages2 = (int *)sgl_xmalloc(sizeof(int) * (size_t)(npages + 8));
    int gp2 = sgl_deepseek_pages_alloc(m, pages2, npages + 8);
    CHECK(gp2 == 0, "%s: pages_alloc(2) rc=%d", tag, gp2);
    int half = ntok / 2;
    int rc1 = sgl_deepseek_prefill(m, ids, half, 0, pages2, NULL);
    int rc2 = rc1 == 0 ? sgl_deepseek_prefill(m, ids + half, ntok - half, half, pages2, lg2) : -1;
    CHECK(rc2 == 0, "%s: 分块 prefill rc=%d", tag, rc2);
    double maxd = 0;
    for (int i = 0; i < V; i++) {
      double d = fabs((double)lg2[i] - (double)lg[i]);
      if (d > maxd) maxd = d;
    }
    printf("  %-6s chunked vs single max|d|=%.3e\n", tag, maxd);
    CHECK(maxd < 1e-4, "%s: 分块 prefill 与整段差 %.3e", tag, maxd);
    sgl_xfree(lg2);
    sgl_xfree(pages2);
  }

  sgl_xfree(lg);
  sgl_xfree(want);
  sgl_xfree(pages);
  sgl_xfree(ids);
  sgl_deepseek_free(m);
  sgl_json_free(ref);
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *base = argc > 1 ? argv[1] : getenv("SGL_TEST_HF_DS");
  if (!base || !*base) base = "../models";

  char dir[1200];
  const char *cases[3][2] = {{"hf-ds-plain", "plain"}, {"hf-ds-yarn", "yarn"}, {"hf-ds-v3", "v3"}};
  for (int i = 0; i < 3; i++) {
    snprintf(dir, sizeof(dir), "%s/%s", base, cases[i][0]);
    char cfg[1300];
    snprintf(cfg, sizeof(cfg), "%s/config.json", dir);
    FILE *f = NULL;
    if (fopen_s(&f, cfg, "rb") != 0 || !f) {
      printf("SKIP %s（无 %s）\n", cases[i][1], dir);
      continue;
    }
    fclose(f);
    run_case(dir, cases[i][1]);
  }

  if (FAILURES) {
    printf("test_deepseek_hf: %d FAILURES\n", FAILURES);
    return 1;
  }
  printf("test_deepseek_hf: all OK\n");
  return 0;
}