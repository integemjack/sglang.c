/* test_yarn — YaRN 逆频率表与 attention_factor，对着 transformers 自己的
 * `modeling_rope_utils._compute_yarn_parameters` 逐值比对（tests/gen_yarn_ref.py 生成）。
 *
 * 这条链路无法用「HF 的 DeepseekV2 前向」当金标准：transformers 5.20 的
 * DeepseekV2RotaryEmbedding 会收下 rope_type=yarn 的配置但算出与 plain 相同的结果
 * （实测 logits 差 3e-8），所以参考值只能取自 llama 家族共用的 _compute_yarn_parameters。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/ops.h"
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
  if (fopen_s(&f, p, "rb") != 0 || !f) { f = fopen(p, "rb"); }
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

/* 相对误差（对接近 0 的值退化为绝对误差）*/
static double relerr(double a, double b) {
  double d = fabs(a - b);
  double s = fabs(b) > 1e-12 ? fabs(b) : 1.0;
  return d / s;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *path = argc > 1 ? argv[1] : "tests/ref_yarn.json";
  size_t n = 0;
  char *txt = read_all(path, &n);
  if (!txt) {
    printf("test_yarn: SKIP（无 %s，先跑 python tests/gen_yarn_ref.py tests/ref_yarn.json）\n",
           path);
    return 0;
  }
  sgl_json *root = sgl_json_parse(txt, n);
  free(txt);
  CHECK(root && root->type == SGL_JSON_ARRAY, "参考 JSON 解析失败");
  if (!root) return 1;

  double *inv = (double *)sgl_xmalloc(sizeof(double) * 512);
  for (int c = 0; c < root->count; c++) {
    const sgl_json *e = root->items[c];
    const char *name = sgl_json_get_str(e, "name", "?");
    int dim = sgl_json_get_int(e, "head_dim", 0);
    double base = sgl_json_get_num(e, "base", 10000.0);
    double factor = sgl_json_get_num(e, "factor", 1.0);
    double orig = sgl_json_get_num(e, "orig", 0);
    double bf = sgl_json_get_num(e, "beta_fast", 32);
    double bs = sgl_json_get_num(e, "beta_slow", 1);
    const sgl_json *tr = sgl_json_get(e, "truncate");
    int truncate = (tr && tr->type == SGL_JSON_BOOL) ? tr->boolean : 1;
    double mscale = sgl_json_get_num(e, "mscale", 0);
    double msall = sgl_json_get_num(e, "mscale_all_dim", 0);
    const sgl_json *ref_inv = sgl_json_get(e, "inv_freq");
    double ref_attn = sgl_json_get_num(e, "attention_factor", 1.0);

    int half = sgl_rope_yarn_invf(inv, dim, base, factor, bf, bs, orig, truncate);
    CHECK(half == dim / 2, "%s: invf 返回 %d（期望 %d）", name, half, dim / 2);

    double max_rel = 0;
    int worst = 0;
    if (ref_inv) {
      for (int i = 0; i < half && i < ref_inv->count; i++) {
        double want = ref_inv->items[i]->num;
        double r = relerr(inv[i], want);
        if (r > max_rel) { max_rel = r; worst = i; }
      }
    }
    double got_attn = sgl_rope_yarn_attention_factor(factor, mscale, msall);
    double attn_rel = relerr(got_attn, ref_attn);
    printf("  %-12s inv max rel=%.3e (worst=%d)  attn=%.9f (ref %.9f, rel %.2e)\n", name, max_rel,
           worst, got_attn, ref_attn, attn_rel);
    CHECK(max_rel < 1e-6, "%s: inv_freq[%d] 相对误差 %.3e", name, worst, max_rel);
    CHECK(attn_rel < 1e-9, "%s: attention_factor 相对误差 %.3e", name, attn_rel);
  }

  sgl_xfree(inv);
  sgl_json_free(root);
  if (FAILURES) {
    printf("test_yarn: %d FAILURES\n", FAILURES);
    return 1;
  }
  printf("test_yarn: all OK\n");
  return 0;
}