/* test_mla.c — DeepSeek 系 MLA（吸收式）数值正确性，对照 numpy 参考
 * 参考由 tests/gen_random_mla.py 生成（非吸收式，按 HF modeling_deepseek_v2.py 公式），
 * 参考与 C 不共享任何代码路径（权重走 safetensors、参考走 numpy）。
 *
 * 覆盖：
 *   1) 全序列 prefill：out[n_pos][hidden] 逐元素比对
 *   2) 压缩 KV cache 行内容：[n_pos][kv_lora_rank+qk_rope_head_dim]（latent + k_pe）
 *   3) 增量 decode：1 个 token 一次 / 3+1... / 1+5+2+... 多种分块，输出必须与 prefill 一致
 *   4) pos 越界返回 -1
 * 容差：逐元素相对误差 < 1e-4（分母 max(|ref|, 1e-3*max|ref|)，避免近零元素放大）。
 *
 * 模型目录：argv[1] 或环境变量 SGL_TEST_MLA（默认 build-mla/mla-rand）；
 *           SGL_TEST_MLA_QLORA0 非空时再跑一个 q_lora_rank=null 的变体。 */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/mla.h"

static int FAILURES = 0;
#define CHECK(cond, ...) \
  do { \
    if (!(cond)) { \
      FAILURES++; \
      printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      printf(__VA_ARGS__); \
      printf("\n"); \
    } \
  } while (0)

static char *read_text_(const char *path) {
  FILE *f = NULL;
  if (fopen_s(&f, path, "rb") != 0 || !f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n <= 0) { fclose(f); return NULL; }
  char *buf = (char *)sgl_xmalloc((size_t)n + 1);
  size_t got = fread(buf, 1, (size_t)n, f);
  fclose(f);
  buf[got] = 0;
  return buf;
}

/* ref json 里 key 的二维数组 → dst[rows][cols]；返回 0 成功 */
static int get_mat_(const sgl_json *root, const char *key, float *dst, int rows, int cols) {
  const sgl_json *a = sgl_json_get(root, key);
  if (!a || a->type != SGL_JSON_ARRAY || a->count != rows) return -1;
  for (int i = 0; i < rows; i++) {
    const sgl_json *r = a->items[i];
    if (!r || r->type != SGL_JSON_ARRAY || r->count != cols) return -2;
    for (int j = 0; j < cols; j++) {
      const sgl_json *v = r->items[j];
      if (!v || v->type != SGL_JSON_NUMBER) return -3;
      dst[(size_t)i * cols + j] = (float)v->num;
    }
  }
  return 0;
}

/* max|a-b|、max 相对误差（分母下限 1e-3*max|ref|）、以及整体相对误差 max|d|/max|ref| */
static void err_stats_(const float *a, const float *b, int n, float *out_abs, float *out_rel,
                       float *out_norm) {
  float maxb = 0, mabs = 0, mrel = 0;
  for (int i = 0; i < n; i++) {
    float d = fabsf(b[i]);
    if (d > maxb) maxb = d;
  }
  float fl = maxb > 0 ? maxb * 1e-3f : 1e-30f;
  for (int i = 0; i < n; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > mabs) mabs = d;
    float den = fabsf(b[i]);
    if (den < fl) den = fl;
    if (d / den > mrel) mrel = d / den;
  }
  *out_abs = mabs;
  *out_rel = mrel;
  if (out_norm) *out_norm = maxb > 0 ? mabs / maxb : mabs;
}

/* 按 chunks[] 给的分段长度跑（段用完后余下的 token 一次跑完），再与参考逐元素比 */
static int check_pattern_(sgl_mla *m, const float *x, int n, int hidden, const int *chunks,
                          int nch, const float *ref_out, float *out, sgl_kv_cache *cache,
                          const char *label) {
  int pos[256];
  sgl_mla_cache_reset(cache);
  int t = 0, seg = 0;
  while (t < n) {
    int len = seg < nch ? chunks[seg] : n - t;
    if (len > n - t) len = n - t;
    if (len <= 0) { seg++; continue; }
    if (len > (int)(sizeof(pos) / sizeof(pos[0]))) return -1;
    for (int i = 0; i < len; i++) pos[i] = t + i;
    if (sgl_mla_forward(m, x + (size_t)t * hidden, len, pos, out + (size_t)t * hidden, cache) !=
        0) {
      CHECK(0, "%s: sgl_mla_forward 失败（t=%d len=%d）", label, t, len);
      return -1;
    }
    t += len;
    seg++;
  }
  float ea, er, en;
  err_stats_(out, ref_out, n * hidden, &ea, &er, &en);
  printf("  %-22s max|d|=%.3e  max rel=%.3e  norm rel=%.3e\n", label, (double)ea, (double)er,
         (double)en);
  CHECK(er < 1e-4f, "%s: 相对误差 %.3e 超 1e-4", label, (double)er);
  return er < 1e-4f ? 0 : -1;
}

static int run_one_(const char *dir) {
  printf("== %s\n", dir);
  char p[1200];
  snprintf(p, sizeof(p), "%s/mla_ref.json", dir);
  char *txt = read_text_(p);
  CHECK(txt != NULL, "缺少参考 %s（先跑 tests/gen_random_mla.py）", p);
  if (!txt) return 1;
  sgl_json *ref = sgl_json_parse(txt, strlen(txt));
  sgl_xfree(txt);
  CHECK(ref != NULL, "mla_ref.json 解析失败");
  if (!ref) return 1;

  const int n_pos = sgl_json_get_int(ref, "n_pos", 0);
  const int hidden = sgl_json_get_int(ref, "hidden", 0);
  const int stride = sgl_json_get_int(ref, "stride", 0);
  const int q_lora = sgl_json_get_int(ref, "q_lora_rank", 0);
  int before = FAILURES;

  sgl_mla *m = sgl_mla_load(dir);
  CHECK(m != NULL, "sgl_mla_load(%s)", dir);
  if (!m) { sgl_json_free(ref); return 1; }

  CHECK(sgl_mla_hidden(m) == hidden, "hidden %d != ref %d", sgl_mla_hidden(m), hidden);
  CHECK(sgl_mla_cache_stride(m) == stride, "stride %d != ref %d", sgl_mla_cache_stride(m),
        stride);
  printf("  hidden=%d heads=%d q_lora=%d kv_lora=%d rope=%d stride=%d n_pos=%d\n", hidden,
         sgl_mla_heads(m), q_lora, sgl_mla_kv_lora_rank(m), sgl_mla_qk_rope_dim(m),
         sgl_mla_cache_stride(m), n_pos);

  float *x = (float *)sgl_xmalloc(sizeof(float) * (size_t)n_pos * hidden);
  float *ref_out = (float *)sgl_xmalloc(sizeof(float) * (size_t)n_pos * hidden);
  float *ref_cache = (float *)sgl_xmalloc(sizeof(float) * (size_t)n_pos * stride);
  float *out = (float *)sgl_xmalloc(sizeof(float) * (size_t)n_pos * hidden);
  int ok = get_mat_(ref, "x", x, n_pos, hidden) == 0 &&
           get_mat_(ref, "out", ref_out, n_pos, hidden) == 0 &&
           get_mat_(ref, "cache", ref_cache, n_pos, stride) == 0;
  CHECK(ok, "参考数组读取失败（n_pos=%d hidden=%d stride=%d）", n_pos, hidden, stride);
  sgl_json_free(ref);
  if (!ok) {
    sgl_xfree(x); sgl_xfree(ref_out); sgl_xfree(ref_cache); sgl_xfree(out);
    sgl_mla_free(m);
    return 1;
  }

  sgl_kv_cache *cache = sgl_mla_cache_new(m, n_pos + 8);
  CHECK(cache != NULL, "cache_new");
  if (!cache) { sgl_xfree(x); sgl_xfree(ref_out); sgl_xfree(ref_cache); sgl_xfree(out); sgl_mla_free(m); return 1; }

  int pos[256];
  for (int i = 0; i < n_pos; i++) pos[i] = i;

  /* 1) 全序列 prefill */
  int rc = sgl_mla_forward(m, x, n_pos, pos, out, cache);
  CHECK(rc == 0, "prefill forward 返回 %d", rc);
  float ea, er, en;
  err_stats_(out, ref_out, n_pos * hidden, &ea, &er, &en);
  printf("[1] prefill n=%d：out  max|d|=%.3e  max rel=%.3e  norm rel=%.3e（阈值 1e-4）\n", n_pos,
         (double)ea, (double)er, (double)en);
  CHECK(er < 1e-4f, "prefill out 相对误差 %.3e 超 1e-4", (double)er);

  /* 2) 压缩 KV cache 内容（latent + k_pe）*/
  err_stats_(cache->data, ref_cache, n_pos * stride, &ea, &er, &en);
  printf("[2] kv cache[%d][%d]：max|d|=%.3e  max rel=%.3e  norm rel=%.3e（阈值 1e-4）\n", n_pos,
         stride, (double)ea, (double)er, (double)en);
  CHECK(er < 1e-4f, "kv cache 相对误差 %.3e 超 1e-4", (double)er);
  CHECK(cache->n_used == n_pos, "n_used=%d != %d", cache->n_used, n_pos);

  /* 3) 增量 decode：不同分块必须与全序列 prefill 逐元素一致 */
  printf("[3] 分块（prefill + 逐步 decode）\n");
  int pat_all[] = {n_pos};
  check_pattern_(m, x, n_pos, hidden, pat_all, 1, ref_out, out, cache, "1 段全序列");
  {
    int *pat = (int *)sgl_xmalloc(sizeof(int) * (size_t)n_pos);
    for (int i = 0; i < n_pos; i++) pat[i] = 1;
    check_pattern_(m, x, n_pos, hidden, pat, n_pos, ref_out, out, cache, "n_pos 段各 1 token");
    if (n_pos > 4) {
      pat[0] = 3; /* 3 个 token 一段，其余逐个 decode */
      check_pattern_(m, x, n_pos, hidden, pat, n_pos, ref_out, out, cache, "3 + 1+1+...");
    }
    if (n_pos >= 8) {
      pat[0] = 1; pat[1] = 5; pat[2] = 2; /* 1 + 5 + 2 + 余下逐个 */
      check_pattern_(m, x, n_pos, hidden, pat, n_pos, ref_out, out, cache, "1+5+2+1+...");
      pat[0] = n_pos / 2;                /* 中段切开 prefill/decode 边界 */
      check_pattern_(m, x, n_pos, hidden, pat, 1, ref_out, out, cache, "半段 prefill + decode");
    }
    sgl_xfree(pat);
  }

  /* 4) pos 越界必须报错而不是越写 */
  {
    int bad = cache->cap + 5;
    int rc2 = sgl_mla_forward(m, x, 1, &bad, out, cache);
    printf("[4] pos 越界（%d >= cap %d）返回 %d（期望 -1）\n", bad, cache->cap, rc2);
    CHECK(rc2 == -1, "pos 越界未返回 -1（返回 %d）", rc2);
  }

  sgl_xfree(x);
  sgl_xfree(ref_out);
  sgl_xfree(ref_cache);
  sgl_xfree(out);
  sgl_mla_cache_free(cache);
  sgl_mla_free(m);
  (void)q_lora;
  return FAILURES == before ? 0 : 1;
}

static int file_exists_(const char *p) {
  FILE *f = NULL;
  if (fopen_s(&f, p, "rb") != 0 || !f) return 0;
  fclose(f);
  return 1;
}

/* tests/gen_random_mla.py（由 __FILE__ 定位；CMake 传的是绝对路径）*/
static const char *gen_py_(void) {
  static char buf[1024];
  static int done = 0;
  if (!done) {
    snprintf(buf, sizeof(buf), "%s", __FILE__);
    char *p = strrchr(buf, '/');
    char *q = strrchr(buf, '\\');
    if (q > p) p = q;
    if (p) *p = 0;
    strncat(buf, "/gen_random_mla.py", sizeof(buf) - strlen(buf) - 1);
    done = 1;
  }
  return buf;
}

/* 没有现成模型就用 python 生成（与 test_bert_embed 同一套约定）*/
static int ensure_model_(const char *out_dir, const char *extra_args) {
  char cfg[1200];
  snprintf(cfg, sizeof(cfg), "%s/config.json", out_dir);
  if (file_exists_(cfg)) return 1;
  const char *py = getenv("PYTHON");
  const char *cands[4];
  int nc = 0;
  if (py && *py) cands[nc++] = py;
  cands[nc++] = "python";
  cands[nc++] = "python3";
  cands[nc++] = "py -3";
  for (int i = 0; i < nc; i++) {
    char cmd[2048];
#ifdef _WIN32
    snprintf(cmd, sizeof(cmd), "%s \"%s\" --out \"%s\" %s > nul 2>&1", cands[i], gen_py_(),
             out_dir, extra_args ? extra_args : "");
#else
    snprintf(cmd, sizeof(cmd), "%s \"%s\" --out \"%s\" %s > /dev/null 2>&1", cands[i],
             gen_py_(), out_dir, extra_args ? extra_args : "");
#endif
    if (system(cmd) == 0 && file_exists_(cfg)) return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  char dir[1200], dir2[1200];
  const char *a1 = argc > 1 ? argv[1] : getenv("SGL_TEST_MLA");
  const char *tmp = getenv("TEMP");
  if (!tmp || !*tmp) tmp = getenv("TMP");
  if (!tmp || !*tmp) tmp = ".";
  if (a1 && *a1) snprintf(dir, sizeof(dir), "%s", a1);
  else snprintf(dir, sizeof(dir), "%s/sgl_mla_rand", tmp);
  const char *a2 = getenv("SGL_TEST_MLA_QLORA0");
  if (a2 && *a2) snprintf(dir2, sizeof(dir2), "%s", a2);
  else snprintf(dir2, sizeof(dir2), "%s/sgl_mla_rand_q0", tmp);

  if (!ensure_model_(dir, NULL)) {
    printf("test_mla: SKIP（无法生成模型 %s：需 python + numpy）\n", dir);
    return 0;
  }
  run_one_(dir);

  /* 第二个模型：q_lora_rank = null（直接 q_proj）*/
  if (ensure_model_(dir2, "--q-lora 0")) run_one_(dir2);

  if (FAILURES) {
    printf("test_mla: %d FAILURES\n", FAILURES);
    return 1;
  }
  printf("test_mla: all OK\n");
  return 0;
}