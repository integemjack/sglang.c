/* test_bert_embed.c — BERT 类 embedding / rerank 数值对照（随机权重小模型）
 *
 * 参考数据由 tests/gen_random_bert.py 生成（numpy 独立前向 + HF tokenizers 的词表编码，
 * 与 C 端零共享代码）：<model_dir>/ref_bert_embed.json。
 * 本测试自己调 python 生成到 $TEMP（不在仓库里留大文件），再逐项断言：
 *   1. sgl_bert_hidden == config.hidden_size
 *   2. WordPiece 编码结果与参考 token id 完全一致（含 [CLS]/[SEP]、max_len 截断）
 *   3. encode(mean pooling + L2 normalize) 与参考向量 maxerr < 1e-4，且 ‖v‖₂ ≈ 1
 *   4. encode(cls pooling) 与参考向量 maxerr < 1e-4
 *   5. rerank 对 4 组 (query, doc) 的分数 maxerr < 1e-4，且排序与参考一致
 *   6. roberta 变体（position_ids = i+padding_idx+1、权重前缀 "roberta."）同样通过 3/5
 * 全部通过打印 "test_bert_embed: all OK"。
 *
 * 手动跑：
 *   python tests/gen_random_bert.py %TEMP%\sgl_bert_gen
 *   build-bert/bin/test_bert_embed.exe %TEMP%\sgl_bert_gen
 * 环境变量：SGL_TEST_BERT（已生成目录）/ SGL_TEST_BERT_REF（参考 JSON 路径）/ PYTHON
 */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/bert.h"
#include "sgl/json.h"

static int FAILURES = 0;
static int CHECKS = 0;
#define CHECK(cond, ...)                                     \
  do {                                                       \
    CHECKS++;                                                \
    if (!(cond)) {                                           \
      FAILURES++;                                            \
      printf("FAIL %s:%d: ", __FILE__, __LINE__);            \
      printf(__VA_ARGS__);                                   \
      printf("\n");                                          \
    }                                                        \
  } while (0)

static const char *GEN_PY(void) {
  /* __FILE__ 是 CMake 传进来的绝对路径 …/tests/test_bert_embed.c */
  static char buf[1024];
  static int done = 0;
  if (!done) {
    snprintf(buf, sizeof(buf), "%s", __FILE__);
    char *p = strrchr(buf, '/');
    char *q = strrchr(buf, '\\');
    if (q > p) p = q;
    if (p) *p = 0; /* …/tests */
    strncat(buf, "/gen_random_bert.py", sizeof(buf) - strlen(buf) - 1);
    done = 1;
  }
  return buf;
}

static int file_exists(const char *p) {
  FILE *f = fopen(p, "rb");
  if (!f) return 0;
  fclose(f);
  return 1;
}

/* 用 system() 调 python 生成模型；成功返回 1 */
static int run_generator(const char *script, const char *out_dir) {
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
    snprintf(cmd, sizeof(cmd), "%s \"%s\" \"%s\" > nul 2>&1", cands[i], script, out_dir);
#else
    snprintf(cmd, sizeof(cmd), "%s \"%s\" \"%s\" > /dev/null 2>&1", cands[i], script,
             out_dir);
#endif
    int rc = system(cmd);
    if (rc == 0) {
      printf("generated: python=%s -> %s\n", cands[i], out_dir);
      return 1;
    }
  }
  return 0;
}

static sgl_json *load_json(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *text = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(text, 1, (size_t)sz, f) != (size_t)sz) {
    fclose(f);
    sgl_xfree(text);
    return NULL;
  }
  text[sz] = 0;
  fclose(f);
  sgl_json *j = sgl_json_parse(text, (size_t)sz);
  sgl_xfree(text);
  return j;
}

/* 逐元素比对，返回 maxerr（out_n 写元素数）*/
static double cmp_vec(const float *got, const sgl_json *ref, int *out_n) {
  double maxerr = 0;
  int n = ref ? ref->count : 0;
  if (out_n) *out_n = n;
  for (int i = 0; i < n; i++) {
    double d = fabs((double)got[i] - ref->items[i]->num);
    if (d > maxerr) maxerr = d;
  }
  return maxerr;
}

static double l2norm(const float *v, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
  return sqrt(s);
}

#define MAXERR_TOL 1e-4
#define NORM_TOL 1e-5

/* 跑一个模型目录的全部断言；tag 用于日志 */
static int check_model(const char *dir, const char *tag) {
  char path[1200];
  snprintf(path, sizeof(path), "%s/ref_bert_embed.json", dir);
  sgl_json *ref = load_json(path);
  CHECK(ref != NULL, "[%s] 参考 JSON 读不到：%s", tag, path);
  if (!ref) return 0;

  const sgl_json *jmodel = sgl_json_get(ref, "model");
  int H = (int)sgl_json_get_int(jmodel, "hidden", 0);

  sgl_bert *m = sgl_bert_load(dir);
  CHECK(m != NULL, "[%s] sgl_bert_load(%s)", tag, dir);
  if (!m) {
    sgl_json_free(ref);
    return 0;
  }
  CHECK(sgl_bert_hidden(m) == H, "[%s] hidden=%d 期望 %d", tag, sgl_bert_hidden(m), H);

  float *emb = (float *)sgl_xmalloc(sizeof(float) * (size_t)H);
  float *emb2 = (float *)sgl_xmalloc(sizeof(float) * (size_t)H);
  int *ids = (int *)sgl_xmalloc(sizeof(int) * 128);

  /* ---- 1) tokenizer 对照 ---- */
  const sgl_json *jtexts = sgl_json_get(ref, "texts");
  int nt = jtexts ? jtexts->count : 0;
  double worst_ids = 0, worst_mean = 0, worst_cls = 0, worst_norm = 0;
  for (int i = 0; i < nt; i++) {
    const sgl_json *c = jtexts->items[i];
    const char *text = sgl_json_get_str(c, "text", "");
    const sgl_json *jids = sgl_json_get(c, "ids");
    int n = sgl_wordpiece_encode(dir, text, ids, 128);
    int ok = n == jids->count;
    int firstbad = -1;
    for (int k = 0; k < n && k < jids->count; k++)
      if (ids[k] != (int)jids->items[k]->num) {
        ok = 0;
        if (firstbad < 0) firstbad = k;
      }
    CHECK(ok, "[%s] WordPiece 编码不一致：text=%.24s… got_n=%d want_n=%d first_bad=%d",
          tag, text, n, jids->count, firstbad);
    if (!ok) continue;

    /* ---- 2) mean pooling + normalize ---- */
    CHECK(sgl_bert_encode(m, text, SGL_BERT_POOL_MEAN, emb, H) == 0,
          "[%s] encode(mean) 失败：%.24s", tag, text);
    double err = cmp_vec(emb, sgl_json_get(c, "embed_mean"), NULL);
    if (err > worst_mean) worst_mean = err;
    CHECK(err < MAXERR_TOL, "[%s] mean-embedding maxerr=%.3e >= %.0e（%.24s…）", tag, err,
          MAXERR_TOL, text);
    double nrm = l2norm(emb, H);
    if (fabs(nrm - 1.0) > worst_norm) worst_norm = fabs(nrm - 1.0);
    CHECK(fabs(nrm - 1.0) < NORM_TOL, "[%s] L2 范数=%.9f 远离 1（%.24s…）", tag, nrm, text);

    /* ---- 3) cls pooling ---- */
    CHECK(sgl_bert_encode(m, text, SGL_BERT_POOL_CLS, emb2, H) == 0,
          "[%s] encode(cls) 失败：%.24s", tag, text);
    err = cmp_vec(emb2, sgl_json_get(c, "embed_cls"), NULL);
    if (err > worst_cls) worst_cls = err;
    CHECK(err < MAXERR_TOL, "[%s] cls-embedding maxerr=%.3e >= %.0e（%.24s…）", tag, err,
          MAXERR_TOL, text);
  }
  CHECK(nt >= 1, "[%s] 参考用例为空", tag);
  (void)worst_ids;
  printf("[%s] tokenizer/mean/cls 对照完成：texts=%d mean_maxerr=%.3e cls_maxerr=%.3e "
         "max|‖v‖-1|=%.3e\n", tag, nt, worst_mean, worst_cls, worst_norm);

  /* ---- 4) rerank 分数 + 排序 ---- */
  const sgl_json *jr = sgl_json_get(ref, "rerank");
  const sgl_json *jrank = sgl_json_get(ref, "rank");
  int nr = (jr && jr->count <= 8) ? jr->count : 8;
  float got[8];
  double worst_score = 0;
  for (int i = 0; i < nr; i++) {
    const sgl_json *c = jr->items[i];
    const char *q = sgl_json_get_str(c, "query", "");
    const char *d = sgl_json_get_str(c, "doc", "");
    double want = sgl_json_get_num(c, "score", 0.0);
    got[i] = sgl_bert_rerank(m, q, d);
    double err = fabs((double)got[i] - want);
    if (err > worst_score) worst_score = err;
    CHECK(err < MAXERR_TOL, "[%s] rerank[%d] score=%.7f 期望 %.7f err=%.3e", tag, i,
          (double)got[i], want, err);
  }
  /* 排序：参考 rank[k] 是第 k 名的下标，要求 C 的分数同序（允许并列时按原序）*/
  if (jrank && jrank->count == nr) {
    for (int k = 0; k + 1 < nr; k++) {
      int a = (int)jrank->items[k]->num;
      int b = (int)jrank->items[k + 1]->num;
      CHECK(a >= 0 && a < nr && b >= 0 && b < nr, "[%s] rank 越界", tag);
      if (a < 0 || a >= nr || b < 0 || b >= nr) continue;
      CHECK(got[a] >= got[b] - 1e-6f,
            "[%s] rerank 排序与参考不符：rank=%d(%.6f) 应 >= rank=%d(%.6f)", tag, a,
            (double)got[a], b, (double)got[b]);
    }
    printf("[%s] rerank: n=%d maxerr=%.3e scores=[", tag, nr, worst_score);
    for (int i = 0; i < nr; i++) printf("%.6f%s", (double)got[i], i + 1 < nr ? ", " : "");
    printf("] order_ok=1\n");
  }

  /* ---- 5) 边界：out_cap 不够 / 空文本 ---- */
  if (H > 1)
    CHECK(sgl_bert_encode(m, "hello", SGL_BERT_POOL_MEAN, emb, H - 1) != 0,
          "[%s] out_cap < hidden 应失败", tag);
  {
    sgl_wp *wp = sgl_wp_load_dir(dir);
    int n0 = sgl_wordpiece_encode(dir, "", ids, 128);
    CHECK(wp != NULL, "[%s] sgl_wp_load_dir", tag);
    if (wp)
      CHECK(n0 == 2 && ids[0] == sgl_wp_cls_id(wp) && ids[1] == sgl_wp_sep_id(wp),
            "[%s] 空文本应只出 [CLS][SEP]（got n=%d ids=%d,%d）", tag, n0, ids[0],
            n0 > 1 ? ids[1] : -1);
    sgl_wp_free(wp);
  }

  sgl_xfree(ids);
  sgl_xfree(emb2);
  sgl_xfree(emb);
  sgl_bert_free(m);
  sgl_json_free(ref);
  return 1;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);

  char dir[1024], dir_rb[1100];
  if (argc > 1) {
    snprintf(dir, sizeof(dir), "%s", argv[1]);
  } else if (getenv("SGL_TEST_BERT") && *getenv("SGL_TEST_BERT")) {
    snprintf(dir, sizeof(dir), "%s", getenv("SGL_TEST_BERT"));
  } else {
    const char *tmp = getenv("TEMP");
    if (!tmp || !*tmp) tmp = getenv("TMP");
    if (!tmp || !*tmp) tmp = ".";
    snprintf(dir, sizeof(dir), "%s/sgl_bert_gen", tmp);
  }
  snprintf(dir_rb, sizeof(dir_rb), "%s_roberta", dir);
  char ptmp[1200];

  /* 没有现成的就自己生成（gen_random_bert.py 同时写 bert / roberta 两个目录）*/
  snprintf(ptmp, sizeof(ptmp), "%s/ref_bert_embed.json", dir);
  if (!file_exists(ptmp)) {
    const char *script = getenv("SGL_TEST_BERT_GEN_PY");
    if (!script || !*script) script = GEN_PY();
    if (!file_exists(script)) {
      printf("test_bert_embed: SKIP（找不到生成脚本 %s）\n", script);
      return 0;
    }
    if (!run_generator(script, dir)) {
      printf("test_bert_embed: SKIP（python/numpy 不可用，无法生成随机模型；"
             "可先手工跑 python %s <dir>）\n", script);
      return 0;
    }
  }

  if (!check_model(dir, "bert")) {
    printf("test_bert_embed: FAILED (bert)\n");
    return 1;
  }
  snprintf(ptmp, sizeof(ptmp), "%s/ref_bert_embed.json", dir_rb);
  if (file_exists(ptmp)) {
    check_model(dir_rb, "roberta");
  } else {
    printf("[roberta] 跳过（没有 %s）\n", ptmp);
  }

  if (FAILURES) {
    printf("test_bert_embed: %d/%d 项失败\n", FAILURES, CHECKS);
    return 1;
  }
  printf("test_bert_embed: all OK（%d 项断言）\n", CHECKS);
  return 0;
}