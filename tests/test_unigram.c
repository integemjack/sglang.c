/* test_unigram.c — Unigram(SentencePiece) tokenizer 对照测试
 *
 * 夹具：models/unigram-t5/tokenizer.json（T5-small 的真实词表，32100 个 piece）
 * 参考：tests/_ref_unigram.json —— 由 tests/gen_unigram_ref.py 用**独立实现**的
 *       Viterbi 算出（只共用词表文件这一份数据，切分算法不共用代码）。
 *
 * 覆盖：词表读的是 [[piece, score], ...] 数组（不是 BPE 的 object）、
 *       最大得分切分、▁ 前缀/空格归一化、decode 回文本（含 ▁→空格）。
 * 夹具或参考缺失时 SKIP（fresh clone 上没有下载的权重也能绿）。
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/json.h"

static int fails;

#define CHECK(cond, ...)                          \
  do {                                            \
    if (!(cond)) {                                \
      printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      printf(__VA_ARGS__);                        \
      printf("\n");                               \
      fails++;                                    \
    }                                             \
  } while (0)

static int file_exists(const char *p) {
  FILE *f = fopen(p, "rb");
  if (!f) return 0;
  fclose(f);
  return 1;
}

static const char *env_or(const char *k, const char *d) {
  const char *v = getenv(k);
  return (v && *v) ? v : d;
}

static char *read_all(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *b = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(b, 1, (size_t)sz, f) != (size_t)sz) {
    fclose(f);
    sgl_xfree(b);
    return NULL;
  }
  fclose(f);
  b[sz] = 0;
  *n = (size_t)sz;
  return b;
}

int main(void) {
  const char *dir = env_or("SGL_TEST_UNIGRAM", "../models/unigram-t5");
  const char *ref = env_or("SGL_TEST_UNIGRAM_REF", "tests/_ref_unigram.json");
  char tpath[1024];
  snprintf(tpath, sizeof(tpath), "%s/tokenizer.json", dir);
  if (!file_exists(tpath) || !file_exists(ref)) {
    printf("SKIP test_unigram: need %s and %s\n", tpath, ref);
    return 0;
  }

  sgl_tokenizer *t = sgl_tokenizer_load_dir(dir);
  if (!t) t = sgl_tokenizer_load(tpath);
  CHECK(t != NULL, "tokenizer load failed: %s", dir);
  if (!t) return 1;

  size_t rn = 0;
  char *txt = read_all(ref, &rn);
  CHECK(txt != NULL, "read %s failed", ref);
  if (!txt) return 1;
  sgl_json *j = sgl_json_parse(txt, rn);
  sgl_xfree(txt);
  CHECK(j != NULL, "parse %s failed", ref);
  if (!j) return 1;

  int want_vocab = sgl_json_get_int(j, "vocab_size", 0);
  CHECK(sgl_tokenizer_vocab_size(t) == want_vocab, "vocab %d != ref %d",
        sgl_tokenizer_vocab_size(t), want_vocab);
  CHECK(sgl_tokenizer_style(t) == 2, "style %d != 2 (unigram)", sgl_tokenizer_style(t));

  const sgl_json *cases = sgl_json_get(j, "cases");
  int ncase = 0;
  if (cases && cases->type == SGL_JSON_ARRAY) ncase = cases->count;
  CHECK(ncase >= 5, "ref has only %d cases", ncase);

  for (int c = 0; c < ncase; c++) {
    const sgl_json *e = cases->items[c];
    const char *text = sgl_json_get_str(e, "text", NULL);
    const sgl_json *ids = sgl_json_get(e, "ids");
    const char *want_dec = sgl_json_get_str(e, "decoded", NULL);
    if (!text || !ids || ids->type != SGL_JSON_ARRAY) continue;

    int n = 0;
    int *got = sgl_tokenizer_encode(t, text, strlen(text), &n);
    int ok = (n == ids->count);
    if (ok)
      for (int i = 0; i < n; i++) {
        if (ids->items[i]->type != SGL_JSON_NUMBER) { ok = 0; break; }
        if (got[i] != (int)ids->items[i]->num) {
          printf("  case %d \"%.40s\": token %d: got %d want %d\n", c, text, i, got[i],
                 (int)ids->items[i]->num);
          ok = 0;
          break;
        }
      }
    CHECK(ok, "encode mismatch: \"%.50s\" (got %d tokens, want %d)", text, n, ids->count);

    char *dec = sgl_tokenizer_decode(t, got, n);
    if (want_dec && dec) CHECK(strcmp(dec, want_dec) == 0, "decode mismatch: got \"%.60s\" want \"%.60s\"",
                               dec, want_dec);
    sgl_xfree(dec);
    sgl_xfree(got);
  }

  sgl_json_free(j);
  sgl_tokenizer_free(t);
  if (fails) {
    printf("test_unigram: %d failures\n", fails);
    return 1;
  }
  printf("test_unigram: all %d cases match the independent Viterbi reference\n", ncase);
  return 0;
}