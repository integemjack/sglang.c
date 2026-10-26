/* test_bpe.c — BPE tokenizer 测试（需要 tokenizer.json；HF 官方 GPT-2 样例）*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"

static int fails = 0;
#define CHECK(cond, ...)                          \
  do {                                            \
    if (!(cond)) {                                \
      printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      printf(__VA_ARGS__);                        \
      printf("\n");                               \
      fails++;                                    \
    }                                             \
  } while (0)

static void check_encode(sgl_tokenizer *t, const char *text, const int *want, int want_n) {
  int n = 0;
  int *ids = sgl_tokenizer_encode(t, text, strlen(text), &n);
  int ok = (n == want_n);
  if (ok)
    for (int i = 0; i < n; i++)
      if (ids[i] != want[i]) { ok = 0; break; }
  if (!ok) {
    printf("  input: \"%s\"\n  want :", text);
    for (int i = 0; i < want_n; i++) printf(" %d", want[i]);
    printf("\n  got  :");
    for (int i = 0; i < n; i++) printf(" %d", ids[i]);
    printf("\n");
    fails++;
  }
  sgl_xfree(ids);
}

int main(void) {
  /* 找 tokenizer.json：环境变量 SGL_TEST_MODEL 目录，或 ../models/gpt2 */
  const char *dir = getenv("SGL_TEST_MODEL");
  char path[1024];
  if (dir && *dir)
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
  else
    snprintf(path, sizeof(path), "models/gpt2/tokenizer.json");
  FILE *f = fopen(path, "rb");
  if (!f) {
    printf("test_bpe: SKIP（没找到 %s；下载 GPT-2 tokenizer 后重跑）\n", path);
    return 0;
  }
  fclose(f);

  sgl_tokenizer *t = sgl_tokenizer_load(path);
  CHECK(t != NULL, "load %s", path);
  if (!t) return 1;
  CHECK(sgl_tokenizer_vocab_size(t) == 50257, "vocab=%d", sgl_tokenizer_vocab_size(t));
  CHECK(sgl_tokenizer_eos_id(t) == 50256, "eos=%d", sgl_tokenizer_eos_id(t));

  /* HF tokenizers 官方 GPT-2 文档样例 */
  check_encode(t, "hello world", (int[]){31373, 995}, 2);
  check_encode(t, "Hello, my dog is cute",
               (int[]){15496, 11, 616, 3290, 318, 13779}, 6);
  check_encode(t, "Hello world", (int[]){15496, 995}, 2);
  check_encode(t, "don't", (int[]){9099, 470}, 2); /* don + 't */
  check_encode(t, "  spaces", (int[]){220, 9029}, 2); /* 双空格：空格段 + " spaces"（tiktoken 验证）*/

  /* decode 回环 */
  int roundtrip_ids[] = {15496, 11, 616, 3290, 318, 13779};
  char *back = sgl_tokenizer_decode(t, roundtrip_ids, 6);
  CHECK(strcmp(back, "Hello, my dog is cute") == 0, "decode=%s", back);
  sgl_xfree(back);

  sgl_tokenizer_free(t);
  printf(fails ? "test_bpe: %d FAILURES\n" : "test_bpe: all OK\n", fails);
  return fails ? 1 : 0;
}
