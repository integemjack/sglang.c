/* test_json.c — JSON 解析器测试 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "sgl/json.h"

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

int main(void) {
  /* 嵌套 + 转义 + unicode */
  const char *s = "{\"name\":\"a\\\"b\\\\c\", \"n\": -1.5e2, \"ok\": true, "
                   "\"arr\": [1, 2, {\"x\": \"\\u4e2d\\u6587\"}], \"z\": null}";
  sgl_json *j = sgl_json_parse(s, strlen(s));
  CHECK(j != NULL, "parse");
  if (!j) return 1;

  CHECK(strcmp(sgl_json_get_str(j, "name", ""), "a\"b\\c") == 0, "name=%s", sgl_json_get_str(j, "name", "?"));
  CHECK(fabs(sgl_json_get_num(j, "n", 0) + 150.0) < 1e-9, "n=%g", sgl_json_get_num(j, "n", 0));
  CHECK(sgl_json_get_int(j, "n", 0) == -150, "n int");
  const sgl_json *ok = sgl_json_get(j, "ok");
  CHECK(ok && ok->type == SGL_JSON_BOOL && ok->boolean == 1, "ok");
  const sgl_json *z = sgl_json_get(j, "z");
  CHECK(z && z->type == SGL_JSON_NULL, "z null");
  const sgl_json *arr = sgl_json_get(j, "arr");
  CHECK(arr && arr->type == SGL_JSON_ARRAY && arr->count == 3, "arr count=%d", arr ? arr->count : -1);
  const sgl_json *x = sgl_json_get(arr->items[2], "x");
  CHECK(x && x->type == SGL_JSON_STRING, "x type");
  /* \u4e2d\u6587 → UTF-8 "中文" */
  CHECK(x && strcmp(x->str, "\xe4\xb8\xad\xe6\x96\x87") == 0, "x=%s", x ? x->str : "?");
  CHECK(sgl_json_get(j, "missing") == NULL, "missing");
  sgl_json_free(j);

  /* 非法输入 */
  CHECK(sgl_json_parse("{", 1) == NULL, "truncated");
  CHECK(sgl_json_parse("{\"a\":1}x", 8) == NULL, "trailing");
  CHECK(sgl_json_parse("[1,]", 4) == NULL, "bad array");

  printf(fails ? "test_json: %d FAILURES\n" : "test_json: all OK\n", fails);
  return fails ? 1 : 0;
}
