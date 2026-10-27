/* test_jinja — P0-2 chat template：把 C 解释器的渲染结果与 Python jinja2 逐字节比对。
 * 参考文件 tests/ref_chat_*.txt 由 jinja2 3.1.6 生成（相同的 messages / kwargs）。
 * 模型目录不存在时跳过对应用例（CI 只有仓库内的随机小模型时也能过）。*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/jinja.h"
#include "sgl/json.h"
#include "sgl/port.h"

static int fails;

static const char *MSGS_JSON =
    "[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},"
    "{\"role\":\"user\",\"content\":\"What is the capital of France?\"},"
    "{\"role\":\"assistant\",\"content\":\"Paris.\"},"
    "{\"role\":\"user\",\"content\":\"And of Japan?\"}]";

static const char *KW_TOOLS_JSON =
    "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"city\":{\"type\":\"string\"}}}}}]}";

static char *read_all(const char *path, size_t *len) {
  FILE *f = NULL;
  if (fopen_s(&f, path, "rb") != 0 || !f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n < 0) {
    fclose(f);
    return NULL;
  }
  char *buf = (char *)malloc((size_t)n + 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  size_t got = fread(buf, 1, (size_t)n, f);
  fclose(f);
  buf[got] = 0;
  if (len) *len = got;
  return buf;
}

/* 与参考文件比对；不一致时打印首个差异位置与上下文 */
static void cmp_ref(const char *name, const char *got, size_t got_len, const char *ref_path) {
  size_t ref_len = 0;
  char *ref = read_all(ref_path, &ref_len);
  if (!ref) {
    printf("SKIP: %s (reference %s not found)\n", name, ref_path);
    return;
  }
  if (got_len == ref_len && memcmp(got, ref, ref_len) == 0) {
    printf("ok  : %s (%.0f bytes identical to jinja2)\n", name, (double)ref_len);
  } else {
    fails++;
    size_t i = 0;
    size_t n = got_len < ref_len ? got_len : ref_len;
    while (i < n && got[i] == ref[i]) i++;
    printf("FAIL: %s (got %llu bytes, want %llu) first diff at %llu\n", name,
           (unsigned long long)got_len, (unsigned long long)ref_len, (unsigned long long)i);
    size_t a = i > 40 ? i - 40 : 0;
    printf("  got: ...%.*s\n", (int)(got_len - a > 80 ? 80 : got_len - a), got + a);
    printf("  ref: ...%.*s\n", (int)(ref_len - a > 80 ? 80 : ref_len - a), ref + a);
  }
  free(ref);
}

static void run_dir_case(const char *label, const char *model_dir, const char *ref, int with_tools) {
  sgl_jinja *j = sgl_jinja_load_dir(model_dir);
  if (!j) {
    printf("SKIP: %s (no chat_template in %s)\n", label, model_dir);
    return;
  }
  sgl_json *msgs = sgl_json_parse(MSGS_JSON, strlen(MSGS_JSON));
  sgl_json *kw = with_tools ? sgl_json_parse(KW_TOOLS_JSON, strlen(KW_TOOLS_JSON)) : NULL;
  char *out = NULL;
  size_t out_len = 0;
  if (!msgs || sgl_jinja_render(j, msgs, 1, kw, &out, &out_len) != 0) {
    printf("FAIL: %s render error\n", label);
    fails++;
  } else {
    cmp_ref(label, out, out_len, ref);
    sgl_xfree(out);
  }
  if (msgs) sgl_json_free(msgs);
  if (kw) sgl_json_free(kw);
  sgl_jinja_free(j);
}

/* 内联小模板：set / namespace / for-else / 过滤器 / 切片 / is defined */
static void inline_cases(void) {
  struct {
    const char *tpl;
    const char *msgs;
    const char *want;
  } cases[] = {
      {"{% for m in messages %}[{{ loop.index0 }}:{{ m.role }}={{ m.content }}]{% endfor %}",
       "[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"yo\"}]",
       "[0:user=hi][1:assistant=yo]"},
      {"{% set ns = namespace(n=0) %}{% for m in messages %}{% set ns.n = ns.n + 1 %}{% endfor %}{{ ns.n }}",
       "[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"user\",\"content\":\"b\"}]", "2"},
      {"{% for m in messages %}x{% else %}empty{% endfor %}", "[]", "empty"},
      {"{{ messages|length }}|{{ messages[0].role }}|{{ messages[::-1][0].content }}",
       "[{\"role\":\"user\",\"content\":\"first\"},{\"role\":\"user\",\"content\":\"last\"}]", "2|user|last"},
      {"{% if messages[0].tool is defined %}has{% else %}none{% endif %}",
       "[{\"role\":\"user\",\"content\":\"x\"}]", "none"},
      {"{{ 'a,b,c'.split(',')|join('-') }}|{{ ' Hello '.strip() }}|{{ 'x'|upper }}{{ 3 * 4 }}",
       "[{\"role\":\"user\",\"content\":\"x\"}]", "a-b-c|Hello|X12"},
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    sgl_jinja *j = sgl_jinja_from_text(cases[i].tpl, strlen(cases[i].tpl));
    sgl_json *msgs = sgl_json_parse(cases[i].msgs, strlen(cases[i].msgs));
    char *out = NULL;
    size_t out_len = 0;
    int bad = 0;
    if (!j || !msgs || sgl_jinja_render(j, msgs, 1, NULL, &out, &out_len) != 0) {
      bad = 1;
    } else if (out_len != strlen(cases[i].want) || strcmp(out, cases[i].want) != 0) {
      bad = 1;
    }
    if (bad) {
      fails++;
      printf("FAIL: inline case %d\n  tpl : %s\n  got : %s\n  want: %s\n", (int)i, cases[i].tpl,
             out ? out : "(render error)", cases[i].want);
    } else {
      printf("ok  : inline case %d\n", (int)i);
    }
    if (out) sgl_xfree(out);
    if (j) sgl_jinja_free(j);
    if (msgs) sgl_json_free(msgs);
  }
}

int main(void) {
  const char *qwen = getenv("SGL_TEST_QWEN"); /* <models>/qwen05 */
  const char *tiny = getenv("SGL_TEST_TINYLLAMA");

  inline_cases();

  if (qwen && qwen[0]) {
    run_dir_case("qwen05 template", qwen, "../tests/ref_chat_qwen05.txt", 0);
    run_dir_case("qwen05 template + tools", qwen, "../tests/ref_chat_qwen05_tools.txt", 1);
    /* 同目录树里的 qwen3-06：namespace / 切片 / is string / split 链 */
    char q3[1200];
    const char *slash = strrchr(qwen, '/');
    const char *bslash = strrchr(qwen, '\\');
    if (!slash || (bslash && bslash > slash)) slash = bslash;
    if (slash) {
      int dirlen = (int)(slash - qwen);
      snprintf(q3, sizeof(q3), "%.*s/qwen3-06", dirlen, qwen);
      run_dir_case("qwen3 template", q3, "../tests/ref_chat_qwen3.txt", 0);
    }
  } else {
    printf("SKIP: qwen cases (SGL_TEST_QWEN not set)\n");
  }
  if (tiny && tiny[0]) run_dir_case("tinyllama template", tiny, "../tests/ref_chat_tinyllama.txt", 0);
  else printf("SKIP: tinyllama case (SGL_TEST_TINYLLAMA not set)\n");

  if (fails) {
    printf("test_jinja: %d failure(s)\n", fails);
    return 1;
  }
  printf("test_jinja: all OK\n");
  return 0;
}