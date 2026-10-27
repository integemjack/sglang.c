/* test_parser — P2（上游 srt/parser/）：reasoning 分离 + tool call 提取 */
#include <stdio.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/parser.h"

/* DeepSeek-R1 的 think 标记：U+003C + "think" + U+FF1E（全角 ＞）*/
#define OPEN_THINK "\x3Cthink\xEF\xBC\x9E"
#define CLOSE_THINK "\x3C/think\xEF\xBC\x9E"

static int fails;
static void check(int cond, const char *msg) {
  if (!cond) {
    fails++;
    printf("FAIL: %s\n", msg);
  }
}

int main(void) {
  /* ---- reasoning ---- */
  {
    char *r = NULL;
    const char *c = NULL;
    int hit = sgl_parse_reasoning(OPEN_THINK " think hard" CLOSE_THINK "The answer is 42.", &r, &c, NULL);
    check(hit == 1, "reasoning: marker found");
    check(r && !strcmp(r, " think hard"), "reasoning: prefix text");
    check(c && !strcmp(c, "The answer is 42."), "reasoning: content after marker");
    sgl_xfree(r);
  }
  {
    char *r = (char *)1;
    const char *c = NULL;
    int hit = sgl_parse_reasoning("<|end_of_thinking|>plain", &r, &c, NULL);
    check(hit == 1 && r && !*r, "reasoning: empty reasoning block");
    check(c && !strcmp(c, "plain"), "reasoning: qwen3 marker");
    sgl_xfree(r);
  }
  {
    char *r = (char *)1;
    const char *c = NULL;
    int hit = sgl_parse_reasoning("no marker here", &r, &c, NULL);
    check(hit == 0 && r == NULL, "reasoning: none -> NULL reasoning");
    check(c && !strcmp(c, "no marker here"), "reasoning: content unchanged");
  }

  /* ---- tool calls ---- */
  {
    int n = 0;
    const char *txt = "sure!\n<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
    char *j = sgl_parse_tool_calls(txt, &n);
    check(n == 1 && j != NULL, "tool: one call");
    check(j && strstr(j, "\"get_weather\"") && strstr(j, "\"Paris\""), "tool: content kept");
    check(j && j[0] == '[' && j[strlen(j) - 1] == ']', "tool: array wrapped");
    sgl_xfree(j);
  }
  {
    int n = 0;
    const char *txt =
        "<tool_call>{\"name\":\"a\"}</tool_call>\n<tool_call>{\"name\":\"b\"}</tool_call>";
    char *j = sgl_parse_tool_calls(txt, &n);
    check(n == 2, "tool: two calls");
    check(j && strstr(j, "\"a\"") && strstr(j, "\"b\""), "tool: both kept");
    sgl_xfree(j);
  }
  {
    int n = 0;
    const char *txt = "<tool_call>[{\"name\":\"a\"},{\"name\":\"b\",\"arguments\":{\"x\":[1,2]}}]</tool_call>";
    char *j = sgl_parse_tool_calls(txt, &n);
    check(n == 2, "tool: array form counted");
    check(j && strstr(j, "\"x\":[1,2]"), "tool: nested array kept intact");
    sgl_xfree(j);
  }
  {
    int n = 7;
    char *j = sgl_parse_tool_calls("plain text, no calls", &n);
    check(j == NULL && n == 0, "tool: none -> NULL");
  }

  if (fails) {
    printf("test_parser: %d failure(s)\n", fails);
    return 1;
  }
  printf("test_parser: all OK\n");
  return 0;
}