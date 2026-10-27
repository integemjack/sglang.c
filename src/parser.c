/* sgl parser 实现：reasoning 分离 + tool call 提取（纯字符串）*/
#include "sgl/parser.h"

#include <string.h>

#include "sgl/alloc.h"

static char *dup_range(const char *p, size_t n) {
  char *d = (char *)sgl_xmalloc(n + 1);
  if (n) memcpy(d, p, n);
  d[n] = 0;
  return d;
}

int sgl_parse_reasoning(const char *text, char **reasoning, const char **content,
                        const char **finish_marker) {
  /* 开/结束标记：开标记有三种写法，取最早出现的结束标记作为切点；
   * 正文里若含开标记，reasoning 从开标记之后开始（开标记本身不进 reasoning）。
   *   - DeepSeek-R1 的 U+003C + "think" + U+FF1E（全角 ＞，字节转义写，免得源码编码出意外）
   *   - Qwen3 的 <|end_of_thinking|>
   *   - 通用 ASCII 的 </thinking> */
  static const char *const OPEN = "\x3Cthink\xEF\xBC\x9E";
  static const char *const closes[] = {"\x3C/think\xEF\xBC\x9E", "<|end_of_thinking|>",
                                       "</thinking>"};
  if (reasoning) *reasoning = NULL;
  if (content) *content = text ? text : "";
  if (finish_marker) *finish_marker = NULL;
  if (!text) return 0;
  const char *close = NULL;
  const char *hit = NULL;
  for (int i = 0; i < 3; i++) {
    const char *p = strstr(text, closes[i]);
    if (p && (!close || p < close)) {
      close = p;
      hit = closes[i];
    }
  }
  const char *begin = text;
  const char *open = strstr(text, OPEN);
  if (open && (!close || open < close)) begin = open + strlen(OPEN);
  if (close) {
    if (reasoning) *reasoning = dup_range(begin, (size_t)(close - begin));
    if (content) *content = close + strlen(hit);
    if (finish_marker) *finish_marker = hit;
    return 1;
  }
  /* 只有开标记、还没结束：剩下的都算 reasoning，正文为空（流式中途会走到这）*/
  if (begin != text) {
    if (reasoning) *reasoning = dup_range(begin, strlen(begin));
    if (content) *content = "";
    return 1;
  }
  return 0;
}

static const char *skip_ws(const char *p) {
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  return p;
}

char *sgl_parse_tool_calls(const char *text, int *n_calls) {
  if (n_calls) *n_calls = 0;
  if (!text) return NULL;
  const char *open = "<tool_call>";
  size_t olen = strlen(open);
  const char *close = "</tool_call>";
  size_t clen = strlen(close);

  /* 第一遍：数出所有片段的总长度，拼成 "[a,b,c]" */
  size_t cap = 128, len = 0;
  char *out = (char *)sgl_xmalloc(cap);
  out[0] = 0;
  int n = 0;
  const char *p = text;
  while ((p = strstr(p, open)) != NULL) {
    const char *body = p + olen;
    const char *e = strstr(body, close);
    const char *end = e ? e : body + strlen(body);
    /* 裁掉首尾空白，并去掉可能存在的 ```json 围栏 */
    const char *b = skip_ws(body);
    if (!strncmp(b, "```", 3)) {
      b = strchr(b, '\n') ? strchr(b, '\n') + 1 : b + 3;
      b = skip_ws(b);
    }
    const char *t = end;
    while (t > b && (t[-1] == ' ' || t[-1] == '\t' || t[-1] == '\n' || t[-1] == '\r')) t--;
    if (t - b >= 3 && !strncmp(t - 3, "```", 3)) {
      t -= 3;
      while (t > b && (t[-1] == ' ' || t[-1] == '\t' || t[-1] == '\n' || t[-1] == '\r')) t--;
    }
    if (t > b) {
      size_t need = (size_t)(t - b) + 4;
      while (len + need + 2 > cap) {
        cap *= 2;
        out = (char *)sgl_xrealloc(out, cap);
      }
      if (n == 0) {
        out[len++] = '[';
      } else {
        out[len++] = ',';
      }
      /* 片段若是数组（<tool_call>[...]</tool_call>），直接内联多个 */
      const char *bb = b;
      int is_arr = (*bb == '[');
      if (is_arr) {
        bb++;
        t = t - 1; /* 去掉首尾 [ ] */
        const char *seg = bb;
        int first_in = 1;
        int depth = 0;
        int instr = 0;
        for (const char *q = bb; q <= t; q++) {
          char c = (q < t) ? *q : ',';
          if (instr) {
            if (c == '\\') {
              q++;
              continue;
            }
            if (c == '"') instr = 0;
            continue;
          }
          if (c == '"') instr = 1;
          else if (c == '{') depth++;
          else if (c == '}') depth--;
          else if (c == ',' && depth == 0) {
            size_t seglen = (size_t)(q - seg);
            while (len + seglen + 3 > cap) {
              cap *= 2;
              out = (char *)sgl_xrealloc(out, cap);
            }
            if (!first_in) out[len++] = ',';
            first_in = 0;
            memcpy(out + len, seg, seglen);
            len += seglen;
            n++;
            seg = q + 1;
          }
        }
      } else {
        memcpy(out + len, b, (size_t)(t - b));
        len += (size_t)(t - b);
        n++;
      }
      out[len] = 0;
    }
    p = e ? e + clen : end;
  }
  if (n == 0) {
    sgl_xfree(out);
    return NULL;
  }
  out[len++] = ']';
  out[len] = 0;
  if (n_calls) *n_calls = n;
  return out;
}