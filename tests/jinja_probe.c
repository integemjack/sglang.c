/* 打印 27B 的 chat_template 对任意消息的渲染结果（离线，不占 GPU）*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sgl/jinja.h"
#include "sgl/json.h"

int main(int argc, char **argv) {
  const char *dir = argc > 1 ? argv[1] : "/root/autodl-tmp/Qwen3.8-27B-FP8";
  const char *user = argc > 2 ? argv[2] : "Say hello in one short sentence.";
  sgl_jinja *j = sgl_jinja_load_dir(dir);
  if (!j) { printf("NO TEMPLATE in %s\n", dir); return 1; }
  char buf[4096];
  snprintf(buf, sizeof(buf),
           "[{\"role\":\"user\",\"content\":\"%s\"}]", user);
  sgl_json *msgs = sgl_json_parse(buf, strlen(buf));
  if (!msgs) { printf("BAD MSGS JSON\n"); return 1; }
  char *out = NULL;
  size_t n = 0;
  int rc = sgl_jinja_render(j, msgs, 1, NULL, &out, &n);
  printf("rc=%d len=%zu\n---PROMPT BEGIN---\n", rc, n);
  if (out) fwrite(out, 1, n, stdout);
  printf("\n---PROMPT END---\n");
  sgl_json_free(msgs);
  sgl_jinja_free(j);
  if (out) sgl_xfree(out);
  return 0;
}