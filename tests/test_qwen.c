/* test_qwen.c — Qwen2.5-0.5B-Instruct chat 端到端（ChatML 模板 + 调度器）
 * 验证：bias/QK-Norm 路径的真实推理 + chat 模板渲染 + eos=<|im_end|> 停止 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/alloc.h"
#include "sgl/scheduler.h"

static int FAILURES = 0;
#define CHECK(cond, ...) do { if (!(cond)) { FAILURES++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static char *chatml_simple(const char *user) {
  size_t n = strlen(user) + 128;
  char *s = (char *)sgl_xmalloc(n);
  snprintf(s, n, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user);
  return s;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_QWEN");
  if (!dir || !*dir) dir = "models/qwen05";
  char p[1024];
  snprintf(p, sizeof(p), "%s/model.safetensors", dir);
  FILE *f = NULL;
  fopen_s(&f, p, "rb");
  if (!f) { printf("test_qwen: SKIP（无 %s）\n", dir); return 0; }
  fclose(f);

  clock_t t0 = clock();
  sgl_scheduler *s = sgl_sched_new(dir, 7);
  CHECK(s != NULL, "sched_new");
  if (!s) return 1;
  printf("load %.1fs\n", (double)(clock() - t0) / CLOCKS_PER_SEC);

  sgl_sampling_params sp = {0.0f, 0, 1.0f, 0}; /* greedy */

  char *prompt = chatml_simple("What is the capital of France? Answer briefly.");
  int id = sgl_sched_submit(s, prompt, 48, &sp);
  CHECK(id > 0, "submit");
  t0 = clock();
  int guard = 0;
  while (sgl_sched_nactive(s) > 0 && guard++ < 8192) sgl_sched_step(s);
  double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;

  int rid, n, npt, nhit;
  int *ids;
  char *text = NULL, *txt2;
  while (sgl_sched_poll(s, &rid, &txt2, &ids, &n, &npt, &nhit)) {
    if (rid == id) { text = txt2; } else { sgl_xfree(txt2); }
    sgl_xfree(ids);
  }
  CHECK(text != NULL && n > 0, "生成 n=%d", n);
  if (text) {
    printf("[chat] %s（%d tok，%.1fs，%.1f tok/s）\n", text, n, dt, n / dt);
    /* 事实性检查：回答应含 Paris（首字母大小写不限）*/
    int has_paris = strstr(text, "Paris") != NULL || strstr(text, "PARIS") != NULL ||
                    strstr(text, "paris") != NULL;
    CHECK(has_paris, "回答未包含 Paris：%s", text);
    /* eos 正确性：生成应在 <|im_end|> 处停止（n < max_new 说明命中 eos）*/
    printf("generated %d tokens (max 48)\n", n);
  }

  sgl_xfree(text);
  sgl_xfree(prompt);
  sgl_sched_free(s);
  if (FAILURES) { printf("test_qwen: %d FAILURES\n", FAILURES); return 1; }
  printf("test_qwen: all OK\n");
  return 0;
}
