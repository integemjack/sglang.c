/* test_reload — P1 权重热更新（/update_weights_from_disk 的后端）：
 * 同一目录重载后输出不变；有在飞请求时拒绝（-2）；实例下标越界报错。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/scheduler.h"

static int failures = 0;
static void check(int ok, const char *what) {
  if (!ok) {
    printf("FAIL: %s\n", what);
    failures++;
  }
}

static char *run_once(sgl_scheduler *s, const char *prompt, int max_new) {
  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp));
  int id = sgl_sched_submit(s, prompt, max_new, &sp);
  if (id < 0) return NULL;
  for (int step = 0; step < 20000; step++) {
    sgl_sched_step(s);
    int rid;
    char *text = NULL;
    int *gids = NULL, n = 0, npt = 0, nhit = 0;
    while (sgl_sched_poll(s, &rid, &text, &gids, &n, &npt, &nhit)) {
      sgl_xfree(gids);
      if (rid == id) return text;
      sgl_xfree(text);
      text = NULL;
    }
  }
  return NULL;
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = getenv("SGL_TEST_MODEL");
  if (!dir || !*dir) dir = "../models/gpt2";
  char cfg[1024];
  snprintf(cfg, sizeof(cfg), "%s/config.json", dir);
  FILE *f = fopen(cfg, "rb");
  if (!f) {
    printf("test_reload: SKIP（无 %s）\n", cfg);
    return 0;
  }
  fclose(f);

  sgl_scheduler *s = sgl_sched_new(dir, 7);
  if (!s) {
    printf("test_reload: FAIL 调度器创建失败\n");
    return 1;
  }
  const char *prompt = "The meaning of life is";
  char *before = run_once(s, prompt, 6);
  check(before != NULL, "重载前生成成功");

  check(sgl_sched_reload_weights(s, -1, dir) == 0, "重载所有 base 副本返回 0");
  char *after = run_once(s, prompt, 6);
  check(after != NULL, "重载后生成成功");
  check(before && after && strcmp(before, after) == 0, "重载同目录权重后输出一致");
  if (before) printf("out: '%s'\n", before);

  /* 越界实例 */
  check(sgl_sched_reload_weights(s, 99, dir) == -1, "越界实例返回 -1");
  /* 不存在的目录 */
  check(sgl_sched_reload_weights(s, 0, "../models/__no_such_model__") == -1,
        "不存在的模型目录返回 -1");
  /* 失败后实例仍可用 */
  char *again = run_once(s, prompt, 6);
  check(again && before && strcmp(again, before) == 0, "失败的重载不影响原模型");

  /* 有在飞请求时拒绝：先提交一个请求再立刻重载 */
  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp));
  int id = sgl_sched_submit(s, prompt, 32, &sp);
  check(id > 0, "提交在飞请求");
  check(sgl_sched_reload_weights(s, 0, dir) == -2, "有在飞请求时返回 -2（busy）");
  /* 让它跑完，清理 */
  for (int step = 0; step < 20000; step++) {
    sgl_sched_step(s);
    int rid;
    char *text = NULL;
    int *gids = NULL, n = 0, npt = 0, nhit = 0;
    while (sgl_sched_poll(s, &rid, &text, &gids, &n, &npt, &nhit)) {
      sgl_xfree(text);
      sgl_xfree(gids);
      if (rid == id) step = 20000;
    }
  }
  check(sgl_sched_reload_weights(s, 0, dir) == 0, "请求跑完后重载成功");

  sgl_xfree(before);
  sgl_xfree(after);
  sgl_xfree(again);
  sgl_sched_free(s);
  if (failures) {
    printf("test_reload: %d FAILURES\n", failures);
    return 1;
  }
  printf("test_reload: all OK\n");
  return 0;
}