/* test_dp — P1 数据并行副本：sgl_sched_add_replica 起多份 base 副本，
 * 请求按「当前运行数最少」路由，各副本 KV 池独立、统计累加。
 * 模型目录用 SGL_TEST_MODEL（默认 models/gpt2），没有就 SKIP。 */
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

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = getenv("SGL_TEST_MODEL");
  if (!dir || !*dir) dir = "../models/gpt2";
  char cfg[1024];
  snprintf(cfg, sizeof(cfg), "%s/config.json", dir);
  FILE *f = fopen(cfg, "rb");
  if (!f) {
    printf("test_dp: SKIP（无 %s）\n", cfg);
    return 0;
  }
  fclose(f);

  /* 单副本基线 */
  sgl_scheduler *one = sgl_sched_new(dir, 1234);
  if (!one) {
    printf("test_dp: FAIL 单副本创建失败\n");
    return 1;
  }
  sgl_sched_stats s1 = sgl_sched_get_stats(one);
  check(s1.n_replicas == 1, "单副本 n_replicas == 1");
  int pages1 = s1.pool_pages;

  /* 加两份副本 → 共 3 份 */
  check(sgl_sched_add_replica(one, dir) == 2, "add_replica 返回 2（含主副本）");
  check(sgl_sched_add_replica(one, dir) == 3, "add_replica 返回 3（含主副本）");
  sgl_sched_stats s3 = sgl_sched_get_stats(one);
  check(s3.n_replicas == 3, "三副本 n_replicas == 3");
  check(s3.pool_pages == 3 * pages1, "池容量按副本累加");
  check(s3.pool_free == 3 * s1.pool_free, "空闲页按副本累加");

  /* 提交 6 个并发请求：应分到 3 个副本（每个最多 2），全部都能跑完且输出正确 */
  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp)); /* greedy */
  int ids[6];
  const char *prompt = "The capital of France is";
  for (int i = 0; i < 6; i++) ids[i] = sgl_sched_submit(one, prompt, 6, &sp);
  check(ids[0] > 0, "submit 返回正 id");
  /* 6 个都提交完时，运行中+等待中的请求应当分布到 3 个副本上：
   * 这里用「统计里 n_running+n_waiting == 6」间接验证它们都进了队列，
   * 再用下面的完成与一致性验证路由没把请求丢掉。*/
  sgl_sched_stats run = sgl_sched_get_stats(one);
  check(run.n_running + run.n_waiting == 6, "6 个请求都在队列里");

  int done = 0;
  char *outs[6];
  memset(outs, 0, sizeof(outs));
  for (int step = 0; step < 20000 && done < 6; step++) {
    sgl_sched_step(one);
    int rid;
    char *text = NULL;
    int *gid = NULL, n = 0, npt = 0, nhit = 0;
    while (sgl_sched_poll(one, &rid, &text, &gid, &n, &npt, &nhit)) {
      /* 找它是第几个请求 */
      for (int i = 0; i < 6; i++)
        if (ids[i] == rid) {
          outs[i] = text;
          done++;
          text = NULL;
          break;
        }
      sgl_xfree(text);
      sgl_xfree(gid);
    }
  }
  check(done == 6, "6 个请求全部完成");
  int same = 1;
  for (int i = 1; i < 6; i++)
    if (!outs[i] || !outs[0] || strcmp(outs[i], outs[0]) != 0) same = 0;
  check(same, "贪心解码下 6 个请求输出一致（副本行为与主副本相同）");
  if (outs[0]) printf("out: '%s'\n", outs[0]);
  for (int i = 0; i < 6; i++) sgl_xfree(outs[i]);

  sgl_sched_free(one);
  if (failures) {
    printf("test_dp: %d FAILURES\n", failures);
    return 1;
  }
  printf("test_dp: all OK\n");
  return 0;
}