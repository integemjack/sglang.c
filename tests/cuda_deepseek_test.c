/* cuda_deepseek_test — GPU 侧的 MLA + MoE 前向是否正确（需要真显卡，不进 ctest）
 *
 * 参考就是 CPU 引擎（src/deepseek.c，本身已由 test_deepseek 对着 numpy 参考验证
 * 到 f32 舍入）。两边都是 f32 + SGEMM，所以这里只应差累加顺序：
 *   - prefill 全部行的 logits：maxerr 目标 < 2e-4（32 维小模型，相对误差 ~1e-6）
 *   - 增量 decode：逐 token 前向，每步与 CPU 同位置比（覆盖 KV 写入/位置）
 *   - 分块 prefill：与一次性 prefill 必须几乎相同（4 个 chunk 拼回去）
 * 路由错了会直接把 logits 打到 1e-1 量级，所以第一条也顺带覆盖 MoE 路由。
 *
 * 用法：cuda_deepseek_test [model_dir]
 *   不给参数时依次试 models/deepseek-rand{,-v3,-3d,-ql0,-il}（也认 SGL_MODELS_DIR）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/cuda_deepseek.h"
#include "sgl/deepseek.h"
#include "sgl/port.h"

static int fails;
#define CHECK(c, msg)                                          \
  do {                                                         \
    if (!(c)) {                                                \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);      \
      fails++;                                                 \
    }                                                          \
  } while (0)

static int file_exists(const char *p) {
  FILE *f = NULL;
  if (fopen_s(&f, p, "rb") != 0 || !f) return 0;
  fclose(f);
  return 1;
}

static const char *pick_model(void) {
  static const char *cand[] = {"deepseek-rand", "deepseek-rand-v3", "deepseek-rand-3d",
                              "deepseek-rand-ql0", "deepseek-rand-il"};
  static char buf[1024];
  const char *root = getenv("SGL_MODELS_DIR");
  if (!root || !*root) root = "../models";
  for (int i = 0; i < 5; i++) {
    snprintf(buf, sizeof(buf), "%s/%s/config.json", root, cand[i]);
    if (file_exists(buf)) {
      snprintf(buf, sizeof(buf), "%s/%s", root, cand[i]);
      return buf;
    }
  }
  return NULL;
}

static double max_abs_diff(const float *a, const float *b, int n) {
  double m = 0;
  for (int i = 0; i < n; i++) {
    double d = fabs((double)a[i] - (double)b[i]);
    if (d > m) m = d;
  }
  return m;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : pick_model();
  if (!dir) {
    printf("SKIP: 找不到 deepseek 小模型（tests/gen_random_deepseek.py）\n");
    return 0;
  }
  char p[1200];
  snprintf(p, sizeof(p), "%s/model.safetensors", dir);
  if (!file_exists(p)) {
    printf("SKIP: %s 没有权重\n", dir);
    return 0;
  }

  sgl_deepseek *cpu = sgl_deepseek_load(dir);
  CHECK(cpu != NULL, "cpu load");
  if (!cpu) return 1;
  int V = sgl_deepseek_vocab(cpu);
  int ctx = sgl_deepseek_ctx(cpu);
  printf("[model] %s  V=%d ctx=%d L=%d\n", dir, V, ctx, cpu->n_layer);

  sgl_cuda_ds *gpu = sgl_cuda_ds_from(cpu);
  CHECK(gpu != NULL, "cuda upload");
  if (!gpu) return 1;
  printf("[vram] %.1f MB\n", (double)sgl_cuda_ds_mem_mb(gpu));

  int n = 12;
  if (n > ctx - 4) n = ctx - 4;
  int toks[64];
  for (int i = 0; i < n; i++) toks[i] = (i * 7 + 3) % V;

  int *pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)ctx);
  CHECK(sgl_deepseek_pages_alloc(cpu, pages, ctx) == 0, "pages alloc");
  float *lc = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * (size_t)n);
  float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * (size_t)n);

  /* CPU：逐 token 前向，拿每行 logits 当参考 */
  for (int t = 0; t < n; t++) {
    if (sgl_deepseek_forward(cpu, &toks[t], 1, t, pages, lc + (size_t)t * V) != 0) {
      printf("FAIL: cpu forward t=%d\n", t);
      return 1;
    }
  }
  /* GPU：一次 prefill 出 n 行 */
  if (sgl_cuda_ds_forward_n(gpu, toks, n, 0, lg, 1) != 0) {
    printf("FAIL: cuda forward_n\n");
    return 1;
  }
  double e = max_abs_diff(lc, lg, V * n);
  printf("[prefill] %d 行 logits maxerr = %.3e（CPU vs GPU）\n", n, e);
  CHECK(e < 2e-4, "prefill logits 与 CPU 一致");

  /* 增量 decode */
  {
    sgl_cuda_ds_reset(gpu);
    float *one = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
    double worst = 0;
    int worst_t = -1;
    for (int t = 0; t < n; t++) {
      if (sgl_cuda_ds_decode(gpu, toks[t], t, one) != 0) {
        printf("FAIL: cuda decode t=%d\n", t);
        fails++;
        break;
      }
      double d = max_abs_diff(lc + (size_t)t * V, one, V);
      if (d > worst) {
        worst = d;
        worst_t = t;
      }
    }
    printf("[decode] 逐 token 增量 maxerr = %.3e（最差 t=%d）\n", worst, worst_t);
    CHECK(worst < 2e-4, "增量 decode 与 CPU 一致");
    sgl_xfree(one);
  }

  /* 分块 prefill == 一次 prefill */
  {
    sgl_cuda_ds_reset(gpu);
    float *chunk = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * (size_t)n);
    int half = n / 2, ok = 1;
    if (sgl_cuda_ds_forward_n(gpu, toks, half, 0, chunk, 1) != 0 ||
        sgl_cuda_ds_forward_n(gpu, toks + half, n - half, half, chunk + (size_t)half * V, 1) != 0) {
      printf("FAIL: cuda 分块 prefill\n");
      fails++;
      ok = 0;
    }
    if (ok) {
      double d = max_abs_diff(lg, chunk, V * n);
      printf("[chunk] 分块(%d+%d) vs 一次 prefill maxerr = %.3e\n", half, n - half, d);
      CHECK(d < 1e-5, "分块 prefill 与整体等价");
    }
    sgl_xfree(chunk);
  }

  sgl_xfree(lc);
  sgl_xfree(lg);
  sgl_deepseek_pages_unref(cpu, pages, ctx);
  sgl_xfree(pages);

  /* 贪婪生成 16 步：CPU 与 GPU 的 token 序列必须完全一致（服务层就是这么采的）*/
  {
    int seq_cpu[16], seq_gpu[16];
    float *row = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
    int *p2 = (int *)sgl_xmalloc(sizeof(int) * (size_t)ctx);
    CHECK(sgl_deepseek_pages_alloc(cpu, p2, ctx) == 0, "greedy pages alloc");
    int np = 0;
    /* CPU：从头 prefill（前面的数组已 free，这里重新算）*/
    for (int t = 0; t < n; t++)
      if (sgl_deepseek_forward(cpu, &toks[t], 1, t, p2, row) != 0) {
        printf("FAIL: cpu re-prefill t=%d\n", t);
        fails++;
        break;
      }
    np = n;
    for (int i = 0; i < 16; i++) {
      int best = 0;
      for (int v = 1; v < V; v++)
        if (row[v] > row[best]) best = v;
      seq_cpu[i] = best;
      if (sgl_deepseek_forward(cpu, &best, 1, np, p2, row) != 0) { seq_cpu[i] = -1; break; }
      np++;
    }
    sgl_cuda_ds_reset(gpu);
    if (sgl_cuda_ds_forward_n(gpu, toks, n, 0, row, 0) != 0) {
      printf("FAIL: greedy prefill\n");
      fails++;
    } else {
      for (int i = 0; i < 16; i++) {
        int best = 0;
        for (int v = 1; v < V; v++)
          if (row[v] > row[best]) best = v;
        seq_gpu[i] = best;
        if (sgl_cuda_ds_decode(gpu, best, n + i, row) != 0) { seq_gpu[i] = -1; break; }
      }
      int same = 1;
      for (int i = 0; i < 16; i++)
        if (seq_cpu[i] != seq_gpu[i]) same = 0;
      printf("[greedy] 16 步 token 序列%s\n", same ? "完全一致" : "不一致");
      if (!same) {
        printf("  cpu:");
        for (int i = 0; i < 16; i++) printf(" %d", seq_cpu[i]);
        printf("\n  gpu:");
        for (int i = 0; i < 16; i++) printf(" %d", seq_gpu[i]);
        printf("\n");
      }
      CHECK(same, "CPU 与 GPU 贪婪 16 步 token 序列一致");
    }
    sgl_deepseek_pages_unref(cpu, p2, ctx);
    sgl_xfree(row);
    sgl_xfree(p2);
  }
  sgl_cuda_ds_free(gpu);
  sgl_deepseek_free(cpu);

  if (fails) {
    printf("cuda_deepseek_test: %d FAILED\n", fails);
    return 1;
  }
  printf("cuda_deepseek_test: all OK\n");
  return 0;
}