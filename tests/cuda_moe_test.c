/* cuda_moe_test.c — GPU MoE 层自测（不需要模型文件，也不需要显卡容量）
 *
 * 覆盖三组形状（都对应真实模型的量级特征）：
 *   1) 小夹具：rows=4  E=64   NE=4   topk=2 mi=32  si=32
 *   2) 解码：  rows=1  E=256  NE=8   topk=2 mi=64  si=64
 *   3) 偏大：  rows=8  E=512  NE=16  topk=4 mi=128 si=128 (norm_topk=0 走另一条分支)
 *
 * 期望：GPU 与 CPU 朴素实现的 max|diff| < 1e-5（f32 同一口径，只差累加顺序）。
 * 编译：CMake 里的 cuda_moe_test（只有开了 CUDA 后端时构建）。*/
#include <math.h>
#include <stdio.h>

#include "sgl/cuda.h"

static int fails = 0;

static void run_case(const char *name, int rows, int E, int NE, int topk, int mi, int si,
                     int norm_topk, float scale) {
  double err = -1.0;
  int rc = sgl_cuda_moe_selftest(rows, E, NE, topk, mi, si, norm_topk, scale, 12345, &err);
  if (rc != 0) {
    printf("FAIL %-10s rows=%d E=%d NE=%d topk=%d mi=%d si=%d: selftest rc=%d\n", name, rows, E,
           NE, topk, mi, si, rc);
    fails++;
    return;
  }
  int ok = err < 1e-5;
  printf("%s %-10s rows=%d E=%d NE=%d topk=%d mi=%d si=%d norm=%d scale=%g  maxerr=%.3e\n",
         ok ? "PASS" : "FAIL", name, rows, E, NE, topk, mi, si, norm_topk, (double)scale, err);
  if (!ok) fails++;
}

int main(void) {
  run_case("fixture", 4, 64, 4, 2, 32, 32, 1, 1.5f);
  run_case("decode", 1, 256, 8, 2, 64, 64, 1, 1.0f);
  run_case("wide", 8, 512, 16, 4, 128, 128, 0, 2.5f);
  run_case("noshared", 3, 128, 6, 3, 48, 0, 1, 1.0f);
  if (fails) {
    printf("\ncuda_moe_test: %d 项失败\n", fails);
    return 1;
  }
  printf("\ncuda_moe_test: 全部通过\n");
  return 0;
}