// probe4: KV 池写入策略的实测 —— slice_update 到底是不是每次复制整个池？
//
// 这决定 MLX 后端的 KV 设计：
//   A) 若 MLX 对 slice_update 做了「引用计数=1 时就地改」（donation），
//      那么 g->kc[l] = slice_update(g->kc[l], ...) 是 O(写入行数)，可以直接用；
//   B) 若每次都整池复制，decode 一步要复制 2*L 个 [ctx,KVH,D]，
//      qwen05@ctx4096 就是 ~48MB/步、B=8 时 ~384MB/步 —— 必须换策略
//      （按序列增长的 concatenate 缓存，或自定义 metal_kernel 就地写）。
// 实测三种写法的每步耗时：slice_update 重赋值 / concatenate 增长 / 只读 slice。
#include <mlx/mlx.h>

#include <chrono>
#include <cstdio>
#include <vector>

namespace mx = mlx::core;

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
  const int ctx = 4096, KVH = 2, D = 64, L = 24; // qwen05 的形状
  const int STEPS = 200;
  printf("池形状 [%d,%d,%d] fp16 = %.1f MB/层，%d 层 K+V 共 %.0f MB\n", ctx, KVH, D,
         (double)ctx * KVH * D * 2 / 1048576.0, L, (double)L * 2 * ctx * KVH * D * 2 / 1048576.0);

  // ---- A) slice_update 重赋值（每层一个池数组）----
  {
    // mx::array 没有默认构造函数 → 不能 vector<array> kc(L)，只能 reserve+push_back
    std::vector<mx::array> kc, vc;
    kc.reserve(L); vc.reserve(L);
    for (int l = 0; l < L; l++) {
      kc.push_back(mx::zeros(mx::Shape{ctx, KVH, D}, mx::float16));
      vc.push_back(mx::zeros(mx::Shape{ctx, KVH, D}, mx::float16));
    }
    for (int l = 0; l < L; l++) { kc[l].eval(); vc[l].eval(); }
    auto upd = mx::ones(mx::Shape{1, KVH, D}, mx::float16);
    upd.eval();
    double t0 = now();
    for (int s = 0; s < STEPS; s++) {
      int row = s % ctx;
      for (int l = 0; l < L; l++) {
        kc[l] = mx::slice_update(kc[l], upd, mx::Shape{row, 0, 0}, mx::Shape{row + 1, KVH, D});
        vc[l] = mx::slice_update(vc[l], upd, mx::Shape{row, 0, 0}, mx::Shape{row + 1, KVH, D});
      }
      // 只 eval 最后一层就足以驱动整图？不行，每层独立 → 全 eval
      for (int l = 0; l < L; l++) {
        kc[l].eval();
        vc[l].eval();
      }
    }
    double dt = (now() - t0) / STEPS * 1000.0;
    printf("A) slice_update 重赋值      : %.3f ms/步（%d 层 x K+V，每步写 1 行）\n", dt, L);
    printf("   → 若接近「整池复制」应有 %.1f MB/步 的搬运量\n",
           (double)L * 2 * ctx * KVH * D * 2 / 1048576.0);
  }

  // ---- B) concatenate 增长（mlx-lm 的做法：只复制已用长度）----
  {
    std::vector<mx::array> kc, vc;
    kc.reserve(L); vc.reserve(L);
    for (int l = 0; l < L; l++) {
      kc.push_back(mx::zeros(mx::Shape{0, KVH, D}, mx::float16));
      vc.push_back(mx::zeros(mx::Shape{0, KVH, D}, mx::float16));
    }
    auto upd = mx::ones(mx::Shape{1, KVH, D}, mx::float16);
    double t0 = now();
    for (int s = 0; s < STEPS; s++) {
      for (int l = 0; l < L; l++) {
        kc[l] = mx::concatenate(std::vector<mx::array>{kc[l], upd}, 0);
        vc[l] = mx::concatenate(std::vector<mx::array>{vc[l], upd}, 0);
      }
      for (int l = 0; l < L; l++) {
        kc[l].eval();
        vc[l].eval();
      }
    }
    double dt = (now() - t0) / STEPS * 1000.0;
    printf("B) concatenate 增长(长度%d) : %.3f ms/步\n", STEPS, dt);
  }

  // ---- C) 基线：什么都不写，只 eval 一遍已有的池（量出 eval 的固定开销）----
  {
    std::vector<mx::array> kc;
    kc.reserve(L);
    for (int l = 0; l < L; l++) kc.push_back(mx::zeros(mx::Shape{ctx, KVH, D}, mx::float16));
    for (int l = 0; l < L; l++) kc[l].eval();
    double t0 = now();
    for (int s = 0; s < STEPS; s++) {
      for (int l = 0; l < L; l++) {
        auto sl = mx::slice(kc[l], mx::Shape{0, 0, 0}, mx::Shape{1024, KVH, D});
        sl.eval();
      }
    }
    double dt = (now() - t0) / STEPS * 1000.0;
    printf("C) 只读 slice[0:1024] x%d 层  : %.3f ms/步（注意力读 KV 的量级）\n", L, dt);
  }
  return 0;
}
