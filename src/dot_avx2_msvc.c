/* sgl dot_avx2_msvc.c — MSVC 版的 AVX2 内积（G18 补记，2026-10-27）
 *
 * 为什么单独一个文件：GCC 可以用 `__attribute__((target("avx2,fma")))` 只给一个函数
 * 开 AVX2（ops.c 的 dot_k_f32_avx2，见 t12.2 的注释），**MSVC 没有等价的函数级
 * target 属性**，intrinsics 是否可用由编译选项 /arch:AVX2 决定。所以这里把 AVX2
 * 版本放进独立编译单元，由 CMake 只给这个文件加 /arch:AVX2，其余代码仍是
 * x86-64-v2 基线（避免 t12.2 记录的「全程序 AVX2 会踩编译器自动向量化 bug」）。
 *
 * 运行时分派：sgl_dot_avx2_available() 用 CPUID 检查 AVX2+FMA+OSXSAVE，不支持就
 * 返回 0，ops.c 会退回 SSE2 版（不存在 SIGILL 风险，QEMU/老 CPU 也能跑）。
 *
 * 归约顺序与 SSE2 版不同（8-wide × 4 路累加 + FMA），数值差 ~1e-6，同属 fp32 噪声。
 * 实测（qwen05 930-token prefill，MinGW/GCC 单线程 31.9s vs MSVC 无 AVX2 326.2s）
 * 见 docs/BENCH.md 的 G18 补记 —— 这个差异几乎全部来自这条内积路径。 */

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
#include <immintrin.h>
#include <intrin.h>

int sgl_dot_avx2_available(void) {
  int r[4] = {0, 0, 0, 0};
  __cpuid(r, 1);
  int osxsave = (r[2] >> 27) & 1;
  int avx = (r[2] >> 28) & 1;
  int fma = (r[2] >> 12) & 1;
  if (!(osxsave && avx && fma)) return 0;
  __cpuid(r, 7);
  int avx2 = (r[1] >> 5) & 1;
  return avx2 && fma;
}

double sgl_dot_f32_avx2(const float *a, const float *b, int K) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  int k = 0;
  for (; k + 32 <= K; k += 32) {
    __m256 a0 = _mm256_loadu_ps(a + k), a1 = _mm256_loadu_ps(a + k + 8);
    __m256 a2 = _mm256_loadu_ps(a + k + 16), a3 = _mm256_loadu_ps(a + k + 24);
    __m256 w0 = _mm256_loadu_ps(b + k), w1 = _mm256_loadu_ps(b + k + 8);
    __m256 w2 = _mm256_loadu_ps(b + k + 16), w3 = _mm256_loadu_ps(b + k + 24);
    acc0 = _mm256_fmadd_ps(a0, w0, acc0);
    acc1 = _mm256_fmadd_ps(a1, w1, acc1);
    acc2 = _mm256_fmadd_ps(a2, w2, acc2);
    acc3 = _mm256_fmadd_ps(a3, w3, acc3);
  }
  for (; k + 8 <= K; k += 8)
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
  __m256 s8 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
  __m128 lo = _mm256_castps256_ps128(s8);
  __m128 hi = _mm256_extractf128_ps(s8, 1);
  __m128 s4 = _mm_add_ps(lo, hi);
  float part[4];
  _mm_storeu_ps(part, s4);
  double s = (double)(part[0] + part[1]) + (part[2] + part[3]);
  for (; k < K; k++) s += (double)a[k] * b[k];
  return s;
}

#else /* 非 MSVC/x64：给一个永远不可用的桩，ops.c 仍走它自己的 GCC AVX2 或 SSE2 */

int sgl_dot_avx2_available(void) { return 0; }
double sgl_dot_f32_avx2(const float *a, const float *b, int K) {
  (void)a; (void)b; (void)K;
  return 0.0;
}

#endif