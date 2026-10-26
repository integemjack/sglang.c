/* sgl ops 实现：OpenMP 并行的 CPU 算子 */
#include "sgl/ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/log.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__SSE2__)
#include <emmintrin.h> /* SSE2：f32 4-wide */
#endif
#if defined(__x86_64__)
#include <immintrin.h> /* __m256/_mm256_* 类型与 intrinsics 声明（仅 target attr 函数实际发射）*/
#endif

/* ---------------- t12.2 AVX2 dot（手写 intrinsics，非自动向量化）---------------
 * 当年 -march=avx2 全程序编译的 0xC0000005 是编译器【自动向量化】bug；这里在
 * x86-64-v2 基线上用 target attribute 只对这一个函数启用 AVX2+FMA 指令，
 * 运行时 __builtin_cpu_supports 检测（QEMU/老 CPU 返回 false 走 SSE2）。
 * 8-wide、4 路累加（32 floats/迭代），尾段标量。归约顺序与 SSE2 版不同，
 * 数值差 ~1e-6（同属 fp32 噪声）。 */
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define SGL_HAVE_AVX2_TARGET 1
__attribute__((target("avx2,fma")))
static double dot_k_f32_avx2(const float *a, const float *b, int K);
static double dot_k_f32_avx2(const float *a, const float *b, int K) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  int k = 0;
  for (; k + 32 <= K; k += 32) {
    __m256 a0 = _mm256_loadu_ps(a + k), a1 = _mm256_loadu_ps(a + k + 8);
    __m256 a2 = _mm256_loadu_ps(a + k + 16), a3 = _mm256_loadu_ps(a + k + 24);
    acc0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b + k), acc0);
    acc1 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b + k + 8), acc1);
    acc2 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b + k + 16), acc2);
    acc3 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b + k + 24), acc3);
  }
  __m256 s4 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
  float part[8];
  _mm256_storeu_ps(part, s4);
  double s = ((double)(part[0] + part[1]) + (double)(part[2] + part[3])) +
             ((double)(part[4] + part[5]) + (double)(part[6] + part[7]));
  for (; k < K; k++) s += (double)a[k] * b[k];
  return s;
}
#else
#define SGL_HAVE_AVX2_TARGET 0
#endif

/* K 维 SIMD 内积（16/迭代，4 路累加）；SSE2 基线保证可用，不受 AVX2 向量化 bug 影响 */
static double dot_k_f32_sse2(const float *a, const float *b, int K) {
#if defined(__SSE2__)
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
  __m128 acc2 = _mm_setzero_ps(), acc3 = _mm_setzero_ps();
  int k = 0;
  for (; k + 16 <= K; k += 16) {
    __m128 a0 = _mm_loadu_ps(a + k), a1 = _mm_loadu_ps(a + k + 4);
    __m128 a2 = _mm_loadu_ps(a + k + 8), a3 = _mm_loadu_ps(a + k + 12);
    __m128 w0 = _mm_loadu_ps(b + k), w1 = _mm_loadu_ps(b + k + 4);
    __m128 w2 = _mm_loadu_ps(b + k + 8), w3 = _mm_loadu_ps(b + k + 12);
    acc0 = _mm_add_ps(acc0, _mm_mul_ps(a0, w0));
    acc1 = _mm_add_ps(acc1, _mm_mul_ps(a1, w1));
    acc2 = _mm_add_ps(acc2, _mm_mul_ps(a2, w2));
    acc3 = _mm_add_ps(acc3, _mm_mul_ps(a3, w3));
  }
  __m128 s4 = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
  float part[4];
  _mm_storeu_ps(part, s4);
  double s = (double)(part[0] + part[1]) + (part[2] + part[3]);
  for (; k < K; k++) s += (double)a[k] * b[k];
  return s;
#else
  double s = 0;
  for (int k = 0; k < K; k++) s += (double)a[k] * b[k];
  return s;
#endif
}

/* 运行时一次性选路（race 无害：各线程写入相同值）*/
static double dot_k_f32(const float *a, const float *b, int K) {
#if SGL_HAVE_AVX2_TARGET
  static int has_avx2 = -1;
  if (has_avx2 < 0)
    has_avx2 = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? 1 : 0;
  if (has_avx2) return dot_k_f32_avx2(a, b, K);
#endif
  return dot_k_f32_sse2(a, b, K);
}

/* ---------------- t12.1 OpenBLAS 运行时分派 ----------------
 * SGL_BLAS=0 强制内置；否则首次使用时尝试加载 libopenblas.dll/.so
 * （CMake SGL_OPENBLAS=ON 会把 third_party 预编译 dll 拷到可执行文件旁；
 * 任何 PATH 可见目录也行；SGL_OPENBLAS_PATH 指定完整路径优先）。
 * 只用于大矩阵（M>=2 且 M*N*K >= SGL_BLAS_MINMAC，默认 4M MAC）；
 * 小矩阵/decode GEMV 保留手写 kernel（线程与 launch 开销占主导）。
 * cblas 常量按 cblas.h 值内联，避免头文件依赖。 */
#define SGL_CBLAS_ROWMAJOR 101
#define SGL_CBLAS_NOTRANS 111
#define SGL_CBLAS_TRANS 112
typedef void (*sgl_sgemm_fn)(int, int, int, int, int, int, float, const float *, int,
                             const float *, int, float, float *, int);
static sgl_sgemm_fn sgl_sgemm;
static double sgl_blas_minmac = 4e6; /* 4M MAC */
static int sgl_blas_state = -1;               /* -1 未初始化 0 关 1 开 */

#if defined(_WIN32)
#include <windows.h>
static void *sgl_dlopen(const char *p) { return (void *)LoadLibraryA(p); }
static void *sgl_dlsym(void *h, const char *n) { return (void *)(intptr_t)GetProcAddress((HMODULE)h, n); }
#else
#include <dlfcn.h>
static void *sgl_dlopen(const char *p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
static void *sgl_dlsym(void *h, const char *n) { return dlsym(h, n); }
#endif

static void sgl_blas_init(void) {
  const char *off = getenv("SGL_BLAS");
  if (off && atoi(off) == 0) { sgl_blas_state = 0; return; }
  const char *mm = getenv("SGL_BLAS_MINMAC");
  if (mm && *mm) sgl_blas_minmac = atof(mm);
  void *h = NULL;
  const char *path = getenv("SGL_OPENBLAS_PATH");
  if (path && *path) h = sgl_dlopen(path);
#if defined(_WIN32)
  if (!h) h = sgl_dlopen("libopenblas.dll");
#else
  if (!h) h = sgl_dlopen("libopenblas.so");
#endif
  if (!h) { sgl_blas_state = 0; return; }
  sgl_sgemm = (sgl_sgemm_fn)sgl_dlsym(h, "cblas_sgemm");
  sgl_blas_state = sgl_sgemm ? 1 : 0;
  SGL_INFO("OpenBLAS %s (sgemm dispatch, min MAC %g)",
           sgl_blas_state == 1 ? "loaded" : "loaded but cblas_sgemm missing", sgl_blas_minmac);
}

void sgl_dot_f32(const float *a, const float *b, int n, float *out_sum);

void sgl_dot_f32(const float *a, const float *b, int n, float *out_sum) {
  *out_sum = (float)dot_k_f32(a, b, n);
}

void sgl_matmul(const float *A, const float *B, const float *bias, float *C,
                int M, int N, int K) {
  /* C 零初始化或 bias 广播 */
  if (bias) {
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
      float *crow = C + (size_t)i * N;
      memcpy(crow, bias, sizeof(float) * N);
    }
  } else {
    memset(C, 0, (size_t)M * N * sizeof(float));
  }

  /* i-k-j：k 推进时对 crow 做 axpy，可向量化 */
#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < M; i++) {
    const float *arow = A + (size_t)i * K;
    float *crow = C + (size_t)i * N;
    for (int k = 0; k < K; k++) {
      const float a = arow[k];
      if (a == 0.0f) continue;
      const float *brow = B + (size_t)k * N;
      /* crow += a * brow */
      int j = 0;
      for (; j + 8 <= N; j += 8) {
        crow[j + 0] += a * brow[j + 0];
        crow[j + 1] += a * brow[j + 1];
        crow[j + 2] += a * brow[j + 2];
        crow[j + 3] += a * brow[j + 3];
        crow[j + 4] += a * brow[j + 4];
        crow[j + 5] += a * brow[j + 5];
        crow[j + 6] += a * brow[j + 6];
        crow[j + 7] += a * brow[j + 7];
      }
      for (; j < N; j++) crow[j] += a * brow[j];
    }
  }
}

void sgl_layernorm(const float *x, const float *gamma, const float *beta,
                   float *y, int M, int N, float eps) {
#pragma omp parallel for
  for (int i = 0; i < M; i++) {
    const float *xr = x + (size_t)i * N;
    float *yr = y + (size_t)i * N;
    double mean = 0;
    for (int j = 0; j < N; j++) mean += xr[j];
    mean /= N;
    double var = 0;
    for (int j = 0; j < N; j++) {
      double d = xr[j] - mean;
      var += d * d;
    }
    var /= N;
    float inv = (float)(1.0 / sqrt(var + eps));
    for (int j = 0; j < N; j++)
      yr[j] = (xr[j] - mean) * inv * gamma[j] + beta[j];
  }
}

void sgl_softmax_inplace(float *row, int n) {
  float maxv = row[0];
  for (int i = 1; i < n; i++)
    if (row[i] > maxv) maxv = row[i];
  double sum = 0;
  for (int i = 0; i < n; i++) {
    row[i] = expf(row[i] - maxv);
    sum += row[i];
  }
  float inv = (float)(1.0 / sum);
  for (int i = 0; i < n; i++) row[i] *= inv;
}

void sgl_gelu_inplace(float *row, int n) {
  /* 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) */
  const float k = 0.7978845608028654f; /* sqrt(2/pi) */
  for (int i = 0; i < n; i++) {
    float x = row[i];
    float inner = k * (x + 0.044715f * x * x * x);
    row[i] = 0.5f * x * (1.0f + tanhf(inner));
  }
}

void sgl_matmul_nt(const float *A, const float *W, const float *bias, float *C,
                   int M, int N, int K) {
  /* W[N,K] 行主序（PyTorch Linear）：C[m,n] = Σ_k A[m,k] * W[n,k] + bias[n]
   * 大矩阵优先 OpenBLAS（t12.1 运行时分派，C[M,N] = A[M,K] @ W[N,K]^T）：
   * CblasRowMajor 下直接传 W 为 Trans（lda=K）——OpenBLAS 文档的 [N,K] 列主
   * Fortran 习惯由 CBLAS RowMajor 接口吸收，bias 随后广播。 */
  if (sgl_blas_state < 0) sgl_blas_init();
  if (sgl_sgemm && M >= 2 && (double)M * N * K >= sgl_blas_minmac) {
    sgl_sgemm(SGL_CBLAS_ROWMAJOR, SGL_CBLAS_NOTRANS, SGL_CBLAS_TRANS,
              M, N, K, 1.0f, A, K, W, K, 0.0f, C, N);
    if (bias) {
#pragma omp parallel for
      for (int m = 0; m < M; m++) {
        float *crow = C + (size_t)m * N;
        for (int n = 0; n < N; n++) crow[n] += bias[n];
      }
    }
    return;
  }
  /* K 维 SSE2 SIMD 内积。
   * n-outer 分块（Spec-v2 / 批量前向复用）：按 m 块（TM 行）切，块内 n 外层——
   * 每个 m 块 W 只从内存过一遍、被块内全部 TM 行激活复用（原来逐行各过一遍 W，
   * M 行 = M×|W| 流量）。每个 C[m][n] 仍是独立的一次 dot_k_f32(ar,wr,K)，
   * 浮点结果与逐行版本逐位一致（只换循环次序，不改累加方式）。 */
#define SGL_MM_NT_TM 8
  for (int m0 = 0; m0 < M; m0 += SGL_MM_NT_TM) {
    const int me = m0 + SGL_MM_NT_TM < M ? m0 + SGL_MM_NT_TM : M;
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      const float *wr = W + (size_t)n * K;
      const float b = bias ? bias[n] : 0.0f;
      for (int m = m0; m < me; m++)
        C[(size_t)m * N + n] = (float)dot_k_f32(A + (size_t)m * K, wr, K) + b;
    }
  }
#undef SGL_MM_NT_TM
}

void sgl_embedding_get(const float *table, const int *ids, int n, int d,
                       float *out) {
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    memcpy(out + (size_t)i * d, table + (size_t)ids[i] * d, sizeof(float) * d);
}

void sgl_silu_inplace(float *row, int n) {
  for (int i = 0; i < n; i++) {
    float x = row[i];
    row[i] = x / (1.0f + expf(-x));
  }
}

void sgl_rmsnorm(const float *x, const float *weight, float *y, int M, int N,
                 float eps) {
#pragma omp parallel for
  for (int m = 0; m < M; m++) {
    const float *xr = x + (size_t)m * N;
    float *yr = y + (size_t)m * N;
    double ss = 0;
    for (int i = 0; i < N; i++) ss += (double)xr[i] * xr[i];
    float inv = (float)(1.0 / sqrt(ss / N + eps));
    for (int i = 0; i < N; i++) yr[i] = xr[i] * inv * weight[i];
  }
}

void sgl_rmsnorm_p1(const float *x, const float *weight, float *y, int M, int N,
                    float eps) {
#pragma omp parallel for
  for (int m = 0; m < M; m++) {
    const float *xr = x + (size_t)m * N;
    float *yr = y + (size_t)m * N;
    double ss = 0;
    for (int i = 0; i < N; i++) ss += (double)xr[i] * xr[i];
    float inv = (float)(1.0 / sqrt(ss / N + eps));
    for (int i = 0; i < N; i++) yr[i] = xr[i] * inv * (1.0f + weight[i]);
  }
}

void sgl_rope_inplace(float *qk, int n, int head_dim, int pos0, double theta) {
  /* HF llama 标准式（GPT-NeoX 风格 rotate_half，配对 (i, i+D/2)，双精度频率）。
 * n 行是【同一位置的 n 个头】（或 n 个 token 的同一头——调用方保证），
 *   全部用同一个位置 pos0：行号不是位置！调用方对 token t 传 pos0+t。 */
#pragma omp parallel for
  for (int t = 0; t < n; t++) {
    float *head = qk + (size_t)t * head_dim;
    int pos = pos0;
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
      double freq = pow(theta, -(double)(2 * i) / head_dim);
      double ang = (double)pos * freq;
      float c = (float)cos(ang), s = (float)sin(ang);
      float a = head[i], b = head[i + half];
      head[i] = a * c - b * s;
      head[i + half] = a * s + b * c;
    }
  }
}
