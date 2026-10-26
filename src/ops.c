/* sgl ops 实现：OpenMP 并行的 CPU 算子 */
#include "sgl/ops.h"

#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__SSE2__)
#include <emmintrin.h> /* SSE2：f32 4-wide */
#endif

/* K 维 SIMD 内积（16/迭代，4 路累加）；SSE2 基线保证可用，不受 AVX2 向量化 bug 影响 */
static inline double dot_k_f32(const float *a, const float *b, int K) {
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
 * K 维 SSE2 SIMD 内积 */
#pragma omp parallel for
  for (int m = 0; m < M; m++) {
    const float *ar = A + (size_t)m * K;
    float *cr = C + (size_t)m * N;
    for (int n = 0; n < N; n++) {
      const float *wr = W + (size_t)n * K;
      cr[n] = (float)dot_k_f32(ar, wr, K) + (bias ? bias[n] : 0.0f);
    }
  }
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
