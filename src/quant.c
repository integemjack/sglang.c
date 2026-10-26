/* sgl quant 实现：int8 权重/激活/KV 量化 */
#include "sgl/quant.h"

#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"

#if defined(__SSE4_1__)
#include <smmintrin.h> /* SSE4.1：_mm_cvtepi8_epi32 */
#endif

float sgl_quant_row_i8(const float *x, int8_t *out, int n) {
  float maxabs = 0;
  for (int i = 0; i < n; i++) {
    float a = x[i] < 0 ? -x[i] : x[i];
    if (a > maxabs) maxabs = a;
  }
  if (maxabs <= 0) {
    memset(out, 0, (size_t)n);
    return 0;
  }
  float scale = maxabs / 127.0f;
  float inv = 1.0f / scale;
  for (int i = 0; i < n; i++) {
    int v = (int)(x[i] * inv + (x[i] >= 0 ? 0.5f : -0.5f));
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    out[i] = (int8_t)v;
  }
  return scale;
}

void sgl_quant_rows_i8(const float *x, int8_t *out, float *scales, int rows, int dim) {
  for (int r = 0; r < rows; r++)
    scales[r] = sgl_quant_row_i8(x + (size_t)r * dim, out + (size_t)r * dim, dim);
}

void sgl_dequant_row_i8(const int8_t *q, float scale, float *out, int n) {
  for (int i = 0; i < n; i++) out[i] = (float)q[i] * scale;
}

void sgl_matmul_w8a16(const float *A, const int8_t *W, const float *sw,
                      const float *bias, float *C, int M, int N, int K) {
#pragma omp parallel for
  for (int m = 0; m < M; m++) {
    const float *ar = A + (size_t)m * K;
    float *cr = C + (size_t)m * N;
    for (int n = 0; n < N; n++) {
      const int8_t *wr = W + (size_t)n * K;
      float s;
#if defined(__SSE4_1__)
      /* int8×16/迭代：cvtepi8_epi32(4×) → f32 mul/acc，4 路累加 */
      __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
      __m128 acc2 = _mm_setzero_ps(), acc3 = _mm_setzero_ps();
      int k = 0;
      for (; k + 16 <= K; k += 16) {
        __m128i wq0 = _mm_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(wr + k)));
        __m128i wq1 = _mm_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(wr + k + 4)));
        __m128i wq2 = _mm_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(wr + k + 8)));
        __m128i wq3 = _mm_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(wr + k + 12)));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(ar + k), _mm_cvtepi32_ps(wq0)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(ar + k + 4), _mm_cvtepi32_ps(wq1)));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(ar + k + 8), _mm_cvtepi32_ps(wq2)));
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(ar + k + 12), _mm_cvtepi32_ps(wq3)));
      }
      __m128 s4 = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
      float part[4];
      _mm_storeu_ps(part, s4);
      float t = (part[0] + part[1]) + (part[2] + part[3]);
      for (; k < K; k++) t += ar[k] * (float)wr[k];
      s = t;
#else
      float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      int k = 0;
      for (; k + 4 <= K; k += 4) {
        s0 += ar[k] * (float)wr[k];
        s1 += ar[k + 1] * (float)wr[k + 1];
        s2 += ar[k + 2] * (float)wr[k + 2];
        s3 += ar[k + 3] * (float)wr[k + 3];
      }
      float t = (s0 + s1) + (s2 + s3);
      for (; k < K; k++) t += ar[k] * (float)wr[k];
      s = t;
#endif
      cr[n] = s * sw[n] + (bias ? bias[n] : 0.0f);
    }
  }
}
