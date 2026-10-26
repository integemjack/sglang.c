/* sgl cuda_backend.cu - llama-family GPU forward (M9 v1)
 * kernels: embed / rmsnorm / gemm_nt (block-reduce dot) / rope (precomputed
 * freqs) / attention (block per (t,h), shared scores) / silu_mul / residual.
 * Numerics match llama.c (fp32; reduction order may differ ~1e-5). */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/cuda.h"
#include "sgl/llama.h"

#define CK(x)                                                            \
  do {                                                                   \
    cudaError_t e = (x);                                                 \
    if (e != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), \
              __FILE__, __LINE__);                                       \
      return -1;                                                         \
    }                                                                    \
  } while (0)

/* M26a: attention path split. upto = positions attended by the last row.
 * <= SHORT_MAX keeps the original shared-scores kernel (numerics unchanged);
 * beyond that the two-pass chunked kernel removes the shared-memory cap. */
#define SGL_ATTN_SHORT_MAX 4096
#define SGL_ATTN_CHUNK 1024

struct sgl_cuda_model {
  int L, H, HK, E, I, V, D, KD, ctx;
  float eps;
  double theta;

  float *wte;        /* [V,E] */
  float *final_norm; /* [E] */
  float *lm_head;    /* [V,E] (aliases wte when tied embeddings) */
  float **ln1, **qw, **kw, **vw, **ow;
  float **qb, **kb, **vb;
  float **ln2, **gw, **uw, **dw;
  float *freqs; /* [D/2] */
  cublasHandle_t blas; /* M22: all projections via cublas */
  float *kc, *vc; /* [ctx][KD] */
  float *x, *h, *q, *k, *v, *attn, *gate, *up;
  int ws_cap; /* rows currently covered by the growable workspace buffers */

  cudaStream_t stream;      /* all launches go here (0 = legacy default) */
  void *blas_ws;            /* cublas workspace (pre-allocated, graph-safe) */
  size_t blas_ws_size;

  /* t1 fp16 path (SGL_CUDA_FP16=1): weights stored as __half, activations /
   * KV / attention / rope / softmax stay fp32. Only GEMMs consume half:
   * M==1 decode GEMV is a hand-written kernel (half2 loads, fp32 accum),
   * M>1 goes through tensor-core cublasGemmEx (16F in, 32F accum+out). */
  int use_fp16;
  __half *wte_h;            /* [V,E] embed table; lm_head aliases when tied */
  __half *lm_head_h;        /* [V,E] (aliases wte_h when tied embeddings) */
  __half **qw_h, **kw_h, **vw_h, **ow_h;
  __half **gw_h, **uw_h, **dw_h;
  __half *ha;               /* fp16 mirror of GEMM inputs (M>1 cublas path) */

  /* persistent I/O buffers for forward_seg / decode_batch (M26b: no
   * per-step cudaMalloc; pinned host side so a captured graph re-reads
   * the latest inputs on every replay) */
  int io_cap_B;             /* rows covered by the buffers below */
  int *d_tok, *d_base, *d_lens, *d_pq, *d_pk;
  int *h_tok, *h_base, *h_lens, *h_pq, *h_pk; /* pinned host mirrors */
  float *d_lgb;             /* logits on device [B*V] */
  float *h_lgb;             /* logits pinned host mirror */
  int lgb_cap;              /* rows covered by d_lgb/h_lgb */

  /* CUDA Graph cache, one captured decode step per batch size (M26b) */
  void *graphs[33];         /* cudaGraph_t per B (index = B) */
  void *gexec[33];          /* cudaGraphExec_t per B */
};

__global__ void k_embed(const float *wte, const int *tok, float *x, int n, int E) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * E) return;
  x[idx] = wte[(long)tok[idx / E] * E + idx % E];
}

__global__ void k_rmsnorm(const float *x, const float *w, float *y, int M, int N,
                          float eps) {
  extern __shared__ float red[];
  int m = blockIdx.x;
  const float *xr = x + (long)m * N;
  float *yr = y + (long)m * N;
  float ss = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) ss += xr[i] * xr[i];
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / N + eps);
  for (int i = threadIdx.x; i < N; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}

__global__ void k_gemm_nt(const float *A, const float *W, const float *bias,
                          float *C, int M, int N, int K) {
  extern __shared__ float red[];
  long mn = (long)blockIdx.x;
  int m = mn / N, n = mn % N;
  const float *ar = A + (long)m * K;
  const float *wr = W + (long)n * K;
  float s = 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x) s += ar[k] * wr[k];
  red[threadIdx.x] = s;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) C[mn] = red[0] + (bias ? bias[n] : 0.0f);
}

/* t1 fp16 GEMV (decode, M==1): C[n] = A[K] . W[n,K] + bias.
 * W rows are __half; vectorized __half2 loads converted to fp32, multiply-
 * accumulate entirely in fp32 (activation A is fp32 and never rounded to
 * half; only the weight storage is fp16). Block per output element, same
 * layout as k_gemm_nt. */
__global__ void k_gemm_nt_h(const float *A, const __half *W, const float *bias,
                            float *C, int M, int N, int K) {
  extern __shared__ float red[];
  long mn = (long)blockIdx.x;
  int m = mn / N, n = mn % N;
  const float *ar = A + (long)m * K;
  const __half *wr = W + (long)n * K;
  float s = 0;
  if ((K & 1) == 0) {
    const __half2 *w2 = (const __half2 *)wr;
    for (int k = threadIdx.x; k < K / 2; k += blockDim.x) {
      float2 wf = __half22float2(w2[k]);
      s += ar[2 * k] * wf.x + ar[2 * k + 1] * wf.y;
    }
  } else {
    for (int k = threadIdx.x; k < K; k += blockDim.x)
      s += ar[k] * __half2float(wr[k]);
  }
  red[threadIdx.x] = s;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) C[mn] = red[0] + (bias ? bias[n] : 0.0f);
}

/* t1 fp16 embed lookup: x = fp32(wte_h[token]) */
__global__ void k_embed_h(const __half *wte, const int *tok, float *x, int n, int E) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * E) return;
  x[idx] = __half2float(wte[(long)tok[idx / E] * E + idx % E]);
}

/* fp32 -> fp16 elementwise conversion (GEMM input mirror for cublasGemmEx) */
__global__ void k_f2h(const float *src, __half *dst, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = __float2half(src[i]);
}


__global__ void k_rope(float *qk, int n, int head_dim, int heads_per_tok,
                       int pos0, const float *freqs) {
  int half = head_dim / 2;
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= (long)n * half) return;
  long tok = idx / ((long)half * heads_per_tok); /* 閻炴稑鑻ぐ鍧楀及椤栨哎浠堢紒渚垮灩缁扁晠鏁嶇仦鑲╃Т缂傚喚鍠楃€?token 闁告帒妫涚划?*/
  int i = idx % half;
  float *head = qk + (long)idx / half * head_dim;
  float ang = (pos0 + tok) * freqs[i];
  float c, s;
  sincosf(ang, &s, &c);
  float a = head[i], b = head[i + half];
  head[i] = a * c - b * s;
  head[i + half] = a * s + b * c;
}

/* per-row rope for batch decode: row r (of n rows) rotated at pos_row[r] */
__global__ void k_rope_rows(float *qk, int n, int head_dim, const int *pos_row,
                            const float *freqs) {
  int half = head_dim / 2;
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= (long)n * half) return;
  int r = (int)(idx / half);
  int i = (int)(idx % half);
  float *head = qk + (long)r * head_dim;
  float ang = pos_row[r] * freqs[i];
  float c, s;
  sincosf(ang, &s, &c);
  float a = head[i], b = head[i + half];
  head[i] = a * c - b * s;
  head[i + half] = a * s + b * c;
}

__global__ void k_attn(const float *q, const float *kc, const float *vc,
                       float *out, int pos0, int kv_base, int n, int H, int HK, int D,
                       float scale) {
  extern __shared__ float sc[];
  int th = blockIdx.x;
  int t = th / H, h = th % H;
  int upto = pos0 + t + 1;
  const float *qh = q + ((long)t * H + h) * D;
  int hkv = h / (H / HK);
  float maxv = -1e30f;
  for (int p = threadIdx.x; p < upto; p += blockDim.x) {
    const float *kh = kc + ((long)(kv_base + p) * HK + hkv) * D;
    float s = 0;
    for (int d = 0; d < D; d++) s += qh[d] * kh[d];
    s *= scale;
    sc[p] = s;
    if (s > maxv) maxv = s;
  }
  __syncthreads();
  __shared__ float shmax[256];
  shmax[threadIdx.x] = maxv;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st && shmax[threadIdx.x + st] > shmax[threadIdx.x])
      shmax[threadIdx.x] = shmax[threadIdx.x + st];
    __syncthreads();
  }
  float invmax = shmax[0];
  float ssum = 0;
  for (int p = threadIdx.x; p < upto; p += blockDim.x) {
    sc[p] = __expf(sc[p] - invmax);
    ssum += sc[p];
  }
  __shared__ float shsum[256];
  shsum[threadIdx.x] = ssum;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) shsum[threadIdx.x] += shsum[threadIdx.x + st];
    __syncthreads();
  }
  float invsum = 1.0f / shsum[0];
  float *o = out + ((long)t * H + h) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] = 0;
  __syncthreads();
  for (int p = 0; p < upto; p++) {
    const float *vh = vc + ((long)(kv_base + p) * HK + hkv) * D;
    float w = sc[p] * invsum;
    for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] += w * vh[d];
  }
}

/* two-pass attention for long contexts (M26a): removes the shared-memory
 * scores cap (SGL_ATTN_SHORT_MAX) by never holding more than SGL_ATTN_CHUNK
 * scores in shared memory.
 * pass 1: stream over KV in chunks; per chunk compute scores, block-reduce
 *         the chunk max, then fold it into a running (max, sum) pair with
 *         the online-softmax rescale  sum = sum*exp(m_old-m_new) + cs.
 * pass 2: recompute scores against the final max, exp(s-m) weights, and
 *         accumulate weighted V; scale by 1/sum at the end.
 * The KV pass runs twice (K dot-products computed twice) in exchange for a
 * fixed shared-memory footprint independent of context length. */
__global__ void k_attn_long(const float *q, const float *kc, const float *vc,
                            float *out, int pos0, int kv_base, int n, int H, int HK, int D,
                            float scale) {
  __shared__ float sc[SGL_ATTN_CHUNK];
  __shared__ float red[256];
  int th = blockIdx.x;
  int t = th / H, h = th % H;
  int upto = pos0 + t + 1;
  const float *qh = q + ((long)t * H + h) * D;
  int hkv = h / (H / HK);

  /* pass 1: global max + online sum of exp(score - running max) */
  float run_max = -1e30f, run_sum = 0.0f;
  for (int c0 = 0; c0 < upto; c0 += SGL_ATTN_CHUNK) {
    int ce = c0 + SGL_ATTN_CHUNK; if (ce > upto) ce = upto;
    float cmax = -1e30f;
    for (int p = c0 + threadIdx.x; p < ce; p += blockDim.x) {
      const float *kh = kc + ((long)(kv_base + p) * HK + hkv) * D;
      float s = 0;
      for (int d = 0; d < D; d++) s += qh[d] * kh[d];
      s *= scale;
      sc[p - c0] = s;
      if (s > cmax) cmax = s;
    }
    red[threadIdx.x] = cmax;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
      if (threadIdx.x < st)
        red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
      __syncthreads();
    }
    cmax = red[0];
    float alpha = __expf(run_max - cmax);
    float cs = 0;
    for (int p = threadIdx.x; p < ce - c0; p += blockDim.x) {
      float e = __expf(sc[p] - cmax);
      sc[p] = e;
      cs += e;
    }
    __syncthreads(); /* everyone has read red[0] above; now safe to reuse red */
    red[threadIdx.x] = cs;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
      if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
      __syncthreads();
    }
    run_sum = run_sum * alpha + red[0];
    run_max = cmax;
    __syncthreads(); /* everyone has read red[0]; next chunk rewrites red */
  }
  float invsum = 1.0f / run_sum;

  /* pass 2: recompute scores with the final max, accumulate weighted V */
  float *o = out + ((long)t * H + h) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] = 0;
  __syncthreads();
  for (int c0 = 0; c0 < upto; c0 += SGL_ATTN_CHUNK) {
    int ce = c0 + SGL_ATTN_CHUNK; if (ce > upto) ce = upto;
    for (int p = c0 + threadIdx.x; p < ce; p += blockDim.x) {
      const float *kh = kc + ((long)(kv_base + p) * HK + hkv) * D;
      float s = 0;
      for (int d = 0; d < D; d++) s += qh[d] * kh[d];
      sc[p - c0] = __expf(s * scale - run_max);
    }
    __syncthreads();
    for (int p = c0; p < ce; p++) {
      const float *vh = vc + ((long)(kv_base + p) * HK + hkv) * D;
      float w = sc[p - c0];
      for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] += w * vh[d];
    }
    __syncthreads();
  }
  for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] *= invsum;
}

/* batch variant: block = (seq, head), KV at absolute (base[b] + p),
 * attends lens[b] + 1 positions (M26a long-context path). */
__global__ void k_attn_long_batch(const float *q, const float *kc, const float *vc,
                                  float *out, const int *base, const int *lens,
                                  int B, int H, int HK, int D, float scale) {
  __shared__ float sc[SGL_ATTN_CHUNK];
  __shared__ float red[256];
  int th = blockIdx.x;
  int b = th / H, h = th % H;
  int base_p = base[b];
  int upto = lens[b] + 1;
  const float *qh = q + ((long)b * H + h) * D;
  int hkv = h / (H / HK);

  float run_max = -1e30f, run_sum = 0.0f;
  for (int c0 = 0; c0 < upto; c0 += SGL_ATTN_CHUNK) {
    int ce = c0 + SGL_ATTN_CHUNK; if (ce > upto) ce = upto;
    float cmax = -1e30f;
    for (int p = c0 + threadIdx.x; p < ce; p += blockDim.x) {
      const float *kh = kc + ((long)(base_p + p) * HK + hkv) * D;
      float s = 0;
      for (int d = 0; d < D; d++) s += qh[d] * kh[d];
      s *= scale;
      sc[p - c0] = s;
      if (s > cmax) cmax = s;
    }
    red[threadIdx.x] = cmax;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
      if (threadIdx.x < st)
        red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
      __syncthreads();
    }
    cmax = red[0];
    float alpha = __expf(run_max - cmax);
    float cs = 0;
    for (int p = threadIdx.x; p < ce - c0; p += blockDim.x) {
      float e = __expf(sc[p] - cmax);
      sc[p] = e;
      cs += e;
    }
    __syncthreads(); /* everyone has read red[0] above; now safe to reuse red */
    red[threadIdx.x] = cs;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
      if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
      __syncthreads();
    }
    run_sum = run_sum * alpha + red[0];
    run_max = cmax;
    __syncthreads(); /* everyone has read red[0]; next chunk rewrites red */
  }
  float invsum = 1.0f / run_sum;

  float *o = out + ((long)b * H + h) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] = 0;
  __syncthreads();
  for (int c0 = 0; c0 < upto; c0 += SGL_ATTN_CHUNK) {
    int ce = c0 + SGL_ATTN_CHUNK; if (ce > upto) ce = upto;
    for (int p = c0 + threadIdx.x; p < ce; p += blockDim.x) {
      const float *kh = kc + ((long)(base_p + p) * HK + hkv) * D;
      float s = 0;
      for (int d = 0; d < D; d++) s += qh[d] * kh[d];
      sc[p - c0] = __expf(s * scale - run_max);
    }
    __syncthreads();
    for (int p = c0; p < ce; p++) {
      const float *vh = vc + ((long)(base_p + p) * HK + hkv) * D;
      float w = sc[p - c0];
      for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] += w * vh[d];
    }
    __syncthreads();
  }
  for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] *= invsum;
}

__global__ void k_silu_mul(float *gate, const float *up, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = gate[i];
  gate[i] = g / (1 + __expf(-g)) * up[i];
}

__global__ void k_residual(float *x, const float *h, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] += h[i];
}

/* row-major C[M,N] = A[M,K] @ W[N,K]^T (+bias) via cublas.
 * mapping: row-major [N,K] with ld=K reads as col-major KxN (OP_T gives NxK),
 * row-major [M,K] ld=K is col-major KxM (OP_N); result col-major NxM with
 * ld=N equals row-major MxN. */
__global__ void k_fill_bias(float *C, const float *bias, int M, int N) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)M * N) return;
  C[i] = bias[i % N];
}

static void gemm_nt_cublas(cublasHandle_t h, cudaStream_t stream,
                           const float *A, const float *W,
                           const float *bias, float *C, int M, int N, int K) {
  /* M23 adaptive dispatch: M=1 GEMV runs faster on the hand-written kernel
   * (cublas launch overhead dominates); M>1 matrices go to cublas.
   * SGL_FORCE_CUBLAS=1 disables the hand-written path (debug/bisect). */
  static int force_blas = -1;
  if (force_blas < 0) {
    const char *e = getenv("SGL_FORCE_CUBLAS");
    force_blas = (e && atoi(e)) ? 1 : 0;
  }
  if (M == 1 && !force_blas) {
    k_gemm_nt<<<N, 256, 256 * sizeof(float), stream>>>(A, W, bias, C, 1, N, K);
    return;
  }
  if (bias)
    k_fill_bias<<<(int)(((long)M * N + 255) / 256), 256, 0, stream>>>((float *)C, bias, M, N);
  const float alpha = 1.0f, beta = bias ? 1.0f : 0.0f;
  cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
              W, K, A, K, &beta, C, N);
}

/* t1 fp16-aware dispatch: same row-major C[M,N] = A[M,K] @ W[N,K]^T (+bias)
 * contract as gemm_nt_cublas, but W arrives as __half when the model was
 * uploaded in fp16. M==1 (decode GEMV) uses the hand kernel (fp32 accum);
 * M>1 converts A to half on the fly and runs tensor-core cublasGemmEx
 * (16F x 16F -> 32F accum, 32F C). Bias is epilogue-added in fp32 either way.
 * Dispatch threshold (M==1 vs cublas) re-measured in sgl_cuda_gemm_bench:
 * at M=1 the hand kernel wins on launch overhead, M>=8 GemmEx wins. */
static void gemm_nt_any(sgl_cuda_model *g, const float *A, const float *W_f,
                        const __half *W_h, const float *bias, float *C,
                        int M, int N, int K) {
  cudaStream_t st = g->stream;
  if (!g->use_fp16) {
    gemm_nt_cublas(g->blas, st, A, W_f, bias, C, M, N, K);
    return;
  }
  if (M == 1) {
    k_gemm_nt_h<<<N, 256, 256 * sizeof(float), st>>>(A, W_h, bias, C, 1, N, K);
    return;
  }
  if ((long)M * K > (long)g->ws_cap * (g->E > g->I ? g->E : g->I)) {
    fprintf(stderr, "gemm_nt_any: ha workspace too small (%d x %d)\n", M, K);
    return;
  }
  k_f2h<<<(int)(((long)M * K + 255) / 256), 256, 0, st>>>(A, g->ha, (long)M * K);
  if (bias)
    k_fill_bias<<<(int)(((long)M * N + 255) / 256), 256, 0, st>>>(C, bias, M, N);
  const float alpha = 1.0f, beta = bias ? 1.0f : 0.0f;
  cublasGemmEx(g->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
               W_h, CUDA_R_16F, K, g->ha, CUDA_R_16F, K, &beta,
               C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}


static float *dev_copy(const float *src, size_t n) {
  float *d = NULL;
  cudaError_t e = cudaMalloc(&d, n * sizeof(float));
  if (e != cudaSuccess) {
    printf("[ZZ alloc-fail] %zu floats: %s\n", n, cudaGetErrorString(e));
    return NULL;
  }
  if (src) {
    e = cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
      printf("[ZZ copy-fail] %zu floats: %s\n", n, cudaGetErrorString(e));
      cudaFree(d);
      return NULL;
    }
  }
  return d;
}

/* t1: fp32 host weights -> fp16 device upload (convert on host, one memcpy) */
static __half *dev_copy_h(const float *src, size_t n) {
  __half *d = NULL;
  cudaError_t e = cudaMalloc(&d, n * sizeof(__half));
  if (e != cudaSuccess) {
    printf("[ZZ alloc-fail-h] %zu halves: %s\n", n, cudaGetErrorString(e));
    return NULL;
  }
  if (src) {
    __half *tmp = (__half *)malloc(n * sizeof(__half));
    if (!tmp) {
      cudaFree(d);
      return NULL;
    }
    for (size_t i = 0; i < n; i++) tmp[i] = __float2half(src[i]);
    e = cudaMemcpy(d, tmp, n * sizeof(__half), cudaMemcpyHostToDevice);
    free(tmp);
    if (e != cudaSuccess) {
      printf("[ZZ copy-fail-h] %zu halves: %s\n", n, cudaGetErrorString(e));
      cudaFree(d);
      return NULL;
    }
  }
  return d;
}

/* growable activation workspace: make sure x/h/q/attn [E rows], k/v [KD
 * rows], gate/up [I rows] each cover n rows. Grows geometrically so decode
 * (n=B) allocates once and prefill grows to the longest segment seen.
 * t1: also the fp16 GEMM-input mirror ha [ws_cap x max(E,I)] (cublas path). */
static int sgl_cuda_reserve(sgl_cuda_model *g, int n) {
  if (n <= g->ws_cap) return 0;
  float *x = dev_copy(NULL, (size_t)n * g->E);
  float *h = dev_copy(NULL, (size_t)n * g->E);
  float *q = dev_copy(NULL, (size_t)n * g->E);
  float *a = dev_copy(NULL, (size_t)n * g->E);
  float *k = dev_copy(NULL, (size_t)n * g->KD);
  float *v = dev_copy(NULL, (size_t)n * g->KD);
  float *ga = dev_copy(NULL, (size_t)n * g->I);
  float *u = dev_copy(NULL, (size_t)n * g->I);
  if (!x || !h || !q || !a || !k || !v || !ga || !u) {
    cudaFree(x); cudaFree(h); cudaFree(q); cudaFree(a);
    cudaFree(k); cudaFree(v); cudaFree(ga); cudaFree(u);
    fprintf(stderr, "sgl_cuda_reserve(%d): out of VRAM\n", n);
    return -1;
  }
  cudaFree(g->x); cudaFree(g->h); cudaFree(g->q); cudaFree(g->attn);
  cudaFree(g->k); cudaFree(g->v); cudaFree(g->gate); cudaFree(g->up);
  g->x = x; g->h = h; g->q = q; g->attn = a;
  g->k = k; g->v = v; g->gate = ga; g->up = u;
  if (g->use_fp16) {
    __half *hn = NULL;
    if (cudaMalloc(&hn, (size_t)n * (g->E > g->I ? g->E : g->I) * sizeof(__half))
        != cudaSuccess) {
      cudaFree(hn);
      fprintf(stderr, "sgl_cuda_reserve(%d): out of VRAM (ha)\n", n);
      return -1;
    }
    cudaFree(g->ha);
    g->ha = hn;
  }
  g->ws_cap = n;
  return 0;
}

/* persistent per-step I/O buffers: device arrays + pinned host mirrors.
 * The pinned mirrors are what a captured graph re-reads on every replay
 * (memcpy nodes capture the host address, not the contents). */
static int sgl_cuda_ensure_io(sgl_cuda_model *g, int B) {
  if (B <= g->io_cap_B) return 0;
  cudaFree(g->d_tok); cudaFree(g->d_base); cudaFree(g->d_lens);
  cudaFree(g->d_pq); cudaFree(g->d_pk);
  cudaFreeHost(g->h_tok); cudaFreeHost(g->h_base); cudaFreeHost(g->h_lens);
  cudaFreeHost(g->h_pq); cudaFreeHost(g->h_pk);
  g->d_tok = g->d_base = g->d_lens = g->d_pq = g->d_pk = NULL;
  g->h_tok = g->h_base = g->h_lens = g->h_pq = g->h_pk = NULL;
  CK(cudaMalloc(&g->d_tok, sizeof(int) * (size_t)B));
  CK(cudaMalloc(&g->d_base, sizeof(int) * (size_t)B));
  CK(cudaMalloc(&g->d_lens, sizeof(int) * (size_t)B));
  CK(cudaMalloc(&g->d_pq, sizeof(int) * (size_t)B * g->H));
  CK(cudaMalloc(&g->d_pk, sizeof(int) * (size_t)B * g->HK));
  CK(cudaMallocHost(&g->h_tok, sizeof(int) * (size_t)B));
  CK(cudaMallocHost(&g->h_base, sizeof(int) * (size_t)B));
  CK(cudaMallocHost(&g->h_lens, sizeof(int) * (size_t)B));
  CK(cudaMallocHost(&g->h_pq, sizeof(int) * (size_t)B * g->H));
  CK(cudaMallocHost(&g->h_pk, sizeof(int) * (size_t)B * g->HK));
  g->io_cap_B = B;
  return 0;
}

/* logits buffers (device + pinned host), grown to `rows` rows of V floats */
static int sgl_cuda_ensure_lgb(sgl_cuda_model *g, int rows) {
  if (rows <= g->lgb_cap) return 0;
  cudaFree(g->d_lgb); cudaFreeHost(g->h_lgb);
  CK(cudaMalloc(&g->d_lgb, sizeof(float) * (size_t)rows * g->V));
  CK(cudaMallocHost(&g->h_lgb, sizeof(float) * (size_t)rows * g->V));
  g->lgb_cap = rows;
  return 0;
}

/* batch attention: block = (seq, head); kv at absolute (base+p) */
__global__ void k_attn_batch(const float *q, const float *kc, const float *vc,
                             float *out, const int *base, const int *lens,
                             int B, int H, int HK, int D, float scale) {
  extern __shared__ float sc[];
  int th = blockIdx.x;
  int b = th / H, h = th % H;
  int base_p = base[b];
  int upto = lens[b] + 1;
  const float *qh = q + ((long)b * H + h) * D;
  int hkv = h / (H / HK);
  float maxv = -1e30f;
  for (int p = threadIdx.x; p < upto; p += blockDim.x) {
    const float *kh = kc + ((long)(base_p + p) * HK + hkv) * D;
    float s = 0;
    for (int d = 0; d < D; d++) s += qh[d] * kh[d];
    s *= scale;
    sc[p] = s;
    if (s > maxv) maxv = s;
  }
  __syncthreads();
  __shared__ float shmax[256];
  shmax[threadIdx.x] = maxv;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st && shmax[threadIdx.x + st] > shmax[threadIdx.x])
      shmax[threadIdx.x] = shmax[threadIdx.x + st];
    __syncthreads();
  }
  float invmax = shmax[0];
  float ssum = 0;
  for (int p = threadIdx.x; p < upto; p += blockDim.x) {
    sc[p] = __expf(sc[p] - invmax);
    ssum += sc[p];
  }
  __shared__ float shsum[256];
  shsum[threadIdx.x] = ssum;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) shsum[threadIdx.x] += shsum[threadIdx.x + st];
    __syncthreads();
  }
  float invsum = 1.0f / shsum[0];
  float *o = out + ((long)b * H + h) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] = 0;
  __syncthreads();
  for (int p = 0; p < upto; p++) {
    const float *vh = vc + ((long)(base_p + p) * HK + hkv) * D;
    float w = sc[p] * invsum;
    for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] += w * vh[d];
  }
}

sgl_cuda_model *sgl_cuda_from_llama(const void *llama_model) {
  const sgl_llama *m = (const sgl_llama *)llama_model;
  if (!m) return NULL;
  sgl_cuda_model *g = (sgl_cuda_model *)calloc(1, sizeof(*g));
  size_t ctx = 0; /* declared up front: goto fail below must not bypass it */
  g->L = m->n_layer; g->H = m->n_head; g->HK = m->n_kv_head;
  g->E = m->n_embd; g->I = m->n_inter; g->V = m->vocab;
  g->D = m->head_dim; g->KD = m->kv_dim; g->ctx = m->n_ctx;
  g->eps = m->rms_eps; g->theta = m->rope_theta;
  /* t1: fp16 weight storage, opt-in via SGL_CUDA_FP16 (default off = the
   * original all-fp32 path, bit-identical behavior). */
  {
    const char *fe = getenv("SGL_CUDA_FP16");
    g->use_fp16 = (fe && atoi(fe)) ? 1 : 0;
  }

  /* M26a: no hard 4096 clamp anymore. Two-pass attention lifts the shared
   * memory limit; the remaining cap is VRAM (KV pool + weights scale with
   * ctx). Default 16384 fits Qwen2.5-0.5B fp32 into a 4GB card (RTX 2060)
   * thanks to tied-lm_head reuse; SGL_CUDA_CTX overrides. */
  {
    const char *e = getenv("SGL_CUDA_CTX");
    int cap = (e && atoi(e) > 0) ? atoi(e) : 16384;
    if (g->ctx > cap) g->ctx = cap;
  }

  if (g->use_fp16) {
    /* t1 fp16 upload: weights as __half, fp32 activations everywhere else */
    g->wte_h = dev_copy_h(m->wte.data, (size_t)m->vocab * m->n_embd);
    g->final_norm = dev_copy(m->final_norm.data, (size_t)m->n_embd);
    /* tied embeddings: the half lm_head aliases the half embed table */
    if (m->lm_head.data == m->wte.data) {
      g->lm_head_h = g->wte_h;
    } else {
      g->lm_head_h = dev_copy_h(m->lm_head.data, (size_t)m->vocab * m->n_embd);
    }
    if (!g->wte_h || !g->final_norm || !g->lm_head_h) goto fail;
  } else {
    g->wte = dev_copy(m->wte.data, (size_t)m->vocab * m->n_embd);
    g->final_norm = dev_copy(m->final_norm.data, (size_t)m->n_embd);
    /* tied embeddings: lm_head aliases wte on the device too (saves V*E*4B,
     * e.g. 545MB for Qwen2.5-0.5B) */
    if (m->lm_head.data == m->wte.data) {
      g->lm_head = g->wte;
    } else {
      g->lm_head = dev_copy(m->lm_head.data, (size_t)m->vocab * m->n_embd);
    }
    if (!g->wte || !g->final_norm || !g->lm_head) goto fail;
  }

  g->ln1 = (float **)calloc(g->L, sizeof(float *));
  g->qw = (float **)calloc(g->L, sizeof(float *));
  g->kw = (float **)calloc(g->L, sizeof(float *));
  g->vw = (float **)calloc(g->L, sizeof(float *));
  g->ow = (float **)calloc(g->L, sizeof(float *));
  g->qb = (float **)calloc(g->L, sizeof(float *));
  g->kb = (float **)calloc(g->L, sizeof(float *));
  g->vb = (float **)calloc(g->L, sizeof(float *));
  g->ln2 = (float **)calloc(g->L, sizeof(float *));
  g->gw = (float **)calloc(g->L, sizeof(float *));
  g->uw = (float **)calloc(g->L, sizeof(float *));
  g->dw = (float **)calloc(g->L, sizeof(float *));
  if (g->use_fp16) {
    g->qw_h = (__half **)calloc(g->L, sizeof(__half *));
    g->kw_h = (__half **)calloc(g->L, sizeof(__half *));
    g->vw_h = (__half **)calloc(g->L, sizeof(__half *));
    g->ow_h = (__half **)calloc(g->L, sizeof(__half *));
    g->gw_h = (__half **)calloc(g->L, sizeof(__half *));
    g->uw_h = (__half **)calloc(g->L, sizeof(__half *));
    g->dw_h = (__half **)calloc(g->L, sizeof(__half *));
  }

  for (int l = 0; l < g->L; l++) {
    const sgl_llama_layer *L = &m->layers[l];
    if (g->use_fp16) {
      g->qw_h[l] = dev_copy_h(L->q_w.data, (size_t)g->E * g->E);
      g->kw_h[l] = dev_copy_h(L->k_w.data, (size_t)g->KD * g->E);
      g->vw_h[l] = dev_copy_h(L->v_w.data, (size_t)g->KD * g->E);
      g->ow_h[l] = dev_copy_h(L->o_w.data, (size_t)g->E * g->E);
      g->gw_h[l] = dev_copy_h(L->gate_w.data, (size_t)g->I * g->E);
      g->uw_h[l] = dev_copy_h(L->up_w.data, (size_t)g->I * g->E);
      g->dw_h[l] = dev_copy_h(L->down_w.data, (size_t)g->E * g->I);
      if (!g->qw_h[l] || !g->gw_h[l] || !g->dw_h[l]) goto fail;
      /* norms and biases stay fp32 (attention path is fp32 throughout) */
      g->ln1[l] = dev_copy(L->inp_norm.data, (size_t)g->E);
      g->ln2[l] = dev_copy(L->post_norm.data, (size_t)g->E);
      g->qb[l] = L->q_b.data ? dev_copy(L->q_b.data, (size_t)g->E) : NULL;
      g->kb[l] = L->k_b.data ? dev_copy(L->k_b.data, (size_t)g->KD) : NULL;
      g->vb[l] = L->v_b.data ? dev_copy(L->v_b.data, (size_t)g->KD) : NULL;
      continue;
    }
    g->ln1[l] = dev_copy(L->inp_norm.data, (size_t)g->E);
    g->qw[l] = dev_copy(L->q_w.data, (size_t)g->E * g->E);
    g->kw[l] = dev_copy(L->k_w.data, (size_t)g->KD * g->E);
    g->vw[l] = dev_copy(L->v_w.data, (size_t)g->KD * g->E);
    g->ow[l] = dev_copy(L->o_w.data, (size_t)g->E * g->E);
    g->qb[l] = L->q_b.data ? dev_copy(L->q_b.data, (size_t)g->E) : NULL;
    g->kb[l] = L->k_b.data ? dev_copy(L->k_b.data, (size_t)g->KD) : NULL;
    g->vb[l] = L->v_b.data ? dev_copy(L->v_b.data, (size_t)g->KD) : NULL;
    g->ln2[l] = dev_copy(L->post_norm.data, (size_t)g->E);
    g->gw[l] = dev_copy(L->gate_w.data, (size_t)g->I * g->E);
    g->uw[l] = dev_copy(L->up_w.data, (size_t)g->I * g->E);
    g->dw[l] = dev_copy(L->down_w.data, (size_t)g->E * g->I);
    if (!g->ln1[l] || !g->qw[l] || !g->gw[l] || !g->dw[l]) goto fail;
  }

  {
    float *f = (float *)malloc(sizeof(float) * (size_t)(g->D / 2));
    for (int i = 0; i < g->D / 2; i++)
      f[i] = (float)pow(g->theta, -(double)(2 * i) / g->D);
    g->freqs = dev_copy(f, (size_t)(g->D / 2));
    free(f);
  }
  if (cublasCreate(&g->blas) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasCreate failed\n");
    goto fail;
  }
  /* M26b: every launch (kernels + cublas) goes through g->stream so a
   * captured decode graph replays into the same ordering domain. */
  if (cudaStreamCreate(&g->stream) != cudaSuccess) {
    fprintf(stderr, "cudaStreamCreate failed\n");
    goto fail;
  }
  cublasSetStream(g->blas, g->stream);
  /* pre-allocated workspace: required for stream capture (cuBLAS must not
   * allocate internally while a graph is being recorded) */
  g->blas_ws_size = (size_t)32 << 20;
  if (cudaMalloc(&g->blas_ws, g->blas_ws_size) != cudaSuccess) {
    g->blas_ws_size = 0;
  } else {
    cublasSetWorkspace(g->blas, g->blas_ws, g->blas_ws_size);
  }

  ctx = (size_t)g->ctx;
  /* KV 閻㈩垰鍟惇?[L][ctx][KD]闁挎稒鑹鹃崹搴ㄦ煀瀹ュ懐绠戝銈堫唺缁犺崵浠﹂崒娑欐 */
  g->kc = dev_copy(NULL, (size_t)g->L * ctx * g->KD);
  g->vc = dev_copy(NULL, (size_t)g->L * ctx * g->KD);
  /* activation workspace is growable now (sgl_cuda_reserve): only the KV
   * pool scales with ctx, long prompts don't need ctx-sized activations */
  if (!g->kc || !g->vc) goto fail;
  if (g->use_fp16) printf("[cuda] fp16 weights ON (SGL_CUDA_FP16=1)\n");
  return g;

fail:
  fprintf(stderr, "sgl_cuda_from_llama: out of VRAM or copy failed\n");
  sgl_cuda_free(g);
  return NULL;
}

void sgl_cuda_free(sgl_cuda_model *g) {
  if (!g) return;
  if (g->blas) cublasDestroy(g->blas);
  if (g->blas_ws) cudaFree(g->blas_ws);
  if (g->stream) cudaStreamDestroy(g->stream);
  for (int b = 0; b <= 32; b++) {
    if (g->graphs[b]) cudaGraphDestroy((cudaGraph_t)g->graphs[b]);
    if (g->gexec[b]) cudaGraphExecDestroy((cudaGraphExec_t)g->gexec[b]);
  }
  cudaFree(g->d_tok); cudaFree(g->d_base); cudaFree(g->d_lens);
  cudaFree(g->d_pq); cudaFree(g->d_pk);
  if (g->h_tok) cudaFreeHost(g->h_tok);
  if (g->h_base) cudaFreeHost(g->h_base);
  if (g->h_lens) cudaFreeHost(g->h_lens);
  if (g->h_pq) cudaFreeHost(g->h_pq);
  if (g->h_pk) cudaFreeHost(g->h_pk);
  if (g->d_lgb) cudaFree(g->d_lgb);
  if (g->h_lgb) cudaFreeHost(g->h_lgb);
  cudaFree(g->wte); cudaFree(g->final_norm);
  /* tied embeddings: lm_head aliases wte, free once */
  if (g->lm_head && g->lm_head != g->wte) cudaFree(g->lm_head);
  /* t1 fp16 mirrors (lm_head_h aliases wte_h when tied) */
  if (g->wte_h) cudaFree(g->wte_h);
  if (g->lm_head_h && g->lm_head_h != g->wte_h) cudaFree(g->lm_head_h);
  cudaFree(g->ha);
  cudaFree(g->freqs); cudaFree(g->kc); cudaFree(g->vc);
  cudaFree(g->x); cudaFree(g->h); cudaFree(g->q); cudaFree(g->k); cudaFree(g->v);
  cudaFree(g->attn); cudaFree(g->gate); cudaFree(g->up);
  for (int l = 0; l < g->L; l++) {
    if (g->ln1 && g->ln1[l]) cudaFree(g->ln1[l]);
    if (g->qw && g->qw[l]) cudaFree(g->qw[l]);
    if (g->kw && g->kw[l]) cudaFree(g->kw[l]);
    if (g->vw && g->vw[l]) cudaFree(g->vw[l]);
    if (g->ow && g->ow[l]) cudaFree(g->ow[l]);
    if (g->qb && g->qb[l]) cudaFree(g->qb[l]);
    if (g->kb && g->kb[l]) cudaFree(g->kb[l]);
    if (g->vb && g->vb[l]) cudaFree(g->vb[l]);
    if (g->ln2 && g->ln2[l]) cudaFree(g->ln2[l]);
    if (g->gw && g->gw[l]) cudaFree(g->gw[l]);
    if (g->uw && g->uw[l]) cudaFree(g->uw[l]);
    if (g->dw && g->dw[l]) cudaFree(g->dw[l]);
    if (g->qw_h && g->qw_h[l]) cudaFree(g->qw_h[l]);
    if (g->kw_h && g->kw_h[l]) cudaFree(g->kw_h[l]);
    if (g->vw_h && g->vw_h[l]) cudaFree(g->vw_h[l]);
    if (g->ow_h && g->ow_h[l]) cudaFree(g->ow_h[l]);
    if (g->gw_h && g->gw_h[l]) cudaFree(g->gw_h[l]);
    if (g->uw_h && g->uw_h[l]) cudaFree(g->uw_h[l]);
    if (g->dw_h && g->dw_h[l]) cudaFree(g->dw_h[l]);
  }
  free(g->ln1); free(g->qw); free(g->kw); free(g->vw); free(g->ow);
  free(g->qb); free(g->kb); free(g->vb);
  free(g->ln2); free(g->gw); free(g->uw); free(g->dw);
  free(g->qw_h); free(g->kw_h); free(g->vw_h); free(g->ow_h);
  free(g->gw_h); free(g->uw_h); free(g->dw_h);
  free(g);
}

long sgl_cuda_free_mem_mb(void) {
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  return (long)(free_b / 1048576);
}

int sgl_cuda_ctx(const sgl_cuda_model *m) { return m->ctx; }

int sgl_cuda_forward(sgl_cuda_model *g, const int *tokens, int n, int pos0,
                     float *logits_out) {
  return sgl_cuda_forward_seg(g, tokens, n, pos0, 0, logits_out);
}

int sgl_cuda_forward_seg(sgl_cuda_model *g, const int *tokens, int n, int pos0,
                         int kv_base, float *logits_out) {
  int E = g->E, H = g->H, HK = g->HK, D = g->D, KD = g->KD;
  if (n <= 0 || pos0 + n > g->ctx) return -1;
  if (kv_base + pos0 + n > g->ctx) return -1;
  if (sgl_cuda_reserve(g, n) != 0) return -1;
  if (sgl_cuda_ensure_io(g, n) != 0) return -1;

  /* input tokens go through the pinned mirror + async copy on g->stream
   * (same ordering domain as the kernels; no per-step cudaMalloc) */
  memcpy(g->h_tok, tokens, sizeof(int) * (size_t)n);
  CK(cudaMemcpyAsync(g->d_tok, g->h_tok, sizeof(int) * (size_t)n,
                     cudaMemcpyHostToDevice, g->stream));

  if (g->use_fp16)
    k_embed_h<<<(int)((n * (long)E + 255) / 256), 256, 0, g->stream>>>(g->wte_h, g->d_tok, g->x, n, E);
  else
    k_embed<<<(int)((n * (long)E + 255) / 256), 256, 0, g->stream>>>(g->wte, g->d_tok, g->x, n, E);
  CK(cudaGetLastError());

  for (int l = 0; l < g->L; l++) {
    k_rmsnorm<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->ln1[l], g->h, n, E, g->eps);
    CK(cudaGetLastError());
    gemm_nt_any(g, g->h, g->qw[l], g->qw_h ? g->qw_h[l] : NULL, g->qb[l], g->q, n, E, E);
    gemm_nt_any(g, g->h, g->kw[l], g->kw_h ? g->kw_h[l] : NULL, g->kb[l], g->k, n, KD, E);
    gemm_nt_any(g, g->h, g->vw[l], g->vw_h ? g->vw_h[l] : NULL, g->vb[l], g->v, n, KD, E);
    CK(cudaGetLastError());
    k_rope<<<(int)(((long)n * H * D / 2 + 255) / 256), 256, 0, g->stream>>>(g->q, n * H, D, H, pos0, g->freqs);
    k_rope<<<(int)(((long)n * HK * D / 2 + 255) / 256), 256, 0, g->stream>>>(g->k, n * HK, D, HK, pos0, g->freqs);
    CK(cudaGetLastError());
    /* KV scatter [L][ctx][KD] <- packed [n][KD]: rows equally strided on both
     * sides (src pitch = KD packed, dst pitch = KD in ctx), so the whole
     * scatter is ONE 2D memcpy instead of 2n tiny async copies per layer.
     * (8192-token prefill did 393k WDDM submits and took 174s; now ~48.) */
    {
      float *kcl = g->kc + (long)l * g->ctx * KD;
      float *vcl = g->vc + (long)l * g->ctx * KD;
      size_t row = (size_t)HK * D, pitch = (size_t)KD * sizeof(float);
      size_t dst_off = (size_t)(kv_base + pos0) * pitch;
      CK(cudaMemcpy2DAsync((char *)kcl + dst_off, pitch, g->k, pitch,
                           row * sizeof(float), n, cudaMemcpyDeviceToDevice, g->stream));
      CK(cudaMemcpy2DAsync((char *)vcl + dst_off, pitch, g->v, pitch,
                           row * sizeof(float), n, cudaMemcpyDeviceToDevice, g->stream));
    }
    /* M26a: attention path split. <= SGL_ATTN_SHORT_MAX keeps the original
     * shared-scores kernel (numerics untouched; shared sized to the real
     * row length instead of the full ctx). Longer rows go to the two-pass
     * chunked kernel, lifting the shared-memory/context cap. Same stream
     * as the D2D KV copies above, so no device sync is needed here.
     * SGL_FORCE_LONG_ATTN=1 runs every prompt through the two-pass kernel
     * (testing: it must agree with the short kernel within fp noise). */
    {
      const char *fe = getenv("SGL_FORCE_LONG_ATTN");
      int force_long = (fe && atoi(fe)) ? 1 : 0;
      if (force_long || pos0 + n > SGL_ATTN_SHORT_MAX) {
        k_attn_long<<<n * H, 256, 0, g->stream>>>(
            g->q, g->kc + (long)l * g->ctx * KD, g->vc + (long)l * g->ctx * KD,
            g->attn, pos0, kv_base, n, H, HK, D, 1.0f / sqrtf((float)D));
      } else {
        k_attn<<<n * H, 256, (size_t)(pos0 + n) * sizeof(float), g->stream>>>(
            g->q, g->kc + (long)l * g->ctx * KD, g->vc + (long)l * g->ctx * KD,
            g->attn, pos0, kv_base, n, H, HK, D, 1.0f / sqrtf((float)D));
      }
    }
    CK(cudaGetLastError());
    gemm_nt_any(g, g->attn, g->ow[l], g->ow_h ? g->ow_h[l] : NULL, NULL, g->h, n, E, E);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)n * E + 255) / 256), 256, 0, g->stream>>>(g->x, g->h, (long)n * E);
    k_rmsnorm<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->ln2[l], g->h, n, E, g->eps);
    gemm_nt_any(g, g->h, g->gw[l], g->gw_h ? g->gw_h[l] : NULL, NULL, g->gate, n, g->I, E);
    gemm_nt_any(g, g->h, g->uw[l], g->uw_h ? g->uw_h[l] : NULL, NULL, g->up, n, g->I, E);
    CK(cudaGetLastError());
    k_silu_mul<<<(int)(((long)n * g->I + 255) / 256), 256, 0, g->stream>>>(g->gate, g->up, (long)n * g->I);
    gemm_nt_any(g, g->gate, g->dw[l], g->dw_h ? g->dw_h[l] : NULL, NULL, g->h, n, E, g->I);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)n * E + 255) / 256), 256, 0, g->stream>>>(g->x, g->h, (long)n * E);
    CK(cudaGetLastError());
    {
      static int dbg = -1;
      if (dbg < 0) { const char *e = getenv("SGL_CUDA_DBG"); dbg = e ? atoi(e) : 0; }
      if (dbg > 0 && l % dbg == 0) {
        float host[4];
        cudaStreamSynchronize(g->stream);
        cudaMemcpy(host, g->x, sizeof(float) * 4, cudaMemcpyDeviceToHost);
        printf("[gpu L%d] x=[%.7g, %.7g, %.7g, %.7g]\n", l, host[0], host[1], host[2], host[3]);
        fflush(stdout);
      }
    }
  }
  k_rmsnorm<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->final_norm, g->h, n, E, g->eps);
  CK(cudaGetLastError());
  {
    static int dbg2 = -1;
    if (dbg2 < 0) { const char *e = getenv("SGL_CUDA_DBG"); dbg2 = e ? atoi(e) : 0; }
    if (dbg2 > 0) {
      float host[4], hx[4];
      cudaStreamSynchronize(g->stream);
      cudaMemcpy(host, g->h + (long)(n - 1) * E, sizeof(float) * 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(hx, g->x + (long)(n - 1) * E, sizeof(float) * 4, cudaMemcpyDeviceToHost);
      float fn[4];
      cudaMemcpy(fn, g->final_norm, sizeof(float) * 4, cudaMemcpyDeviceToHost);
      printf("[gpu fin] x=[%.7g,%.7g] w=[%.7g,%.7g] h=[%.7g, %.7g]\n",
             hx[0], hx[1], fn[0], fn[1], host[0], host[1]);
      fflush(stdout);
    }
  }
  {
    if (logits_out) {
      if (sgl_cuda_ensure_lgb(g, 1) != 0) return -1;
      gemm_nt_any(g, g->h + (long)(n - 1) * E, g->lm_head, g->lm_head_h, NULL,
                  g->d_lgb, 1, g->V, E);
      CK(cudaGetLastError());
      CK(cudaMemcpyAsync(g->h_lgb, g->d_lgb, sizeof(float) * (size_t)g->V,
                         cudaMemcpyDeviceToHost, g->stream));
      CK(cudaStreamSynchronize(g->stream));
      memcpy(logits_out, g->h_lgb, sizeof(float) * (size_t)g->V);
    }
  }
  return 0;
}

/* graph-safe KV row write (M26b): destination computed from the DEVICE
 * base/lens arrays, so a captured graph writes to the right slot even as
 * lens[] grows every decode step (the old host-computed D2D copies would
 * replay with stale addresses). One block covers all B rows. */
__global__ void k_kv_write(float *kc, float *vc, const float *k, const float *v,
                           const int *base, const int *lens, int HK, int D, int B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * HK * D;
  if (i >= total) return;
  int b = i / (HK * D);
  int r = i % (HK * D);
  int pos = base[b] + lens[b];
  kc[(long)pos * HK * D + r] = k[(long)b * HK * D + r];
  vc[(long)pos * HK * D + r] = v[(long)b * HK * D + r];
}

/* one decode step (B sequences x 1 token): every kernel launch for a step,
 * reading inputs from the persistent device buffers. Shared by the plain
 * path and the captured CUDA graph (M26b). Graph rules respected: no
 * cudaMalloc/cudaFree, no syncs, all launches on g->stream, per-sequence
 * positions/KV addresses read from device arrays.
 * maxu = max(lens[b])+1 as seen from the host. maxu <= SGL_ATTN_SHORT_MAX
 * keeps the original shared-scores attention kernel (numerics untouched);
 * capture passes maxu = ctx+1 to force the two-pass kernel, whose fixed
 * shared-memory footprint works for any length a replayed graph sees. */
static int decode_step_launch(sgl_cuda_model *g, int B, int maxu) {
  int E = g->E, H = g->H, HK = g->HK, D = g->D, KD = g->KD;
  cudaStream_t s = g->stream;

  if (g->use_fp16)
    k_embed_h<<<(int)((B * (long)E + 255) / 256), 256, 0, s>>>(g->wte_h, g->d_tok, g->x, B, E);
  else
    k_embed<<<(int)((B * (long)E + 255) / 256), 256, 0, s>>>(g->wte, g->d_tok, g->x, B, E);
  CK(cudaGetLastError());

  for (int l = 0; l < g->L; l++) {
    k_rmsnorm<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->ln1[l], g->h, B, E, g->eps);
    CK(cudaGetLastError());
    gemm_nt_any(g, g->h, g->qw[l], g->qw_h ? g->qw_h[l] : NULL, g->qb[l], g->q, B, E, E);
    gemm_nt_any(g, g->h, g->kw[l], g->kw_h ? g->kw_h[l] : NULL, g->kb[l], g->k, B, KD, E);
    gemm_nt_any(g, g->h, g->vw[l], g->vw_h ? g->vw_h[l] : NULL, g->vb[l], g->v, B, KD, E);
    CK(cudaGetLastError());
    k_rope_rows<<<(int)(((long)B * H * D / 2 + 255) / 256), 256, 0, s>>>(g->q, B * H, D, g->d_pq, g->freqs);
    k_rope_rows<<<(int)(((long)B * HK * D / 2 + 255) / 256), 256, 0, s>>>(g->k, B * HK, D, g->d_pk, g->freqs);
    CK(cudaGetLastError());
    k_kv_write<<<(int)((B * HK * D + 255) / 256), 256, 0, s>>>(
        g->kc + (long)l * g->ctx * KD, g->vc + (long)l * g->ctx * KD,
        g->k, g->v, g->d_base, g->d_lens, HK, D, B);
    CK(cudaGetLastError());
    if (maxu > SGL_ATTN_SHORT_MAX) {
      k_attn_long_batch<<<B * H, 256, 0, s>>>(
          g->q, g->kc + (long)l * g->ctx * KD, g->vc + (long)l * g->ctx * KD,
          g->attn, g->d_base, g->d_lens, B, H, HK, D, 1.0f / sqrtf((float)D));
    } else {
      k_attn_batch<<<B * H, 256, (size_t)maxu * sizeof(float), s>>>(
          g->q, g->kc + (long)l * g->ctx * KD, g->vc + (long)l * g->ctx * KD,
          g->attn, g->d_base, g->d_lens, B, H, HK, D, 1.0f / sqrtf((float)D));
    }
    CK(cudaGetLastError());
    gemm_nt_any(g, g->attn, g->ow[l], g->ow_h ? g->ow_h[l] : NULL, NULL, g->h, B, E, E);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)B * E + 255) / 256), 256, 0, s>>>(g->x, g->h, (long)B * E);
    k_rmsnorm<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->ln2[l], g->h, B, E, g->eps);
    gemm_nt_any(g, g->h, g->gw[l], g->gw_h ? g->gw_h[l] : NULL, NULL, g->gate, B, g->I, E);
    gemm_nt_any(g, g->h, g->uw[l], g->uw_h ? g->uw_h[l] : NULL, NULL, g->up, B, g->I, E);
    CK(cudaGetLastError());
    k_silu_mul<<<(int)(((long)B * g->I + 255) / 256), 256, 0, s>>>(g->gate, g->up, (long)B * g->I);
    gemm_nt_any(g, g->gate, g->dw[l], g->dw_h ? g->dw_h[l] : NULL, NULL, g->h, B, E, g->I);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)B * E + 255) / 256), 256, 0, s>>>(g->x, g->h, (long)B * E);
    CK(cudaGetLastError());
  }
  k_rmsnorm<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->final_norm, g->h, B, E, g->eps);
  CK(cudaGetLastError());
  /* lm_head as ONE GEMM over all B rows (h rows are contiguous) */
  gemm_nt_any(g, g->h, g->lm_head, g->lm_head_h, NULL, g->d_lgb, B, g->V, E);
  CK(cudaGetLastError());
  return 0;
}

int sgl_cuda_decode_batch(sgl_cuda_model *g, const int *tokens, const int *base_pos,
                          const int *lens, int B, float *logits_out) {
  int H = g->H, HK = g->HK;
  if (B <= 0 || B > 32) return -1;
  for (int b = 0; b < B; b++)
    if (base_pos[b] + lens[b] + 1 > g->ctx) return -1;
  if (!logits_out) return -1;
  if (sgl_cuda_reserve(g, B) != 0) return -1;
  if (sgl_cuda_ensure_io(g, B) != 0) return -1;
  if (sgl_cuda_ensure_lgb(g, B) != 0) return -1;

  /* fill the pinned input mirrors; a replayed graph re-reads them via its
   * captured H2D memcpy nodes, the plain path copies them explicitly */
  memcpy(g->h_tok, tokens, sizeof(int) * (size_t)B);
  memcpy(g->h_base, base_pos, sizeof(int) * (size_t)B);
  memcpy(g->h_lens, lens, sizeof(int) * (size_t)B);
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) g->h_pq[b * H + h] = lens[b];
    for (int hk = 0; hk < HK; hk++) g->h_pk[b * HK + hk] = lens[b];
  }

  static int use_graph = -1;
  if (use_graph < 0) {
    const char *e = getenv("SGL_CUDA_GRAPH");
    use_graph = (e && atoi(e) == 0) ? 0 : 1; /* default ON, =0 disables */
  }

  int maxu = 1;
  for (int b = 0; b < B; b++)
    if (lens[b] + 1 > maxu) maxu = lens[b] + 1;

  if (use_graph) {
    /* capture a decode step for this B once; replay afterwards. The graph
     * embeds H2D input copies from the pinned mirrors and the D2H of the
     * logits, so a replay only needs the mirrors refreshed. Attention is
     * forced to the two-pass kernel (fixed shared memory): the captured
     * grid must stay valid as lens[] grows across replays. */
    if (!g->gexec[B]) {
      cudaGraph_t graph = NULL;
      cudaGraphExec_t exec = NULL;
      int rc;
      CK(cudaStreamSynchronize(g->stream));
      CK(cudaStreamBeginCapture(g->stream, cudaStreamCaptureModeThreadLocal));
      /* the input upload from the pinned mirrors is part of the graph:
       * every replay re-copies whatever the caller wrote into the mirrors */
      cudaMemcpyAsync(g->d_tok, g->h_tok, sizeof(int) * (size_t)B,
                      cudaMemcpyHostToDevice, g->stream);
      cudaMemcpyAsync(g->d_base, g->h_base, sizeof(int) * (size_t)B,
                      cudaMemcpyHostToDevice, g->stream);
      cudaMemcpyAsync(g->d_lens, g->h_lens, sizeof(int) * (size_t)B,
                      cudaMemcpyHostToDevice, g->stream);
      cudaMemcpyAsync(g->d_pq, g->h_pq, sizeof(int) * (size_t)B * H,
                      cudaMemcpyHostToDevice, g->stream);
      cudaMemcpyAsync(g->d_pk, g->h_pk, sizeof(int) * (size_t)B * HK,
                      cudaMemcpyHostToDevice, g->stream);
      rc = decode_step_launch(g, B, g->ctx + 1 /* force long-path kernel */);
      if (rc == 0) {
        rc = cudaMemcpyAsync(g->h_lgb, g->d_lgb,
                             sizeof(float) * (size_t)B * g->V,
                             cudaMemcpyDeviceToHost, g->stream) == cudaSuccess ? 0 : -1;
      }
      if (rc == 0) {
        rc = cudaStreamEndCapture(g->stream, &graph) == cudaSuccess ? 0 : -1;
      } else {
        cudaStreamEndCapture(g->stream, &graph);
      }
      if (rc != 0 || !graph) {
        fprintf(stderr, "cuda capture failed; falling back to plain decode\n");
        if (graph) cudaGraphDestroy(graph);
        use_graph = 0;
      } else if (cudaGraphInstantiate(&exec, graph, NULL, NULL, 0) != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed; falling back\n");
        cudaGraphDestroy(graph);
        use_graph = 0;
      } else {
        g->graphs[B] = graph;
        g->gexec[B] = exec;
        size_t nn = 0;
        cudaGraphGetNodes(graph, NULL, &nn);
        printf("[cuda] decode graph captured (B=%d, %d nodes)\n", B, (int)nn);
      }
    }
    if (use_graph && g->gexec[B]) {
      if (cudaGraphLaunch((cudaGraphExec_t)g->gexec[B], g->stream) != cudaSuccess)
        return -1;
      CK(cudaStreamSynchronize(g->stream));
      memcpy(logits_out, g->h_lgb, sizeof(float) * (size_t)B * g->V);
      return 0;
    }
  }

  /* plain path: explicit H2D of the pinned mirrors, one step, D2H logits */
  CK(cudaMemcpyAsync(g->d_tok, g->h_tok, sizeof(int) * (size_t)B,
                     cudaMemcpyHostToDevice, g->stream));
  CK(cudaMemcpyAsync(g->d_base, g->h_base, sizeof(int) * (size_t)B,
                     cudaMemcpyHostToDevice, g->stream));
  CK(cudaMemcpyAsync(g->d_lens, g->h_lens, sizeof(int) * (size_t)B,
                     cudaMemcpyHostToDevice, g->stream));
  CK(cudaMemcpyAsync(g->d_pq, g->h_pq, sizeof(int) * (size_t)B * H,
                     cudaMemcpyHostToDevice, g->stream));
  CK(cudaMemcpyAsync(g->d_pk, g->h_pk, sizeof(int) * (size_t)B * HK,
                     cudaMemcpyHostToDevice, g->stream));
  if (decode_step_launch(g, B, maxu) != 0) return -1;
  CK(cudaMemcpyAsync(g->h_lgb, g->d_lgb, sizeof(float) * (size_t)B * g->V,
                     cudaMemcpyDeviceToHost, g->stream));
  CK(cudaStreamSynchronize(g->stream));
  memcpy(logits_out, g->h_lgb, sizeof(float) * (size_t)B * g->V);
  return 0;
}

/* ---- t1: GEMM micro-benchmark, cublasSgemm vs fp16 cublasGemmEx ----
 * Real model shapes; M in {1,8,64}; best-of-N cudaEvent timing. Also times
 * the hand-written M=1 GEMV kernels (fp32 vs fp16 weights) for the dispatch
 * threshold. Pure diagnostic: allocates its own scratch, no global state. */
void sgl_cuda_gemm_bench(void) {
  struct { int N, K; const char *name; } shapes[] = {
      {896, 896, "qwen05 qkv/ow [896,896]"},
      {4864, 896, "qwen05 mlp up [4864,896]"},
      {896, 4864, "qwen05 mlp down [896,4864]"},
      {151936, 896, "qwen05 lm_head [151936,896]"},
      {2048, 2048, "tinyllama qkv/ow [2048,2048]"},
      {5632, 2048, "tinyllama mlp up [5632,2048]"},
      {2048, 5632, "tinyllama mlp down [2048,5632]"},
      {32000, 2048, "tinyllama lm_head [32000,2048]"},
  };
  const int Ms[] = {1, 8, 64};
  const int MMAX = 64, ITERS = 20, WARM = 3;
  cublasHandle_t h;
  cudaEvent_t ev0, ev1;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  printf("[gemm-bench] C[M,N]=A[M,K]@W[N,K]^T; sgemm(fp32) vs hgemm(16F in, 32F acc), %d iters best\n",
         ITERS);
  for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
    int N = shapes[si].N, K = shapes[si].K;
    float *Wf = NULL, *Af = NULL, *Cf = NULL, *Af_h = NULL;
    __half *Wh = NULL, *Ah = NULL;
    if (cudaMalloc(&Wf, (size_t)N * K * 4) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Wh, (size_t)N * K * 2) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Af, (size_t)MMAX * K * 4) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Ah, (size_t)MMAX * K * 2) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Cf, (size_t)MMAX * N * 4) != cudaSuccess) goto next_shape;
    if (cudaMallocHost(&Af_h, (size_t)MMAX * K * 2) != cudaSuccess) goto next_shape;
    {
      /* deterministic fill, values in [-0.25, 0.25] */
      unsigned int rng = 12345u + (unsigned int)(N * 31 + K);
      float *tw = (float *)malloc((size_t)N * K * 4);
      if (!tw) goto next_shape;
      for (long i = 0; i < (long)N * K; i++) {
        rng = rng * 1664525u + 1013904223u;
        tw[i] = ((rng >> 8) % 4096) / 8192.0f - 0.25f;
      }
      cudaMemcpy(Wf, tw, (size_t)N * K * 4, cudaMemcpyHostToDevice);
      __half *th = (__half *)malloc((size_t)N * K * 2);
      if (!th) { free(tw); goto next_shape; }
      for (long i = 0; i < (long)N * K; i++) th[i] = __float2half(tw[i]);
      cudaMemcpy(Wh, th, (size_t)N * K * 2, cudaMemcpyHostToDevice);
      free(th);
      float *ta = (float *)malloc((size_t)MMAX * K * 4);
      if (!ta) { free(tw); goto next_shape; }
      for (long i = 0; i < (long)MMAX * K; i++) {
        rng = rng * 1664525u + 1013904223u;
        ta[i] = ((rng >> 8) % 4096) / 8192.0f - 0.25f;
      }
      cudaMemcpy(Af, ta, (size_t)MMAX * K * 4, cudaMemcpyHostToDevice);
      for (long i = 0; i < (long)MMAX * K; i++) Af_h[i] = __float2half(ta[i]);
      cudaMemcpy(Ah, Af_h, (size_t)MMAX * K * 2, cudaMemcpyHostToDevice);
      free(ta); free(tw);
    }
    for (int mi = 0; mi < 3; mi++) {
      int M = Ms[mi];
      const float al = 1.0f, be = 0.0f;
      float ms_s = 1e30f, ms_h = 1e30f, ms_k32 = 1e30f, ms_k16 = 1e30f;
      /* fp32 cublas */
      for (int it = -WARM; it < ITERS; it++) {
        cudaEventRecord(ev0);
        cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &al, Wf, K, Af, K, &be, Cf, N);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        float ms;
        cudaEventElapsedTime(&ms, ev0, ev1);
        if (it >= 0 && ms < ms_s) ms_s = ms;
      }
      /* fp16 tensor-core cublas (16F in, 32F accumulate + 32F out) */
      for (int it = -WARM; it < ITERS; it++) {
        cudaEventRecord(ev0);
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &al,
                     Wh, CUDA_R_16F, K, Ah, CUDA_R_16F, K, &be,
                     Cf, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        float ms;
        cudaEventElapsedTime(&ms, ev0, ev1);
        if (it >= 0 && ms < ms_h) ms_h = ms;
      }
      if (M == 1) {
        /* hand-written GEMV: fp32 weights vs fp16 weights */
        for (int it = -WARM; it < ITERS; it++) {
          cudaEventRecord(ev0);
          k_gemm_nt<<<N, 256, 256 * sizeof(float)>>>(Af, Wf, NULL, Cf, 1, N, K);
          cudaEventRecord(ev1);
          cudaEventSynchronize(ev1);
          float ms;
          cudaEventElapsedTime(&ms, ev0, ev1);
          if (it >= 0 && ms < ms_k32) ms_k32 = ms;
          cudaEventRecord(ev0);
          k_gemm_nt_h<<<N, 256, 256 * sizeof(float)>>>(Af, Wh, NULL, Cf, 1, N, K);
          cudaEventRecord(ev1);
          cudaEventSynchronize(ev1);
          cudaEventElapsedTime(&ms, ev0, ev1);
          if (it >= 0 && ms < ms_k16) ms_k16 = ms;
        }
      }
      printf("[gemm-bench] %-28s M=%3d  sgemm %8.4f ms  hgemm %8.4f ms (%.2fx)",
             shapes[si].name, M, ms_s, ms_h, ms_s / ms_h);
      if (M == 1)
        printf("  hand fp32 %7.4f ms  hand fp16 %7.4f ms", ms_k32, ms_k16);
      printf("\n");
    }
  next_shape:
    if (Wf) cudaFree(Wf);
    if (Wh) cudaFree(Wh);
    if (Af) cudaFree(Af);
    if (Ah) cudaFree(Ah);
    if (Cf) cudaFree(Cf);
    if (Af_h) cudaFreeHost(Af_h);
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  cublasDestroy(h);
  fflush(stdout);
}

