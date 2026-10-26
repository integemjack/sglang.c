/* sgl cuda_backend.cu - llama-family GPU forward (M9 v1)
 * kernels: embed / rmsnorm / gemm_nt (block-reduce dot) / rope (precomputed
 * freqs) / attention (block per (t,h), shared scores) / silu_mul / residual.
 * Numerics match llama.c (fp32; reduction order may differ ~1e-5). */
#include <cuda_runtime.h>
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

/* growable activation workspace: make sure x/h/q/attn [E rows], k/v [KD
 * rows], gate/up [I rows] each cover n rows. Grows geometrically so decode
 * (n=B) allocates once and prefill grows to the longest segment seen. */
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
  /* M26a: no hard 4096 clamp anymore. Two-pass attention lifts the shared
   * memory limit; the remaining cap is VRAM (KV pool + weights scale with
   * ctx). Default 16384 fits Qwen2.5-0.5B fp32 into a 4GB card (RTX 2060)
   * thanks to tied-lm_head reuse; SGL_CUDA_CTX overrides. */
  {
    const char *e = getenv("SGL_CUDA_CTX");
    int cap = (e && atoi(e) > 0) ? atoi(e) : 16384;
    if (g->ctx > cap) g->ctx = cap;
  }

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

  for (int l = 0; l < g->L; l++) {
    const sgl_llama_layer *L = &m->layers[l];
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
  }
  free(g->ln1); free(g->qw); free(g->kw); free(g->vw); free(g->ow);
  free(g->qb); free(g->kb); free(g->vb);
  free(g->ln2); free(g->gw); free(g->uw); free(g->dw);
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

  k_embed<<<(int)((n * (long)E + 255) / 256), 256, 0, g->stream>>>(g->wte, g->d_tok, g->x, n, E);
  CK(cudaGetLastError());

  for (int l = 0; l < g->L; l++) {
    k_rmsnorm<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->ln1[l], g->h, n, E, g->eps);
    CK(cudaGetLastError());
    gemm_nt_cublas(g->blas, g->stream, g->h, g->qw[l], g->qb[l], g->q, n, E, E);
    gemm_nt_cublas(g->blas, g->stream, g->h, g->kw[l], g->kb[l], g->k, n, KD, E);
    gemm_nt_cublas(g->blas, g->stream, g->h, g->vw[l], g->vb[l], g->v, n, KD, E);
    CK(cudaGetLastError());
    k_rope<<<(int)(((long)n * H * D / 2 + 255) / 256), 256, 0, g->stream>>>(g->q, n * H, D, H, pos0, g->freqs);
    k_rope<<<(int)(((long)n * HK * D / 2 + 255) / 256), 256, 0, g->stream>>>(g->k, n * HK, D, HK, pos0, g->freqs);
    CK(cudaGetLastError());
    for (int t = 0; t < n; t++) {
      /* KV 閻㈩垰鍟惇?[L][ctx][KD]闁挎稒纰嶉惁锛勪沪閸屾碍鍊楅柛妤冨С缁旀潙鈻撶喊澶岀闁?閻犲洤顭烽崗妯跨疀閸涙番鈧繘宕濋悩鑼勾闁稿绻掍簺 l*ctx*KD */
      float *kcl = g->kc + (long)l * g->ctx * KD;
      float *vcl = g->vc + (long)l * g->ctx * KD;
      cudaMemcpyAsync(kcl + ((long)(kv_base + pos0 + t) * HK) * D,
                      g->k + ((long)t * HK) * D, sizeof(float) * (size_t)HK * D,
                      cudaMemcpyDeviceToDevice, g->stream);
      cudaMemcpyAsync(vcl + ((long)(kv_base + pos0 + t) * HK) * D,
                      g->v + ((long)t * HK) * D, sizeof(float) * (size_t)HK * D,
                      cudaMemcpyDeviceToDevice, g->stream);
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
    gemm_nt_cublas(g->blas, g->stream, g->attn, g->ow[l], NULL, g->h, n, E, E);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)n * E + 255) / 256), 256, 0, g->stream>>>(g->x, g->h, (long)n * E);
    k_rmsnorm<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->ln2[l], g->h, n, E, g->eps);
    gemm_nt_cublas(g->blas, g->stream, g->h, g->gw[l], NULL, g->gate, n, g->I, E);
    gemm_nt_cublas(g->blas, g->stream, g->h, g->uw[l], NULL, g->up, n, g->I, E);
    CK(cudaGetLastError());
    k_silu_mul<<<(int)(((long)n * g->I + 255) / 256), 256, 0, g->stream>>>(g->gate, g->up, (long)n * g->I);
    gemm_nt_cublas(g->blas, g->stream, g->gate, g->dw[l], NULL, g->h, n, E, g->I);
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
      gemm_nt_cublas(g->blas, g->stream, g->h + (long)(n - 1) * E, g->lm_head, NULL,
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

  k_embed<<<(int)((B * (long)E + 255) / 256), 256, 0, s>>>(g->wte, g->d_tok, g->x, B, E);
  CK(cudaGetLastError());

  for (int l = 0; l < g->L; l++) {
    k_rmsnorm<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->ln1[l], g->h, B, E, g->eps);
    CK(cudaGetLastError());
    gemm_nt_cublas(g->blas, s, g->h, g->qw[l], g->qb[l], g->q, B, E, E);
    gemm_nt_cublas(g->blas, s, g->h, g->kw[l], g->kb[l], g->k, B, KD, E);
    gemm_nt_cublas(g->blas, s, g->h, g->vw[l], g->vb[l], g->v, B, KD, E);
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
    gemm_nt_cublas(g->blas, s, g->attn, g->ow[l], NULL, g->h, B, E, E);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)B * E + 255) / 256), 256, 0, s>>>(g->x, g->h, (long)B * E);
    k_rmsnorm<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->ln2[l], g->h, B, E, g->eps);
    gemm_nt_cublas(g->blas, s, g->h, g->gw[l], NULL, g->gate, B, g->I, E);
    gemm_nt_cublas(g->blas, s, g->h, g->uw[l], NULL, g->up, B, g->I, E);
    CK(cudaGetLastError());
    k_silu_mul<<<(int)(((long)B * g->I + 255) / 256), 256, 0, s>>>(g->gate, g->up, (long)B * g->I);
    gemm_nt_cublas(g->blas, s, g->gate, g->dw[l], NULL, g->h, B, E, g->I);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)B * E + 255) / 256), 256, 0, s>>>(g->x, g->h, (long)B * E);
    CK(cudaGetLastError());
  }
  k_rmsnorm<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->final_norm, g->h, B, E, g->eps);
  CK(cudaGetLastError());
  /* lm_head as ONE GEMM over all B rows (h rows are contiguous) */
  gemm_nt_cublas(g->blas, s, g->h, g->lm_head, NULL, g->d_lgb, B, g->V, E);
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
