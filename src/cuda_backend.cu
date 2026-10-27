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
#include "sgl/gpt2.h"
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
 * beyond that the two-pass chunked kernel removes the shared-memory cap.
 * G16: the flash kernel is no longer just a long-row fallback. 实测（qwen05 int8,
 * 同进程 best-of-3, cuda_perf）short 核 vs flash：64 tok 32.0/34.8ms、128 tok
 * 44.1/33.4ms、256 tok 68.0/39.8ms、512 169/46、1024 583/76、2048 2201/187、
 * 4096 11445/568ms —— 所以 64 个 token 以上一律走 flash（前提是 head_dim 有
 * 特化；没有特化时仍退回 short/两遍，行为与改前一致）。 */
#define SGL_ATTN_SHORT_MAX 4096 /* short 核的 shared 上限（不支持 flash 的 D 用）*/
#define SGL_ATTN_FLASH_MIN 64   /* 以下 short 核略快（8%），以上 flash 全面占优 */
#define SGL_ATTN_CHUNK 1024
#define SGL_PROF_MAXL 64 /* G1: per-layer attention event pairs (max L) */
/* G3: gexec[B] sentinel meaning "capturing a graph for this B failed". The
 * batch size then runs the plain path; invalidate_graphs() clears it. */
#define SGL_NO_GRAPH ((void *)~(size_t)0)

/* t8 W8A8 per-channel int8 weights: w[N,K] int8 + scale[N]. */
typedef struct { int8_t *w; float *scale; } sgl_w8w;

struct sgl_cuda_model {
  int L, H, HK, E, I, V, D, KD, ctx;
  int sliding_window; /* t9: >0 = banded causal mask (mistral); gpt2 keeps 0 */
  float eps;
  double theta;

  float *wte;        /* [V,E] */
  float *final_norm; /* [E] */
  float *lm_head;    /* [V,E] (aliases wte when tied embeddings) */
  float **ln1, **qw, **kw, **vw, **ow;
  float **qb, **kb, **vb;
  float **ln2, **gw, **uw, **dw;
  sgl_w8w *qw8, *kw8, *vw8, *ow8, *gw8, *uw8, *dw8; /* t8: per-layer [L], NULL when off */
  float *freqs; /* [D/2] */
  cublasHandle_t blas; /* M22: all projections via cublas */
  float *kc, *vc; /* [ctx][KD] */
  float *x, *h, *q, *k, *v, *attn, *gate, *up;
  int ws_cap; /* rows currently covered by the growable workspace buffers */

  cudaStream_t stream;      /* all launches go here (0 = legacy default) */
  void *blas_ws;            /* cublas workspace (pre-allocated, graph-safe) */
  size_t blas_ws_size;

  /* t4 GPT-2 backend (kind=1): llama slots are reused where the math matches
   * (ln1/ln2 = gamma-only norm weight, qw = transposed qkv W [3E,E],
   * ow = transposed attn c_proj, gw = transposed c_fc [4E,E], dw = transposed
   * mlp c_proj [E,4E], wte/lm_head = tied). Extras live below. */
  int kind;                 /* 0 = llama, 1 = gpt2 */
  float *wpe;               /* [ctx,E] absolute position embeddings */
  float *lnf_b;             /* final layernorm beta [E] */
  float **lnb1, **lnb2;     /* per-layer layernorm betas [E] */
  float **qkvb, **ob, **fcb, **mb; /* per-layer proj biases: [3E] [E] [4E] [E] */

  /* t1 fp16 path (SGL_CUDA_FP16=1): weights stored as __half, activations /
   * KV / attention / rope / softmax stay fp32. Only GEMMs consume half:
   * M==1 decode GEMV is a hand-written kernel (half2 loads, fp32 accum),
   * M>1 goes through tensor-core cublasGemmEx (16F in, 32F accum+out). */
  int use_fp16;
  int use_w8;   /* t8: SGL_CUDA_W8=1 -> int8 hidden projections (dp4a decode) */
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

  /* G1 profiling: SGL_CUDA_PROF=1 brackets each layer's attention launch with
   * a CUDA event pair on the forward stream; summed at the end of the forward
   * this is the attention share of prefill time (no intermediate syncs, so the
   * measurement does not serialise the pipeline). */
  int prof;
  cudaEvent_t pf0[SGL_PROF_MAXL], pf1[SGL_PROF_MAXL];

  /* G2: int8 activations + int32 accumulators for the IMMA big-M path.
   * Sized with the workspace (ws_cap rows x max(E,I) cols): every W8
   * projection has K <= max(E,I) and N <= max(E,I). */
  int8_t *w8_a8;   /* [ws_cap][max(E,I)] */
  int32_t *w8_c32; /* [ws_cap][max(E,I)] */
  float *w8_as;    /* [ws_cap] per-row activation scales */
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

/* G14 small-M GEMV：M 很小（2..8，验证型前向 / 小批量）时，k_gemm_nt 每个输出元素
 * 一个 block 会把权重行读 M 遍；这里改成每个输出行 n 一个 block，**权重只读一遍**，
 * 同时累 MT 个 query 行的点积（A 很小，全在 L2）。
 * MT 必须是编译期常量：动态 M 的循环会让累加器 acc[] 落到 local memory，反而把
 * 带宽优势吃光（M=2 实测 22.1ms vs M=1 的 14.6ms；改成模板后见 BENCH G14 节）。 */
/* 跨线程归约：先 warp 内 shuffle（每行 5 步，无同步），再每个 warp 的 lane0 写一份
 * 共享内存，最后线程 0 汇总。MT=8 的旧写法是 MT 轮树形归约，每轮都 8 次
 * __syncthreads（共 64 次同步），实测让 n=5..8 反而比 n=4 慢。 */
template <int MT>
__device__ __forceinline__ void mrow_emit(float *C, const float *acc, const float *bias,
                                          int n, int N, int M, int tid, int nt) {
  __shared__ float wred[MT * 8];
  int lane = tid & 31, warp = tid >> 5, nwarp = nt >> 5;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    float v = acc[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) wred[i * nwarp + warp] = v;
  }
  __syncthreads();
  if (tid == 0)
    for (int i = 0; i < M; i++) {
      float s = 0.f;
      for (int w = 0; w < nwarp; w++) s += wred[i * nwarp + w];
      C[(long)i * N + n] = s + (bias ? bias[n] : 0.0f);
    }
}

template <int MT>
__global__ void k_gemm_nt_mrow(const float *A, const float *W, const float *bias,
                               float *C, int M, int N, int K) {
  int n = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const float *wr = W + (long)n * K;
  float acc[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = 0.f;
  for (int k = tid; k < K; k += nt) {
    float w = wr[k];
#pragma unroll
    for (int i = 0; i < MT; i++)
      if (i < M) acc[i] += A[(long)i * K + k] * w; /* i>=M 时 A 没那行，别越界读 */
  }
  mrow_emit<MT>(C, acc, bias, n, N, M, tid, nt);
}

/* 同上，权重是 __half（fp16 上传）；累加仍全 fp32（与 k_gemm_nt_h 一致）。 */
template <int MT>
__global__ void k_gemm_nt_h_mrow(const float *A, const __half *W, const float *bias,
                                 float *C, int M, int N, int K) {
  int n = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const __half *wr = W + (long)n * K;
  float acc[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = 0.f;
  if ((K & 1) == 0) {
    const __half2 *w2 = (const __half2 *)wr;
    for (int k = tid; k < K / 2; k += nt) {
      float2 wf = __half22float2(w2[k]);
#pragma unroll
      for (int i = 0; i < MT; i++) {
        if (i >= M) continue; /* 同上：A 只分配了 M 行 */
        const float *ar = A + (long)i * K + 2 * k;
        acc[i] += ar[0] * wf.x + ar[1] * wf.y;
      }
    }
  } else {
    for (int k = tid; k < K; k += nt) {
      float w = __half2float(wr[k]);
#pragma unroll
      for (int i = 0; i < MT; i++)
        if (i < M) acc[i] += A[(long)i * K + k] * w;
    }
  }
  mrow_emit<MT>(C, acc, bias, n, N, M, tid, nt);
}

/* t1 fp16 embed lookup: x = fp32(wte_h[token]) */
__global__ void k_embed_h(const __half *wte, const int *tok, float *x, int n, int E) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * E) return;
  x[idx] = __half2float(wte[(long)tok[idx / E] * E + idx % E]);
}

/* t4 gpt2: x[t] = wte[tok[t]] + wpe[pos0+t] (absolute positions, fp32) */
__global__ void k_embed_pos(const float *wte, const float *wpe, const int *tok,
                            float *x, int n, int E, int pos0) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * E) return;
  int t = idx / E;
  x[idx] = wte[(long)tok[t] * E + idx % E] + wpe[(long)(pos0 + t) * E + idx % E];
}

/* t4 gpt2 fp16 weights: embed table is half, wpe stays fp32 (small) */
__global__ void k_embed_pos_h(const __half *wte, const float *wpe, const int *tok,
                              float *x, int n, int E, int pos0) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * E) return;
  int t = idx / E;
  x[idx] = __half2float(wte[(long)tok[t] * E + idx % E]) + wpe[(long)(pos0 + t) * E + idx % E];
}

/* t4 gpt2: LayerNorm WITH beta (mean/var form, matches sgl_layernorm numerics) */
__global__ void k_layernorm_b(const float *x, const float *g, const float *b,
                              float *y, int M, int N, float eps) {
  extern __shared__ float red[];
  int m = blockIdx.x;
  const float *xr = x + (long)m * N;
  float *yr = y + (long)m * N;
  float sm = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) sm += xr[i];
  red[threadIdx.x] = sm;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float mean = red[0] / N;
  float sv = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float d = xr[i] - mean;
    sv += d * d;
  }
  red[threadIdx.x] = sv;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / N + eps);
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    yr[i] = (xr[i] - mean) * inv * g[i] + b[i];
}

/* t4 gpt2: tanh-approx GELU in place (same formula as sgl_gelu_inplace) */
__global__ void k_gelu(float *row, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float k = 0.7978845608028654f;
  float x = row[i];
  float inner = k * (x + 0.044715f * x * x * x);
  row[i] = 0.5f * x * (1.0f + tanhf(inner));
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
                       float scale, int q_stride, int win) {
  extern __shared__ float sc[];
  int th = blockIdx.x;
  int t = th / H, h = th % H;
  int upto = pos0 + t + 1;
  /* win>0 (mistral sliding window): attend only p in [upto-win, upto) */
  int from = (win > 0 && upto > win) ? upto - win : 0;
  const float *qh = q + (long)t * q_stride + h * D;
  int hkv = h / (H / HK);
  float maxv = -1e30f;
  for (int p = from + threadIdx.x; p < upto; p += blockDim.x) {
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
  for (int p = from + threadIdx.x; p < upto; p += blockDim.x) {
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
  for (int p = from; p < upto; p++) {
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
                            float scale, int win) {
  __shared__ float sc[SGL_ATTN_CHUNK];
  __shared__ float red[256];
  int th = blockIdx.x;
  int t = th / H, h = th % H;
  int upto = pos0 + t + 1;
  const float *qh = q + ((long)t * H + h) * D;
  int hkv = h / (H / HK);
  int from = (win > 0 && upto > win) ? upto - win : 0; /* t9 sliding window */

  /* pass 1: global max + online sum of exp(score - running max) */
  float run_max = -1e30f, run_sum = 0.0f;
  for (int c0 = from; c0 < upto; c0 += SGL_ATTN_CHUNK) {
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
  for (int c0 = from; c0 < upto; c0 += SGL_ATTN_CHUNK) {
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
                                  int B, int H, int HK, int D, float scale, int win) {
  __shared__ float sc[SGL_ATTN_CHUNK];
  __shared__ float red[256];
  int th = blockIdx.x;
  int b = th / H, h = th % H;
  int base_p = base[b];
  int upto = lens[b] + 1;
  const float *qh = q + ((long)b * H + h) * D;
  int hkv = h / (H / HK);
  int from = (win > 0 && upto > win) ? upto - win : 0; /* t9 sliding window */

  float run_max = -1e30f, run_sum = 0.0f;
  for (int c0 = from; c0 < upto; c0 += SGL_ATTN_CHUNK) {
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
  for (int c0 = from; c0 < upto; c0 += SGL_ATTN_CHUNK) {
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

/* ------------------------------------------------------------------------
 * G1: single-pass tiled flash attention for long-context prefill.
 *
 * The two-pass k_attn_long above costs 66.2s for an 8192-token qwen05 prefill.
 * Measured cause (SGL_CUDA_PROF=1 per-stage events, docs/NOTES_g1.md): the
 * kernel is global-memory bound, not FLOP bound -- one block per (token, head)
 * with a thread-per-position dot loop reads K[p][0..D) one float at a time,
 * so consecutive lanes touch addresses HK*D floats apart: every 32-lane load
 * moves 128 B of useful data spread over 32 sectors => ~1/8 sector efficiency,
 * and the whole KV range is re-read for every one of the n tokens (twice,
 * because pass 2 recomputes the scores).
 *
 * This kernel keeps the same math but reorganises it:
 *   block  = (tile of TQ query tokens, head), 8 warps, 256 threads
 *   warp w = QPW consecutive queries, lane j = KV position c0+j (TK per tile)
 *   K/V tiles are cooperatively staged into shared memory (coalesced along d,
 *   read ONCE per TQ-token block instead of once per token), the dot loop
 *   reads Qs broadcast + Ks with pitch D+1 (bank-conflict-free), the online
 *   softmax (running max/sum) is a warp reduction, and each lane's weight is
 *   __shfl-broadcast into the per-lane output accumulator (ND = D/32 dims).
 * K is dot-producted exactly once: no second pass.
 * ------------------------------------------------------------------------ */
template <int D, int QPW, int TK>
__global__ __launch_bounds__(256) void k_attn_flash(
    const float *q, const float *kc, const float *vc, float *out,
    int pos0, int kv_base, int n, int H, int HK, float scale, int win) {
  const int TQ = 8 * QPW;
  const int ND = D / 32;
  __shared__ float Qs[TQ][D + 1];
  __shared__ float Ks[TK][D + 1];
  __shared__ float Vs[TK][D + 1];

  int th = blockIdx.x;
  int tile = th / H, h = th % H;
  int tq0 = tile * TQ;
  int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int hkv = h / (H / HK);

  for (int idx = threadIdx.x; idx < TQ * D; idx += 256) {
    int t = idx / D, d = idx - t * D;
    int tq = tq0 + t;
    Qs[t][d] = (tq < n) ? q[((long) tq * H + h) * D + d] : 0.0f;
  }
  __syncthreads();

  int tlast = (tq0 + TQ <= n) ? (tq0 + TQ - 1) : (n - 1);
  int upto = pos0 + tlast + 1;
  int from = 0;
  if (win > 0) { int lo = pos0 + tq0 + 1 - win; from = lo > 0 ? lo : 0; }

  float mrun[QPW], lrun[QPW], acc[QPW][ND];
#pragma unroll
  for (int i = 0; i < QPW; i++) {
    mrun[i] = -1e30f;
    lrun[i] = 0.0f;
#pragma unroll
    for (int nd = 0; nd < ND; nd++) acc[i][nd] = 0.0f;
  }

  for (int c0 = from; c0 < upto; c0 += TK) {
    int ce = c0 + TK;
    if (ce > upto) ce = upto;
    for (int idx = threadIdx.x; idx < TK * D; idx += 256) {
      int p = idx / D, d = idx - p * D;
      int gp = c0 + p;
      if (gp < ce) {
        long o = ((long) (kv_base + gp) * HK + hkv) * D + d;
        Ks[p][d] = kc[o];
        Vs[p][d] = vc[o];
      } else {
        Ks[p][d] = 0.0f;
        Vs[p][d] = 0.0f;
      }
    }
    __syncthreads();

    int gp = c0 + lane;
    float e[QPW];
#pragma unroll
    for (int i = 0; i < QPW; i++) {
      int qi = w * QPW + i;
      int tq = tq0 + qi;
      float s = 0.0f;
#pragma unroll
      for (int d = 0; d < D; d++) s += Qs[qi][d] * Ks[lane][d];
      s *= scale;
      int hi = pos0 + tq;
      int lo = (win > 0) ? (pos0 + tq + 1 - win) : 0;
      if (lo < 0) lo = 0;
      if (gp > hi || gp < lo || gp >= ce) s = -1e30f;
      float mt = s;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
      float mn = fmaxf(mrun[i], mt);
      float alpha = (mrun[i] <= -1e29f) ? 0.0f : __expf(mrun[i] - mn);
      e[i] = (s <= -1e29f) ? 0.0f : __expf(s - mn);
      float esum = e[i];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) esum += __shfl_xor_sync(0xffffffffu, esum, off);
      lrun[i] = lrun[i] * alpha + esum;
#pragma unroll
      for (int nd = 0; nd < ND; nd++) acc[i][nd] *= alpha;
      mrun[i] = mn;
    }
#pragma unroll
    for (int pp = 0; pp < TK; pp++) {
#pragma unroll
      for (int i = 0; i < QPW; i++) {
        float wp = __shfl_sync(0xffffffffu, e[i], pp);
#pragma unroll
        for (int nd = 0; nd < ND; nd++) acc[i][nd] += wp * Vs[pp][lane + 32 * nd];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < QPW; i++) {
    int qi = w * QPW + i, tq = tq0 + qi;
    if (tq >= n) continue;
    float inv = 1.0f / lrun[i];
    float *o = out + ((long) tq * H + h) * D;
#pragma unroll
    for (int nd = 0; nd < ND; nd++) o[lane + 32 * nd] = acc[i][nd] * inv;
  }
}

/* flash launch: TQ/TK chosen per head_dim so shared memory stays under the
 * 64 KB/SM budget (2 blocks/SM). Returns 0 on success, -1 if D is unsupported
 * (caller then falls back to the two-pass kernel).
 * G16: the supported head dims are the instantiations below — the kernel needs
 * D % 32 == 0 (ND = D/32 output dims per lane); D=80/96 were added for
 * gemma/qwen-style models that are not 64/128 (D=96 verified numerically via
 * tests/long_vcpu.c against the numpy-equivalent CPU path, see BENCH.md). */
static int launch_flash(sgl_cuda_model *g, int l, int n, int pos0, int kv_base,
                        float scale) {
  int H = g->H, HK = g->HK, D = g->D, KD = g->KD;
  const float *kc = g->kc + (long) l * g->ctx * KD;
  const float *vc = g->vc + (long) l * g->ctx * KD;
  int win = g->sliding_window;
  cudaStream_t s = g->stream;
  /* G16: ND = D/32 必须是整数，否则 lane 只覆盖 D 的前 32*ND 维（D=80 就是这种：
   * 32*2=64 < 80，用 flash 会错算且不报错）。没特化的 D 一律回退到 short/两遍。*/
  if (D % 32 != 0) {
    if (getenv("SGL_ATTN_DEBUG"))
      fprintf(stderr, "[attn] flash skipped: D=%d not a multiple of 32 -> slow path\n", D);
    return -1;
  }
  if (D == 64) {
    const int TQ = 8 * 4;
    if (getenv("SGL_ATTN_DEBUG"))
      fprintf(stderr, "[attn] flash D=64 QPW=4 TK=32 n=%d pos0=%d\n", n, pos0);
    k_attn_flash<64, 4, 32><<<(n + TQ - 1) / TQ * H, 256, 0, s>>>(
        g->q, kc, vc, g->attn, pos0, kv_base, n, H, HK, scale, win);
  } else if (D == 128) {
    const int TQ = 8 * 2;
    if (getenv("SGL_ATTN_DEBUG"))
      fprintf(stderr, "[attn] flash D=128 QPW=2 TK=16 n=%d pos0=%d\n", n, pos0);
    k_attn_flash<128, 2, 16><<<(n + TQ - 1) / TQ * H, 256, 0, s>>>(
        g->q, kc, vc, g->attn, pos0, kv_base, n, H, HK, scale, win);
  } else if (D == 96) {
    /* shared = 16*(97) + 2*32*(97) floats = 30.3 KB -> 2 blocks/SM */
    const int TQ = 8 * 2;
    if (getenv("SGL_ATTN_DEBUG"))
      fprintf(stderr, "[attn] flash D=96 QPW=2 TK=32 n=%d pos0=%d\n", n, pos0);
    k_attn_flash<96, 2, 32><<<(n + TQ - 1) / TQ * H, 256, 0, s>>>(
        g->q, kc, vc, g->attn, pos0, kv_base, n, H, HK, scale, win);
  } else {
    if (getenv("SGL_ATTN_DEBUG"))
      fprintf(stderr, "[attn] flash unsupported D=%d -> two-pass\n", D);
    return -1;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* G16: 单点决定走哪个 attention 核（以前这段在 prof / 非 prof 两分支里各抄一份）。
 * 优先级：SGL_ATTN_TWOPASS=1 -> 两遍核（A/B）；否则 pos0+n > FLASH_MIN（或
 * SGL_FORCE_LONG_ATTN）时先试 flash，flash 不支持该 head_dim 才退回：
 *   行不长（<= SHORT_MAX）-> short 核（shared 分数，保持老行为）
 *   行很长              -> 两遍核（shared 无上限） */
static void attn_dispatch(sgl_cuda_model *g, int l, int n, int pos0, int kv_base,
                          float sc, int two_pass, int force_long, int E) {
  int D = g->D, H = g->H, HK = g->HK;
  const float *kc = g->kc + (long) l * g->ctx * g->KD;
  const float *vc = g->vc + (long) l * g->ctx * g->KD;
  int long_row = two_pass || force_long || pos0 + n > SGL_ATTN_FLASH_MIN;
  if (!long_row) {
    k_attn<<<n * H, 256, (size_t)(pos0 + n) * sizeof(float), g->stream>>>(
        g->q, kc, vc, g->attn, pos0, kv_base, n, H, HK, D, sc, E, g->sliding_window);
    return;
  }
  if (!two_pass && launch_flash(g, l, n, pos0, kv_base, sc) == 0) return;
  k_attn_long<<<n * H, 256, 0, g->stream>>>(
      g->q, kc, vc, g->attn, pos0, kv_base, n, H, HK, D, sc, g->sliding_window);
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

/* t8: W8A8 GEMV via __dp4a (Turing native int8 MAC).
 * A row (K floats) is quantized per-tensor-absmax to int8 on the fly in the
 * kernel (read twice: pass 1 absmax, pass 2 pack); W is per-output-channel
 * int8 with scale[n]. Accurate enough for hidden projections; lm_head stays
 * full precision. C[n] = dp4a(A8,W8) * ascale * wscale[n] + bias. */
__global__ void k_gemm_w8a8(const float *A, const int8_t *W, const float *ws,
                            const float *bias, float *C, int M, int N, int K) {
  int n = blockIdx.x;
  int m = blockIdx.y;
  if (m >= M || n >= N) return;
  const float *a = A + (long)m * K;
  /* per-row activation absmax */
  float amax = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    float v = fabsf(a[i]);
    if (v > amax) amax = v;
  }
  __shared__ float red[256];
  red[threadIdx.x] = amax;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s && red[threadIdx.x + s] > red[threadIdx.x]) red[threadIdx.x] = red[threadIdx.x + s];
    __syncthreads();
  }
  float as_inv = red[0] > 1e-8f ? 127.0f / red[0] : 0.f;
  float as = red[0] > 1e-8f ? red[0] / 127.0f : 0.f;
  const int8_t *w = W + (long)n * K;
  int acc = 0;
  /* 16-wide stripes: 4x dp4a per iteration, activation bytes re-derived from
   * float each time (no extra smem pass; bandwidth-bound anyway) */
  for (int i = threadIdx.x * 16; i + 15 < K; i += blockDim.x * 16) {
    signed char abuf[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) {
      float q = __float2int_rn(a[i + j] * as_inv);
      q = q > 127.f ? 127.f : (q < -127.f ? -127.f : q);
      abuf[j] = (signed char)q;
    }
    int acc0 = __dp4a(*(const int *)(abuf + 0), *(const int *)(w + i + 0), 0);
    acc0 = __dp4a(*(const int *)(abuf + 4), *(const int *)(w + i + 4), acc0);
    acc0 = __dp4a(*(const int *)(abuf + 8), *(const int *)(w + i + 8), acc0);
    acc0 = __dp4a(*(const int *)(abuf + 12), *(const int *)(w + i + 12), acc0);
    acc += acc0;
  }
  /* tail (K % 16): scalar fallback, rare (K=896/4864/2048 all %16==0) */
  for (int i = K - (K % 16) + threadIdx.x; i < K; i += blockDim.x) {
    float q = __float2int_rn(a[i] * as_inv);
    q = q > 127.f ? 127.f : (q < -127.f ? -127.f : q);
    acc += (int)((signed char)q) * (int)w[i];
  }
  __shared__ int iacc[256];
  iacc[threadIdx.x] = acc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) iacc[threadIdx.x] += iacc[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    C[(long)m * N + n] = (float)iacc[0] * as * ws[n] + (bias ? bias[n] : 0.f);
}

/* G2: big-M int8 path. The dp4a kernel above is a per-(m,n) GEMV that
 * re-derives the activation row absmax once per output channel (N times per
 * row!) and reads int8 weights from global per (m,n) pair -- fine at M=1
 * (decode), hopeless at M>1. sgl_cuda_gemm_bench measured, tinyllama
 * [2048,2048] M=8: dp4a 0.9250 ms vs int8 IMMA 0.0692 ms (13.4x). So for
 * prefill we quantize the activations ONCE into int8 and run the tensor-core
 * IMMA GEMM (int8 x int8 -> int32, exact integer accumulation), then apply
 * the per-row x per-channel scales in a dequant epilogue.
 * Integer dot products are order-independent (no overflow: K<=4864, |sum| <=
 * K*127*127 < 2^31), so the IMMA result is bit-identical to the dp4a result --
 * switching paths cannot change W8 numerics, which cuda_test [w8] checks. */
__global__ void k_quant_rows(const float *A, int8_t *A8, float *as, int M, int K) {
  int m = blockIdx.x;
  if (m >= M) return;
  const float *a = A + (long) m * K;
  int8_t *o = A8 + (long) m * K;
  float amax = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    float v = fabsf(a[i]);
    if (v > amax) amax = v;
  }
  __shared__ float red[256];
  red[threadIdx.x] = amax;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s && red[threadIdx.x + s] > red[threadIdx.x]) red[threadIdx.x] = red[threadIdx.x + s];
    __syncthreads();
  }
  float am0 = red[0];
  float inv = am0 > 1e-8f ? 127.0f / am0 : 0.f;
  if (threadIdx.x == 0) as[m] = am0 / 127.0f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    float q = __float2int_rn(a[i] * inv);
    q = q > 127.f ? 127.f : (q < -127.f ? -127.f : q);
    o[i] = (int8_t) q;
  }
}

__global__ void k_w8_dequant(const int32_t *C32, const float *as, const float *ws,
                             const float *bias, float *C, int M, int N) {
  long i = blockIdx.x * (long) blockDim.x + threadIdx.x;
  if (i >= (long) M * N) return;
  int n = (int) (i % N);
  C[i] = (float) C32[i] * as[i / N] * ws[n] + (bias ? bias[n] : 0.f);
}

/* returns 0 when the IMMA path ran, -1 when the caller must use dp4a */
static int gemm_nt_w8_imma(sgl_cuda_model *g, const float *A, const sgl_w8w *W8,
                           const float *bias, float *C, int M, int N, int K) {
  /* G2 A/B knob: SGL_W8_DP4A=1 keeps the old per-(m,n) dp4a path for every M
   * (read per call so cuda_test can compare both paths in one process). */
  { const char *ab = getenv("SGL_W8_DP4A"); if (ab && atoi(ab)) return -1; }
  if (M < 2 || (K % 4) || (N % 4) || !g->w8_a8 || !g->w8_c32) return -1;
  const int32_t ai = 1, bi = 0;
  k_quant_rows<<<M, 256, 0, g->stream>>>(A, g->w8_a8, g->w8_as, M, K);
  cublasStatus_t st = cublasGemmEx(g->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &ai,
                                   W8->w, CUDA_R_8I, K, g->w8_a8, CUDA_R_8I, K, &bi,
                                   g->w8_c32, CUDA_R_32I, N, CUBLAS_COMPUTE_32I,
                                   CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) return -1;
  k_w8_dequant<<<(int) (((long) M * N + 255) / 256), 256, 0, g->stream>>>(
      g->w8_c32, g->w8_as, W8->scale, bias, C, M, N);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* G14: M in [2,8] also goes to a hand kernel — one block per output row,
 * one pass over the weight row for all M query rows. cublas needs ~9 ms more
 * than the M=1 kernel for M=2 on tinyllama (measured), because each of the
 * 66 per-layer GEMMs pays its own launch/tile overhead while the traffic is
 * identical.
 * SGL_SMALLM_CUBLAS=1 forces the old cublas path for A/B. */
static int small_m_hand(void) {
  static int on = -1;
  if (on < 0) on = getenv("SGL_SMALLM_CUBLAS") ? 0 : 1;
  return on;
}

/* G14: dispatch the templated small-M GEMV by rounding M up to 2/4/8 so the
 * accumulator loop is compile-time unrolled (dynamic M spills acc[] to local
 * memory and loses the whole point). */
static void gemm_nt_mrow_launch(cudaStream_t s, const float *A, const float *W,
                                const float *bias, float *C, int M, int N, int K) {
  if (M <= 2) k_gemm_nt_mrow<2><<<N, 256, 2 * 256 * sizeof(float), s>>>(A, W, bias, C, M, N, K);
  else if (M <= 4) k_gemm_nt_mrow<4><<<N, 256, 4 * 256 * sizeof(float), s>>>(A, W, bias, C, M, N, K);
  else k_gemm_nt_mrow<8><<<N, 256, 8 * 256 * sizeof(float), s>>>(A, W, bias, C, M, N, K);
}
static void gemm_nt_h_mrow_launch(cudaStream_t s, const float *A, const __half *W,
                                  const float *bias, float *C, int M, int N, int K) {
  if (M <= 2) k_gemm_nt_h_mrow<2><<<N, 256, 2 * 256 * sizeof(float), s>>>(A, W, bias, C, M, N, K);
  else if (M <= 4) k_gemm_nt_h_mrow<4><<<N, 256, 4 * 256 * sizeof(float), s>>>(A, W, bias, C, M, N, K);
  else k_gemm_nt_h_mrow<8><<<N, 256, 8 * 256 * sizeof(float), s>>>(A, W, bias, C, M, N, K);
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
  if (M >= 2 && M <= 8 && !force_blas && small_m_hand()) {
    gemm_nt_mrow_launch(stream, A, W, bias, C, M, N, K);
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
  if (M >= 2 && M <= 8 && small_m_hand()) {
    gemm_nt_h_mrow_launch(st, A, W_h, bias, C, M, N, K);
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

/* t5 fix: captured graphs bind device pointers (x/h/q/... activation buffers)
 * and pinned host addresses; any realloc below frees those, so cached graphs
 * must be dropped. They re-capture lazily on the next decode. */
static void invalidate_graphs(sgl_cuda_model *g) {
  for (int b = 0; b <= 32; b++) {
    if (g->graphs[b]) { cudaGraphDestroy((cudaGraph_t)g->graphs[b]); g->graphs[b] = NULL; }
    if (g->gexec[b] && g->gexec[b] != SGL_NO_GRAPH) {
      cudaGraphExecDestroy((cudaGraphExec_t)g->gexec[b]);
    }
    g->gexec[b] = NULL; /* also clears the per-B "cannot capture" marker */
  }
}

/* growable activation workspace: make sure x/h/q/attn [E rows], k/v [KD
 * rows], gate/up [I rows] each cover n rows. Grows geometrically so decode
 * (n=B) allocates once and prefill grows to the longest segment seen.
 * t1: also the fp16 GEMM-input mirror ha [ws_cap x max(E,I)] (cublas path). */
static int sgl_cuda_reserve(sgl_cuda_model *g, int n) {
  if (n <= g->ws_cap) return 0;
  invalidate_graphs(g); /* realloc below frees buffers old graphs point to */
  float *x = dev_copy(NULL, (size_t)n * g->E);
  float *h = dev_copy(NULL, (size_t)n * g->E);
  /* t4 gpt2: the q slot doubles as the fused-qkv buffer [n,3E] */
  int q_cols = g->kind == 1 ? 3 * g->E : g->E;
  float *q = dev_copy(NULL, (size_t)n * q_cols);
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
  if (g->use_w8) {
    /* G2: IMMA scratch. invalidate_graphs() already ran at the top of this
     * function, so growing these buffers is graph-safe (rule 4). */
    int EI = g->E > g->I ? g->E : g->I;
    int8_t *a8 = NULL;
    int32_t *c32 = NULL;
    float *as = NULL;
    if (cudaMalloc(&a8, (size_t)n * EI) != cudaSuccess ||
        cudaMalloc(&c32, (size_t)n * EI * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&as, (size_t)n * sizeof(float)) != cudaSuccess) {
      cudaFree(a8); cudaFree(c32); cudaFree(as);
      fprintf(stderr, "sgl_cuda_reserve(%d): out of VRAM (w8 imma scratch)\n", n);
      return -1;
    }
    cudaFree(g->w8_a8); cudaFree(g->w8_c32); cudaFree(g->w8_as);
    g->w8_a8 = a8; g->w8_c32 = c32; g->w8_as = as;
  }
  g->ws_cap = n;
  return 0;
}

/* persistent per-step I/O buffers: device arrays + pinned host mirrors.
 * The pinned mirrors are what a captured graph re-reads on every replay
 * (memcpy nodes capture the host address, not the contents). */
static int sgl_cuda_ensure_io(sgl_cuda_model *g, int B) {
  if (B <= g->io_cap_B) return 0;
  /* t5 fix: captured graphs bind the CURRENT pinned host addresses in their
   * memcpy nodes; growing the buffers frees those addresses, so every cached
   * graph must be dropped (they re-capture on the next decode with the new
   * pointers). Symptom was a sticky CUDA error surfacing in the NEXT
   * unrelated cudaGetLastError() — e.g. /generate batch B=8 captured
   * graphs, then a longer chat prefill grew h_tok and replaying B<=8 read a
   * freed pinned buffer. */
  invalidate_graphs(g);
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

/* logits buffers (device + pinned host), grown to `rows` rows of V floats.
 * G7: captured decode graphs bind d_lgb/h_lgb addresses in their D2H logits
 * memcpy node (see capture_decode_graph), so a growth (free + realloc) must
 * drop them first — same rule as the activation workspace (rule 4). */
static int sgl_cuda_ensure_lgb(sgl_cuda_model *g, int rows) {
  if (rows <= g->lgb_cap) return 0;
  invalidate_graphs(g);
  cudaFree(g->d_lgb); cudaFreeHost(g->h_lgb);
  CK(cudaMalloc(&g->d_lgb, sizeof(float) * (size_t)rows * g->V));
  CK(cudaMallocHost(&g->h_lgb, sizeof(float) * (size_t)rows * g->V));
  g->lgb_cap = rows;
  return 0;
}

/* batch attention: block = (seq, head); kv at absolute (base+p) */
__global__ void k_attn_batch(const float *q, const float *kc, const float *vc,
                             float *out, const int *base, const int *lens,
                             int B, int H, int HK, int D, float scale, int win) {
  extern __shared__ float sc[];
  int th = blockIdx.x;
  int b = th / H, h = th % H;
  int base_p = base[b];
  int upto = lens[b] + 1;
  const float *qh = q + ((long)b * H + h) * D;
  int hkv = h / (H / HK);
  int from = (win > 0 && upto > win) ? upto - win : 0; /* t9 sliding window */
  float maxv = -1e30f;
  for (int p = from + threadIdx.x; p < upto; p += blockDim.x) {
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
  for (int p = from + threadIdx.x; p < upto; p += blockDim.x) {
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
  for (int p = from; p < upto; p++) {
    const float *vh = vc + ((long)(base_p + p) * HK + hkv) * D;
    float w = sc[p] * invsum;
    for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] += w * vh[d];
  }
}

/* t8: quantize [N,K] fp32 host weights -> int8 device + scale[N] device */
static int dev_copy_w8(const float *src, int N, int K, sgl_w8w *out) {
  int8_t *q = (int8_t *)malloc((size_t)N * K);
  float *s = (float *)malloc(sizeof(float) * (size_t)N);
  if (!q || !s) { free(q); free(s); return -1; }
  for (int n = 0; n < N; n++) {
    float amax = 0;
    const float *r = src + (size_t)n * K;
    for (int k = 0; k < K; k++) { float v = fabsf(r[k]); if (v > amax) amax = v; }
    float inv = amax > 1e-8f ? 127.0f / amax : 0.f;
    s[n] = amax / 127.0f;
    for (int k = 0; k < K; k++) {
      float v = r[k] * inv;
      v = v > 127.f ? 127.f : (v < -127.f ? -127.f : v);
      q[(size_t)n * K + k] = (int8_t)lrintf(v);
    }
  }
  cudaError_t e1 = cudaMalloc((void **)&out->w, (size_t)N * K);
  cudaError_t e2 = cudaMalloc((void **)&out->scale, sizeof(float) * (size_t)N);
  if (e1 != cudaSuccess || e2 != cudaSuccess) {
    if (e1 == cudaSuccess) cudaFree(out->w);
    if (e2 == cudaSuccess) cudaFree(out->scale);
    free(q); free(s);
    return -1;
  }
  cudaMemcpy(out->w, q, (size_t)N * K, cudaMemcpyHostToDevice);
  cudaMemcpy(out->scale, s, sizeof(float) * (size_t)N, cudaMemcpyHostToDevice);
  free(q); free(s);
  return 0;
}

static int w8_alloc_layers(sgl_w8w **arr, int L) {
  *arr = (sgl_w8w *)calloc((size_t)L, sizeof(sgl_w8w));
  return *arr ? 0 : -1;
}

static void w8_free_layers(sgl_w8w *arr, int L) {
  if (!arr) return;
  for (int l = 0; l < L; l++) {
    if (arr[l].w) cudaFree(arr[l].w);
    if (arr[l].scale) cudaFree(arr[l].scale);
  }
  free(arr);
}

/* t8 projection dispatch: W8A8 for every M when int8 weights exist
 * (prefill included -- accuracy verified against fp32 greedy; fp32 originals
 * are NOT uploaded in W8 mode, that's where the VRAM saving comes from).
 * G2: M>1 goes to the int8 tensor-core (IMMA) path; M==1 and the shapes IMMA
 * cannot take stay on the hand dp4a GEMV, which is the faster of the two at
 * batch 1 (weight bandwidth bound, one row of activations). */
#define PROJ(g, A, Wf, Wh, W8, bias, C, M, N, K)                                   \
  do {                                                                             \
    if ((g)->use_w8 && (W8)) {                                                     \
      if (gemm_nt_w8_imma(g, A, W8, bias, C, M, N, K) == 0) break;                 \
      k_gemm_w8a8<<<dim3(N, M), 256, 0, (g)->stream>>>(                            \
          A, (W8)->w, (W8)->scale, bias, C, M, N, K);                              \
      CK(cudaGetLastError());                                                      \
      break;                                                                       \
    }                                                                              \
    gemm_nt_any(g, A, Wf, Wh, bias, C, M, N, K);                                   \
  } while (0)

/* P1 诊断：SGL_CUDA_TRACE=1 时把上传每步打一行，定位「上传时进程崩」到底停在哪次 dev_copy。
 * （Windows + WDDM 下驱动内部段错误拿不到我们的栈，只能看走到哪。）*/
static int cuda_trace_on(void) {
  static int on = -1;
  if (on < 0) on = getenv("SGL_CUDA_TRACE") ? 1 : 0;
  return on;
}
#define CT(...)                        \
  do {                                 \
    if (cuda_trace_on()) {             \
      printf("[ct] " __VA_ARGS__);      \
      printf("\n");                    \
      fflush(stdout);                  \
    }                                  \
  } while (0)

sgl_cuda_model *sgl_cuda_from_llama(const void *llama_model) {
  const sgl_llama *m = (const sgl_llama *)llama_model;
  if (!m) return NULL;
  sgl_cuda_model *g = (sgl_cuda_model *)calloc(1, sizeof(*g));
  size_t ctx = 0; /* declared up front: goto fail below must not bypass it */
  g->L = m->n_layer; g->H = m->n_head; g->HK = m->n_kv_head;
  g->E = m->n_embd; g->I = m->n_inter; g->V = m->vocab;
  g->D = m->head_dim; g->KD = m->kv_dim; g->ctx = m->n_ctx;
  g->eps = m->rms_eps; g->theta = m->rope_theta;
  g->sliding_window = m->sliding_window; /* t9: mistral banded mask, 0 = off */
  /* t1: fp16 weight storage, opt-in via SGL_CUDA_FP16 (default off = the
   * original all-fp32 path, bit-identical behavior). */
  {
    const char *fe = getenv("SGL_CUDA_FP16");
    g->use_fp16 = (fe && atoi(fe)) ? 1 : 0;
    const char *we = getenv("SGL_CUDA_W8");
    g->use_w8 = (we && atoi(we)) ? 1 : 0; /* t8: int8 projections (dp4a) */
    if (g->use_w8) g->use_fp16 = 1;       /* embed table fp16, projections int8 */
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
    CT("upload wte (fp16) V=%d E=%d", m->vocab, m->n_embd);
    g->wte_h = dev_copy_h(m->wte.data, (size_t)m->vocab * m->n_embd);
    CT("upload final_norm");
    g->final_norm = dev_copy(m->final_norm.data, (size_t)m->n_embd);
    /* tied embeddings: the half lm_head aliases the half embed table */
    if (m->lm_head.data == m->wte.data) {
      g->lm_head_h = g->wte_h;
    } else {
      g->lm_head_h = dev_copy_h(m->lm_head.data, (size_t)m->vocab * m->n_embd);
    }
    if (!g->wte_h || !g->final_norm || !g->lm_head_h) goto fail;
  } else {
    CT("upload wte (fp32) V=%d E=%d n=%lld", m->vocab, m->n_embd,
       (long long)((size_t)m->vocab * m->n_embd));
    g->wte = dev_copy(m->wte.data, (size_t)m->vocab * m->n_embd);
    CT("upload final_norm");
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
  if (g->use_w8) { /* t8: int8 copies of the 7 hidden projections (llama path) */
    if (w8_alloc_layers(&g->qw8, g->L) || w8_alloc_layers(&g->kw8, g->L) ||
        w8_alloc_layers(&g->vw8, g->L) || w8_alloc_layers(&g->ow8, g->L) ||
        w8_alloc_layers(&g->gw8, g->L) || w8_alloc_layers(&g->uw8, g->L) ||
        w8_alloc_layers(&g->dw8, g->L))
      goto fail;
  }

  for (int l = 0; l < g->L; l++) {
    const sgl_llama_layer *L = &m->layers[l];
    if (g->use_w8) {
      if (dev_copy_w8(L->q_w.data, g->E, g->E, &g->qw8[l]) != 0 ||
          dev_copy_w8(L->k_w.data, g->KD, g->E, &g->kw8[l]) != 0 ||
          dev_copy_w8(L->v_w.data, g->KD, g->E, &g->vw8[l]) != 0 ||
          dev_copy_w8(L->o_w.data, g->E, g->E, &g->ow8[l]) != 0 ||
          dev_copy_w8(L->gate_w.data, g->I, g->E, &g->gw8[l]) != 0 ||
          dev_copy_w8(L->up_w.data, g->I, g->E, &g->uw8[l]) != 0 ||
          dev_copy_w8(L->down_w.data, g->E, g->I, &g->dw8[l]) != 0)
        goto fail;
      /* norms and biases still fp32 */
      g->ln1[l] = dev_copy(L->inp_norm.data, (size_t)g->E);
      g->ln2[l] = dev_copy(L->post_norm.data, (size_t)g->E);
      g->qb[l] = L->q_b.data ? dev_copy(L->q_b.data, (size_t)g->E) : NULL;
      g->kb[l] = L->k_b.data ? dev_copy(L->k_b.data, (size_t)g->KD) : NULL;
      g->vb[l] = L->v_b.data ? dev_copy(L->v_b.data, (size_t)g->KD) : NULL;
      continue; /* skip fp32/fp16 projection upload — VRAM saving */
    }
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
    CT("layer %d/%d uploaded", l + 1, g->L);
  }

  {
    float *f = (float *)malloc(sizeof(float) * (size_t)(g->D / 2));
    for (int i = 0; i < g->D / 2; i++)
      f[i] = (float)pow(g->theta, -(double)(2 * i) / g->D);
    g->freqs = dev_copy(f, (size_t)(g->D / 2));
    free(f);
  }
  CT("cublasCreate");
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
  CT("kv alloc done L=%d ctx=%zu KD=%d", g->L, ctx, g->KD);
  if (g->use_fp16) printf("[cuda] fp16 weights ON (SGL_CUDA_FP16=1)\n");
  return g;

fail:
  fprintf(stderr, "sgl_cuda_from_llama: out of VRAM or copy failed\n");
  sgl_cuda_free(g);
  return NULL;
}

/* t4: transpose-copy helper — HF Conv1D weight is [K,N] row-major; llama's
 * gemm path wants [N,K] (W row n = weight column n). Host-side one-shot. */
static float *dev_copy_transposed(const float *src, int K, int N, int to_half) {
  float *buf = (float *)malloc((size_t)N * K * sizeof(float));
  if (!buf) return NULL;
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      buf[(size_t)n * K + k] = src[(size_t)k * N + n];
  float *d = to_half ? (float *)dev_copy_h(buf, (size_t)N * K) : dev_copy(buf, (size_t)N * K);
  free(buf);
  return d;
}

sgl_cuda_model *sgl_cuda_from_gpt2(const void *gpt2_model) {
  const sgl_gpt2 *m = (const sgl_gpt2 *)gpt2_model;
  if (!m) return NULL;
  sgl_cuda_model *g = (sgl_cuda_model *)calloc(1, sizeof(*g));
  size_t ctx = 0;
  g->kind = 1;
  g->L = m->n_layer; g->H = m->n_head; g->HK = m->n_head; /* no GQA */
  g->E = m->n_embd; g->I = 4 * m->n_embd; g->V = m->vocab;
  g->D = m->n_embd / m->n_head; g->KD = m->n_embd; g->ctx = m->n_ctx;
  g->eps = m->ln_eps; g->theta = 10000.0; /* unused */
  g->sliding_window = 0; /* gpt2: no sliding window */
  {
    const char *fe = getenv("SGL_CUDA_FP16");
    g->use_fp16 = (fe && atoi(fe)) ? 1 : 0; /* gpt2: no W8 path (t8 is llama-only) */
  }
  {
    const char *e = getenv("SGL_CUDA_CTX");
    int cap = (e && atoi(e) > 0) ? atoi(e) : m->n_ctx; /* gpt2 ctx=1024 fits */
    if (g->ctx > cap) g->ctx = cap;
  }

  if (g->use_fp16) {
    g->wte_h = dev_copy_h(m->wte.data, (size_t)m->vocab * m->n_embd);
    g->lm_head_h = g->wte_h; /* tied */
    if (!g->wte_h) goto fail;
  } else {
    g->wte = dev_copy(m->wte.data, (size_t)m->vocab * m->n_embd);
    g->lm_head = g->wte; /* tied */
    if (!g->wte) goto fail;
  }
  g->wpe = dev_copy(m->wpe.data, (size_t)g->ctx * m->n_embd);
  g->final_norm = dev_copy(m->lnf_g.data, (size_t)m->n_embd);
  g->lnf_b = dev_copy(m->lnf_b.data, (size_t)m->n_embd);
  if (!g->wpe || !g->final_norm || !g->lnf_b) goto fail;

  g->ln1 = (float **)calloc(g->L, sizeof(float *));
  g->ln2 = (float **)calloc(g->L, sizeof(float *));
  g->lnb1 = (float **)calloc(g->L, sizeof(float *));
  g->lnb2 = (float **)calloc(g->L, sizeof(float *));
  g->qkvb = (float **)calloc(g->L, sizeof(float *));
  g->ob = (float **)calloc(g->L, sizeof(float *));
  g->fcb = (float **)calloc(g->L, sizeof(float *));
  g->mb = (float **)calloc(g->L, sizeof(float *));
  /* qkv W lives in qw slot (transposed [3E,E]); attn c_proj in ow; c_fc in
   * gw ([4E,E]); mlp c_proj in dw ([E,4E]). fp16 mirrors use the _h slots. */
  g->qw = (float **)calloc(g->L, sizeof(float *));
  g->ow = (float **)calloc(g->L, sizeof(float *));
  g->gw = (float **)calloc(g->L, sizeof(float *));
  g->dw = (float **)calloc(g->L, sizeof(float *));
  if (g->use_fp16) {
    g->qw_h = (__half **)calloc(g->L, sizeof(__half *));
    g->ow_h = (__half **)calloc(g->L, sizeof(__half *));
    g->gw_h = (__half **)calloc(g->L, sizeof(__half *));
    g->dw_h = (__half **)calloc(g->L, sizeof(__half *));
  }
  if (!g->lnb1 || !g->qkvb || !g->ob || !g->fcb || !g->mb || !g->lnb2) goto fail;

  for (int l = 0; l < g->L; l++) {
    const sgl_gpt2_layer *L = &m->layers[l];
    int E = g->E, E3 = 3 * E, E4 = 4 * E;
    if (g->use_fp16) {
      g->qw_h[l] = (__half *)dev_copy_transposed(L->qkv_w.data, E, E3, 1);
      g->ow_h[l] = (__half *)dev_copy_transposed(L->proj_w.data, E, E, 1);
      g->gw_h[l] = (__half *)dev_copy_transposed(L->fc_w.data, E, E4, 1);
      g->dw_h[l] = (__half *)dev_copy_transposed(L->mlp_proj_w.data, E4, E, 1);
      if (!g->qw_h[l] || !g->gw_h[l] || !g->dw_h[l]) goto fail;
    } else {
      g->qw[l] = dev_copy_transposed(L->qkv_w.data, E, E3, 0);
      g->ow[l] = dev_copy_transposed(L->proj_w.data, E, E, 0);
      g->gw[l] = dev_copy_transposed(L->fc_w.data, E, E4, 0);
      g->dw[l] = dev_copy_transposed(L->mlp_proj_w.data, E4, E, 0);
      if (!g->qw[l] || !g->gw[l] || !g->dw[l]) goto fail;
    }
    g->ln1[l] = dev_copy(L->ln1_g.data, (size_t)E);
    g->lnb1[l] = dev_copy(L->ln1_b.data, (size_t)E);
    g->ln2[l] = dev_copy(L->ln2_g.data, (size_t)E);
    g->lnb2[l] = dev_copy(L->ln2_b.data, (size_t)E);
    g->qkvb[l] = dev_copy(L->qkv_b.data, (size_t)E3);
    g->ob[l] = dev_copy(L->proj_b.data, (size_t)E);
    g->fcb[l] = dev_copy(L->fc_b.data, (size_t)E4);
    g->mb[l] = dev_copy(L->mlp_proj_b.data, (size_t)E);
    if (!g->ln1[l] || !g->lnb1[l] || !g->ln2[l] || !g->lnb2[l] ||
        !g->qkvb[l] || !g->ob[l] || !g->fcb[l] || !g->mb[l]) goto fail;
  }

  if (cublasCreate(&g->blas) != CUBLAS_STATUS_SUCCESS) goto fail;
  if (cudaStreamCreate(&g->stream) != cudaSuccess) goto fail;
  cublasSetStream(g->blas, g->stream);
  g->blas_ws_size = (size_t)32 << 20;
  if (cudaMalloc(&g->blas_ws, g->blas_ws_size) != cudaSuccess) g->blas_ws_size = 0;
  else cublasSetWorkspace(g->blas, g->blas_ws, g->blas_ws_size);

  ctx = (size_t)g->ctx;
  g->kc = dev_copy(NULL, (size_t)g->L * ctx * g->KD);
  g->vc = dev_copy(NULL, (size_t)g->L * ctx * g->KD);
  if (!g->kc || !g->vc) goto fail;
  if (g->use_fp16) printf("[cuda] gpt2 fp16 weights ON (SGL_CUDA_FP16=1)\n");
  return g;

fail:
  fprintf(stderr, "sgl_cuda_from_gpt2: out of VRAM or copy failed\n");
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
  /* t4 gpt2 extras */
  cudaFree(g->wpe); cudaFree(g->lnf_b);
  for (int l = 0; l < g->L; l++) {
    if (g->lnb1 && g->lnb1[l]) cudaFree(g->lnb1[l]);
    if (g->lnb2 && g->lnb2[l]) cudaFree(g->lnb2[l]);
    if (g->qkvb && g->qkvb[l]) cudaFree(g->qkvb[l]);
    if (g->ob && g->ob[l]) cudaFree(g->ob[l]);
    if (g->fcb && g->fcb[l]) cudaFree(g->fcb[l]);
    if (g->mb && g->mb[l]) cudaFree(g->mb[l]);
  }
  free(g->lnb1); free(g->lnb2); free(g->qkvb); free(g->ob); free(g->fcb); free(g->mb);
  w8_free_layers(g->qw8, g->L); w8_free_layers(g->kw8, g->L); w8_free_layers(g->vw8, g->L);
  w8_free_layers(g->ow8, g->L); w8_free_layers(g->gw8, g->L); w8_free_layers(g->uw8, g->L);
  w8_free_layers(g->dw8, g->L);
  cudaFree(g->w8_a8); cudaFree(g->w8_c32); cudaFree(g->w8_as); /* G2 IMMA scratch */
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
int sgl_cuda_vocab(const sgl_cuda_model *m) { return m->V; }

/* G6: 把 [src_base, src_base+len) 这 len 行的 KV（所有层、k 和 v）拷到
 * [dst_base, dst_base+len)。服务端前缀缓存用：命中的前缀不必重算 prefill，
 * 直接把已租让段里的 KV 行搬过来。RoPE 把位置编进了 K，所以只有当两边
 * 都是从位置 0 开始的前缀时才能复用（服务端正是：每条序列在自己的段内
 * pos0 从 0 起）。src==dst 时 no-op。返回 0 成功。 */
int sgl_cuda_copy_kv(sgl_cuda_model *g, int src_base, int dst_base, int len) {
  if (!g || len <= 0) return 0;
  if (src_base == dst_base) return 0;
  if (src_base < 0 || dst_base < 0 || src_base + len > g->ctx || dst_base + len > g->ctx)
    return -1;
  size_t bytes = (size_t) len * g->KD * sizeof(float); /* 同层内行连续：一次 1-D 拷贝 */
  for (int l = 0; l < g->L; l++) {
    size_t off = (size_t) l * g->ctx * g->KD;
    if (cudaMemcpyAsync(g->kc + off + (size_t) dst_base * g->KD,
                        g->kc + off + (size_t) src_base * g->KD,
                        bytes, cudaMemcpyDeviceToDevice, g->stream) != cudaSuccess)
      return -1;
    if (cudaMemcpyAsync(g->vc + off + (size_t) dst_base * g->KD,
                        g->vc + off + (size_t) src_base * g->KD,
                        bytes, cudaMemcpyDeviceToDevice, g->stream) != cudaSuccess)
      return -1;
  }
  return cudaStreamSynchronize(g->stream) == cudaSuccess ? 0 : -1;
}

/* ---- G5: 上传前的显存预检 ----
 * 口径与 sgl_cuda_from_llama 的分配一致（同一个环境变量开关决定权重位宽）：
 *   权重   embed V*E + (不 tied 时 lm_head V*E)
 *          + L*(q E*E + o E*E + k KD*E + v KD*E + gate/up/down 3*I*E)
 *   KV     2 * L * ctx * KD * 4
 *   工作区 rows 行：x/h/attn(E) + q(E) + k/v(KD) + gate/up(I)
 *          + fp16 镜像 rows*max(E,I)*2 + int8 暂存 rows*max(E,I)*(1+4+4)
 *   logits 设备镜像 V*4 * conc
 *   cublas workspace 32MB
 *   CUDA 上下文 + 内核模块：本机实测（RTX 2060 / CUDA 13.3）第一次 cudaMalloc
 *          之后 free 掉约 300MB，取 320MB。
 * 估算不是精确值，所以调用方还要留余量（cuda_http 用 192MB）。 */
long sgl_cuda_estimate_vram_mb(const void *llama_model, int ctx, int rows) {
  const sgl_llama *m = (const sgl_llama *)llama_model;
  if (!m) return -1;
  if (ctx <= 0) ctx = m->n_ctx;
  if (rows <= 0) rows = ctx / 8;
  int L = m->n_layer, E = m->n_embd, I = m->n_inter, V = m->vocab;
  int KD = m->kv_dim > 0 ? m->kv_dim : m->n_embd;
  const char *fe = getenv("SGL_CUDA_FP16");
  int fp16 = (fe && atoi(fe)) ? 1 : 0;
  const char *we = getenv("SGL_CUDA_W8");
  int w8 = (we && atoi(we)) ? 1 : 0;
  if (w8) fp16 = 1; /* embed stays fp16, projections go int8 */
  double wsz = fp16 ? 2.0 : 4.0;   /* embed / lm_head element size */
  double psz = w8 ? 1.0 : wsz;     /* projection element size */
  double ei = (double)(E > I ? E : I);
  double bytes = 0;
  bytes += (double)V * E * wsz;                          /* wte */
  if (m->lm_head.data && m->lm_head.data != m->wte.data) /* untied lm_head */
    bytes += (double)V * E * wsz;
  bytes += (double)L * (2.0 * E * E + 2.0 * KD * E + 3.0 * I * E) * psz;
  bytes += (double)L * (6.0 * E + 2.0 * I) * 4.0;        /* norms + biases fp32 */
  bytes += 2.0 * L * ctx * KD * 4.0;                     /* KV pool */
  bytes += (4.0 * E + 2.0 * KD + 2.0 * I) * rows * 4.0;  /* activation workspace */
  if (fp16) bytes += ei * rows * 2.0;                    /* cublas fp16 input mirror */
  if (w8) bytes += ei * rows * 9.0;                      /* int8 A + int32 C + scales */
  bytes += (double)V * 4.0 * 33.0;                       /* d_lgb grows to 32*B rows */
  bytes += 32.0 * 1048576.0;                             /* cublas workspace */
  bytes += 320.0 * 1048576.0;                            /* context + loaded modules */
  return (long)(bytes / 1048576.0);
}

long sgl_cuda_avail_vram_mb(void) {
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  return (long)(free_b / 1048576);
}

int sgl_cuda_forward(sgl_cuda_model *g, const int *tokens, int n, int pos0,
                     float *logits_out) {
  return sgl_cuda_forward_seg(g, tokens, n, pos0, 0, logits_out);
}

/* t4: GPT-2 block forward (embed+pos / LN+bias / fused qkv / shared-scores
 * attention / GELU MLP / tied lm_head). KV rows scatter from the fused qkv
 * buffer via 2D memcpy (row layout [q|k|v], src pitch 3E). */
static int gpt2_forward_seg(sgl_cuda_model *g, const int *tokens, int n, int pos0,
                            int kv_base, float *logits_out) {
  int E = g->E, H = g->H, D = g->D, KD = g->KD, E3 = 3 * E;
  if (n <= 0 || pos0 + n > g->ctx) return -1;
  if (kv_base + pos0 + n > g->ctx) return -1;
  if (sgl_cuda_reserve(g, n) != 0) return -1;
  if (sgl_cuda_ensure_io(g, n) != 0) return -1;

  memcpy(g->h_tok, tokens, sizeof(int) * (size_t)n);
  CK(cudaMemcpyAsync(g->d_tok, g->h_tok, sizeof(int) * (size_t)n,
                     cudaMemcpyHostToDevice, g->stream));
  if (g->use_fp16)
    k_embed_pos_h<<<(int)((n * (long)E + 255) / 256), 256, 0, g->stream>>>(
        g->wte_h, g->wpe, g->d_tok, g->x, n, E, pos0);
  else
    k_embed_pos<<<(int)((n * (long)E + 255) / 256), 256, 0, g->stream>>>(
        g->wte, g->wpe, g->d_tok, g->x, n, E, pos0);
  CK(cudaGetLastError());

  for (int l = 0; l < g->L; l++) {
    k_layernorm_b<<<n, 256, 256 * sizeof(float), g->stream>>>(
        g->x, g->ln1[l], g->lnb1[l], g->h, n, E, g->eps);
    CK(cudaGetLastError());
    /* fused qkv: [n,3E] lands in the q slot */
    PROJ(g, g->h, g->qw[l], g->qw_h ? g->qw_h[l] : NULL, g->qw8 ? &g->qw8[l] : NULL, g->qkvb[l],
         g->q, n, E3, E);
    CK(cudaGetLastError());
    /* KV scatter: k/v rows are at +E/+2E inside each 3E-stride qkv row */
    {
      float *kcl = g->kc + (long)l * g->ctx * KD;
      float *vcl = g->vc + (long)l * g->ctx * KD;
      size_t row = (size_t)KD * sizeof(float);
      size_t src_pitch = (size_t)E3 * sizeof(float);
      size_t dst_off = (size_t)(kv_base + pos0) * row;
      CK(cudaMemcpy2DAsync((char *)kcl + dst_off, row, (char *)g->q + E * sizeof(float),
                           src_pitch, row, n, cudaMemcpyDeviceToDevice, g->stream));
      CK(cudaMemcpy2DAsync((char *)vcl + dst_off, row, (char *)g->q + 2 * E * sizeof(float),
                           src_pitch, row, n, cudaMemcpyDeviceToDevice, g->stream));
    }
    k_attn<<<n * H, 256, (size_t)(pos0 + n) * sizeof(float), g->stream>>>(
        g->q, g->kc + (long)l * g->ctx * KD, g->vc + (long)l * g->ctx * KD,
        g->attn, pos0, kv_base, n, H, H, D, 1.0f / sqrtf((float)D), E3,
        g->sliding_window);
    CK(cudaGetLastError());
    PROJ(g, g->attn, g->ow[l], g->ow_h ? g->ow_h[l] : NULL, g->ow8 ? &g->ow8[l] : NULL, g->ob[l],
         g->h, n, E, E);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)n * E + 255) / 256), 256, 0, g->stream>>>(g->x, g->h, (long)n * E);
    k_layernorm_b<<<n, 256, 256 * sizeof(float), g->stream>>>(
        g->x, g->ln2[l], g->lnb2[l], g->h, n, E, g->eps);
    CK(cudaGetLastError());
    PROJ(g, g->h, g->gw[l], g->gw_h ? g->gw_h[l] : NULL, g->gw8 ? &g->gw8[l] : NULL, g->fcb[l],
         g->gate, n, g->I, E);
    CK(cudaGetLastError());
    k_gelu<<<(int)(((long)n * g->I + 255) / 256), 256, 0, g->stream>>>(g->gate, (long)n * g->I);
    PROJ(g, g->gate, g->dw[l], g->dw_h ? g->dw_h[l] : NULL, g->dw8 ? &g->dw8[l] : NULL, g->mb[l],
         g->h, n, E, g->I);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)n * E + 255) / 256), 256, 0, g->stream>>>(g->x, g->h, (long)n * E);
    CK(cudaGetLastError());
  }
  k_layernorm_b<<<n, 256, 256 * sizeof(float), g->stream>>>(
      g->x, g->final_norm, g->lnf_b, g->h, n, E, g->eps);
  CK(cudaGetLastError());
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
  return 0;
}

/* t10 llava: shared llama forward — tokens==NULL takes emb ([n][E] host,
 * pre-fused visual embeddings) and skips the embed lookup; same layer loop.
 * G7: all_rows!=0 returns every position's logits in logits_out ([n][V],
 * row i = prediction after consuming tokens[0..i]) via ONE M=n lm_head GEMM;
 * otherwise logits_out is the single last-position row (original behavior). */
static int fwd_seg_llama(sgl_cuda_model *g, const int *tokens, const float *emb,
                         int n, int pos0, int kv_base, float *logits_out,
                         int all_rows) {
  int E = g->E, H = g->H, HK = g->HK, D = g->D, KD = g->KD;
  if (n <= 0 || pos0 + n > g->ctx) return -1;
  if (kv_base + pos0 + n > g->ctx) return -1;
  if (sgl_cuda_reserve(g, n) != 0) return -1;
  if (sgl_cuda_ensure_io(g, n) != 0) return -1;

  /* G1 profiling: event-pair every attention launch (see struct comment). */
  {
    static int prof_env = -1;
    if (prof_env < 0) {
      const char *pe = getenv("SGL_CUDA_PROF");
      prof_env = (pe && atoi(pe)) ? 1 : 0;
    }
    g->prof = (prof_env && g->L <= SGL_PROF_MAXL) ? 1 : 0;
    if (g->prof && g->pf0[0] == NULL) {
      for (int l = 0; l < g->L; l++) {
        cudaEventCreate(&g->pf0[l]);
        cudaEventCreate(&g->pf1[l]);
      }
    }
  }
  const int prof = g->prof;

  if (emb) {
    /* visual tokens: activations come precomputed from the host (CLIP+projector) */
    CK(cudaMemcpyAsync(g->x, emb, sizeof(float) * (size_t)n * E,
                       cudaMemcpyHostToDevice, g->stream));
  } else {
    /* input tokens go through the pinned mirror + async copy on g->stream
     * (same ordering domain as the kernels; no per-step cudaMalloc) */
    memcpy(g->h_tok, tokens, sizeof(int) * (size_t)n);
    CK(cudaMemcpyAsync(g->d_tok, g->h_tok, sizeof(int) * (size_t)n,
                       cudaMemcpyHostToDevice, g->stream));
    if (g->use_fp16)
      k_embed_h<<<(int)((n * (long)E + 255) / 256), 256, 0, g->stream>>>(g->wte_h, g->d_tok, g->x, n, E);
    else
      k_embed<<<(int)((n * (long)E + 255) / 256), 256, 0, g->stream>>>(g->wte, g->d_tok, g->x, n, E);
  }
  CK(cudaGetLastError());

  for (int l = 0; l < g->L; l++) {
    k_rmsnorm<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->ln1[l], g->h, n, E, g->eps);
    CK(cudaGetLastError());
    PROJ(g, g->h, g->qw[l], g->qw_h ? g->qw_h[l] : NULL, g->qw8 ? &g->qw8[l] : NULL, g->qb[l],
         g->q, n, E, E);
    PROJ(g, g->h, g->kw[l], g->kw_h ? g->kw_h[l] : NULL, g->kw8 ? &g->kw8[l] : NULL, g->kb[l],
         g->k, n, KD, E);
    PROJ(g, g->h, g->vw[l], g->vw_h ? g->vw_h[l] : NULL, g->vw8 ? &g->vw8[l] : NULL, g->vb[l],
         g->v, n, KD, E);
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
    /* M26a/G16: attention path split —— 具体选哪个核见 attn_dispatch()。
     * SGL_FORCE_LONG_ATTN=1 把所有 prefill 都丢给长核（验证两条路径一致）。
     * SGL_ATTN_TWOPASS=1 强制旧两遍核（A/B 基准）。
     * SGL_CUDA_PROF=1 给每层 attention 包 CUDA event（G1 的耗时分解）。*/
    {
      const char *fe = getenv("SGL_FORCE_LONG_ATTN");
      const char *tp = getenv("SGL_ATTN_TWOPASS");
      int force_long = (fe && atoi(fe)) ? 1 : 0;
      int two_pass = (tp && atoi(tp)) ? 1 : 0;
      float sc = 1.0f / sqrtf((float)D);
      if (prof && g->L <= SGL_PROF_MAXL) {
        cudaEventRecord(g->pf0[l], g->stream);
        attn_dispatch(g, l, n, pos0, kv_base, sc, two_pass, force_long, E);
        cudaEventRecord(g->pf1[l], g->stream);
      } else {
        attn_dispatch(g, l, n, pos0, kv_base, sc, two_pass, force_long, E);
      }
    }
    CK(cudaGetLastError());
    PROJ(g, g->attn, g->ow[l], g->ow_h ? g->ow_h[l] : NULL, g->ow8 ? &g->ow8[l] : NULL, NULL,
         g->h, n, E, E);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)n * E + 255) / 256), 256, 0, g->stream>>>(g->x, g->h, (long)n * E);
    k_rmsnorm<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->ln2[l], g->h, n, E, g->eps);
    PROJ(g, g->h, g->gw[l], g->gw_h ? g->gw_h[l] : NULL, g->gw8 ? &g->gw8[l] : NULL, NULL,
         g->gate, n, g->I, E);
    PROJ(g, g->h, g->uw[l], g->uw_h ? g->uw_h[l] : NULL, g->uw8 ? &g->uw8[l] : NULL, NULL,
         g->up, n, g->I, E);
    CK(cudaGetLastError());
    k_silu_mul<<<(int)(((long)n * g->I + 255) / 256), 256, 0, g->stream>>>(g->gate, g->up, (long)n * g->I);
    PROJ(g, g->gate, g->dw[l], g->dw_h ? g->dw_h[l] : NULL, g->dw8 ? &g->dw8[l] : NULL, NULL,
         g->h, n, E, g->I);
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
      int rows = all_rows ? n : 1;
      const float *hbase = all_rows ? g->h : g->h + (long)(n - 1) * E;
      if (sgl_cuda_ensure_lgb(g, rows) != 0) return -1;
      gemm_nt_any(g, hbase, g->lm_head, g->lm_head_h, NULL,
                  g->d_lgb, rows, g->V, E);
      CK(cudaGetLastError());
      CK(cudaMemcpyAsync(g->h_lgb, g->d_lgb, sizeof(float) * (size_t)rows * g->V,
                         cudaMemcpyDeviceToHost, g->stream));
      CK(cudaStreamSynchronize(g->stream));
      memcpy(logits_out, g->h_lgb, sizeof(float) * (size_t)rows * g->V);
    }
  }
  if (prof) {
    CK(cudaStreamSynchronize(g->stream));
    float tot = 0.0f;
    for (int l = 0; l < g->L; l++) {
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, g->pf0[l], g->pf1[l]);
      tot += ms;
    }
    printf("[prof] attention %.1f ms / %d layers (n=%d pos0=%d kv_base=%d)\n",
           tot, g->L, n, pos0, kv_base);
    fflush(stdout);
  }
  return 0;
}

int sgl_cuda_forward_seg(sgl_cuda_model *g, const int *tokens, int n, int pos0,
                         int kv_base, float *logits_out) {
  if (g->kind == 1) return gpt2_forward_seg(g, tokens, n, pos0, kv_base, logits_out);
  return fwd_seg_llama(g, tokens, NULL, n, pos0, kv_base, logits_out, 0);
}

/* G7: verify forward — every one of the n positions gets its own logits row
 * (single M=n lm_head GEMM). See cuda.h for row semantics. */
int sgl_cuda_forward_seg_rows(sgl_cuda_model *g, const int *tokens, int n,
                              int pos0, int kv_base, float *logits_out) {
  if (!logits_out || n <= 0) return -1;
  if (g->kind == 1) return -1; /* llama backend only */
  return fwd_seg_llama(g, tokens, NULL, n, pos0, kv_base, logits_out, 1);
}

/* t10 llava: pre-fused visual embeddings [n][E] straight into the layer loop
 * (llama path only; gpt2 returns -1). */
int sgl_cuda_forward_seg_emb(sgl_cuda_model *g, const float *emb, int n, int pos0,
                             int kv_base, float *logits_out) {
  if (g->kind == 1) return -1;
  return fwd_seg_llama(g, NULL, emb, n, pos0, kv_base, logits_out, 0);
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
    PROJ(g, g->h, g->qw[l], g->qw_h ? g->qw_h[l] : NULL, g->qw8 ? &g->qw8[l] : NULL, g->qb[l],
         g->q, B, E, E);
    PROJ(g, g->h, g->kw[l], g->kw_h ? g->kw_h[l] : NULL, g->kw8 ? &g->kw8[l] : NULL, g->kb[l],
         g->k, B, KD, E);
    PROJ(g, g->h, g->vw[l], g->vw_h ? g->vw_h[l] : NULL, g->vw8 ? &g->vw8[l] : NULL, g->vb[l],
         g->v, B, KD, E);
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
          g->attn, g->d_base, g->d_lens, B, H, HK, D, 1.0f / sqrtf((float)D),
          g->sliding_window);
    } else {
      k_attn_batch<<<B * H, 256, (size_t)maxu * sizeof(float), s>>>(
          g->q, g->kc + (long)l * g->ctx * KD, g->vc + (long)l * g->ctx * KD,
          g->attn, g->d_base, g->d_lens, B, H, HK, D, 1.0f / sqrtf((float)D),
          g->sliding_window);
    }
    CK(cudaGetLastError());
    PROJ(g, g->attn, g->ow[l], g->ow_h ? g->ow_h[l] : NULL, g->ow8 ? &g->ow8[l] : NULL, NULL,
         g->h, B, E, E);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)B * E + 255) / 256), 256, 0, s>>>(g->x, g->h, (long)B * E);
    k_rmsnorm<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->ln2[l], g->h, B, E, g->eps);
    PROJ(g, g->h, g->gw[l], g->gw_h ? g->gw_h[l] : NULL, g->gw8 ? &g->gw8[l] : NULL, NULL,
         g->gate, B, g->I, E);
    PROJ(g, g->h, g->uw[l], g->uw_h ? g->uw_h[l] : NULL, g->uw8 ? &g->uw8[l] : NULL, NULL,
         g->up, B, g->I, E);
    CK(cudaGetLastError());
    k_silu_mul<<<(int)(((long)B * g->I + 255) / 256), 256, 0, s>>>(g->gate, g->up, (long)B * g->I);
    PROJ(g, g->gate, g->dw[l], g->dw_h ? g->dw_h[l] : NULL, g->dw8 ? &g->dw8[l] : NULL, NULL,
         g->h, B, E, g->I);
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

/* G3: capture one decode-step graph for batch size B.
 * Two stability fixes over the original inline capture:
 *  (1) WARM-UP RUN before capturing. With CUDA_MODULE_LOADING=LAZY (the CUDA
 *      13 default) the first launch of a kernel loads its module, and cuBLAS
 *      picks (and may allocate for) an algorithm on its first call per shape;
 *      doing that inside stream capture is what made captures fail
 *      intermittently. One plain step first, then capture.
 *  (2) PER-B FAILURE. A failed capture used to set a process-global
 *      "graphs off" flag, so one unlucky capture degraded every batch size for
 *      the rest of the server's life (the "capture fails then decode crawls"
 *      symptom). Now only this B is marked SGL_NO_GRAPH; other batch sizes
 *      keep replaying their graphs, and a later buffer growth
 *      (invalidate_graphs) clears the marker so B gets another chance. */
static int capture_decode_graph(sgl_cuda_model *g, int B) {
  int H = g->H, HK = g->HK;
  cudaGraph_t graph = NULL;
  cudaGraphExec_t exec = NULL;
  if (B < 1 || B > 32) return -1;
  if (sgl_cuda_reserve(g, B) != 0) return -1;
  if (sgl_cuda_ensure_io(g, B) != 0) return -1;
  if (sgl_cuda_ensure_lgb(g, B) != 0) return -1;
  CK(cudaStreamSynchronize(g->stream));
  /* (1) warm-up: a full plain step with the mirrors as they are. The device
   * mirrors must be loaded from the host mirrors FIRST: ensure_io() only
   * cudaMallocs them (uninitialized), and launching with garbage base_pos
   * reads KV out of range -> illegal access (this is what the original
   * capture avoided by never executing the step during capture). */
  if (cudaMemcpyAsync(g->d_tok, g->h_tok, sizeof(int) * (size_t) B,
                      cudaMemcpyHostToDevice, g->stream) != cudaSuccess) return -1;
  if (cudaMemcpyAsync(g->d_base, g->h_base, sizeof(int) * (size_t) B,
                      cudaMemcpyHostToDevice, g->stream) != cudaSuccess) return -1;
  if (cudaMemcpyAsync(g->d_lens, g->h_lens, sizeof(int) * (size_t) B,
                      cudaMemcpyHostToDevice, g->stream) != cudaSuccess) return -1;
  if (cudaMemcpyAsync(g->d_pq, g->h_pq, sizeof(int) * (size_t) B * H,
                      cudaMemcpyHostToDevice, g->stream) != cudaSuccess) return -1;
  if (cudaMemcpyAsync(g->d_pk, g->h_pk, sizeof(int) * (size_t) B * HK,
                      cudaMemcpyHostToDevice, g->stream) != cudaSuccess) return -1;
  if (decode_step_launch(g, B, g->ctx + 1) != 0) return -1;
  if (cudaMemcpyAsync(g->h_lgb, g->d_lgb, sizeof(float) * (size_t)B * g->V,
                      cudaMemcpyDeviceToHost, g->stream) != cudaSuccess) return -1;
  if (cudaStreamSynchronize(g->stream) != cudaSuccess) return -1;

  if (cudaStreamBeginCapture(g->stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    g->gexec[B] = SGL_NO_GRAPH;
    return -1;
  }
  /* the input upload from the pinned mirrors is part of the graph: every
   * replay re-copies whatever the caller wrote into the mirrors */
  int rc = 0;
  rc |= cudaMemcpyAsync(g->d_tok, g->h_tok, sizeof(int) * (size_t)B,
                        cudaMemcpyHostToDevice, g->stream) == cudaSuccess ? 0 : -1;
  rc |= cudaMemcpyAsync(g->d_base, g->h_base, sizeof(int) * (size_t)B,
                        cudaMemcpyHostToDevice, g->stream) == cudaSuccess ? 0 : -1;
  rc |= cudaMemcpyAsync(g->d_lens, g->h_lens, sizeof(int) * (size_t)B,
                        cudaMemcpyHostToDevice, g->stream) == cudaSuccess ? 0 : -1;
  rc |= cudaMemcpyAsync(g->d_pq, g->h_pq, sizeof(int) * (size_t)B * H,
                        cudaMemcpyHostToDevice, g->stream) == cudaSuccess ? 0 : -1;
  rc |= cudaMemcpyAsync(g->d_pk, g->h_pk, sizeof(int) * (size_t)B * HK,
                        cudaMemcpyHostToDevice, g->stream) == cudaSuccess ? 0 : -1;
  rc |= decode_step_launch(g, B, g->ctx + 1 /* force long-path kernel */);
  if (rc == 0)
    rc = cudaMemcpyAsync(g->h_lgb, g->d_lgb, sizeof(float) * (size_t)B * g->V,
                         cudaMemcpyDeviceToHost, g->stream) == cudaSuccess ? 0 : -1;
  if (rc != 0) {
    cudaStreamEndCapture(g->stream, &graph); /* abort the capture */
    if (graph) cudaGraphDestroy(graph);
    g->gexec[B] = SGL_NO_GRAPH;
    fprintf(stderr, "[cuda] decode graph capture failed (B=%d); this size runs plain\n", B);
    cudaGetLastError(); /* clear any sticky error from the aborted capture */
    return -1;
  }
  if (cudaStreamEndCapture(g->stream, &graph) != cudaSuccess || !graph) {
    if (graph) cudaGraphDestroy(graph);
    g->gexec[B] = SGL_NO_GRAPH;
    fprintf(stderr, "[cuda] decode graph end-capture failed (B=%d)\n", B);
    cudaGetLastError();
    return -1;
  }
  if (cudaGraphInstantiate(&exec, graph, NULL, NULL, 0) != cudaSuccess) {
    cudaGraphDestroy(graph);
    g->gexec[B] = SGL_NO_GRAPH;
    fprintf(stderr, "[cuda] decode graph instantiate failed (B=%d)\n", B);
    cudaGetLastError();
    return -1;
  }
  g->graphs[B] = graph;
  g->gexec[B] = exec;
  size_t nn = 0;
  cudaGraphGetNodes(graph, NULL, &nn);
  printf("[cuda] decode graph captured (B=%d, %d nodes)\n", B, (int)nn);
  fflush(stdout); /* server log must show captures even when piped */
  return 0;
}

/* G3: pre-warm everything the decode graphs bake in, then capture the graphs
 * for B = 1..max_b. Call once at startup (before serving): the activation
 * workspace then never grows (rows = the largest prefill chunk the server will
 * issue), so the cached graphs stay valid for the whole server lifetime, and
 * the ~0.5s capture cost per B is paid before the first request instead of
 * inside a live batch. Returns the number of graphs captured, or -1. */
int sgl_cuda_prewarm(sgl_cuda_model *g, int rows, int max_b) {
  if (!g) return -1;
  if (rows < 1) rows = 1;
  if (max_b < 1) max_b = 1;
  if (max_b > 32) max_b = 32;
  if (sgl_cuda_reserve(g, rows) != 0) return -1;
  if (sgl_cuda_ensure_io(g, max_b) != 0) return -1;
  if (sgl_cuda_ensure_lgb(g, max_b) != 0) return -1;
  /* mirrors: any values in range; the capture only reads the addresses */
  for (int b = 0; b < max_b; b++) {
    g->h_tok[b] = 1;
    g->h_base[b] = b * (g->ctx / max_b);
    g->h_lens[b] = 16;
    for (int h = 0; h < g->H; h++) g->h_pq[b * g->H + h] = 16;
    for (int hk = 0; hk < g->HK; hk++) g->h_pk[b * g->HK + hk] = 16;
    if (g->h_base[b] + 17 >= g->ctx) g->h_base[b] = g->ctx - 18;
    if (g->h_base[b] < 0) g->h_base[b] = 0;
  }
  int got = 0;
  for (int B = 1; B <= max_b; B++) {
    if (!g->gexec[B]) {
      if (capture_decode_graph(g, B) == 0) got++;
    } else got++;
  }
  /* Prefill warm-up too: the first real prompt otherwise pays the lazy kernel
   * load of the prefill-only kernels (rope rows / short + flash attention /
   * silu) and cuBLAS algorithm selection for the M>1 shapes — measured as a
   * ~20s first round under 12-way concurrency vs ~4s afterwards. Any KV the
   * warm-up writes is overwritten by the first real prefill of that segment,
   * and attention only reads [kv_base, kv_base+len) of the requesting
   * sequence, so the garbage rows can never be attended. */
  {
    int wn = rows < 64 ? rows : 64;
    int *wt = (int *)malloc(sizeof(int) * (size_t)wn);
    if (wt) {
      float *wl = (float *)malloc(sizeof(float) * (size_t)g->V);
      for (int i = 0; i < wn; i++) wt[i] = 1 + (i % 100);
      if (wl) {
        if (sgl_cuda_forward_seg(g, wt, wn, 0, 0, wl) != 0)
          fprintf(stderr, "[cuda] prewarm prefill failed (rows=%d)\n", wn);
        free(wl);
      }
      free(wt);
    }
  }
  CK(cudaStreamSynchronize(g->stream));
  return got;
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
      if (capture_decode_graph(g, B) != 0) {
        /* (G3) only this B gave up; fall through to the plain path below */
      }
    }
    if (g->gexec[B] && g->gexec[B] != SGL_NO_GRAPH) {
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
    int8_t *Wi8 = NULL, *Ai8 = NULL; /* t8: IMMA vs dp4a bench buffers */
    int *Ci32 = NULL;
    float *Wsc = NULL, *Asc = NULL;
    if (cudaMalloc(&Wf, (size_t)N * K * 4) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Wh, (size_t)N * K * 2) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Af, (size_t)MMAX * K * 4) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Ah, (size_t)MMAX * K * 2) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Cf, (size_t)MMAX * N * 4) != cudaSuccess) goto next_shape;
    if (cudaMallocHost(&Af_h, (size_t)MMAX * K * 2) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Wi8, (size_t)N * K) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Ai8, (size_t)MMAX * K) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Ci32, (size_t)MMAX * N * 4) != cudaSuccess) goto next_shape;
    if (cudaMalloc(&Wsc, (size_t)N * 4) != cudaSuccess) goto next_shape;
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
      /* t8: per-channel int8 W + per-tensor int8 A (bench-only quantization) */
      {
        int8_t *qw = (int8_t *)malloc((size_t)N * K);
        int8_t *qa = (int8_t *)malloc((size_t)MMAX * K);
        float *sc = (float *)malloc((size_t)N * 4);
        for (int n = 0; n < N; n++) {
          float am = 0;
          for (int k2 = 0; k2 < K; k2++) {
            float v = tw[(size_t)n * K + k2];
            if (v > am) am = v; else if (-v > am) am = -v;
          }
          float inv = am > 1e-8f ? 127.0f / am : 0;
          sc[n] = am / 127.0f;
          for (int k2 = 0; k2 < K; k2++) {
            float v = tw[(size_t)n * K + k2] * inv;
            qw[(size_t)n * K + k2] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
          }
        }
        float aam = 0;
        for (long i = 0; i < (long)MMAX * K; i++) {
          float v = ta[i];
          if (v > aam) aam = v; else if (-v > aam) aam = -v;
        }
        float ainv = aam > 1e-8f ? 127.0f / aam : 0;
        for (long i = 0; i < (long)MMAX * K; i++) {
          float v = ta[i] * ainv;
          qa[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
        cudaMemcpy(Wi8, qw, (size_t)N * K, cudaMemcpyHostToDevice);
        cudaMemcpy(Ai8, qa, (size_t)MMAX * K, cudaMemcpyHostToDevice);
        cudaMemcpy(Wsc, sc, (size_t)N * 4, cudaMemcpyHostToDevice);
        free(qw); free(qa); free(sc);
      }
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
      /* t8: int8 tensor core (cublasGemmEx IMMA) vs hand dp4a kernel */
      if (K % 4 == 0 && N % 4 == 0) {
        int32_t ai = 1, bi = 0;
        float ms_imma = 1e30f;
        for (int it = -WARM; it < ITERS; it++) {
          cudaEventRecord(ev0);
          cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &ai,
                       Wi8, CUDA_R_8I, K, Ai8, CUDA_R_8I, K, &bi,
                       Ci32, CUDA_R_32I, N, CUBLAS_COMPUTE_32I,
                       CUBLAS_GEMM_DEFAULT);
          cudaEventRecord(ev1);
          cudaEventSynchronize(ev1);
          float ms;
          cudaEventElapsedTime(&ms, ev0, ev1);
          if (it >= 0 && ms < ms_imma) ms_imma = ms;
        }
        float ms_dp4a = 1e30f;
        for (int it = -WARM; it < ITERS; it++) {
          cudaEventRecord(ev0);
          k_gemm_w8a8<<<dim3(N, M), 256>>>(Af, Wi8, Wsc, NULL, Cf, M, N, K);
          cudaEventRecord(ev1);
          cudaEventSynchronize(ev1);
          float ms;
          cudaEventElapsedTime(&ms, ev0, ev1);
          if (it >= 0 && ms < ms_dp4a) ms_dp4a = ms;
        }
        printf("[gemm-bench] %-28s M=%3d  int8 imma %8.4f ms  dp4a %8.4f ms (%.2fx)\n",
               shapes[si].name, M, ms_imma, ms_dp4a, ms_imma / ms_dp4a);
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
    if (Wi8) cudaFree(Wi8);
    if (Ai8) cudaFree(Ai8);
    if (Ci32) cudaFree(Ci32);
    if (Wsc) cudaFree(Wsc);
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  cublasDestroy(h);
  fflush(stdout);
}

