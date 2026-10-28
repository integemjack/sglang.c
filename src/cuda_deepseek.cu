/* sgl cuda_deepseek.cu — DeepSeek 系（MLA + MoE）的 GPU 前向
 *
 * 目标：与 CPU 引擎（src/deepseek.c）**同一套公式、同一套 f32 口径**，只是把
 * 逐行标量循环换成 cublas SGEMM + 并行 kernel。所以两者只差累加顺序：
 * 测试拿 CPU 当参考比 logits（见 tests/cuda_deepseek_test.c）。
 *
 * 与 llama 的 GPU 路径（cuda_backend.cu）刻意分开，原因见 include/sgl/cuda_deepseek.h：
 * MLA 的 KV cache 行是 [latent | k_pe]（stride = kv_lora + qk_rope，与 head 数无关），
 * 每头 q 是 [qk_nope + qk_rope]、value 只有 kv_lora 维 —— 和 llama 的 [ctx][KD]
 * 布局正交，混在一起只会互相污染。
 *
 * 吸收式（absorbed）MLA，和 CPU 侧完全一致：
 *   q_abs[h] = W_UK[h]ᵀ q_nope[h]            （W_UK[h] = kv_b_proj 里该头 k_nope 那几行）
 *   score_p  = (q_abs[h]·c_p + q_pe[h]·k_pe_p) * softmax_scale
 *   o_lat[h] = Σ_p a_p c_p                    （c 就是 cache 行前 kv_lora 维）
 *   o[h]     = W_UV[h] o_lat[h]               （W_UV[h] = kv_b_proj 里该头 v 那几行）
 * kv_b_proj 在上传时就被拆成 W_UK / W_UV，推理期不再需要它。
 *
 * MoE 按 HF 的 DeepseekV2/V3 路由公式（softmax 或 sigmoid、e_score_correction_bias、
 * group_limited_greedy、topk 权重取未修正打分、norm_topk_prob、routed_scaling_factor、
 * 共享专家直加），与 CPU 的 ds_moe_row_ 逐条对应。
 *
 * 路由需要"把 token 按专家分组"，本实现的做法：routing 结果拷回主机 → 主机按专家
 * 排序（n*topk 个元素，几微秒）→ 一次 k_gather_rows 把行复制成连续块 → 每专家一次
 * cublas GEMM → k_scatter_moe 用 atomicAdd 按权重累加回行。这是标准的 gather/scatter
 * MoE；省事且正确，代价是每层一次 D2H 同步（后续可以改成设备端排序避免）。
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/cuda_deepseek.h"
/* 这几个是纯 C 头（没有 extern "C"外衣）：nvcc 把 .cu 当 C++ 编译，
 * 包一层才能链到 C 库里未变形的符号（sgl_xmalloc / sgl_tensor_numel …）。*/
extern "C" {
#include "sgl/alloc.h"
#include "sgl/deepseek.h"
#include "sgl/log.h"
#include "sgl/port.h"
}

#define CK(x)                                                                        \
  do {                                                                               \
    cudaError_t _e = (x);                                                             \
    if (_e != cudaSuccess) {                                                           \
      SGL_ERROR("cuda %s:%d %s -> %s", __FILE__, __LINE__, #x, cudaGetErrorString(_e)); \
      return -1;                                                                       \
    }                                                                                  \
  } while (0)

#define BLAS_OK(x)                                                                   \
  do {                                                                               \
    cublasStatus_t _s = (x);                                                          \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                                 \
      SGL_ERROR("cublas %s:%d %s -> %d", __FILE__, __LINE__, #x, (int)_s);             \
      return -1;                                                                       \
    }                                                                                  \
  } while (0)

struct sgl_cuda_ds {
  int L, E, H, Dn, Dr, Dc, Dv, qhd, stride, V, ctx, I;
  int q_lora;
  int NE, topk, mi, n_shared, sh_inter;
  int norm_topk, sigmoid_score, group_limited, n_group, topk_group;
  int rope_interleave, yarn;
  float routed_scale, eps, sm_scale, rope_attn_factor;

  /* ---- 权重（设备）---- */
  float *wte, *lm_head, *final_norm;
  float **inp_norm, **post_norm;
  float **q_a_w, **q_a_norm, **q_b_w, **q_a_b;   /* q_a_* 仅 q_lora>0 时有效 */
  float **kv_a_w, **kv_a_norm, **kv_a_b;
  float **w_ukt; /* [H][Dc][Dn]：kv_b 的 k_nope 段转置（为了用 gemm_nt）*/
  float **w_uv;  /* [H][Dv][Dc]：kv_b 的 v 段，原样 */
  float **o_w, **o_b;
  int *is_moe;
  float **g_w, **u_w, **d_w;        /* dense MLP（[mi/E..]）*/
  float **router_w, **ec_bias;      /* MoE 路由 */
  float **exp_gu, **exp_dn;         /* 3D 融合布局：[NE][2*mi][E] / [NE][E][mi] */
  float ***e_gate, ***e_up, ***e_dn; /* 逐专家布局：[层][专家] 指向各自 [mi][E] / [E][mi] */
  float **sh_g, **sh_u, **sh_d;     /* 共享专家 */

  /* ---- 激活/临时（按 cap 行分配）---- */
  int cap;
  float *x, *h, *q, *kva, *qa, *lat, *qn, *qpe, *qabs, *olat, *obuf, *att;
  float *gu, *upd, *mlp, *route;
  int *d_tok;                        /* [cap] 输入 token（避免每步 cudaMalloc）*/
  float *gath, *eout;                /* MoE gather/scatter 缓冲（cap*topk 行）*/
  int *d_sel;                        /* [cap*topk] 选中的专家号 */
  float *d_wsel;                     /* [cap*topk] 对应权重 */
  int *d_idx;                        /* [cap*topk] gather 用行号 */
  float *h_wsel;                     /* pinned host，routing 结果镜像 */
  int *h_sel;
  float *lat_pin;                    /* 小缓冲的 pinned 主体不需要 */

  float *kc; /* [L][ctx][stride] */
  float *invf; /* [Dr/2]（设备不需要，kernel 直接用主机表拷过去）*/
  float *d_invf;

  float *d_lgb, *h_lgb; /* logits */
  int lgb_cap;

  cublasHandle_t blas;
  cudaStream_t stream;
  size_t mem_bytes;
};

/* ------------------------------ kernels ------------------------------ */

__global__ void k_ds_embed(const float *wte, const int *tok, float *x, int n, int E) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)n * E;
  if (i >= tot) return;
  int t = (int)(i / E), e = (int)(i % E);
  x[i] = wte[(size_t)tok[t] * E + e];
}

/* y[r][:] = x[r][:] / rms(x[r]) * w[:]；行步长 stride（用于从 kva 里抽 latent）*/
__global__ void k_ds_rmsnorm(const float *x, const float *w, float *y, int rows, int N,
                             int xstride, int ystride, float eps) {
  int r = blockIdx.x;
  const float *xr = x + (size_t)r * xstride;
  float *yr = y + (size_t)r * ystride;
  float s = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) s += xr[i] * xr[i];
  __shared__ float red[256];
  red[threadIdx.x] = s;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
    __syncthreads();
  }
  float rms = sqrtf(red[0] / (float)N + eps);
  for (int i = threadIdx.x; i < N; i += blockDim.x) yr[i] = xr[i] / rms * w[i];
}

/* MLA 的 RoPE：只旋转每行 [off, off+Dr) 这一段。
 * rows 里每 groups 行属于同一个位置（q 是 n*H 行、H 个一头一位置之外交错，
 * k_pe 是 n 行 groups=1）。interleave=1 时按 (2i, 2i+1) 配对（HF V3 的交错式）。*/
__global__ void k_ds_rope(float *base, int rows, int rowstride, int off, int Dr, int pos0,
                          int groups, const float *invf, int interleave, float scale) {
  int r = blockIdx.x;
  int pos = pos0 + (groups > 0 ? r / groups : r);
  float *row = base + (size_t)r * rowstride + off;
  int half = Dr / 2;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    double ang = (double)pos * invf[i];
    float c = (float)cos(ang) * scale, s = (float)sin(ang) * scale;
    int a_i = interleave ? 2 * i : i;
    int b_i = interleave ? 2 * i + 1 : i + half;
    float a = row[a_i], b = row[b_i];
    row[a_i] = a * c - b * s;
    row[b_i] = a * s + b * c;
  }
}

/* q [n][H][qhd] → qn [H][n][Dn] + qpe [H][n][Dr]（每头连续，便于后面 cublas 按头跑）*/
__global__ void k_ds_pack_q(const float *q, float *qn, float *qpe, int n, int H, int qhd,
                            int Dn, int Dr) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)n * H;
  if (i >= tot) return;
  int t = (int)(i / H), h = (int)(i % H);
  const float *src = q + ((size_t)t * H + h) * qhd;
  float *dn = qn + ((size_t)h * n + t) * Dn;
  float *dr = qpe + ((size_t)h * n + t) * Dr;
  for (int k = threadIdx.y; k < Dn; k += blockDim.y) dn[k] = src[k];
  for (int k = threadIdx.y; k < Dr; k += blockDim.y) dr[k] = src[Dn + k];
}

/* 写压缩 KV 行：[latent | k_pe]（latent 已过 kv_a_layernorm）*/
__global__ void k_ds_kv_write(float *kcl, const float *lat, const float *kva, int n, int pos0,
                              int Dc, int Dr, int stride) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)n * (Dc + Dr);
  if (i >= tot) return;
  int t = (int)(i / (Dc + Dr)), d = (int)(i % (Dc + Dr));
  float *dst = kcl + (size_t)(pos0 + t) * stride;
  if (d < Dc)
    dst[d] = lat[(size_t)t * Dc + d];
  else
    dst[d] = kva[(size_t)t * stride + Dc + (d - Dc)];
}

/* 吸收式 MLA 注意力（一行一头一个 block）：online softmax，两处 block 同步/位置。
 * q 行 = [qabs(Dc) | qpe(Dr)]，cache 行 = [latent(Dc) | k_pe(Dr)]，
 * 输出 o_lat = Σ_p a_p * latent_p（只有前 Dc 维参与加权）。*/
__global__ void k_ds_mla_attn(const float *qabs, const float *qpe, const float *kcl, int n,
                              int Dc, int Dr, int stride, int pos0, float sm_scale,
                              float *olat) {
  /* acc[ITEMS]：每个线程负责 Dc 维里每隔 blockDim.x 的那些位（ITEMS*256 >= 2048 维，
   * DeepSeek 的 kv_lora_rank 最大 512，余量够）。*/
  const int ITEMS = 8;
  int t = blockIdx.x, h = blockIdx.y;
  int pos = pos0 + t, np = pos + 1;
  extern __shared__ float sm[];
  float *qr = sm;            /* Dc+Dr */
  float *sc = &sm[Dc + Dr];  /* 2: 本位置权重、历史缩放因子 */
  __shared__ float red[256];
  __shared__ float m_shared, l_shared;
  const float *qa = qabs + ((size_t)h * n + t) * Dc;
  const float *qp = qpe + ((size_t)h * n + t) * Dr;
  for (int d = threadIdx.x; d < Dc + Dr; d += blockDim.x)
    qr[d] = d < Dc ? qa[d] : qp[d - Dc];
  if (threadIdx.x == 0) {
    m_shared = -1e30f;
    l_shared = 0.0f;
  }
  float acc[8];
  for (int i = 0; i < ITEMS; i++) acc[i] = 0.0f;
  __syncthreads();
  for (int p = 0; p < np; p++) {
    const float *row = kcl + (size_t)p * stride;
    float s = 0.0f;
    for (int d = threadIdx.x; d < Dc + Dr; d += blockDim.x) s += qr[d] * row[d];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
      if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      float sv = red[0] * sm_scale;
      float m_new = sv > m_shared ? sv : m_shared;
      float corr = expf(m_shared - m_new);
      float w = expf(sv - m_new);
      l_shared = l_shared * corr + w;
      m_shared = m_new;
      sc[0] = w;
      sc[1] = corr;
    }
    __syncthreads();
    float w = sc[0], corr = sc[1];
    for (int i = 0; i < ITEMS; i++) {
      int d = threadIdx.x + i * blockDim.x;
      if (d < Dc) acc[i] = acc[i] * corr + w * row[d];
      else
        acc[i] = 0.0f;
    }
    __syncthreads();
  }
  float inv = l_shared > 0.0f ? 1.0f / l_shared : 0.0f;
  float *out = olat + ((size_t)h * n + t) * Dc;
  for (int i = 0; i < ITEMS; i++) {
    int d = threadIdx.x + i * blockDim.x;
    if (d < Dc) out[d] = acc[i] * inv;
  }
}

/* o [H][n][Dv] → att [n][H*Dv] */
__global__ void k_ds_pack_att(const float *obuf, float *att, int n, int H, int Dv) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)n * H;
  if (i >= tot) return;
  int t = (int)(i / H), h = (int)(i % H);
  const float *src = obuf + ((size_t)h * n + t) * Dv;
  float *dst = att + (size_t)t * H * Dv + (size_t)h * Dv;
  for (int k = 0; k < Dv; k++) dst[k] = src[k];
}

__global__ void k_ds_silu_mul(float *gate, const float *up, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) gate[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
}

__global__ void k_ds_add(float *x, const float *h, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) x[i] += h[i];
}

__global__ void k_ds_add_bias(float *C, const float *bias, int M, int N) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)M * N;
  if (i >= tot) return;
  C[i] += bias[i % N];
}

/* MoE 路由（一行一个 block）。逻辑与 CPU ds_moe_row_ 的 1)~5) 一一对应：
 * softmax/sigmoid → +bias → group_limited_greedy 过滤 → top-k → 归一化 × scaling。*/
__global__ void k_ds_route(const float *route, const float *ec_bias, int NE, int k, int n_group,
                           int topk_group, int sigmoid_score, int group_limited, int norm_topk,
                           float routed_scale, int *sel, float *wsel) {
  int r = blockIdx.x;
  extern __shared__ float s[]; /* [NE] 打分 + [NE] 选择分 + [n_group] 组分数 */
  float *sc = s;
  float *ch = s + NE;
  float *gs = s + 2 * NE;
  const float *src = route + (size_t)r * NE;
  for (int i = threadIdx.x; i < NE; i += blockDim.x) sc[i] = src[i];
  __syncthreads();
  if (sigmoid_score) {
    for (int i = threadIdx.x; i < NE; i += blockDim.x)
      sc[i] = 1.0f / (1.0f + expf(-sc[i]));
  } else {
    /* softmax：并行求 max 与 sum */
    __shared__ float red[256];
    float mx = -1e30f;
    for (int i = threadIdx.x; i < NE; i += blockDim.x) mx = fmaxf(mx, sc[i]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
      if (threadIdx.x < off) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + off]);
      __syncthreads();
    }
    float m = red[0];
    __syncthreads();
    float sum = 0;
    for (int i = threadIdx.x; i < NE; i += blockDim.x) {
      sc[i] = expf(sc[i] - m);
      sum += sc[i];
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
      if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
      __syncthreads();
    }
    float tot = red[0];
    for (int i = threadIdx.x; i < NE; i += blockDim.x) sc[i] /= tot;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < NE; i += blockDim.x) ch[i] = sc[i] + (ec_bias ? ec_bias[i] : 0.0f);
  __syncthreads();
  /* 组过滤与 top-k 都是串行小循环（NE、group 都很小），交给线程 0 保证与 CPU 顺序一致 */
  if (threadIdx.x == 0) {
    if (group_limited && n_group > 0) {
      int epg = NE / n_group;
      int chosen[64];
      for (int g = 0; g < n_group; g++) {
        const float *base = ch + (size_t)g * epg;
        float t1 = -1e30f, t2 = -1e30f;
        for (int i = 0; i < epg; i++) {
          float v = base[i];
          if (v > t1) {
            t2 = t1;
            t1 = v;
          } else if (v > t2) {
            t2 = v;
          }
        }
        gs[g] = (epg > 1 ? t1 + t2 : t1);
      }
      for (int j = 0; j < (topk_group < 64 ? topk_group : 64); j++) {
        int best = -1;
        float bv = -1e30f;
        for (int i = 0; i < n_group; i++) {
          int used = 0;
          for (int q = 0; q < j; q++)
            if (chosen[q] == i) used = 1;
          if (!used && gs[i] > bv) {
            bv = gs[i];
            best = i;
          }
        }
        chosen[j] = best;
      }
      for (int i = 0; i < NE; i++) {
        int g = i / epg, keep = 0;
        for (int j = 0; j < topk_group && j < 64; j++)
          if (chosen[j] == g) keep = 1;
        if (!keep) ch[i] = -1e30f;
      }
    }
    for (int j = 0; j < k; j++) {
      int best = -1;
      float bv = -1e30f;
      for (int i = 0; i < NE; i++) {
        int used = 0;
        for (int q = 0; q < j; q++)
          if (sel[(size_t)r * k + q] == i) used = 1;
        if (!used && ch[i] > bv) {
          bv = ch[i];
          best = i;
        }
      }
      sel[(size_t)r * k + j] = best;
      wsel[(size_t)r * k + j] = best >= 0 ? sc[best] : 0.0f;
    }
    float t = 0;
    if (k > 1 && norm_topk) {
      for (int j = 0; j < k; j++) t += wsel[(size_t)r * k + j];
      t += 1e-20f;
      if (t != 0.0f)
        for (int j = 0; j < k; j++) wsel[(size_t)r * k + j] /= t;
    }
    for (int j = 0; j < k; j++) wsel[(size_t)r * k + j] *= routed_scale;
  }
}

/* dst[i][:] = src[idx[i]][:]（MoE 按专家 gather 行）*/
__global__ void k_ds_gather_rows(float *dst, const float *src, const int *idx, int rows,
                                  int cols) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)rows * cols;
  if (i >= tot) return;
  int r = (int)(i / cols), c = (int)(i % cols);
  dst[i] = src[(size_t)idx[r] * cols + c];
}

/* out[idx[i]][:] += w[i] * src[i][:]（不同 slot 可能命中同一行 → atomicAdd）*/
__global__ void k_ds_scatter_moe(float *out, const float *src, const int *idx, const float *w,
                                 int rows, int cols) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)rows * cols;
  if (i >= tot) return;
  int r = (int)(i / cols), c = (int)(i % cols);
  atomicAdd(out + (size_t)idx[r] * cols + c, w[r] * src[i]);
}

/* ------------------------------ host helpers ------------------------------ */

/* C[M][N] = A[M][K] · W[N][K]ᵀ (+bias)，row-major，cublas 列主序映射 */
static int gemm_nt(cublasHandle_t h, cudaStream_t s, const float *A, const float *W,
                   const float *bias, float *C, int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) return 0;
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st =
      cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, K, A, K, &beta, C, N);
  if (st != CUBLAS_STATUS_SUCCESS) {
    SGL_ERROR("cublasSgemm failed (%d) M=%d N=%d K=%d", (int)st, M, N, K);
    return -1;
  }
  if (bias) {
    long tot = (long)M * N;
    k_ds_add_bias<<<(int)((tot + 255) / 256), 256, 0, s>>>(C, bias, M, N);
    CK(cudaGetLastError());
  }
  return 0;
}

static float *dev_dup(const float *src, size_t n) {
  if (!src || n == 0) return NULL;
  float *d = NULL;
  if (cudaMalloc((void **)&d, n * sizeof(float)) != cudaSuccess) return NULL;
  if (cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d);
    return NULL;
  }
  return d;
}
/* ------------------------------ 分配 / 上传 / 释放 ------------------------------ */

static void ds_free_scratch(sgl_cuda_ds *g) {
#define FREEP(p)          \
  do {                    \
    if (p) {              \
      cudaFree((void *)(p)); \
      (p) = NULL;         \
    }                     \
  } while (0)
  g->qpe = NULL; /* 指向 qn 内部，不能单独 free */
  FREEP(g->x);
  FREEP(g->h);
  FREEP(g->q);
  FREEP(g->kva);
  FREEP(g->qa);
  FREEP(g->lat);
  FREEP(g->qn);
  FREEP(g->qabs);
  FREEP(g->olat);
  FREEP(g->obuf);
  FREEP(g->att);
  FREEP(g->gu);
  FREEP(g->upd);
  FREEP(g->mlp);
  FREEP(g->route);
  FREEP(g->gath);
  FREEP(g->eout);
  FREEP(g->d_wsel);
  FREEP(g->d_sel);
  FREEP(g->d_idx);
  FREEP(g->d_tok);
  sgl_xfree(g->h_sel);
  sgl_xfree(g->h_wsel);
  g->h_sel = NULL;
  g->h_wsel = NULL;
  g->cap = 0;
#undef FREEP
}

static int ds_reserve(sgl_cuda_ds *g, int n) {
  if (n <= 0) return -1;
  if (n <= g->cap) return 0;
  int cap = g->cap > 0 ? g->cap : 64;
  while (cap < n) cap *= 2;
  int E = g->E, H = g->H, Dc = g->Dc, Dr = g->Dr, Dv = g->Dv, I = g->I;
  int mi = g->mi > 0 ? g->mi : 1;
  int sh = g->sh_inter > 0 ? g->sh_inter : 1;
  int rw = I > 2 * mi ? I : 2 * mi;
  if (sh > rw) rw = sh;
  int k = g->topk > 0 ? g->topk : 1;
  int NE = g->NE > 0 ? g->NE : 1;
  int ql = g->q_lora > Dc ? g->q_lora : Dc;

  ds_free_scratch(g);
#define A(field, count)                                                                      \
  do {                                                                                       \
    if (cudaMalloc((void **)&g->field, sizeof(float) * (size_t)(count)) != cudaSuccess) {      \
      SGL_ERROR("cuda_deepseek: cudaMalloc %s failed (cap=%d)", #field, cap);                  \
      ds_free_scratch(g);                                                                     \
      return -1;                                                                              \
    }                                                                                         \
  } while (0)
  A(x, (size_t)cap * E);
  A(h, (size_t)cap * E);
  A(q, (size_t)cap * H * g->qhd);
  A(kva, (size_t)cap * g->stride);
  A(qa, (size_t)cap * ql);
  A(lat, (size_t)cap * Dc);
  A(qn, (size_t)cap * H * (Dc + Dr));
  g->qpe = g->qn + (size_t)cap * H * Dc; /* qpe 与 qn 共用一块：qn 在前 */
  A(qabs, (size_t)cap * H * Dc);
  A(olat, (size_t)cap * H * Dc);
  A(obuf, (size_t)cap * H * Dv);
  A(att, (size_t)cap * H * Dv);
  A(gu, (size_t)cap * rw);
  A(upd, (size_t)cap * rw);
  A(mlp, (size_t)cap * E);
  A(route, (size_t)cap * NE);
  A(gath, (size_t)cap * k * E);
  A(eout, (size_t)cap * k * E);
  A(d_wsel, (size_t)cap * k);
#undef A
  if (cudaMalloc((void **)&g->d_sel, sizeof(int) * (size_t)cap * k) != cudaSuccess ||
      cudaMalloc((void **)&g->d_idx, sizeof(int) * (size_t)cap * k) != cudaSuccess ||
      cudaMalloc((void **)&g->d_tok, sizeof(int) * (size_t)cap) != cudaSuccess) {
    SGL_ERROR("cuda_deepseek: cudaMalloc routing buffers failed (cap=%d k=%d)", cap, k);
    ds_free_scratch(g);
    return -1;
  }
  g->h_sel = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap * k);
  g->h_wsel = (float *)sgl_xmalloc(sizeof(float) * (size_t)cap * k);
  if (!g->h_sel || !g->h_wsel) {
    ds_free_scratch(g);
    return -1;
  }
  g->cap = cap;
  return 0;
}

static int ds_ensure_logits(sgl_cuda_ds *g, int rows) {
  if (rows <= g->lgb_cap) return 0;
  if (g->d_lgb) cudaFree(g->d_lgb);
  if (g->h_lgb) cudaFreeHost(g->h_lgb);
  g->d_lgb = NULL;
  g->h_lgb = NULL;
  int cap = rows < 32 ? 32 : rows;
  if (cudaMalloc((void **)&g->d_lgb, sizeof(float) * (size_t)cap * g->V) != cudaSuccess)
    return -1;
  if (cudaMallocHost((void **)&g->h_lgb, sizeof(float) * (size_t)cap * g->V) != cudaSuccess)
    return -1;
  g->lgb_cap = cap;
  return 0;
}

static int dev_copy_field(float **dst, const sgl_tensor *t) {
  *dst = NULL;
  if (!t || !t->data) return 0;
  *dst = dev_dup(t->data, (size_t)sgl_tensor_numel(t));
  return *dst ? 0 : -1;
}

sgl_cuda_ds *sgl_cuda_ds_from(const void *ds_model) {
  const sgl_deepseek *m = (const sgl_deepseek *)ds_model;
  if (!m) return NULL;
  sgl_cuda_ds *g = (sgl_cuda_ds *)calloc(1, sizeof(*g));
  if (!g) return NULL;
  /* nvcc 把 .cu 当 C++ 编译：goto fail 不能跨过带初始化的声明 → 循环里用到的东西先集中定义 */
  int L = 0;
  g->L = m->n_layer;
  g->E = m->n_embd;
  g->H = m->n_head;
  g->Dn = m->qk_nope;
  g->Dr = m->qk_rope;
  g->Dc = m->kv_lora;
  g->Dv = m->v_head;
  g->qhd = m->q_head_dim;
  g->stride = m->stride;
  g->V = m->vocab;
  g->ctx = m->n_ctx;
  g->I = m->n_inter;
  g->q_lora = m->q_lora;
  g->NE = m->n_experts;
  g->topk = m->n_topk;
  g->mi = m->moe_inter;
  g->n_shared = m->n_shared;
  g->sh_inter = m->shared_inter;
  g->norm_topk = m->norm_topk;
  g->sigmoid_score = m->sigmoid_score;
  g->group_limited = m->group_limited;
  g->n_group = m->n_group;
  g->topk_group = m->topk_group;
  g->rope_interleave = m->rope_interleave;
  g->yarn = m->yarn;
  g->rope_attn_factor = m->rope_attn_factor;
  g->routed_scale = m->routed_scale;
  g->eps = m->rms_eps;
  /* softmax_scale 直接用 CPU 侧算好的那个值（YaRN 的 mscale² 已在那边乘进去）*/
  g->sm_scale = m->softmax_scale;

  if (cudaStreamCreate(&g->stream) != cudaSuccess) goto fail;
  if (cublasCreate(&g->blas) != CUBLAS_STATUS_SUCCESS) goto fail;
  cublasSetStream(g->blas, g->stream);

  {
    int half = g->Dr / 2;
    float *f = (float *)sgl_xmalloc(sizeof(float) * (size_t)(half > 0 ? half : 1));
    for (int i = 0; i < half; i++) f[i] = (float)m->invf[i];
    g->invf = f;
    g->d_invf = dev_dup(f, (size_t)(half > 0 ? half : 1));
    if (!g->d_invf) goto fail;
  }

  if (dev_copy_field(&g->wte, &m->wte)) goto fail;
  if (m->lm_head.data == m->wte.data)
    g->lm_head = g->wte; /* tied embeddings：同一块显存 */
  else if (dev_copy_field(&g->lm_head, &m->lm_head))
    goto fail;
  if (dev_copy_field(&g->final_norm, &m->final_norm)) goto fail;

  L = g->L;
  g->inp_norm = (float **)calloc(L, sizeof(float *));
  g->post_norm = (float **)calloc(L, sizeof(float *));
  g->q_a_w = (float **)calloc(L, sizeof(float *));
  g->q_a_norm = (float **)calloc(L, sizeof(float *));
  g->q_b_w = (float **)calloc(L, sizeof(float *));
  g->q_a_b = (float **)calloc(L, sizeof(float *));
  g->kv_a_w = (float **)calloc(L, sizeof(float *));
  g->kv_a_norm = (float **)calloc(L, sizeof(float *));
  g->kv_a_b = (float **)calloc(L, sizeof(float *));
  g->w_ukt = (float **)calloc(L, sizeof(float *));
  g->w_uv = (float **)calloc(L, sizeof(float *));
  g->o_w = (float **)calloc(L, sizeof(float *));
  g->o_b = (float **)calloc(L, sizeof(float *));
  g->is_moe = (int *)calloc(L, sizeof(int));
  g->g_w = (float **)calloc(L, sizeof(float *));
  g->u_w = (float **)calloc(L, sizeof(float *));
  g->d_w = (float **)calloc(L, sizeof(float *));
  g->router_w = (float **)calloc(L, sizeof(float *));
  g->ec_bias = (float **)calloc(L, sizeof(float *));
  g->exp_gu = (float **)calloc(L, sizeof(float *));
  g->exp_dn = (float **)calloc(L, sizeof(float *));
  g->e_gate = (float ***)calloc(L, sizeof(float **));
  g->e_up = (float ***)calloc(L, sizeof(float **));
  g->e_dn = (float ***)calloc(L, sizeof(float **));
  g->sh_g = (float **)calloc(L, sizeof(float *));
  g->sh_u = (float **)calloc(L, sizeof(float *));
  g->sh_d = (float **)calloc(L, sizeof(float *));
  if (!g->inp_norm || !g->post_norm || !g->w_ukt || !g->w_uv || !g->is_moe ||
      !g->e_gate || !g->e_up || !g->e_dn)
    goto fail;

  for (int l = 0; l < L; l++) {
    const sgl_deepseek_layer *Lp = &m->layers[l];
    const int Dn = g->Dn, Dc = g->Dc, Dv = g->Dv, H = g->H;
    const int per_head = (Dn + Dv) * Dc;
    const float *kvb = NULL;
    float *ukt = NULL, *uv = NULL;
    int NE = g->NE;
    if (dev_copy_field(&g->inp_norm[l], &Lp->inp_norm)) goto fail;
    if (dev_copy_field(&g->post_norm[l], &Lp->post_norm)) goto fail;
    if (g->q_lora > 0) {
      if (dev_copy_field(&g->q_a_w[l], &Lp->q_a_w)) goto fail;
      if (dev_copy_field(&g->q_a_norm[l], &Lp->q_a_norm)) goto fail;
    }
    if (dev_copy_field(&g->q_b_w[l], &Lp->q_b_w)) goto fail;
    if (dev_copy_field(&g->kv_a_w[l], &Lp->kv_a_w)) goto fail;
    if (dev_copy_field(&g->kv_a_norm[l], &Lp->kv_a_norm)) goto fail;
    if (dev_copy_field(&g->o_w[l], &Lp->o_w)) goto fail;
    if (dev_copy_field(&g->q_a_b[l], &Lp->q_a_b)) goto fail;
    if (dev_copy_field(&g->kv_a_b[l], &Lp->kv_a_b)) goto fail;
    if (dev_copy_field(&g->o_b[l], &Lp->o_b)) goto fail;

    /* kv_b_proj [H*(Dn+Dv)][Dc] → W_UKᵀ[H][Dc][Dn]（主机侧转置）+ W_UV[H][Dv][Dc]（原样切片）*/
    {
      kvb = Lp->kv_b_w.data;
      if (!kvb) {
        SGL_ERROR("cuda_deepseek: layer %d 缺 kv_b_proj", l);
        goto fail;
      }
      ukt = (float *)sgl_xmalloc(sizeof(float) * (size_t)H * Dc * Dn);
      uv = (float *)sgl_xmalloc(sizeof(float) * (size_t)H * Dv * Dc);
      if (!ukt || !uv) {
        sgl_xfree(ukt);
        sgl_xfree(uv);
        goto fail;
      }
      for (int h = 0; h < H; h++) {
        const float *uk = kvb + (size_t)h * per_head; /* [Dn][Dc] */
        const float *uvp = uk + (size_t)Dn * Dc;      /* [Dv][Dc] */
        float *dst = ukt + (size_t)h * Dc * Dn;       /* [Dc][Dn] */
        for (int j = 0; j < Dn; j++)
          for (int d = 0; d < Dc; d++) dst[(size_t)d * Dn + j] = uk[(size_t)j * Dc + d];
        memcpy(uv + (size_t)h * Dv * Dc, uvp, sizeof(float) * (size_t)Dv * Dc);
      }
      g->w_ukt[l] = dev_dup(ukt, (size_t)H * Dc * Dn);
      g->w_uv[l] = dev_dup(uv, (size_t)H * Dv * Dc);
      sgl_xfree(ukt);
      sgl_xfree(uv);
      if (!g->w_ukt[l] || !g->w_uv[l]) goto fail;
    }

    g->is_moe[l] = Lp->is_moe;
    if (!Lp->is_moe) {
      if (dev_copy_field(&g->g_w[l], &Lp->gate_w)) goto fail;
      if (dev_copy_field(&g->u_w[l], &Lp->up_w)) goto fail;
      if (dev_copy_field(&g->d_w[l], &Lp->down_w)) goto fail;
    } else {
      if (dev_copy_field(&g->router_w[l], &Lp->router_w)) goto fail;
      if (Lp->ec_bias) {
        g->ec_bias[l] = dev_dup(Lp->ec_bias, (size_t)g->NE);
        if (!g->ec_bias[l]) goto fail;
      }
      if (Lp->exp_stacked) {
        if (dev_copy_field(&g->exp_gu[l], &Lp->exp_gu3)) goto fail;
        if (dev_copy_field(&g->exp_dn[l], &Lp->exp_dn3)) goto fail;
      } else {
        g->e_gate[l] = (float **)calloc(NE, sizeof(float *));
        g->e_up[l] = (float **)calloc(NE, sizeof(float *));
        g->e_dn[l] = (float **)calloc(NE, sizeof(float *));
        if (!g->e_gate[l] || !g->e_up[l] || !g->e_dn[l]) goto fail;
        for (int e = 0; e < NE; e++) {
          if (dev_copy_field(&g->e_gate[l][e], &Lp->exp_gate[e])) goto fail;
          if (dev_copy_field(&g->e_up[l][e], &Lp->exp_up[e])) goto fail;
          if (dev_copy_field(&g->e_dn[l][e], &Lp->exp_down[e])) goto fail;
        }
      }
      if (g->n_shared > 0) {
        if (dev_copy_field(&g->sh_g[l], &Lp->sh_gate_w)) goto fail;
        if (dev_copy_field(&g->sh_u[l], &Lp->sh_up_w)) goto fail;
        if (dev_copy_field(&g->sh_d[l], &Lp->sh_down_w)) goto fail;
      }
    }
  }

  {
    size_t nkv = (size_t)g->L * g->ctx * g->stride;
    if (cudaMalloc((void **)&g->kc, sizeof(float) * nkv) != cudaSuccess) {
      SGL_ERROR("cuda_deepseek: KV cache alloc failed (%zu floats)", nkv);
      goto fail;
    }
    if (cudaMemset(g->kc, 0, sizeof(float) * nkv) != cudaSuccess) goto fail;
  }
  if (ds_reserve(g, 64) != 0) goto fail;
  {
    size_t freeb = 0, totalb = 0;
    cudaMemGetInfo(&freeb, &totalb);
    g->mem_bytes = totalb - freeb;
  }
  SGL_INFO("cuda_deepseek: uploaded L=%d E=%d H=%d nope=%d rope=%d lora=%d vhead=%d stride=%d "
           "moe=%d(top%d) ctx=%d",
           g->L, g->E, g->H, g->Dn, g->Dr, g->Dc, g->Dv, g->stride, g->NE, g->topk, g->ctx);
  return g;
fail:
  sgl_cuda_ds_free(g);
  return NULL;
}

void sgl_cuda_ds_free(sgl_cuda_ds *g) {
  if (!g) return;
  int L = g->L;
#define F(p)              \
  do {                    \
    if (p) cudaFree((void *)(p)); \
  } while (0)
  if (g->wte) cudaFree(g->wte);
  if (g->lm_head && g->lm_head != g->wte) cudaFree(g->lm_head);
  F(g->final_norm);
  F(g->d_invf);
  F(g->kc);
  F(g->d_lgb);
  ds_free_scratch(g);
  if (g->h_lgb) cudaFreeHost(g->h_lgb);
  sgl_xfree(g->invf);
  for (int l = 0; l < L; l++) {
    if (g->inp_norm) F(g->inp_norm[l]);
    if (g->post_norm) F(g->post_norm[l]);
    if (g->q_a_w) F(g->q_a_w[l]);
    if (g->q_a_norm) F(g->q_a_norm[l]);
    if (g->q_b_w) F(g->q_b_w[l]);
    if (g->q_a_b) F(g->q_a_b[l]);
    if (g->kv_a_w) F(g->kv_a_w[l]);
    if (g->kv_a_norm) F(g->kv_a_norm[l]);
    if (g->kv_a_b) F(g->kv_a_b[l]);
    if (g->w_ukt) F(g->w_ukt[l]);
    if (g->w_uv) F(g->w_uv[l]);
    if (g->o_w) F(g->o_w[l]);
    if (g->o_b) F(g->o_b[l]);
    if (g->g_w) F(g->g_w[l]);
    if (g->u_w) F(g->u_w[l]);
    if (g->d_w) F(g->d_w[l]);
    if (g->router_w) F(g->router_w[l]);
    if (g->ec_bias) F(g->ec_bias[l]);
    if (g->exp_gu) F(g->exp_gu[l]);
    if (g->exp_dn) F(g->exp_dn[l]);
    if (g->sh_g) F(g->sh_g[l]);
    if (g->sh_u) F(g->sh_u[l]);
    if (g->sh_d) F(g->sh_d[l]);
    if (g->e_gate && g->e_gate[l]) {
      for (int e = 0; e < g->NE; e++) {
        F(g->e_gate[l][e]);
        if (g->e_up && g->e_up[l]) F(g->e_up[l][e]);
        if (g->e_dn && g->e_dn[l]) F(g->e_dn[l][e]);
      }
      sgl_xfree(g->e_gate[l]);
    }
    if (g->e_up && g->e_up[l]) sgl_xfree(g->e_up[l]);
    if (g->e_dn && g->e_dn[l]) sgl_xfree(g->e_dn[l]);
  }
  sgl_xfree(g->inp_norm);
  sgl_xfree(g->post_norm);
  sgl_xfree(g->q_a_w);
  sgl_xfree(g->q_a_norm);
  sgl_xfree(g->q_b_w);
  sgl_xfree(g->q_a_b);
  sgl_xfree(g->kv_a_w);
  sgl_xfree(g->kv_a_norm);
  sgl_xfree(g->kv_a_b);
  sgl_xfree(g->w_ukt);
  sgl_xfree(g->w_uv);
  sgl_xfree(g->o_w);
  sgl_xfree(g->o_b);
  sgl_xfree(g->is_moe);
  sgl_xfree(g->g_w);
  sgl_xfree(g->u_w);
  sgl_xfree(g->d_w);
  sgl_xfree(g->router_w);
  sgl_xfree(g->ec_bias);
  sgl_xfree(g->exp_gu);
  sgl_xfree(g->exp_dn);
  sgl_xfree(g->e_gate);
  sgl_xfree(g->e_up);
  sgl_xfree(g->e_dn);
  sgl_xfree(g->sh_g);
  sgl_xfree(g->sh_u);
  sgl_xfree(g->sh_d);
#undef F
  if (g->blas) cublasDestroy(g->blas);
  if (g->stream) cudaStreamDestroy(g->stream);
  free(g);
}

int sgl_cuda_ds_vocab(const sgl_cuda_ds *g) { return g ? g->V : 0; }
int sgl_cuda_ds_ctx(const sgl_cuda_ds *g) { return g ? g->ctx : 0; }
long sgl_cuda_ds_mem_mb(const sgl_cuda_ds *g) {
  return g ? (long)(g->mem_bytes / (1024 * 1024)) : 0;
}

void sgl_cuda_ds_reset(sgl_cuda_ds *g) {
  if (!g || !g->kc) return;
  cudaMemsetAsync(g->kc, 0, sizeof(float) * (size_t)g->L * g->ctx * g->stride, g->stream);
  cudaStreamSynchronize(g->stream);
}

/* ------------------------------ MLA 一段 ------------------------------ */

static int ds_mla_attn(sgl_cuda_ds *g, int l, int n, int pos0) {
  int H = g->H, Dc = g->Dc, Dr = g->Dr, Dn = g->Dn, Dv = g->Dv;
  cudaStream_t s = g->stream;
  {
    /* k_ds_pack_q 的 block 是 (32,8)：一个 block 只覆盖 32 个 (t,h)，
     * 所以块数要按 32 算，不能按 128（之前写成 128，n*H>32 时后一半行没打包）。*/
    dim3 grid((unsigned)(((long)n * H + 31) / 32));
    dim3 blk(32, 8);
    k_ds_pack_q<<<grid, blk, 0, s>>>(g->q, g->qn, g->qpe, n, H, g->qhd, Dn, Dr);
    if (cudaGetLastError() != cudaSuccess) return -1;
  }
  for (int h = 0; h < H; h++) {
    const float *A = g->qn + (size_t)h * n * Dn;
    const float *W = g->w_ukt[l] + (size_t)h * Dc * Dn;
    float *C = g->qabs + (size_t)h * n * Dc;
    if (gemm_nt(g->blas, s, A, W, NULL, C, n, Dc, Dn) != 0) return -1;
  }
  {
    size_t shmem = sizeof(float) * (size_t)(Dc + Dr + 2);
    dim3 grid((unsigned)n, (unsigned)H);
    k_ds_mla_attn<<<grid, 256, shmem, s>>>(g->qabs, g->qpe,
                                           g->kc + (size_t)l * g->ctx * g->stride, n, Dc, Dr,
                                           g->stride, pos0, g->sm_scale, g->olat);
    if (cudaGetLastError() != cudaSuccess) return -1;
  }
  for (int h = 0; h < H; h++) {
    const float *A = g->olat + (size_t)h * n * Dc;
    const float *W = g->w_uv[l] + (size_t)h * Dv * Dc;
    float *C = g->obuf + (size_t)h * n * Dv;
    if (gemm_nt(g->blas, s, A, W, NULL, C, n, Dv, Dc) != 0) return -1;
  }
  {
    k_ds_pack_att<<<(unsigned)(((long)n * H + 127) / 128), 128, 0, s>>>(g->obuf, g->att, n, H, Dv);
    if (cudaGetLastError() != cudaSuccess) return -1;
  }
  if (gemm_nt(g->blas, s, g->att, g->o_w[l], g->o_b[l], g->h, n, g->E, H * Dv) != 0) return -1;
  k_ds_add<<<(int)(((long)n * g->E + 255) / 256), 256, 0, s>>>(g->x, g->h, (long)n * g->E);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* ------------------------------ MoE / MLP 一段 ------------------------------ */

static int ds_mlp(sgl_cuda_ds *g, int l, int n) {
  cudaStream_t s = g->stream;
  int E = g->E, I = g->I, mi = g->mi, NE = g->NE, k = g->topk;
  if (!g->is_moe[l]) {
    if (gemm_nt(g->blas, s, g->h, g->g_w[l], NULL, g->gu, n, I, E) != 0) return -1;
    if (gemm_nt(g->blas, s, g->h, g->u_w[l], NULL, g->upd, n, I, E) != 0) return -1;
    k_ds_silu_mul<<<(int)(((long)n * I + 255) / 256), 256, 0, s>>>(g->gu, g->upd, (long)n * I);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (gemm_nt(g->blas, s, g->gu, g->d_w[l], NULL, g->mlp, n, E, I) != 0) return -1;
    return 0;
  }

  if (gemm_nt(g->blas, s, g->h, g->router_w[l], NULL, g->route, n, NE, E) != 0) return -1;
  {
    size_t shmem = sizeof(float) * (size_t)(2 * NE + (g->n_group > 0 ? g->n_group : 1));
    k_ds_route<<<n, 128, shmem, s>>>(g->route, g->ec_bias[l], NE, k, g->n_group, g->topk_group,
                                     g->sigmoid_score, g->group_limited, g->norm_topk,
                                     g->routed_scale, g->d_sel, g->d_wsel);
    if (cudaGetLastError() != cudaSuccess) return -1;
  }
  int m = n * k;
  if (cudaMemcpyAsync(g->h_sel, g->d_sel, sizeof(int) * (size_t)m, cudaMemcpyDeviceToHost, s) !=
          cudaSuccess ||
      cudaMemcpyAsync(g->h_wsel, g->d_wsel, sizeof(float) * (size_t)m, cudaMemcpyDeviceToHost, s) !=
          cudaSuccess ||
      cudaStreamSynchronize(s) != cudaSuccess)
    return -1;

  /* 按专家分组（顺序与 CPU 选择一致；starts 是每段起点）*/
  int *prow = (int *)sgl_xmalloc(sizeof(int) * (size_t)m);
  float *pw = (float *)sgl_xmalloc(sizeof(float) * (size_t)m);
  int *starts = (int *)sgl_xmalloc(sizeof(int) * (size_t)(NE + 1));
  if (!prow || !pw || !starts) {
    sgl_xfree(prow);
    sgl_xfree(pw);
    sgl_xfree(starts);
    return -1;
  }
  int cnt = 0;
  for (int e = 0; e < NE; e++) {
    starts[e] = cnt;
    for (int i = 0; i < m; i++) {
      if (g->h_sel[i] != e) continue;
      prow[cnt] = i / k;
      pw[cnt] = g->h_wsel[i];
      cnt++;
    }
  }
  starts[NE] = cnt;

  if (cnt > 0) {
    if (cudaMemcpyAsync(g->d_idx, prow, sizeof(int) * (size_t)cnt, cudaMemcpyHostToDevice, s) !=
            cudaSuccess ||
        cudaMemcpyAsync(g->d_wsel, pw, sizeof(float) * (size_t)cnt, cudaMemcpyHostToDevice, s) !=
            cudaSuccess) {
      sgl_xfree(prow);
      sgl_xfree(pw);
      sgl_xfree(starts);
      return -1;
    }
    k_ds_gather_rows<<<(int)(((long)cnt * E + 255) / 256), 256, 0, s>>>(g->gath, g->h, g->d_idx,
                                                                       cnt, E);
    if (cudaGetLastError() != cudaSuccess) {
      sgl_xfree(prow);
      sgl_xfree(pw);
      sgl_xfree(starts);
      return -1;
    }
  }

  for (int e = 0; e < NE; e++) {
    int a = starts[e], b = starts[e + 1];
    if (b <= a) continue;
    int rows = b - a;
    float *src = g->gath + (size_t)a * E;
    float *dst = g->eout + (size_t)a * E;
    if (g->exp_gu[l]) {
      const float *gu3 = g->exp_gu[l] + (size_t)e * 2 * mi * E;
      const float *dn3 = g->exp_dn[l] + (size_t)e * E * mi;
      if (gemm_nt(g->blas, s, src, gu3, NULL, g->gu, rows, 2 * mi, E) != 0) {
        sgl_xfree(prow);
        sgl_xfree(pw);
        sgl_xfree(starts);
        return -1;
      }
      k_ds_silu_mul<<<(int)(((long)rows * mi + 255) / 256), 256, 0, s>>>(g->gu, g->gu + mi,
                                                                        (long)rows * mi);
      if (gemm_nt(g->blas, s, g->gu, dn3, NULL, dst, rows, E, mi) != 0) {
        sgl_xfree(prow);
        sgl_xfree(pw);
        sgl_xfree(starts);
        return -1;
      }
    } else {
      if (gemm_nt(g->blas, s, src, g->e_gate[l][e], NULL, g->gu, rows, mi, E) != 0 ||
          gemm_nt(g->blas, s, src, g->e_up[l][e], NULL, g->upd, rows, mi, E) != 0) {
        sgl_xfree(prow);
        sgl_xfree(pw);
        sgl_xfree(starts);
        return -1;
      }
      k_ds_silu_mul<<<(int)(((long)rows * mi + 255) / 256), 256, 0, s>>>(g->gu, g->upd,
                                                                        (long)rows * mi);
      if (gemm_nt(g->blas, s, g->gu, g->e_dn[l][e], NULL, dst, rows, E, mi) != 0) {
        sgl_xfree(prow);
        sgl_xfree(pw);
        sgl_xfree(starts);
        return -1;
      }
    }
  }

  if (cudaMemsetAsync(g->mlp, 0, sizeof(float) * (size_t)n * E, s) != cudaSuccess) {
    sgl_xfree(prow);
    sgl_xfree(pw);
    sgl_xfree(starts);
    return -1;
  }
  if (cnt > 0) {
    k_ds_scatter_moe<<<(int)(((long)cnt * E + 255) / 256), 256, 0, s>>>(g->mlp, g->eout, g->d_idx,
                                                                       g->d_wsel, cnt, E);
    if (cudaGetLastError() != cudaSuccess) {
      sgl_xfree(prow);
      sgl_xfree(pw);
      sgl_xfree(starts);
      return -1;
    }
  }
  sgl_xfree(prow);
  sgl_xfree(pw);
  sgl_xfree(starts);

  if (g->n_shared > 0 && g->sh_g[l]) {
    int sh = g->sh_inter;
    if (gemm_nt(g->blas, s, g->h, g->sh_g[l], NULL, g->gu, n, sh, E) != 0) return -1;
    if (gemm_nt(g->blas, s, g->h, g->sh_u[l], NULL, g->upd, n, sh, E) != 0) return -1;
    k_ds_silu_mul<<<(int)(((long)n * sh + 255) / 256), 256, 0, s>>>(g->gu, g->upd, (long)n * sh);
    if (gemm_nt(g->blas, s, g->gu, g->sh_d[l], NULL, g->upd, n, E, sh) != 0) return -1;
    k_ds_add<<<(int)(((long)n * E + 255) / 256), 256, 0, s>>>(g->mlp, g->upd, (long)n * E);
    if (cudaGetLastError() != cudaSuccess) return -1;
  }
  return 0;
}

/* ------------------------------ forward ------------------------------ */

int sgl_cuda_ds_forward_n(sgl_cuda_ds *g, const int *tokens, int n, int pos0, float *logits,
                          int all_rows) {
  if (!g || !tokens || n <= 0) return -1;
  if (pos0 < 0 || pos0 + n > g->ctx) {
    SGL_ERROR("cuda_deepseek forward: pos0=%d n=%d 超出 ctx=%d", pos0, n, g->ctx);
    return -1;
  }
  if (ds_reserve(g, n) != 0) return -1;
  cudaStream_t s = g->stream;
  int E = g->E, H = g->H, Dc = g->Dc, Dr = g->Dr, Dn = g->Dn;

  if (cudaMemcpyAsync(g->d_tok, tokens, sizeof(int) * (size_t)n, cudaMemcpyHostToDevice, s) !=
      cudaSuccess)
    return -1;
  k_ds_embed<<<(int)(((long)n * E + 255) / 256), 256, 0, s>>>(g->wte, g->d_tok, g->x, n, E);
  if (cudaGetLastError() != cudaSuccess) return -1;

  for (int l = 0; l < g->L; l++) {
    k_ds_rmsnorm<<<n, 256, 0, s>>>(g->x, g->inp_norm[l], g->h, n, E, E, E, g->eps);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (g->q_lora > 0) {
      if (gemm_nt(g->blas, s, g->h, g->q_a_w[l], g->q_a_b[l], g->qa, n, g->q_lora, E) != 0)
        return -1;
      k_ds_rmsnorm<<<n, 256, 0, s>>>(g->qa, g->q_a_norm[l], g->qa, n, g->q_lora, g->q_lora,
                                     g->q_lora, g->eps);
      if (cudaGetLastError() != cudaSuccess) return -1;
      if (gemm_nt(g->blas, s, g->qa, g->q_b_w[l], NULL, g->q, n, H * g->qhd, g->q_lora) != 0)
        return -1;
    } else {
      if (gemm_nt(g->blas, s, g->h, g->q_b_w[l], g->q_a_b[l], g->q, n, H * g->qhd, E) != 0)
        return -1;
    }
    if (gemm_nt(g->blas, s, g->h, g->kv_a_w[l], g->kv_a_b[l], g->kva, n, g->stride, E) != 0)
      return -1;
    {
      float scale = g->yarn ? g->rope_attn_factor : 1.0f;
      k_ds_rope<<<(unsigned)(n * H), 64, 0, s>>>(g->q, n * H, g->qhd, Dn, Dr, pos0, H, g->d_invf,
                                                 g->rope_interleave, scale);
      k_ds_rope<<<(unsigned)n, 64, 0, s>>>(g->kva, n, g->stride, Dc, Dr, pos0, 1, g->d_invf,
                                           g->rope_interleave, scale);
      if (cudaGetLastError() != cudaSuccess) return -1;
    }
    k_ds_rmsnorm<<<n, 256, 0, s>>>(g->kva, g->kv_a_norm[l], g->lat, n, Dc, g->stride, Dc, g->eps);
    k_ds_kv_write<<<(int)(((long)n * (Dc + Dr) + 255) / 256), 256, 0, s>>>(
        g->kc + (size_t)l * g->ctx * g->stride, g->lat, g->kva, n, pos0, Dc, Dr, g->stride);
    if (cudaGetLastError() != cudaSuccess) return -1;

    if (ds_mla_attn(g, l, n, pos0) != 0) return -1;

    k_ds_rmsnorm<<<n, 256, 0, s>>>(g->x, g->post_norm[l], g->h, n, E, E, E, g->eps);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (ds_mlp(g, l, n) != 0) return -1;
    k_ds_add<<<(int)(((long)n * E + 255) / 256), 256, 0, s>>>(g->x, g->mlp, (long)n * E);
    if (cudaGetLastError() != cudaSuccess) return -1;
  }

  k_ds_rmsnorm<<<n, 256, 0, s>>>(g->x, g->final_norm, g->h, n, E, E, E, g->eps);
  if (cudaGetLastError() != cudaSuccess) return -1;

  if (logits) {
    int rows = all_rows ? n : 1;
    const float *base = all_rows ? g->h : g->h + (size_t)(n - 1) * E;
    if (ds_ensure_logits(g, rows) != 0) return -1;
    if (gemm_nt(g->blas, s, base, g->lm_head, NULL, g->d_lgb, rows, g->V, E) != 0) return -1;
    if (cudaMemcpyAsync(g->h_lgb, g->d_lgb, sizeof(float) * (size_t)rows * g->V,
                        cudaMemcpyDeviceToHost, s) != cudaSuccess)
      return -1;
    if (cudaStreamSynchronize(s) != cudaSuccess) return -1;
    memcpy(logits, g->h_lgb, sizeof(float) * (size_t)rows * g->V);
  } else if (cudaStreamSynchronize(s) != cudaSuccess) {
    return -1;
  }
  return 0;
}

int sgl_cuda_ds_decode(sgl_cuda_ds *g, int token, int pos, float *logits) {
  return sgl_cuda_ds_forward_n(g, &token, 1, pos, logits, 0);
}