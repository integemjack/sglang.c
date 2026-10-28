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
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

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

/* t8 W8A8 per-channel int8 weights: w[N,K] int8 + scale[N].
 * M1：w4 指向同一投影的 FP4 副本（NULL = 没有）。use_w4 时 PROJ 走它，
 * int8 数据不再上传——两套同时放显存放不下。*/
typedef struct { int8_t *w; float *scale; void *w4; } sgl_w8w;

/* ================= M3: MTP（multi-token prediction）草稿头 =================
 * 目的：让一次前向产出多个 token。单发上限是 61.9 tok/s（实测：每 token 必读
 * 27.296 GB / 1690 GB/s），只有「一次前向出多 token」能过 100，且不动精度。
 *
 * 结构（与 transformers modeling_layers.py:313 MtpLayer 一致）：
 *   proj_in = cat([enorm(emb(t_p)), hnorm(hidden_p)])   [2E]
 *   x       = fc @ proj_in                              [E]  (无 bias)
 *   x       = DecoderLayer(x)  一个 full-attention 层，自带 KV cache
 *   h       = post_norm(x)                              [E]
 *   tok'    = argmax(lm_head @ h)
 * 注意 emb 与 hidden 取**同一位置 p**，预测 p+1（不是错开一格）。
 *
 * MTP 层的 q/k/v/o 与 mlp 维度跟主模型的 full_attention 层完全一致，所以
 * 直接复用 PROJ / k_hrmsnorm / k_rope / k_attn_long，不另写一套。*/
typedef struct {
  int ok;                     /* 1 = 权重已加载可用 */
  sgl_w8w fc8;                /* [E, 2E] */
  sgl_w8w q8, k8, v8, o8, g8, u8, d8;
  float *enorm, *hnorm;       /* [E] pre_fc_norm_embedding / _hidden */
  float *ln1, *ln2;           /* [E] input_layernorm / post_attention_layernorm */
  float *post_norm;           /* [E] mtp.norm */
  float *qn, *kn;             /* [D] q_norm / k_norm */
  float *e, *proj_in;         /* [E], [2E] */
  float *x, *h, *q, *q2, *qgate, *k, *v, *attn, *gate, *up; /* 单行 scratch */
  float *kc, *vc;             /* [mtp_ctx][KD] 自有 KV（不与主模型共享）*/
  int ctx;                    /* MTP KV 的容量 */
  int hsrc;                   /* 0 = final-norm 之前, 1 = 之后（用接受率裁决）*/
  int *d_tok;                 /* [1] 草稿输入 token */
  int *d_arg;                 /* [1] device argmax 结果 */
  /* 链式草稿（W>1）：本层输出 hidden 给下一步当输入。只分配 E 大小的两个副本。*/
  float *chain_pre, *chain_post;
  int chain_use_post;         /* SGL_MTP_CHAIN=post 时用 chain_post */
  float *hlast;               /* [E] 「最后一个已落定位置」的 hidden（草稿输入）*/
  /* sgl_cuda_mtp_fill_kv 的多行 scratch（每块 ≤ fill_cap 行）与跨块衔接的 hidden */
  int fill_cap;
  float *fe, *fhid, *fpin, *fx, *fh, *fk, *fv; /* [cap][E]×5、[cap][2E]、[cap][KD]×2 */
  int *ftok;                  /* [cap] */
  float *hprev;               /* [E] 上一块最后一行的 hidden（下一块第 0 行用）*/
  int base;                   /* 当前序列在 kc/vc 里的起始行（= 主模型 KV 段基址，前缀复用时整段搬）*/
  float *d_p;                 /* [1] device：草稿 token 的 softmax 概率 */
  float last_p;               /* 最近一次 mtp_draft 的草稿概率（host；want_p=0 时不算）*/
  int want_p;                 /* 1 = mtp_draft 要算草稿概率（阈值或统计要用；默认路径不多付这一趟）*/
} sgl_mtp;

/* ================= M3: 投机解码的 GDN 状态快照 =================
 * 拒绝部分草稿时，KDN 递归状态已经多推进了几行，必须回滚到「第 k 行处理完」
 * 的精确状态。状态在寄存器里（k_lin_persist_reg 的 st[CH]），kernel 只在结尾
 * 写回全局，所以逐行快照只能写在 kernel 的行循环里。
 * 内存：每行 = 全部 lin 层的 state+hist；W=4 时 5 行，约 720 MiB（只对 1 个槽）。*/
typedef struct {
  int rows;                   /* = W+1 */
  float *S;                   /* [rows][lin_n][nh][kd][vd] */
  float *H;                   /* [rows][lin_n][convd][K-1] */
  long s_layer;               /* 单层 state 的 float 数 */
  long h_layer;               /* 单层 hist 的 float 数 */
} sgl_specsnap;

/* ================= M2/M6：量化分页 KV =================
 * 为什么必须量化：16 层 full-attn 的 KV 在 FP32 下每 token 128 KiB，
 * 262K 上下文每 token 要读 137 GB（24 个 q head 各读一遍同一个 kv head），
 * 1690 GB/s 也要 81 ms——比 27 GB 权重读取贵 5 倍。4-bit 后降到 17 GB。
 *
 * 布局（每 token 每层 tok_stride 字节）：
 *   [0, nvals/2)                K 的 int4 nibble（低半字节 = 偶数 d）
 *   [nvals/2, nvals)            V 的 int4
 *   [nvals, nvals+nblk*4)       K 的 scale（每 SGL_KVQ_BLK 个值一个 float）
 *   [nvals+nblk*4, ...)         V 的 scale
 * 其中 nvals = HK*D = 1024，nblk = nvals/128 = 8 → tok_stride = 1088 B，
 * 16 层 = 17.0 KiB/token（FP32 是 128 KiB，降 7.5 倍）。
 *
 * 分页：页 = SGL_KVQ_P(64) 个位置。物理池 [nkv][n_slots][P][tok_stride]，
 * 页表 pt[page] = slot 或 -1。M2 阶段 n_slots = n_pages（页表恒等、全驻留）；
 * M6 阶段 n_slots < n_pages，冷页的权威副本在 pinned host，靠 resolve 换入。*/
#define SGL_KVQ_P 64
#define SGL_KVQ_BLK 128
typedef struct {
  int on;                 /* 1 = 量化分页路径启用 */
  int nkv;                /* KV 层数 */
  int P;                  /* 页大小（token）*/
  int n_pages;            /* 页表长度 = ceil(ctx/P) */
  int n_slots;            /* 物理槽数；= n_pages 即全驻留 */
  int HK, D, nvals, nblk;
  long tok_stride;        /* 每 token 每层字节 */
  long slot_layer;        /* 每 slot 每层字节 = P*tok_stride */
  long layer_slots;       /* 每层全部 slot 字节 */
  unsigned char *d_pool;  /* [nkv][n_slots][P][tok_stride] */
  int *d_pt;              /* [n_pages] page -> slot / -1 */
  int *h_pt;              /* host 镜像（resolve 决策用）*/
  unsigned char *h_pool;  /* [nkv][n_pages][P][tok_stride] pinned 权威副本（M6）*/
  int host_slots;         /* 1 = h_pool 已分配 */
  unsigned *d_clock;      /* [n_pages] CLOCK 参考位（M6）*/
  float *d_split;         /* [Bmax*H*nsplit][2+D] split-KV 部分和 */
  int split_cap;          /* 已分配条目数 */
  int nsplit;             /* 每 (b,h) 切几段 */
} sgl_kvq;

/* 定义在 M2/M6 区段，但 KV 分配处（更早）要用 */
static int kvq_init(sgl_cuda_model *g, long ctx, int nkv, int HK, int D, int host_resident,
                    int n_slots_want);
/* M1: NVFP4 风格的 4-bit 权重。数据 2 个值/字节（e2m1），缩放按 16 元素一块
 * 存 fp16。相对 int8 的 1 字节/参数，这里是 0.5 + 2/16 = 0.625 字节/参数
 * （1.6× 减容）；e2m1 只有 16 个电平，必须靠小块缩放才能保住精度。*/
typedef struct { uint8_t *w; __half *bs; } sgl_w4w;

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
  /* M1 FP4 权重（与 w8 互斥：只上其中一套，否则显存放不下两套 27 GB）*/
  sgl_w4w *qw4, *kw4, *vw4, *ow4, *gw4, *uw4, *dw4;
  sgl_w4w lm_head_w4;
  int use_w4;
  /* M1：FP4 GEMM 的 int8 激活 scratch（M 行 × K 列 + 每行一个缩放）。
   * K 最大 17408（down_proj）、M 最大 32，共 557 KB，按需扩。*/
  int8_t *w4_a8;
  float *w4_as;
  size_t w4_acap;   /* 已分配的 M*K 字节数 */
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
 int emb_h;    /* t11：embed/lm_head 存 fp16（w8 模式默认）*/
  __half *wte_h;            /* [V,E] embed table; lm_head aliases when tied */
  __half *wte_h_host;       /* M0: 若非 NULL，embed 在 mapped pinned host（wte_h 是其设备别名）*/
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
  float *h_log1;            /* t11：hybrid 逐行 decode 的单行临时出口（pinned, V floats） */
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

  /* t11 qwen3_5 hybrid（G9 linear/deltanet + G13 full 特性）*/
  int QD;                /* q/o 投影宽：显式 head_dim 时 = H*D ≠ E */
  int q_gate;            /* 1 = q_proj 输出 2*QD（按头交错 [q(D)|gate(D)]）*/
  int norm_p1;           /* 1 = RMSNorm 用 (1+w)（Qwen3NextRMSNorm）*/
  int rot_dim;           /* partial rotary 维度（0 = 整头）*/
  unsigned char *lkind;  /* [L] 0=full/sliding attn, 2=linear deltanet */
  int n_kv_layers;       /* t11: number of layers needing KV (full/sliding) */
  int *kvo;              /* t11：层 l 的紧凑 KV 序号（lin 层不占 KV，重复利用省显存）*/
  int *linvo;            /* t11：层 l 的紧凑 lin 序号（-1 = 非 lin 层）*/
  float **qn, **kn;      /* full 层 q/k rmsnorm 权重 [D]（可 NULL）*/
  float *q2;               /* t11：q_gate 时 PROJ 的 2QD 输出 [ws][2QD] */
  float *qgate;            /* t11：gate 半边 [ws][QD] */
  double *pinvf;         /* [L][max_half] partial-rope 逆频率表 */
  /* lin 层（int8 复用 qw8/ow8 槽：qkvz->qw8, out_proj->ow8；mlp->gw8/uw8/dw8）*/
  float **l_ba;          /* [2*lin_nh][E] f32 */
  float **l_conv;        /* [convd][K] f32 */
  float **l_norm;        /* [lin_vd] gated-norm 权重 */
  float **l_alog, **l_dtb; /* [lin_nh] */
  float *l_state;        /* [n_lin][slots][nh][vd][kd] 递归状态（kd 最内层，零初始化）*/
  float *l_hist;         /* t11：[n_lin][slots][convd*(K-1)] conv 输入历史（零初始化）*/
  int lin_n, lin_slots, lin_nh, lin_nk, lin_kd, lin_vd, lin_conv_k, lin_ratio;
  int lin_scratch_cap;   /* 下方 lin 临时缓冲覆盖的行数 */
  float *l_qkvz;         /* [cap][2keyd+2vald] */
  float *l_conv_io;      /* [cap][convd] conv 后 siLU 结果 */
  float *l_qn, *l_kn;    /* [cap][keyd] l2norm 后 */
  float *l_o;            /* [cap][vald] deltanet 输出（gnorm 前）*/
  /* t11 prefill 加速：把 lin 层的 qkvz / out_proj 从「逐 token 的 M=1 GEMV」
   * 改成「一次 M=n 的 GEMM」。GEMV 每 token 要把 84MB 权重完整读一遍，
   * 48 层 × 1024 token × 2 次 = 4.6TB 权重流（实际只受带宽限制，2.6s+）；
   * 批量后权重只读 48×2 遍（≈4.5GB）。逐 token 只保留不可并行的递归主体。*/
  float *l_qkvz_all;     /* [ws_cap][rowN] 整块的 qkvz 投影结果（只读下游）*/
  float *l_o_all;        /* [ws_cap][vald] 整块的 deltanet 输出 */
  float *l_ba_all;       /* [ws_cap][2*lin_nh] 每行的门控输入 h·W_ba^T（k_lin_ba 算）*/
  int *d_lin_slot;       /* [1] graph 重放前更新的 lin slot */
  int *h_lin_slot;       /* pinned 镜像 */
  int *h_dec_slots;      /* t11：decode batch 每行的 lin 槽（pinned, 32 个）*/

  /* G2: int8 activations + int32 accumulators for the IMMA big-M path.
   * Sized with the workspace (ws_cap rows x max(E,I) cols): every W8
   * projection has K <= max(E,I) and N <= max(E,I). */
  int8_t *w8_a8;   /* [ws_cap][max(E,I)] */
  int32_t *w8_c32; /* [ws_cap][max(E,I)] */
  float *w8_as;    /* [ws_cap] per-row activation scales */

  sgl_mtp mtp;     /* M3: MTP 草稿头（mtp.ok=0 时全部跳过）*/
  sgl_specsnap snap; /* M3: 投机解码的逐行状态快照（spec_on=0 时全空）*/
  sgl_kvq kvq;       /* M2/M6: 量化分页 KV（kvq.on=0 时全部跳过）*/
  int spec_on;       /* 1 = 正在跑投机验证，lin kernel 要逐行快照 */
  int last_rows;     /* 上一次前向写了几行激活（prefill 末位 hidden 靠它定位）*/
  void *st;        /* llama 模型的 safetensors 上下文（MTP 加载用，不拥有）*/
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

/* out[r] = [a[r] | b[r]]（各 E 宽）：MTP 的 proj_in 按行拼接 */
__global__ void k_cat2_rows(const float *a, const float *b, float *out, int n, int E) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= (long)n * 2 * E) return;
  long r = idx / (2 * E);
  int c = (int)(idx % (2 * E));
  out[idx] = c < E ? a[r * E + c] : b[r * E + (c - E)];
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

/* t11 G13: partial rope —— 只旋转每头前 rot 维（用 per-layer double 逆频率表）*/
__global__ void k_rope_p(float *qk, int n, int head_dim, int heads_per_tok,
                         int pos0, const double *invf, int rot) {
  int half = rot / 2;
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  /* n 是头行数（调用方传 tokens*H），与 k_rope 同约定；上界多乘 heads_per_tok 会让
   * 末块的补齐线程旋转缓冲区之外的头（rows*H*half 不是 256 的倍数时越界）。*/
  if (idx >= (long)n * half) return;
  long tok = idx / ((long)half * heads_per_tok);
  int i = idx % half;
  float *head = qk + (long)(idx / half) * head_dim;
  float ang = (float)((pos0 + tok) * invf[i]);
  float c, s;
  sincosf(ang, &s, &c);
  float a = head[i], b = head[i + half];
  head[i] = a * c - b * s;
  head[i + half] = a * s + b * c;
}

__global__ void k_rope_p_rows(float *qk, int n, int head_dim, const int *pos_row,
                              const double *invf, int rot) {
  int half = rot / 2;
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= (long)n * half) return;
  int r = (int)(idx / half);
  int i = (int)(idx % half);
  float *head = qk + (long)r * head_dim;
  float ang = (float)(pos_row[r] * invf[i]);
  float c, s;
  sincosf(ang, &s, &c);
  float a = head[i], b = head[i + half];
  head[i] = a * c - b * s;
  head[i + half] = a * s + b * c;
}

/* t11 G13: RMSNorm 用 (1+w)（Qwen3NextRMSNorm）*/
__global__ void k_rmsnorm_p1(const float *x, const float *w, float *y, int M, int N,
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
  for (int i = threadIdx.x; i < N; i += blockDim.x) yr[i] = xr[i] * inv * (1.0f + w[i]);
}

/* t11: per-head RMSNorm（q_norm/k_norm，权重 [D]，每头独立归约）。
 * x 行主序 [M][H][D] 压平处理：block = 一个 head。p1=1 时用 (1+w)（qwen3_5）。*/
__global__ void k_hrmsnorm(float *x, const float *w, int M, int H, int D, float eps,
                           int p1) {
  extern __shared__ float red[];
  long hb = blockIdx.x;
  float *hd = x + hb * D;
  float ss = 0;
  for (int i = threadIdx.x; i < D; i += blockDim.x) ss += hd[i] * hd[i];
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  (void)M; (void)H;
  float inv = rsqrtf(red[0] / D + eps);
  if (p1)
    for (int i = threadIdx.x; i < D; i += blockDim.x) hd[i] = hd[i] * inv * (1.0f + w[i]);
  else
    for (int i = threadIdx.x; i < D; i += blockDim.x) hd[i] = hd[i] * inv * w[i];
}

/* t11 G13(a): q_proj 输出 2QD，按头交错 [q(D)|gate(D)] → 拆到 q / gate */
__global__ void k_qgate_split(const float *q2, float *q, float *g, int n, int H, int D) {
  long total = (long)n * H * D;
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= total) return;
  long src = (idx / D) * 2 * D; /* 每 (行,头) 起点（行内已含头序）*/
  int d = (int)(idx % D);
  q[idx] = q2[src + d];
  g[idx] = q2[src + D + d];
}

__global__ void k_qgate_apply(float *a, const float *g, long total) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= total) return;
  a[idx] /= (1.0f + __expf(-g[idx]));
}

/* t11 G9: 因果 depthwise conv + SiLU（交错布局直接寻址，免 gather）。
 * qkvz 行 [nk 组][q kd|k kd|v r*vd|z r*vd]；conv 通道 c 按下面三段映射：
 *   c∈[0,keyd)            → q 段：qkvz[(c/kd)*gw + c%kd]
 *   c∈[keyd,2keyd)        → k 段：qkvz[(c/kd-?)...] —— 组号 g = (c-keyd)/kd
 *   c∈[2keyd, 2keyd+nh*vd) → v 段：g=(c-2keyd)/(r*vd)*/
__global__ void k_lin_conv(const float *cw, float *hist, float *qkvz, float *io,
                           int nk, int kd, int nh, int vd, int ratio, int K,
                           long hoff) {
  int keyd = nk * kd;
  int vtot = nh * vd;
  int convd = 2 * keyd + vtot;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < convd; c += gridDim.x * blockDim.x) {
    int gw = 2 * kd + 2 * ratio * vd;
    long src;
    if (c < keyd) src = (long)(c / kd) * gw + (c % kd);
    else if (c < 2 * keyd) { int g = (c - keyd) / kd; src = (long)g * gw + kd + ((c - keyd) % kd); }
    else { int g = (c - 2 * keyd) / (ratio * vd); src = (long)g * gw + 2 * kd + ((c - 2 * keyd) % (ratio * vd)); }
    float x = qkvz[src];
    const float *wc = cw + (long)c * K;
    float *hc = hist + hoff + (long)c * (K - 1);
    float acc = wc[K - 1] * x;
    for (int j = 0; j < K - 1; j++) acc += wc[j] * hc[j];
    for (int j = 0; j + 1 < K - 1; j++) hc[j] = hc[j + 1];
    hc[K - 2] = x;
    io[c] = acc / (1.0f + __expf(-acc));
  }
}

/* t11 G9: q/k l2norm（每 k-group；q 乘 inv_sqrt_kd，k 不乘）*/
__global__ void k_lin_l2norm(const float *conv, float *qn, float *kn, int nk, int kd) {
  extern __shared__ float red[];
  int g = blockIdx.x / 2, is_q = blockIdx.x % 2;
  const float *src = conv + (is_q ? 0 : (long)nk * kd) + (long)g * kd;
  float *dst = is_q ? qn + (long)g * kd : kn + (long)g * kd;
  float ss = 0;
  for (int i = threadIdx.x; i < kd; i += blockDim.x) ss += src[i] * src[i];
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] + 1e-6f);
  if (is_q) inv /= sqrtf((float)kd);
  for (int i = threadIdx.x; i < kd; i += blockDim.x) dst[i] = src[i] * inv;
}

/* 门控输入 ba[m][j] = h[m] · W_ba[j]（W_ba [nba][E] f32，nba = 2*nh）。grid(nba, M)。
 * 每行的归约顺序与 M 无关：投机 verify（M=n）与 decode（M=1）得到逐位相同的门控。*/
__global__ void k_lin_ba(const float *h, const float *w, float *ba, int nba, int E) {
  extern __shared__ float red[];
  int j = blockIdx.x, m = blockIdx.y;
  const float *x = h + (long)m * E, *wr = w + (long)j * E;
  float s = 0;
  for (int i = threadIdx.x; i < E; i += blockDim.x) s += x[i] * wr[i];
  red[threadIdx.x] = s;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) ba[(long)m * nba + j] = red[0];
}

/* t11 G9: gated deltanet 单步（状态转置布局 [h][d][i]，行访问合并）。
 * block = 128 threads，每 thread 管 d 列（vd<=128 时一一对应；vd>128 循环）。
 * 内存布局：S[h][d*kd + i]；q/k 为 k-group 级（h/ratio 取）。
 * ba 是本 token 的门控输入（k_lin_ba 的一行，组内 [b(r)|a(r)]）：
 * b[h] = ba[g*2r + h%r], a = ba[g*2r + r + h%r] */
__global__ void k_lin_step(float *S, const float *qn, const float *kn, const float *v,
                           const float *z, const float *ba, const float *alog,
                           const float *dtb, float *o, int nh, int kd, int vd,
                           int ratio, int gw, float beta_scratch, long soff,
                           int ntile) {
  /* t11 性能：改成「一个 warp 管一列」。
   * 原来的版本是一个 block 管一个 head、128 个线程各管一个 vd 列，全模型
   * 只有 nh(48) 个 block = 6144 个线程 —— GPU 170 个 SM，每个 SM 平均只有
   * 一个 warp，延迟完全藏不住（实测 1661ms/块，占 prefill 75%）。
   * 现在：grid = nh × (vd/cpb)，每个 warp 负责一列，warp 内 32 个 lane 分摊
   * kd 维并用 __shfl_xor 归约 → 线程数 48×32×128 ≈ 20 万，真正吃满 GPU。
   * 状态布局改为 [vd][kd]（kd 最内层），lane 连续读 row[i] 天然合并。
   * 归约顺序从串行变成蝶式，结果只差浮点末位。*/
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int cpb = blockDim.x >> 5;
  int h = blockIdx.x / ntile;
  int tile = blockIdx.x - h * ntile;
  int d = tile * cpb + warp;
  if (d >= vd) return;
  int g = h / ratio, r2 = 2 * ratio;
  const float *qh = qn + (long)g * kd;
  const float *kh = kn + (long)g * kd;
  (void)nh; (void)z; (void)beta_scratch; /* z 在 gnorm 里用 */
  float bval = ba[(long)g * r2 + (h % ratio)];
  float aval = ba[(long)g * r2 + ratio + (h % ratio)];
  float beta = 1.0f / (1.0f + __expf(-bval));
  float ax = aval + dtb[h];
  float sp = ax > 20.0f ? ax : log1pf(__expf(ax));
  float alpha = __expf(-__expf(alog[h]) * sp);
  float *row = S + soff + (long)h * kd * vd + (long)d * kd;
  float mem = 0;
  for (int i = lane; i < kd; i += 32) {
    float s = row[i] * alpha;
    row[i] = s;
    mem += s * kh[i];
  }
  for (int off = 16; off > 0; off >>= 1) mem += __shfl_xor_sync(0xffffffffu, mem, off);
  float delta = (v[(long)h * vd + d] - mem) * beta;
  float oo = 0;
  for (int i = lane; i < kd; i += 32) {
    float s = row[i] + kh[i] * delta;
    row[i] = s;
    oo += s * qh[i];
  }
  for (int off = 16; off > 0; off >>= 1) oo += __shfl_xor_sync(0xffffffffu, oo, off);
  if (lane == 0) o[(long)h * vd + d] = oo;
}

/* t11 G9: gated RMSNorm：oh = oh*inv*norm_w[d]*silu(z[d])（double 累加与 CPU 一致）*/
__global__ void k_lin_gnorm(float *o, const float *z, const float *nw, int nh, int vd,
                            int kd, int ratio, int gw, float eps) {
  extern __shared__ float red[];
  int h = blockIdx.x;
  float *oh = o + (long)h * vd;
  const float *zh = z + (long)(h / ratio) * gw + 2 * kd + (long)ratio * vd + (long)(h % ratio) * vd;
  (void)nh;
  float ss = 0;
  for (int i = threadIdx.x; i < vd; i += blockDim.x) ss += (double)oh[i] * oh[i];
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / vd + eps);
  for (int d = threadIdx.x; d < vd; d += blockDim.x)
    oh[d] = oh[d] * inv * nw[d] * (zh[d] / (1.0f + __expf(-zh[d])));
}

/* ---- M0: 批量 GDN 四个 kernel（B 行 = B 个独立请求，各用自己的 state slot）----
 * 起因：下面 path=0 的四个 kernel 都是「单 token、单 slot、slot 偏移由 host 传」，
 * 所以 hybrid 模型的 B>1 decode 只能 for(b) 串行跑完 64 层。实测（5090，ctx=2048，
 * lin_slots=16）聚合吞吐恒 36.8 tok/s 与并发完全无关，每请求严格按 1/N 衰减：
 *   conc=1 36.75, conc=2 18.42, conc=4 9.22, conc=8 4.59 tok/s —— 0% 并行度。
 *
 * batch 版把「行」放进 grid.y，slot 从 device 数组 dslot[b] 读 → 一次 launch
 * 处理全部 B 行；slot 偏移在 kernel 内算，host 不再参与，因此也能进 CUDA graph。
 *
 * 刻意用 path=0（逐 token 四段）而不是 path=2 的融合 kernel：融合版假设「n 个
 * token 属于同一序列、同一条递归链」，而 batch decode 是 B 个不同序列各 1 个
 * token，递归互不相关，语义不同。
 * hbase/hstride 事先算好「层偏移」与「槽步长」：
 *   hoff = hbase + slot*hstride,  hbase = ll*lin_slots*convd*(K-1),
 *   hstride = convd*(K-1);  sbase = ll*lin_slots*nh*kd*vd, sstride = nh*kd*vd。*/

/* conv：grid(ceil(convd/128), B)；qkvz [B][rowN]，io [B][convd] */
__global__ void k_lin_conv_b(const float *cw, float *hist, const float *qkvz,
                             float *io, int nk, int kd, int nh, int vd,
                             int ratio, int K, const int *dslot,
                             long hbase, long hstride, int rowN) {
  int b = blockIdx.y;
  int keyd = nk * kd;
  int convd = 2 * keyd + nh * vd;
  int gw = 2 * kd + 2 * ratio * vd;
  const float *qv = qkvz + (long)b * rowN;
  float *oi = io + (long)b * (long)convd;
  float *hof = hist + hbase + (long)(dslot ? dslot[b] : 0) * hstride;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < convd;
       c += gridDim.x * blockDim.x) {
    long src;
    if (c < keyd) src = (long)(c / kd) * gw + (c % kd);
    else if (c < 2 * keyd) { int g = (c - keyd) / kd; src = (long)g * gw + kd + ((c - keyd) % kd); }
    else { int g = (c - 2 * keyd) / (ratio * vd); src = (long)g * gw + 2 * kd + ((c - 2 * keyd) % (ratio * vd)); }
    float x = qv[src];
    const float *wc = cw + (long)c * K;
    float *hc = hof + (long)c * (K - 1);
    float acc = wc[K - 1] * x;
    for (int j = 0; j < K - 1; j++) acc += wc[j] * hc[j];
    for (int j = 0; j + 1 < K - 1; j++) hc[j] = hc[j + 1];
    hc[K - 2] = x;
    oi[c] = acc / (1.0f + __expf(-acc));
  }
}

/* q/k l2norm：grid(2*nk, B) */
__global__ void k_lin_l2norm_b(const float *conv, float *qn, float *kn,
                               int nk, int kd, int convd) {
  extern __shared__ float red[];
  int b = blockIdx.y;
  int keyd = nk * kd;
  int g = blockIdx.x / 2, is_q = blockIdx.x % 2;
  const float *base = conv + (long)b * convd;
  const float *src = base + (is_q ? 0 : (long)keyd) + (long)g * kd;
  float *dst = (is_q ? qn : kn) + (long)b * keyd + (long)g * kd;
  float ss = 0;
  for (int i = threadIdx.x; i < kd; i += blockDim.x) ss += src[i] * src[i];
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] + 1e-6f);
  if (is_q) inv /= sqrtf((float)kd);
  for (int i = threadIdx.x; i < kd; i += blockDim.x) dst[i] = src[i] * inv;
}

/* 递归主体：grid(nh*ntile, B)；v 已按 convd 行步长偏移（传 conv_io 首地址）*/
__global__ void k_lin_step_b(float *S, const float *qn, const float *kn,
                             const float *v, const float *z, const float *ba,
                             const float *alog, const float *dtb, float *o,
                             int nh, int kd, int vd, int ratio,
                             const int *dslot, long sbase, long sstride,
                             int ntile, int keyd, int convd) {
  int b = blockIdx.y;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int cpb = blockDim.x >> 5;
  int h = blockIdx.x / ntile;
  int tile = blockIdx.x - h * ntile;
  int d = tile * cpb + warp;
  if (d >= vd) return;
  int vald = nh * vd;
  int g = h / ratio, r2 = 2 * ratio;
  long soff = sbase + (long)(dslot ? dslot[b] : 0) * sstride;
  const float *qh = qn + (long)b * keyd + (long)g * kd;
  const float *kh = kn + (long)b * keyd + (long)g * kd;
  /* v 段从 conv_io 的第 2*keyd 通道开始（与单 token 版传 l_conv_io+2*keyd 对齐）*/
  const float *vb = v + (long)b * convd + (long)2 * keyd;
  (void)z;
  const float *bab = ba + (long)b * 2 * nh; /* 本行的门控输入 */
  float bval = bab[(long)g * r2 + (h % ratio)];
  float aval = bab[(long)g * r2 + ratio + (h % ratio)];
  float beta = 1.0f / (1.0f + __expf(-bval));
  float ax = aval + dtb[h];
  float sp = ax > 20.0f ? ax : log1pf(__expf(ax));
  float alpha = __expf(-__expf(alog[h]) * sp);
  float *row = S + soff + (long)h * kd * vd + (long)d * kd;
  float mem = 0;
  for (int i = lane; i < kd; i += 32) {
    float s = row[i] * alpha;
    row[i] = s;
    mem += s * kh[i];
  }
  for (int off = 16; off > 0; off >>= 1) mem += __shfl_xor_sync(0xffffffffu, mem, off);
  float delta = (vb[(long)h * vd + d] - mem) * beta;
  float oo = 0;
  for (int i = lane; i < kd; i += 32) {
    float s = row[i] + kh[i] * delta;
    row[i] = s;
    oo += s * qh[i];
  }
  for (int off = 16; off > 0; off >>= 1) oo += __shfl_xor_sync(0xffffffffu, oo, off);
  if (lane == 0) o[(long)b * (long)vald + (long)h * vd + d] = oo;
}

/* gated RMSNorm：grid(nh, B) */
__global__ void k_lin_gnorm_b(float *o, const float *z, const float *nw,
                              int nh, int vd, int kd, int ratio, int gw,
                              float eps, int rowN, int vald) {
  extern __shared__ float red[];
  int b = blockIdx.y;
  int h = blockIdx.x;
  float *oh = o + (long)b * vald + (long)h * vd;
  const float *zh = z + (long)b * rowN + (long)(h / ratio) * gw + 2 * kd +
                    (long)ratio * vd + (long)(h % ratio) * vd;
  (void)nh;
  float ss = 0;
  for (int i = threadIdx.x; i < vd; i += blockDim.x) ss += (double)oh[i] * oh[i];
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / vd + eps);
  for (int d = threadIdx.x; d < vd; d += blockDim.x)
    oh[d] = oh[d] * inv * nw[d] * (zh[d] / (1.0f + __expf(-zh[d])));
}

/* t11 prefill 加速：整层常驻的 gated-deltanet 融合内核。
 * 原来每个 (层, token) 要 4 次 kernel launch（conv/l2norm/step/gnorm），
 * 一块 1024 token 就是 48×1024×4 ≈ 20 万次 launch；而且状态是 3.1MB/层，
 * 每 token 要从显存读一遍再写一遍（48×1024×6.2MB ≈ 306GB）。实测 lin-rec
 * 639ms，几乎全是 launch 开销 + 状态往返。
 * 现在：grid = nh（一个 block 一个 v-head），状态和 conv 历史常驻 shared，
 * 四段在同 block 内用 __syncthreads 串起来，内部循环整块的 token ——
 * 每层只需 1 次 launch，状态全程不碰显存。
 * 数学与时序契约与 k_lin_conv / k_lin_l2norm / k_lin_step / k_lin_gnorm 一致。*/
__global__ void k_lin_layer_persist(
    const float *qkvz_all, const float *cw, float *hist, float *S,
    const float *ba, const float *alog, const float *dtb, const float *nw,
    float *o_all, int n, int nk, int kd, int nh, int vd, int K, int ratio,
    int gw, float eps, long soff, long hoff) {
  int h = blockIdx.x;
  int g = h / ratio, r = h % ratio;
  int keyd = nk * kd;
  int rowN = 2 * keyd + 2 * nh * vd;
  int ovd = nh * vd;
  int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ float sh[];
  float *St = sh;                        /* [kd][vd] 递归状态（线程 d 读 St[i*vd+d]）*/
  float *hq = St + (long)kd * vd;        /* [(K-1)][kd] conv 历史 */
  float *hk = hq + (long)(K - 1) * kd;
  float *hv = hk + (long)(K - 1) * kd;   /* [(K-1)][vd] */
  float *cq = hv + (long)(K - 1) * vd;   /* [kd] conv+SiLU 后的 q */
  float *ck = cq + kd;                   /* [kd] */
  float *cv = ck + kd;                   /* [vd] */
  float *ov = cv + vd;                   /* [vd] 本 token 输出（gnorm 要整行）*/
  float *red = ov + vd;                  /* [2*nt] 归约用 */
  const long sbase = (long)h * kd * vd;  /* 全局状态按 [vd][kd] 存（与 k_lin_step 一致）*/
  const long qsrc = (long)g * gw;
  const long ksrc = qsrc + kd;
  const long vsrc = qsrc + 2 * kd + (long)r * vd;
  const long zsrc = qsrc + 2 * kd + (long)ratio * vd + (long)r * vd;
  const float *wcq = cw + (long)(g * kd) * K;
  const float *wck = cw + (long)(keyd + g * kd) * K;
  const float *wcv = cw + (long)(2 * keyd + h * vd) * K;

  /* 载入状态（[vd][kd] → shared [kd][vd]，转置一次，每层一次）*/
  for (int d = tid; d < vd; d += nt)
    for (int i = 0; i < kd; i++) St[(long)i * vd + d] = S[soff + sbase + (long)d * kd + i];
  if (tid < kd)
    for (int j = 0; j < K - 1; j++) {
      hq[(long)j * kd + tid] = hist[hoff + (long)(g * kd + tid) * (K - 1) + j];
      hk[(long)j * kd + tid] = hist[hoff + (long)(keyd + g * kd + tid) * (K - 1) + j];
    }
  if (tid < vd)
    for (int j = 0; j < K - 1; j++)
      hv[(long)j * vd + tid] = hist[hoff + (long)(2 * keyd + h * vd + tid) * (K - 1) + j];
  __syncthreads();

  for (int t = 0; t < n; t++) {
    const float *row = qkvz_all + (long)t * rowN;
    const float *bat = ba + (long)t * 2 * nh; /* 门控随 token 变（第 t 行）*/
    float beta = 1.0f / (1.0f + __expf(-bat[(long)g * 2 * ratio + r]));
    float ax = bat[(long)g * 2 * ratio + ratio + r] + dtb[h];
    float sp = ax > 20.0f ? ax : log1pf(__expf(ax));
    float alpha = __expf(-__expf(alog[h]) * sp);
    /* --- 因果 depthwise conv + SiLU --- */
    if (tid < kd) {
      float xq = row[qsrc + tid], xk = row[ksrc + tid];
      float aq = wcq[(long)tid * K + (K - 1)] * xq;
      float ak = wck[(long)tid * K + (K - 1)] * xk;
      for (int j = 0; j < K - 1; j++) {
        aq += wcq[(long)tid * K + j] * hq[(long)j * kd + tid];
        ak += wck[(long)tid * K + j] * hk[(long)j * kd + tid];
      }
      for (int j = 0; j + 1 < K - 1; j++) {
        hq[(long)j * kd + tid] = hq[(long)(j + 1) * kd + tid];
        hk[(long)j * kd + tid] = hk[(long)(j + 1) * kd + tid];
      }
      if (K >= 2) {
        hq[(long)(K - 2) * kd + tid] = xq;
        hk[(long)(K - 2) * kd + tid] = xk;
      }
      cq[tid] = aq / (1.0f + __expf(-aq));
      ck[tid] = ak / (1.0f + __expf(-ak));
    }
    if (tid < vd) {
      float xv = row[vsrc + tid];
      float av = wcv[(long)tid * K + (K - 1)] * xv;
      for (int j = 0; j < K - 1; j++) av += wcv[(long)tid * K + j] * hv[(long)j * vd + tid];
      for (int j = 0; j + 1 < K - 1; j++) hv[(long)j * vd + tid] = hv[(long)(j + 1) * vd + tid];
      if (K >= 2) hv[(long)(K - 2) * vd + tid] = xv;
      cv[tid] = av / (1.0f + __expf(-av));
    }
    __syncthreads();
    /* --- q/k l2norm（每 k-group，block 内归约）--- */
    red[tid] = (tid < kd) ? cq[tid] * cq[tid] : 0.0f;
    red[nt + tid] = (tid < kd) ? ck[tid] * ck[tid] : 0.0f;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
      if (tid < s) { red[tid] += red[tid + s]; red[nt + tid] += red[nt + tid + s]; }
      __syncthreads();
    }
    float invq = rsqrtf(red[0] + 1e-6f) / sqrtf((float)kd);
    float invk = rsqrtf(red[nt] + 1e-6f);
    if (tid < kd) { cq[tid] *= invq; ck[tid] *= invk; }
    __syncthreads();
    /* --- delta-rule 递归一步（状态在 shared）--- */
    if (tid < vd) {
      float mem = 0.0f;
      for (int i = 0; i < kd; i++) {
        long idx = (long)i * vd + tid;
        float s = St[idx] * alpha;
        St[idx] = s;
        mem += s * ck[i];
      }
      float delta = (cv[tid] - mem) * beta;
      float oo = 0.0f;
      for (int i = 0; i < kd; i++) {
        long idx = (long)i * vd + tid;
        float s = St[idx] + ck[i] * delta;
        St[idx] = s;
        oo += s * cq[i];
      }
      ov[tid] = oo;
    }
    __syncthreads();
    /* --- gated RMSNorm + SiLU(z) --- */
    red[tid] = (tid < vd) ? ov[tid] * ov[tid] : 0.0f;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float inv = rsqrtf(red[0] / vd + eps);
    if (tid < vd) {
      float z = row[zsrc + tid];
      o_all[(long)t * ovd + h * vd + tid] =
          ov[tid] * inv * nw[tid] * (z / (1.0f + __expf(-z)));
    }
    __syncthreads();
  }

  /* 状态 / 历史 写回（shared [kd][vd] → 全局 [vd][kd]）*/
  for (int d = tid; d < vd; d += nt)
    for (int i = 0; i < kd; i++) S[soff + sbase + (long)d * kd + i] = St[(long)i * vd + d];
  if (tid < kd)
    for (int j = 0; j < K - 1; j++) {
      hist[hoff + (long)(g * kd + tid) * (K - 1) + j] = hq[(long)j * kd + tid];
      hist[hoff + (long)(keyd + g * kd + tid) * (K - 1) + j] = hk[(long)j * kd + tid];
    }
  if (tid < vd)
    for (int j = 0; j < K - 1; j++)
      hist[hoff + (long)(2 * keyd + h * vd + tid) * (K - 1) + j] = hv[(long)j * vd + tid];
}

/* t11 prefill 加速 v2：整层常驻 + 递归状态放【寄存器】。
 * v1 把状态放 shared：一个 block 只有 4 个 warp，且每 token 要 2×kd 次
 * shared 读改写（每线程 768 条指令、依赖链完全裸露），实测 363ms。
 * v2：block = 512 线程，一个 block 仍管一个 v-head 的 VD 列，但 i（kd）方向
 * 再拆 4 组，每组 128 线程管 CH=kd/4 个 i —— 状态就 32 个 float，直接驻留
 * 寄存器（循环全展开、下标全是编译期常量）。每 token 每线程只剩
 * CH 次广播读 + CH 次 FFMA，跨组用 shared 交换部分和。
 * 要求 kd == vd == 128（qwen3.8 的 lin 层就是 128/128），不满足时调用方回退。*/
template <int KD, int VD>
__global__ __launch_bounds__(512) void k_lin_persist_reg(
    const float *qkvz_all, const float *cw, float *hist, float *S,
    const float *ba, const float *alog, const float *dtb, const float *nw,
    float *o_all, int n, int nk, int nh, int K, int ratio, int gw, float eps,
    long soff, long hoff, float *snapS, float *snapH, long snapS_rs, long snapH_rs) {
  const int G = 4;         /* i 方向分组数 */
  const int CH = KD / G;   /* 每组负责的 i 个数 */
  const int NW = KD / 32;  /* 归约参与的 warp 数 */
  int h = blockIdx.x;
  int g = h / ratio, r = h % ratio;
  int keyd = nk * KD;
  int rowN = 2 * keyd + 2 * nh * VD;
  int ovd = nh * VD;
  int tid = threadIdx.x;
  int col = tid % VD;
  int grp = tid / VD;
  int i0 = grp * CH;
  const long sbase = (long)h * KD * VD;
  const long qsrc = (long)g * gw;
  const long ksrc = qsrc + KD;
  const long vsrc = qsrc + 2 * KD + (long)r * VD;
  const long zsrc = qsrc + 2 * KD + (long)ratio * VD + (long)r * VD;
  const float *wcq = cw + (long)(g * KD) * K;
  const float *wck = cw + (long)(keyd + g * KD) * K;
  const float *wcv = cw + (long)(2 * keyd + h * VD) * K;

  extern __shared__ float sh[];
  float *hq = sh;                       /* [(K-1)][KD] */
  float *hk = hq + (long)(K - 1) * KD;
  float *hv = hk + (long)(K - 1) * KD;  /* [(K-1)][VD] */
  float *cq = hv + (long)(K - 1) * VD;  /* [KD] */
  float *ck = cq + KD;                  /* [KD] */
  float *cv = ck + KD;                  /* [VD] */
  float *ov = cv + VD;                  /* [VD] */
  float *pmem = ov + VD;                /* [G][VD] */
  float *poo = pmem + (long)G * VD;     /* [G][VD] */
  float *pdelta = poo + (long)G * VD;   /* [VD] */
  float *red = pdelta + VD;             /* [2*NW] */

  /* 状态进寄存器（全局布局 [vd][kd]，本线程的 CH 个 i 连续）*/
  float st[CH];
#pragma unroll
  for (int k = 0; k < CH; k++)
    st[k] = S[soff + sbase + (long)col * KD + (i0 + k)];

  if (tid < KD)
    for (int j = 0; j < K - 1; j++) {
      hq[(long)j * KD + tid] = hist[hoff + (long)(g * KD + tid) * (K - 1) + j];
      hk[(long)j * KD + tid] = hist[hoff + (long)(keyd + g * KD + tid) * (K - 1) + j];
    }
  if (tid >= KD && tid < KD + VD)
    for (int j = 0; j < K - 1; j++)
      hv[(long)j * VD + col] = hist[hoff + (long)(2 * keyd + h * VD + col) * (K - 1) + j];
  __syncthreads();

  for (int t = 0; t < n; t++) {
    const float *row = qkvz_all + (long)t * rowN;
    const float *bat = ba + (long)t * 2 * nh; /* 门控随 token 变（第 t 行）*/
    float beta = 1.0f / (1.0f + __expf(-bat[(long)g * 2 * ratio + r]));
    float ax = bat[(long)g * 2 * ratio + ratio + r] + dtb[h];
    float sp = ax > 20.0f ? ax : log1pf(__expf(ax));
    float alpha = __expf(-__expf(alog[h]) * sp);
    /* --- 因果 depthwise conv + SiLU（第 0 组算 q/k，第 1 组算 v）--- */
    if (tid < KD) {
      float xq = row[qsrc + tid], xk = row[ksrc + tid];
      float aq = wcq[(long)tid * K + (K - 1)] * xq;
      float ak = wck[(long)tid * K + (K - 1)] * xk;
      for (int j = 0; j < K - 1; j++) {
        aq += wcq[(long)tid * K + j] * hq[(long)j * KD + tid];
        ak += wck[(long)tid * K + j] * hk[(long)j * KD + tid];
      }
      for (int j = 0; j + 1 < K - 1; j++) {
        hq[(long)j * KD + tid] = hq[(long)(j + 1) * KD + tid];
        hk[(long)j * KD + tid] = hk[(long)(j + 1) * KD + tid];
      }
      if (K >= 2) {
        hq[(long)(K - 2) * KD + tid] = xq;
        hk[(long)(K - 2) * KD + tid] = xk;
      }
      cq[tid] = aq / (1.0f + __expf(-aq));
      ck[tid] = ak / (1.0f + __expf(-ak));
    } else if (tid < KD + VD) {
      float xv = row[vsrc + col];
      float av = wcv[(long)col * K + (K - 1)] * xv;
      for (int j = 0; j < K - 1; j++) av += wcv[(long)col * K + j] * hv[(long)j * VD + col];
      for (int j = 0; j + 1 < K - 1; j++) hv[(long)j * VD + col] = hv[(long)(j + 1) * VD + col];
      if (K >= 2) hv[(long)(K - 2) * VD + col] = xv;
      cv[col] = av / (1.0f + __expf(-av));
    }
    __syncthreads();
    /* --- q/k l2norm（warp 内蝶式 + 跨 warp 汇总）--- */
    if (tid < KD) {
      float a = cq[tid] * cq[tid], b = ck[tid] * ck[tid];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, off);
        b += __shfl_xor_sync(0xffffffffu, b, off);
      }
      if ((tid & 31) == 0) { red[(tid >> 5) * 2] = a; red[(tid >> 5) * 2 + 1] = b; }
    }
    __syncthreads();
    if (tid < KD) {
      float sa = 0.0f, sb = 0.0f;
#pragma unroll
      for (int w2 = 0; w2 < NW; w2++) { sa += red[w2 * 2]; sb += red[w2 * 2 + 1]; }
      cq[tid] *= rsqrtf(sa + 1e-6f) / sqrtf((float)KD);
      ck[tid] *= rsqrtf(sb + 1e-6f);
    }
    __syncthreads();
    /* --- delta-rule 递归一步（状态在寄存器，跨组交换部分和）--- */
    float mem = 0.0f;
#pragma unroll
    for (int k = 0; k < CH; k++) { st[k] *= alpha; mem += st[k] * ck[i0 + k]; }
    pmem[grp * VD + col] = mem;
    __syncthreads();
    if (tid < VD) {
      float t2 = 0.0f;
#pragma unroll
      for (int w2 = 0; w2 < G; w2++) t2 += pmem[w2 * VD + tid];
      pdelta[tid] = (cv[tid] - t2) * beta;
    }
    __syncthreads();
    float delta = pdelta[col];
    float oo = 0.0f;
#pragma unroll
    for (int k = 0; k < CH; k++) { st[k] += ck[i0 + k] * delta; oo += st[k] * cq[i0 + k]; }
    poo[grp * VD + col] = oo;
    __syncthreads();
    /* --- gated RMSNorm + SiLU(z)：先把 G 份部分和合成整行 --- */
    if (tid < VD) {
      float t2 = 0.0f;
#pragma unroll
      for (int w2 = 0; w2 < G; w2++) t2 += poo[w2 * VD + tid];
      ov[tid] = t2;
      float a = t2 * t2;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
      if ((tid & 31) == 0) red[tid >> 5] = a;
    }
    __syncthreads();
    if (tid < VD) {
      float ss = 0.0f;
#pragma unroll
      for (int w2 = 0; w2 < NW; w2++) ss += red[w2];
      float inv = rsqrtf(ss / VD + eps);
      float z = row[zsrc + tid];
      o_all[(long)t * ovd + h * VD + tid] =
          ov[tid] * inv * nw[tid] * (z / (1.0f + __expf(-z)));
    }
    __syncthreads();
    /* M3：逐行快照（只在本行处理完之后写）。snapS 指向本层的区段、行步长
     * snapS_rs，所以 snapS + t*snapS_rs 就是「第 t 行处理完后」的状态。
     * 拒绝部分草稿时靠它把状态回滚到精确的 k 行。*/
    if (snapS) {
      long so2 = (long)t * snapS_rs + sbase;
#pragma unroll
      for (int k = 0; k < CH; k++) snapS[so2 + (long)col * KD + (i0 + k)] = st[k];
      long ho2 = (long)t * snapH_rs;
      if (tid < KD)
        for (int j = 0; j < K - 1; j++) {
          snapH[ho2 + (long)(g * KD + tid) * (K - 1) + j] = hq[(long)j * KD + tid];
          snapH[ho2 + (long)(keyd + g * KD + tid) * (K - 1) + j] = hk[(long)j * KD + tid];
        }
      if (tid >= KD && tid < KD + VD)
        for (int j = 0; j < K - 1; j++)
          snapH[ho2 + (long)(2 * keyd + h * VD + col) * (K - 1) + j] = hv[(long)j * VD + col];
    }
  }
#pragma unroll
  for (int k = 0; k < CH; k++) S[soff + sbase + (long)col * KD + (i0 + k)] = st[k];
  if (tid < KD)
    for (int j = 0; j < K - 1; j++) {
      hist[hoff + (long)(g * KD + tid) * (K - 1) + j] = hq[(long)j * KD + tid];
      hist[hoff + (long)(keyd + g * KD + tid) * (K - 1) + j] = hk[(long)j * KD + tid];
    }
  if (tid >= KD && tid < KD + VD)
    for (int j = 0; j < K - 1; j++)
      hist[hoff + (long)(2 * keyd + h * VD + col) * (K - 1) + j] = hv[(long)j * VD + col];
}


/* ================= M2/M6：量化 KV 的存取 kernel ================= */

/* 把一个量化块（SGL_KVQ_BLK 个值）的 amax 归约出来（block 内 128 线程）*/
__device__ __forceinline__ float kvq_amax(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  __shared__ float sm[4];
  int w = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) sm[w] = v;
  __syncthreads();
  v = sm[0];
  if (w == 0) {
    float t = (threadIdx.x < 4) ? sm[threadIdx.x] : 0.0f;
#pragma unroll
    for (int off = 2; off > 0; off >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, off));
    sm[threadIdx.x] = t;
  }
  __syncthreads();
  return sm[0];
}

/* 写入：把 [n][HK*D] 的 k/v（已做完 qk-norm + rope）量化进分页池。
 * grid = n * (2 * nvals/128)，block = 128 线程；一个 block 处理一个量化块。
 * bpos<0：用 pos = base[t] + lens[t]（decode 批量清晰路径）。
 * slot<0（页未驻留）时直接跳过：权威副本在 host（M6 写侧另写）。*/
__global__ void k_kvq_write(unsigned char *pool, const int *pt, const float *k, const float *v,
                            int n, int pos0, int kv_base, int HK, int D, int P,
                            long slot_layer, long layer_slots, long tok_stride,
                            const int *base, const int *lens) {
  int bpb = 2 * ((HK * D) / SGL_KVQ_BLK); /* 每 token 的量化块数 */
  int blk = blockIdx.x;
  int t = blk / bpb, sub = blk % bpb;
  int isv = sub / ((HK * D) / SGL_KVQ_BLK);
  int qb = sub % ((HK * D) / SGL_KVQ_BLK);
  int nvals = HK * D;
  int pos = base ? (base[t] + lens[t]) : (kv_base + pos0 + t);
  int page = pos / P, off = pos - page * P;
  int slot = pt ? pt[page] : page;
  if (slot < 0) return;
  const float *src = (isv ? v : k) + (long)t * nvals + (long)qb * SGL_KVQ_BLK;
  float x = src[threadIdx.x];
  float amax = kvq_amax(fabsf(x));
  float sc = amax > 0.0f ? amax / 7.0f : 1.0f;
  int q = __float2int_rn(x / sc);
  if (q > 7) q = 7;
  if (q < -7) q = -7;
  int qp = __shfl_xor_sync(0xffffffffu, q, 1); /* 配对成 1 字节 */
  unsigned char *dst = pool + (long)slot * slot_layer + (long)off * tok_stride;
  int qbase = isv ? (nvals / 2) : 0;
  if ((threadIdx.x & 1) == 0)
    dst[qbase + (qb * SGL_KVQ_BLK + threadIdx.x) / 2] =
        (unsigned char)((q & 0xF) | ((qp & 0xF) << 4));
  /* scale：每块一个 float */
  if (threadIdx.x == 0) {
    int nblk = nvals / SGL_KVQ_BLK;
    int sbase = nvals + (isv ? nblk * 4 : 0);
    *(float *)(dst + sbase + qb * 4) = sc;
  }
}

/* 读一个 4-bit 值（dq 必须是偶数起点成对读，外面循环步长 2）*/
__device__ __forceinline__ void kvq_deq2(const unsigned char *base, int d, int nvals,
                                         float *o0, float *o1) {
  unsigned char b = base[d >> 1];
  int n0 = (b & 0xF), n1 = (b >> 4);
  n0 = (n0 << 28) >> 28;
  n1 = (n1 << 28) >> 28;
  int nblk = nvals / SGL_KVQ_BLK;
  float sc = *(const float *)(base + nvals + (d >> 7) * 4);
  *o0 = (float)n0 * sc;
  *o1 = (float)n1 * sc;
}

/* 单 KN 值（attention 里按需取用，d 任意）*/
__device__ __forceinline__ float kvq_deq1(const unsigned char *base, int d, int nvals) {
  unsigned char b = base[d >> 1];
  int q = (d & 1) ? (b >> 4) : (b & 0xF);
  q = (q << 28) >> 28;
  float sc = *(const float *)(base + nvals + (d >> 7) * 4);
  return (float)q * sc;
}

__device__ __forceinline__ float kvq_deqv(const unsigned char *base, int d, int nvals) {
  int half = nvals / 2;
  unsigned char b = base[half + (d >> 1)];
  int q = (d & 1) ? (b >> 4) : (b & 0xF);
  q = (q << 28) >> 28;
  int nblk = nvals / SGL_KVQ_BLK;
  float sc = *(const float *)(base + nvals + nblk * 4 + (d >> 7) * 4);
  return (float)q * sc;
}

/* M2：分页 + 4-bit 反量化的 attention（prefill 用，两遍法）*/
/* M2：分页 + 4-bit 反量化的 flash attention（warp-per-(t,h)，单遍 online softmax）。
 *
 * 为什么重写：第一版按「每线程一个 p、逐值标量反量化」写，每个值要算两次
 * 整数除法（page/off）+ 读一次 scale，实测 11K 上下文 decode 147 ms/token
 * （6.8 tok/s），差异 20 倍以上。这里改成：
 *   - 一个 warp 包一个 (t, h)；lane 负责连续的 ND=D/32 个维度 → int4 一次读 4 B
 *   - scale 每 128 值一个：D=256、ND=8 时每 16 个 lane 共享一个 scale
 *   - 单遍 online softmax（不再读两遍 KV）
 *   - 按页循环：同页的 64 个位置只查一次页表
 * 要求 D % 32 == 0（不满足时调用方退回k_attn_q）。*/
#define KVQ_FLASH_WARPS 4
__global__ void k_attn_q_flash(const float *q, const unsigned char *pool, const int *pt,
                               float *out, int pos0, int kv_base, int n, int H, int HK, int D,
                               float scale, int win, int P, int nvals, long slot_layer,
                               long tok_stride) {
  const int ND = D / 32;
  const int nblk = nvals / SGL_KVQ_BLK;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int th = blockIdx.x * KVQ_FLASH_WARPS + warp;
  if (th >= n * H) return;
  int t = th / H, h = th % H;
  int upto = pos0 + t + 1;
  int hkv = h / (H / HK);
  int from = (win > 0 && upto > win) ? upto - win : 0;
  const float *qh = q + ((long)t * H + h) * D;
  /* lane 负责 d = lane*ND .. +ND-1（连续） => int4 恰好 4 字节 */
  float qr[16];
  if (ND > 16) return;
#pragma unroll
  for (int i = 0; i < ND; i++) qr[i] = qh[lane * ND + i];
  int vbase = hkv * D + lane * ND; /* 该 lane 的全局值起点 */
  int kbyte = vbase >> 1;          /* K 区字节偏移（相对 token 起点）*/
  int vbyte = (nvals >> 1) + (vbase >> 1);
  int sblk_k = vbase >> 7; /* scale 块号 */
  int sblk_v = sblk_k;
  unsigned sfree = 0;
  (void)sfree;

  float m = -1e30f, lsum = 0.0f;
  float acc[16];
#pragma unroll
  for (int i = 0; i < ND; i++) acc[i] = 0.0f;

  int p = from;
  while (p < upto) {
    int gpos = kv_base + p;
    int page = gpos / P, off = gpos - page * P;
    int pend = (page + 1) * P - kv_base;
    if (pend > upto) pend = upto;
    int slot = pt[page];
    if (slot < 0) { /* 未驻留：跳过（M6 先 resolve 再跑）*/
      p = pend;
      continue;
    }
    const unsigned char *pg = pool + (long)slot * slot_layer;
    for (; p < pend; p++, off++) {
      const unsigned char *kb = pg + (long)off * tok_stride;
      /* ---- QK^T：lane 读 4 字节 = 8 个 int4 ---- */
      float szk = *(const float *)(kb + nvals + sblk_k * 4);
      float dot = 0.0f;
      {
        const unsigned char *bp = kb + kbyte;
#pragma unroll
        for (int j = 0; j < ND; j += 2) {
          unsigned char by = bp[j >> 1];
          int n0 = (by & 0xF), n1 = (by >> 4);
          n0 = (n0 << 28) >> 28;
          n1 = (n1 << 28) >> 28;
          dot += qr[j] * (float)n0 + qr[j + 1] * (float)n1;
        }
        dot *= szk;
      }
#pragma unroll
      for (int off2 = 16; off2 > 0; off2 >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off2);
      float s = dot * scale;
      /* ---- online softmax + V 累加 ---- */
      float mn = fmaxf(m, s);
      float alpha = __expf(m - mn);
      float e = __expf(s - mn);
      lsum = lsum * alpha + e;
      float szv = *(const float *)(kb + nvals + nblk * 4 + sblk_v * 4);
      const unsigned char *bp = kb + vbyte;
#pragma unroll
      for (int j = 0; j < ND; j += 2) {
        unsigned char by = bp[j >> 1];
        int n0 = (by & 0xF), n1 = (by >> 4);
        n0 = (n0 << 28) >> 28;
        n1 = (n1 << 28) >> 28;
        acc[j] = acc[j] * alpha + e * ((float)n0 * szv);
        acc[j + 1] = acc[j + 1] * alpha + e * ((float)n1 * szv);
      }
      m = mn;
    }
  }
  float inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;
  float *o = out + ((long)t * H + h) * D;
#pragma unroll
  for (int i = 0; i < ND; i++) o[lane * ND + i] = acc[i] * inv;
}

__global__ void k_attn_q(const float *q, const unsigned char *pool, const int *pt,
                         float *out, int pos0, int kv_base, int n, int H, int HK, int D,
                         float scale, int win, int P, int nvals, long slot_layer,
                         long tok_stride) {
  __shared__ float sc[SGL_ATTN_CHUNK];
  __shared__ float red[256];
  int th = blockIdx.x;
  int t = th / H, h = th % H;
  int upto = pos0 + t + 1;
  const float *qh = q + ((long)t * H + h) * D;
  int hkv = h / (H / HK);
  int from = (win > 0 && upto > win) ? upto - win : 0;
  int koff = (hkv * D) / 2; /* K 区起点（配对打包）*/
  int voff = nvals / 2 + (hkv * D) / 2;

  /* pass 1：全局 max + 在线归一化因子 */
  float run_max = -1e30f, run_sum = 0.0f;
  for (int c0 = from; c0 < upto; c0 += SGL_ATTN_CHUNK) {
    int ce = c0 + SGL_ATTN_CHUNK; if (ce > upto) ce = upto;
    float cmax = -1e30f;
    for (int p = c0 + threadIdx.x; p < ce; p += blockDim.x) {
      int gpos = kv_base + p;
      int page = gpos / P, off = gpos - page * P;
      int slot = pt[page];
      float s = 0;
      if (slot >= 0) {
        const unsigned char *kb = pool + (long)slot * slot_layer + (long)off * tok_stride;
        for (int d = 0; d < D; d += 2) {
          float v0, v1;
          const unsigned char *bb = kb + koff + (d >> 1);
          unsigned char by = *bb;
          int n0 = (by & 0xF), n1 = (by >> 4);
          n0 = (n0 << 28) >> 28; n1 = (n1 << 28) >> 28;
          float sc0 = *(const float *)(kb + nvals + ((hkv * D + d) >> 7) * 4);
          v0 = (float)n0 * sc0; v1 = (float)n1 * sc0;
          s += qh[d] * v0 + qh[d + 1] * v1;
        }
        s *= scale;
      } else
        s = -1e30f;
      sc[p - c0] = s;
      if (s > cmax) cmax = s;
    }
    red[threadIdx.x] = cmax;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
      if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
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
    __syncthreads();
    red[threadIdx.x] = cs;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
      if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
      __syncthreads();
    }
    run_sum = run_sum * alpha + red[0];
    run_max = cmax;
    __syncthreads();
  }
  float invsum = 1.0f / run_sum;

  /* pass 2：用最终 max 重算权重，累加 V */
  float *o = out + ((long)t * H + h) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] = 0;
  __syncthreads();
  for (int c0 = from; c0 < upto; c0 += SGL_ATTN_CHUNK) {
    int ce = c0 + SGL_ATTN_CHUNK; if (ce > upto) ce = upto;
    for (int p = c0 + threadIdx.x; p < ce; p += blockDim.x) {
      int gpos = kv_base + p;
      int page = gpos / P, off = gpos - page * P;
      int slot = pt[page];
      float s = 0;
      if (slot >= 0) {
        const unsigned char *kb = pool + (long)slot * slot_layer + (long)off * tok_stride;
        for (int d = 0; d < D; d += 2) {
          const unsigned char *bb = kb + koff + (d >> 1);
          unsigned char by = *bb;
          int n0 = (by & 0xF), n1 = (by >> 4);
          n0 = (n0 << 28) >> 28; n1 = (n1 << 28) >> 28;
          float sc0 = *(const float *)(kb + nvals + ((hkv * D + d) >> 7) * 4);
          s += qh[d] * ((float)n0 * sc0) + qh[d + 1] * ((float)n1 * sc0);
        }
        s *= scale;
      } else
        s = -1e30f;
      sc[p - c0] = __expf(s - run_max);
    }
    __syncthreads();
    for (int p = c0; p < ce; p++) {
      int gpos = kv_base + p;
      int page = gpos / P, off = gpos - page * P;
      int slot = pt[page];
      if (slot < 0) continue;
      const unsigned char *vb = pool + (long)slot * slot_layer + (long)off * tok_stride;
      float w = sc[p - c0];
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const unsigned char *bb = vb + voff + (d >> 1);
        unsigned char by = *bb;
        int q4 = (d & 1) ? (by >> 4) : (by & 0xF);
        q4 = (q4 << 28) >> 28;
        float scv = *(const float *)(vb + nvals + (nvals / SGL_KVQ_BLK) * 4 +
                                    ((hkv * D + d) >> 7) * 4);
        o[d] += w * ((float)q4 * scv);
      }
    }
    __syncthreads();
  }
  for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] *= invsum;
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
template <int D, int QPW, int TK, bool VG = false>
__global__ __launch_bounds__(256) void k_attn_flash(
    const float *q, const float *kc, const float *vc, float *out,
    int pos0, int kv_base, int n, int H, int HK, float scale, int win) {
  const int TQ = 8 * QPW;
  const int ND = D / 32;
  /* t11：改成动态 shared —— TK=32（D=256） 需要 74KB，超过 48KB 静态上限。
   * 行指针写法让下面所有 Qs[qi][d] / Ks[p][d] 的下标不用改。
   * VG=true 时 V 不进 shared（不从 smem 划大小）：加权和里每个 lane 读的是
   * Vs[pp][lane+32*nd]，warp 内 32 个 lane 正好连续 → 直接读全局就是合并访问，
   * 且同一个 hkv 被 H/HK 个 head 共享，L1/L2 复用率很高。省出来的 32*257*4 B
   * 让 TK 能开到 32 而仍保 2 blocks/SM（41KB）。*/
  extern __shared__ float smem[];
  float (*Qs)[D + 1] = reinterpret_cast<float (*)[D + 1]>(smem);
  float (*Ks)[D + 1] = reinterpret_cast<float (*)[D + 1]>(smem + (long)TQ * (D + 1));
  float (*Vs)[D + 1] = VG ? nullptr
                          : reinterpret_cast<float (*)[D + 1]>(smem + (long)(TQ + TK) * (D + 1));

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
        if (!VG) Vs[p][d] = vc[o];
      } else {
        Ks[p][d] = 0.0f;
        if (!VG) Vs[p][d] = 0.0f;
      }
    }
    __syncthreads();

    int gp = c0 + lane;
    /* t11: lane 跑 0..31，但 Ks 只有 TK 行——原来的 Ks[lane] 在 TK<32 时
     * 会读出界（TK=16 靠后面的 Vs 塔住侥幸不崩，TK=8 就直接越过整个
     * static shared 报 illegal memory access）。这里用位与钳到有效行：
     * 超出 TK 的 lane 本来就在下面被置成 -1e30，读到的垃圾不影响结果。
     * TK 必须是 2 的幂（当前实例 8/16/32 都满足）。*/
    const float *krow = Ks[lane & (TK - 1)];
    float e[QPW];
#pragma unroll
    for (int i = 0; i < QPW; i++) {
      int qi = w * QPW + i;
      int tq = tq0 + qi;
      float s = 0.0f;
#pragma unroll
      for (int d = 0; d < D; d++) s += Qs[qi][d] * krow[d];
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
      /* VG：超出本块有效范围的 pp 在 shared 版里是靠 Vs=0 消掉的，
       * 直读全局时不能去碰（会读越 KV 池），直接跳过。*/
      if (VG && c0 + pp >= ce) continue;
      const float *vrow = VG ? (vc + ((long)(kv_base + c0 + pp) * HK + hkv) * D) : nullptr;
#pragma unroll
      for (int i = 0; i < QPW; i++) {
        float wp = __shfl_sync(0xffffffffu, e[i], pp);
#pragma unroll
        for (int nd = 0; nd < ND; nd++)
          acc[i][nd] += wp * (VG ? vrow[lane + 32 * nd] : Vs[pp][lane + 32 * nd]);
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
  const float *kc = g->kc + (long)g->kvo[l] * g->ctx * KD;
  const float *vc = g->vc + (long)g->kvo[l] * g->ctx * KD;
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
    k_attn_flash<64, 4, 32><<<(n + TQ - 1) / TQ * H, 256,
                               (size_t)(TQ + 2 * 32) * 65 * sizeof(float), s>>>(
        g->q, kc, vc, g->attn, pos0, kv_base, n, H, HK, scale, win);
  } else if (D == 128) {
    const int TQ = 8 * 2;
    if (getenv("SGL_ATTN_DEBUG"))
      fprintf(stderr, "[attn] flash D=128 QPW=2 TK=16 n=%d pos0=%d\n", n, pos0);
    k_attn_flash<128, 2, 16><<<(n + TQ - 1) / TQ * H, 256,
                                (size_t)(TQ + 2 * 16) * 129 * sizeof(float), s>>>(
        g->q, kc, vc, g->attn, pos0, kv_base, n, H, HK, scale, win);
  } else if (D == 96) {
    /* shared = 16*(97) + 2*32*(97) floats ≈ 30.3 KB */
    const int TQ = 8 * 2;
    if (getenv("SGL_ATTN_DEBUG"))
      fprintf(stderr, "[attn] flash D=96 QPW=2 TK=32 n=%d pos0=%d\n", n, pos0);
    k_attn_flash<96, 2, 32><<<(n + TQ - 1) / TQ * H, 256,
                               (size_t)(TQ + 2 * 32) * 97 * sizeof(float), s>>>(
        g->q, kc, vc, g->attn, pos0, kv_base, n, H, HK, scale, win);
  } else if (D == 256) {
    /* t11：Qwen3.8 的 head_dim=256 以前没有特化 → 一直回退到两遍核，
     * 而那个核是「一个 block 管一个 (t,h)」，每个查询行都把整段 K/V
     * 重读一遍（n=1024、ctx=4096 时单层 200GB 流量，实测 158ms/层）。
     * lane 跑 0..31 对应 TK 个 key 位置，所以 TK 越接近 32 利用率越高：
     *   TK=8/16 + V 进 shared → 41KB，2 blocks/SM，但 score 段只用 8/16 个 lane
     *   TK=32  + V 进 shared → 74KB，1 block/SM → 实测反而慢 10~20%
     *   TK=32  + V 直读全局 → 40KB，2 blocks/SM，lane 全满（当前默认）*/
    const int TQ = 8 * 1;
    int tk = 32;
    { const char *e2 = getenv("SGL_ATTN_TK"); if (e2) { int v2 = atoi(e2); if (v2 == 16 || v2 == 32) tk = v2; } }
    if (tk == 32) {
      if (getenv("SGL_ATTN_DEBUG"))
        fprintf(stderr, "[attn] flash D=256 QPW=1 TK=32 VG=1 n=%d pos0=%d\n", n, pos0);
      k_attn_flash<256, 1, 32, true><<<(n + TQ - 1) / TQ * H, 256,
                                        (size_t)(TQ + 32) * 257 * sizeof(float), s>>>(
          g->q, kc, vc, g->attn, pos0, kv_base, n, H, HK, scale, win);
    } else {
      if (getenv("SGL_ATTN_DEBUG"))
        fprintf(stderr, "[attn] flash D=256 QPW=1 TK=16 VG=0 n=%d pos0=%d\n", n, pos0);
      k_attn_flash<256, 1, 16><<<(n + TQ - 1) / TQ * H, 256,
                                  (size_t)(TQ + 2 * 16) * 257 * sizeof(float), s>>>(
          g->q, kc, vc, g->attn, pos0, kv_base, n, H, HK, scale, win);
    }
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
/* M2：分页 + 4-bit 反量化的 batch attention（decode 用）*/
/* M2：decode 的 split-KV attention。
 * 问题：B 小时 (B*H/KVQ_FLASH_WARPS) 只有 6 个 block，170 个 SM 只用了 6 个
 * （实测 11K 上下文 decode 191 ms/token，有效算力 3.5%）。
 * 解法：每 (b,h) 的历史切成 nsplit 段，各段并行算部分 (m,l,acc)，再合并。
 * 这就是 flash-decoding 的做法。*/
typedef struct {
  int nsplit;
  float *part; /* [B*H*nsplit][2 + Dmax] : m, l, acc[D] */
  int cap;     /* 已分配的 B*H*nsplit 条目数 */
  int dm;      /* D */
} sgl_kvq_split;

/* 算一段，输出部分归约结果。warp 内 lane 负责连续的 ND 维。*/
__global__ void k_attn_q_split_partial(const float *q, const unsigned char *pool, const int *pt,
                                       const int *base, const int *lens, int B, int H, int HK,
                                       int D, float scale, int win, int P, int nvals,
                                       long slot_layer, long tok_stride, int nsplit,
                                       float *part) {
  const int ND = D / 32;
  const int nblk = nvals / SGL_KVQ_BLK;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int wid = blockIdx.x * KVQ_FLASH_WARPS + warp;
  int total = B * H * nsplit;
  if (wid >= total || ND > 16 || ND <= 0) return;
  int seg = wid % nsplit;
  int bh = wid / nsplit;
  int b = bh / H, h = bh % H;
  int upto = lens[b] + 1;
  int hkv = h / (H / HK);
  int from = (win > 0 && upto > win) ? upto - win : 0;
  int span = upto - from;
  if (span <= 0) {
    if (lane == 0) {
      float *o = part + (long)wid * (2 + D);
      o[0] = -1e30f;
      o[1] = 0.0f;
    }
    return;
  }
  /* 把 [from, upto) 均分给 nsplit 段 */
  int step = (span + nsplit - 1) / nsplit;
  int s0 = from + seg * step;
  int s1 = s0 + step;
  if (s1 > upto) s1 = upto;
  float qr[16];
  const float *qh = q + ((long)b * H + h) * D;
#pragma unroll
  for (int i = 0; i < ND; i++) qr[i] = qh[lane * ND + i];
  int vbase = hkv * D + lane * ND;
  int kbyte = vbase >> 1;
  int vbyte = (nvals >> 1) + (vbase >> 1);
  int sblk = vbase >> 7;
  float m = -1e30f, lsum = 0.0f;
  float acc[16];
#pragma unroll
  for (int i = 0; i < ND; i++) acc[i] = 0.0f;
  int bpos = base[b];
  int p = s0;
  while (p < s1) {
    int gpos = bpos + p;
    int page = gpos / P, off = gpos - page * P;
    int pend = (page + 1) * P - bpos;
    if (pend > s1) pend = s1;
    int slot = pt[page];
    if (slot < 0) {
      p = pend;
      continue;
    }
    const unsigned char *pg = pool + (long)slot * slot_layer;
    for (; p < pend; p++, off++) {
      const unsigned char *kb = pg + (long)off * tok_stride;
      float szk = *(const float *)(kb + nvals + sblk * 4);
      float dot = 0.0f;
      const unsigned char *bp = kb + kbyte;
#pragma unroll
      for (int j = 0; j < ND; j += 2) {
        unsigned char by = bp[j >> 1];
        int n0 = (by & 0xF), n1 = (by >> 4);
        n0 = (n0 << 28) >> 28;
        n1 = (n1 << 28) >> 28;
        dot += qr[j] * (float)n0 + qr[j + 1] * (float)n1;
      }
      dot *= szk;
#pragma unroll
      for (int o2 = 16; o2 > 0; o2 >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o2);
      float s = dot * scale;
      float mn = fmaxf(m, s);
      float alpha = __expf(m - mn);
      float e = __expf(s - mn);
      lsum = lsum * alpha + e;
      float szv = *(const float *)(kb + nvals + nblk * 4 + sblk * 4);
      const unsigned char *vp = kb + vbyte;
#pragma unroll
      for (int j = 0; j < ND; j += 2) {
        unsigned char by = vp[j >> 1];
        int n0 = (by & 0xF), n1 = (by >> 4);
        n0 = (n0 << 28) >> 28;
        n1 = (n1 << 28) >> 28;
        acc[j] = acc[j] * alpha + e * ((float)n0 * szv);
        acc[j + 1] = acc[j + 1] * alpha + e * ((float)n1 * szv);
      }
      m = mn;
    }
  }
  float *o = part + (long)wid * (2 + D);
  if (lane == 0) {
    o[0] = m;
    o[1] = lsum;
  }
#pragma unroll
  for (int i = 0; i < ND; i++) o[2 + lane * ND + i] = acc[i];
}

/* 合并 nsplit 份：warp per (b,h)，lane 沿 D 分 */
__global__ void k_attn_q_split_merge(float *out, const float *part, int B, int H, int D,
                                     int nsplit) {
  const int ND = D / 32;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int bh = blockIdx.x * KVQ_FLASH_WARPS + warp;
  if (bh >= B * H || ND > 16 || ND <= 0) return;
  float m = -1e30f;
  for (int s = 0; s < nsplit; s++) {
    const float *o = part + (long)(bh * nsplit + s) * (2 + D);
    m = fmaxf(m, o[0]);
  }
  float lsum = 0.0f;
  float acc[16];
#pragma unroll
  for (int i = 0; i < ND; i++) acc[i] = 0.0f;
  for (int s = 0; s < nsplit; s++) {
    const float *o = part + (long)(bh * nsplit + s) * (2 + D);
    float w = __expf(o[0] - m);
    lsum += o[1] * w;
#pragma unroll
    for (int i = 0; i < ND; i++) acc[i] += o[2 + lane * ND + i] * w;
  }
  float inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;
  float *p = out + (long)bh * D;
#pragma unroll
  for (int i = 0; i < ND; i++) p[lane * ND + i] = acc[i] * inv;
}

/* M2：batch 版 flash（decode 用；每行有自己的 base/lens）*/
__global__ void k_attn_q_batch_flash(const float *q, const unsigned char *pool, const int *pt,
                                     float *out, const int *base, const int *lens, int B, int H,
                                     int HK, int D, float scale, int win, int P, int nvals,
                                     long slot_layer, long tok_stride) {
  const int ND = D / 32;
  const int nblk = nvals / SGL_KVQ_BLK;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int th = blockIdx.x * KVQ_FLASH_WARPS + warp;
  if (th >= B * H) return;
  if (ND > 16 || ND <= 0) return;
  int b = th / H, h = th % H;
  int upto = lens[b] + 1;
  int hkv = h / (H / HK);
  int from = (win > 0 && upto > win) ? upto - win : 0;
  const float *qh = q + ((long)b * H + h) * D;
  float qr[16];
#pragma unroll
  for (int i = 0; i < ND; i++) qr[i] = qh[lane * ND + i];
  int vbase = hkv * D + lane * ND;
  int kbyte = vbase >> 1;
  int vbyte = (nvals >> 1) + (vbase >> 1);
  int sblk = vbase >> 7;

  float m = -1e30f, lsum = 0.0f;
  float acc[16];
#pragma unroll
  for (int i = 0; i < ND; i++) acc[i] = 0.0f;

  int bpos = base[b];
  int p = from;
  while (p < upto) {
    int gpos = bpos + p;
    int page = gpos / P, off = gpos - page * P;
    int pend = (page + 1) * P - bpos;
    if (pend > upto) pend = upto;
    int slot = pt[page];
    if (slot < 0) {
      p = pend;
      continue;
    }
    const unsigned char *pg = pool + (long)slot * slot_layer;
    for (; p < pend; p++, off++) {
      const unsigned char *kb = pg + (long)off * tok_stride;
      float szk = *(const float *)(kb + nvals + sblk * 4);
      float dot = 0.0f;
      const unsigned char *bp = kb + kbyte;
#pragma unroll
      for (int j = 0; j < ND; j += 2) {
        unsigned char by = bp[j >> 1];
        int n0 = (by & 0xF), n1 = (by >> 4);
        n0 = (n0 << 28) >> 28;
        n1 = (n1 << 28) >> 28;
        dot += qr[j] * (float)n0 + qr[j + 1] * (float)n1;
      }
      dot *= szk;
#pragma unroll
      for (int off2 = 16; off2 > 0; off2 >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off2);
      float s = dot * scale;
      float mn = fmaxf(m, s);
      float alpha = __expf(m - mn);
      float e = __expf(s - mn);
      lsum = lsum * alpha + e;
      float szv = *(const float *)(kb + nvals + nblk * 4 + sblk * 4);
      const unsigned char *vp = kb + vbyte;
#pragma unroll
      for (int j = 0; j < ND; j += 2) {
        unsigned char by = vp[j >> 1];
        int n0 = (by & 0xF), n1 = (by >> 4);
        n0 = (n0 << 28) >> 28;
        n1 = (n1 << 28) >> 28;
        acc[j] = acc[j] * alpha + e * ((float)n0 * szv);
        acc[j + 1] = acc[j + 1] * alpha + e * ((float)n1 * szv);
      }
      m = mn;
    }
  }
  float inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;
  float *o = out + ((long)b * H + h) * D;
#pragma unroll
  for (int i = 0; i < ND; i++) o[lane * ND + i] = acc[i] * inv;
}

__global__ void k_attn_q_batch(const float *q, const unsigned char *pool, const int *pt,
                               float *out, const int *base, const int *lens, int B, int H,
                               int HK, int D, float scale, int win, int P, int nvals,
                               long slot_layer, long tok_stride) {
  __shared__ float sc[SGL_ATTN_CHUNK];
  __shared__ float red[256];
  int th = blockIdx.x;
  int b = th / H, h = th % H;
  int upto = lens[b] + 1;
  const float *qh = q + ((long)b * H + h) * D;
  int hkv = h / (H / HK);
  int from = (win > 0 && upto > win) ? upto - win : 0;
  int koff = (hkv * D) / 2;
  int voff = nvals / 2 + (hkv * D) / 2;
  int nblk = nvals / SGL_KVQ_BLK;
  float run_max = -1e30f, run_sum = 0.0f;

  for (int c0 = from; c0 < upto; c0 += SGL_ATTN_CHUNK) {
    int ce = c0 + SGL_ATTN_CHUNK; if (ce > upto) ce = upto;
    float cmax = -1e30f;
    for (int p = c0 + threadIdx.x; p < ce; p += blockDim.x) {
      int gpos = base[b] + p;
      int page = gpos / P, off = gpos - page * P;
      int slot = pt[page];
      float s = 0;
      if (slot >= 0) {
        const unsigned char *kb = pool + (long)slot * slot_layer + (long)off * tok_stride;
        for (int d = 0; d < D; d += 2) {
          const unsigned char *bb = kb + koff + (d >> 1);
          unsigned char by = *bb;
          int n0 = (by & 0xF), n1 = (by >> 4);
          n0 = (n0 << 28) >> 28; n1 = (n1 << 28) >> 28;
          float sc0 = *(const float *)(kb + nvals + ((hkv * D + d) >> 7) * 4);
          s += qh[d] * ((float)n0 * sc0) + qh[d + 1] * ((float)n1 * sc0);
        }
        s *= scale;
      } else
        s = -1e30f;
      sc[p - c0] = s;
      if (s > cmax) cmax = s;
    }
    red[threadIdx.x] = cmax;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
      if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
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
    __syncthreads();
    red[threadIdx.x] = cs;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
      if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
      __syncthreads();
    }
    run_sum = run_sum * alpha + red[0];
    run_max = cmax;
    __syncthreads();
  }
  float invsum = 1.0f / run_sum;
  float *o = out + ((long)b * H + h) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] = 0;
  __syncthreads();
  for (int c0 = from; c0 < upto; c0 += SGL_ATTN_CHUNK) {
    int ce = c0 + SGL_ATTN_CHUNK; if (ce > upto) ce = upto;
    for (int p = c0 + threadIdx.x; p < ce; p += blockDim.x) {
      int gpos = base[b] + p;
      int page = gpos / P, off = gpos - page * P;
      int slot = pt[page];
      float s = 0;
      if (slot >= 0) {
        const unsigned char *kb = pool + (long)slot * slot_layer + (long)off * tok_stride;
        for (int d = 0; d < D; d += 2) {
          const unsigned char *bb = kb + koff + (d >> 1);
          unsigned char by = *bb;
          int n0 = (by & 0xF), n1 = (by >> 4);
          n0 = (n0 << 28) >> 28; n1 = (n1 << 28) >> 28;
          float sc0 = *(const float *)(kb + nvals + ((hkv * D + d) >> 7) * 4);
          s += qh[d] * ((float)n0 * sc0) + qh[d + 1] * ((float)n1 * sc0);
        }
        s *= scale;
      } else
        s = -1e30f;
      sc[p - c0] = __expf(s - run_max);
    }
    __syncthreads();
    for (int p = c0; p < ce; p++) {
      int gpos = base[b] + p;
      int page = gpos / P, off = gpos - page * P;
      int slot = pt[page];
      if (slot < 0) continue;
      const unsigned char *vb = pool + (long)slot * slot_layer + (long)off * tok_stride;
      float w = sc[p - c0];
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const unsigned char *bb = vb + voff + (d >> 1);
        unsigned char by = *bb;
        int q4 = (d & 1) ? (by >> 4) : (by & 0xF);
        q4 = (q4 << 28) >> 28;
        float scv = *(const float *)(vb + nvals + nblk * 4 + ((hkv * D + d) >> 7) * 4);
        o[d] += w * ((float)q4 * scv);
      }
    }
    __syncthreads();
  }
  for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] *= invsum;
}

static void attn_dispatch(sgl_cuda_model *g, int l, int n, int pos0, int kv_base,
                          float sc, int two_pass, int force_long, int E) {
  int D = g->D, H = g->H, HK = g->HK;
  const float *kc = g->kc + (long)g->kvo[l] * g->ctx * g->KD;
  const float *vc = g->vc + (long)g->kvo[l] * g->ctx * g->KD;
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

/* M3：device 侧 argmax。MTP 每步要为每个草稿算一次 lm_head（V=248320），
 * 把 1 MB 的 logits D2H 每草稿一次不划算；在 device 上取完只回 4 字节。*/
__global__ void k_argmax(const float *v, int n, int *out) {
  __shared__ float bv[256];
  __shared__ int bi[256];
  int tid = threadIdx.x;
  float best = -INFINITY;
  int bidx = 0;
  for (int i = tid; i < n; i += blockDim.x) {
    float x = v[i];
    if (x > best) { best = x; bidx = i; }
  }
  bv[tid] = best;
  bi[tid] = bidx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && bv[tid + s] > bv[tid]) { bv[tid] = bv[tid + s]; bi[tid] = bi[tid + s]; }
    __syncthreads();
  }
  if (tid == 0) *out = bi[0];
}

/* k_argmax 再加 argmax 的 softmax 概率 1/Σexp(v-max)；选出的下标与 k_argmax 相同 */
__global__ void k_argmax_p(const float *v, int n, int *out, float *pout) {
  __shared__ float bv[256];
  __shared__ int bi[256];
  int tid = threadIdx.x;
  float best = -INFINITY;
  int bidx = 0;
  for (int i = tid; i < n; i += blockDim.x) {
    float x = v[i];
    if (x > best) { best = x; bidx = i; }
  }
  bv[tid] = best;
  bi[tid] = bidx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && bv[tid + s] > bv[tid]) { bv[tid] = bv[tid + s]; bi[tid] = bi[tid + s]; }
    __syncthreads();
  }
  float mx = bv[0];
  __syncthreads();
  float sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) sum += __expf(v[i] - mx);
  bv[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) bv[tid] += bv[tid + s];
    __syncthreads();
  }
  if (tid == 0) {
    *out = bi[0];
    *pout = 1.0f / bv[0];
  }
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
/* t11 性能：M==1 的 GEMV——激活已预量化成 int8（每行只做一次 absmax/quant），
 * 不再每个输出通道重读 20KB 的 f32 激活行。27B 的 lin qkvz (N=16384,K=5120)
 * 原先要重复读 327MB 激活，而权重才 84MB——激活性读比权重多 4 倍。*/
__global__ void k_gemm_w8a8_q(const int8_t *a8, const int8_t *W, const float *ws,
                              const float *as, const float *bias, float *C, int N, int K) {
  int n = blockIdx.x;
  if (n >= N) return;
  const int8_t *w = W + (long)n * K;
  int acc = 0;
  for (int i = threadIdx.x * 16; i + 15 < K; i += blockDim.x * 16) {
    int acc0 = __dp4a(*(const int *)(a8 + i), *(const int *)(w + i), 0);
    acc0 = __dp4a(*(const int *)(a8 + i + 4), *(const int *)(w + i + 4), acc0);
    acc0 = __dp4a(*(const int *)(a8 + i + 8), *(const int *)(w + i + 8), acc0);
    acc0 = __dp4a(*(const int *)(a8 + i + 12), *(const int *)(w + i + 12), acc0);
    acc += acc0;
  }
  for (int i = K - (K % 16) + threadIdx.x; i < K; i += blockDim.x)
    acc += (int)a8[i] * (int)w[i];
  __shared__ int iaccq[256];
  iaccq[threadIdx.x] = acc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) iaccq[threadIdx.x] += iaccq[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    C[n] = (float)iaccq[0] * as[0] * ws[n] + (bias ? bias[n] : 0.f);
}

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

/* t11 诊断计数器（见 gemm_nt_w8_imma）*/
long g_imma_ok = 0, g_imma_fail = 0, g_imma_skip = 0, g_fallback_mn = 0;
int g_imma_status = 0;

/* returns 0 when the IMMA path ran, -1 when the caller must use dp4a */
static int gemm_nt_w8_imma(sgl_cuda_model *g, const float *A, const sgl_w8w *W8,
                           const float *bias, float *C, int M, int N, int K) {
  /* t11 诊断：数一数 IMMA 到底成功几次，失败时 cublas 返回什么。
   * 之前 prefill 一块 1024 token 要 7.6 s，而 int8 张量核理论上 ~0.4 s/
   * 整套 GEMM——如果这里全是 fail，就坐实了“回退到逐 (m,n) dp4a 核，
   * 权重被重读 M 次”这个猜想（权重 27GB × 1024 = 27TB）。*/
  extern long g_imma_ok, g_imma_fail, g_imma_skip, g_fallback_mn;
  extern int g_imma_status;
  /* G2 A/B knob: SGL_W8_DP4A=1 keeps the old per-(m,n) dp4a path for every M
   * (read per call so cuda_test can compare both paths in one process). */
  { const char *ab = getenv("SGL_W8_DP4A"); if (ab && atoi(ab)) { g_imma_skip++; return -1; } }
  if (M < 2 || (K % 4) || (N % 4) || !g->w8_a8 || !g->w8_c32) { g_imma_skip++; return -1; }
  const int32_t ai = 1, bi = 0;
  k_quant_rows<<<M, 256, 0, g->stream>>>(A, g->w8_a8, g->w8_as, M, K);
  cublasStatus_t st = cublasGemmEx(g->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &ai,
                                   W8->w, CUDA_R_8I, K, g->w8_a8, CUDA_R_8I, K, &bi,
                                   g->w8_c32, CUDA_R_32I, N, CUBLAS_COMPUTE_32I,
                                   CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) { g_imma_status = (int)st; g_imma_fail++; return -1; }
  k_w8_dequant<<<(int) (((long) M * N + 255) / 256), 256, 0, g->stream>>>(
      g->w8_c32, g->w8_as, W8->scale, bias, C, M, N);
  if (cudaGetLastError() != cudaSuccess) { g_imma_fail++; return -1; }
  g_imma_ok++;
  return 0;
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
/* ---- M0-显存：embed_tokens 放在 mapped pinned host 内存 ----
 * 动机：embed 每 token 只查 1 行（E=5120，fp16 = 10 KiB），却常驻 VRAM 占
 * 2.54 GB（248320×5120×2B）。实测 GPU 直读 mapped host 51.2 GB/s，一次 10 KiB
 * 只 0.2 µs，完全可忽略。腾出的 2.54 GB 正好让 32 个 GDN 递归状态槽
 * （1.81 GB）+ conv 历史（188 MB）入住 —— 之前 32 槽因差 335 MB 直接崩。
 * 失败（无 UVA / 不可映射）时返回 NULL，调用方回退到显存。*/
static __half *host_map_h(const float *src, size_t n) {
  __half *h = NULL;
  if (cudaHostAlloc((void **)&h, n * sizeof(__half), cudaHostAllocMapped) != cudaSuccess)
    return NULL;
  if (src)
    for (size_t i = 0; i < n; i++) h[i] = __float2half(src[i]);
  else
    memset(h, 0, n * sizeof(__half));
  return h;
}

/* device-visible alias of a mapped host buffer (UVA: same address on our setup) */
static __half *map_dev_ptr(__half *h) {
  void *d = NULL;
  if (!h) return NULL;
  if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess || !d) return h;
  return (__half *)d;
}

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
  if (g->kind == 0 && g->QD > q_cols) q_cols = g->QD; /* t11：显式 head_dim 时 q 半边 = QD */
  int a_cols = g->kind == 0 && g->QD > g->E ? g->QD : g->E; /* t11：attn 输出行宽 = H*D（可能 >E）*/
  float *q = dev_copy(NULL, (size_t)n * q_cols);
  float *a = dev_copy(NULL, (size_t)n * a_cols);
  float *k = dev_copy(NULL, (size_t)n * g->KD);
  float *v = dev_copy(NULL, (size_t)n * g->KD);
  float *ga = dev_copy(NULL, (size_t)n * g->I);
  float *u = dev_copy(NULL, (size_t)n * g->I);
  float *q2 = NULL, *qg = NULL; /* t11：q_gate 时 PROJ 的 2QD 输出与 gate 半边 */
  if (g->q_gate && g->kind == 0) {
    q2 = dev_copy(NULL, (size_t)n * 2 * g->QD);
    qg = dev_copy(NULL, (size_t)n * g->QD);
    if (!q2 || !qg) {
      cudaFree(x); cudaFree(h); cudaFree(q); cudaFree(a);
      cudaFree(k); cudaFree(v); cudaFree(ga); cudaFree(u);
      cudaFree(q2); cudaFree(qg);
      fprintf(stderr, "sgl_cuda_reserve(%d): out of VRAM (qgate)\n", n);
      return -1;
    }
  }
  if (!x || !h || !q || !a || !k || !v || !ga || !u) {
    cudaFree(x); cudaFree(h); cudaFree(q); cudaFree(a);
    cudaFree(k); cudaFree(v); cudaFree(ga); cudaFree(u);
    cudaFree(q2); cudaFree(qg);
    fprintf(stderr, "sgl_cuda_reserve(%d): out of VRAM\n", n);
    return -1;
  }
  cudaFree(g->x); cudaFree(g->h); cudaFree(g->q); cudaFree(g->attn);
  cudaFree(g->k); cudaFree(g->v); cudaFree(g->gate); cudaFree(g->up);
  cudaFree(g->q2); cudaFree(g->qgate); /* t11（NULL 安全）*/
  g->x = x; g->h = h; g->q = q; g->attn = a;
  g->k = k; g->v = v; g->gate = ga; g->up = u;
  g->q2 = q2; g->qgate = qg;
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
  /* t11 prefill 加速：lin 层整块批处理的两个暂存区
   *   l_qkvz_all [n][rowN]  一次 M=n 投影的 qkvz（下游只读）
   *   l_o_all    [n][vald]  逐 token 递归的输出，最后再一次 M=n 的 out_proj
   * 1024 行时约 1024×(16384+6144)×4B ≈ 92MB。*/
  if (g->lin_n > 0 && g->lin_nh > 0) {
    int vald = g->lin_nh * g->lin_vd;
    int rowN = 2 * g->lin_nk * g->lin_kd + 2 * vald;
    float *qa = NULL, *oa = NULL, *bba = NULL;
    if (cudaMalloc((void **)&qa, (size_t)n * rowN * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&oa, (size_t)n * vald * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&bba, (size_t)n * 2 * g->lin_nh * sizeof(float)) != cudaSuccess) {
      cudaFree(qa); cudaFree(oa); cudaFree(bba);
      fprintf(stderr, "sgl_cuda_reserve(%d): out of VRAM (lin batch scratch)\n", n);
      return -1;
    }
    cudaFree(g->l_qkvz_all); cudaFree(g->l_o_all); cudaFree(g->l_ba_all);
    g->l_qkvz_all = qa; g->l_o_all = oa; g->l_ba_all = bba;
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

/* t11：host 已量化的 int8 权重（llama.c 流式 W8）直接上传，免 f32 中转。
 *（量化语义与 dev_copy_w8 完全一致：per-row absmax）*/
static int dev_copy_w8_i8(const int8_t *hq, const float *hs, int N, int K, sgl_w8w *out) {
  if (!hq || !hs) return -1;
  if (cudaMalloc((void **)&out->w, (size_t)N * K) != cudaSuccess) return -1;
  if (cudaMalloc((void **)&out->scale, sizeof(float) * (size_t)N) != cudaSuccess) {
    cudaFree(out->w);
    return -1;
  }
  cudaMemcpy(out->w, hq, (size_t)N * K, cudaMemcpyHostToDevice);
  cudaMemcpy(out->scale, hs, sizeof(float) * (size_t)N, cudaMemcpyHostToDevice);
  return 0;
}

static void w8_free_layers(sgl_w8w *arr, int L) {
  if (!arr) return;
  for (int l = 0; l < L; l++) {
    if (arr[l].w) cudaFree(arr[l].w);
    if (arr[l].scale) cudaFree(arr[l].scale);
  }
  free(arr);
}

/* ================= M1: NVFP4 权重 =================
 * e2m1 的 16 个电平（bit3 = 符号，低 3 位 = 幅度序号）。
 * 幅度：0, .5, 1, 1.5, 2, 3, 4, 6 —— 这是 NVFP4 规定的码书。*/
static const float k_e2m1_pos[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
__constant__ float c_e2m1[16] = {0.f,   0.5f, 1.f,   1.5f, 2.f,   3.f,   4.f,   6.f,
                                 -0.f,  -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

/* 量化时每 16 元素一块；K 不是 16 的倍数时把尾巴并进最后一块。*/
#define W4_BLK 16
static inline int w4_nblk(int K) { return (K + W4_BLK - 1) / W4_BLK; }

/* 把一个 16 元块量化成 8 字节（16 个 int4 补码 nibble）+ 一个 fp16 块缩放。
 *
 * 为什么不用 e2m1：微基准实测（docs/DESIGN_1M_32C.md §14）在 sm_120 上，
 *「FP4 码书 → float 解码再 FMA」无论用 smem 查表还是算术构造 float 位型，
 * 都只能跑到 **272–286 GB/s**（峰值的 16%）；而 int4 补码 + prmt 展开 +
 * int8 激活 → __dp4a 能跑到 **1677–1930 GB/s（lm_head 上就是峰值）**。
 * 差 6.2× 的原因是 dp4a 一次算 4 个乘加且没有查表延迟，而 float 路径
 * 每值一次 30-cycle 的 LDS/长依赖链。
 * 代价：int4 只有 16 个电平（−7..7），靠每 16 元素一个缩放保精度。*/
static inline void w4_quant_block(const float *x, int n, uint8_t *out8, __half *sc) {
  float amax = 0.f;
  for (int i = 0; i < n; i++) { float v = fabsf(x[i]); if (v > amax) amax = v; }
  float s = (amax > 1e-12f) ? amax / 7.f : 0.f;
  *sc = __float2half(s);
  float inv = s > 0.f ? 1.f / s : 0.f;
  for (int i = 0; i < n; i++) {
    float v = x[i] * inv;
    int q = (int)lrintf(v);
    if (q > 7) q = 7;
    if (q < -7) q = -7;
    unsigned nib = (unsigned)(q & 0x0f);   /* 4-bit 补码 */
    if (i & 1) out8[i >> 1] |= (uint8_t)(nib << 4);
    else       out8[i >> 1] = (uint8_t)(nib & 0x0f);
  }
}

/* 从 host f32 权重量化成 FP4（N 行 × K 列）。*/
static int dev_copy_w4(const float *src, int N, int K, sgl_w4w *out) {
  if (!src) return -1;
  int nb = w4_nblk(K);
  size_t wbytes = ((size_t)N * (size_t)K + 1) / 2;
  uint8_t *q = (uint8_t *)calloc(wbytes ? wbytes : 1, 1);
  __half *s = (__half *)malloc(sizeof(__half) * (size_t)N * nb);
  if (!q || !s) { free(q); free(s); return -1; }
  for (int n = 0; n < N; n++) {
    const float *r = src + (size_t)n * K;
    for (int b = 0; b < nb; b++) {
      int k0 = b * W4_BLK;
      int cnt = (K - k0 < W4_BLK) ? (K - k0) : W4_BLK;
      w4_quant_block(r + k0, cnt, q + (size_t)n * K / 2 + (size_t)k0 / 2, &s[(size_t)n * nb + b]);
    }
  }
  cudaError_t e1 = cudaMalloc((void **)&out->w, wbytes);
  cudaError_t e2 = cudaMalloc((void **)&out->bs, sizeof(__half) * (size_t)N * nb);
  if (e1 != cudaSuccess || e2 != cudaSuccess) {
    if (e1 == cudaSuccess) cudaFree(out->w);
    if (e2 == cudaSuccess) cudaFree(out->bs);
    free(q); free(s);
    return -1;
  }
  cudaMemcpy(out->w, q, wbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(out->bs, s, sizeof(__half) * (size_t)N * nb, cudaMemcpyHostToDevice);
  free(q); free(s);
  return 0;
}

/* 从已量化的 host int8 权重（llama.c 流式 W8 的产物）转成 FP4、
 * 免回 f32 中转。反量化用行缩放 hs[n]，重量化用 16 元块缩放。
 * 精度损失：int8 已有一次误差，此处再加一次 FP4 误差；
 * 实测对 greedy 输出的影响见 docs/DESIGN_1M_32C.md §13 的对照方法。*/
static int dev_copy_w4_from_i8(const int8_t *hq, const float *hs, int N, int K, sgl_w4w *out) {
  if (!hq || !hs) return -1;
  int nb = w4_nblk(K);
  size_t wbytes = ((size_t)N * (size_t)K + 1) / 2;
  uint8_t *q = (uint8_t *)calloc(wbytes ? wbytes : 1, 1);
  __half *s = (__half *)malloc(sizeof(__half) * (size_t)N * nb);
  float *buf = (float *)malloc(sizeof(float) * W4_BLK);
  if (!q || !s || !buf) { free(q); free(s); free(buf); return -1; }
  for (int n = 0; n < N; n++) {
    const int8_t *r = hq + (size_t)n * K;
    float sc = hs[n];
    for (int b = 0; b < nb; b++) {
      int k0 = b * W4_BLK;
      int cnt = (K - k0 < W4_BLK) ? (K - k0) : W4_BLK;
      for (int i = 0; i < cnt; i++) buf[i] = (float)r[k0 + i] * sc;
      w4_quant_block(buf, cnt, q + (size_t)n * K / 2 + (size_t)k0 / 2, &s[(size_t)n * nb + b]);
    }
  }
  cudaError_t e1 = cudaMalloc((void **)&out->w, wbytes);
  cudaError_t e2 = cudaMalloc((void **)&out->bs, sizeof(__half) * (size_t)N * nb);
  if (e1 != cudaSuccess || e2 != cudaSuccess) {
    if (e1 == cudaSuccess) cudaFree(out->w);
    if (e2 == cudaSuccess) cudaFree(out->bs);
    free(q); free(s); free(buf);
    return -1;
  }
  cudaMemcpy(out->w, q, wbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(out->bs, s, sizeof(__half) * (size_t)N * nb, cudaMemcpyHostToDevice);
  free(q); free(s); free(buf);
  return 0;
}

static int w4_alloc_layers(sgl_w4w **arr, int L) {
  *arr = (sgl_w4w *)calloc((size_t)L, sizeof(sgl_w4w));
  return *arr ? 0 : -1;
}
static void w4_free_layers(sgl_w4w *arr, int L) {
  if (!arr) return;
  for (int l = 0; l < L; l++) {
    if (arr[l].w) cudaFree(arr[l].w);
    if (arr[l].bs) cudaFree(arr[l].bs);
  }
  free(arr);
}

/* ---- FP4(int4) GEMV / 小 M：dp4a 路径 ----
 * 一个 uint32 装 8 个 int4。用 __byte_perm 把 nibble 展开成字节（2 条指令出
 * 4 字节），再用 __vsubss4 做符号扩展，最后 4 次 __dp4a 算完 16 个乘加。
 * 每值约 1.25 条指令（vs float 解码的 4 条且有查表依赖）。
 *
 * 为什么不能直接拿 int4 权重乘 float 激活：
 * 实测（.mb2.cu，mlp gate N=17408 K=5120）：smem-LUT 266 GB/s、
 * 算术解码 280 GB/s、int4+dp4a **1930 GB/s**。差 6–7 倍。
 * 所以激活必须先量化成 int8（k_quant_rows），与仓库 int8 路径同一套语义。*/
__device__ __forceinline__ void w4_x8(unsigned v, int *a, int *b) {
  unsigned lom = v & 0x0F0F0F0Fu;
  unsigned him = (v >> 4) & 0x0F0F0F0Fu;
  unsigned lo = __byte_perm(lom, him, 0x5140);  /* → [n0][n1][n2][n3] */
  unsigned hi = __byte_perm(lom, him, 0x7362);  /* → [n4][n5][n6][n7] */
  *a = (int)__vsubss4(lo ^ 0x08080808u, 0x08080808u);
  *b = (int)__vsubss4(hi ^ 0x08080808u, 0x08080808u);
}

__global__ void k_gemm_w4_q(const int8_t *A8, const uint8_t *W, const __half *bs,
                            const float *as, const float *bias, float *C, int N, int K) {
  extern __shared__ float red[];
  int n = blockIdx.x;
  if (n >= N) return;
  int nb = (K + 15) / 16;
  int nch = (K + 31) / 32;
  const uint4 *wr = (const uint4 *)(W + (size_t)n * K / 2);
  const __half *br = bs + (size_t)n * nb;
  float acc = 0.f;
  for (int c = threadIdx.x; c < nch; c += blockDim.x) {
    uint4 v = wr[c];
    unsigned q[4] = {v.x, v.y, v.z, v.w};
    int b8[8];
#pragma unroll
    for (int u = 0; u < 4; u++) { int x, y; w4_x8(q[u], &x, &y); b8[u * 2] = x; b8[u * 2 + 1] = y; }
    int base = c * 32;
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
      int bb = c * 2 + hf;
      if (bb >= nb) break;
      int sacc = 0;
#pragma unroll
      for (int t = 0; t < 4; t++)
        sacc = __dp4a(*(const int *)(A8 + base + hf * 16 + t * 4), b8[hf * 4 + t], sacc);
      acc += (float)sacc * __half2float(br[bb]);
    }
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) C[n] = red[0] * as[0] + (bias ? bias[n] : 0.f);
}

/* M=2..8：一个 block 一个输出行，权重读一遍，累计 MT 个激活行。
 * C 是 M 行×N 列行主序，第 i 行的第 n 个元素在 C[i*N+n]。*/
template <int MT>
__global__ void k_gemm_w4_mrow(const int8_t *A8, const uint8_t *W, const __half *bs,
                               const float *as, const float *bias, float *C, int M, int N, int K) {
  extern __shared__ float wred[];
  int n = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  if (n >= N) return;
  int nb = (K + 15) / 16;
  int nch = (K + 31) / 32;
  const uint4 *wr = (const uint4 *)(W + (size_t)n * K / 2);
  const __half *br = bs + (size_t)n * nb;
  float acc[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = 0.f;
  for (int c = tid; c < nch; c += nt) {
    uint4 v = wr[c];
    unsigned q[4] = {v.x, v.y, v.z, v.w};
    int b8[8];
#pragma unroll
    for (int u = 0; u < 4; u++) { int x, y; w4_x8(q[u], &x, &y); b8[u * 2] = x; b8[u * 2 + 1] = y; }
    int base = c * 32;
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
      int bb = c * 2 + hf;
      if (bb >= nb) break;
      float fs = __half2float(br[bb]);
#pragma unroll
      for (int i = 0; i < MT; i++) {
        if (i >= M) continue;
        const int8_t *ar = A8 + (size_t)i * K + base + hf * 16;
        int sacc = 0;
#pragma unroll
        for (int t = 0; t < 4; t++) sacc = __dp4a(*(const int *)(ar + t * 4), b8[hf * 4 + t], sacc);
        acc[i] += (float)sacc * fs;
      }
    }
  }
  /* 归约：warp 内 shuffle，再每 warp lane0 汇总 */
  int lane = tid & 31, warp = tid >> 5, nwarp = nt >> 5;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    float v = acc[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) wred[i * 8 + warp] = v;
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < MT; i++) {
      if (i >= M) continue;
      float t = 0.f;
      for (int w = 0; w < nwarp && w < 8; w++) t += wred[i * 8 + w];
      C[(size_t)i * N + n] = t * as[i] + (bias ? bias[n] : 0.f);
    }
  }
}

/* lm_head 分派：FP4 优先，否则 fp16/fp32。两处调用点都走这里。*/
static void gemm_w4_launch(sgl_cuda_model *g, const float *A, const sgl_w4w *W,
                           const float *bias, float *C, int M, int N, int K);
static void lm_head_gemm(sgl_cuda_model *g, const float *A, float *C, int M) {
  if (g->use_w4 && g->lm_head_w4.w)
    gemm_w4_launch(g, A, &g->lm_head_w4, NULL, C, M, g->V, g->E);
  else
    gemm_nt_any(g, A, g->lm_head, g->lm_head_h, NULL, C, M, g->V, g->E);
}

static void gemm_w4_launch(sgl_cuda_model *g, const float *A, const sgl_w4w *W,
                           const float *bias, float *C, int M, int N, int K) {
  cudaStream_t s = g->stream;
  /* 先把激活量化成 int8（每行一个缩放），与 int8 权重路径同一套语义。
   * dp4a 要求两边都是 int8；这正是 v2 跑到 1677 GB/s 的前提。*/
  size_t need = (size_t)M * (size_t)K;
  if (need > g->w4_acap) {
    if (g->w4_a8) cudaFree(g->w4_a8);
    if (cudaMalloc((void **)&g->w4_a8, need) != cudaSuccess) { g->w4_acap = 0; return; }
    if (!g->w4_as && cudaMalloc((void **)&g->w4_as, sizeof(float) * 32) != cudaSuccess) {
      cudaFree(g->w4_a8); g->w4_a8 = NULL; g->w4_acap = 0; return;
    }
    g->w4_acap = need;
  }
  if (!g->w4_a8 || !g->w4_as) return;
  k_quant_rows<<<M, 256, 0, s>>>(A, g->w4_a8, g->w4_as, M, K);
  if (M == 1) {
    k_gemm_w4_q<<<N, 256, 256 * sizeof(float), s>>>(g->w4_a8, W->w, W->bs, g->w4_as, bias, C, N, K);
    return;
  }
  /* M>8 时按 8 行一组走多趟（权重多读 M/8 遍）。组内一次读完，
   * 代价可接受：0.625 B/参数下 32 并发也只有 4×16.25 GB ≈ 38 ms/步。*/
  for (int m0 = 0; m0 < M; m0 += 8) {
    int rem = M - m0;
    const float *Asub = A + (size_t)m0 * K;
    (void)Asub;
    /* 激活用组起点偏移：kernel 内部按 i*K 取相对行 */
    const int8_t *a8sub = g->w4_a8 + (size_t)m0 * K;
    const float *assub = g->w4_as + m0;
    float *Csub = C + (size_t)m0 * N;
    int Ms = rem < 8 ? rem : 8;
    switch (Ms) {
      case 8: k_gemm_w4_mrow<8><<<N, 256, 8 * 8 * sizeof(float), s>>>(a8sub, W->w, W->bs, assub, bias, Csub, Ms, N, K); break;
      case 7: k_gemm_w4_mrow<7><<<N, 256, 8 * 8 * sizeof(float), s>>>(a8sub, W->w, W->bs, assub, bias, Csub, Ms, N, K); break;
      case 6: k_gemm_w4_mrow<6><<<N, 256, 8 * 8 * sizeof(float), s>>>(a8sub, W->w, W->bs, assub, bias, Csub, Ms, N, K); break;
      case 5: k_gemm_w4_mrow<5><<<N, 256, 8 * 8 * sizeof(float), s>>>(a8sub, W->w, W->bs, assub, bias, Csub, Ms, N, K); break;
      case 4: k_gemm_w4_mrow<4><<<N, 256, 8 * 8 * sizeof(float), s>>>(a8sub, W->w, W->bs, assub, bias, Csub, Ms, N, K); break;
      case 3: k_gemm_w4_mrow<3><<<N, 256, 8 * 8 * sizeof(float), s>>>(a8sub, W->w, W->bs, assub, bias, Csub, Ms, N, K); break;
      case 2: k_gemm_w4_mrow<2><<<N, 256, 8 * 8 * sizeof(float), s>>>(a8sub, W->w, W->bs, assub, bias, Csub, Ms, N, K); break;
      default: k_gemm_w4_q<<<N, 256, 256 * sizeof(float), s>>>(a8sub, W->w, W->bs, assub, bias, Csub, N, K); break;
    }
  }
}

/* t8 projection dispatch: W8A8 for every M when int8 weights exist
 * (prefill included -- accuracy verified against fp32 greedy; fp32 originals
 * are NOT uploaded in W8 mode, that's where the VRAM saving comes from).
 * G2: M>1 goes to the int8 tensor-core (IMMA) path; M==1 and the shapes IMMA
 * cannot take stay on the hand dp4a GEMV, which is the faster of the two at
 * batch 1 (weight bandwidth bound, one row of activations). */
/* P2 诊断（SGL_PROF_NOGEMM=1）：整个 PROJ 不发射任何 kernel，用来量
 * 「一步 decode 里 GEMM 之外到底占多少」。配合 SGL_CUDA_PROF=1 读 total。
 * 输出是垃圾（C 保持旧值），纯计时用。*/
static int prof_nogemm(void) {
  static int v = -1;
  if (v < 0) { const char *e = getenv("SGL_PROF_NOGEMM"); v = e ? atoi(e) : 0; }
  return v;
}

#define PROJ(g, A, Wf, Wh, W8, bias, C, M, N, K)                                   \
  do {                                                                             \
    if (prof_nogemm()) break;                                                      \
    if ((g)->use_w4 && (W8) && (W8)->w4) {                                         \
      gemm_w4_launch((g), A, (const sgl_w4w *)(W8)->w4, bias, C, M, N, K);         \
      break;                                                                       \
    }                                                                              \
    if ((g)->use_w8 && (W8)) {                                                     \
      if ((M) > 1 && gemm_nt_w8_imma(g, A, W8, bias, C, M, N, K) == 0) break;      \
      if ((M) == 1 && (g)->w8_a8 && (g)->w8_as) {                                  \
        k_quant_rows<<<1, 256, 0, (g)->stream>>>(A, (g)->w8_a8, (g)->w8_as, 1, K);  \
        k_gemm_w8a8_q<<<(N), 256, 0, (g)->stream>>>(                               \
            (g)->w8_a8, (W8)->w, (W8)->scale, (g)->w8_as, bias, C, N, K);          \
        CK(cudaGetLastError());                                                    \
        break;                                                                     \
      }                                                                            \
      g_fallback_mn++;                                                             \
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
    if (m->w8) g->use_w8 = 1;             /* t11：llama 流式已量化（SGL_W8）——直接 int8 上传 */
    /* M1：SGL_W4=1 时权重改用 NVFP4（0.625 B/参数 vs int8 的 1.0），
     * int8 副本不再上传。PROJ 通过 qw8[l].w4 转发到 FP4 kernel。*/
    {
      const char *w4e = getenv("SGL_W4");
      g->use_w4 = (w4e && atoi(w4e)) ? 1 : 0;
      if (g->use_w4) g->use_w8 = 1; /* 复用同一套槽位与 norms/biases 加载 */
    }
    if (g->use_w8) g->use_fp16 = 1;       /* embed table fp16, projections int8 */
    g->emb_h = g->use_fp16; /* t11：embed/lm_head 存 fp16（w8 模式默认）*/
  }

  /* M26a: no hard 4096 clamp anymore. Two-pass attention lifts the shared
   * memory limit; the remaining cap is VRAM (KV pool + weights scale with
   * ctx). Default 16384 fits Qwen2.5-0.5B fp32 into a 4GB card (RTX 2060)
   * thanks to tied-lm_head reuse; SGL_CUDA_CTX overrides. */
  {
    const char *e = getenv("SGL_CUDA_CTX");
    /* M2：量化 KV 下 17 KiB/token，旧上限 16384（为 FP32 KV 设的）不再适用 */
    const char *kqe = getenv("SGL_KVQ");
    int kvq_on = (kqe && atoi(kqe)) ? 1 : 0;
    int cap = (e && atoi(e) > 0) ? atoi(e) : (kvq_on ? (1 << 23) : 16384);
    if (g->ctx > cap) g->ctx = cap;
  }

  if (g->use_fp16) {
    /* t1 fp16 upload: weights as __half, fp32 activations everywhere else */
    int tied = (m->lm_head.data == m->wte.data);
    /* M0-显存：untied 时 embed 走 mapped host（每 token 只查 1 行，实测直读
     * 51 GB/s、一行 10 KiB 仅 0.2 µs），腾出 2.54 GB 给 32 个 GDN 状态槽。
     * tied 时 wte_h 同时是 lm_head（每 token 全读），必须留在显存。
     * SGL_EMB_DEV=1 可强制回显存（对照实验 / 无 UVA 环境）。*/
    g->wte_h = NULL;
    g->wte_h_host = NULL;
    if (!tied && !getenv("SGL_EMB_DEV")) {
      g->wte_h_host = host_map_h(m->wte.data, (size_t)m->vocab * m->n_embd);
      if (g->wte_h_host) {
        g->wte_h = map_dev_ptr(g->wte_h_host);
        printf("[cuda] embed on mapped host: frees %.2f GiB VRAM\n",
               (double)m->vocab * m->n_embd * 2.0 / 1073741824.0);
      }
    }
    if (!g->wte_h) CT("upload wte (fp16) V=%d E=%d", m->vocab, m->n_embd);
    if (!g->wte_h) g->wte_h = dev_copy_h(m->wte.data, (size_t)m->vocab * m->n_embd);
    CT("upload final_norm");
    g->final_norm = dev_copy(m->final_norm.data, (size_t)m->n_embd);
    /* tied embeddings: the half lm_head aliases the half embed table */
    if (tied) {
      g->lm_head_h = g->wte_h;
    } else if (g->use_w4) {
      /* M1：lm_head 每 token 全读，是第三大权重（2.54 GB fp16）。
       * 换 FP4 后 0.795 GB，直接减少 1.75 GB 的每 token 带宽。*/
      if (dev_copy_w4(m->lm_head.data, m->vocab, m->n_embd, &g->lm_head_w4) != 0) goto fail;
      g->lm_head_h = NULL;
    } else {
      g->lm_head_h = dev_copy_h(m->lm_head.data, (size_t)m->vocab * m->n_embd);
    }
    if (!g->wte_h || !g->final_norm) goto fail;
    if (!g->use_w4 && !g->lm_head_h) goto fail;
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
  /* t11 qwen3_5 hybrid：qk-norm / partial rope / lin 权重指针 */
  g->qn = (float **)calloc(g->L, sizeof(float *));
  g->kn = (float **)calloc(g->L, sizeof(float *));
  g->l_ba = (float **)calloc(g->L, sizeof(float *));
  g->l_conv = (float **)calloc(g->L, sizeof(float *));
  g->l_norm = (float **)calloc(g->L, sizeof(float *));
  g->l_alog = (float **)calloc(g->L, sizeof(float *));
  g->l_dtb = (float **)calloc(g->L, sizeof(float *));
  g->lkind = (unsigned char *)calloc(g->L, 1);
  g->kvo = (int *)malloc(sizeof(int) * (size_t)g->L);
  g->linvo = (int *)malloc(sizeof(int) * (size_t)g->L);
  g->QD = m->q_dim > 0 ? m->q_dim : g->E;
  g->q_gate = m->q_gate;
  g->norm_p1 = m->norm_p1;
  g->rot_dim = m->rot_dim > 0 ? m->rot_dim : g->D;
  g->lin_nk = m->lin_nk; g->lin_nh = m->lin_nh; g->lin_kd = m->lin_kd;
  g->lin_vd = m->lin_vd; g->lin_conv_k = m->lin_conv; g->lin_ratio = m->lin_ratio;
  g->lin_n = m->n_lin;
  if (m->layer_kind)
    for (int l = 0; l < g->L; l++) g->lkind[l] = (unsigned char)m->layer_kind[l];
  { /* t11：紧凑 KV 映射——lin 层不占 KV（27B 模型 64 层只有 16 层要 KV，省 3/4）*/
    int nn = 0, nl = 0;
    for (int l = 0; l < g->L; l++) {
      if (g->lkind[l] == 2) {
        g->kvo[l] = -1;
        g->linvo[l] = nl++;
      } else {
        g->kvo[l] = nn++;
        g->linvo[l] = -1;
      }
    }
    g->n_kv_layers = nn;
  }
  { /* partial-rope 逆频率表（m->rot_invf 已是层主序 [L][half]，直接上传）*/
    int half = g->rot_dim / 2;
    if (m->rot_invf && half > 0) {
      size_t nb = sizeof(double) * (size_t)g->L * half;
      void *dst = NULL;
      if (cudaMalloc(&dst, nb) == cudaSuccess) {
        cudaMemcpy(dst, m->rot_invf, nb, cudaMemcpyHostToDevice);
        g->pinvf = (double *)dst;
      }
    }
  }
  /* lin 递归状态池：[n_lin][slots][nh][vd][kd] 转置布局（零初始化）。
   * 与 CPU 布局 [h][i][d] 互为转置——GPU 内核按行访问 [d][i] 才合并。*/
  g->lin_slots = m->n_lin_slots > 0 ? m->n_lin_slots : 8;
  {
    const char *se = getenv("SGL_LIN_SLOTS");
    if (se && atoi(se) > 0) g->lin_slots = atoi(se);
  }
  if (g->lin_n > 0 && g->lin_nh > 0) {
    size_t ssz = (size_t)g->lin_n * g->lin_slots * g->lin_nh * g->lin_kd * g->lin_vd;
    if (cudaMalloc((void **)&g->l_state, ssz * sizeof(float)) != cudaSuccess)
      g->l_state = NULL;
    else
      cudaMemset(g->l_state, 0, ssz * sizeof(float));
  }
  if (cudaMalloc((void **)&g->d_lin_slot, sizeof(int) * 32) == cudaSuccess) {
    int z32[32] = {0};
    cudaMemcpy(g->d_lin_slot, z32, sizeof(int) * 32, cudaMemcpyHostToDevice);
    cudaHostAlloc((void **)&g->h_lin_slot, sizeof(int), cudaHostAllocDefault);
    *g->h_lin_slot = 0;
    cudaHostAlloc((void **)&g->h_dec_slots, sizeof(int) * 32, cudaHostAllocDefault);
    if (g->h_dec_slots)
      for (int i = 0; i < 32; i++) g->h_dec_slots[i] = 0;
    cudaHostAlloc((void **)&g->h_log1, sizeof(float) * (size_t)g->V, cudaHostAllocDefault);
  }
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
  if (g->use_w4) {
    if (w4_alloc_layers(&g->qw4, g->L) || w4_alloc_layers(&g->kw4, g->L) ||
        w4_alloc_layers(&g->vw4, g->L) || w4_alloc_layers(&g->ow4, g->L) ||
        w4_alloc_layers(&g->gw4, g->L) || w4_alloc_layers(&g->uw4, g->L) ||
        w4_alloc_layers(&g->dw4, g->L))
      goto fail;
  }

  for (int l = 0; l < g->L; l++) {
    const sgl_llama_layer *L = &m->layers[l];
    int is_lin = (m->layer_kind && m->layer_kind[l] == 2) ||
                 (!m->layer_kind && L->lin_idx >= 0);
    /* t11：投影宽度按层类型/门控定：lin 层 q 槽=qkvz、o 槽=[E,vald]；
     * full 层 q_gate 时 q 槽=2QD，o 槽=[E,QD]（QD 可能 ≠ E）*/
    int qN = is_lin ? (2 * m->lin_nk * m->lin_kd + 2 * m->lin_nh * m->lin_vd)
                    : (g->q_gate ? 2 * g->QD : g->QD);
    int lin_vald = m->lin_nh * m->lin_vd; /* t11：lin 层 v/o 总宽（nh*vd，不是 vd）*/
    if (g->use_w8) {
      /* t11：流式加载时 host 已是 int8（L->q8...），直传；否则 f32 现量化。
     * o 槽：lin=[E,vald]（out_proj）、full=[E,QD]。*/
      if (g->use_w4) {
        /* M1：直接 int8 → FP4，不必先上传 int8（显存只够一套）。
         * 有 host int8 时用它，否则回退到 f32 现量化。*/
        int rq, rk = 0, rv = 0, ro, rg, ru, rd;
        if (L->q8 && L->sq8) rq = dev_copy_w4_from_i8(L->q8, L->sq8, qN, g->E, &g->qw4[l]);
        else if (is_lin) rq = dev_copy_w4(L->lin_qkvz_w.data, qN, g->E, &g->qw4[l]);
        else rq = dev_copy_w4(L->q_w.data, qN, g->E, &g->qw4[l]);
        if (!is_lin) {
          if (L->k8 && L->sk8) rk = dev_copy_w4_from_i8(L->k8, L->sk8, g->KD, g->E, &g->kw4[l]);
          else rk = dev_copy_w4(L->k_w.data, g->KD, g->E, &g->kw4[l]);
          if (L->v8 && L->sv8) rv = dev_copy_w4_from_i8(L->v8, L->sv8, g->KD, g->E, &g->vw4[l]);
          else rv = dev_copy_w4(L->v_w.data, g->KD, g->E, &g->vw4[l]);
          if (L->o8 && L->so8) ro = dev_copy_w4_from_i8(L->o8, L->so8, g->E, g->QD, &g->ow4[l]);
          else ro = dev_copy_w4(L->o_w.data, g->E, g->QD, &g->ow4[l]);
        } else {
          if (L->o8 && L->so8) ro = dev_copy_w4_from_i8(L->o8, L->so8, g->E, lin_vald, &g->ow4[l]);
          else ro = dev_copy_w4(L->lin_out_w.data, g->E, lin_vald, &g->ow4[l]);
        }
        if (L->g8 && L->sg8) rg = dev_copy_w4_from_i8(L->g8, L->sg8, g->I, g->E, &g->gw4[l]);
        else rg = dev_copy_w4(L->gate_w.data, g->I, g->E, &g->gw4[l]);
        if (L->u8 && L->su8) ru = dev_copy_w4_from_i8(L->u8, L->su8, g->I, g->E, &g->uw4[l]);
        else ru = dev_copy_w4(L->up_w.data, g->I, g->E, &g->uw4[l]);
        if (L->d8 && L->sd8) rd = dev_copy_w4_from_i8(L->d8, L->sd8, g->E, g->I, &g->dw4[l]);
        else rd = dev_copy_w4(L->down_w.data, g->E, g->I, &g->dw4[l]);
        if (rq || rk || rv || ro || rg || ru || rd) goto fail;
        /* PROJ 靠这些指针转发到 FP4 kernel（int8 数据不上传）*/
        g->qw8[l].w4 = &g->qw4[l];
        g->kw8[l].w4 = is_lin ? NULL : &g->kw4[l];
        g->vw8[l].w4 = is_lin ? NULL : &g->vw4[l];
        g->ow8[l].w4 = &g->ow4[l];
        g->gw8[l].w4 = &g->gw4[l];
        g->uw8[l].w4 = &g->uw4[l];
        g->dw8[l].w4 = &g->dw4[l];
      } else {
      int rq = dev_copy_w8_i8(L->q8, L->sq8, qN, g->E, &g->qw8[l]);
      int rk = is_lin ? 0 : dev_copy_w8_i8(L->k8, L->sk8, g->KD, g->E, &g->kw8[l]);
      int rv = is_lin ? 0 : dev_copy_w8_i8(L->v8, L->sv8, g->KD, g->E, &g->vw8[l]);
      int ro = is_lin ? dev_copy_w8_i8(L->o8, L->so8, g->E, lin_vald, &g->ow8[l])
                      : dev_copy_w8_i8(L->o8, L->so8, g->E, g->QD, &g->ow8[l]);
      int rg = dev_copy_w8_i8(L->g8, L->sg8, g->I, g->E, &g->gw8[l]);
      int ru = dev_copy_w8_i8(L->u8, L->su8, g->I, g->E, &g->uw8[l]);
      int rd = dev_copy_w8_i8(L->d8, L->sd8, g->E, g->I, &g->dw8[l]);
      if (rq || rk || rv || ro || rg || ru || rd) {
        /* 无 host int8（小模型 f32 检查点且未设 SGL_W8）时回退现量化。
         * lin 层没有 q/k/v/o_proj——q 槽是 in_proj_qkvz、o 槽是 out_proj。*/
        if (is_lin) {
          if (dev_copy_w8(L->lin_qkvz_w.data, qN, g->E, &g->qw8[l]) != 0) goto fail;
          if (dev_copy_w8(L->lin_out_w.data, g->E, lin_vald, &g->ow8[l]) != 0) goto fail;
        } else {
          if (dev_copy_w8(L->q_w.data, qN, g->E, &g->qw8[l]) != 0) goto fail;
          if (dev_copy_w8(L->k_w.data, g->KD, g->E, &g->kw8[l]) != 0 ||
              dev_copy_w8(L->v_w.data, g->KD, g->E, &g->vw8[l]) != 0 ||
              dev_copy_w8(L->o_w.data, g->E, g->QD, &g->ow8[l]) != 0)
            goto fail;
        }
        if (dev_copy_w8(L->gate_w.data, g->I, g->E, &g->gw8[l]) != 0 ||
            dev_copy_w8(L->up_w.data, g->I, g->E, &g->uw8[l]) != 0 ||
            dev_copy_w8(L->down_w.data, g->E, g->I, &g->dw8[l]) != 0)
          goto fail;
      }
      }
      /* norms and biases still fp32 */
      g->ln1[l] = dev_copy(L->inp_norm.data, (size_t)g->E);
      g->ln2[l] = dev_copy(L->post_norm.data, (size_t)g->E);
      if (!is_lin) {
        g->qn[l] = L->q_norm.data ? dev_copy(L->q_norm.data, (size_t)g->D) : NULL;
        g->kn[l] = L->k_norm.data ? dev_copy(L->k_norm.data, (size_t)g->D) : NULL;
        g->qb[l] = L->q_b.data ? dev_copy(L->q_b.data, (size_t)g->QD) : NULL;
        g->kb[l] = L->k_b.data ? dev_copy(L->k_b.data, (size_t)g->KD) : NULL;
        g->vb[l] = L->v_b.data ? dev_copy(L->v_b.data, (size_t)g->KD) : NULL;
      } else {
        /* t11 lin：小张量 f32 直传（ba/conv/norm/a_log/dt_bias）*/
        g->l_ba[l] = dev_copy(L->lin_ba_w.data, (size_t)2 * m->lin_nh * g->E);
        g->l_conv[l] = dev_copy(L->lin_conv_w.data,
                                (size_t)(2 * m->lin_nk * m->lin_kd + lin_vald) *
                                    m->lin_conv);
        g->l_norm[l] = dev_copy(L->lin_norm_w.data, (size_t)m->lin_vd);
        g->l_alog[l] = dev_copy(L->lin_a_log, (size_t)m->lin_nh);
        g->l_dtb[l] = dev_copy(L->lin_dt_bias, (size_t)m->lin_nh);
        if (!g->l_ba[l] || !g->l_conv[l] || !g->l_norm[l] || !g->l_alog[l] || !g->l_dtb[l])
          goto fail;
      }
      continue; /* skip fp32/fp16 projection upload — VRAM saving */
    }
    if (g->use_fp16) {
      if (is_lin) {
        fprintf(stderr, "sgl_cuda_from_llama: linear-attn layer %d needs SGL_CUDA_W8=1\n", l);
        goto fail;
      }
      g->qw_h[l] = dev_copy_h(L->q_w.data, (size_t)qN * g->E);
      g->kw_h[l] = dev_copy_h(L->k_w.data, (size_t)g->KD * g->E);
      g->vw_h[l] = dev_copy_h(L->v_w.data, (size_t)g->KD * g->E);
      g->ow_h[l] = dev_copy_h(L->o_w.data, (size_t)g->E * g->QD);
      g->gw_h[l] = dev_copy_h(L->gate_w.data, (size_t)g->I * g->E);
      g->uw_h[l] = dev_copy_h(L->up_w.data, (size_t)g->I * g->E);
      g->dw_h[l] = dev_copy_h(L->down_w.data, (size_t)g->E * g->I);
      if (!g->qw_h[l] || !g->gw_h[l] || !g->dw_h[l]) goto fail;
      /* norms and biases stay fp32 (attention path is fp32 throughout) */
      g->ln1[l] = dev_copy(L->inp_norm.data, (size_t)g->E);
      g->ln2[l] = dev_copy(L->post_norm.data, (size_t)g->E);
      g->qn[l] = L->q_norm.data ? dev_copy(L->q_norm.data, (size_t)g->D) : NULL;
      g->kn[l] = L->k_norm.data ? dev_copy(L->k_norm.data, (size_t)g->D) : NULL;
      g->qb[l] = L->q_b.data ? dev_copy(L->q_b.data, (size_t)g->QD) : NULL;
      g->kb[l] = L->k_b.data ? dev_copy(L->k_b.data, (size_t)g->KD) : NULL;
      g->vb[l] = L->v_b.data ? dev_copy(L->v_b.data, (size_t)g->KD) : NULL;
      continue;
    }
    if (is_lin) {
      fprintf(stderr, "sgl_cuda_from_llama: linear-attn layer %d needs SGL_CUDA_W8=1\n", l);
      goto fail;
    }
    g->ln1[l] = dev_copy(L->inp_norm.data, (size_t)g->E);
    g->qw[l] = dev_copy(L->q_w.data, (size_t)qN * g->E);
    g->kw[l] = dev_copy(L->k_w.data, (size_t)g->KD * g->E);
    g->vw[l] = dev_copy(L->v_w.data, (size_t)g->KD * g->E);
    g->ow[l] = dev_copy(L->o_w.data, (size_t)g->E * g->QD);
    g->qb[l] = L->q_b.data ? dev_copy(L->q_b.data, (size_t)g->QD) : NULL;
    g->kb[l] = L->k_b.data ? dev_copy(L->k_b.data, (size_t)g->KD) : NULL;
    g->vb[l] = L->v_b.data ? dev_copy(L->v_b.data, (size_t)g->KD) : NULL;
    g->qn[l] = L->q_norm.data ? dev_copy(L->q_norm.data, (size_t)g->D) : NULL;
    g->kn[l] = L->k_norm.data ? dev_copy(L->k_norm.data, (size_t)g->D) : NULL;
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
  /* t11: 自动顶满上下文——SGL_CTX_AUTO=1 时按「权重上传完后的真实剩余显存」
   * 反算 ctx，而不是让人反复试数值。
   * KV 每 token = n_kv_layers(16) × KD(1024) × K/V × 4B = 128 KiB，
   * 因此 262144 上下文光 KV 就要 32 GiB —— 这张 32GB 卡物理放不下，
   * 自动顶满会落在显存允许的最大值（约 18-20K）。*/
  {
    const char *auto_e = getenv("SGL_CTX_AUTO");
    if (auto_e && atoi(auto_e) > 0) {
      size_t fb = 0, tb = 0;
      if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) {
        /* M2：量化池已启时按 17 KiB/token 算，而不是 FP32 的 128 KiB */
        const char *kqe = getenv("SGL_KVQ");
        int kqw = (kqe && atoi(kqe)) ? 1 : 0;
        size_t per_tok;
        if (kqw) {
          int nvals0 = g->HK * g->D;
          long tok_stride0 = (long)nvals0 + 2L * (nvals0 / SGL_KVQ_BLK) * 4L;
          per_tok = (size_t)(g->n_kv_layers ? g->n_kv_layers : g->L) * (size_t)tok_stride0;
        } else {
          per_tok = (size_t)(g->n_kv_layers ? g->n_kv_layers : g->L) *
                    (size_t)g->KD * 2ULL * 4ULL;
        }
        size_t reserve = 896ULL << 20; /* prefill 工作区峰值 ~405MB + logits/碎片 */
        size_t fit = fb > reserve ? (fb - reserve) / per_tok : 0;
        size_t cur = (size_t)g->ctx;
        fprintf(stderr, "[kv] auto-fit: free=%zuMB per_token=%zuB fit=%zu cur=%zu\n",
                fb >> 20, per_tok, fit, cur);
        if (fit >= 2048 && fit < cur) {
          size_t nc = (fit / 1024) * 1024; /* 向下取整到 1024 */
          if (nc < 2048) nc = 2048;
          fprintf(stderr, "[kv] auto-fit: ctx %zu -> %zu\n", cur, nc);
          ctx = nc;
          g->ctx = (int)nc;
        }
      }
    }
  }
  /* t11: lin layers do not occupy KV (compact layer number via kvo) */
  fprintf(stderr, "[kv] n_kv_layers=%d L=%d ctx=%zu KD=%d floats/each=%zu\n",
          g->n_kv_layers, g->L, ctx, g->KD,
          (size_t)(g->n_kv_layers ? g->n_kv_layers : g->L) * ctx * g->KD);
  /* M2/M6：先试量化分页池（SGL_KVQ=1）。成功就不分配 FP32 池：
   * 262K 下 FP32 要 32 GiB（装不下），4-bit 只要 4.26 GiB。*/
  {
    const char *kq = getenv("SGL_KVQ");
    int want = (kq && atoi(kq)) ? 1 : 0;
    if (want) {
      const char *hr = getenv("SGL_KVQ_HOST");
      const char *ns = getenv("SGL_KVQ_SLOTS");
      int host_res = (hr && atoi(hr)) ? 1 : 0;
      int nsw = ns ? atoi(ns) : 0;
      int nkv = g->n_kv_layers ? g->n_kv_layers : g->L;
      if (kvq_init(g, (long)ctx, nkv, g->HK, g->D, host_res, nsw) != 0) {
        fprintf(stderr, "[kvq] init failed -> falling back to FP32 KV\n");
        g->kvq.on = 0;
      }
    }
  }
  if (!g->kvq.on) {
    g->kc = dev_copy(NULL, (size_t)(g->n_kv_layers ? g->n_kv_layers : g->L) * ctx * g->KD);
    g->vc = dev_copy(NULL, (size_t)(g->n_kv_layers ? g->n_kv_layers : g->L) * ctx * g->KD);
  }
  /* activation workspace is growable now (sgl_cuda_reserve): only the KV
   * pool scales with ctx, long prompts don't need ctx-sized activations */
  if (!g->kvq.on && (!g->kc || !g->vc)) goto fail;
  CT("kv alloc done L=%d ctx=%zu KD=%d", g->L, ctx, g->KD);
  /* t11 lin：conv 历史池 + 每 token 复用的 scratch（行级，不随 prefill 长度增长）*/
  if (g->lin_n > 0 && g->lin_nh > 0) {
    int convd = 2 * g->lin_nk * g->lin_kd + g->lin_nh * g->lin_vd;
    size_t hs = (size_t)g->lin_n * g->lin_slots * convd * (g->lin_conv_k - 1);
    if (!g->l_state) goto fail;
    if (cudaMalloc((void **)&g->l_hist, hs * sizeof(float)) != cudaSuccess) goto fail;
    cudaMemset(g->l_hist, 0, hs * sizeof(float));
    /* M0: 这几个 scratch 改成按「最大 batch 行数」分配。batch decode 时 GDN 的
     * 中间结果（conv/l2norm/step/gnorm）按行存 [B][...]；原先只有 1 行，
     * B>1 会直接越界（实测 B=2 就 illegal memory access）。
     * 32 行 × rowN 合计约 4.7 MB，可忽略。*/
    const size_t lrows = 32;
    if (cudaMalloc((void **)&g->l_qkvz, lrows * (size_t)(convd + g->lin_nh * g->lin_vd) * sizeof(float)) != cudaSuccess)
      goto fail; /* rowN = 2keyd+2vald = convd + nh*vd */
    if (cudaMalloc((void **)&g->l_conv_io, lrows * (size_t)convd * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc((void **)&g->l_qn, lrows * (size_t)g->lin_nk * g->lin_kd * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&g->l_kn, lrows * (size_t)g->lin_nk * g->lin_kd * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&g->l_o, lrows * (size_t)g->lin_nh * g->lin_vd * sizeof(float)) != cudaSuccess)
      goto fail;
    g->lin_scratch_cap = (int)lrows;
    CT("lin state pool %d layers x %d slots (state %.1f MB + hist %.1f MB)",
       g->lin_n, g->lin_slots,
       (size_t)g->lin_n * g->lin_slots * g->lin_nh * g->lin_kd * g->lin_vd * 4.0 / 1048576,
       hs * 4.0 / 1048576);
  }
  if (g->use_fp16) printf("[cuda] fp16 weights ON (SGL_CUDA_FP16=1)\n");
  g->st = (void *)m->st; /* M3：MTP 加载复用主模型的 safetensors 上下文 */
  return g;

fail:
  fprintf(stderr, "sgl_cuda_from_llama: out of VRAM or copy failed\n");
  sgl_cuda_free(g);
  return NULL;
}

/* ================= M3: MTP 权重加载与草稿 =================
 *
 * 张量直接从主模型的 safetensors 上下文取——mtp.safetensors 是一个散分片，
 * sgl_st_get 的散分片扫描已经把它打开了。FP8 权重由 safetensors 层按
 * 128x128 块 + <name>_scale_inv 解成 f32，这里再量化成 int8（与主模型同一
 * 口径：per-row absmax），因为 M=1 的 GEMV 走 int8 dp4a。*/

static int mtp_get_1d(sgl_cuda_model *g, const char *name, int n, float **dst) {
  sgl_st_ctx *st = (sgl_st_ctx *)g->st;
  int64_t sh[1] = {n};
  sgl_tensor t = sgl_st_get(st, name, 1, sh);
  if (sgl_tensor_is_null(t)) {
    fprintf(stderr, "[mtp] missing tensor %s\n", name);
    return -1;
  }
  if (cudaMalloc((void **)dst, sizeof(float) * (size_t)n) != cudaSuccess) {
    sgl_st_drop(st, &t);
    return -1;
  }
  cudaMemcpy(*dst, t.data, sizeof(float) * (size_t)n, cudaMemcpyHostToDevice);
  sgl_st_drop(st, &t);
  return 0;
}

static int mtp_get_w8(sgl_cuda_model *g, const char *name, int N, int K, sgl_w8w *out) {
  sgl_st_ctx *st = (sgl_st_ctx *)g->st;
  int64_t sh[2] = {N, K};
  sgl_tensor t = sgl_st_get(st, name, 2, sh);
  if (sgl_tensor_is_null(t)) {
    fprintf(stderr, "[mtp] missing tensor %s\n", name);
    return -1;
  }
  int rc = dev_copy_w8(t.data, N, K, out);
  sgl_st_drop(st, &t);
  return rc;
}

static int mtp_alloc(float **p, size_t n) {
  if (cudaMalloc((void **)p, n * sizeof(float)) != cudaSuccess) return -1;
  cudaMemset(*p, 0, n * sizeof(float));
  return 0;
}

/* 加载 MTP 草稿头。kv_ctx = MTP 自有 KV 的容量。
 * 返回 0 = 就绪；非 0 = 没有 MTP 权重 / 环境不支持。*/
int sgl_cuda_mtp_load(sgl_cuda_model *g, int kv_ctx) {
  sgl_mtp *t;
  int E, D, KD, QD, I;
  if (!g) return -1;
  t = &g->mtp;
  if (t->ok) return 0;
  E = g->E; D = g->D; KD = g->KD; QD = g->QD; I = g->I;
  if (!g->st) { fprintf(stderr, "[mtp] no safetensors ctx\n"); return -1; }
  if (!g->use_w8) { fprintf(stderr, "[mtp] needs int8 weights (SGL_CUDA_W8=1)\n"); return -1; }
  if (kv_ctx < 2) kv_ctx = 2;
  t->ctx = kv_ctx;
  {
    const char *e = getenv("SGL_MTP_HIDDEN");
    t->hsrc = (e && !strcmp(e, "post")) ? 1 : 0;
  }

  if (mtp_get_w8(g, "mtp.fc.weight", E, 2 * E, &t->fc8)) goto fail;
  if (mtp_get_w8(g, "mtp.layers.0.self_attn.q_proj.weight", 2 * QD, E, &t->q8)) goto fail;
  if (mtp_get_w8(g, "mtp.layers.0.self_attn.k_proj.weight", KD, E, &t->k8)) goto fail;
  if (mtp_get_w8(g, "mtp.layers.0.self_attn.v_proj.weight", KD, E, &t->v8)) goto fail;
  if (mtp_get_w8(g, "mtp.layers.0.self_attn.o_proj.weight", E, QD, &t->o8)) goto fail;
  if (mtp_get_w8(g, "mtp.layers.0.mlp.gate_proj.weight", I, E, &t->g8)) goto fail;
  if (mtp_get_w8(g, "mtp.layers.0.mlp.up_proj.weight", I, E, &t->u8)) goto fail;
  if (mtp_get_w8(g, "mtp.layers.0.mlp.down_proj.weight", E, I, &t->d8)) goto fail;

  if (mtp_get_1d(g, "mtp.pre_fc_norm_embedding.weight", E, &t->enorm)) goto fail;
  if (mtp_get_1d(g, "mtp.pre_fc_norm_hidden.weight", E, &t->hnorm)) goto fail;
  if (mtp_get_1d(g, "mtp.layers.0.input_layernorm.weight", E, &t->ln1)) goto fail;
  if (mtp_get_1d(g, "mtp.layers.0.post_attention_layernorm.weight", E, &t->ln2)) goto fail;
  if (mtp_get_1d(g, "mtp.norm.weight", E, &t->post_norm)) goto fail;
  if (mtp_get_1d(g, "mtp.layers.0.self_attn.q_norm.weight", D, &t->qn)) goto fail;
  if (mtp_get_1d(g, "mtp.layers.0.self_attn.k_norm.weight", D, &t->kn)) goto fail;

  if (mtp_alloc(&t->e, E) || mtp_alloc(&t->proj_in, 2 * (size_t)E) ||
      mtp_alloc(&t->x, E) || mtp_alloc(&t->h, E) ||
      mtp_alloc(&t->q2, 2 * (size_t)QD) || mtp_alloc(&t->q, QD) ||
      mtp_alloc(&t->qgate, (size_t)g->H * D) || mtp_alloc(&t->k, KD) ||
      mtp_alloc(&t->v, KD) || mtp_alloc(&t->attn, (size_t)g->H * D) ||
      mtp_alloc(&t->gate, I) || mtp_alloc(&t->up, I))
    goto fail;
  /* 链式草稿：本层输出 hidden 两个副本 + 「最后落定位置」的 hidden */
  if (mtp_alloc(&t->chain_pre, E) || mtp_alloc(&t->chain_post, E) ||
      mtp_alloc(&t->hlast, E))
    goto fail;
  {
    /* 默认 post：27B 实测每步 2.83 token（pre 2.72）。SGL_MTP_CHAIN=pre 可切回 */
    const char *cc = getenv("SGL_MTP_CHAIN");
    t->chain_use_post = (cc && (cc[0] == 'p' || cc[0] == 'P') &&
                         (cc[1] == 'r' || cc[1] == 'R'))
                            ? 0
                            : 1;
  }
  {
    size_t kv = (size_t)kv_ctx * (size_t)KD;
    if (cudaMalloc((void **)&t->kc, kv * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc((void **)&t->vc, kv * sizeof(float)) != cudaSuccess) goto fail;
    cudaMemset(t->kc, 0, kv * sizeof(float));
    cudaMemset(t->vc, 0, kv * sizeof(float));
  }
  if (cudaMalloc((void **)&t->d_tok, sizeof(int)) != cudaSuccess) goto fail;
  if (cudaMalloc((void **)&t->d_arg, sizeof(int)) != cudaSuccess) goto fail;
  if (cudaMalloc((void **)&t->d_p, sizeof(float)) != cudaSuccess) goto fail;

  t->ok = 1;
  {
    double wb = (double)E * 2 * E + 4.0 * (double)QD * E + 2.0 * (double)KD * E +
                (double)E * QD + 2.0 * (double)I * E + (double)E * I;
    printf("[mtp] loaded: int8 weights %.2f MiB, kv_ctx %d (%.1f MiB), hsrc=%s\n",
           wb / 1048576.0, kv_ctx,
           2.0 * (double)kv_ctx * (double)KD * 4.0 / 1048576.0,
           t->hsrc ? "post-final-norm" : "pre-final-norm");
  }
  return 0;

fail:
  fprintf(stderr, "[mtp] load failed (out of VRAM or bad tensors)\n");
  return -1;
}

int sgl_cuda_mtp_ready(const sgl_cuda_model *g) { return g ? g->mtp.ok : 0; }
const float *sgl_cuda_mtp_hprev(const sgl_cuda_model *g) { return g ? g->mtp.hprev : NULL; }
const float *sgl_cuda_mtp_hlast(const sgl_cuda_model *g) { return g ? g->mtp.hlast : NULL; }

/* 一步草稿：给定位置 pos 的 hidden（device [E] 单行）与该位置的 token，
 * 预测 pos+1 的 token。
 *
 * hidden 应指 SGL_MTP_HIDDEN=pre（默认）时的 final-norm 之前残差流（g->x），
 * 或 =post 时的之后值（g->h）。哪个对由接受率裁决。
 * 副作用：MTP 把自己的 K/V 写在自有 KV 的 pos 行（不与主模型共享）。*/
int sgl_cuda_mtp_draft(sgl_cuda_model *g, int token, int pos, const float *hidden,
                       int *out_tok) {
  sgl_mtp *t;
  int E, D, KD, QD, I, H, HK;
  cudaStream_t s;
  if (!g || !out_tok) return -1;
  t = &g->mtp;
  if (!t->ok) return -1;
  if (pos < 0 || t->base + pos >= t->ctx) return -1;
  E = g->E; D = g->D; KD = g->KD; QD = g->QD; I = g->I; H = g->H; HK = g->HK;
  s = g->stream;
  float *kc = t->kc + (size_t)t->base * KD, *vc = t->vc + (size_t)t->base * KD;

  /* 1. emb(token) 与 hidden 各过一次 RMSNorm，拼成 proj_in（emb 在前）。
   *
   * 注意 grid：k_embed 是「一线程一元素、无 stride 循环」的，行宽 E=5120 必须
   * ceil(E/256) 个 block。写成 <<<1,256>>> 只会写前 256 个元素，剩下 4864
   * 是上一个请求的残留——症状是草稿变成「高频泛词、偶尔对」，接受率卡在 0.5。*/
  cudaMemcpyAsync(t->d_tok, &token, sizeof(int), cudaMemcpyHostToDevice, s);
  {
    int eb = (int)(((long)E + 255) / 256);
    if (g->wte_h)
      k_embed_h<<<eb, 256, 0, s>>>(g->wte_h, t->d_tok, t->e, 1, E);
    else
      k_embed<<<eb, 256, 0, s>>>(g->wte, t->d_tok, t->e, 1, E);
  }
  if (g->norm_p1) {
    k_rmsnorm_p1<<<1, 256, 256 * sizeof(float), s>>>(t->e, t->enorm, t->proj_in, 1, E, g->eps);
    k_rmsnorm_p1<<<1, 256, 256 * sizeof(float), s>>>(hidden, t->hnorm, t->proj_in + E, 1, E,
                                                    g->eps);
  } else {
    k_rmsnorm<<<1, 256, 256 * sizeof(float), s>>>(t->e, t->enorm, t->proj_in, 1, E, g->eps);
    k_rmsnorm<<<1, 256, 256 * sizeof(float), s>>>(hidden, t->hnorm, t->proj_in + E, 1, E,
                                                  g->eps);
  }
  /* 2. fc：x = W(2E->E) @ proj_in */
  PROJ(g, t->proj_in, NULL, NULL, &t->fc8, NULL, t->x, 1, E, 2 * E);

  /* 3. 一个完整的 full-attention decoder 层 */
  if (g->norm_p1)
    k_rmsnorm_p1<<<1, 256, 256 * sizeof(float), s>>>(t->x, t->ln1, t->h, 1, E, g->eps);
  else
    k_rmsnorm<<<1, 256, 256 * sizeof(float), s>>>(t->x, t->ln1, t->h, 1, E, g->eps);
  PROJ(g, t->h, NULL, NULL, &t->q8, NULL, t->q2, 1, 2 * QD, E);
  k_qgate_split<<<(QD + 255) / 256, 256, 0, s>>>(t->q2, t->q, t->qgate, 1, H, D);
  PROJ(g, t->h, NULL, NULL, &t->k8, NULL, t->k, 1, KD, E);
  PROJ(g, t->h, NULL, NULL, &t->v8, NULL, t->v, 1, KD, E);
  CK(cudaGetLastError());
  k_hrmsnorm<<<H, D, D * sizeof(float), s>>>(t->q, t->qn, 1, H, D, 1e-6f, g->norm_p1);
  k_hrmsnorm<<<HK, D, D * sizeof(float), s>>>(t->k, t->kn, 1, HK, D, 1e-6f, g->norm_p1);
  if (g->rot_dim && g->rot_dim < D && g->pinvf) {
    int half = g->rot_dim / 2;
    k_rope_p<<<(H * half + 255) / 256, 256, 0, s>>>(t->q, H, D, H, pos, g->pinvf, g->rot_dim);
    k_rope_p<<<(HK * half + 255) / 256, 256, 0, s>>>(t->k, HK, D, HK, pos, g->pinvf, g->rot_dim);
  } else {
    k_rope<<<(H * D / 2 + 255) / 256, 256, 0, s>>>(t->q, H, D, H, pos, g->freqs);
    k_rope<<<(HK * D / 2 + 255) / 256, 256, 0, s>>>(t->k, HK, D, HK, pos, g->freqs);
  }
  CK(cudaGetLastError());
  /* KV 写到自有池的 base+pos 行 */
  {
    size_t pitch = (size_t)KD * sizeof(float);
    if (cudaMemcpy2DAsync(kc + (size_t)pos * KD, pitch, t->k, pitch, pitch, 1,
                          cudaMemcpyDeviceToDevice, s) != cudaSuccess)
      return -1;
    if (cudaMemcpy2DAsync(vc + (size_t)pos * KD, pitch, t->v, pitch, pitch, 1,
                          cudaMemcpyDeviceToDevice, s) != cudaSuccess)
      return -1;
  }
  /* attention 覆盖 0..pos（含自身）*/
  k_attn_long<<<H, 256, 0, s>>>(t->q, kc, vc, t->attn, pos, 0, 1, H, HK, D,
                                1.0f / sqrtf((float)D), 0);
  k_qgate_apply<<<(QD + 255) / 256, 256, 0, s>>>(t->attn, t->qgate, (long)QD);
  CK(cudaGetLastError());
  PROJ(g, t->attn, NULL, NULL, &t->o8, NULL, t->h, 1, E, QD);
  k_residual<<<(E + 255) / 256, 256, 0, s>>>(t->x, t->h, E);
  if (g->norm_p1)
    k_rmsnorm_p1<<<1, 256, 256 * sizeof(float), s>>>(t->x, t->ln2, t->h, 1, E, g->eps);
  else
    k_rmsnorm<<<1, 256, 256 * sizeof(float), s>>>(t->x, t->ln2, t->h, 1, E, g->eps);
  PROJ(g, t->h, NULL, NULL, &t->g8, NULL, t->gate, 1, I, E);
  PROJ(g, t->h, NULL, NULL, &t->u8, NULL, t->up, 1, I, E);
  k_silu_mul<<<(I + 255) / 256, 256, 0, s>>>(t->gate, t->up, I);
  PROJ(g, t->gate, NULL, NULL, &t->d8, NULL, t->h, 1, E, I);
  k_residual<<<(E + 255) / 256, 256, 0, s>>>(t->x, t->h, E);
  CK(cudaGetLastError());

  /* 4. post_norm，然后共享 lm_head（不拥有自己的词表投影）*/
  if (g->norm_p1)
    k_rmsnorm_p1<<<1, 256, 256 * sizeof(float), s>>>(t->x, t->post_norm, t->h, 1, E, g->eps);
  else
    k_rmsnorm<<<1, 256, 256 * sizeof(float), s>>>(t->x, t->post_norm, t->h, 1, E, g->eps);
  /* 投机链式（M_k 形式）：本层输出 hidden 留给下一步当输入。pre = post_norm
   * 之前（t->x），post = 之后（t->h）。哪个对不确定，SGL_MTP_CHAIN 选，错了
   * 只会让后续草稿变差、被拒绝，不影响正确性。*/
  if (t->chain_pre) CK(cudaMemcpyAsync(t->chain_pre, t->x, sizeof(float) * (size_t)E,
                                       cudaMemcpyDeviceToDevice, s));
  if (t->chain_post) CK(cudaMemcpyAsync(t->chain_post, t->h, sizeof(float) * (size_t)E,
                                        cudaMemcpyDeviceToDevice, s));
  lm_head_gemm(g, t->h, g->d_lgb, 1);
  if (t->want_p)
    k_argmax_p<<<1, 256, 0, s>>>(g->d_lgb, g->V, t->d_arg, t->d_p);
  else
    k_argmax<<<1, 256, 0, s>>>(g->d_lgb, g->V, t->d_arg);
  if (cudaMemcpyAsync(out_tok, t->d_arg, sizeof(int), cudaMemcpyDeviceToHost, s) != cudaSuccess)
    return -1;
  if (t->want_p &&
      cudaMemcpyAsync(&t->last_p, t->d_p, sizeof(float), cudaMemcpyDeviceToHost, s) != cudaSuccess)
    return -1;
  return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -1;
}

void sgl_cuda_mtp_set_base(sgl_cuda_model *g, int base) {
  if (g) g->mtp.base = base > 0 ? base : 0;
}

int sgl_cuda_mtp_fill_kv(sgl_cuda_model *g, const int *toks, int n, int pos0,
                         const float *hprev) {
  sgl_mtp *t;
  int E, KD, HK, D, cap, b0;
  const float *src;
  cudaStream_t s;
  if (!g || !toks || n < 1) return -1;
  t = &g->mtp;
  if (!t->ok || pos0 < 0 || t->base + pos0 + n > t->ctx) return -1;
  E = g->E; KD = g->KD; HK = g->HK; D = g->D;
  s = g->stream;
  src = t->hsrc ? g->h : g->x;
  /* 块 ≤ ws_cap 行（IMMA scratch 的上限）；64 行时 scratch 约 6 MiB */
  cap = g->ws_cap < 64 ? g->ws_cap : 64;
  if (cap < 1) return -1;
  if (t->fill_cap < cap) {
    cudaFree(t->fe); cudaFree(t->fhid); cudaFree(t->fpin); cudaFree(t->fx);
    cudaFree(t->fh); cudaFree(t->fk); cudaFree(t->fv); cudaFree(t->ftok);
    t->fill_cap = 0;
    if (mtp_alloc(&t->fe, (size_t)cap * E) || mtp_alloc(&t->fhid, (size_t)cap * E) ||
        mtp_alloc(&t->fpin, (size_t)cap * 2 * E) || mtp_alloc(&t->fx, (size_t)cap * E) ||
        mtp_alloc(&t->fh, (size_t)cap * E) || mtp_alloc(&t->fk, (size_t)cap * KD) ||
        mtp_alloc(&t->fv, (size_t)cap * KD) ||
        cudaMalloc((void **)&t->ftok, sizeof(int) * (size_t)cap) != cudaSuccess)
      return -1;
    t->fill_cap = cap;
  }
  if (!t->hprev && mtp_alloc(&t->hprev, E)) return -1;

  for (b0 = 0; b0 < n; b0 += cap) {
    int nb = n - b0 < cap ? n - b0 : cap;
    size_t rowb = sizeof(float) * (size_t)E;
    /* 全局第 i 行配 hidden[pos0+i-1] = 前向的第 i-1 行；第 0 行配 hprev */
    if (b0 == 0) {
      CK(cudaMemcpyAsync(t->fhid, hprev ? hprev : src, rowb, cudaMemcpyDeviceToDevice, s));
      if (nb > 1)
        CK(cudaMemcpyAsync(t->fhid + E, src, rowb * (size_t)(nb - 1), cudaMemcpyDeviceToDevice, s));
    } else {
      CK(cudaMemcpyAsync(t->fhid, src + (size_t)(b0 - 1) * E, rowb * (size_t)nb,
                         cudaMemcpyDeviceToDevice, s));
    }
    CK(cudaMemcpyAsync(t->ftok, toks + b0, sizeof(int) * (size_t)nb, cudaMemcpyHostToDevice, s));
    {
      int eb = (int)(((long)nb * E + 255) / 256);
      if (g->wte_h)
        k_embed_h<<<eb, 256, 0, s>>>(g->wte_h, t->ftok, t->fe, nb, E);
      else
        k_embed<<<eb, 256, 0, s>>>(g->wte, t->ftok, t->fe, nb, E);
    }
    /* 与 sgl_cuda_mtp_draft 的前半段相同，只到 K/V 为止 */
    if (g->norm_p1) {
      k_rmsnorm_p1<<<nb, 256, 256 * sizeof(float), s>>>(t->fe, t->enorm, t->fh, nb, E, g->eps);
      k_rmsnorm_p1<<<nb, 256, 256 * sizeof(float), s>>>(t->fhid, t->hnorm, t->fx, nb, E, g->eps);
    } else {
      k_rmsnorm<<<nb, 256, 256 * sizeof(float), s>>>(t->fe, t->enorm, t->fh, nb, E, g->eps);
      k_rmsnorm<<<nb, 256, 256 * sizeof(float), s>>>(t->fhid, t->hnorm, t->fx, nb, E, g->eps);
    }
    k_cat2_rows<<<(int)(((long)nb * 2 * E + 255) / 256), 256, 0, s>>>(t->fh, t->fx, t->fpin, nb, E);
    PROJ(g, t->fpin, NULL, NULL, &t->fc8, NULL, t->fx, nb, E, 2 * E);
    if (g->norm_p1)
      k_rmsnorm_p1<<<nb, 256, 256 * sizeof(float), s>>>(t->fx, t->ln1, t->fh, nb, E, g->eps);
    else
      k_rmsnorm<<<nb, 256, 256 * sizeof(float), s>>>(t->fx, t->ln1, t->fh, nb, E, g->eps);
    PROJ(g, t->fh, NULL, NULL, &t->k8, NULL, t->fk, nb, KD, E);
    PROJ(g, t->fh, NULL, NULL, &t->v8, NULL, t->fv, nb, KD, E);
    CK(cudaGetLastError());
    k_hrmsnorm<<<nb * HK, D, D * sizeof(float), s>>>(t->fk, t->kn, nb, HK, D, 1e-6f, g->norm_p1);
    if (g->rot_dim && g->rot_dim < D && g->pinvf) {
      int half = g->rot_dim / 2;
      k_rope_p<<<(int)(((long)nb * HK * half + 255) / 256), 256, 0, s>>>(
          t->fk, nb * HK, D, HK, pos0 + b0, g->pinvf, g->rot_dim);
    } else {
      k_rope<<<(int)(((long)nb * HK * D / 2 + 255) / 256), 256, 0, s>>>(
          t->fk, nb * HK, D, HK, pos0 + b0, g->freqs);
    }
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(t->kc + (size_t)(t->base + pos0 + b0) * KD, t->fk,
                       sizeof(float) * (size_t)nb * KD, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(t->vc + (size_t)(t->base + pos0 + b0) * KD, t->fv,
                       sizeof(float) * (size_t)nb * KD, cudaMemcpyDeviceToDevice, s));
  }
  CK(cudaMemcpyAsync(t->hprev, src + (size_t)(n - 1) * E, sizeof(float) * (size_t)E,
                     cudaMemcpyDeviceToDevice, s));
  return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -1;
}

/* 便捷计时（单调秒），MTP 诊断用。sgl_hnow 定义在文件后半段，这里需要
 * 前面的声明。*/
static double mtp_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* MTP 自验证：teacher-forced 接受率。
 *
 * 关键点（第一版错在这里）：MTP 层有**自己的** KV cache，它必须像主模型
 * 一样从位置 0 顺序填满。只给草稿位置写 KV、让 prompt 那段留零，等于让草稿
 * 看不见上下文——实测接受率只有 40%。正确做法：
 *   1) 主模型贪心跑出参考流
 *   2) 整条序列（prompt + 参考流）过一遍 forward_seg_rows，把每个位置的
 *      hidden 拿全（g->x 本来就有 n 行）
 *   3) 从位置 0 开始逐位置跑 MTP（顺序写它自己的 KV），预测下一个 token
 *
 * 正确实现约 0.6–0.85；拼接顺序/归一化位置错了约 0–0.05。*/
int sgl_cuda_mtp_accept_test(sgl_cuda_model *m, const int *prompt, int np, int ngen,
                             double *acc, int *n_ok) {
  int E, i, ok = 0, tot = 0, N;
  int *gen = NULL, *seq = NULL;
  float *hidall = NULL;
  double dr_ms = 0.0;
  int ndr = 0;
  static float *logits = NULL;
  if (!m || !prompt || np <= 0 || ngen <= 1) return -1;
  if (!m->mtp.ok) return -1;
  m->mtp.base = 0; /* 自测从位置 0 起，用第 0 段 */
  if (getenv("SGL_MTP_DBG")) fprintf(stderr, "[mtp] stage: accept_test enter\n");
  E = m->E;
  N = np + ngen;
  gen = (int *)calloc((size_t)ngen, sizeof(int));
  seq = (int *)calloc((size_t)N, sizeof(int));
  if (!gen || !seq) { free(gen); free(seq); return -1; }
  if (!logits) logits = (float *)malloc(sizeof(float) * (size_t)m->V);
  if (!logits) { free(gen); free(seq); return -1; }
  if (cudaMalloc((void **)&hidall, sizeof(float) * (size_t)N * E) != cudaSuccess) {
    free(gen); free(seq);
    return -1;
  }

  /* 1. 参考流：逐 token 贪心 */
  {
    int pos = np;
    if (getenv("SGL_MTP_DBG")) fprintf(stderr, "[mtp] stage: prefill np=%d\n", np);
    if (sgl_cuda_forward_seg(m, prompt, np, 0, 0, logits) != 0) goto fail;
    for (i = 0; i < ngen; i++) {
      int tk = 0;
      float best = -INFINITY;
      for (int v = 0; v < m->V; v++)
        if (logits[v] > best) { best = logits[v]; tk = v; }
      gen[i] = tk;
      if (i + 1 < ngen) {
        if (sgl_cuda_forward_seg(m, &tk, 1, pos, 0, logits) != 0) goto fail;
        pos++;
      }
    }
  }

  /* 2. 整条序列过一遍，拿全部位置的 hidden（=pre 取 x，=post 取 h）*/
  memcpy(seq, prompt, sizeof(int) * (size_t)np);
  for (i = 0; i < ngen; i++) seq[np + i] = gen[i];
  if (getenv("SGL_MTP_DBG")) fprintf(stderr, "[mtp] stage: ref done, forward all %d rows\n", N);
  /* logits_out 传 NULL：这一步只要 g->x/g->h 的 N 行 hidden，不要 lm_head。
   * （传个小缓冲会写 N*V 个 float 把堆冲掉。）*/
  if (sgl_cuda_forward_seg_rows(m, seq, N, 0, 0, NULL) != 0) goto fail;
  {
    const float *src = m->mtp.hsrc ? m->h : m->x;
    if (cudaMemcpy(hidall, src, sizeof(float) * (size_t)N * E, cudaMemcpyDeviceToDevice) !=
        cudaSuccess)
      goto fail;
  }

  /* 3. 从位置 0 顺序跑 MTP，顺带把它自己的 KV 顺序填满。
   *
   * 配对是**错开一格**的（MtpModel.forward 的文档写得很清楚）：
   *   位置 p 的输入 = (hidden[p-1], token[p])，输出预测 token[p+1]。
   * 直接同位置配对（hidden[p], token[p]）实测只有 37%，错就错在这里。
   * 序列真实长度下 p 从 np 开始才是真正的草稿位置（prompt 那段的运行只是
   * 为了填 MTP 自己的 KV），但两边都要跑，因为 KV 必须从 0 顺序建。*/
  for (i = 0; i + 1 < N; i++) {
    int d = 0;
    int hp = (i == 0) ? 0 : i - 1;
    double t0 = mtp_now();
    if (sgl_cuda_mtp_draft(m, seq[i], i, hidall + (size_t)hp * E, &d) != 0) goto fail;
    dr_ms += (mtp_now() - t0) * 1000.0;
    ndr++;
    if (i >= np) {
      if (getenv("SGL_MTP_DBG"))
        fprintf(stderr, "[mtp] dbg pos=%d draft=%d ref=%d %s\n", i, d, seq[i + 1],
                d == seq[i + 1] ? "HIT" : "miss");
      tot++;
      if (d == seq[i + 1]) ok++;
    }
  }
  if (acc) *acc = tot > 0 ? (double)ok / (double)tot : 0.0;
  if (n_ok) *n_ok = ok;
  printf("[mtp] acceptance (teacher-forced): %d/%d = %.1f%%  hsrc=%s  draft %.2f ms/step\n", ok,
         tot, tot > 0 ? 100.0 * (double)ok / (double)tot : 0.0,
         m->mtp.hsrc ? "post" : "pre", ndr > 0 ? dr_ms / ndr : 0.0);
  cudaFree(hidall);
  free(gen);
  free(seq);
  return 0;
fail:
  if (hidall) cudaFree(hidall);
  free(gen);
  free(seq);
  fprintf(stderr, "[mtp] accept_test failed\n");
  return -1;
}

/* ================= M2/M6：量化分页 KV 的分配 ================= */
/* 返回 0 成功；-1 失败（调用方便于降级到 FP32 路径）。
 * host_resident=1 时额外分配 pinned 权威副本（M6）；0 = 全驻留 VRAM（M2）。*/
static int kvq_init(sgl_cuda_model *g, long ctx, int nkv, int HK, int D, int host_resident,
                    int n_slots_want) {
  sgl_kvq *q = &g->kvq;
  long n_pages, slots;
  memset(q, 0, sizeof(*q));
  if (nkv <= 0 || HK <= 0 || D <= 0) return -1;
  if ((HK * D) % SGL_KVQ_BLK != 0) {
    fprintf(stderr, "[kvq] HK*D=%d not a multiple of %d -> quant KV unavailable\n", HK * D,
            SGL_KVQ_BLK);
    return -1;
  }
  q->nkv = nkv;
  q->P = SGL_KVQ_P;
  q->HK = HK;
  q->D = D;
  q->nvals = HK * D;
  q->nblk = q->nvals / SGL_KVQ_BLK;
  q->tok_stride = (long)q->nvals + 2L * q->nblk * 4L; /* 4bit 打包 + 两组 float scale */
  q->slot_layer = (long)q->P * q->tok_stride;
  n_pages = (ctx + q->P - 1) / q->P;
  q->n_pages = (int)n_pages;
  slots = (n_slots_want > 0 && n_slots_want < n_pages) ? n_slots_want : n_pages;
  q->n_slots = (int)slots;
  q->layer_slots = slots * q->slot_layer;

  if (cudaMalloc((void **)&q->d_pool, (size_t)q->nkv * q->layer_slots) != cudaSuccess) {
    q->d_pool = NULL;
    cudaGetLastError();
    return -1;
  }
  if (cudaMalloc((void **)&q->d_pt, sizeof(int) * (size_t)q->n_pages) != cudaSuccess) {
    cudaFree(q->d_pool);
    q->d_pool = NULL;
    cudaGetLastError();
    return -1;
  }
  q->h_pt = (int *)malloc(sizeof(int) * (size_t)q->n_pages);
  if (!q->h_pt) {
    cudaFree(q->d_pool);
    cudaFree(q->d_pt);
    q->d_pool = NULL;
    q->d_pt = NULL;
    return -1;
  }
  /* M2：页表恒等（全驻留）。M6：前 n_slots 页先驻留，其余 -1。*/
  for (long p = 0; p < n_pages; p++) q->h_pt[p] = (p < slots) ? (int)p : -1;
  if (cudaMemcpy(q->d_pt, q->h_pt, sizeof(int) * (size_t)q->n_pages, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    free(q->h_pt);
    cudaFree(q->d_pool);
    cudaFree(q->d_pt);
    q->h_pt = NULL;
    q->d_pool = NULL;
    q->d_pt = NULL;
    cudaGetLastError();
    return -1;
  }
  cudaMemset(q->d_pool, 0, (size_t)q->nkv * q->layer_slots);

  if (host_resident) {
    size_t bytes = (size_t)q->nkv * n_pages * q->slot_layer;
    if (cudaHostAlloc((void **)&q->h_pool, bytes, cudaHostAllocMapped) != cudaSuccess) {
      fprintf(stderr, "[kvq] pinned host KV for %zu pages failed: %.1f GiB\n", (size_t)n_pages,
              (double)bytes / 1073741824.0);
      cudaGetLastError();
      free(q->h_pt);
      cudaFree(q->d_pool);
      cudaFree(q->d_pt);
      q->h_pt = NULL;
      q->d_pool = NULL;
      q->d_pt = NULL;
      return -1;
    }
    memset(q->h_pool, 0, bytes);
    q->host_slots = 1;
  }
  q->on = 1;
  /* M2：split-KV 的部分和缓冲。只在 decode 且 B 小（并行度不够）时用。
   * 条目 = Bmax(32) × nkv_H(24...) × nsplit。按 H 真实值算。*/
  {
    int Hm = g->H > 0 ? g->H : 1;
    const char *ne = getenv("SGL_KVQ_NSPLIT");
    int ns = ne ? atoi(ne) : 32;
    if (ns < 1) ns = 1;
    if (ns > 128) ns = 128;
    q->nsplit = ns;
    long entries = 32L * Hm * ns;
    size_t bytes = (size_t)entries * (size_t)(2 + q->D) * sizeof(float);
    if (bytes < (64ULL << 20)) bytes = (64ULL << 20); /* 最小 64 MiB，避免小 D 不够用 */
    if (cudaMalloc((void **)&q->d_split, bytes) == cudaSuccess) {
      q->split_cap = (int)(bytes / ((size_t)(2 + q->D) * sizeof(float)));
      fprintf(stderr, "[kvq] split-KV buffer %.1f MiB (nsplit=%d, cap=%d entries)\n",
              (double)bytes / 1048576.0, ns, q->split_cap);
    } else {
      q->d_split = NULL;
      q->split_cap = 0;
      cudaGetLastError();
    }
  }
  fprintf(stderr,
          "[kvq] quant paged KV ON: nkv=%d P=%d n_pages=%d n_slots=%d nvals=%d tok_stride=%ld B\n"
          "[kvq]   VRAM pool %.2f GiB%s | per-token %.2f KiB (FP32 was %.2f KiB)\n",
          q->nkv, q->P, q->n_pages, q->n_slots, q->nvals, q->tok_stride,
          (double)(q->nkv * q->layer_slots) / 1073741824.0,
          q->host_slots ? " + full pinned host copy" : " (all resident)",
          (double)(q->nkv * q->tok_stride) / 1024.0, (double)(q->nkv * q->nvals * 8) / 1024.0);
  return 0;
}

void sgl_cuda_kvq_free(sgl_cuda_model *m) {
  sgl_kvq *q;
  if (!m) return;
  q = &m->kvq;
  if (q->d_pool) cudaFree(q->d_pool);
  if (q->d_pt) cudaFree(q->d_pt);
  if (q->d_split) cudaFree(q->d_split);
  if (q->h_pool) cudaFreeHost(q->h_pool);
  if (q->h_pt) free(q->h_pt);
  memset(q, 0, sizeof(*q));
}

int sgl_cuda_kvq_enabled(const sgl_cuda_model *m) { return m ? m->kvq.on : 0; }

/* ================= M3: 投机解码 ================= */
/* 把上一次前向末位的 hidden 存成草稿输入（row<0 = 末行）。*/
int sgl_cuda_mtp_set_hidden(sgl_cuda_model *m, int row) {
  const float *src;
  if (!m || !m->mtp.ok || !m->mtp.hlast) return -1;
  if (row < 0) row = (m->last_rows > 0) ? m->last_rows - 1 : 0;
  src = m->mtp.hsrc ? m->h : m->x;
  if (cudaMemcpyAsync(m->mtp.hlast, src + (size_t)row * (size_t)m->E,
                      sizeof(float) * (size_t)m->E, cudaMemcpyDeviceToDevice,
                      m->stream) != cudaSuccess)
    return -1;
  return cudaStreamSynchronize(m->stream) == cudaSuccess ? 0 : -1;
}

int sgl_cuda_spec_begin(sgl_cuda_model *m, int W) {
  int rows, convd;
  long s_layer, h_layer;
  size_t nS, nH;
  if (!m || W < 1) return -1;
  if (m->lin_n <= 0 || !m->l_state || !m->l_hist) return -1;
  rows = W + 1;
  if (m->snap.S && m->snap.rows >= rows) return 0;
  if (m->snap.S) {
    cudaFree(m->snap.S);
    cudaFree(m->snap.H);
    m->snap.S = m->snap.H = NULL;
    m->snap.rows = 0;
  }
  convd = 2 * m->lin_nk * m->lin_kd + m->lin_nh * m->lin_vd;
  s_layer = (long)m->lin_nh * m->lin_kd * m->lin_vd;
  h_layer = (long)convd * (m->lin_conv_k - 1);
  nS = (size_t)rows * (size_t)m->lin_n * (size_t)s_layer;
  nH = (size_t)rows * (size_t)m->lin_n * (size_t)h_layer;
  if (cudaMalloc((void **)&m->snap.S, nS * sizeof(float)) != cudaSuccess) {
    m->snap.S = NULL;
    cudaGetLastError();
    return -1;
  }
  if (cudaMalloc((void **)&m->snap.H, nH * sizeof(float)) != cudaSuccess) {
    cudaFree(m->snap.S);
    m->snap.S = NULL;
    m->snap.H = NULL;
    cudaGetLastError();
    return -1;
  }
  m->snap.s_layer = s_layer;
  m->snap.h_layer = h_layer;
  m->snap.rows = rows;
  fprintf(stderr, "[spec] snapshot room for %d rows: %.1f MiB (state) + %.1f MiB (hist)\n", rows,
          (double)nS * 4.0 / 1048576.0, (double)nH * 4.0 / 1048576.0);
  return 0;
}

int sgl_cuda_mtp_drafts(sgl_cuda_model *m, int t0, int pos, int W, float minp, int *out,
                        float *probs) {
  int i, tok, d;
  float *hp;
  double joint = 1.0;
  if (!m || !m->mtp.ok || !out || W < 1) return -1;
  tok = t0;
  hp = m->mtp.hlast;
  m->mtp.want_p = (minp > 0.0f || probs) ? 1 : 0;
  m->mtp.last_p = 1.0f;
  for (i = 0; i < W; i++) {
    if (!hp) return i;
    if (sgl_cuda_mtp_draft(m, tok, pos + i, hp, &d) != 0) return i;
    joint *= m->mtp.last_p;
    /* 第 1 个草稿总留着（1 行草稿的 verify 与普通 decode 同价）；之后整条链都对的
     * 概率低于 minp 就停，这一个也不要 */
    if (i > 0 && joint < (double)minp) return i;
    out[i] = d;
    if (probs) probs[i] = m->mtp.last_p;
    tok = d;
    /* 链式：下一步用本层的输出 hidden（而不是主模型的）*/
    hp = m->mtp.chain_use_post ? m->mtp.chain_post : m->mtp.chain_pre;
  }
  return W;
}

int sgl_cuda_spec_verify(sgl_cuda_model *m, const int *toks, int n, int pos0, int kv_base,
                         float *logits_rows, int *k_out) {
  int j, k = 0, rc;
  if (!m || !toks || n < 2 || !logits_rows) return -1;
  m->spec_on = m->snap.S ? 1 : 0;
  rc = sgl_cuda_forward_seg_rows(m, toks, n, pos0, kv_base, logits_rows);
  m->spec_on = 0;
  if (rc != 0) return rc;
  for (j = 0; j + 1 < n; j++) {
    const float *L = logits_rows + (size_t)j * (size_t)m->V;
    int best = 0, v;
    float bv = L[0];
    for (v = 1; v < m->V; v++)
      if (L[v] > bv) {
        bv = L[v];
        best = v;
      }
    if (best == toks[j + 1])
      k++;
    else
      break;
  }
  if (k_out) *k_out = k;
  return 0;
}

int sgl_cuda_spec_commit(sgl_cuda_model *m, int slot, int k) {
  int l, rows = k + 1, t;
  size_t soff, hoff;
  if (!m || !m->snap.S) return -1;
  if (rows < 1 || rows > m->snap.rows) return -1;
  t = rows - 1; /* snapshot[t] = 处理完 t+1 行的状态 */
  for (l = 0; l < m->L; l++) {
    int ll = m->linvo ? m->linvo[l] : -1;
    if (ll < 0) continue;
    soff = ((size_t)ll * m->lin_slots + (size_t)slot) * (size_t)m->snap.s_layer;
    hoff = ((size_t)ll * m->lin_slots + (size_t)slot) * (size_t)m->snap.h_layer;
    size_t ssrc = (size_t)t * (size_t)m->lin_n * (size_t)m->snap.s_layer +
                  (size_t)ll * (size_t)m->snap.s_layer;
    size_t hsrc = (size_t)t * (size_t)m->lin_n * (size_t)m->snap.h_layer +
                  (size_t)ll * (size_t)m->snap.h_layer;
    if (cudaMemcpyAsync(m->l_state + soff, m->snap.S + ssrc,
                        sizeof(float) * (size_t)m->snap.s_layer, cudaMemcpyDeviceToDevice,
                        m->stream) != cudaSuccess)
      return -1;
    if (cudaMemcpyAsync(m->l_hist + hoff, m->snap.H + hsrc,
                        sizeof(float) * (size_t)m->snap.h_layer, cudaMemcpyDeviceToDevice,
                        m->stream) != cudaSuccess)
      return -1;
  }
  /* 最后落定位置的 hidden → 下一步草稿的输入 */
  if (m->mtp.ok && m->mtp.hlast) {
    const float *src = m->mtp.hsrc ? m->h : m->x;
    if (cudaMemcpyAsync(m->mtp.hlast, src + (size_t)k * (size_t)m->E,
                        sizeof(float) * (size_t)m->E, cudaMemcpyDeviceToDevice,
                        m->stream) != cudaSuccess)
      return -1;
  }
  return cudaStreamSynchronize(m->stream) == cudaSuccess ? 0 : -1;
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

    if (!g->kvo) g->kvo = (int *)malloc(sizeof(int) * (size_t)g->L);
  for (int l = 0; l < g->L; l++) g->kvo[l] = l; /* gpt2: every layer has KV (identity map) */
  ctx = (size_t)g->ctx;
  /* t11: 自动顶满上下文——SGL_CTX_AUTO=1 时按「权重上传完后的真实剩余显存」
   * 反算 ctx，而不是让人反复试数值。
   * KV 每 token = n_kv_layers(16) × KD(1024) × K/V × 4B = 128 KiB，
   * 因此 262144 上下文光 KV 就要 32 GiB —— 这张 32GB 卡物理放不下，
   * 自动顶满会落在显存允许的最大值（约 18-20K）。*/
  {
    const char *auto_e = getenv("SGL_CTX_AUTO");
    if (auto_e && atoi(auto_e) > 0) {
      size_t fb = 0, tb = 0;
      if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) {
        /* M2：量化池已启时按 17 KiB/token 算，而不是 FP32 的 128 KiB */
        const char *kqe = getenv("SGL_KVQ");
        int kqw = (kqe && atoi(kqe)) ? 1 : 0;
        size_t per_tok;
        if (kqw) {
          int nvals0 = g->HK * g->D;
          long tok_stride0 = (long)nvals0 + 2L * (nvals0 / SGL_KVQ_BLK) * 4L;
          per_tok = (size_t)(g->n_kv_layers ? g->n_kv_layers : g->L) * (size_t)tok_stride0;
        } else {
          per_tok = (size_t)(g->n_kv_layers ? g->n_kv_layers : g->L) *
                    (size_t)g->KD * 2ULL * 4ULL;
        }
        size_t reserve = 896ULL << 20; /* prefill 工作区峰值 ~405MB + logits/碎片 */
        size_t fit = fb > reserve ? (fb - reserve) / per_tok : 0;
        size_t cur = (size_t)g->ctx;
        fprintf(stderr, "[kv] auto-fit: free=%zuMB per_token=%zuB fit=%zu cur=%zu\n",
                fb >> 20, per_tok, fit, cur);
        if (fit >= 2048 && fit < cur) {
          size_t nc = (fit / 1024) * 1024; /* 向下取整到 1024 */
          if (nc < 2048) nc = 2048;
          fprintf(stderr, "[kv] auto-fit: ctx %zu -> %zu\n", cur, nc);
          ctx = nc;
          g->ctx = (int)nc;
        }
      }
    }
  }
  /* t11: lin layers do not occupy KV (compact layer number via kvo) */
  fprintf(stderr, "[kv] n_kv_layers=%d L=%d ctx=%zu KD=%d floats/each=%zu\n",
          g->n_kv_layers, g->L, ctx, g->KD,
          (size_t)(g->n_kv_layers ? g->n_kv_layers : g->L) * ctx * g->KD);
  g->kc = dev_copy(NULL, (size_t)(g->n_kv_layers ? g->n_kv_layers : g->L) * ctx * g->KD);
  g->vc = dev_copy(NULL, (size_t)(g->n_kv_layers ? g->n_kv_layers : g->L) * ctx * g->KD);
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
  if (g->wte_h_host) cudaFreeHost(g->wte_h_host);
  else if (g->wte_h) cudaFree(g->wte_h);
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
  /* t11 qwen3_5 hybrid */
  cudaFree(g->q2); cudaFree(g->qgate); cudaFree(g->pinvf);
  cudaFree(g->l_state); cudaFree(g->l_hist);
  cudaFree(g->l_qkvz); cudaFree(g->l_conv_io);
  cudaFree(g->l_qkvz_all); cudaFree(g->l_o_all); cudaFree(g->l_ba_all);
  cudaFree(g->l_qn); cudaFree(g->l_kn); cudaFree(g->l_o);
  cudaFree(g->d_lin_slot);
  if (g->h_lin_slot) cudaFreeHost(g->h_lin_slot);
  if (g->h_dec_slots) cudaFreeHost(g->h_dec_slots);
  if (g->h_log1) cudaFreeHost(g->h_log1);
  for (int l = 0; l < g->L; l++) {
    if (g->qn && g->qn[l]) cudaFree(g->qn[l]);
    if (g->kn && g->kn[l]) cudaFree(g->kn[l]);
    if (g->l_ba && g->l_ba[l]) cudaFree(g->l_ba[l]);
    if (g->l_conv && g->l_conv[l]) cudaFree(g->l_conv[l]);
    if (g->l_norm && g->l_norm[l]) cudaFree(g->l_norm[l]);
    if (g->l_alog && g->l_alog[l]) cudaFree(g->l_alog[l]);
    if (g->l_dtb && g->l_dtb[l]) cudaFree(g->l_dtb[l]);
  }
  free(g->qn); free(g->kn); free(g->l_ba); free(g->l_conv); free(g->l_norm);
  free(g->l_alog); free(g->l_dtb);
  free(g->lkind); free(g->kvo); free(g->linvo);
  w8_free_layers(g->qw8, g->L); w8_free_layers(g->kw8, g->L); w8_free_layers(g->vw8, g->L);
  w8_free_layers(g->ow8, g->L); w8_free_layers(g->gw8, g->L); w8_free_layers(g->uw8, g->L);
  w8_free_layers(g->dw8, g->L);
  w4_free_layers(g->qw4, g->L); w4_free_layers(g->kw4, g->L); w4_free_layers(g->vw4, g->L);
  w4_free_layers(g->ow4, g->L); w4_free_layers(g->gw4, g->L); w4_free_layers(g->uw4, g->L);
  w4_free_layers(g->dw4, g->L);
  if (g->lm_head_w4.w) cudaFree(g->lm_head_w4.w);
  if (g->lm_head_w4.bs) cudaFree(g->lm_head_w4.bs);
  if (g->w4_a8) cudaFree(g->w4_a8);
  if (g->w4_as) cudaFree(g->w4_as);
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
  /* M3：MTP 自有 KV 与主模型同段布局，前缀复用时一起搬（否则命中段的草稿看到的是旧值）*/
  if (g->mtp.ok && g->mtp.kc && src_base + len <= g->mtp.ctx && dst_base + len <= g->mtp.ctx) {
    size_t mb = (size_t)len * g->KD * sizeof(float);
    if (cudaMemcpyAsync(g->mtp.kc + (size_t)dst_base * g->KD, g->mtp.kc + (size_t)src_base * g->KD,
                        mb, cudaMemcpyDeviceToDevice, g->stream) != cudaSuccess ||
        cudaMemcpyAsync(g->mtp.vc + (size_t)dst_base * g->KD, g->mtp.vc + (size_t)src_base * g->KD,
                        mb, cudaMemcpyDeviceToDevice, g->stream) != cudaSuccess)
      return -1;
  }
  if (g->kvq.on) {
    /* M2：量化池的前缀复用。逐页搬（源/目都同偏移时才能一次搬一页）。
     * 请求段通常 64 对齐（seg_len 是 1024 的倍数），所以这里按页走。*/
    sgl_kvq *q = &g->kvq;
    if ((src_base % q->P) != 0 || (dst_base % q->P) != 0 || (len % q->P) != 0) {
      fprintf(stderr, "[kvq] copy_kv unaligned (%d->%d len %d, P=%d) -> skip reuse\n", src_base,
              dst_base, len, q->P);
      return -1;
    }
    for (int l = 0; l < g->L; l++) {
      int ll = g->kvo[l];
      if (ll < 0) continue;
      unsigned char *base = q->d_pool + (long)ll * q->layer_slots;
      for (int p = 0; p < len / q->P; p++) {
        int sp = (src_base / q->P) + p, dp = (dst_base / q->P) + p;
        int ss = q->h_pt[sp], ds = q->h_pt[dp];
        if (ss < 0 || ds < 0) return -1; /* 未驻留：不支持（M6 后续补）*/
        if (cudaMemcpyAsync(base + (long)ds * q->slot_layer, base + (long)ss * q->slot_layer,
                            (size_t)q->slot_layer, cudaMemcpyDeviceToDevice,
                            g->stream) != cudaSuccess)
          return -1;
      }
    }
    return cudaStreamSynchronize(g->stream) == cudaSuccess ? 0 : -1;
  }
  size_t bytes = (size_t) len * g->KD * sizeof(float); /* 同层内行连续：一次 1-D 拷贝 */
  for (int l = 0; l < g->L; l++) {
    size_t off = (size_t)g->kvo[l] * g->ctx * g->KD;
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
  int QD = m->q_dim > 0 ? m->q_dim : E;
  const char *fe = getenv("SGL_CUDA_FP16");
  int fp16 = (fe && atoi(fe)) ? 1 : 0;
  const char *we = getenv("SGL_CUDA_W8");
  int w8 = (we && atoi(we)) ? 1 : 0;
  if (m->w8) w8 = 1; /* t11: stream-quantized model (SGL_W8) saves the same */
  if (w8) fp16 = 1; /* embed stays fp16, projections go int8 */
  double wsz = fp16 ? 2.0 : 4.0;   /* embed / lm_head element size */
  double psz = w8 ? 1.0 : wsz;     /* projection element size */
  double ei = (double)(E > I ? E : I);
  /* t11: per layer-kind accounting (lin layers have no k/v, use qkvz/out rows;
   * full layers with q_gate double the q rows; QD may differ from E) */
  int n_full = 0, n_lin = 0;
  for (int l = 0; l < L; l++) {
    if (m->layer_kind && m->layer_kind[l] == 2) n_lin++;
    else n_full++;
  }
  if (!m->layer_kind) n_full = L;
  int q_gate = m->q_gate ? 1 : 0;
  double vald = m->lin_vd > 0 ? (double)m->n_head * m->lin_vd : 0; /* v 总宽 */
  double lin_rowN = m->lin_nk > 0 ? (2.0 * m->lin_nk * m->lin_kd + 2.0 * vald) : 0;
  double bytes = 0;
  bytes += (double)V * E * wsz;                          /* wte */
  if (m->lm_head.data && m->lm_head.data != m->wte.data) /* untied lm_head */
    bytes += (double)V * E * wsz;
  bytes += (double)n_full * ((2.0 + q_gate) * QD * E + 2.0 * KD * E + E * QD + 3.0 * I * E) * psz;
  bytes += (double)n_lin * ((lin_rowN + 96.0) * E + E * vald + 3.0 * I * E) * psz;
  bytes += (double)L * (6.0 * E + 2.0 * I) * 4.0;        /* norms + biases fp32 */
  /* M2：SGL_KVQ=1 时 KV 是 4-bit 分页池（每 token 约 17 KiB），不再是 FP32 128 KiB */
  {
    const char *kqe = getenv("SGL_KVQ");
    int kqw = (kqe && atoi(kqe)) ? 1 : 0;
    if (kqw && m->n_kv_head > 0 && m->head_dim > 0) {
      int nvals = m->n_kv_head * m->head_dim;
      int nblk = nvals / 128;
      long tok_stride = (long)nvals + 2L * nblk * 4L;
      bytes += (double)n_full * ctx * (double)tok_stride;
    } else {
      bytes += 2.0 * n_full * ctx * KD * 4.0;            /* KV pool (lin layers have none) */
    }
  }
  bytes += (4.0 * E + 2.0 * KD + 2.0 * I) * rows * 4.0;  /* activation workspace */
  if (fp16) bytes += ei * rows * 2.0;                    /* cublas fp16 input mirror */
  if (w8) bytes += ei * rows * 9.0;                      /* int8 A + int32 C + scales */
  if (n_lin > 0) { /* t11: lin recurrent state + conv history pools */
    int convd = 2 * m->lin_nk * m->lin_kd + m->lin_nh * m->lin_vd;
    bytes += (double)n_lin * m->n_lin_slots * m->lin_nh * m->lin_kd * m->lin_vd * 4.0;
    bytes += (double)n_lin * m->n_lin_slots * convd * (m->lin_conv - 1) * 4.0;
  }
  bytes += (double)V * 4.0 * 33.0;                       /* d_lgb grows to 32*B rows */
  bytes += 32.0 * 1048576.0;                             /* cublas workspace */
  bytes += 320.0 * 1048576.0;                             /* context + loaded modules */
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
      float *kcl = g->kc + (long)g->kvo[l] * g->ctx * KD;
      float *vcl = g->vc + (long)g->kvo[l] * g->ctx * KD;
      size_t row = (size_t)KD * sizeof(float);
      size_t src_pitch = (size_t)E3 * sizeof(float);
      size_t dst_off = (size_t)(kv_base + pos0) * row;
      CK(cudaMemcpy2DAsync((char *)kcl + dst_off, row, (char *)g->q + E * sizeof(float),
                           src_pitch, row, n, cudaMemcpyDeviceToDevice, g->stream));
      CK(cudaMemcpy2DAsync((char *)vcl + dst_off, row, (char *)g->q + 2 * E * sizeof(float),
                           src_pitch, row, n, cudaMemcpyDeviceToDevice, g->stream));
    }
    k_attn<<<n * H, 256, (size_t)(pos0 + n) * sizeof(float), g->stream>>>(
        g->q, g->kc + (long)g->kvo[l] * g->ctx * KD, g->vc + (long)g->kvo[l] * g->ctx * KD,
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
    lm_head_gemm(g, g->h + (long)(n - 1) * E, g->d_lgb, 1);
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
/* t11 性能分析：prefill 到底花在 attention 还是 GEMM。
 * 原 [prof] 只报 attention，看不出剩下的是什么。这里再加一个每次 forward
 * 的墙钟总时间（prof 分支末尾已经 cudaStreamSynchronize 过，所以从函数入口
 * 到这里的差值就是这一块的真实 GPU 时间）。*/
static double sgl_hnow(void) {
#ifdef _WIN32
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart / (double)f.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}
static int prof_on(void) {
  static int v = -1;
  if (v < 0) { const char *pe = getenv("SGL_CUDA_PROF"); v = (pe && atoi(pe)) ? 1 : 0; }
  return v;
}

static int fwd_seg_llama(sgl_cuda_model *g, const int *tokens, const float *emb,
                         int n, int pos0, int kv_base, float *logits_out,
                         int all_rows) {
  int E = g->E, H = g->H, HK = g->HK, D = g->D, KD = g->KD;
  double t_fwd0 = prof_on() ? sgl_hnow() : 0.0;
  if (n <= 0 || pos0 + n > g->ctx) return -1;
  if (kv_base + pos0 + n > g->ctx) return -1;
  g->last_rows = n; /* M3：投机首次开跑时靠它取 prefill 末位 hidden */
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
    int lin = g->lkind && g->lkind[l] == 2; /* t11 qwen3_5 hybrid */
    if (g->norm_p1)
      k_rmsnorm_p1<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->ln1[l], g->h, n, E,
                                                                g->eps);
    else
      k_rmsnorm<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->ln1[l], g->h, n, E,
                                                            g->eps);
    CK(cudaGetLastError());
    if (lin) {
      /* t11 G9：gated deltanet —— 逐 token（递归依赖）；qkvz/out_proj 走 PROJ（int8）*/
      int nk2 = g->lin_nk, kd2 = g->lin_kd, nh2 = g->lin_nh, vd2 = g->lin_vd;
      int keyd2 = nk2 * kd2;
      int gw2 = 2 * kd2 + 2 * g->lin_ratio * vd2;
      int rowN2 = 2 * keyd2 + 2 * nh2 * vd2;
      int convd2 = 2 * keyd2 + nh2 * vd2;
      int ll2 = g->linvo[l];
      int K2 = g->lin_conv_k;
      const size_t slot2 = (size_t)(g->h_lin_slot ? *g->h_lin_slot : 0);
      if (pos0 == 0 && n > 0) { /* t11：新序列从零状态开始（清**本序列自己的**槽）*/
        /* 之前基址是 ll2*lin_slots（即 slot 0），不管当前用哪个槽——并发时
         * 只有槽 0 被清零，其余请求从上一轮的脏递归状态起步，第二个 token 起全乱。*/
        size_t so = ((size_t)ll2 * g->lin_slots + slot2) * (size_t)nh2 * kd2 * vd2;
        size_t ho = ((size_t)ll2 * g->lin_slots + slot2) * (size_t)convd2 * (K2 - 1);
        CK(cudaMemsetAsync(g->l_state + so, 0, sizeof(float) * (size_t)nh2 * kd2 * vd2,
                           g->stream));
        CK(cudaMemsetAsync(g->l_hist + ho, 0, sizeof(float) * (size_t)convd2 * (K2 - 1),
                           g->stream));
      }
      /* t11 prefill 加速：整块的 qkvz 投影一次算完（M=n 的 GEMM）。
       * 以前是逐 token 的 M=1 GEMV：每 token 把 84MB 权重完整读一遍。*/
      if (prof) cudaEventRecord(g->pf0[l], g->stream);
      PROJ(g, g->h, g->qw[l], g->qw_h ? g->qw_h[l] : NULL,
           g->qw8 ? &g->qw8[l] : NULL, NULL, g->l_qkvz_all, n, rowN2, E);
      CK(cudaGetLastError());
      k_lin_ba<<<dim3(2 * nh2, n), 256, 256 * sizeof(float), g->stream>>>(
          g->h, g->l_ba[l], g->l_ba_all, 2 * nh2, E);
      CK(cudaGetLastError());
      /* t11 计时口径：lin 层的 pf0/pf1 只包住「递归本体」（批量投影是 GEMM，
       * 归到“其余”里）。上次 pf0 在 PROJ 之前，这里重记一次。*/
      /* t11 prefill 加速：整层常驻融合内核（状态在寄存器 / shared，内部循环整块 token）。
       * path=2：v2 寄存器版（kd=vd=128 时，最快）；path=1：v1 shared 版；
       * path=0：原来的逐 token 四段式（数值基准，SGL_LIN_PERSIST=0 可强制）。*/
      int path = (kd2 == 128 && vd2 == 128 && n > 0 && g->l_qkvz_all && g->l_o_all) ? 2 : 0;
      if (path == 0 && kd2 <= 128 && vd2 <= 128 && n > 0 && g->l_qkvz_all && g->l_o_all)
        path = 1;
      {
        const char *pe2 = getenv("SGL_LIN_PERSIST");
        if (pe2) { int v2 = atoi(pe2); if (v2 >= 0 && v2 <= 2) path = v2; }
      }
      size_t shb = 0;
      if (path == 2) {
        shb = (size_t)(2L * (K2 - 1) * kd2 + (long)(K2 - 1) * vd2 + 2L * kd2 +
                       3L * vd2 + 8L * vd2 + 2L * (kd2 / 32)) * sizeof(float);
      } else if (path == 1) {
        const int nt2 = 128;
        shb = (size_t)((long)kd2 * vd2 + 2L * (K2 - 1) * kd2 + (long)(K2 - 1) * vd2 +
                       2L * kd2 + 2L * vd2 + 2L * nt2) * sizeof(float);
        static int shopt = -1, shlim = 0;
        if (shopt < 0) {
          /* 上限要问设备：这卡（RTX 5090）每 block 动态 shared 只允许 99KB，
           * SM 总量也才 100KB（不是 Hopper 的 227KB）。之前写死 128KB 会被
           * cudaFuncSetAttribute 报 invalid argument，整层直接回退旧路径。*/
          int dev = 0;
          cudaGetDevice(&dev);
          if (cudaDeviceGetAttribute(&shlim, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                     dev) != cudaSuccess) {
            shlim = 48 * 1024;
            cudaGetLastError();
          }
          shopt = (cudaFuncSetAttribute((const void *)k_lin_layer_persist,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        shlim) == cudaSuccess) ? 1 : 0;
          if (!shopt) {
            fprintf(stderr, "[lin] persist shared opt-in failed (limit %d B): %s\n",
                    shlim, cudaGetErrorString(cudaGetLastError()));
            cudaGetLastError();
          }
        }
        if (!shopt || (long)shb > (long)shlim) path = 0;
      }
      if (prof) cudaEventRecord(g->pf0[l], g->stream);
      if (path == 2) {
        float *ss = NULL, *sh = NULL;
        long sr = 0, hr = 0;
        if (g->spec_on && g->snap.S && g->snap.rows >= n) {
          ss = g->snap.S + (long)ll2 * g->snap.s_layer;
          sh = g->snap.H + (long)ll2 * g->snap.h_layer;
          sr = (long)g->lin_n * g->snap.s_layer;
          hr = (long)g->lin_n * g->snap.h_layer;
        }
        k_lin_persist_reg<128, 128><<<nh2, 512, shb, g->stream>>>(
            g->l_qkvz_all, g->l_conv[l], g->l_hist, g->l_state, g->l_ba_all,
            g->l_alog[l], g->l_dtb[l], g->l_norm[l], g->l_o_all, n, nk2, nh2, K2,
            g->lin_ratio, gw2, g->eps,
            (long)(((size_t)ll2 * g->lin_slots + slot2) * (size_t)nh2 * kd2 * vd2),
            (long)(((size_t)ll2 * g->lin_slots + slot2) * (size_t)convd2 * (K2 - 1)),
            ss, sh, sr, hr);
        CK(cudaGetLastError());
      } else if (path == 1) {
        k_lin_layer_persist<<<nh2, 128, shb, g->stream>>>(
            g->l_qkvz_all, g->l_conv[l], g->l_hist, g->l_state, g->l_ba_all,
            g->l_alog[l], g->l_dtb[l], g->l_norm[l], g->l_o_all, n, nk2, kd2, nh2,
            vd2, K2, g->lin_ratio, gw2, g->eps,
            (long)(((size_t)ll2 * g->lin_slots + slot2) * (size_t)nh2 * kd2 * vd2),
            (long)(((size_t)ll2 * g->lin_slots + slot2) * (size_t)convd2 * (K2 - 1)));
        CK(cudaGetLastError());
      } else
      for (int t = 0; t < n; t++) {
        /* t11：投影已批量完成（见外层 PROJ），这里只跑递归主体 */
        float *qkvz_t = g->l_qkvz_all + (size_t)t * rowN2;
        float *o_t = g->l_o_all + (size_t)t * (nh2 * vd2);
        k_lin_conv<<<(convd2 + 127) / 128, 128, 0, g->stream>>>(
            g->l_conv[l], g->l_hist, qkvz_t, g->l_conv_io, nk2, kd2, nh2, vd2,
            g->lin_ratio, K2,
            ((size_t)ll2 * g->lin_slots + (size_t)*g->h_lin_slot) * convd2 * (K2 - 1));
        k_lin_l2norm<<<2 * nk2, 256, 256 * sizeof(float), g->stream>>>(
            g->l_conv_io, g->l_qn, g->l_kn, nk2, kd2);
        k_lin_step<<<nh2 * ((vd2 + 3) / 4), 128, 0, g->stream>>>(
            g->l_state, g->l_qn, g->l_kn, g->l_conv_io + 2 * keyd2, qkvz_t,
            g->l_ba_all + (size_t)t * 2 * nh2, g->l_alog[l], g->l_dtb[l], o_t, nh2, kd2, vd2,
            g->lin_ratio, gw2, 0.0f,
            ((size_t)ll2 * g->lin_slots + (size_t)*g->h_lin_slot) * nh2 * kd2 * vd2,
            (vd2 + 3) / 4);
        k_lin_gnorm<<<nh2, 256, 256 * sizeof(float), g->stream>>>(
            o_t, qkvz_t, g->l_norm[l], nh2, vd2, kd2, g->lin_ratio, gw2, g->eps);
        CK(cudaGetLastError());
        {
          static int dbgg = -1;
          if (dbgg < 0) { const char *e = getenv("SGL_CUDA_DBG"); dbgg = e ? atoi(e) : 0; }
          if (dbgg > 0 && t == 0) {
            float ov[2], zv[2], nw[2], cv[2];
            int zoff = 2 * kd2 + g->lin_ratio * vd2;
            cudaStreamSynchronize(g->stream);
            cudaMemcpy(ov, o_t, sizeof(float) * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(zv, qkvz_t + zoff, sizeof(float) * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(nw, g->l_norm[l], sizeof(float) * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(cv, g->l_conv_io + 2 * kd2 * g->lin_nk, sizeof(float) * 2,
                       cudaMemcpyDeviceToHost);
            fprintf(stderr, "[gpu gnorm L%d] o=[%.4g,%.4g] z=[%.4g,%.4g] nw=[%.4g,%.4g] v=[%.4g,%.4g]\n",
                    l, ov[0], ov[1], zv[0], zv[1], nw[0], nw[1], cv[0], cv[1]);
          }
        }
        {
          static int dbga = -1;
          if (dbga < 0) { const char *e = getenv("SGL_CUDA_DBG"); dbga = e ? atoi(e) : 0; }
          if (dbga > 0) {
            float zq[4];
            cudaStreamSynchronize(g->stream);
            for (int s2 = 0; s2 < 4; s2++)
              cudaMemcpy(zq + s2, qkvz_t + (size_t)s2 * (rowN2 / 4), sizeof(float),
                         cudaMemcpyDeviceToHost);
            fprintf(stderr, "[zdbg rowN=%d] [0]=%.4g [1/4]=%.4g [1/2]=%.4g [3/4]=%.4g\n", rowN2,
                    zq[0], zq[1], zq[2], zq[3]);
            float hh[2], oo[2], qv[2], ci[2], qn2[2], st[2];
            cudaMemcpy(hh, g->h + (size_t)t * E, sizeof(float) * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(oo, o_t, sizeof(float) * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(qv, qkvz_t, sizeof(float) * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(ci, g->l_conv_io, sizeof(float) * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(qn2, g->l_qn, sizeof(float) * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(st, g->l_state, sizeof(float) * 2, cudaMemcpyDeviceToHost);
            fprintf(stderr,
                    "[gpu linattn L%d t%d] qkvz=[%.4g,%.4g] conv=[%.4g,%.4g] qn=[%.4g,%.4g] "
                    "state=[%.4g,%.4g] o=[%.4g,%.4g] h=[%.4g,%.4g]\n",
                    l, t, qv[0], qv[1], ci[0], ci[1], qn2[0], qn2[1], st[0], st[1], oo[0], oo[1],
                    hh[0], hh[1]);
          }
        }
      }
      if (prof) cudaEventRecord(g->pf1[l], g->stream);
      /* t11 prefill 加速：整块 out_proj（M=n）一次算完，直接写回 h。*/
      PROJ(g, g->l_o_all, g->ow[l], g->ow_h ? g->ow_h[l] : NULL,
           g->ow8 ? &g->ow8[l] : NULL, NULL, g->h, n, E, nh2 * vd2);
      CK(cudaGetLastError());
      k_residual<<<(int)(((long)n * E + 255) / 256), 256, 0, g->stream>>>(g->x, g->h,
                                                                          (long)n * E);
      CK(cudaGetLastError());
      if (g->norm_p1)
        k_rmsnorm_p1<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->ln2[l], g->h, n,
                                                                  E, g->eps);
      else
        k_rmsnorm<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->ln2[l], g->h, n, E,
                                                              g->eps);
      CK(cudaGetLastError());
      PROJ(g, g->h, g->gw[l], g->gw_h ? g->gw_h[l] : NULL, g->gw8 ? &g->gw8[l] : NULL,
           NULL, g->gate, n, g->I, E);
      PROJ(g, g->h, g->uw[l], g->uw_h ? g->uw_h[l] : NULL, g->uw8 ? &g->uw8[l] : NULL,
           NULL, g->up, n, g->I, E);
      CK(cudaGetLastError());
      k_silu_mul<<<(int)(((long)n * g->I + 255) / 256), 256, 0, g->stream>>>(
          g->gate, g->up, (long)n * g->I);
      PROJ(g, g->gate, g->dw[l], g->dw_h ? g->dw_h[l] : NULL, g->dw8 ? &g->dw8[l] : NULL,
           NULL, g->h, n, E, g->I);
      CK(cudaGetLastError());
      k_residual<<<(int)(((long)n * E + 255) / 256), 256, 0, g->stream>>>(g->x, g->h,
                                                                          (long)n * E);
      CK(cudaGetLastError());
      {
        static int dbgl = -1;
        if (dbgl < 0) { const char *e = getenv("SGL_CUDA_DBG"); dbgl = e ? atoi(e) : 0; }
        if (dbgl > 0) {
          float hx[2];
          cudaStreamSynchronize(g->stream);
          cudaMemcpy(hx, g->x + (long)(n - 1) * E, sizeof(float) * 2, cudaMemcpyDeviceToHost);
          fprintf(stderr, "[gpu lin L%d] x=[%.7g,%.7g]\n", l, hx[0], hx[1]);
        }
      }
      continue; /* lin 层完 */
    }
    /* full/sliding 层（t11：q_gate / qk-norm / partial rope / [E,QD] o_proj）*/
    {
      float *qdst = g->q;
      int qN2 = g->QD;
      if (g->q_gate) {
        qdst = g->q2;
        qN2 = 2 * g->QD;
      }
      PROJ(g, g->h, g->qw[l], g->qw_h ? g->qw_h[l] : NULL, g->qw8 ? &g->qw8[l] : NULL,
           g->qb[l], qdst, n, qN2, E);
      if (g->q_gate) {
        k_qgate_split<<<(int)(((long)n * H * D + 255) / 256), 256, 0, g->stream>>>(
            g->q2, g->q, g->qgate, n, H, D);
        CK(cudaGetLastError());
      }
    }
    PROJ(g, g->h, g->kw[l], g->kw_h ? g->kw_h[l] : NULL, g->kw8 ? &g->kw8[l] : NULL, g->kb[l],
         g->k, n, KD, E);
    PROJ(g, g->h, g->vw[l], g->vw_h ? g->vw_h[l] : NULL, g->vw8 ? &g->vw8[l] : NULL, g->vb[l],
         g->v, n, KD, E);
    CK(cudaGetLastError());
    if (g->qn && g->qn[l])
      k_hrmsnorm<<<n * H, D, D * sizeof(float), g->stream>>>(g->q, g->qn[l], n, H, D,
                                                             1e-6f, g->norm_p1);
    if (g->kn && g->kn[l])
      k_hrmsnorm<<<n * HK, D, D * sizeof(float), g->stream>>>(g->k, g->kn[l], n, HK, D,
                                                             1e-6f, g->norm_p1);
    if (g->rot_dim && g->rot_dim < D && g->pinvf) {
      int half = g->rot_dim / 2;
      k_rope_p<<<(int)(((long)n * H * half + 255) / 256), 256, 0, g->stream>>>(
          g->q, n * H, D, H, pos0, g->pinvf + (size_t)l * half, g->rot_dim);
      k_rope_p<<<(int)(((long)n * HK * half + 255) / 256), 256, 0, g->stream>>>(
          g->k, n * HK, D, HK, pos0, g->pinvf + (size_t)l * half, g->rot_dim);
    } else {
      k_rope<<<(int)(((long)n * H * D / 2 + 255) / 256), 256, 0, g->stream>>>(
          g->q, n * H, D, H, pos0, g->freqs);
      k_rope<<<(int)(((long)n * HK * D / 2 + 255) / 256), 256, 0, g->stream>>>(
          g->k, n * HK, D, HK, pos0, g->freqs);
    }
    CK(cudaGetLastError());
    /* KV scatter [L][ctx][KD] <- packed [n][KD]: rows equally strided on both
     * sides (src pitch = KD packed, dst pitch = KD in ctx), so the whole
     * scatter is ONE 2D memcpy instead of 2n tiny async copies per layer.
     * (8192-token prefill did 393k WDDM submits and took 174s; now ~48.) */
    if (g->kvq.on) {
      /* M2/M6：量化写入。grid = n × 每 token 量化块数 */
      sgl_kvq *q = &g->kvq;
      int bpb = 2 * (q->nvals / SGL_KVQ_BLK);
      int ll = g->kvo[l];
      k_kvq_write<<<n * bpb, SGL_KVQ_BLK, 0, g->stream>>>(
          q->d_pool + (long)ll * q->layer_slots, q->d_pt, g->k, g->v, n, pos0, kv_base,
          q->HK, q->D, q->P, q->slot_layer, q->layer_slots, q->tok_stride, NULL, NULL);
      CK(cudaGetLastError());
    } else {
      float *kcl = g->kc + (long)g->kvo[l] * g->ctx * KD;
      float *vcl = g->vc + (long)g->kvo[l] * g->ctx * KD;
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
      int qstride = g->QD > 0 ? g->QD : E; /* t11：显式 head_dim 时每行 QD（非 E）*/
      if (g->kvq.on) {
        /* M2：分页 + 4-bit 反量化 attention。D%32==0 时走 flash 版（单遍、
         * 向量化读取），否则退回两遍的通用版。*/
        sgl_kvq *q = &g->kvq;
        int ll = g->kvo[l];
        if (prof && g->L <= SGL_PROF_MAXL) cudaEventRecord(g->pf0[l], g->stream);
        if (D % 32 == 0 && D / 32 <= 16) {
          int nw = n * H;
          int nb = (nw + KVQ_FLASH_WARPS - 1) / KVQ_FLASH_WARPS;
          k_attn_q_flash<<<nb, 32 * KVQ_FLASH_WARPS, 0, g->stream>>>(
              g->q, q->d_pool + (long)ll * q->layer_slots, q->d_pt, g->attn, pos0, kv_base,
              n, H, q->HK, q->D, sc, g->sliding_window, q->P, q->nvals, q->slot_layer,
              q->tok_stride);
        } else {
          k_attn_q<<<n * H, 256, 0, g->stream>>>(
              g->q, q->d_pool + (long)ll * q->layer_slots, q->d_pt, g->attn, pos0, kv_base,
              n, H, q->HK, q->D, sc, g->sliding_window, q->P, q->nvals, q->slot_layer,
              q->tok_stride);
        }
        CK(cudaGetLastError());
        if (prof && g->L <= SGL_PROF_MAXL) cudaEventRecord(g->pf1[l], g->stream);
      } else if (prof && g->L <= SGL_PROF_MAXL) {
        cudaEventRecord(g->pf0[l], g->stream);
        attn_dispatch(g, l, n, pos0, kv_base, sc, two_pass, force_long, qstride);
        cudaEventRecord(g->pf1[l], g->stream);
      } else {
        attn_dispatch(g, l, n, pos0, kv_base, sc, two_pass, force_long, qstride);
      }
    }
    CK(cudaGetLastError());
    if (g->q_gate) { /* t11 G13(a)：attn 输出乘 sigmoid(gate) 后再 o_proj */
      k_qgate_apply<<<(int)(((long)n * g->QD + 255) / 256), 256, 0, g->stream>>>(
          g->attn, g->qgate, (long)n * g->QD);
      CK(cudaGetLastError());
    }
    PROJ(g, g->attn, g->ow[l], g->ow_h ? g->ow_h[l] : NULL, g->ow8 ? &g->ow8[l] : NULL, NULL,
         g->h, n, E, g->QD > 0 ? g->QD : E);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)n * E + 255) / 256), 256, 0, g->stream>>>(g->x, g->h, (long)n * E);
    if (g->norm_p1)
      k_rmsnorm_p1<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->ln2[l], g->h, n, E,
                                                               g->eps);
    else
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
  if (g->norm_p1)
    k_rmsnorm_p1<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->final_norm, g->h, n, E,
                                                             g->eps);
  else
    k_rmsnorm<<<n, 256, 256 * sizeof(float), g->stream>>>(g->x, g->final_norm, g->h, n, E,
                                                          g->eps);
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
      lm_head_gemm(g, hbase, g->d_lgb, rows);
      CK(cudaGetLastError());
      CK(cudaMemcpyAsync(g->h_lgb, g->d_lgb, sizeof(float) * (size_t)rows * g->V,
                         cudaMemcpyDeviceToHost, g->stream));
      CK(cudaStreamSynchronize(g->stream));
      memcpy(logits_out, g->h_lgb, sizeof(float) * (size_t)rows * g->V);
    }
  }
  if (prof) {
    CK(cudaStreamSynchronize(g->stream));
    /* t11：要容错——没被记录的事件 cudaEventElapsedTime 会返回
     * cudaErrorInvalidResourceHandle，而这个错误会粘在 last error 上，
     * 下一块 forward 的 CK(cudaGetLastError()) 会把它报成 “invalid resource
     * handle”，整块 prefill 直接失败（这就是 chunk 2 必崩的原因）。*/
    double total_ms = (sgl_hnow() - t_fwd0) * 1000.0;
    float attn_ms = 0.0f, lin_ms = 0.0f;
    for (int l = 0; l < g->L; l++) {
      float ms = 0.0f;
      if (cudaEventElapsedTime(&ms, g->pf0[l], g->pf1[l]) != cudaSuccess) {
        cudaGetLastError(); /* 吞掉：未记录事件的错误不能污染后续调用 */
        continue;
      }
      if (g->lkind && g->lkind[l] == 2) lin_ms += ms;
      else attn_ms += ms;
    }
    printf("[prof] n=%d pos0=%d  total %.1f ms | attn(full) %.1f ms | lin-recur %.1f ms | rest %.1f ms\n",
           n, pos0, total_ms, attn_ms, lin_ms,
           total_ms - attn_ms - lin_ms);
    extern long g_imma_ok, g_imma_fail, g_imma_skip, g_fallback_mn;
    extern int g_imma_status;
    printf("[prof2] imma ok=%ld fail=%ld skip=%ld fallback_mn=%ld cublas_status=%d\n",
           g_imma_ok, g_imma_fail, g_imma_skip, g_fallback_mn, g_imma_status);
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
 * (single M=n lm_head GEMM). See cuda.h for row semantics.
 * M3：logits_out 允许为 NULL = 只算到 hidden（每行的 x/h 仍然写满），
 * 给 MTP 预热/取全位置 hidden 用——那时不需要 lm_head，省一次 248320 的 GEMM。*/
int sgl_cuda_forward_seg_rows(sgl_cuda_model *g, const int *tokens, int n,
                              int pos0, int kv_base, float *logits_out) {
  if (n <= 0) return -1;
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
/* ---- t11: 每请求线性注意力状态槽 ---- */
void sgl_cuda_set_lin_slot(sgl_cuda_model *m, int slot) {
  if (!m || !m->h_lin_slot) return;
  int ns = m->lin_slots;
  if (ns > 0) slot = ((slot % ns) + ns) % ns;
  else slot = 0;
  *m->h_lin_slot = slot;
}

void sgl_cuda_set_decode_slots(sgl_cuda_model *m, const int *slots, int B) {
  if (!m || !m->h_dec_slots) return;
  int n = B < 32 ? B : 32;
  for (int i = 0; i < n; i++) {
    int sl = slots ? slots[i] : 0;
    int ns = m->lin_slots;
    if (ns > 0) sl = ((sl % ns) + ns) % ns;
    else sl = 0;
    m->h_dec_slots[i] = sl;
  }
}

int sgl_cuda_lin_slots(const sgl_cuda_model *m) { return m ? m->lin_slots : 0; }

static int decode_step_launch(sgl_cuda_model *g, int B, int maxu) {
  int E = g->E, H = g->H, HK = g->HK, D = g->D, KD = g->KD;
  cudaStream_t s = g->stream;

  /* B==1：hybrid（lin 层/q_gate）走 fwd_seg_llama（n=1）—— 数值基准路径，不动。
   * B>1：走下面的批量路径（M0），每层一次 launch 处理 B 行；原先这里是
   * for(b) 串行跑 B 遍 64 层，实测聚合吞吐与并发无关（恒 36.8 tok/s）。*/
  if ((g->lin_n > 0 || g->q_gate) && B == 1) {
    for (int b = 0; b < B; b++) {
      /* fwd_seg_llama 约定：KV 写 (kv_base + pos0)、rope 用 pos0 ——
       * 所以 pos0 是【序列内相对位置】= h_lens[b]，kv_base 是绝对页。
       * 把 base 再加进 pos0 会让 KV 偏移变成 3×base → 越界（prewarm 就是这样炸的）。*/
      int pos0 = g->h_lens[b];
      /* t11：每行必须用**自己的**线性注意力递归状态槽。之前所有行都读同一个
       * *h_lin_slot(=0) → 并发时 4 个序列的 state 互相覆盖，第一个 token 对、
       * 后续全乱（“7+3” 答 “1”）。槽由 HTTP 层按段号传入。*/
      if (g->h_lin_slot) {
        int sl = g->h_dec_slots ? g->h_dec_slots[b] : 0;
        if (g->lin_slots > 0) sl = ((sl % g->lin_slots) + g->lin_slots) % g->lin_slots;
        else sl = 0;
        *g->h_lin_slot = sl;
      }
      {
        static int dbgd = -1;
        if (dbgd < 0) { const char *e = getenv("SGL_CUDA_DBG"); dbgd = e ? atoi(e) : 0; }
        if (dbgd > 0)
          fprintf(stderr, "[gpu dec] b=%d tok=%d base=%d len=%d pos0=%d slot=%d\n", b,
                  g->h_tok[b], g->h_base[b], g->h_lens[b], pos0,
                  g->h_lin_slot ? (int)*g->h_lin_slot : -1);
      }
      /* t11：fwd_seg_llama 内部会先把 logits D2H 到 g->h_lgb[0..V)，然后再
       * memcpy 到 logits_out。之前直接传 h_lgb + b*V → 第 b 行写入前就把
       * 第 0 行覆盖成自己的结果，且 B 行结果根本沒回到调用方的 batch_logits。
       * 现在每行先落到独立 scratch h_log1，再搬到 h_lgb 的第 b 行。*/
      float *lout = g->h_log1 ? g->h_log1 : g->h_lgb;
      if (fwd_seg_llama(g, &g->h_tok[b], NULL, 1, pos0, g->h_base[b], lout, 0) != 0)
        return -1;
      if (g->h_log1 && g->h_lgb)
        memcpy(g->h_lgb + (size_t)b * g->V, g->h_log1, sizeof(float) * (size_t)g->V);
    }
    CK(cudaStreamSynchronize(s));
    /* 行结果已在 h_lgb[b]；拷贝到调用方的 logits_out 由 decode_batch 统一做。*/
    return 0;
  }

  /* M0: batch decode 的每行 slot 送进 device —— batch GDN kernel 内按 b 取，
   * host 不再参与（这正是原先无法批量、也无法进 graph 的原因）。*/
  if (g->lin_n > 0 && g->d_lin_slot && g->h_dec_slots)
    CK(cudaMemcpyAsync(g->d_lin_slot, g->h_dec_slots, sizeof(int) * (size_t)B,
                       cudaMemcpyHostToDevice, s));
  if (g->use_fp16)
    k_embed_h<<<(int)((B * (long)E + 255) / 256), 256, 0, s>>>(g->wte_h, g->d_tok, g->x, B, E);
  else
    k_embed<<<(int)((B * (long)E + 255) / 256), 256, 0, s>>>(g->wte, g->d_tok, g->x, B, E);
  CK(cudaGetLastError());

  for (int l = 0; l < g->L; l++) {
    int is_lin = (g->lkind && g->lkind[l] == 2);
    if (g->norm_p1)
      k_rmsnorm_p1<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->ln1[l], g->h, B, E, g->eps);
    else
      k_rmsnorm<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->ln1[l], g->h, B, E, g->eps);
    CK(cudaGetLastError());
    if (is_lin) {
      /* M0: 批量 gated deltanet —— B 行各自用自己的 slot，一次 launch 算完。
       * 单 token 版靠 host 传 slot 偏移，这里改成 kernel 内从 d_lin_slot[b] 取。*/
      int nk2 = g->lin_nk, kd2 = g->lin_kd, nh2 = g->lin_nh, vd2 = g->lin_vd;
      int keyd2 = nk2 * kd2, vald2 = nh2 * vd2;
      int convd2 = 2 * keyd2 + vald2;
      int gw2 = 2 * kd2 + 2 * g->lin_ratio * vd2;
      int rowN2 = 2 * keyd2 + 2 * vald2;
      int K2 = g->lin_conv_k;
      int ntile2 = (vd2 + 3) / 4;
      int ll2 = g->linvo[l];
      long sbase = (long)ll2 * g->lin_slots * vald2 * kd2;
      long sstride = (long)vald2 * kd2;
      long hbase = (long)ll2 * g->lin_slots * convd2 * (K2 - 1);
      long hstride = (long)convd2 * (K2 - 1);
      PROJ(g, g->h, g->qw[l], g->qw_h ? g->qw_h[l] : NULL, g->qw8 ? &g->qw8[l] : NULL,
           NULL, g->l_qkvz_all, B, rowN2, E);
      CK(cudaGetLastError());
      k_lin_ba<<<dim3(2 * nh2, B), 256, 256 * sizeof(float), s>>>(g->h, g->l_ba[l],
                                                                  g->l_ba_all, 2 * nh2, E);
      k_lin_conv_b<<<dim3((convd2 + 127) / 128, B), 128, 0, s>>>(
          g->l_conv[l], g->l_hist, g->l_qkvz_all, g->l_conv_io, nk2, kd2, nh2, vd2,
          g->lin_ratio, K2, g->d_lin_slot, hbase, hstride, rowN2);
      k_lin_l2norm_b<<<dim3(2 * nk2, B), 256, 256 * sizeof(float), s>>>(
          g->l_conv_io, g->l_qn, g->l_kn, nk2, kd2, convd2);
      k_lin_step_b<<<dim3(nh2 * ntile2, B), 128, 0, s>>>(
          g->l_state, g->l_qn, g->l_kn, g->l_conv_io, g->l_qkvz_all,
          g->l_ba_all, g->l_alog[l], g->l_dtb[l], g->l_o_all, nh2, kd2, vd2,
          g->lin_ratio, g->d_lin_slot, sbase, sstride, ntile2, keyd2, convd2);
      k_lin_gnorm_b<<<dim3(nh2, B), 256, 256 * sizeof(float), s>>>(
          g->l_o_all, g->l_qkvz_all, g->l_norm[l], nh2, vd2, kd2, g->lin_ratio,
          gw2, g->eps, rowN2, vald2);
      CK(cudaGetLastError());
      PROJ(g, g->l_o_all, g->ow[l], g->ow_h ? g->ow_h[l] : NULL,
           g->ow8 ? &g->ow8[l] : NULL, NULL, g->h, B, E, vald2);
      CK(cudaGetLastError());
      k_residual<<<(int)(((long)B * E + 255) / 256), 256, 0, s>>>(g->x, g->h, (long)B * E);
      CK(cudaGetLastError());
      /* lin 层的 MLP 与 full 层相同 */
      if (g->norm_p1)
        k_rmsnorm_p1<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->ln2[l], g->h, B, E, g->eps);
      else
        k_rmsnorm<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->ln2[l], g->h, B, E, g->eps);
      PROJ(g, g->h, g->gw[l], g->gw_h ? g->gw_h[l] : NULL, g->gw8 ? &g->gw8[l] : NULL,
           NULL, g->gate, B, g->I, E);
      PROJ(g, g->h, g->uw[l], g->uw_h ? g->uw_h[l] : NULL, g->uw8 ? &g->uw8[l] : NULL,
           NULL, g->up, B, g->I, E);
      CK(cudaGetLastError());
      k_silu_mul<<<(int)(((long)B * g->I + 255) / 256), 256, 0, s>>>(g->gate, g->up,
                                                                    (long)B * g->I);
      PROJ(g, g->gate, g->dw[l], g->dw_h ? g->dw_h[l] : NULL, g->dw8 ? &g->dw8[l] : NULL,
           NULL, g->h, B, E, g->I);
      CK(cudaGetLastError());
      k_residual<<<(int)(((long)B * E + 255) / 256), 256, 0, s>>>(g->x, g->h, (long)B * E);
      CK(cudaGetLastError());
      continue;
    }
    /* full/sliding 层 —— 对齐 fwd_seg_llama 的完整流程：q_gate
     * （attn_output_gate=True）/ qk per-head norm / partial rope（factor 0.25，
     * rot_dim=64）/ o_proj 的 K 是 QD 而非 E。
     * M0 之前这段只服务过非 hybrid 模型；Qwen3.8 走它时上面四项全缺，
     * 实测输出退化成垃圾 token（状态本身对，只是 full 层算错）。*/
    {
      float *qdst = g->q;
      int qN2 = g->QD > 0 ? g->QD : E;
      if (g->q_gate) { qdst = g->q2; qN2 = 2 * qN2; }
      PROJ(g, g->h, g->qw[l], g->qw_h ? g->qw_h[l] : NULL, g->qw8 ? &g->qw8[l] : NULL,
           g->qb[l], qdst, B, qN2, E);
      if (g->q_gate) {
        k_qgate_split<<<(int)(((long)B * H * D + 255) / 256), 256, 0, s>>>(
            g->q2, g->q, g->qgate, B, H, D);
        CK(cudaGetLastError());
      }
    }
    PROJ(g, g->h, g->kw[l], g->kw_h ? g->kw_h[l] : NULL, g->kw8 ? &g->kw8[l] : NULL, g->kb[l],
         g->k, B, KD, E);
    PROJ(g, g->h, g->vw[l], g->vw_h ? g->vw_h[l] : NULL, g->vw8 ? &g->vw8[l] : NULL, g->vb[l],
         g->v, B, KD, E);
    CK(cudaGetLastError());
    if (g->qn && g->qn[l])
      k_hrmsnorm<<<B * H, D, D * sizeof(float), s>>>(g->q, g->qn[l], B, H, D, 1e-6f,
                                                     g->norm_p1);
    if (g->kn && g->kn[l])
      k_hrmsnorm<<<B * HK, D, D * sizeof(float), s>>>(g->k, g->kn[l], B, HK, D, 1e-6f,
                                                      g->norm_p1);
    if (g->rot_dim && g->rot_dim < D && g->pinvf) {
      int hf2 = g->rot_dim / 2;
      k_rope_p_rows<<<(int)(((long)B * H * hf2 + 255) / 256), 256, 0, s>>>(
          g->q, B * H, D, g->d_pq, g->pinvf + (size_t)l * hf2, g->rot_dim);
      k_rope_p_rows<<<(int)(((long)B * HK * hf2 + 255) / 256), 256, 0, s>>>(
          g->k, B * HK, D, g->d_pk, g->pinvf + (size_t)l * hf2, g->rot_dim);
    } else {
      k_rope_rows<<<(int)(((long)B * H * D / 2 + 255) / 256), 256, 0, s>>>(g->q, B * H, D, g->d_pq, g->freqs);
      k_rope_rows<<<(int)(((long)B * HK * D / 2 + 255) / 256), 256, 0, s>>>(g->k, B * HK, D, g->d_pk, g->freqs);
    }
    CK(cudaGetLastError());
    if (g->kvq.on) {
      sgl_kvq *q = &g->kvq;
      int bpb = 2 * (q->nvals / SGL_KVQ_BLK);
      int ll = g->kvo[l];
      k_kvq_write<<<B * bpb, SGL_KVQ_BLK, 0, s>>>(
          q->d_pool + (long)ll * q->layer_slots, q->d_pt, g->k, g->v, B, 0, 0, q->HK, q->D,
          q->P, q->slot_layer, q->layer_slots, q->tok_stride, g->d_base, g->d_lens);
      CK(cudaGetLastError());
      int nw0 = B * H;
      int ns0 = q->nsplit;
      int use_split = 0;
      if (q->d_split && ns0 > 1 && nw0 * ns0 <= q->split_cap) {
        /* 并行度：不切时 nw/4 个 block，切后 nw*ns/4。上下文长（maxu 大）
         * 且 nw < 256 才值得（否则本来就填满了 GPU）。*/
        if (nw0 < 256 && maxu > 1024) use_split = 1;
      }
      if (use_split) {
        int tot = nw0 * ns0;
        { /* 只在开关变化时打一行，不吵 */
          static int last_key = -1;
          int key = use_split * 1000 + nw0;
          if (key != last_key) {
            last_key = key;
            fprintf(stderr, "[kvq] decode split-KV: B=%d H=%d nw=%d nsplit=%d maxu=%d -> %d blocks\n",
                    B, H, nw0, ns0, maxu, (tot + KVQ_FLASH_WARPS - 1) / KVQ_FLASH_WARPS);
          }
        }
        k_attn_q_split_partial<<<(tot + KVQ_FLASH_WARPS - 1) / KVQ_FLASH_WARPS,
                                  32 * KVQ_FLASH_WARPS, 0, s>>>(
            g->q, q->d_pool + (long)ll * q->layer_slots, q->d_pt, g->d_base, g->d_lens, B, H,
            q->HK, q->D, 1.0f / sqrtf((float)D), g->sliding_window, q->P, q->nvals,
            q->slot_layer, q->tok_stride, ns0, q->d_split);
        CK(cudaGetLastError());
        k_attn_q_split_merge<<<(nw0 + KVQ_FLASH_WARPS - 1) / KVQ_FLASH_WARPS,
                                32 * KVQ_FLASH_WARPS, 0, s>>>(
            g->attn, q->d_split, B, H, q->D, ns0);
        CK(cudaGetLastError());
      } else {
        k_attn_q_batch_flash<<<(nw0 + KVQ_FLASH_WARPS - 1) / KVQ_FLASH_WARPS,
                                32 * KVQ_FLASH_WARPS, 0, s>>>(
            g->q, q->d_pool + (long)ll * q->layer_slots, q->d_pt, g->attn, g->d_base, g->d_lens,
            B, H, q->HK, q->D, 1.0f / sqrtf((float)D), g->sliding_window, q->P, q->nvals,
            q->slot_layer, q->tok_stride);
        CK(cudaGetLastError());
      }
    } else if (maxu > SGL_ATTN_SHORT_MAX) {
      k_kv_write<<<(int)((B * HK * D + 255) / 256), 256, 0, s>>>(
          g->kc + (long)g->kvo[l] * g->ctx * KD, g->vc + (long)g->kvo[l] * g->ctx * KD,
          g->k, g->v, g->d_base, g->d_lens, HK, D, B);
      CK(cudaGetLastError());
      k_attn_long_batch<<<B * H, 256, 0, s>>>(
          g->q, g->kc + (long)g->kvo[l] * g->ctx * KD, g->vc + (long)g->kvo[l] * g->ctx * KD,
          g->attn, g->d_base, g->d_lens, B, H, HK, D, 1.0f / sqrtf((float)D),
          g->sliding_window);
    } else {
      k_kv_write<<<(int)((B * HK * D + 255) / 256), 256, 0, s>>>(
          g->kc + (long)g->kvo[l] * g->ctx * KD, g->vc + (long)g->kvo[l] * g->ctx * KD,
          g->k, g->v, g->d_base, g->d_lens, HK, D, B);
      CK(cudaGetLastError());
      k_attn_batch<<<B * H, 256, (size_t)maxu * sizeof(float), s>>>(
          g->q, g->kc + (long)g->kvo[l] * g->ctx * KD, g->vc + (long)g->kvo[l] * g->ctx * KD,
          g->attn, g->d_base, g->d_lens, B, H, HK, D, 1.0f / sqrtf((float)D),
          g->sliding_window);
    }
    CK(cudaGetLastError());
    if (g->q_gate) {
      k_qgate_apply<<<(int)(((long)B * g->QD + 255) / 256), 256, 0, s>>>(
          g->attn, g->qgate, (long)B * g->QD);
      CK(cudaGetLastError());
    }
    PROJ(g, g->attn, g->ow[l], g->ow_h ? g->ow_h[l] : NULL, g->ow8 ? &g->ow8[l] : NULL, NULL,
         g->h, B, E, g->QD > 0 ? g->QD : E);
    CK(cudaGetLastError());
    k_residual<<<(int)(((long)B * E + 255) / 256), 256, 0, s>>>(g->x, g->h, (long)B * E);
    if (g->norm_p1)
      k_rmsnorm_p1<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->ln2[l], g->h, B, E, g->eps);
    else
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
  if (g->norm_p1)
    k_rmsnorm_p1<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->final_norm, g->h, B, E, g->eps);
  else
    k_rmsnorm<<<B, 256, 256 * sizeof(float), s>>>(g->x, g->final_norm, g->h, B, E, g->eps);
  CK(cudaGetLastError());
  /* lm_head as ONE GEMM over all B rows (h rows are contiguous) */
  lm_head_gemm(g, g->h, g->d_lgb, B);
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
  /* t11：hybrid（lin/q_gate）不进 graph——其 decode 是逐行 fwd（含 host 指针），
   * 但 prefill 预热仍然要做（避免首次真实 prompt 付内核 lazy-load 的代价）。*/
  if (!(g->lin_n > 0 || g->q_gate))
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
    if (g->lin_n > 0 || g->q_gate) use_graph = 0; /* t11 v1：hybrid 逐行 fwd，不 capture */
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
  /* t11：只有「hybrid 且 B==1」走 fwd_seg_llama 串行路径时，它内部已把每行
   * logits D2H 到 h_lgb（pinned host），这里再拷会盖成残留旧值。
   * M0：B>1 的批量路径（以及所有非 hybrid 模型）写的是 device 侧 d_lgb，
   * 必须照常 D2H —— 之前无条件跳过，导致并发请求拿到 prewarm 残留的 logits
   * （现象：输出退化成 “Paris Paris Paris…”，但速度指标看起来正常）。*/
  if (!((g->lin_n > 0 || g->q_gate) && B == 1)) {
    CK(cudaMemcpyAsync(g->h_lgb, g->d_lgb, sizeof(float) * (size_t)B * g->V,
                       cudaMemcpyDeviceToHost, g->stream));
    CK(cudaStreamSynchronize(g->stream));
  }
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

