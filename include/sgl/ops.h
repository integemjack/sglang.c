/* sgl/ops.h — CPU 张量算子（f32 行主序）
 * 对应 sglang srt/layers/{linear,layernorm,activation}.py 调用的 torch 算子子集。 */
#ifndef SGL_OPS_H
#define SGL_OPS_H

/* C[M,N] = A[M,K] @ B[K,N] + bias[N]（bias 可 NULL）
 * 分块 + OpenMP 并行；A、B、C 均行主序。 */
void sgl_matmul(const float *A, const float *B, const float *bias, float *C,
                int M, int N, int K);

/* 同 sgl_matmul_nt，但 W 为 PyTorch Linear 布局 [N,K]（out×in，行主序）；
 * HF 的 llama/qwen 等投影权重都是这种（gpt2 的 Conv1D 才是 [K,N]）。 */
void sgl_matmul_nt(const float *A, const float *W, const float *bias, float *C,
                   int M, int N, int K);

/* SIMD 点积（SSE2 4-wide；供 attention 的 D 维内积用）*/
void sgl_dot_f32(const float *a, const float *b, int n, float *out_sum);

/* y[M,N] = layernorm(x[M,N]) * gamma[N] + beta[N] */
void sgl_layernorm(const float *x, const float *gamma, const float *beta,
                   float *y, int M, int N, float eps);

/* 行内 softmax（数值稳定）*/
void sgl_softmax_inplace(float *row, int n);

/* GELU（GPT-2 使用的 tanh 近似，即 HF 的 gelu_new）*/
void sgl_gelu_inplace(float *row, int n);

/* SiLU（Llama 激活）：x * sigmoid(x) */
void sgl_silu_inplace(float *row, int n);

/* RMSNorm（Llama）：y = x / sqrt(mean(x²)+eps) * weight（无 bias/mean）*/
void sgl_rmsnorm(const float *x, const float *weight, float *y, int M, int N,
                 float eps);

/* RoPE（Llama 推理）：对 q/k 的每个头（head_dim=D）按位置 pos 旋转。
 * 旋转式（HF llama 标准式）：对 (2i, 2i+1) 对应用
 *   out[2i]   = in[2i] * cos(θ) - in[2i+1] * sin(θ)
 *   out[2i+1] = in[2i] * sin(θ) + in[2i+1] * cos(θ)
 * θ_p,i = pos * theta^(-2i/D)，theta 即 rope_theta（10000）。
 * qk[n, D] 行主序，in-place 安全。 */
void sgl_rope_inplace(float *qk, int n, int head_dim, int pos0, double theta);

/* embedding 查表：out[n,d] = table[ids[i], :] */
void sgl_embedding_get(const float *table, const int *ids, int n, int d,
                       float *out);

#endif
