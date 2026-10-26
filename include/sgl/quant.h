/* sgl/quant.h — int8 量化算子（对应 sglang 的 W8A8 / int8 KV 路线的 CPU 版）
 * 方案：
 *   权重：per-channel（输出通道）对称量化 W_i8[n][k] + scale[n]
 *   激活：per-token 对称量化 A_i8[m][k] + scale[m]
 *   y[m][n] = Σ A_i8*W_i8 * sa[m] * sw[n]  （int32 累加）
 *   KV：per-page（一个槽位整段）对称量化 */
#ifndef SGL_QUANT_H
#define SGL_QUANT_H

#include <stdint.h>

/* f32 行 [n] → int8 对称量化；返回 scale（= max|abs|/127）。scale=NULL 时只返回 */
float sgl_quant_row_i8(const float *x, int8_t *out, int n);

/* f32 矩阵 [rows][dim] 逐行量化；scales[rows] */
void sgl_quant_rows_i8(const float *x, int8_t *out, float *scales, int rows, int dim);

/* int8 行 → f32（乘 scale）*/
void sgl_dequant_row_i8(const int8_t *q, float scale, float *out, int n);

/* y[m][n] = Σ_k A_f32[m,k] * (W_i8[n,k]*sw[n]) + bias[n]
 * W8A16：激活保 f32（LLM 激活有重尾 outlier，逐 token 量化会崩精度）；
 * 权重 int8 反量化累加——decode 的 GEMV 受权重带宽限制，体积减半直接提速。 */
void sgl_matmul_w8a16(const float *A, const int8_t *W, const float *sw,
                      const float *bias, float *C, int M, int N, int K);

#endif
