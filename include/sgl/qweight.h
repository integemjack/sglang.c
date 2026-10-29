/* sgl/qweight.h — AWQ / GPTQ 量化权重的解包（load 时反量化成 f32）
 *
 * 对应上游 sglang 的 `srt/model_loader/weight_utils.py` +
 * `srt/layers/quantization/{awq,gptq}.py`（那里调 CUDA kernel，这里在加载期
 * 解成 f32，forward 不需要区分）。
 *
 * 两种格式的张量布局**不同**，都按各自官方实现的实际落盘形状处理：
 *
 *   AWQ（quant_method=awq，version=gemm，AutoAWQ 的 WQLinear_GEMM.from_linear）
 *     qweight  int32 [K, N/pack]   —— 沿**输出维**打包，含 AWQ 的 8 元重排
 *     qzeros   int32 [K/G, N/pack] —— 同上重排；zero_point=false 时不存在
 *     scales   f16   [K/G, N]
 *     G = group_size（-1 = 整行共享一个 scale）
 *     重排表（官方 packing_utils.AWQ_REVERSE_ORDER）：自然列 n 的 4bit 槽位是
 *       rev[n&7]，rev = [0,4,1,5,2,6,3,7]
 *     dequant: w[k][n] = (q - z) * scale[k/G][n]
 *
 *   GPTQ（quant_method=gptq，AutoGPTQ / optimum 的落盘格式）
 *     qweight  int32 [K/pack, N]   —— 沿**输入维**打包
 *     qzeros   int32 [K/G, N/pack] —— 沿输出维打包
 *     scales   f16   [K/G, N]
 *     g_idx    int32 [K]           —— 每行属于哪个 group；desc_act（act-order）
 *                                     时 group 在自然列序里不连续，必须靠它；
 *                                     没有该张量时 group = k/G
 *     dequant: w[k][n] = (q[k][n] - z[g][n]) * scale[g][n]，g = g_idx[k] 或 k/G
 *     注：act-order 只影响"哪些列共享 scale"，**不改变行序**（GPTQModel 的
 *     `_dequantize_from_codes` 直接 scales[g_idx] * (w - zeros[g_idx])，
 *     行序即自然序），所以不需要任何置换。
 *
 * pack = 32/bits（4bit → 每 int32 装 8 个值）。
 */
#ifndef SGL_QWEIGHT_H
#define SGL_QWEIGHT_H

#include <stddef.h> /* size_t：GCC/Linux 不会从别的头文件间接带进来 */
#include <stdint.h>

typedef enum {
  SGL_Q_NONE = 0,
  SGL_Q_AWQ = 1,
  SGL_Q_GPTQ = 2,
  SGL_Q_CT = 3, /* compressed-tensors 的 pack-quantized（W4A16 / W8A16）*/
  SGL_Q_MXFP4 = 4, /* gpt-oss 系列的 mxfp4（e2m1 + e8m0 块缩放）*/
} sgl_qmethod;
typedef struct {
  int method;      /* sgl_qmethod */
  int bits;        /* 4 或 8（AWQ 官方只支持 4）*/
  int group_size;  /* -1 = 每行（整条 K）共享 */
  int zero_point;  /* AWQ: 0/1（config 的 zero_point）；GPTQ: 恒为 1 */
  int desc_act;    /* GPTQ: 是否有 g_idx 的非连续分组 */
  int transposed;  /* AWQ: qweight 是否按 [N/pack, K] 存放（gemv 变体）*/
  int v2;          /* GPTQ: format=gptq_v2（qzeros 存的是 v1 值 - 1，解时 +1）*/
  int packed_dim;  /* CT: 沿哪一维打包（0 或 1；Linear 权重是 1 = 沿 K）*/
  int offset;      /* CT: 无符号码的零点偏移（4bit 是 8，8bit 是 128）*/
  int layout;      /* 0 = 各格式的自然落盘布局；1 = marlin 布局（见下）*/
} sgl_qcfg;

/* layout=1：marlin（vLLM 的 awq_marlin / gptq_marlin 在 process_weights_after_loading
 * 里做出来的布局，kernel 直接吃的那份内存形态）。跟自然落盘的区别：
 *   qweight int32 [K/16, N*16/pack]   16x64 的 tile，tile 内按 kernel 的 word/nibble 序
 *   scales  f16   [K/G, N]            列做过 64 列置换（marlin_permute_scales）
 *   qzeros  int32 [K/G, N/pack]       可选；行内 64 列置换 + 8 元 interleave
 * 注意：这不是 HF 生态的落盘格式（上游没有哪个工具把权重存成 marlin），见
 * docs/GAP_vs_upstream.md 的 A5 与 tests/marlin_layout.py 的说明。*/
#define SGL_QLAYOUT_NATURAL 0
#define SGL_QLAYOUT_MARLIN 1

/* 每 int32 装几个值 */
int sgl_q_pack(int bits);

/* 反量化成一个 [N, K] 的 f32 行主序矩阵（HF 的 weight 布局：行 = out_features）。
 * 入参都是裸缓冲（qweight/qzeros 为 int32、scales 为 f32、g_idx 为 int32 或 NULL）。
 * 形状校验失败/类型不支持时返回 -1 并打印原因；成功返回 0。
 * out 需由调用方按 N*K*sizeof(float) 准备。*/
int sgl_qweight_dequant(const sgl_qcfg *cfg, const int32_t *qweight, int64_t qw_rows,
                        int64_t qw_cols, const int32_t *qzeros, int64_t qz_rows,
                        int64_t qz_cols, const float *scales, int64_t sc_rows, int64_t sc_cols,
                        const int32_t *g_idx, int64_t g_n, int64_t N, int64_t K, float *out);

/* compressed-tensors pack-quantized：直接吃 weight_packed[rows, cols] 的 int32。
 * 官方解包（vllm-project/compressed-tensors
 *   src/compressed_tensors/compressors/pack_quantized/helpers.py::unpack_from_int32）
 * 对 4bit 的做法：把 int32 看成 4 个字节，每字节低半字节是偶数元素、高半字节是奇数元素，
 * 拼成一维后取前 cols 个，再整体减 1<<(bits-1) 得到有符号值。
 * scales 是 [N, K/G]（沿 K 分组，对称量化，无 zero point）。*/
int sgl_qweight_dequant_ct(const sgl_qcfg *cfg, const int32_t *wpacked, int64_t rows,
                           int64_t cols, const float *scales, int64_t sc_rows, int64_t sc_cols,
                           int64_t N, int64_t K, float *out);

/* gpt-oss 的 mxfp4：weight 是 uint8（两个 e2m1 挤一字节，低半字节在前），
 * scales 是 e8m0（uint8，值 = 2^(x-127)），每 32 个元素一个。*/
int sgl_qweight_dequant_mxfp4(const int32_t *cfg_unused, const uint8_t *wpacked, int64_t rows,
                              int64_t cols, const uint8_t *scales, int64_t sc_rows,
                              int64_t sc_cols, int64_t N, int64_t K, float *out);

/* marlin 布局（layout=1）的解包：形状/置换按 cfg 里的 bits/group_size 走。
 * 只支持 4bit（AWQ 官方也只支持 4bit）；scales/qzeros 必须已经是 marlin 置换后的样子。
 * 没有 qzeros 时按 marlin 的约定 w = (q - 8) * scale。*/
int sgl_qweight_dequant_marlin(const sgl_qcfg *cfg, const int32_t *qweight, int64_t qw_rows,
                               int64_t qw_cols, const int32_t *qzeros, int64_t qz_rows,
                               int64_t qz_cols, const float *scales, int64_t sc_rows,
                               int64_t sc_cols, int64_t N, int64_t K, float *out);

/* 从 config.json 的 quantization_config 文本填 cfg（返回 1 = 认出了 AWQ/GPTQ）。*/
int sgl_qcfg_from_json(const char *json, size_t len, sgl_qcfg *cfg);

#endif /* SGL_QWEIGHT_H */