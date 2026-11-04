/* sgl/cuda_spec.h — G7: GPU 侧投机解码（greedy + temperature>0 修正采样）
 * greedy 版与 CPU 版 sgl_spec_generate 逐 token 等价：每步最终 token 都是 target 在该
 * 位置的 argmax ⇒ 输出与同 target 非 spec greedy 全等（fp 噪声级 tie 翻转除外，
 * 属 GPU 批量/单 token kernel 归约顺序差异，非算法差异）。 */
#ifndef SGL_CUDA_SPEC_H
#define SGL_CUDA_SPEC_H

#include "sgl/cuda.h"
#include "sgl/sampling.h"

#ifdef __cplusplus
extern "C" {
#endif

/* target/draft 是两个 GPU 实例（可同一实例：draft==target 用于接受率=100%
 * 的正确性测试），各占自己的 KV 段 [base, base+len)（同实例时两段不得重叠，
 * 且 base+max len 须 ≤ sgl_cuda_ctx）。两模型 vocab 必须相同（token id 直接
 * 可比）。k_spec 1..8。返回生成数；*out_ids（sgl_xfree）不含 eos 之后的 token。
 * out_drafted/out_accepted 报接受率统计（可 NULL）。失败 -1。 */
int sgl_cuda_spec_generate(sgl_cuda_model *target, int t_base,
                           sgl_cuda_model *draft, int d_base,
                           const int *prompt, int n_prompt, int max_new,
                           int k_spec, int eos_id, int **out_ids,
                           int *out_drafted, int *out_accepted);

/* temperature>0 的修正采样（speculative sampling），语义/数学同 CPU 版
 * sgl_spec_sample_generate：draft 从 q̃ 抽、target 逐位 u<min(1,p̃/q̃) 检验，
 * 拒绝处从 norm((p̃-q̃)₊) 重采样 ⇒ 产出分布 == 直接从 target 同参数抽样。
 * 分布对拍（cuda_test 的 TV 自校准检验）是验收依据，不逐 token 相等。
 * temperature<=0 时退化为 sgl_cuda_spec_generate。失败 -1。 */
int sgl_cuda_spec_sample_generate(sgl_cuda_model *target, int t_base,
                                  sgl_cuda_model *draft, int d_base,
                                  const int *prompt, int n_prompt, int max_new,
                                  int k_spec, int eos_id,
                                  const sgl_sampling_params *sp, uint64_t *rng,
                                  int **out_ids, int *out_drafted, int *out_accepted);

#ifdef __cplusplus
}
#endif

#endif
