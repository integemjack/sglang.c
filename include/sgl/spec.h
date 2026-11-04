/* sgl/spec.h — speculative decoding（greedy 与 temperature>0 修正采样两版）
 * 对应 sglang srt/speculative/ 的最小正确版。 */
#ifndef SGL_SPEC_H
#define SGL_SPEC_H

#include "sgl/llama.h"
#include "sgl/sampling.h"

/* target/dDraft 需同 vocab（同 tokenizer 模型）。greedy 下输出与 target-only
 * 逐 token 全等。返回生成数；*out_ids（sgl_xfree）不含 eos。失败 -1。 */
int sgl_spec_generate(sgl_llama *target, sgl_llama *draft,
                      const int *prompt, int n_prompt, int max_new,
                      int k_spec, int eos_id, int **out_ids);

/* temperature>0 的修正采样（speculative sampling）：draft 从 q̃ 抽候选，
 * target 逐位验 u<min(1, p̃(x)/q̃(x))，拒绝处从 norm((p̃-q̃)₊) 重采样。
 * 数学保证：每个产出 token 的边缘分布 == 直接从 target 同参数抽样（与
 * draft 好坏无关，draft 只影响接受率/速度）——验收看分布对拍（test_spec
 * 的 TV 自校准检验），不是逐 token 相等。temperature/top_k/top_p/min_p/
 * penalty 均支持（penalty 的 history 与非投机路径一致）；stop_ids 由服务层
 * 检查，这里只处理 eos（同 greedy 版）。rng 由调用方初始化（如 sp->seed）。
 * sp==NULL 或 temperature<=0 时退化为 sgl_spec_generate（greedy）。
 * out_drafted/out_accepted 报统计（退化时为 -1）。失败 -1。 */
int sgl_spec_sample_generate(sgl_llama *target, sgl_llama *draft,
                             const int *prompt, int n_prompt, int max_new,
                             int k_spec, int eos_id,
                             const sgl_sampling_params *sp, uint64_t *rng,
                             int **out_ids, int *out_drafted, int *out_accepted);

#endif
