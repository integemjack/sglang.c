/* sgl/spec.h — speculative decoding（greedy draft-verify）
 * 对应 sglang srt/speculative/ 的最小正确版。 */
#ifndef SGL_SPEC_H
#define SGL_SPEC_H

#include "sgl/llama.h"

/* target/dDraft 需同 vocab（同 tokenizer 模型）。greedy 下输出与 target-only
 * 逐 token 全等。返回生成数；*out_ids（sgl_xfree）不含 eos。失败 -1。 */
int sgl_spec_generate(sgl_llama *target, sgl_llama *draft,
                      const int *prompt, int n_prompt, int max_new,
                      int k_spec, int eos_id, int **out_ids);

#endif
