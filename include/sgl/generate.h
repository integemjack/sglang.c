/* sgl/generate.h — 生成循环（对应 sglang srt/managers/scheduler.py 的单请求同步路径：
 * prefill 一次前向 + decode 逐 token，含 KV cache 复用）*/
#ifndef SGL_GENERATE_H
#define SGL_GENERATE_H

#include "sgl/bpe.h"
#include "sgl/gpt2.h"
#include "sgl/sampling.h"

/* 每产出一个 token 回调一次：piece 为该 token 解码后的文本片段（可能为空），
 * 返回非 0 则提前终止 */
typedef int (*sgl_gen_callback)(void *user, int token_id, const char *piece);

/* 返回生成的 token 数（不含 prompt）；-1 出错 */
int sgl_generate(sgl_gpt2 *m, const sgl_tokenizer *tok, const int *prompt_ids,
                 int prompt_n, const sgl_sampling_params *sp, int max_new,
                 sgl_gen_callback cb, void *user);

#endif
