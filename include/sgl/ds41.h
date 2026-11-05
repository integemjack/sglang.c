/* ds41.h — DeepSeek-V4.1 (mini 金标准) MLX 前向的 C 接口。
 *
 * 对拍数据由 tools/dsv41_ref/npz_to_st.py 生成：
 *   mini_ref.st     权重 + ids + freqs（mx::load_safetensors 读）
 *   mini_ref_io.bin 期望输出（logits_pre + 每步 logits/tok）
 * sgl_ds41_run 复现金标准循环：prefill(ids) → argmax → 逐步 decode，输出与 bin 对比。
 */
#ifndef SGL_DS41_H
#define SGL_DS41_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 加载 dir 下的 mini_ref.st；成功返回 0 */
int sgl_ds41_load(const char* dir);

/* 清空全部缓存（win/compress/indexer kv_state 等），可重新 run */
void sgl_ds41_reset(void);

/* 完整跑一遍金标准循环。
 * 输入 ids 从 st 里读（B=2,S=12 已固定在 cfg）；n_dec 步 decode。
 * 输出：logits_pre [B*V]、toks [n_dec][B]、logits_dec [n_dec][B*V]（调用方分配）。
 * 返回 0 成功。 */
int sgl_ds41_run(int n_dec, float* logits_pre, uint32_t* toks, float* logits_dec);

#ifdef __cplusplus
}
#endif
#endif /* SGL_DS41_H */
