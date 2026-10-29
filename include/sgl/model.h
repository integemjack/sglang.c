/* sgl/model.h — 模型后端抽象：统一 gpt2 / llama，供调度器与 HTTP 使用
 * （对应 sglang 的 ModelRunner 对不同 model 类型的分发）。 */
#ifndef SGL_MODEL_H
#define SGL_MODEL_H

#include <stdint.h>

typedef struct sgl_model sgl_model;

/* 按目录自动识别（config.json：architectures/hidden_size→llama，n_embd→gpt2）*/
sgl_model *sgl_model_load(const char *model_dir);
void sgl_model_free(sgl_model *m);

int sgl_model_vocab(const sgl_model *m);
int sgl_model_ctx(const sgl_model *m);

/* G21：文本隐藏维（视觉 emb 行宽也是它）*/
int sgl_model_hidden(const sgl_model *m);

/* 分页 KV 池（两后端语义一致）*/
int sgl_model_pages_alloc(sgl_model *m, int *idx, int n);
void sgl_model_pages_unref(sgl_model *m, const int *idx, int n);
void sgl_model_page_ref(sgl_model *m, int page, int delta);
int sgl_model_nfree(const sgl_model *m);
int sgl_model_pool_pages(const sgl_model *m);
/* 页的当前引用数（radix 驱逐判断用）*/
int sgl_model_page_refs(const sgl_model *m, int page);

/* prefill（tokens[n] 写 [pos0,pos0+n)，页表 idx 长度 ≥ pos0+n）*/
int sgl_model_prefill(sgl_model *m, const int *tokens, int n, int pos0,
                      const int *idx, float *logits_out);

/* 批量 decode：B 序列各 1 token；logits_out 需 B*vocab */
int sgl_model_decode_batch(sgl_model *m, const int *tokens, int *const *idx,
                           const int *seq_lens, int B, float *logits_out);

/* G8 多模态：emb 版 prefill（[n][E_lm] 直接进模型，页表语义同 sgl_model_prefill）。
 * 只有 llama 后端支持；gpt2 返回 -1。 */
int sgl_model_prefill_emb(sgl_model *m, const float *emb, int n, int pos0,
                          const int *idx, float *logits_out);

#endif
