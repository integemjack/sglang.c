/* sgl/bpe.h — GPT-2 字节级 BPE tokenizer（加载 HF tokenizer.json）
 * 对应 sglang srt/tokenizer/tokenizer_manager.py 底下的 HF tokenizers。 */
#ifndef SGL_BPE_H
#define SGL_BPE_H

#include <stddef.h>
#include <stdint.h>

typedef struct sgl_tokenizer sgl_tokenizer;
typedef struct sgl_gguf sgl_gguf;

/* 失败返回 NULL（原因打日志）*/
sgl_tokenizer *sgl_tokenizer_load(const char *tokenizer_json_path);
/* 按目录加载（tokenizer.json + 可选 tokenizer_config.json 的 eos/bos 覆盖）*/
sgl_tokenizer *sgl_tokenizer_load_dir(const char *model_dir);
/* 从 GGUF 内嵌元数据构造（单文件 .gguf 场景；sgl_tokenizer_load_dir 会自动回退到它）*/
sgl_tokenizer *sgl_tokenizer_load_gguf(sgl_gguf *gg);
void sgl_tokenizer_free(sgl_tokenizer *t);

int sgl_tokenizer_vocab_size(const sgl_tokenizer *t);
int sgl_tokenizer_eos_id(const sgl_tokenizer *t);
int sgl_tokenizer_bos_id(const sgl_tokenizer *t); /* Llama 风格 <s>；无则 -1 */
int sgl_tokenizer_style(const sgl_tokenizer *t); /* 0=gpt2 byte-level 1=llama SP */

/* 返回 malloc 的 int 数组，*out_n 写个数；调用方 free。
 * 已知简化：非 ASCII 多字节字符按「符号类」预切分（与 \p{L} 语义有差异，
 * 英文输入与 HF 完全一致，中文等可能得到不同但合法的 token 序列）。 */
int *sgl_tokenizer_encode(const sgl_tokenizer *t, const char *text, size_t len,
                          int *out_n);

/* G12: 查一个「字面量 token」的 id（added_tokens 里的特殊 token，如 "<image>"）。
 * 返回 -1 表示不在词表里。注意：它不是 sgl_tokenizer_encode 的逆向——只查整段
 * 与 piece 完全相等的那一个 token。*/
int sgl_tokenizer_piece_id(const sgl_tokenizer *t, const char *piece);

/* 返回 malloc 的 NUL 结尾 UTF-8 字符串；调用方 free。 */
char *sgl_tokenizer_decode(const sgl_tokenizer *t, const int *ids, int n);

/* P1: 取单个 token 的真实 UTF-8 字节（GPT-2 byte-level 反映射 / Llama ▁ 与 <0xXX>），
 * 给 grammar 词表用。写入 buf（容量 cap），返回字节数；不可用（越界/特殊 token）返回 0。*/
int sgl_tokenizer_piece_bytes(const sgl_tokenizer *t, int id, char *buf, int cap);

#endif
