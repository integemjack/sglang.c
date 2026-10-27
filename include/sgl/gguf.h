/* sgl/gguf.h — GGUF（llama.cpp）权重只读加载
 * 对应 sglang 的 GGUF 加载路径（srt/model_loader/gguf_loader.py + gguf 库）
 * 与 llama.cpp 的 ggml 量化格式。
 *
 * 设计：GGUF 不是「另一种后端」，而是权重源。对外只用 HF 张量名提问
 * （sgl_gguf_get_f32("model.layers.0.self_attn.q_proj.weight", ...)），
 * 内部完成 GGUF 名 → HF 名映射 + 量化反解 → f32 行主序缓冲。
 * 于是 model/llama/deepseek 的上层代码一行不用改。
 *
 * 支持的张量类型（ggml_type）：
 *   F32 / F16 / BF16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 /
 *   Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K
 * 明确不支持的（IQ1/IQ2/IQ3/IQ4 系列、TQ1_0/TQ2_0、MXFP4）：打开时给出
 * 可执行的报错，而不是读出乱码。
 */
#ifndef SGL_GGUF_H
#define SGL_GGUF_H

#include <stddef.h>
#include <stdint.h>

typedef struct sgl_gguf sgl_gguf;

/* 打开单个 .gguf；失败返回 NULL */
sgl_gguf *sgl_gguf_open(const char *path);

/* 在目录里找 .gguf（唯一一个；有多个时取 split 的第一个分片或名字最小的）*/
sgl_gguf *sgl_gguf_open_dir(const char *dir);

void sgl_gguf_close(sgl_gguf *g);

/* 按 HF 张量名取张量并反量化成 f32（行主序、紧凑）。
 * 返回 malloc 的缓冲（调用方 sgl_xfree）；shape 可选。
 * 找不到返回 NULL（可选张量很常见，调用方按 NULL 处理即可），
 * 找到但类型不支持时打印错误并返回 NULL。*/
float *sgl_gguf_get_f32(sgl_gguf *g, const char *hf_name, int *out_ndim,
                        int64_t out_shape[4]);

/* 同名张量是否存在（HF 名）*/
int sgl_gguf_has(sgl_gguf *g, const char *hf_name);

/* 查张量类型：返回 ggml_type 编号（-1 = 不存在），type_name 可选输出字符串 */
int sgl_gguf_tensor_type(sgl_gguf *g, const char *hf_name, const char **type_name);

/* 打开的文件路径（测试/日志用）*/
const char *sgl_gguf_path(const sgl_gguf *g);

/* llama.cpp 的 convert 脚本在写 GGUF 时会把 q_proj/k_proj 的行按 head 交错重排
 * （LlamaModel.permute：r=hd*hd_dim+half*hd_dim/2+k ↦ r'=hd*hd_dim+2k+half），
 * 目的是让 ggml 的交错 RoPE 直接可用。本项目的 forward 是 HF 的 rotate_half
 * 配对 (i, i+D/2)，必须逆置换回来。
 * 返回被处理的张量数（0 = 不适用），可设 SGL_GGUF_NO_PERMUTE=1 关掉。*/
int sgl_gguf_unpermute_qk(sgl_gguf *g, const char *hf_name, float *buf, int ndim,
                          const int64_t *shape);

/* ---- 元数据访问（tokenizer / 配置需要读数组）---- */
int sgl_gguf_meta_i32(sgl_gguf *g, const char *key, int def);
float sgl_gguf_meta_f32(sgl_gguf *g, const char *key, float def);
const char *sgl_gguf_meta_str(sgl_gguf *g, const char *key, const char *def);
/* 字符串数组：元素个数 / 第 i 个（返回内部指针，非 NUL 结尾，用 len 截断）*/
int sgl_gguf_meta_str_array_n(sgl_gguf *g, const char *key);
int sgl_gguf_meta_str_array_at(sgl_gguf *g, const char *key, int i, const char **s,
                               size_t *len);
/* int32 数组 */
int sgl_gguf_meta_i32_array_n(sgl_gguf *g, const char *key);
int sgl_gguf_meta_i32_array_at(sgl_gguf *g, const char *key, int i);

/* 列出映射后的 HF 张量名（调试/测试用）*/
int sgl_gguf_names(sgl_gguf *g, const char ***names);

/* 从元数据合成一份 HF 风格 config.json 文本（malloc，调用方 free）。
 * 没有 config.json 的 GGUF 目录靠它给上层提供结构参数。*/
char *sgl_gguf_config_json(sgl_gguf *g, size_t *out_len);

/* 读模型配置文本：优先 <dir>/config.json；没有则找 <dir>/*.gguf 用元数据合成。
 * 返回 malloc 的 NUL 结尾缓冲（调用方 free），失败 NULL。*/
char *sgl_config_read_text(const char *dir, size_t *out_len);

/* 张量类型名（日志用）*/
const char *sgl_gguf_type_name(int ggml_type);

/* GGUF 名 ↔ HF 名（映射成功返回 1，失败返回 0；两个方向都提供，便于测试）*/
int sgl_gguf_name_to_hf(const char *gguf_name, char *out, size_t cap);
int sgl_gguf_name_from_hf(const char *hf_name, char *out, size_t cap);

#endif /* SGL_GGUF_H */