/* sgl/safetensors.h — safetensors 只读加载（mmap 零拷贝，支持 HF 分片）
 * 对应 sglang srt/model_loader/loader.py（其调 safetensors 库）。 */
#ifndef SGL_SAFETENSORS_H
#define SGL_SAFETENSORS_H

#include "sgl/tensor.h"

/* C++ 调用方（cuda_backend.cu / 测试）需要 C 链接：本头文件此前没有这个保护，
 * 所以一旦 .cu 里调用 sgl_st_get 就会得到 mangled 名字、链接期 undefined
 * reference。M3 的 MTP 加载是第一个在 C++ 里用它的地方。*/
#ifdef __cplusplus
extern "C" {
#endif

typedef struct sgl_st_ctx sgl_st_ctx;

/* 打开单个 .safetensors 文件；失败返回 NULL */
sgl_st_ctx *sgl_st_open(const char *path);

/* 打开模型目录：优先 model.safetensors.index.json（多分片），
 * 否则 model.safetensors（兼容 pytorch_model.safetensors / consolidated.safetensors）。
 * 分片模式下 sgl_st_get 按 weight_map 在对应分片里查，不需要把整个模型拼成一个文件。 */
sgl_st_ctx *sgl_st_open_dir(const char *dir);

void sgl_st_close(sgl_st_ctx *c);

/* 按名字取张量（F32 直接零拷贝；F16/BF16 转成 fp32 的 owned 缓冲）
 * 返回的张量不持有内存；找不到/不匹配返回 null tensor。
 * FP8（E4M3/E5M2）加载时解成 f32；同分片里存在 <name>_scale_inv
 *（DeepSeek/Qwen FP8 的 128x128 块状缩放，BF16/F32）时按块乘上。 */
sgl_tensor sgl_st_get(sgl_st_ctx *c, const char *name, int expect_ndim,
                      const int64_t *expect_shape);

/* t11 qwen3_5（VLM 包装检查点）：权重都在 model.language_model.* 下。
 * 设置后 sgl_st_get 按名字找不到时会再试 prefix+name。 */
void sgl_st_set_prefix(sgl_st_ctx *c, const char *pfx);

/* t11：释放张量背后的 owned 转换缓冲（f32），并把张量置空。
 * 流式 int8 加载用：大矩阵取到 f32 → 量化 → 立刻 drop，宿存峰值不随层数增长。
 * 张量不是 owned（f32 零拷贝直指 mmap）时仅置空。 */
void sgl_st_drop(sgl_st_ctx *c, sgl_tensor *t);

/* 列出所有张量名（调试用），返回个数，names 指向内部字符串 */
int sgl_st_names(sgl_st_ctx *c, const char ***names);

/* 已加载的分片数（单文件 = 1） */
int sgl_st_shard_count(const sgl_st_ctx *c);

/* 按名字取**原始字节**（不解码、不转换、零拷贝指向 mmap）。
 *
 * 用途：量化权重。sgl_st_get 只认浮点，而 MLX/mlx-community 的 4-bit 检查点把稠密
 * 矩阵存在 `<base>.weight`(U32，每 uint32 打包 8 个 4bit 值) + `<base>.scales`(F16)
 * + `<base>.biases`(F16) 三件套里；要保住 4-bit 就必须拿到原始 U32，不能让上层
 * 先解成 fp32。scales/biases 是 F16，同样用这个拿（sgl_st_get 会转 fp32，多一次拷贝
 * 且丢掉 fp16 精度）。
 *
 * 参数：dtype 指向内部字符串（"F16"/"U32"/...，不归调用方释放）；dims 至少 4 元素；
 * bytes/nbytes 是 mmap 内的视图，**生命周期跟 ctx 走**。
 * 返回 1 = 找到；0 = 没有（或这个 ctx 是 GGUF 源 —— GGUF 请走 sgl_st_get）。
 * 注：向 sgl_st_get 一样会试 prefix 回退（VLM 包装检查点）。 */
int sgl_st_get_raw(sgl_st_ctx *c, const char *name, const char **dtype, int *ndim,
                   int64_t *dims, const void **bytes, size_t *nbytes);

#ifdef __cplusplus
}
#endif

#endif