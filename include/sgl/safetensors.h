/* sgl/safetensors.h — safetensors 只读加载（mmap 零拷贝，支持 HF 分片）
 * 对应 sglang srt/model_loader/loader.py（其调 safetensors 库）。 */
#ifndef SGL_SAFETENSORS_H
#define SGL_SAFETENSORS_H

#include "sgl/tensor.h"

typedef struct sgl_st_ctx sgl_st_ctx;

/* 打开单个 .safetensors 文件；失败返回 NULL */
sgl_st_ctx *sgl_st_open(const char *path);

/* 打开模型目录：优先 model.safetensors.index.json（多分片），
 * 否则 model.safetensors（兼容 pytorch_model.safetensors / consolidated.safetensors）。
 * 分片模式下 sgl_st_get 按 weight_map 在对应分片里查，不需要把整个模型拼成一个文件。 */
sgl_st_ctx *sgl_st_open_dir(const char *dir);

void sgl_st_close(sgl_st_ctx *c);

/* 按名字取张量（F32 直接零拷贝；F16/BF16 转成 fp32 的 owned 缓冲）
 * 返回的张量不持有内存；找不到/不匹配返回 null tensor */
sgl_tensor sgl_st_get(sgl_st_ctx *c, const char *name, int expect_ndim,
                      const int64_t *expect_shape);

/* 列出所有张量名（调试用），返回个数，names 指向内部字符串 */
int sgl_st_names(sgl_st_ctx *c, const char ***names);

/* 已加载的分片数（单文件 = 1） */
int sgl_st_shard_count(const sgl_st_ctx *c);

#endif