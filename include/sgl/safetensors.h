/* sgl/safetensors.h — safetensors 只读加载（mmap 零拷贝）
 * 对应 sglang srt/model_loader/loader.py（其调 safetensors 库）。 */
#ifndef SGL_SAFETENSORS_H
#define SGL_SAFETENSORS_H

#include "sgl/tensor.h"

typedef struct sgl_st_ctx sgl_st_ctx;

/* 失败返回 NULL */
sgl_st_ctx *sgl_st_open(const char *path);
void sgl_st_close(sgl_st_ctx *c);

/* 按名字取张量（dtype 必须为 F32，shape 校验可选传 NULL）
 * 返回的张量指向 mmap 只读区，不持有内存；找不到/不匹配返回 null tensor */
sgl_tensor sgl_st_get(sgl_st_ctx *c, const char *name, int expect_ndim,
                      const int64_t *expect_shape);

/* 列出所有张量名（调试用），返回个数，names 指向内部字符串 */
int sgl_st_names(sgl_st_ctx *c, const char ***names);

#endif
