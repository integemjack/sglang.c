/* sgl/alloc.h — 内存：对齐分配、内存池、跨平台 mmap（权重零拷贝加载用）
 * 对应 sglang 依赖的 PyTorch CUDA caching allocator / pinned memory 的 CPU 侧替代。 */
#ifndef SGL_ALLOC_H
#define SGL_ALLOC_H

#include <stddef.h>
#include <stdint.h>

/* 64 字节对齐分配（缓存行），失败时 abort（推理引擎启动期分配，不可恢复） */
void *sgl_xmalloc(size_t size);
void *sgl_xcalloc(size_t count, size_t size);
void *sgl_xrealloc(void *ptr, size_t size);
void sgl_xfree(void *ptr);

/* ---- 只读文件映射（safetensors 权重零拷贝） ---- */
typedef struct {
  void *data;      /* 映射起始地址 */
  size_t size;     /* 文件字节数 */
  void *_handle;   /* 平台句柄（Windows: HANDLE） */
} sgl_mmap;

/* 只读映射整个文件；失败返回 -1 */
int sgl_mmap_open(sgl_mmap *m, const char *path);
/* 按字节偏移取只读指针 */
const void *sgl_mmap_at(const sgl_mmap *m, size_t offset, size_t len);
void sgl_mmap_close(sgl_mmap *m);

#endif
