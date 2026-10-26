/* sgl/tensor.h — 张量（PyTorch torch.Tensor 的最小 C 子集：1D/2D/3D f32） */
#ifndef SGL_TENSOR_H
#define SGL_TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
  SGL_CPU = 0,
  SGL_CUDA = 1, /* M10 起用 */
} sgl_device;

typedef struct {
  int ndim;             /* 0..4 */
  int64_t shape[4];     /* 各维大小 */
  int64_t strides[4];   /* 元素步长（当前只支持紧凑行主序） */
  sgl_device device;
  float *data;          /* 指向 buffer 或 mmap 只读区 */
  void *_owner;         /* 非 NULL 表示 data 需要释放 */
} sgl_tensor;

sgl_tensor sgl_tensor_null(void);
int sgl_tensor_is_null(sgl_tensor t);

/* 新建（分配内存、清零可选）*/
sgl_tensor sgl_tensor_zeros(int ndim, const int64_t *shape);
sgl_tensor sgl_tensor_empty(int ndim, const int64_t *shape);

/* 包装外部 buffer（如 mmap 里的权重，不持有所有权）*/
sgl_tensor sgl_tensor_from_data(int ndim, const int64_t *shape, float *data);

/* 从其他 buffer 复制 */
sgl_tensor sgl_tensor_copy_from(int ndim, const int64_t *shape, const float *src);

int64_t sgl_tensor_numel(const sgl_tensor *t);
void sgl_tensor_free(sgl_tensor *t);

/* element-wise 归约 */
float sgl_tensor_sum(const sgl_tensor *t);
float sgl_tensor_max_abs(const sgl_tensor *t);

#endif
