/* sgl/cuda_deepseek.h — DeepSeek 系（MLA + MoE）的 CUDA 前向
 *
 * 与 CPU 侧 sgl/deepseek.h 的数值口径完全一致：吸收式（absorbed）MLA + HF 的
 * MoE 路由公式。fp32 权重、fp32 激活、SGEMM（cublas），所以和 CPU 引擎的差异
 * 只剩累加顺序带来的 f32 舍入 —— 测试就是拿 CPU 当参考比 logits。
 *
 * 为什么要独立一份（而不是塞进 cuda_backend.cu 的 sgl_cuda_model）：
 *   llama 那张结构（H 个 KV head、[ctx][KD] 的 K/V cache、--fp16/--w8 分支、
 *   CUDA Graph）和 MLA 的布局正交：MLA 的 cache 行是 [latent | k_pe]（stride =
 *   kv_lora + qk_rope，与 head 数无关），每头的 q 是 [qk_nope + qk_rope] 而
 *   value 只有 kv_lora 维。混进同一结构只会让两边的分支互相污染。
 *
 * KV cache：本实现用**连续**缓冲 [L][ctx][stride]，不做分页（分页是为了前缀共享
 * 与服务端调度；GPU 侧的页表化与 CUDA Graph 是后续独立的一步）。
 */
#ifndef SGL_CUDA_DEEPSEEK_H
#define SGL_CUDA_DEEPSEEK_H

/* nvcc 把 .cu 当 C++ 编译：这里必须给 C 链接，否则 C 侧测试链不到（与
 * cuda_backend.h 的做法一致）。*/
#ifdef __cplusplus
extern "C" {
#endif

typedef struct sgl_cuda_ds sgl_cuda_ds;

/* 把 CPU 侧已加载的 sgl_deepseek* 上传到 GPU。失败返回 NULL。 */
sgl_cuda_ds *sgl_cuda_ds_from(const void *ds_model);
void sgl_cuda_ds_free(sgl_cuda_ds *g);

int sgl_cuda_ds_vocab(const sgl_cuda_ds *g);
int sgl_cuda_ds_ctx(const sgl_cuda_ds *g);
/* 上次 forward 用掉的设备显存（MB），日志/测试用 */
long sgl_cuda_ds_mem_mb(const sgl_cuda_ds *g);

/* prefill/forward：n 个连续位置 [pos0, pos0+n)，写回 n 行 logits（每行 vocab）。
 * all_rows=0 时只回最后一行（decode 口径）。返回 0 = 成功。 */
int sgl_cuda_ds_forward_n(sgl_cuda_ds *g, const int *tokens, int n, int pos0, float *logits,
                          int all_rows);

/* 单 token 解码（pos = 该 token 的绝对位置）。 */
int sgl_cuda_ds_decode(sgl_cuda_ds *g, int token, int pos, float *logits);

/* 清空 KV（新序列）。 */
void sgl_cuda_ds_reset(sgl_cuda_ds *g);

#ifdef __cplusplus
}
#endif

#endif