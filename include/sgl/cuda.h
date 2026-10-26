/* sgl/cuda.h — CUDA 后端（M9 v1）：llama 系模型的 GPU 单流前向
 * 权重在加载时上传设备；KV 连续存储（不分页，v1）；采样在主机。
 * 复用 llama.c 的加载器（sgl_llama*）作为权重源与配置来源。 */
#ifndef SGL_CUDA_H
#define SGL_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sgl_cuda_model sgl_cuda_model;

/* 从已加载的 CPU 模型构建 GPU 实例（拷贝全部权重到设备）；失败 NULL */
sgl_cuda_model *sgl_cuda_from_llama(const void *llama_model /* sgl_llama* */);

/* t4: GPT-2 后端。Conv1D [K,N] 权重在上传时转置为 [N,K] 复用现有 gemm/attention
 * 路径；绝对位置 wpe / 带 bias 的 LayerNorm / GELU 走 gpt2 专用 kernel 组合。
 * sgl_cuda_forward* 内部按 kind 分派。fp16 开关同样生效（转置时一并转 half）。 */
sgl_cuda_model *sgl_cuda_from_gpt2(const void *gpt2_model /* sgl_gpt2* */);

void sgl_cuda_free(sgl_cuda_model *m);

/* tokens[n] 位于绝对位置 [pos0, pos0+n)；logits_out[vocab]（主机内存）为
 * 最后一个位置的分布。v1：KV 连续，pos0+n ≤ n_ctx。成功 0。 */
int sgl_cuda_forward(sgl_cuda_model *m, const int *tokens, int n, int pos0,
                     float *logits_out);
/* same, with explicit KV storage base (kv_base+p addresses storage while
 * causal positions stay 0-based relative: enables per-sequence segments) */
int sgl_cuda_forward_seg(sgl_cuda_model *m, const int *tokens, int n, int pos0,
                         int kv_base, float *logits_out);

/* model geometry accessors (opaque struct; ctx = context length on GPU) */
int sgl_cuda_ctx(const sgl_cuda_model *m);

/* batch decode: B sequences x 1 token; sequence i uses KV segment at absolute
 * position base_pos[i] + p (caller keeps segments non-overlapping in m->ctx;
 * e.g. base = i * seg_len). lens[i] = current length; token writes at lens[i].
 * logits_out needs B*vocab floats. */
int sgl_cuda_decode_batch(sgl_cuda_model *m, const int *tokens, const int *base_pos,
                          const int *lens, int B, float *logits_out);

/* GEMM micro-benchmark (t1): times cublasSgemm vs fp16 cublasGemmEx on real
 * model shapes (qwen05 / tinyllama), M in {1,8,64}; prints ms and ratio.
 * ASCII output; harmless to call any time. */
void sgl_cuda_gemm_bench(void);

/* 自由显存（MB，调试/规划用）*/
long sgl_cuda_free_mem_mb(void);

#ifdef __cplusplus
}
#endif

#endif
