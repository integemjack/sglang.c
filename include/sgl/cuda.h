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

/* t10 llava: 预融合视觉 embedding [n][E]（CLIP+projector 输出）直接进层循环，
 * 跳过 token 查表；与 token 前向同一路径。仅 llama 后端（gpt2 返回 -1）。 */
int sgl_cuda_forward_seg_emb(sgl_cuda_model *g, const float *emb, int n, int pos0,
                             int kv_base, float *logits_out);

void sgl_cuda_free(sgl_cuda_model *m);

/* tokens[n] 位于绝对位置 [pos0, pos0+n)；logits_out[vocab]（主机内存）为
 * 最后一个位置的分布。v1：KV 连续，pos0+n ≤ n_ctx。成功 0。 */
int sgl_cuda_forward(sgl_cuda_model *m, const int *tokens, int n, int pos0,
                     float *logits_out);
/* same, with explicit KV storage base (kv_base+p addresses storage while
 * causal positions stay 0-based relative: enables per-sequence segments) */
int sgl_cuda_forward_seg(sgl_cuda_model *m, const int *tokens, int n, int pos0,
                         int kv_base, float *logits_out);

/* G7: multi-row logits for speculative-decoding verify. Same forward as
 * sgl_cuda_forward_seg, but logits_out receives ALL n rows: row i is the
 * distribution after consuming tokens[0..i] (prediction for position
 * pos0+i+1), V floats each, host memory [n][V]. lm_head runs as ONE M=n
 * GEMM (not n M=1 calls). llama backend only (gpt2 returns -1). */
int sgl_cuda_forward_seg_rows(sgl_cuda_model *m, const int *tokens, int n,
                              int pos0, int kv_base, float *logits_out);

/* model geometry accessors (opaque struct; ctx = context length on GPU) */
int sgl_cuda_ctx(const sgl_cuda_model *m);
/* G7: vocab size on GPU (logits row length; spec verify buffers need it) */
int sgl_cuda_vocab(const sgl_cuda_model *m);

/* batch decode: B sequences x 1 token; sequence i uses KV segment at absolute
 * position base_pos[i] + p (caller keeps segments non-overlapping in m->ctx;
 * e.g. base = i * seg_len). lens[i] = current length; token writes at lens[i].
 * logits_out needs B*vocab floats. */
int sgl_cuda_decode_batch(sgl_cuda_model *m, const int *tokens, const int *base_pos,
                          const int *lens, int B, float *logits_out);

/* G3: 预热 + 预捕获。把激活工作区一次性撑到 rows（服务端 = 最大 prefill 块），
 * batch 镜像撑到 max_b，然后为 B=1..max_b 各捕获一张 decode graph。返回捕获成功
 * 的张数，失败 -1。服务端在起服务前调用：之后 ws_cap 不再增长，缓存的 graph 全程
 * 有效（增长会 invalidate 全部 graph 并在下一次 decode 重新捕获，~0.5s/张）。 */
int sgl_cuda_prewarm(sgl_cuda_model *m, int rows, int max_b);

/* G6: 前缀缓存的 KV 搬运：把 [src,src+len) 行（所有层 k+v）拷到 [dst,dst+len)。
 * 只有两边都从位置 0 开始（同一段内）才能复用——RoPE 把位置写进了 K。
 * 成功 0；越界/拷贝失败 -1；src==dst 或 len<=0 直接算成功。 */
int sgl_cuda_copy_kv(sgl_cuda_model *m, int src, int dst, int len);

/* G5: 上传前的显存估算（MB）。按与 sgl_cuda_from_llama 相同的分配口径算：
 * 权重（fp32/fp16/int8 按当前环境变量开关）+ KV 池(ctx) + 激活工作区(rows) +
 * logits 设备镜像 + cublas workspace。llama_model 为 sgl_llama*。 */
long sgl_cuda_estimate_vram_mb(const void *llama_model, int ctx, int rows);
/* G5: 当前可用显存（cudaMemGetInfo 的 free，MB）；失败返回 -1 */
long sgl_cuda_avail_vram_mb(void);

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
