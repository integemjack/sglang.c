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

/* t11: hybrid（linear attention）模型的每请求递归状态槽。
 * 递归状态（state+conv hist）不能像 KV 那样按位置寻址，所以每个并发序列必须
 * 用独立的槽：prefill/decode 前告诉引擎“本序列用哪个槽”（一般 = 段号 seg）。 */
void sgl_cuda_set_lin_slot(sgl_cuda_model *m, int slot);
/* 一次为 batch 的每一行设槽（decode 内部逐行前向时用），B > 32 会被截断 */
void sgl_cuda_set_decode_slots(sgl_cuda_model *m, const int *slots, int B);
int sgl_cuda_lin_slots(const sgl_cuda_model *m);

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

/* ============ M3: MTP（multi-token prediction）草稿头 ============
 * 从同一检查点的 mtp.safetensors 读一个 full-attention 草稿层，量化成 int8。
 * kv_ctx = MTP 自有 KV 的容量（只影响草稿能看多远的历史）。
 * 返回 0 = 就绪；非 0 = 没有 mtp 权重 / 环境不支持（调用方应静默降级）。 */
int sgl_cuda_mtp_load(sgl_cuda_model *m, int kv_ctx);
int sgl_cuda_mtp_ready(const sgl_cuda_model *m);

/* 一步草稿：给定位置 pos 的 hidden（device [E] 单行）与该位置的 token，
 * 预测 pos+1 的 token。hidden 取 final-norm 之前还是之后由 SGL_MTP_HIDDEN
 * 决定（pre 默认 / post）。返回 0 成功。 */
int sgl_cuda_mtp_draft(sgl_cuda_model *m, int token, int pos, const float *hidden,
                       int *out_tok);

/* MTP 自验证：在同一 prompt 上让主模型贪心跑 ngen 个 token 作为参考流，
 * 再 teacher-forced 逐步跑 MTP，统计草稿与参考流的一致率（接受率）。
 * 正确实现约 0.6–0.85，实现错了约 0–0.05——判别极锐利。
 * 结果写进 *acc（接受率）与 *n_ok（命中数）。返回 0 成功。 */
int sgl_cuda_mtp_accept_test(sgl_cuda_model *m, const int *prompt, int np, int ngen,
                             double *acc, int *n_ok);

/* ============ M3: 投机解码（草稿 → 一次验证 → 接受最长前缀）============
 * 只对贪心请求（temperature=0 / top_k=1）开：验证就是「argmax 是否相等」。
 * 验证走 fwd_seg_llama（k_lin_persist_reg、prefill attention），普通 decode 走
 * decode_step_launch（k_lin_step_b、decode attention），归约顺序不同，所以输出只在
 * 近 tie 的 token（off 的 top-1 概率 0.3–0.55）处与不开投机分叉，不保证逐位一致。
 * 采样请求需拒绝采样才能保分布，走原路径。*/

/* 分配逐行 GDN 状态快照（W+1 行），返回 0 成功。只对 lin kernel 的 path=2
 * （k_lin_persist_reg，默认路径）生效；其它路径上会静默关掉投机。*/
int sgl_cuda_spec_begin(sgl_cuda_model *m, int W);

/* 把上一次前向的第 row 行（row<0 = 末行）hidden 存成草稿输入。
 * prefill → 第一次投机之间用一次。*/
int sgl_cuda_mtp_set_hidden(sgl_cuda_model *m, int row);

/* 链式草稿：从位置 pos 开始草拟 W 个 token，写进 out[0..W-1]。
 * 第 i 步用上一步的输出 hidden（MTP 自己的，不是主模型的）——因为主模型对
 * 尚未存在的 token 没有 hidden。返回实际生成的个数（<= W）。*/
int sgl_cuda_mtp_drafts(sgl_cuda_model *m, int t0, int pos, int W, int *out);

/* 一次验证：对 toks[0..n-1]（= [t0, 草稿...]）在 pos0 处做带因果掩码的前向，
 * 每行一份 logits（host，[n][V]），同时开启逐行状态快照。
 * 回写 *k_out = 最长匹配前缀长度（第 j 行的 argmax == toks[j+1] 则 j 计入）。*/
int sgl_cuda_spec_verify(sgl_cuda_model *m, const int *toks, int n, int pos0, int kv_base,
                         float *logits_rows, int *k_out);

/* 提交：把第 slot 槽的 GDN 状态回滚到「处理完 k+1 行」（k 个草稿被接受），
 * 并把「最后落定位置的 hidden」存成下一步草稿的输入。返回 0 成功。*/
int sgl_cuda_spec_commit(sgl_cuda_model *m, int slot, int k);

/* 最后一行的 logits（commit 后预测下一个 token 用）*/
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
