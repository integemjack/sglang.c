/* sgl/mlx.h — MLX 后端（Apple Silicon / Metal）：llama 系模型的设备侧前向
 *
 * 与 include/sgl/cuda.h 同形状、同语义（那份注释是权威说明），差别只在
 * 「显存」→「统一内存」。服务层不直接调这里，而是经 include/sgl/accel.h 的
 * 函数表（apps/mlx_http.c 提供 sgl_accel_mlx()）。
 *
 * T1 范围：llama / qwen2 / qwen3 / mistral / gemma 的 fp16 前向 + 批量 decode +
 * KV 段池 + 前缀缓存搬运。GDN(hybrid) / MLA / MoE / gpt2 / MTP 还没做，
 * 对应入口返回 -1 或 NULL，服务层据此降级。
 *
 * 实现要点（都在 tools/mlx_smoke/probe2.cpp 里实测钉死过）：
 *   - 权重 [out,in] 布局，前向一律 x @ W^T（与 CPU 的 sgl_matmul_nt 同）；
 *   - rope 用手写实现（host 用 double 算 cos/sin 再传设备），因为 mx::fast::rope
 *     的 freqs 参数与本仓库的 layer_invf 语义不等价，且它做不了 attn_factor /
 *     partial rotary；手写版实测与 CPU 公式一致（probe2 3e）；
 *   - q/k/v 排成 [B,H,S,D]（rope 的位置轴是 -2）；
 *   - KV 池按层拆成独立数组：mx::slice_update 产生新数组，整池一个数组会让
 *     每层写入都复制整个池（L 倍放大）。
 */
#ifndef SGL_MLX_H
#define SGL_MLX_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sgl_mlx_model sgl_mlx_model;

/* 告知模型目录：量化检查点（4-bit）的权重必须由 MLX 自己从 safetensors 读
 * （见 mlx_backend.cpp 的 mxw_* 说明），而 sgl_llama 结构里不存目录。
 * 不设也能跑 fp16（走 sgl_st 路径）；服务层在 from_llama 之前调一次。*/
void sgl_mlx_set_model_dir(const char *dir);

/* 从已加载的 CPU 模型（sgl_llama*）构建设备实例（权重转 fp16/fp32 常驻设备）。
 * 失败返回 NULL。ctx 取 SGL_MLX_CTX（默认 = 模型的 n_ctx，上限见 estimate）。 */
sgl_mlx_model *sgl_mlx_from_llama(const void *llama_model);

/* 直接从模型目录建实例，不经过 CPU 侧的权重加载。
 *
 * 用途：mlx-community 的 4-bit / 8-bit 检查点（`<base>.weight` 是 U32 打包 +
 * `.scales`/`.biases`），sgl_llama_load 拿不到这些张量（会返回 NULL）；
 * 但 fp16 检查点也能走这条，两条路展开是同一套加载代码。
 * 几何与 config 由 sgl_llama_load_full(dir, 1) 解析（不加载权重）。 */
sgl_mlx_model *sgl_mlx_from_dir(const char *dir);

void sgl_mlx_free(sgl_mlx_model *m);

/* 几何 */
int sgl_mlx_ctx(const sgl_mlx_model *m);
int sgl_mlx_vocab(const sgl_mlx_model *m);

/* tokens[n] 在因果位置 [pos0, pos0+n)，KV 写到 kv_base+pos0+i 行；
 * logits_out[vocab] = 最后一个位置的分布（NULL = 不要）。成功 0。 */
int sgl_mlx_forward_seg(sgl_mlx_model *m, const int *tokens, int n, int pos0, int kv_base,
                        float *logits_out);

/* 同上，但 logits_out 收全部 n 行（[n][vocab]），投机解码的 verify 用。 */
int sgl_mlx_forward_seg_rows(sgl_mlx_model *m, const int *tokens, int n, int pos0, int kv_base,
                             float *logits_out);

/* 预融合视觉 embedding [n][E] 直接进层循环（多模态）。 */
int sgl_mlx_forward_seg_emb(sgl_mlx_model *m, const float *emb, int n, int pos0, int kv_base,
                            float *logits_out);

/* B 条序列各 1 token：序列 i 的 KV 段基址 base_pos[i]，当前长度 lens[i]，
 * 新 token 写在 base_pos[i]+lens[i] 行、因果位置 lens[i]。logits_out 需 B*vocab。 */
int sgl_mlx_decode_batch(sgl_mlx_model *m, const int *tokens, const int *base_pos,
                         const int *lens, int B, float *logits_out);

/* 前缀缓存：把 [src,src+len) 行（所有层 k+v）拷到 [dst,dst+len)。成功 0。 */
int sgl_mlx_copy_kv(sgl_mlx_model *m, int src, int dst, int len);

/* 预热：把常用形状先跑一遍，让 MLX 把 kernel 编好（mx::compile 的效果）。
 * 返回完成项数，<0 = 失败。 */
int sgl_mlx_prewarm(sgl_mlx_model *m, int rows, int max_b);

/* 建实例前的内存估算（MB）：权重 + KV 池(ctx) + 激活(rows)。<=0 = 估不出来。 */
long sgl_mlx_estimate_mem_mb(const void *llama_model, int ctx, int rows);

/* 当前可用设备内存（MB）。Apple Silicon 是统一内存，取 Metal 的
 * maxRecommendedWorkingSetSize 减去已用；<0 = 问不出来。 */
long sgl_mlx_avail_mem_mb(void);

/* 设备名（日志用），如 "Apple M1"。 */
const char *sgl_mlx_device_name(void);

#ifdef __cplusplus
}
#endif

#endif /* SGL_MLX_H */
