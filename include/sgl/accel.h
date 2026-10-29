/* sgl/accel.h — 加速后端抽象（W1）：把「设备侧推理引擎」从 HTTP/调度层解耦
 *
 * 背景：CUDA 后端原先是一条平行栈——apps/cuda_http.c 直接调 29 个 sgl_cuda_* 符号
 * （36 处调用），自带一套 HTTP。要再加 MLX 就得抄第二套 2000 行的服务代码，
 * 以后每加一个端点要改三处。这里把服务层真正用到的入口收成一张函数指针表：
 * CUDA / MLX 各提供一张，apps/accel_http.c 只通过表调用。
 *
 * 约定：
 *   - **必选**入口（见 sgl_accel_api_check）为 NULL = 这张表不合法，起服务前就报错；
 *   - **可选**入口为 NULL = 该特性关闭，服务层必须自己降级（前缀缓存关、投机关、
 *     图像请求拒、hybrid 模型拒），不能崩、也不能静默给出错误输出；
 *   - 表是进程级只读的（静态常量），实例句柄 sgl_accel* 由后端自己定义。
 *
 * 语义与 include/sgl/cuda.h 中同名函数逐条对齐（那份注释是权威说明），
 * 差别只在「显存」→「设备内存」：CUDA 是 VRAM，MLX 是统一内存。
 */
#ifndef SGL_ACCEL_H
#define SGL_ACCEL_H

#ifdef __cplusplus
extern "C" {
#endif

/* 表结构版本：字段增删就 +1，防止半新半旧的后端与服务层错配 */
#define SGL_ACCEL_ABI 2

typedef struct sgl_accel sgl_accel; /* 后端实例（CUDA 侧即 sgl_cuda_model）*/

typedef struct sgl_accel_api {
  const char *name; /* "cuda" / "mlx"，日志与 /health 里用 */
  int abi;          /* 必须 == SGL_ACCEL_ABI */

  /* ---- 生命周期（必选）---- */
  /* 从已加载的 CPU 模型（sgl_llama*）构建设备实例；失败 NULL。
   * 统一内存下没有「上传」阶段，但接口形状保持一致。 */
  sgl_accel *(*from_llama)(const void *llama_model);
  void (*free)(sgl_accel *m);

  /* ---- 几何（必选）---- */
  int (*ctx)(const sgl_accel *m);   /* 设备侧上下文长度（= KV 池总行数）*/
  int (*vocab)(const sgl_accel *m); /* logits 行长；NULL 时服务层用 tokenizer 的 vocab */

  /* ---- 前向（必选）---- */
  /* tokens[n] 在因果位置 [pos0, pos0+n)，KV 写到 kv_base+pos；
   * logits_out[vocab] 只出最后一行（NULL = 不要 logits）。成功 0。 */
  int (*forward_seg)(sgl_accel *m, const int *tokens, int n, int pos0, int kv_base,
                     float *logits_out);
  /* B 条序列各 1 token 的批量 decode：base_pos[i] 是段基址，lens[i] 是当前长度，
   * logits_out 需要 B*vocab。成功 0。 */
  int (*decode_batch)(sgl_accel *m, const int *tokens, const int *base_pos, const int *lens,
                      int B, float *logits_out);

  /* ---- 设备内存预算（必选）---- */
  /* 上传/建实例前的估算（MB），口径由后端自己定；<=0 = 估不出来（跳过预检）。 */
  long (*estimate_mem_mb)(const void *llama_model, int ctx, int rows);
  /* 当前可用设备内存（MB）；<0 = 问不出来。 */
  long (*avail_mem_mb)(void);

  /* ---- 可选：多行 logits（投机解码的 verify 用）---- */
  int (*forward_seg_rows)(sgl_accel *m, const int *tokens, int n, int pos0, int kv_base,
                          float *logits_out);
  /* ---- 可选：预融合视觉 embedding 进层循环（多模态）---- */
  int (*forward_seg_emb)(sgl_accel *m, const float *emb, int n, int pos0, int kv_base,
                         float *logits_out);
  /* ---- 可选：gpt2 后端 ---- */
  sgl_accel *(*from_gpt2)(const void *gpt2_model);

  /* ---- 可选：前缀缓存的 KV 搬运（[src,src+len) → [dst,dst+len)，所有层 k+v）---- */
  int (*copy_kv)(sgl_accel *m, int src, int dst, int len);

  /* ---- 可选：预热（CUDA = 撑工作区 + 捕获 decode graph；MLX = mx::compile）----
   * 返回完成量，<0 = 失败/不支持。 */
  int (*prewarm)(sgl_accel *m, int rows, int max_b);

  /* ---- 可选：hybrid（线性注意力）模型的每请求递归状态槽 ----
   * lin_slots() <= 0 表示这个后端/模型没有递归状态 → 前缀缓存可用。 */
  int (*lin_slots)(const sgl_accel *m);
  void (*set_lin_slot)(sgl_accel *m, int slot);
  void (*set_decode_slots)(sgl_accel *m, const int *slots, int B);

  /* ---- 可选：MTP 草稿头 + 投机解码（NULL = 特性关闭）---- */
  int (*mtp_load)(sgl_accel *m, int kv_ctx);
  int (*mtp_ready)(const sgl_accel *m);
  void (*mtp_set_base)(sgl_accel *m, int base);
  int (*mtp_set_hidden)(sgl_accel *m, int row);
  int (*mtp_drafts)(sgl_accel *m, int t0, int pos, int W, float minp, int *out, float *probs);
  const float *(*mtp_hprev)(const sgl_accel *m);
  const float *(*mtp_hlast)(const sgl_accel *m);
  int (*mtp_fill_kv)(sgl_accel *m, const int *toks, int n, int pos0, const float *hprev);
  int (*spec_begin)(sgl_accel *m, int W);
  int (*spec_verify)(sgl_accel *m, const int *toks, int n, int pos0, int kv_base,
                     float *logits_rows, int *k_out);
  int (*spec_commit)(sgl_accel *m, int slot, int k);
  int (*mtp_accept_test)(sgl_accel *m, const int *prompt, int np, int ngen, double *acc,
                         int *n_ok);

  /* ---- 可选：MLA + MoE 模型（DeepSeek-V2/V3 系）----
   * 与 llama 那张表正交：KV 行是 [latent | k_pe]，每头的 q 是 [qk_nope + qk_rope]。
   * ds_load 吃的是 CPU 侧已加载的 sgl_deepseek*，返回设备实例（不透明）。
   * 服务层有这张表且模型目录是 DeepSeek 时，生成走单请求阻塞路径（不做段租让/批量）。*/
  void *(*ds_load)(const void *ds_cpu_model);
  void (*ds_free)(void *ds);
  int (*ds_vocab)(const void *ds);
  int (*ds_ctx)(const void *ds);
  void (*ds_reset)(void *ds);
  /* n 个连续位置 [pos0,pos0+n)；只回最后一行 logits（vocab 个 float）*/
  int (*ds_forward)(void *ds, const int *tokens, int n, int pos0, float *logits_out);
  /* 单 token 解码（pos = 绝对位置）*/
  int (*ds_decode)(void *ds, int token, int pos, float *logits_out);
} sgl_accel_api;

/* 校验一张表：必选入口齐、abi 对得上返回 0；否则返回 -1 并把原因写进 err（可 NULL）。*/
int sgl_accel_api_check(const sgl_accel_api *api, char *err, int errlen);

/* 各后端的表。没编进这个构建时返回 NULL（调用方据此回落）。*/
const sgl_accel_api *sgl_accel_cuda(void); /* SGL_CUDA=ON */
const sgl_accel_api *sgl_accel_mlx(void);  /* SGL_MLX=ON（Apple Silicon）*/

/* 后端无关的 HTTP 推理服务（原 sgl_cuda_http_serve，实现见 apps/accel_http.c）。
 * host 语义同 http.c：NULL/"127.0.0.1"/"localhost" 只听本机；"0.0.0.0"/"any"/"*" 听全部。
 * 返回 0 正常退出；2 = 模型/分词器加载失败或表不合法。 */
int sgl_accel_http_serve(const sgl_accel_api *api, const char *dir, int port, const char *host);

#ifdef __cplusplus
}
#endif

#endif /* SGL_ACCEL_H */
