/* sgl mlx_http.c — MLX 后端薄壳（T1）
 *
 * 与 apps/cuda_http.c 完全对称，只做两件事：
 *   1) 把 include/sgl/mlx.h 的入口装进 sgl_accel_api 函数表（sgl_accel_mlx()）；
 *   2) 提供 main 与 sgl_mlx_http_serve()（sglc --device mlx 走它）。
 * 服务逻辑全在 apps/accel_http.c（后端无关），这里不含任何逻辑。
 *
 * T1 还没做的入口一律留 NULL —— 按 accel.h 的约定，服务层会据此降级：
 *   from_gpt2（T2）、lin_slots / set_*（hybrid 模型 T3，NULL 表示无递归状态，
 *   前缀缓存因此可用）、mtp_ 与 spec_ 系列（T3）、ds_ 系列（MLA+MoE，T3）。
 *
 * 编宏 SGL_EMBED_MLX_HTTP：sglc 把本文件与 accel_http.c 一起编进去时定义，
 * 此时不出 main（sglc 自己有）。 */
#include <stdio.h>
#include <stdlib.h>

#include "sgl/accel.h"
#include "sgl/mlx.h"

#define MX(m) ((sgl_mlx_model *)(m))
#define MXC(m) ((const sgl_mlx_model *)(m))
#define AC(p) ((sgl_accel *)(p))

static sgl_accel *mx_from_llama(const void *m) { return AC(sgl_mlx_from_llama(m)); }
static void mx_free(sgl_accel *m) { sgl_mlx_free(MX(m)); }
static int mx_ctx(const sgl_accel *m) { return sgl_mlx_ctx(MXC(m)); }
static int mx_vocab(const sgl_accel *m) { return sgl_mlx_vocab(MXC(m)); }

static int mx_forward_seg(sgl_accel *m, const int *t, int n, int p0, int kb, float *lg) {
  return sgl_mlx_forward_seg(MX(m), t, n, p0, kb, lg);
}
static int mx_forward_seg_rows(sgl_accel *m, const int *t, int n, int p0, int kb, float *lg) {
  return sgl_mlx_forward_seg_rows(MX(m), t, n, p0, kb, lg);
}
static int mx_forward_seg_emb(sgl_accel *m, const float *e, int n, int p0, int kb, float *lg) {
  return sgl_mlx_forward_seg_emb(MX(m), e, n, p0, kb, lg);
}
static int mx_decode_batch(sgl_accel *m, const int *t, const int *bp, const int *l, int B,
                           float *lg) {
  return sgl_mlx_decode_batch(MX(m), t, bp, l, B, lg);
}
static int mx_copy_kv(sgl_accel *m, int s, int d, int n) { return sgl_mlx_copy_kv(MX(m), s, d, n); }
static int mx_prewarm(sgl_accel *m, int rows, int max_b) {
  return sgl_mlx_prewarm(MX(m), rows, max_b);
}
/* 「设备内存」在 Apple Silicon 上是统一内存 */
static long mx_estimate_mem_mb(const void *m, int ctx, int rows) {
  return sgl_mlx_estimate_mem_mb(m, ctx, rows);
}
static long mx_avail_mem_mb(void) { return sgl_mlx_avail_mem_mb(); }

/* 用指定初始化器：字段增删/换序都不会静默错位；没写的字段自动是 NULL */
static const sgl_accel_api MLX_API = {
    .name = "mlx",
    .abi = SGL_ACCEL_ABI,
    .from_llama = mx_from_llama,
    .free = mx_free,
    .ctx = mx_ctx,
    .vocab = mx_vocab,
    .forward_seg = mx_forward_seg,
    .decode_batch = mx_decode_batch,
    .estimate_mem_mb = mx_estimate_mem_mb,
    .avail_mem_mb = mx_avail_mem_mb,
    .forward_seg_rows = mx_forward_seg_rows,
    .forward_seg_emb = mx_forward_seg_emb,
    .copy_kv = mx_copy_kv,
    .prewarm = mx_prewarm,
};

const sgl_accel_api *sgl_accel_mlx(void) { return &MLX_API; }

/* 兼容入口：sglc --device mlx 与独立 mlx_http 都走这里 */
int sgl_mlx_http_serve(const char *dir, int port, const char *host) {
  /* 先告知目录：4-bit 检查点在 CPU 侧是 config-only（权重全 NULL），
   * MLX 必须自己从 safetensors 读，而 sgl_llama 不存目录。*/
  sgl_mlx_set_model_dir(dir);
  return sgl_accel_http_serve(&MLX_API, dir, port, host);
}

#ifndef SGL_EMBED_MLX_HTTP
int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model_dir> [port=30001] [host]\n", argv[0]);
    return 1;
  }
  return sgl_mlx_http_serve(argv[1], argc > 2 ? atoi(argv[2]) : 30001,
                            argc > 3 ? argv[3] : NULL);
}
#endif
