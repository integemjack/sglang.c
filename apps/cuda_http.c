/* sgl cuda_http.c — CUDA 后端薄壳（W1）
 *
 * 只做两件事：
 *   1) 把 include/sgl/cuda.h 的入口装进 sgl_accel_api 函数表（sgl_accel_cuda()）；
 *   2) 提供 main 与 sgl_cuda_http_serve()（sglc --device cuda 走它）。
 * 服务逻辑全在 apps/accel_http.c（后端无关）。下面的适配函数只把不透明句柄在
 * sgl_cuda_model* 与 sgl_accel* 之间转一下，不含任何逻辑——语义以 cuda.h 的注释为准。
 *
 * 编宏 SGL_EMBED_CUDA_HTTP：sglc 把本文件与 accel_http.c 一起编进去时定义，
 * 此时不出 main（sglc 自己有）。 */
#include <stdio.h>
#include <stdlib.h>

#include "sgl/accel.h"
#include "sgl/cuda.h"
#include "sgl/cuda_deepseek.h"

#define CU(m) ((sgl_cuda_model *)(m))
#define CUC(m) ((const sgl_cuda_model *)(m))
#define AC(p) ((sgl_accel *)(p))

static sgl_accel *cu_from_llama(const void *m) { return AC(sgl_cuda_from_llama(m)); }
static sgl_accel *cu_from_gpt2(const void *m) { return AC(sgl_cuda_from_gpt2(m)); }
static void cu_free(sgl_accel *m) { sgl_cuda_free(CU(m)); }
static int cu_ctx(const sgl_accel *m) { return sgl_cuda_ctx(CUC(m)); }
static int cu_vocab(const sgl_accel *m) { return sgl_cuda_vocab(CUC(m)); }

static int cu_forward_seg(sgl_accel *m, const int *t, int n, int p0, int kb, float *lg) {
  return sgl_cuda_forward_seg(CU(m), t, n, p0, kb, lg);
}
static int cu_forward_seg_rows(sgl_accel *m, const int *t, int n, int p0, int kb, float *lg) {
  return sgl_cuda_forward_seg_rows(CU(m), t, n, p0, kb, lg);
}
static int cu_forward_seg_emb(sgl_accel *m, const float *e, int n, int p0, int kb, float *lg) {
  return sgl_cuda_forward_seg_emb(CU(m), e, n, p0, kb, lg);
}
static int cu_decode_batch(sgl_accel *m, const int *t, const int *bp, const int *l, int B,
                           float *lg) {
  return sgl_cuda_decode_batch(CU(m), t, bp, l, B, lg);
}
static int cu_copy_kv(sgl_accel *m, int s, int d, int n) {
  return sgl_cuda_copy_kv(CU(m), s, d, n);
}
/* 「设备内存」在 CUDA 上就是 VRAM */
static long cu_estimate_mem_mb(const void *m, int ctx, int rows) {
  return sgl_cuda_estimate_vram_mb(m, ctx, rows);
}
static long cu_avail_mem_mb(void) { return sgl_cuda_avail_vram_mb(); }
static int cu_prewarm(sgl_accel *m, int rows, int max_b) {
  return sgl_cuda_prewarm(CU(m), rows, max_b);
}

static int cu_lin_slots(const sgl_accel *m) { return sgl_cuda_lin_slots(CUC(m)); }
static void cu_set_lin_slot(sgl_accel *m, int s) { sgl_cuda_set_lin_slot(CU(m), s); }
static void cu_set_decode_slots(sgl_accel *m, const int *s, int B) {
  sgl_cuda_set_decode_slots(CU(m), s, B);
}

static int cu_mtp_load(sgl_accel *m, int kv_ctx) { return sgl_cuda_mtp_load(CU(m), kv_ctx); }
static int cu_mtp_ready(const sgl_accel *m) { return sgl_cuda_mtp_ready(CUC(m)); }
static void cu_mtp_set_base(sgl_accel *m, int b) { sgl_cuda_mtp_set_base(CU(m), b); }
static int cu_mtp_set_hidden(sgl_accel *m, int row) { return sgl_cuda_mtp_set_hidden(CU(m), row); }
static int cu_mtp_drafts(sgl_accel *m, int t0, int pos, int W, float minp, int *out, float *pr) {
  return sgl_cuda_mtp_drafts(CU(m), t0, pos, W, minp, out, pr);
}
static const float *cu_mtp_hprev(const sgl_accel *m) { return sgl_cuda_mtp_hprev(CUC(m)); }
static const float *cu_mtp_hlast(const sgl_accel *m) { return sgl_cuda_mtp_hlast(CUC(m)); }
static int cu_mtp_fill_kv(sgl_accel *m, const int *t, int n, int p0, const float *hp) {
  return sgl_cuda_mtp_fill_kv(CU(m), t, n, p0, hp);
}
static int cu_spec_begin(sgl_accel *m, int W) { return sgl_cuda_spec_begin(CU(m), W); }
static int cu_spec_verify(sgl_accel *m, const int *t, int n, int p0, int kb, float *rows, int *k) {
  return sgl_cuda_spec_verify(CU(m), t, n, p0, kb, rows, k);
}
static int cu_spec_commit(sgl_accel *m, int slot, int k) {
  return sgl_cuda_spec_commit(CU(m), slot, k);
}
static int cu_mtp_accept_test(sgl_accel *m, const int *p, int np, int ng, double *acc, int *nok) {
  return sgl_cuda_mtp_accept_test(CU(m), p, np, ng, acc, nok);
}

/* ---- DeepSeek（MLA + MoE）：sgl_cuda_ds 是另一套结构，不经 sgl_accel 句柄 ---- */
static void *cu_ds_load(const void *ds_cpu) { return (void *)sgl_cuda_ds_from(ds_cpu); }
static void cu_ds_free(void *ds) { sgl_cuda_ds_free((sgl_cuda_ds *)ds); }
static int cu_ds_vocab(const void *ds) { return sgl_cuda_ds_vocab((const sgl_cuda_ds *)ds); }
static int cu_ds_ctx(const void *ds) { return sgl_cuda_ds_ctx((const sgl_cuda_ds *)ds); }
static void cu_ds_reset(void *ds) { sgl_cuda_ds_reset((sgl_cuda_ds *)ds); }
static int cu_ds_forward(void *ds, const int *t, int n, int p0, float *lg) {
  return sgl_cuda_ds_forward_n((sgl_cuda_ds *)ds, t, n, p0, lg, 0);
}
static int cu_ds_decode(void *ds, int token, int pos, float *lg) {
  return sgl_cuda_ds_decode((sgl_cuda_ds *)ds, token, pos, lg);
}

/* 用指定初始化器：字段增删/换序都不会静默错位 */
static const sgl_accel_api CUDA_API = {
    .name = "cuda",
    .abi = SGL_ACCEL_ABI,
    .from_llama = cu_from_llama,
    .free = cu_free,
    .ctx = cu_ctx,
    .vocab = cu_vocab,
    .forward_seg = cu_forward_seg,
    .decode_batch = cu_decode_batch,
    .estimate_mem_mb = cu_estimate_mem_mb,
    .avail_mem_mb = cu_avail_mem_mb,
    .forward_seg_rows = cu_forward_seg_rows,
    .forward_seg_emb = cu_forward_seg_emb,
    .from_gpt2 = cu_from_gpt2,
    .copy_kv = cu_copy_kv,
    .prewarm = cu_prewarm,
    .lin_slots = cu_lin_slots,
    .set_lin_slot = cu_set_lin_slot,
    .set_decode_slots = cu_set_decode_slots,
    .mtp_load = cu_mtp_load,
    .mtp_ready = cu_mtp_ready,
    .mtp_set_base = cu_mtp_set_base,
    .mtp_set_hidden = cu_mtp_set_hidden,
    .mtp_drafts = cu_mtp_drafts,
    .mtp_hprev = cu_mtp_hprev,
    .mtp_hlast = cu_mtp_hlast,
    .mtp_fill_kv = cu_mtp_fill_kv,
    .spec_begin = cu_spec_begin,
    .spec_verify = cu_spec_verify,
    .spec_commit = cu_spec_commit,
    .mtp_accept_test = cu_mtp_accept_test,

    /* ---- DeepSeek（MLA + MoE）：cuda_deepseek.cu 的独立结构 ---- */
    .ds_load = cu_ds_load,
    .ds_free = cu_ds_free,
    .ds_vocab = cu_ds_vocab,
    .ds_ctx = cu_ds_ctx,
    .ds_reset = cu_ds_reset,
    .ds_forward = cu_ds_forward,
    .ds_decode = cu_ds_decode,
};

const sgl_accel_api *sgl_accel_cuda(void) { return &CUDA_API; }

/* 兼容入口：sglc --device cuda 与独立 cuda_http.exe 都走这里 */
int sgl_cuda_http_serve(const char *dir, int port, const char *host) {
  return sgl_accel_http_serve(&CUDA_API, dir, port, host);
}

#ifndef SGL_EMBED_CUDA_HTTP
int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model_dir> [port=30001] [host]\n", argv[0]);
    return 1;
  }
  return sgl_cuda_http_serve(argv[1], argc > 2 ? atoi(argv[2]) : 30001,
                             argc > 3 ? argv[3] : NULL);
}
#endif
