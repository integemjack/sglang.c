/* tests/mlx_test.c — MLX 后端对拍 CPU（金标准）+ 性能
 *
 * 用法: mlx_test <model_dir> [decode_steps=12]
 *
 * 与 tests/cuda_test.c 同思路：CPU 的 sgl_llama_forward 是金标准。fp16 下不要求
 * 逐位相同，要求的是用户能感知的正确性：
 *   1) prefill logits 的 top-1 与 CPU 一致，概率误差在 fp16 量级；
 *   2) 贪心解码（teacher forcing，喂 CPU 选的 token）每步 top-1 都一致；
 *   3) 分块 prefill == 一次 prefill（段内 KV 复用 + 因果 mask 正确）；
 *   4) decode_batch(B=1) == forward_seg(n=1)；
 *   5) decode_batch(B=2, 两个不同段) == 两次单序列前向（段隔离正确）；
 *   6) copy_kv 搬运后继续解码 == 原地解码（前缀缓存可用）；
 *   7) 性能：prefill tok/s、decode tok/s（B=1 与 B=4）。
 * 环境变量：SGL_MLX_FP16=0 走 fp32（容忍收紧到 1e-3），SGL_MLX_CTX 压上下文。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/llama.h"
#include "sgl/mlx.h"

#ifndef _WIN32
#include <pthread.h>
static double now_s(void); /* 定义在后面（L65 附近），这里先用 */
/* [8] 线程因素：accel_http 的 worker 是独立线程，bench 在主线程。
 * MLX 默认 stream 是 per-thread 的 —— 两端差得多就说明开销在线程而不是计算。*/
typedef struct {
  sgl_mlx_model *g;
  int tok, np, nd;
  float *lg;
  double sec;
} th_arg;
static void *th_loop(void *p) {
  th_arg *a = (th_arg *)p;
  int bp1[1] = {0}, ln1[1] = {a->np};
  int t = a->tok;
  double t0 = now_s();
  for (int s = 0; s < a->nd; s++) {
    ln1[0] = a->np + s;
    sgl_mlx_decode_batch(a->g, &t, bp1, ln1, 1, a->lg);
  }
  a->sec = now_s() - t0;
  return NULL;
}
#endif

static int FAILURES = 0, CHECKS = 0;

#define CHECK(cond, ...)                       \
  do {                                         \
    CHECKS++;                                  \
    if (cond) {                                \
      printf("  ok   ");                       \
      printf(__VA_ARGS__);                     \
    } else {                                   \
      FAILURES++;                              \
      printf("  FAIL ");                       \
      printf(__VA_ARGS__);                     \
    }                                          \
    printf("\n");                              \
  } while (0)

/* 墙钟，不是 CPU 时间：MLX 的 eval 会等 GPU，这段时间不算进 clock()，
 * 用 clock() 会把 decode 报得比实际快好几倍（实测 42 ms vs 66 ms）。*/
static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
/* top1 与 top2 的概率差。随机权重的 fixture（*-rand）输出分布近乎均匀，
 * 这个差只有 1e-6 量级 —— 此时 top-1 是器噪声，拿它当断言会假失败。
 * 只有 gap 明显大于两边的概率误差时，top-1 一致才是有意义的信号。*/
static double top_gap(const float *l, int V) {
  double m1 = -1e30, m2 = -1e30, s = 0;
  for (int i = 0; i < V; i++) {
    if (l[i] > m1) {
      m2 = m1;
      m1 = l[i];
    } else if (l[i] > m2) {
      m2 = l[i];
    }
  }
  for (int i = 0; i < V; i++) s += exp((double)l[i] - m1);
  return (1.0 - exp(m2 - m1)) / s; /* p1 - p2 */
}

static int top1_meaningful(const float *ref, const float *got, int V, double pe) {
  double g = top_gap(ref, V);
  if (g <= 20.0 * pe + 1e-9) {
    printf("  (分布近乎均匀：top1-top2 概率差 %.3g ≤ 误差量级，top-1 不作断言)\n", g);
    return 0;
  }
  (void)got;
  return 1;
}

static int top1(const float *l, int V) {
  int b = 0;
  for (int i = 1; i < V; i++)
    if (l[i] > l[b]) b = i;
  return b;
}

/* 概率域最大误差：softmax 后比 logits 绝对值更能反映「输出会不会变」 */
static double prob_err(const float *a, const float *b, int V) {
  double ma = -1e30, mb = -1e30, sa = 0, sb = 0, e = 0;
  for (int i = 0; i < V; i++) {
    if (a[i] > ma) ma = a[i];
    if (b[i] > mb) mb = b[i];
  }
  for (int i = 0; i < V; i++) {
    sa += exp((double)a[i] - ma);
    sb += exp((double)b[i] - mb);
  }
  for (int i = 0; i < V; i++) {
    double d = fabs(exp((double)a[i] - ma) / sa - exp((double)b[i] - mb) / sb);
    if (d > e) e = d;
  }
  return e;
}

static double max_abs_err(const float *a, const float *b, int n) {
  double e = 0;
  for (int i = 0; i < n; i++) {
    double d = fabs((double)a[i] - (double)b[i]);
    if (d > e) e = d;
  }
  return e;
}

int main(int argc, char **argv) {
  const char *dir = argc > 1 ? argv[1] : "models/qwen05";
  const int steps = argc > 2 ? atoi(argv[2]) : 12;
  const char *fpe = getenv("SGL_MLX_FP16");
  const int fp16 = !(fpe && !atoi(fpe));
  const double tol = fp16 ? 5e-2 : 2e-3;

  printf("== mlx_test: %s (%s, %d 步) ==\n", dir, fp16 ? "fp16" : "fp32", steps);
  sgl_llama *cpu = sgl_llama_load(dir);
  if (!cpu) {
    printf("SKIP: CPU 侧模型加载失败\n");
    return 2;
  }
  sgl_mlx_model *g = sgl_mlx_from_llama(cpu);
  if (!g) {
    printf("SKIP: MLX 后端拒绝了这个模型（原因见上）\n");
    sgl_llama_free(cpu);
    return 2;
  }
  const int V = sgl_mlx_vocab(g), ctx = sgl_mlx_ctx(g);
  printf("  device=%s  V=%d ctx=%d  avail=%ld MB estimate=%ld MB\n", sgl_mlx_device_name(), V,
         ctx, sgl_mlx_avail_mem_mb(), sgl_mlx_estimate_mem_mb(cpu, ctx, 128));

  /* CPU 侧用恒等页表（slot[p]=p），与 MLX 的行号一一对应 */
  int *slot = (int *)malloc(sizeof(int) * (size_t)(ctx > 0 ? ctx : 1));
  for (int i = 0; i < (ctx > 0 ? ctx : 1); i++) slot[i] = i;

  const int P = 8;
  int prompt[8];
  for (int i = 0; i < P; i++) prompt[i] = (int)(((unsigned)i * 7919u + 13u) % (unsigned)V);

  float *lgc = (float *)malloc(sizeof(float) * (size_t)V);
  float *lgm = (float *)malloc(sizeof(float) * (size_t)V);
  float *lg2 = (float *)malloc(sizeof(float) * (size_t)V);
  float *lgp = (float *)malloc(sizeof(float) * (size_t)V); /* prefill 的 MLX logits 存档：
                                                            * [2] 会反复覆写 lgm，[3] 要拿原始的比 */

  /* ---- 1) prefill 对拍 ---- */
  printf("\n[1] prefill n=%d\n", P);
  if (sgl_llama_forward(cpu, prompt, P, 0, slot, lgc) != 0) {
    printf("SKIP: CPU 前向失败\n");
    return 2;
  }
  if (sgl_mlx_forward_seg(g, prompt, P, 0, 0, lgm) != 0) {
    printf("FAIL: MLX 前向返回非 0\n");
    return 1;
  }
  {
    double pe = prob_err(lgc, lgm, V), ae = max_abs_err(lgc, lgm, V);
    memcpy(lgp, lgm, sizeof(float) * (size_t)V);
    if (top1_meaningful(lgc, lgm, V, pe))
      CHECK(top1(lgc, V) == top1(lgm, V), "top-1 一致 (cpu=%d mlx=%d)", top1(lgc, V),
            top1(lgm, V));
    CHECK(pe < tol, "概率最大误差 %.4g < %.3g（logits 绝对误差 %.4g）", pe, tol, ae);
  }

  /* ---- 2) 贪心解码（teacher forcing：两边都喂 CPU 选的 token）---- */
  printf("\n[2] greedy decode %d 步\n", steps);
  {
    int tok = top1(lgc, V), mism = 0, pos = P, counted = 0;
    printf("  cpu:");
    for (int s = 0; s < steps; s++) {
      printf(" %d", tok);
      if (sgl_llama_forward(cpu, &tok, 1, pos, slot, lgc) != 0) break;
      if (sgl_mlx_forward_seg(g, &tok, 1, pos, 0, lgm) != 0) break;
      double pe = prob_err(lgc, lgm, V);
      if (top1_meaningful(lgc, lgm, V, pe)) { /* 分布够尖才数进不一致 */
        counted++;
        if (top1(lgc, V) != top1(lgm, V)) mism++;
      }
      tok = top1(lgc, V);
      pos++;
      if (pos >= ctx) break;
    }
    printf("\n");
    CHECK(mism == 0, "%d 步 top-1 全部一致（有效 %d 步，不一致 %d 步）", steps, counted, mism);
  }

  /* ---- 3) 分块 prefill == 一次 prefill（新实例，干净 KV）---- */
  printf("\n[3] 分块 prefill\n");
  {
    sgl_mlx_model *g2 = sgl_mlx_from_llama(cpu);
    if (!g2) {
      printf("  SKIP: 建不出第二个实例\n");
    } else {
      sgl_mlx_forward_seg(g2, prompt, 4, 0, 0, NULL);
      sgl_mlx_forward_seg(g2, prompt + 4, 4, 4, 0, lg2);
      double pe = prob_err(lgp, lg2, V);
      CHECK(pe < 1e-4, "4+4 分块 vs 一次 8：概率误差 %.4g < 1e-4", pe);
      if (top1_meaningful(lgp, lg2, V, pe))
        CHECK(top1(lgp, V) == top1(lg2, V), "top-1 一致 (%d vs %d)", top1(lgp, V),
              top1(lg2, V));
      sgl_mlx_free(g2);
    }
  }

  float *lgb = (float *)malloc(sizeof(float) * (size_t)V * 4); /* 批量 decode 最多 B=4 */

  /* ---- 4/5/6) 段隔离、批量 decode、copy_kv：都在一个干净实例上做 ---- */
  sgl_mlx_model *g3 = sgl_mlx_from_llama(cpu);
  if (!g3) {
    printf("\n[4-6] SKIP: 建不出第二个实例\n");
  } else {
    const int KB2 = 64, P2 = 5;
    sgl_mlx_forward_seg(g3, prompt, P, 0, 0, NULL);      /* 段 A：rows 0..7 */
    sgl_mlx_forward_seg(g3, prompt, P2, 0, KB2, NULL);   /* 段 B：rows 64..68 */
    int tokA = prompt[0], tokB = prompt[1];

    printf("\n[4] decode_batch(B=1) vs forward_seg(n=1)\n");
    sgl_mlx_forward_seg(g3, &tokA, 1, P, 0, lg2);                  /* 参考：写 row 8 */
    {
      int bp = 0, ln = P;
      CHECK(sgl_mlx_decode_batch(g3, &tokA, &bp, &ln, 1, lgb) == 0, "decode_batch B=1 返回 0");
      double pe = prob_err(lg2, lgb, V);
      CHECK(pe < 1e-3, "与 forward_seg 同分布（概率误差 %.4g）", pe);
    }

    printf("\n[5] decode_batch(B=2, 两个段) vs 两次单序列前向\n");
    /* 两条参考必须与 decode_batch 算【同一步】：
     *   段 A：forward_seg(pos0=P+1) ↔ lens[0]=P+1（都是位置 P+1、写 row P+1）
     *   段 B：forward_seg(pos0=P2)   ↔ lens[1]=P2  （都是位置 P2、写 row KB2+P2）
     * 之前把 lens[1] 写成 P2+1，等于拿「多走一步」的分布去比，玩具模型上分布近乎
     * 均匀看不出来，真模型（Qwen2.5-0.5B）上立刻差到 8e-2。*/
    sgl_mlx_forward_seg(g3, &tokB, 1, P2, KB2, lg2);               /* 段 B 参考：写 row 69 */
    {
      float *refA = (float *)malloc(sizeof(float) * (size_t)V);
      sgl_mlx_forward_seg(g3, &tokA, 1, P + 1, 0, refA);           /* 段 A 参考：写 row 9 */
      int tk[2] = {tokA, tokB}, bp[2] = {0, KB2}, ln[2] = {P + 1, P2};
      CHECK(sgl_mlx_decode_batch(g3, tk, bp, ln, 2, lgb) == 0, "decode_batch B=2 返回 0");
      double peA = prob_err(refA, lgb, V);
      double peB = prob_err(lg2, lgb + V, V);
      /* 两边都是 MLX，只是批量与否。阈值跟 dtype 走：
       *   fp32 实测 8.9e-07（逻辑逐位对）；
       *   fp16 实测 3.1e-03 —— B=2 时 Smax 被另一段拉长（本例 10 → 71），
       *   被 mask 掉的行虽然权重恰好为 0，但 matmul 的归约长度变了，
       *   fp16 的分块累加顺序跟着变 → 舍入不同。不是逻辑错（fp32 已证）。
       * 取 tol/5：比「对 CPU」紧 5 倍，又足以抓住真错（之前 lens 写错一步时是 8e-2）。*/
      double xtol = tol / 5.0;
      CHECK(peA < xtol, "序列 0 与单跑一致（概率误差 %.4g < %.3g）", peA, xtol);
      CHECK(peB < xtol, "序列 1 与单跑一致（概率误差 %.4g < %.3g）—— 段隔离正确", peB, xtol);
      int mA = top1_meaningful(refA, lgb, V, peA), mB = top1_meaningful(lg2, lgb + V, V, peB);
      if (mA || mB)
        CHECK((!mA || top1(refA, V) == top1(lgb, V)) &&
                  (!mB || top1(lg2, V) == top1(lgb + V, V)),
              "两条序列 top-1 都一致 (%d/%d, %d/%d)", top1(refA, V), top1(lgb, V),
              top1(lg2, V), top1(lgb + V, V));
      free(refA);
    }

    printf("\n[6] copy_kv（前缀缓存搬运）\n");
    if (ctx < 200) {
      printf("  SKIP: ctx=%d 太小\n", ctx);
    } else {
      const int SRC = 0, DST = 128, LEN = P + 1; /* 段 A 已写到 row 8 */
      CHECK(sgl_mlx_copy_kv(g3, SRC, DST, LEN) == 0, "copy_kv [%d,%d) → [%d,%d) 返回 0", SRC,
            SRC + LEN, DST, DST + LEN);
      int tok = prompt[2];
      sgl_mlx_forward_seg(g3, &tok, 1, LEN, SRC, lg2);  /* 原地再走一步 */
      sgl_mlx_forward_seg(g3, &tok, 1, LEN, DST, lgb);  /* 搬运后的副本上走同一步 */
      double pe = prob_err(lg2, lgb, V);
      CHECK(pe < 1e-3, "搬运后继续解码与原地一致（概率误差 %.4g）", pe);
    }

    /* ---- 7) 性能 ---- */
    printf("\n[7] 性能（BENCH 行供汇总）\n");
    {
      const int NP = 128;
      int *big = (int *)malloc(sizeof(int) * (size_t)NP);
      for (int i = 0; i < NP; i++) big[i] = prompt[i % P];
      double t0 = now_s();
      sgl_mlx_forward_seg(g3, big, NP, 0, 0, NULL);
      double tp = now_s() - t0;
      printf("  prefill %d tokens: %.3f s → %.1f tok/s\n", NP, tp, tp > 0 ? NP / tp : 0);

      const int ND = 32;
      int tok = prompt[0];
      t0 = now_s();
      for (int s = 0; s < ND; s++) sgl_mlx_forward_seg(g3, &tok, 1, NP + s, 0, NULL);
      double td = now_s() - t0;
      printf("  decode B=1 x%d: %.3f s → %.1f tok/s（%.2f ms/步）【prefill 路径，不算 lm_head】\n",
             ND, td, td > 0 ? ND / td : 0, td * 1000.0 / ND);

      /* 真实服务路径：accel_http 走的是 decode_batch，要算 lm_head（[V,E]，
       * 0.5B 模型下 272 MB fp16）并把 logits 拷回主机。以前这一项没量，
       * bench 报 63 tok/s 而 HTTP 只有 13 tok/s，就是因为差距全在这里。*/
      {
        int bp1[1] = {0}, ln1[1] = {NP};
        t0 = now_s();
        for (int s = 0; s < ND; s++) {
          ln1[0] = NP + s;
          sgl_mlx_decode_batch(g3, &tok, bp1, ln1, 1, lgb);
        }
        double tb = now_s() - t0;
        printf("  decode_batch B=1 x%d: %.3f s → %.1f tok/s（%.2f ms/步）【服务路径，含 lm_head】\n",
               ND, tb, tb > 0 ? ND / tb : 0, tb * 1000.0 / ND);
      }

      if (ctx >= 4 * 256 + ND) { /* B=4：四段各 32 token */
        int tk[4], bp[4], ln[4];
        for (int b = 0; b < 4; b++) {
          bp[b] = b * 256;
          sgl_mlx_forward_seg(g3, big, 32, 0, bp[b], NULL);
          tk[b] = prompt[b % P];
          ln[b] = 32;
        }
        t0 = now_s();
        for (int s = 0; s < ND; s++) {
          for (int b = 0; b < 4; b++) ln[b] = 32 + s;
          sgl_mlx_decode_batch(g3, tk, bp, ln, 4, lgb);
        }
        double t4 = now_s() - t0;
        printf("  decode B=4 x%d: %.3f s → %.1f tok/s 合计（%.2f ms/步）\n", ND, t4,
               t4 > 0 ? ND * 4 / t4 : 0, t4 * 1000.0 / ND);
      }
      printf("BENCH prefill_tok_s=%.1f decode1_tok_s=%.1f\n", tp > 0 ? NP / tp : 0,
             td > 0 ? ND / td : 0);

      /* [8] 线程因素：accel_http 的 worker 是个独立线程，bench 在主线程。
       * MLX 的默认 stream 是 per-thread 的 —— 若这里的数字与上面主线程那行
       * 差一大截，说明 HTTP 的额外开销主要来自线程切换而不是模型计算。
       * （注：上面那一行不带 lm_head，MLX 是异步的，量到的是「提交速度」，
       *  不代表单步真实耗时；下面两行都带 lm_head 且每步同步读回 logits。）*/
#ifndef _WIN32
      {
        th_arg a1;
        a1.g = g3;
        a1.tok = tok;
        a1.np = NP;
        a1.nd = ND;
        a1.lg = lgb;
        a1.sec = 0;
        /* 先在主线程跑一遍（已图为热），再开新线程跑同样的循环 */
        double t1 = now_s();
        {
          int bp1[1] = {0}, ln1[1] = {NP};
          int tt = tok;
          for (int s = 0; s < ND; s++) {
            ln1[0] = NP + s;
            sgl_mlx_decode_batch(g3, &tt, bp1, ln1, 1, lgb);
          }
        }
        double tmain = now_s() - t1;
        pthread_t th;
        if (pthread_create(&th, NULL, th_loop, &a1) == 0) {
          pthread_join(th, NULL);
          printf("  decode_batch B=1 主线程: %.2f ms/步   新线程: %.2f ms/步（都含 lm_head）\n",
                 tmain * 1000.0 / ND, a1.sec * 1000.0 / ND);
        } else {
          printf("  decode_batch B=1 主线程: %.2f ms/步（pthread 创建失败）\n",
                 tmain * 1000.0 / ND);
        }
      }
#endif
      free(big);
    }
    sgl_mlx_free(g3);
  }
  free(lgb);

  printf("\n== %d 项检查，%d 项失败 ==\n", CHECKS, FAILURES);
  free(slot);
  free(lgc);
  free(lgm);
  free(lg2);
  free(lgp);
  sgl_mlx_free(g);
  sgl_llama_free(cpu);
  return FAILURES ? 1 : 0;
}
