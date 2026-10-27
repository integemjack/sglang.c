/* cuda_perf.c — GPU 性能基准（G1 prefill / G2 W8 / G3 graph 的 A/B 工具）
 *
 * 用法: cuda_perf <model_dir> [--prompt N[,N..]] [--decode M] [--batch B[,B..]]
 *                   [--rounds R] [--seg S] [--spec <draft_dir> [--spec-k K] [--newtok N]]
 * 档位由环境变量决定（这样同一份二进制可以 A/B 对照）:
 *   SGL_CUDA_FP16 / SGL_CUDA_W8 / SGL_W8_DP4A / SGL_CUDA_GRAPH / SGL_CUDA_CTX
 * G7: --spec 在 target(dir)+draft(draft_dir) 上报 spec vs 非 spec greedy tok/s
 *   与接受率（同 prompt 同 max_new，best-of-rounds；非 spec 走 decode_batch B=1
 *   图快路径 = 生产基线）。tinyllama+llama-160m 请配 SGL_CUDA_FP16=1。
 * 输出每行一个 [perf] 记录，best-of-rounds，单位 ms / tok/s。
 * 不进 ctest：纯测量工具，不做断言。
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sgl/port.h" /* _putenv 等 MSVC/C11-K 垫片（Linux 构建需要）*/

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/cuda.h"
#include "sgl/cuda_spec.h"
#include "sgl/llama.h"

static double now_sec(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int parse_list(const char *s, int *out, int max) {
  int n = 0;
  const char *p = s;
  while (*p && n < max) {
    out[n++] = atoi(p);
    const char *c = strchr(p, ',');
    if (!c) break;
    p = c + 1;
  }
  return n;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = "models/qwen05";
  int pn[8] = {512, 2048, 8192};
  int npn = 3;
  int bs[8] = {1, 4, 8};
  int nbs = 3;
  int decode_steps = 32;
  int rounds = 3;
  int ab = 0;   /* G1/G2: same-process A/B (toggle the path per round) */
  int seg = 0; /* 0 => ctx/8 per sequence */
  const char *spec_dir = NULL; /* G7: draft model dir for --spec */
  int spec_k = 4, newtok = 32;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
    else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) npn = parse_list(argv[++i], pn, 8);
    else if (!strcmp(argv[i], "--batch") && i + 1 < argc) nbs = parse_list(argv[++i], bs, 8);
    else if (!strcmp(argv[i], "--decode") && i + 1 < argc) decode_steps = atoi(argv[i + 1]), i++;
    else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[i + 1]), i++;
    else if (!strcmp(argv[i], "--seg") && i + 1 < argc) seg = atoi(argv[i + 1]), i++;
    else if (!strcmp(argv[i], "--spec") && i + 1 < argc) spec_dir = argv[++i];
    else if (!strcmp(argv[i], "--spec-k") && i + 1 < argc) spec_k = atoi(argv[i + 1]), i++;
    else if (!strcmp(argv[i], "--newtok") && i + 1 < argc) newtok = atoi(argv[i + 1]), i++;
    else if (!strcmp(argv[i], "--ab")) ab = 1;
    else dir = argv[i]; /* positional model dir */
  }

  sgl_llama *cpu = sgl_llama_load(dir);
  if (!cpu) { fprintf(stderr, "cuda_perf: cannot load %s\n", dir); return 1; }
  int V = cpu->vocab;
  sgl_cuda_model *gpu = sgl_cuda_from_llama(cpu);
  if (!gpu) { fprintf(stderr, "cuda_perf: cannot upload %s to GPU\n", dir); return 1; }
  int ctx = sgl_cuda_ctx(gpu);
  if (!seg) seg = ctx / 8;
  printf("[cfg] model=%s ctx=%d seg=%d rounds=%d free_vram=%ldMB\n", dir, ctx, seg, rounds,
         sgl_cuda_free_mem_mb());

  /* synthetic prompt: cycling ids in a safe range (timing is content-free) */
  int maxn = pn[0];
  for (int i = 1; i < npn; i++) if (pn[i] > maxn) maxn = pn[i];
  int *ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)maxn);
  for (int i = 0; i < maxn; i++) ids[i] = 100 + (i % 900) % V;
  float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * 8);

  /* ---- prefill ----
   * --ab: alternate the two code paths inside one process so both see the
   * identical allocation state. WDDM spills to system memory once the working
   * set exceeds free VRAM, which inflates absolute numbers by 3-4x (this card
   * normally has ~2.7GB free with the desktop running), so the same-process
   * ratio is the number to trust. */
  for (int i = 0; i < npn; i++) {
    int n = pn[i];
    if (n > ctx) { printf("[perf] prefill n=%-6d skipped (> ctx %d)\n", n, ctx); continue; }
    if (ab) {
      double b_old = 1e30, b_new = 1e30;
      int rc = 0;
      for (int r = 0; r < rounds; r++) {
        _putenv("SGL_ATTN_TWOPASS=1");
        double t0 = now_sec();
        rc = sgl_cuda_forward_seg(gpu, ids, n, 0, 0, lg);
        double dt = now_sec() - t0;
        _putenv("SGL_ATTN_TWOPASS=0");
        if (rc != 0) break;
        if (dt < b_old) b_old = dt;
        t0 = now_sec();
        rc = sgl_cuda_forward_seg(gpu, ids, n, 0, 0, lg);
        dt = now_sec() - t0;
        if (rc != 0) break;
        if (dt < b_new) b_new = dt;
      }
      if (rc != 0) { printf("[perf] prefill n=%-6d A/B FAILED rc=%d\n", n, rc); continue; }
      printf("[perf] prefill n=%-6d two-pass %10.1f ms | flash %8.1f ms | %.1fx\n",
             n, b_old * 1e3, b_new * 1e3, b_old / b_new);
      /* W8 A/B: same trick for the int8 projections (dp4a vs IMMA) */
      if (getenv("SGL_CUDA_W8")) {
        double w_old = 1e30, w_new = 1e30;
        for (int r = 0; r < rounds; r++) {
          _putenv("SGL_W8_DP4A=1");
          double t0 = now_sec();
          rc = sgl_cuda_forward_seg(gpu, ids, n, 0, 0, lg);
          double dt = now_sec() - t0;
          _putenv("SGL_W8_DP4A=0");
          if (rc != 0) break;
          if (dt < w_old) w_old = dt;
          t0 = now_sec();
          rc = sgl_cuda_forward_seg(gpu, ids, n, 0, 0, lg);
          dt = now_sec() - t0;
          if (rc != 0) break;
          if (dt < w_new) w_new = dt;
        }
        if (rc == 0)
          printf("[perf] prefill n=%-6d w8-dp4a %10.1f ms | w8-imma %8.1f ms | %.1fx\n",
                 n, w_old * 1e3, w_new * 1e3, w_old / w_new);
        else printf("[perf] prefill n=%-6d w8 A/B FAILED rc=%d\n", n, rc);
      }
      continue;
    }
    double best = 1e30;
    int rc = 0;
    for (int r = 0; r < rounds; r++) {
      double t0 = now_sec();
      rc = sgl_cuda_forward_seg(gpu, ids, n, 0, 0, lg);
      double dt = now_sec() - t0;
      if (rc != 0) break;
      if (dt < best) best = dt;
    }
    if (rc != 0) { printf("[perf] prefill n=%-6d FAILED rc=%d\n", n, rc); continue; }
    printf("[perf] prefill n=%-6d %10.1f ms  %8.1f tok/s\n", n, best * 1e3, n / best);
  }

  /* ---- decode (B sequences, one step each) ---- */
  for (int i = 0; i < nbs; i++) {
    int B = bs[i];
    if (B < 1 || B > 32) continue;
    int *toks = (int *)sgl_xmalloc(sizeof(int) * (size_t)B * 3);
    int *base = toks + B, *lens = toks + 2 * B;
    for (int b = 0; b < B; b++) { toks[b] = 1 + (b * 7) % (V - 2); base[b] = b * seg; lens[b] = 16; }
    if (base[B - 1] + lens[B - 1] + decode_steps + 1 > ctx) {
      printf("[perf] decode B=%-3d skipped (seg %d x B > ctx %d)\n", B, seg, ctx);
      sgl_xfree(toks); continue;
    }
    double best = 1e30;
    int rc = 0, steps_done = 0;
    for (int r = 0; r < rounds; r++) {
      /* warm the segment (also warms graph capture / workspace growth) */
      for (int b = 0; b < B; b++) lens[b] = 16;
      double t0 = now_sec();
      for (int s = 0; s < decode_steps; s++) {
        rc = sgl_cuda_decode_batch(gpu, toks, base, lens, B, lg);
        if (rc != 0) break;
        for (int b = 0; b < B; b++) {
          int top = 0;
          for (int j = 1; j < V; j++) if (lg[(size_t)b * V + j] > lg[(size_t)b * V + top]) top = j;
          toks[b] = top;
          lens[b]++;
        }
      }
      double dt = now_sec() - t0;
      if (rc != 0) break;
      steps_done = decode_steps;
      if (dt < best) best = dt;
    }
    if (rc != 0) { printf("[perf] decode B=%-3d FAILED rc=%d\n", B, rc); sgl_xfree(toks); continue; }
    double tps = (double)steps_done * B / best;
    printf("[perf] decode B=%-3d %10.1f ms/step  %8.1f tok/s aggregate\n", B,
           best * 1e3 / steps_done, tps);
    sgl_xfree(toks);
  }

  /* ---- G7: speculative decoding vs plain greedy decode ----
   * 同 prompt（target tokenizer 真实文本，默认 'The meaning of life is'，
   * SGL_SPEC_PROMPT 可改）、同 max_new（eos=-1 固定长度，两边产出相同 token
   * 数）；非 spec 走 prefill + decode_batch B=1（图快路径，生产基线）；
   * spec 走 sgl_cuda_spec_generate（draft=decode_batch B=1，verify=一次
   * forward_seg_rows）。best-of-rounds；接受率取最快轮（确定性，各轮相同）。
   * 逐 token 等价性断言在 cuda_test [spec]，这里只测速。 */
  if (spec_dir) {
    char tp[600];
    snprintf(tp, sizeof(tp), "%s/tokenizer.json", dir);
    sgl_tokenizer *stk = sgl_tokenizer_load(tp);
    const char *ptxt = getenv("SGL_SPEC_PROMPT");
    if (!ptxt || !*ptxt) ptxt = "The meaning of life is";
    int np = 0;
    int *pids = stk ? sgl_tokenizer_encode(stk, ptxt, strlen(ptxt), &np) : NULL;
    sgl_llama *dcpu = sgl_llama_load(spec_dir);
    sgl_cuda_model *dgpu = dcpu ? sgl_cuda_from_llama(dcpu) : NULL;
    if (!pids || !dgpu) {
      printf("[perf] spec SKIPPED (tokenizer/draft upload failed)\n");
    } else if (sgl_cuda_vocab(dgpu) != V) {
      printf("[perf] spec SKIPPED (draft vocab %d != target %d)\n",
             sgl_cuda_vocab(dgpu), V);
    } else {
      int need = np + newtok + spec_k + 8;
      int c7 = ctx < sgl_cuda_ctx(dgpu) ? ctx : sgl_cuda_ctx(dgpu);
      if (need > c7) {
        newtok = c7 - np - spec_k - 8;
        if (newtok < 8) { printf("[perf] spec SKIPPED (ctx too small)\n"); newtok = 0; }
        else printf("[perf] spec: newtok clamped to %d (ctx %d)\n", newtok, c7);
      }
      if (newtok > 0) {
        double b_plain = 1e30, b_spec = 1e30;
        int best_dr = 0, best_ac = 0, rc7 = 0;
        int base0 = 0;
        for (int r = 0; r < rounds && rc7 == 0; r++) {
          /* plain: prefill + decode_batch B=1（生产基线） */
          double t0 = now_sec();
          rc7 = sgl_cuda_forward_seg(gpu, pids, np, 0, 0, lg);
          int l = np, last = -1;
          for (int s = 0; rc7 == 0 && s < newtok; s++) {
            int t = 0;
            for (int j = 1; j < V; j++) if (lg[j] > lg[t]) t = j;
            last = t;
            if (s + 1 < newtok) {
              rc7 = sgl_cuda_decode_batch(gpu, &t, &base0, &l, 1, lg);
              l++;
            }
          }
          double dt = now_sec() - t0;
          if (rc7 == 0 && dt < b_plain) b_plain = dt;
          (void)last;
        }
        for (int r = 0; r < rounds && rc7 == 0; r++) {
          /* spec：同长度、同起点（段 0，覆写 plain 的 KV 无妨，串行） */
          int *out7 = NULL, dr7 = 0, ac7 = 0;
          double t0 = now_sec();
          int n7 = sgl_cuda_spec_generate(gpu, 0, dgpu, 0, pids, np, newtok,
                                          spec_k, -1, &out7, &dr7, &ac7);
          double dt = now_sec() - t0;
          if (n7 != newtok) { printf("[perf] spec FAILED n=%d want %d\n", n7, newtok);
                              rc7 = -1; sgl_xfree(out7); break; }
          if (dt < b_spec) { b_spec = dt; best_dr = dr7; best_ac = ac7; }
          sgl_xfree(out7);
        }
        if (rc7 == 0 && b_plain < 1e29 && b_spec < 1e29) {
          double ptps = newtok / b_plain, sts = newtok / b_spec;
          printf("[perf] spec target=%s draft=%s K=%d prompt='%s' new=%d rounds=%d: "
                 "plain %6.1f tok/s | spec %6.1f tok/s (%.2fx) | accept %.1f%% (%d/%d)\n",
                 dir, spec_dir, spec_k, ptxt, newtok, rounds,
                 ptps, sts, sts / ptps,
                 best_dr ? 100.0 * best_ac / best_dr : 0.0, best_ac, best_dr);
        }
      }
    }
    if (stk) sgl_tokenizer_free(stk);
    if (pids) sgl_xfree(pids);
    if (dgpu) sgl_cuda_free(dgpu);
    if (dcpu) sgl_llama_free(dcpu);
  }

  sgl_xfree(ids); sgl_xfree(lg);
  sgl_cuda_free(gpu);
  sgl_llama_free(cpu);
  return 0;
}
