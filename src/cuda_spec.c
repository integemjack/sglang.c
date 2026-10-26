/* sgl CUDA 侧投机解码 v1（G7）— draft-verify，与 CPU 版 src/spec.c 逐 token
 * 等价：每轮 draft 连出 K 候选（decode_batch B=1 图快路径，写 draft 自己的
 * KV 段）→ target 一次 n-token 批量前向（sgl_cuda_forward_seg_rows，lm_head
 * 一张 M=n 的 GEMM）拿全部候选位 logits → 逐步用【上一位置分布】比对
 * （forward 返回的是吃掉 token 后的分布，比较对象错前一位，同 CPU v2）。
 * 命中接受；首个不匹配采纳 target argmax；全接受 bonus 一个。
 * 每个最终 token 均为 target argmax ⇒ 与同 target 非 spec greedy 全等
 * （批量/单 token kernel 归约顺序不同的 fp 噪声级 tie 翻转除外）。
 * KV 段隔离：target 用 [t_base, ...)，draft 用 [d_base, ...)，互不覆盖；
 * draft==target（同实例两段）时接受率应为 100%，用作正确性哨兵。
 * temperature>0 的修正采样未实现（同 CPU 版）。 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/cuda_spec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/alloc.h"
#include "sgl/log.h"

/* G14 诊断：SGL_SPEC_PROF=1 时累计各阶段耗时（draft 图步 / verify / 两侧补齐），
 * 结束时打一行到 stderr —— 这是“该往哪个方向优化”的唯一可靠依据（WDDM 下
 * 绝对值会飘，但同一进程内的占比是稳定的）。*/
typedef struct {
  double draft, verify, catch_t, catch_d, bonus_prefill;
  int n_round;
} spec_prof;
static int spec_prof_on(void) {
  static int on = -1;
  if (on < 0) on = getenv("SGL_SPEC_PROF") ? 1 : 0;
  return on;
}
static double now_sec(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int argmax_v(const float *lg, int n) {
  int best = 0;
  for (int i = 1; i < n; i++) if (lg[i] > lg[best]) best = i;
  return best;
}

/* KV 从 *kv_len 补到 want。n==1（每轮的 reject/bonus 补齐）走 decode_batch
 * 图快路径：G7 实测 plain forward_seg 在 WDDM 下比图步贵 2-3 倍，而补齐每轮
 * 固定发生两次（target+draft），是 GPU spec 的主要常数开销；
 * decode_batch==forward 逐位等价已由 cuda_test [batch] 断言。
 * n>1（prompt prefill/异常长跳）仍走批量 forward_seg（本就只回末行）。 */
static int kv_catch_up_gpu(sgl_cuda_model *m, int base, const int *seq,
                           int want, int *kv_len, float *dl, double *acc) {
  int n = want - *kv_len;
  if (n <= 0) return 0;
  double t0 = acc ? now_sec() : 0.0;
  if (n == 1) {
    int b = base, l = *kv_len;
    if (sgl_cuda_decode_batch(m, seq + *kv_len, &b, &l, 1, dl) != 0) return -1;
    *kv_len = want;
    if (acc) *acc += now_sec() - t0;
    return 0;
  }
  if (sgl_cuda_forward_seg(m, seq + *kv_len, n, *kv_len, base, dl) != 0)
    return -1;
  *kv_len = want;
  if (acc) *acc += now_sec() - t0;
  return 0;
}

int sgl_cuda_spec_generate(sgl_cuda_model *target, int t_base,
                           sgl_cuda_model *draft, int d_base,
                           const int *prompt, int n_prompt, int max_new,
                           int k_spec, int eos_id, int **out_ids,
                           int *out_drafted, int *out_accepted) {
  if (!target || !draft || !prompt || !out_ids) return -1;
  if (n_prompt <= 0 || max_new <= 0) return -1;
  int V = sgl_cuda_vocab(target);
  if (V != sgl_cuda_vocab(draft)) {
    SGL_ERROR("cuda_spec: draft vocab=%d != target %d（需同词表）",
              sgl_cuda_vocab(draft), V);
    return -1;
  }
  int K = k_spec < 1 ? 1 : k_spec > 8 ? 8 : k_spec;
  int cap = n_prompt + max_new + K + 8;
  if (t_base + cap > sgl_cuda_ctx(target) ||
      d_base + cap > sgl_cuda_ctx(draft)) {
    SGL_ERROR("cuda_spec: segment overflow (cap=%d t_ctx=%d d_ctx=%d)", cap,
              sgl_cuda_ctx(target), sgl_cuda_ctx(draft));
    return -1;
  }
  int rc = -1;
  int *seq = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap);
  float *dl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  float *tl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  float *vl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * (size_t)(K + 1));
  if (!seq || !dl || !tl || !vl) goto done;
  memcpy(seq, prompt, sizeof(int) * (size_t)n_prompt);
  int len = n_prompt, t_kv = 0, d_kv = 0;
  int produced = 0, n_accepts = 0, n_drafted = 0;
  spec_prof prof;
  memset(&prof, 0, sizeof(prof));
  double *acc_t = spec_prof_on() ? &prof.catch_t : NULL;
  double *acc_d = spec_prof_on() ? &prof.catch_d : NULL;

  /* G14 流水线：**预填 np-1 行**，把 prompt 末 token 留作“挂起 token”（pending），
   * 它由下一轮的 verify 第 0 行吞下去。这样每轮 verify 都是固定 n = K+1 行，
   * 而 reject/bonus token 的 KV 不再需要单独跑一次 target 前向（旧结构每轮固定
   * 一次 target 补齐 = 11.9ms，占总时长 30-46%）。实测分解见 BENCH G14 节。*/
  int npf = n_prompt > 1 ? n_prompt - 1 : n_prompt; /* prefill 行数 */
  if (npf > 0) {
    if (kv_catch_up_gpu(target, t_base, seq, npf, &t_kv, tl, acc_t) != 0) goto done;
    if (kv_catch_up_gpu(draft, d_base, seq, npf, &d_kv, dl, acc_d) != 0) goto done;
  }

  while (produced < max_new) {
    prof.n_round++;
    /* npend = 已输出但未进 target KV 的 token 数（≤1）。首轮为 0（np==1 时
     * 预填了整个 prompt），之后恒为 1。*/
    int npend = len - t_kv;
    if (npend > 1) npend = 1; /* 防御：不应发生 */
    /* 1) draft：先吞 pending（如有），再连出 K 候选（B=1 图快路径）*/
    int cand[8], nk = 0;
    double td0 = spec_prof_on() ? now_sec() : 0.0;
    if (npend == 1) {
      int p = seq[len - 1];
      if (sgl_cuda_decode_batch(draft, &p, &d_base, &d_kv, 1, dl) != 0) goto done;
      d_kv++;
    }
    for (int i = 0; i < K && produced + nk < max_new; i++) {
      int t = argmax_v(dl, V);
      cand[nk++] = t;
      if (sgl_cuda_decode_batch(draft, &t, &d_base, &d_kv, 1, dl) != 0) goto done;
      d_kv++;
    }
    n_drafted += nk;
    if (spec_prof_on()) prof.draft += now_sec() - td0;
    if (nk == 0) break;

    /* 2) target 一次前向验证固定 n = npend + nk 行：
     *    row r 吃掉 tokens[r]（位于 pos0+r），输出的是 pos0+r+1 位置的分布。
     *    所以“位于 len+i 的候选 cand[i]”要跟 vl[npend-1+i] 比（npend==0 时
     *    i=0 用 tl = prefill 末位分布）。*/
    int vt[9];
    int vn = 0;
    if (npend == 1) vt[vn++] = seq[len - 1];
    for (int i = 0; i < nk; i++) vt[vn++] = cand[i];
    int pos0 = len - npend;
    if ((long)pos0 + vn > sgl_cuda_ctx(target) || (long)d_base + len + nk > sgl_cuda_ctx(draft))
      break; /* 段用完就收工（与旧行为一致：不写非法 KV 行）*/
    int accepted = 0, reject_tok = -1;
    double tv0 = spec_prof_on() ? now_sec() : 0.0;
    if (sgl_cuda_forward_seg_rows(target, vt, vn, pos0, t_base, vl) != 0) goto done;
    if (spec_prof_on()) prof.verify += now_sec() - tv0;
    t_kv = pos0 + vn;
    for (int i = 0; i < nk; i++) {
      int ridx = npend - 1 + i; /* 候选 cand[i] 位置上的分布所在行 */
      const float *p = ridx >= 0 ? vl + (size_t)ridx * V : tl;
      int g = argmax_v(p, V);
      if (g == cand[i]) { accepted++; n_accepts++; }
      else { reject_tok = g; break; }
    }

    /* 3) 写入确认 token（拒绝时采纳 g；全接受 bonus=argmax(末候选位分布)）*/
    for (int i = 0; i < accepted; i++) seq[len + i] = cand[i];
    int n_new = accepted;
    if (reject_tok >= 0) {
      seq[len + accepted] = reject_tok;
      n_new++;
    } else if (produced + nk < max_new) {
      seq[len + nk] = argmax_v(vl + (size_t)(vn - 1) * V, V);
      n_new++;
    }
    len += n_new;
    produced += n_new;

    /* 4) 流水线推进：末 token（reject/bonus）成为下一轮的 pending。
     *    target KV 里 [len-1, t_kv) 的候选行可能已作废，但下一轮 verify 从
     *    pos0 = len-1 重新写起，必然覆写；draft 同理（它下轮先吞 pending）。*/
    t_kv = len - 1;
    d_kv = len - 1;

    /* 5) eos：截去其后的本轮 token（同 CPU 版）*/
    if (eos_id >= 0) {
      for (int i = len - n_new; i < len; i++) {
        if (seq[i] == eos_id) {
          int new_len = i + 1;
          produced -= (len - new_len);
          len = new_len;
          produced = max_new;
          t_kv = len; /* 收工，不再需要 pending */
          d_kv = len;
          break;
        }
      }
    }
  }

  if (spec_prof_on()) {
    double tot = prof.draft + prof.verify + prof.catch_t + prof.catch_d;
    fprintf(stderr,
            "[spec-prof] K=%d rounds=%d drafted=%d accepted=%d (%.1f%%) | "
            "draft %.1fms verify %.1fms catchT %.1fms catchD %.1fms | sum %.1fms "
            "(draft %.0f%% verify %.0f%% catch %.0f%%)\n",
            K, prof.n_round, n_drafted, n_accepts,
            n_drafted ? 100.0 * n_accepts / n_drafted : 0.0, prof.draft * 1e3,
            prof.verify * 1e3, prof.catch_t * 1e3, prof.catch_d * 1e3, tot * 1e3,
            tot > 0 ? 100 * prof.draft / tot : 0.0,
            tot > 0 ? 100 * prof.verify / tot : 0.0,
            tot > 0 ? 100 * (prof.catch_t + prof.catch_d) / tot : 0.0);
  }

  *out_ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)(len - n_prompt));
  if (!*out_ids) goto done;
  memcpy(*out_ids, seq + n_prompt, sizeof(int) * (size_t)(len - n_prompt));
  if (out_drafted) *out_drafted = n_drafted;
  if (out_accepted) *out_accepted = n_accepts;
  rc = len - n_prompt;

done:
  sgl_xfree(seq); sgl_xfree(dl); sgl_xfree(tl); sgl_xfree(vl);
  return rc;
}
