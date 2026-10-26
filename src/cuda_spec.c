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

#include "sgl/alloc.h"
#include "sgl/log.h"

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
                           int want, int *kv_len, float *dl) {
  int n = want - *kv_len;
  if (n <= 0) return 0;
  if (n == 1) {
    int b = base, l = *kv_len;
    if (sgl_cuda_decode_batch(m, seq + *kv_len, &b, &l, 1, dl) != 0) return -1;
    *kv_len = want;
    return 0;
  }
  if (sgl_cuda_forward_seg(m, seq + *kv_len, n, *kv_len, base, dl) != 0)
    return -1;
  *kv_len = want;
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
  float *vl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * (size_t)K);
  if (!seq || !dl || !tl || !vl) goto done;
  memcpy(seq, prompt, sizeof(int) * (size_t)n_prompt);
  int len = n_prompt, t_kv = 0, d_kv = 0;
  int produced = 0, n_accepts = 0, n_drafted = 0;

  if (kv_catch_up_gpu(target, t_base, seq, n_prompt, &t_kv, tl) != 0) goto done;
  if (kv_catch_up_gpu(draft, d_base, seq, n_prompt, &d_kv, dl) != 0) goto done;

  while (produced < max_new) {
    /* 1) draft 连出 K 候选（B=1 批量解码路径 = graph 快路径；token 写在
     *    d_base+d_kv 行，attend [d_base, d_base+d_kv)）*/
    int cand[8], nk = 0;
    for (int i = 0; i < K && produced + nk < max_new; i++) {
      int t = argmax_v(dl, V);
      cand[nk++] = t;
      if (sgl_cuda_decode_batch(draft, &t, &d_base, &d_kv, 1, dl) != 0) goto done;
      d_kv++;
    }
    n_drafted += nk;

    /* 2) target 一次批量验证：vl[i] = 吃掉 cand[0..i] 后的分布。
     *    i=0 用 tl（prompt 末位），i>0 用 vl 第 i-1 行（同 CPU v2）。 */
    int accepted = 0, reject_tok = -1;
    if (sgl_cuda_forward_seg_rows(target, cand, nk, t_kv, t_base, vl) != 0) goto done;
    t_kv += nk; /* 批量一次写入全部候选位 KV；作废时轮末回退计数 */
    for (int i = 0; i < nk; i++) {
      const float *p = (i == 0) ? tl : vl + (size_t)(i - 1) * V;
      int g = argmax_v(p, V); /* 上一位置对 cand[i] 位置的预测 */
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
      seq[len + nk] = argmax_v(vl + (size_t)(nk - 1) * V, V);
      n_new++;
    }
    len += n_new;
    produced += n_new;

    /* 4) KV 有效长度回退（作废候选行靠下轮覆写，同 CPU 版）：
     *    末 token（reject/bonus）未入 KV 则 valid=len-1，否则 len。 */
    {
      int valid = (n_new > accepted) ? len - 1 : len;
      t_kv = valid;
      d_kv = valid;
    }

    /* 5) eos：截去其后的本轮 token（同 CPU 版）*/
    if (eos_id >= 0) {
      for (int i = len - n_new; i < len; i++) {
        if (seq[i] == eos_id) {
          int new_len = i + 1;
          produced -= (len - new_len);
          len = new_len;
          produced = max_new;
          break;
        }
      }
    }
    /* 6) 两模型 KV 补齐到新 len（reject/bonus token 的 KV；dl/tl 同步末位分布）*/
    if (produced < max_new) {
      if (kv_catch_up_gpu(target, t_base, seq, len, &t_kv, tl) != 0) goto done;
      if (kv_catch_up_gpu(draft, d_base, seq, len, &d_kv, dl) != 0) goto done;
    }
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
