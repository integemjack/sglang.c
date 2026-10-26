/* sgl speculative decoding v2（draft-verify；对应 sglang srt/speculative 的 greedy 版）
 * 每轮：draft 连出 K 候选 → target 【一次 n-token 批量前向】拿到全部候选位
 * logits（n-outer 分块 GEMM，权重驻留复用；verify 成本 ≈ 单步×1.2 而非×K），
 * 再逐步用【上一位置分布】验证（forward 返回的是吃掉 token 后的分布，比较
 * 对象要错前一位；批量前向与逐 token 前向数值逐位一致，test_spec 有断言）。
 * 命中接受；首个不匹配采纳 target argmax；全接受 bonus 一个。
 * 每个最终 token 均为 target argmax ⇒ 与 target-only greedy 全等。
 * KV 补齐（含 prompt prefill）也走批量前向；轮末作废候选只回退计数不释放页。 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/spec.h"

#include <math.h>
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

/* KV 从 *kv_len 补到 want：一次 n-token 批量前向覆盖整段（含 prompt prefill），
 * dl = 末位分布。批量与逐 token 前向数值逐位一致，语义不变。
 * n>1 时 logits 是 [n][V]，dl 只有一行——就地借 n 行临时缓冲，末行拷回 dl
 * （曾因直接写 dl 越界毁堆：prefill n=6 写 6*V 到 V 大小的缓冲）。 */
static int kv_catch_up(sgl_llama *m, const int *seq, int want, int *kv_len,
                       int *pages, float *dl) {
  int n = want - *kv_len;
  if (n <= 0) return 0;
  float *tmp = (n == 1) ? dl
      : (float *)sgl_xmalloc(sizeof(float) * (size_t)n * (size_t)m->vocab);
  if (sgl_llama_forward_n(m, seq + *kv_len, n, *kv_len, pages, tmp) != 0) {
    if (tmp != dl) sgl_xfree(tmp);
    return -1;
  }
  if (tmp != dl) {
    memcpy(dl, tmp + (size_t)(n - 1) * m->vocab,
           sizeof(float) * (size_t)m->vocab);
    sgl_xfree(tmp);
  }
  *kv_len = want;
  return 0;
}

int sgl_spec_generate(sgl_llama *target, sgl_llama *draft,
                      const int *prompt, int n_prompt, int max_new,
                      int k_spec, int eos_id, int **out_ids) {
  int K = k_spec < 1 ? 1 : k_spec > 8 ? 8 : k_spec;
  int cap = n_prompt + max_new + K + 8;
  int *seq = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap);
  memcpy(seq, prompt, sizeof(int) * (size_t)n_prompt);
  int len = n_prompt;
  int t_kv = 0, d_kv = 0;

  int *tp = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap);
  int *dp = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap);
  if (sgl_llama_pages_alloc(target, tp, cap) != 0 ||
      sgl_llama_pages_alloc(draft, dp, cap) != 0) {
    SGL_ERROR("spec: pool exhausted");
    goto fail;
  }
  int V = target->vocab;
  float *dl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  float *tl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  float *vl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * (size_t)K); /* 批量 verify：[K][V] */
  int produced = 0, n_accepts = 0, n_drafted = 0;

  if (kv_catch_up(target, seq, n_prompt, &t_kv, tp, tl) != 0) goto fail;
  if (kv_catch_up(draft, seq, n_prompt, &d_kv, dp, dl) != 0) goto fail;

  while (produced < max_new) {
    /* 1) draft 连出 K 候选（写 draft KV）*/
    int cand[8], nk = 0;
    for (int i = 0; i < K && produced + nk < max_new; i++) {
      int t = argmax_v(dl, V);
      cand[nk++] = t;
      if (sgl_llama_forward(draft, &t, 1, d_kv, dp, dl) != 0) goto fail;
      d_kv++;
    }
    n_drafted += nk;

    /* 2) target 验证（v2）：一次 n-token 批量前向拿全部候选位 logits，
     *    再逐步用上一位置分布比对（i=0 用 tl，i>0 用批量结果第 i-1 行）。
     *    K 个候选一次过权重 ⇒ verify 成本 ≈ 单步×1.2 而非 ×K。 */
    int accepted = 0, reject_tok = -1;
    if (sgl_llama_forward_n(target, cand, nk, t_kv, tp, vl) != 0) goto fail;
    t_kv += nk; /* 批量一次写入全部候选位 KV；作废时轮末回退计数 */
    for (int i = 0; i < nk; i++) {
      const float *p = (i == 0) ? tl : vl + (size_t)(i - 1) * V;
      int g = argmax_v(p, V); /* 上一位置对 cand[i] 位置的预测 */
      if (g == cand[i]) { accepted++; n_accepts++; }
      else { reject_tok = g; break; }
    }

    /* 3) 写入确认 token（拒绝时采纳 g；全接受 bonus=argmax(tl)）*/
    for (int i = 0; i < accepted; i++) seq[len + i] = cand[i];
    int n_new = accepted;
    if (reject_tok >= 0) {
      seq[len + accepted] = reject_tok;
      n_new++;
    } else if (produced + nk < max_new) {
      seq[len + nk] = argmax_v(vl + (size_t)(nk - 1) * V, V); /* 末候选位分布 */
      n_new++;
    }
    len += n_new;
    produced += n_new;

    /* 4) 作废候选不释放页（槽复用靠覆写，ref 恒 1；轮中 unref 会导致下轮
     *    catch_up 重用 ref=0 页 → 结尾 double-unref 栈越界）。但 KV 有效长度必须
     *    回退：valid = 末 token（reject/bonus）未入 KV 则 len-1，否则 len。*/
    {
      int valid = (n_new > accepted) ? len - 1 : len;
      t_kv = valid;
      d_kv = valid;
    }

    /* 5) eos：截去其后的本轮 token（页不释放，回退计数即可）*/
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
    /* 6) 两模型 KV 补齐到新 len（reject/bonus token 的 KV；dl/tl 同步为末位分布）*/
    if (produced < max_new) {
      if (kv_catch_up(target, seq, len, &t_kv, tp, tl) != 0) goto fail;
      if (kv_catch_up(draft, seq, len, &d_kv, dp, dl) != 0) goto fail;
    }
  }

  if (n_drafted > 0)
    SGL_INFO("spec: drafted=%d accepted=%d（%.0f%%）", n_drafted, n_accepts,
             100.0 * n_accepts / n_drafted);
  *out_ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)(len - n_prompt));
  memcpy(*out_ids, seq + n_prompt, sizeof(int) * (size_t)(len - n_prompt));
  sgl_llama_pages_unref(target, tp, cap);
  sgl_llama_pages_unref(draft, dp, cap);
  sgl_xfree(seq); sgl_xfree(tp); sgl_xfree(dp); sgl_xfree(dl); sgl_xfree(tl); sgl_xfree(vl);
  return len - n_prompt;

fail:
  if (tp) sgl_llama_pages_unref(target, tp, cap);
  if (dp) sgl_llama_pages_unref(draft, dp, cap);
  sgl_xfree(seq); sgl_xfree(tp); sgl_xfree(dp); sgl_xfree(dl); sgl_xfree(tl); sgl_xfree(vl);
  return -1;
}
