/* sgl speculative decoding（draft-verify；对应 sglang srt/speculative 的 greedy 版）
 * 每轮：draft 连出 K 候选 → target 逐步 forward 候选，用【上一位置分布】验证
 * （forward 返回的是吃掉 token 后的分布，比较对象要错前一位）。
 * 命中接受；首个不匹配采纳 target argmax；全接受 bonus 一个。
 * 每个最终 token 均为 target argmax ⇒ 与 target-only greedy 全等。
 * 轮末把 target/draft KV 补齐到新序列长（reject/bonus token 非 draft 自产）。 */
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

/* KV 从 *kv_len 补到 want（写 seq[kv_len..want)）；dl 收尾为最后位置分布 */
static int kv_catch_up(sgl_llama *m, const int *seq, int want, int *kv_len,
                       int *pages, float *dl) {
  while (*kv_len < want) {
    if (sgl_llama_forward(m, &seq[*kv_len], 1, *kv_len, pages, dl) != 0) return -1;
    (*kv_len)++;
  }
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
  float *prev = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
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

    /* 2) target 验证：prev = 当前末位分布（对下一 token 的预测）*/
    memcpy(prev, tl, sizeof(float) * (size_t)V);
    int accepted = 0, reject_tok = -1;
    for (int i = 0; i < nk; i++) {
      if (sgl_llama_forward(target, &cand[i], 1, t_kv, tp, tl) != 0) goto fail;
      t_kv++;
      int g = argmax_v(prev, V); /* 上一位置对 cand[i] 位置的预测 */
      if (g == cand[i]) { accepted++; n_accepts++; }
      else { reject_tok = g; break; }
      memcpy(prev, tl, sizeof(float) * (size_t)V);
    }

    /* 3) 写入确认 token（拒绝时采纳 g；全接受 bonus=argmax(tl)）*/
    for (int i = 0; i < accepted; i++) seq[len + i] = cand[i];
    int n_new = accepted;
    if (reject_tok >= 0) {
      seq[len + accepted] = reject_tok;
      n_new++;
    } else if (produced + nk < max_new) {
      seq[len + nk] = argmax_v(tl, V);
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
  sgl_xfree(seq); sgl_xfree(tp); sgl_xfree(dp); sgl_xfree(dl); sgl_xfree(tl); sgl_xfree(prev);
  return len - n_prompt;

fail:
  if (tp) sgl_llama_pages_unref(target, tp, cap);
  if (dp) sgl_llama_pages_unref(draft, dp, cap);
  sgl_xfree(seq); sgl_xfree(tp); sgl_xfree(dp); sgl_xfree(dl); sgl_xfree(tl); sgl_xfree(prev);
  return -1;
}
