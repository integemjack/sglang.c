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
#include "sgl/sampling.h"

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
  int V = target->vocab;
  /* 这三个必须在可能 goto fail 的代码**之前**声明并初始化为 NULL：
   * 旧写法把它们声明在 pages_alloc 之后，于是“KV 池耗尽”那条路上 fail: 里会
   * sgl_xfree() 三个未初始化的指针（未定义行为 / 堆损坏）。Clang 的
   * -Wsometimes-uninitialized 抓住的就是这个（GCC 没报）。sgl_xfree 对 NULL 安全
   * （POSIX 版判空；MSVC 的 _aligned_free(NULL) 也是 no-op）。*/
  float *dl = NULL;
  float *tl = NULL;
  float *vl = NULL;
  if (!tp || !dp) { /* 主机内存不足：页表根本没填，不能走 pages_unref */
    SGL_ERROR("spec: out of host memory (page tables)");
    sgl_xfree(seq);
    sgl_xfree(tp);
    sgl_xfree(dp);
    return -1;
  }
  if (sgl_llama_pages_alloc(target, tp, cap) != 0 ||
      sgl_llama_pages_alloc(draft, dp, cap) != 0) {
    SGL_ERROR("spec: pool exhausted");
    goto fail;
  }
  dl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  tl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  vl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * (size_t)K); /* 批量 verify：[K][V] */
  if (!dl || !tl || !vl) {
    SGL_ERROR("spec: out of host memory");
    goto fail;
  }
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

/* ===================== temperature>0：修正采样（speculative sampling） ====================
 *
 * 与 greedy 版同一套 KV/轮次结构（上面的 sgl_spec_generate），只换「草稿怎么出、
 * 验证怎么判」：
 *   草稿：从 q̃ = sgl_dist_transform(dl) 直接抽（抽牌顺序与 sgl_sample_ex 不同
 *         没关系——q̃ 记录的是真实提议分布，修正数学只依赖它）；
 *   验证：target 各候选位构建 p̃，u < min(1, p̃(x)/q̃(x)) 接受；
 *   拒绝：从 norm((p̃-q̃)₊) 重采样（数学上：产出 = min(p̃,q̃) + (p̃-q̃)₊ = p̃）；
 *   全接受：从末位 p̃ 直抽 bonus。
 * 产出 token 的边缘分布 == 直接从 target 抽样 ⇒ 分布对拍是验收依据
 * （test_spec 的 TV 自校准检验），不像 greedy 版能逐 token 相等。
 * penalty 的 history：草稿/验证各位置用「已确认 token + 到该位置为止的候选」
 * （草稿先写进 seq 作草稿区，接受时覆写、拒绝后面自然丢弃），与非投机路径
 * 逐步采样时可见的历史完全一致。 */
static int draw_from_dist(const float *d, int n, uint64_t *rng) {
  double mass = 0;
  for (int i = 0; i < n; i++)
    if (d[i] > 0) mass += d[i];
  if (mass <= 0) return argmax_v(d, n);
  double r = sgl_rng_uniform(rng) * mass, acc = 0;
  for (int i = 0; i < n; i++) {
    if (d[i] <= 0) continue;
    acc += d[i];
    if (r < acc) return i;
  }
  return argmax_v(d, n); /* 浮点舍入見底 */
}

int sgl_spec_sample_generate(sgl_llama *target, sgl_llama *draft,
                             const int *prompt, int n_prompt, int max_new,
                             int k_spec, int eos_id,
                             const sgl_sampling_params *sp, uint64_t *rng,
                             int **out_ids, int *out_drafted, int *out_accepted) {
  if (out_drafted) *out_drafted = -1;
  if (out_accepted) *out_accepted = -1;
  if (!sp || sp->temperature <= 0.0f)
    return sgl_spec_generate(target, draft, prompt, n_prompt, max_new,
                             k_spec, eos_id, out_ids);
  if (!rng || !out_ids || !target || !draft || !prompt) return -1;
  int K = k_spec < 1 ? 1 : k_spec > 8 ? 8 : k_spec;
  int cap = n_prompt + max_new + K + 8;
  int *seq = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap);
  memcpy(seq, prompt, sizeof(int) * (size_t)n_prompt);
  int len = n_prompt;
  int t_kv = 0, d_kv = 0;

  int *tp = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap);
  int *dp = (int *)sgl_xmalloc(sizeof(int) * (size_t)cap);
  int V = target->vocab;
  float *dl = NULL, *tl = NULL, *vl = NULL;
  float *qd = NULL, *pd = NULL; /* q̃ 逐候选存档 [K][V]；p̃ 只需当前位 [V] */
  if (!tp || !dp) {
    SGL_ERROR("spec: out of host memory (page tables)");
    sgl_xfree(seq);
    sgl_xfree(tp);
    sgl_xfree(dp);
    return -1;
  }
  if (sgl_llama_pages_alloc(target, tp, cap) != 0 ||
      sgl_llama_pages_alloc(draft, dp, cap) != 0) {
    SGL_ERROR("spec: pool exhausted");
    goto fail;
  }
  dl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  tl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  vl = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * (size_t)K); /* [K][V] */
  qd = (float *)sgl_xmalloc(sizeof(float) * (size_t)V * (size_t)K);
  pd = (float *)sgl_xmalloc(sizeof(float) * (size_t)V);
  if (!dl || !tl || !vl || !qd || !pd) {
    SGL_ERROR("spec: out of host memory");
    goto fail;
  }
  int produced = 0, n_accepts = 0, n_drafted = 0;

  if (kv_catch_up(target, seq, n_prompt, &t_kv, tp, tl) != 0) goto fail;
  if (kv_catch_up(draft, seq, n_prompt, &d_kv, dp, dl) != 0) goto fail;

  while (produced < max_new) {
    /* 1) draft 连抽 K 候选（写 draft KV）；候选同时写进 seq 草稿区，
     *    供后续 penalty 的 history 引用 */
    int cand[8], nk = 0;
    for (int i = 0; i < K && produced + nk < max_new; i++) {
      float *q = qd + (size_t)nk * V;
      sgl_dist_transform(dl, V, sp, seq + n_prompt, len - n_prompt + nk, q);
      int t = draw_from_dist(q, V, rng);
      cand[nk] = t;
      seq[len + nk] = t;
      nk++;
      if (sgl_llama_forward(draft, &t, 1, d_kv, dp, dl) != 0) goto fail;
      d_kv++;
    }
    n_drafted += nk;

    /* 2) target 一次 n-token 批量前向拿全部候选位 logits，逐位拒绝检验 */
    int accepted = 0, reject_tok = -1;
    if (sgl_llama_forward_n(target, cand, nk, t_kv, tp, vl) != 0) goto fail;
    t_kv += nk;
    for (int i = 0; i < nk; i++) {
      const float *prow = (i == 0) ? tl : vl + (size_t)(i - 1) * V;
      sgl_dist_transform(prow, V, sp, seq + n_prompt, len - n_prompt + i, pd);
      const float *q = qd + (size_t)i * V;
      int x = cand[i];
      double u = sgl_rng_uniform(rng);
      if (pd[x] > 0.0f && u * (double)q[x] < (double)pd[x]) {
        accepted++;
        n_accepts++;
      } else {
        for (int j = 0; j < V; j++) {
          float d = pd[j] - q[j];
          pd[j] = d > 0.0f ? d : 0.0f;
        }
        reject_tok = draw_from_dist(pd, V, rng);
        break;
      }
    }

    /* 3) 写入确认 token（拒绝时采纳重采样；全接受 bonus=从末位 p̃ 直抽）*/
    for (int i = 0; i < accepted; i++) seq[len + i] = cand[i];
    int n_new = accepted;
    if (reject_tok >= 0) {
      seq[len + accepted] = reject_tok;
      n_new++;
    } else if (produced + nk < max_new) {
      sgl_dist_transform(vl + (size_t)(nk - 1) * V, V, sp,
                         seq + n_prompt, len - n_prompt + nk, pd);
      seq[len + nk] = draw_from_dist(pd, V, rng);
      n_new++;
    }
    len += n_new;
    produced += n_new;

    /* 4) 作废候选 KV 回退计数（同 greedy 版：不释放页，只回退有效长度） */
    {
      int valid = (n_new > accepted) ? len - 1 : len;
      t_kv = valid;
      d_kv = valid;
    }

    /* 5) eos：截去其后的本轮 token（同 greedy 版） */
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
    /* 6) 两模型 KV 补齐到新 len（同 greedy 版） */
    if (produced < max_new) {
      if (kv_catch_up(target, seq, len, &t_kv, tp, tl) != 0) goto fail;
      if (kv_catch_up(draft, seq, len, &d_kv, dp, dl) != 0) goto fail;
    }
  }

  if (n_drafted > 0)
    SGL_INFO("spec(sample): drafted=%d accepted=%d（%.0f%%）", n_drafted,
             n_accepts, 100.0 * n_accepts / n_drafted);
  *out_ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)(len - n_prompt));
  memcpy(*out_ids, seq + n_prompt, sizeof(int) * (size_t)(len - n_prompt));
  if (out_drafted) *out_drafted = n_drafted;
  if (out_accepted) *out_accepted = n_accepts;
  sgl_llama_pages_unref(target, tp, cap);
  sgl_llama_pages_unref(draft, dp, cap);
  sgl_xfree(seq); sgl_xfree(tp); sgl_xfree(dp); sgl_xfree(dl); sgl_xfree(tl);
  sgl_xfree(vl); sgl_xfree(qd); sgl_xfree(pd);
  return len - n_prompt;

fail:
  if (tp) sgl_llama_pages_unref(target, tp, cap);
  if (dp) sgl_llama_pages_unref(draft, dp, cap);
  sgl_xfree(seq); sgl_xfree(tp); sgl_xfree(dp); sgl_xfree(dl); sgl_xfree(tl);
  sgl_xfree(vl); sgl_xfree(qd); sgl_xfree(pd);
  return -1;
}
