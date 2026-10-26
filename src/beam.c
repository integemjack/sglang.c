/* sgl beam search（llama 后端；对应 sglang srt/beam_search 的最小正确版）
 * W 条 beam，逐步保留累计 logprob 最高的 W 个候选；eos 候选直接终束并记录；
 * 分叉时复制源 beam 的 KV 页（每步全量复制，W≤8、CPU 可接受）。 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/llama.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/log.h"

typedef struct {
  int *seq;      /* prompt + 已生成 */
  int len;       /* 总长（含 prompt）*/
  int cap;
  int *pages;    /* KV 页表 */
  int n_pages;   /* 实际持有的页数（unref 按此数，绝不能按 cap_pages：未分配槽是垃圾值）*/
  int cap_pages;
  double lp;     /* 累计 log P(generated) */
  int alive;
} sgl_beam;

static double logprob_of(const float *lg, int vocab, int tok) {
  float mx = lg[0];
  for (int i = 1; i < vocab; i++) if (lg[i] > mx) mx = lg[i];
  double s = 0;
  for (int i = 0; i < vocab; i++) s += exp((double)lg[i] - mx);
  return (double)lg[tok] - mx - log(s);
}

/* 把 src 前 n 页复制到新分配页，写进 dst_pages */
static void kv_copy_pages(sgl_llama *m, int *dst_pages, const int *src_pages, int n) {
  int row = m->kv_dim;
  for (int p = 0; p < n; p++) {
    int np;
    if (sgl_llama_pages_alloc(m, &np, 1) != 0) {
      SGL_ERROR("beam: kv pool exhausted at page %d/%d", p, n);
      np = 0;
    }
    dst_pages[p] = np;
    if (m->kv8) {
      memcpy(m->kc8 + (size_t)np * row, m->kc8 + (size_t)src_pages[p] * row, (size_t)row);
      memcpy(m->vc8 + (size_t)np * row, m->vc8 + (size_t)src_pages[p] * row, (size_t)row);
      m->kcs[np] = m->kcs[src_pages[p]];
      m->vcs[np] = m->vcs[src_pages[p]];
    } else {
      for (int l = 0; l < m->n_layer; l++) {
        size_t base = (size_t)l * m->n_pool * m->kv_dim;
        memcpy(m->kc + base + (size_t)np * row, m->kc + base + (size_t)src_pages[p] * row,
               (size_t)row * sizeof(float));
        memcpy(m->vc + base + (size_t)np * row, m->vc + base + (size_t)src_pages[p] * row,
               (size_t)row * sizeof(float));
      }
    }
  }
}

int sgl_llama_beam_search(sgl_llama *m, const int *prompt, int n_prompt,
                          int beam_width, int max_new, int eos_id,
                          sgl_llama_beam_result *out) {
  int W = beam_width < 1 ? 1 : beam_width > 8 ? 8 : beam_width;
  sgl_beam *bm = (sgl_beam *)sgl_xcalloc(W, sizeof(sgl_beam));
  for (int i = 0; i < W; i++) {
    bm[i].cap = n_prompt + max_new + 8;
    bm[i].seq = (int *)sgl_xmalloc(sizeof(int) * (size_t)bm[i].cap);
    bm[i].alive = 0;
  }
  bm[0].cap_pages = n_prompt + max_new + 8;
  bm[0].pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)bm[0].cap_pages);
  if (sgl_llama_pages_alloc(m, bm[0].pages, bm[0].cap_pages) != 0) {
    SGL_ERROR("beam: pool too small");
    goto fail;
  }
  memcpy(bm[0].seq, prompt, sizeof(int) * (size_t)n_prompt);
  bm[0].len = n_prompt;
  bm[0].n_pages = bm[0].cap_pages;
  bm[0].lp = 0;
  bm[0].alive = 1;

  float *logits = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->vocab);
  typedef struct { int b, tok, term; double lp; } cand;
  int C = W * W + 8;
  cand *cs = (cand *)sgl_xmalloc(sizeof(cand) * (size_t)C);
  /* 已终束结果（最多 W + 越界保护用 C）*/
  cand *done = (cand *)sgl_xmalloc(sizeof(cand) * (size_t)C);
  int n_done = 0;

  if (sgl_llama_forward(m, bm[0].seq, bm[0].len, 0, bm[0].pages, logits) != 0) goto fail;
  int started = 1; /* logits 已就绪（对应 bm[0] 最后一 token）*/

  for (int step = 0; step < max_new; step++) {
    int nc = 0;
    for (int bi = 0; bi < W; bi++) {
      if (!bm[bi].alive) continue;
      if (!started) {
        if (sgl_llama_forward(m, &bm[bi].seq[bm[bi].len - 1], 1, bm[bi].len - 1,
                              bm[bi].pages, logits) != 0) goto fail;
      }
      /* top-W token（选择排序 W 次）*/
      cand local[8];
      for (int r = 0; r < W; r++) local[r].tok = -1;
      for (int r = 0; r < W; r++) {
        int best = -1;
        for (int i = 0; i < m->vocab; i++) {
          int dup = 0;
          for (int q = 0; q < r; q++) if (local[q].tok == i) { dup = 1; break; }
          if (dup) continue;
          if (best < 0 || logits[i] > logits[best]) best = i;
        }
        if (best < 0) break;
        local[r].tok = best;
      }
      for (int r = 0; r < W; r++) {
        if (local[r].tok < 0 || nc >= C) break;
        cs[nc].b = bi;
        cs[nc].tok = local[r].tok;
        cs[nc].lp = bm[bi].lp + logprob_of(logits, m->vocab, local[r].tok);
        cs[nc].term = (local[r].tok == eos_id);
        nc++;
      }
      started = 0; /* 后续 beam 都要自己前向 */
    }
    if (nc == 0) break;

    for (int i = 0; i < nc - 1; i++)
      for (int j = i + 1; j < nc; j++)
        if (cs[j].lp > cs[i].lp) { cand t = cs[i]; cs[i] = cs[j]; cs[j] = t; }

    /* 旧 beam 快照（复制源），随后重建 */
    sgl_beam *old = (sgl_beam *)sgl_xmalloc(sizeof(sgl_beam) * (size_t)W);
    memcpy(old, bm, sizeof(sgl_beam) * (size_t)W);
    for (int i = 0; i < W; i++) { bm[i].alive = 0; bm[i].pages = NULL; bm[i].n_pages = 0; bm[i].len = 0; bm[i].lp = 0; }

    int slot = 0;
    for (int r = 0; r < nc && (slot < W || n_done < W); r++) {
      cand *c = &cs[r];
      if (c->term) {
        if (n_done < W) { done[n_done] = *c; n_done++; }
      } else if (slot < W) {
        sgl_beam *src = &old[c->b];
        bm[slot].cap_pages = src->len + 2 + (max_new - step);
        bm[slot].pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)bm[slot].cap_pages);
        kv_copy_pages(m, bm[slot].pages, src->pages, src->len);
        /* 新 token 落在位置 src->len：预先分配写页（下一轮 forward 要用）*/
        int npg;
        if (sgl_llama_pages_alloc(m, &npg, 1) == 0)
          bm[slot].pages[src->len] = npg;
        else { SGL_ERROR("beam: no page for new token"); }
        bm[slot].n_pages = src->len + 1;
        memcpy(bm[slot].seq, src->seq, sizeof(int) * (size_t)src->len);
        bm[slot].len = src->len;
        bm[slot].seq[bm[slot].len++] = c->tok;
        bm[slot].lp = c->lp;
        bm[slot].alive = 1;
        slot++;
      }
    }
    /* 释放旧 beam 的页 */
    for (int i = 0; i < W; i++)
      if (old[i].pages) { sgl_llama_pages_unref(m, old[i].pages, old[i].n_pages); sgl_xfree(old[i].pages); }
    sgl_xfree(old);

    if (slot == 0) break; /* 全部终束 */
    /* 下一轮各 beam 需各自前向（started=0）*/
  }

  /* 收尾：存活 beam 按 lp 降序输出 ids（生成部分）。
   * 终束候选（eos）仅用于提前断流：全部终束时 slot==0 退出循环。 */
  {
    typedef struct { double lp; const int *seq; int len; } row;
    row rows[16];
    int nr = 0;
    for (int i = 0; i < W && nr < 16; i++) {
      if (!bm[i].alive || bm[i].len <= n_prompt) continue;
      rows[nr].lp = bm[i].lp;
      rows[nr].seq = bm[i].seq;
      rows[nr].len = bm[i].len;
      nr++;
    }
    for (int i = 0; i < nr - 1; i++)
      for (int j = i + 1; j < nr; j++)
        if (rows[j].lp > rows[i].lp) { row t = rows[i]; rows[i] = rows[j]; rows[j] = t; }
    for (int i = 0; i < nr; i++) {
      out[i].n = rows[i].len - n_prompt;
      out[i].ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)(out[i].n > 0 ? out[i].n : 1));
      memcpy(out[i].ids, rows[i].seq + n_prompt, sizeof(int) * (size_t)out[i].n);
      out[i].logprob = rows[i].lp;
    }
    for (int i = 0; i < W; i++) {
      if (bm[i].pages) { sgl_llama_pages_unref(m, bm[i].pages, bm[i].n_pages); sgl_xfree(bm[i].pages); }
      sgl_xfree(bm[i].seq);
    }
    sgl_xfree(bm);
    sgl_xfree(cs);
    sgl_xfree(done);
    sgl_xfree(logits);
    return nr;
  }

fail:
  for (int i = 0; i < W; i++) {
    if (bm[i].pages) { sgl_llama_pages_unref(m, bm[i].pages, bm[i].n_pages); sgl_xfree(bm[i].pages); }
    sgl_xfree(bm[i].seq);
  }
  sgl_xfree(bm);
  sgl_xfree(cs);
  sgl_xfree(done);
  sgl_xfree(logits);
  return -1;
}
