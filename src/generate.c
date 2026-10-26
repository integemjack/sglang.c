/* sgl generate 实现（单请求，直接走分页 KV 池；无 radix）*/
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/generate.h"

#include <stdio.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/log.h"

int sgl_generate(sgl_gpt2 *m, const sgl_tokenizer *tok, const int *prompt_ids,
                 int prompt_n, const sgl_sampling_params *sp, int max_new,
                 sgl_gen_callback cb, void *user) {
  if (prompt_n <= 0) { SGL_ERROR("empty prompt"); return -1; }
  if (prompt_n + max_new > m->n_ctx) {
    SGL_ERROR("prompt(%d)+max_new(%d) > ctx(%d)", prompt_n, max_new, m->n_ctx);
    return -1;
  }

  float *logits = (float *)sgl_xmalloc(sizeof(float) * (size_t)m->vocab);
  uint64_t rng = sp->seed ? sp->seed : 0x9E3779B97F4A7C15ULL;
  int eos = sgl_tokenizer_eos_id(tok);
  int produced = 0;

  /* 分配本序列的页表并 prefill（整段 prompt 一次前向）*/
  int need = prompt_n + max_new + 1;
  int *pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)need);
  if (sgl_gpt2_pages_alloc(m, pages, need) != 0) {
    SGL_ERROR("kv pool exhausted");
    sgl_xfree(pages); sgl_xfree(logits);
    return -1;
  }
  int kv_len = 0;

  if (sgl_gpt2_prefill(m, prompt_ids, prompt_n, 0, pages, logits) != 0) {
    sgl_gpt2_pages_unref(m, pages, need);
    sgl_xfree(pages); sgl_xfree(logits);
    return -1;
  }
  kv_len = prompt_n;

  for (int step = 0; step < max_new; step++) {
    int next_tok = sgl_sample(logits, m->vocab, sp, &rng);
    if (next_tok == eos) break;

    char *piece = sgl_tokenizer_decode(tok, &next_tok, 1);
    int stop = 0;
    if (cb) stop = cb(user, next_tok, piece ? piece : "");
    sgl_xfree(piece);
    produced++;
    if (stop) break;
    if (kv_len >= m->n_ctx) break;

    /* decode：单 token 增量前向 */
    if (sgl_gpt2_prefill(m, &next_tok, 1, kv_len, pages, logits) != 0) break;
    kv_len++;
  }
  sgl_gpt2_pages_unref(m, pages, need);
  sgl_xfree(pages);
  sgl_xfree(logits);
  return produced;
}
