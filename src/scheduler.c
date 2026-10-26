/* sgl scheduler 实现：连续批处理调度器 + RadixCache 前缀复用 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/scheduler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/log.h"
#include "sgl/model.h"
#include "sgl/radix.h"

typedef enum { SGL_REQ_WAITING = 0, SGL_REQ_RUNNING, SGL_REQ_DONE } sgl_req_state;

/* 多 LoRA 实例池的一个实例：base（insts[0]，name=NULL）或某个 adapter 的
 * 「base+合并 delta」完整副本（独立 KV 池 + 独立 RadixCache）*/
typedef struct {
  char *name;      /* NULL = base；否则路由名 */
  sgl_model *model;
  sgl_radix *radix;
} sgl_inst;

typedef struct {
  int id;
  sgl_req_state state;
  int *prompt;
  int n_prompt;
  int *gen;       /* 已生成 token（不含 eos）*/
  int n_gen, cap_gen;
  int max_new;
  sgl_sampling_params sp;
  uint64_t rng;
  int *pages;     /* 页表：位置 p → KV 页号；容量 cap_pages */
  int cap_pages;
  int hit_len;    /* RadixCache 命中的前缀长度 */
  int kv_len;     /* 已写入 KV 的长度 = n_prompt + n_gen */
  int n_streamed; /* 已被流式取走的生成 token 数（≤ n_gen）*/
  int streaming;  /* 流式请求：poll 不回收（由 stream 结束时 reclaim）*/
  int inst;       /* 路由到的实例下标（0=base，≥1=lora）*/
} sgl_req;

struct sgl_scheduler {
  sgl_inst *insts;   /* insts[0] = base */
  int n_insts, cap_insts;
  sgl_tokenizer *tok;
  int eos_id;
  sgl_req *reqs;
  int n_reqs, cap_reqs;
  int next_id;
  /* step 用的批缓冲 */
  int *b_tokens, *b_lens;
  int **b_pages;
  float *b_logits; /* [SGL_GPT2_MAX_BATCH][vocab] */
  sgl_sched_stats stats;
};

static uint64_t mix64(uint64_t x) {
  x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
  x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL;
  x ^= x >> 33;
  return x;
}

static char *path_join(const char *dir, const char *name) {
  size_t n = strlen(dir) + strlen(name) + 2;
  char *p = (char *)sgl_xmalloc(n);
  snprintf(p, n, "%s/%s", dir, name);
  return p;
}

sgl_scheduler *sgl_sched_new(const char *model_dir, uint64_t seed) {
  sgl_scheduler *s = (sgl_scheduler *)sgl_xcalloc(1, sizeof(*s));
  s->cap_insts = 4;
  s->insts = (sgl_inst *)sgl_xmalloc(sizeof(sgl_inst) * (size_t)s->cap_insts);
  memset(s->insts, 0, sizeof(sgl_inst) * (size_t)s->cap_insts);
  s->n_insts = 1;
  s->insts[0].model = sgl_model_load(model_dir);
  if (!s->insts[0].model) { sgl_xfree(s->insts); sgl_xfree(s); return NULL; }
  char *tpath = path_join(model_dir, "tokenizer.json");
  s->tok = sgl_tokenizer_load_dir(model_dir);
  sgl_xfree(tpath);
  if (!s->tok) { sgl_model_free(s->insts[0].model); sgl_xfree(s->insts); sgl_xfree(s); return NULL; }
  s->insts[0].radix = sgl_radix_new(s->insts[0].model);
  s->eos_id = sgl_tokenizer_eos_id(s->tok);
  s->next_id = 1;
  s->cap_reqs = 16;
  s->reqs = (sgl_req *)sgl_xmalloc(sizeof(sgl_req) * s->cap_reqs);
  int mb = SGL_GPT2_MAX_BATCH;
  s->b_tokens = (int *)sgl_xmalloc(sizeof(int) * mb);
  s->b_pages = (int **)sgl_xmalloc(sizeof(int *) * mb);
  s->b_lens = (int *)sgl_xmalloc(sizeof(int) * mb);
  s->b_logits = (float *)sgl_xmalloc(sizeof(float) * mb * sgl_model_vocab(s->insts[0].model));
  if (seed == 0) seed = 0x853C49E6748FEA9BULL;
  (void)seed; /* 各请求的 rng 由 submit 时按 id 派生 */
  return s;
}

void sgl_sched_free(sgl_scheduler *s) {
  if (!s) return;
  for (int i = 0; i < s->n_reqs; i++) {
    sgl_xfree(s->reqs[i].prompt);
    sgl_xfree(s->reqs[i].gen);
    sgl_xfree(s->reqs[i].pages);
  }
  sgl_xfree(s->reqs);
  sgl_xfree(s->b_tokens); sgl_xfree(s->b_pages);
  sgl_xfree(s->b_lens); sgl_xfree(s->b_logits);
  for (int i = 0; i < s->n_insts; i++) {
    sgl_radix_free(s->insts[i].radix);
    sgl_model_free(s->insts[i].model);
    sgl_xfree(s->insts[i].name);
  }
  sgl_xfree(s->insts);
  sgl_tokenizer_free(s->tok);
  sgl_xfree(s);
}

static sgl_req *req_new(sgl_scheduler *s, const char *text, int max_new,
                        const sgl_sampling_params *sp, int inst) {
  if (s->n_reqs == s->cap_reqs) {
    s->cap_reqs *= 2;
    s->reqs = (sgl_req *)sgl_xrealloc(s->reqs, sizeof(sgl_req) * s->cap_reqs);
  }
  sgl_req *r = &s->reqs[s->n_reqs++];
  memset(r, 0, sizeof(*r));
  r->id = s->next_id++;
  r->state = SGL_REQ_WAITING;
  r->prompt = sgl_tokenizer_encode(s->tok, text, strlen(text), &r->n_prompt);
  if (!r->prompt || r->n_prompt <= 0) { r->prompt = NULL; return NULL; }
  r->max_new = max_new > 0 ? max_new : 16;
  r->sp = sp ? *sp : (sgl_sampling_params){0};
  r->rng = mix64((uint64_t)r->id * 0x9E3779B97F4A7C15ULL + 1);
  r->inst = inst;
  r->cap_gen = r->max_new + 1;
  r->gen = (int *)sgl_xmalloc(sizeof(int) * (size_t)r->cap_gen);
  return r;
}

int sgl_sched_submit(sgl_scheduler *s, const char *text, int max_new,
                     const sgl_sampling_params *sp) {
  return sgl_sched_submit_lora(s, text, max_new, sp, NULL);
}

/* 请求完成：整段序列（prompt+生成）入本实例的 RadixCache，释放本请求的页引用 */
static void req_finish(sgl_scheduler *s, sgl_req *r) {
  sgl_inst *in = &s->insts[r->inst];
  if (r->pages) {
    int full_len = r->n_prompt + (r->n_gen > 0 ? r->n_gen : 0);
    if (r->n_gen > 0 && full_len > r->hit_len) {
      /* 拼完整 token 序列入树（生成部分独有，必是新分支）*/
      int *full = (int *)sgl_xmalloc(sizeof(int) * (size_t)full_len);
      memcpy(full, r->prompt, sizeof(int) * (size_t)r->n_prompt);
      memcpy(full + r->n_prompt, r->gen, sizeof(int) * (size_t)r->n_gen);
      sgl_radix_insert(in->radix, full, r->pages, full_len);
      sgl_xfree(full);
    }
    /* 释放本请求的全部页引用（命中借用的 + 新分配的 + 未用尾部的）*/
    sgl_model_pages_unref(in->model, r->pages, r->cap_pages);
    sgl_xfree(r->pages);
    r->pages = NULL;
  }
  r->state = SGL_REQ_DONE;
  s->stats.n_finished++;
}

/* 采样并把 token 追加进 r->gen；返回 1 表示请求应结束（eos/max_new/满）*/
static int req_sample_append(sgl_scheduler *s, sgl_req *r, const float *logits) {
  sgl_model *m = s->insts[r->inst].model;
  int tok = sgl_sample(logits, sgl_model_vocab(m), &r->sp, &r->rng);
  if (tok == s->eos_id) return 1; /* eos：不追加 */
  r->gen[r->n_gen++] = tok;
  s->stats.decode_tokens++;
  if (r->n_gen >= r->max_new) return 1;
  if (r->kv_len + 1 >= sgl_model_ctx(m)) return 1; /* ctx 满 */
  return 0;
}

/* prefill 一个等待请求：RadixCache 命中部分借用页，仅对剩余后缀计算 */
static int step_prefill(sgl_scheduler *s, sgl_req *r) {
  sgl_inst *in = &s->insts[r->inst];
  sgl_model *m = in->model;
  int need_total = r->n_prompt + r->max_new + 1;
  r->cap_pages = need_total;
  r->pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)need_total);

  /* 1) 前缀匹配：命中的页借用（引用 +1，防驱逐）*/
  int matched = sgl_radix_match(in->radix, r->prompt, r->n_prompt, r->pages);
  r->hit_len = matched;
  for (int i = 0; i < matched; i++) sgl_model_page_ref(m, r->pages[i], +1);
  s->stats.cache_hit_tokens += matched;

  /* 2) 分配后缀 + 生成所需的页；不够先驱逐（借用的页不会被驱逐），
   *    仍不够则本次不推进（请求继续等待，等运行中的释放）*/
  int need_alloc = need_total - matched;
  if (sgl_model_pages_alloc(m, r->pages + matched, need_alloc) != 0) {
    sgl_radix_evict(in->radix, need_alloc - sgl_model_nfree(m));
    if (sgl_model_pages_alloc(m, r->pages + matched, need_alloc) != 0) {
      sgl_model_pages_unref(m, r->pages, matched); /* 归还借用 */
      sgl_xfree(r->pages); r->pages = NULL;
      return 1; /* 稍后重试 */
    }
  }

  /* 3) 只对未命中后缀前向；全命中时补最后一个 token 取 logits */
  int from = matched < r->n_prompt ? matched : r->n_prompt - 1;
  float *lg = s->b_logits;
  int n_suf = r->n_prompt - from;
  if (sgl_model_prefill(m, r->prompt + from, n_suf, from, r->pages, lg) != 0) return -1;
  s->stats.prefill_tokens += n_suf;
  r->kv_len = r->n_prompt;
  r->state = SGL_REQ_RUNNING;
  if (req_sample_append(s, r, lg)) req_finish(s, r);
  return 0;
}

int sgl_sched_step(sgl_scheduler *s) {
  /* 1) prefill 优先：取最早的等待请求（任一实例；池不足时保持等待，下步重试）*/
  for (int i = 0; i < s->n_reqs; i++) {
    sgl_req *r = &s->reqs[i];
    if (r->state != SGL_REQ_WAITING) continue;
    int rc = step_prefill(s, r);
    if (rc < 0) { r->state = SGL_REQ_DONE; r->n_gen = -1; } /* 前向致命错 */
    return 0;
  }

  /* 2) 批量 decode：逐实例把该实例运行中的请求组成一批各自前向
   *    （不同实例的权重不同，不能混入同一 batch）*/
  int finished = 0;
  for (int t = 0; t < s->n_insts; t++) {
    sgl_model *m = s->insts[t].model;
    int B = 0;
    for (int i = 0; i < s->n_reqs && B < SGL_GPT2_MAX_BATCH; i++) {
      sgl_req *r = &s->reqs[i];
      if (r->state != SGL_REQ_RUNNING || r->inst != t) continue;
      s->b_tokens[B] = r->gen[r->n_gen - 1]; /* 最后生成的 token 尚未入 KV */
      s->b_pages[B] = r->pages;
      s->b_lens[B] = r->kv_len;
      B++;
    }
    if (B == 0) continue;
    if (sgl_model_decode_batch(m, s->b_tokens, s->b_pages, s->b_lens, B, s->b_logits) != 0)
      return -1;

    int bi = 0;
    for (int i = 0; i < s->n_reqs; i++) {
      sgl_req *r = &s->reqs[i];
      if (r->state != SGL_REQ_RUNNING || r->inst != t) continue;
      r->kv_len++; /* 刚 decode 的 token 已写入 KV */
      if (req_sample_append(s, r, s->b_logits + (size_t)bi * sgl_model_vocab(m))) {
        req_finish(s, r);
        finished++;
      }
      bi++;
    }
  }
  return finished;
}

int sgl_sched_poll(sgl_scheduler *s, int *req_id, char **text, int **ids,
                   int *n_ids, int *n_prompt, int *n_hit) {
  for (int i = 0; i < s->n_reqs; i++) {
    sgl_req *r = &s->reqs[i];
    if (r->state != SGL_REQ_DONE) continue;
    if (r->n_gen == -1) continue; /* 失败或已取走 */
    if (r->streaming) continue;   /* 流式请求由 reclaim 回收 */
    if (req_id) *req_id = r->id;
    if (n_ids) *n_ids = r->n_gen;
    if (n_prompt) *n_prompt = r->n_prompt;
    if (n_hit) *n_hit = r->hit_len;
    if (ids) {
      *ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)(r->n_gen > 0 ? r->n_gen : 1));
      memcpy(*ids, r->gen, sizeof(int) * (size_t)r->n_gen);
    }
    if (text) {
      if (r->n_gen > 0) {
        *text = sgl_tokenizer_decode(s->tok, r->gen, r->n_gen);
      } else {
        *text = (char *)sgl_xmalloc(1);
        (*text)[0] = 0;
      }
    }
    r->n_gen = -1; /* 已取走标记 */
    return 1;
  }
  return 0;
}

int sgl_sched_nactive(sgl_scheduler *s) {
  int n = 0;
  for (int i = 0; i < s->n_reqs; i++)
    if (s->reqs[i].state == SGL_REQ_WAITING || s->reqs[i].state == SGL_REQ_RUNNING) n++;
  return n;
}

int sgl_sched_submit_stream(sgl_scheduler *s, const char *text, int max_new,
                            const sgl_sampling_params *sp) {
  return sgl_sched_submit_stream_lora(s, text, max_new, sp, NULL);
}

void sgl_sched_reclaim_stream(sgl_scheduler *s, int req_id) {
  for (int i = 0; i < s->n_reqs; i++) {
    sgl_req *r = &s->reqs[i];
    if (r->id == req_id) {
      r->n_gen = -1;
      r->streaming = 0;
      return;
    }
  }
}

/* ---- 流式 ---- */
sgl_stream_ev sgl_sched_stream_next(sgl_scheduler *s, int req_id,
                                    char **piece, int *tok_id) {
  for (int i = 0; i < s->n_reqs; i++) {
    sgl_req *r = &s->reqs[i];
    if (r->id != req_id) continue;
    if (r->n_gen < 0) return SGL_STREAM_DONE; /* 失败请求 */
    if (r->n_gen > r->n_streamed) {
      int t = r->gen[r->n_streamed];
      r->n_streamed++;
      if (piece) *piece = sgl_tokenizer_decode(s->tok, &t, 1);
      if (tok_id) *tok_id = t;
      return SGL_STREAM_TOKEN;
    }
    return r->state == SGL_REQ_DONE ? SGL_STREAM_DONE : SGL_STREAM_MORE;
  }
  return SGL_STREAM_DONE; /* 不存在（已回收）：当作结束 */
}

sgl_sched_stats sgl_sched_get_stats(const sgl_scheduler *s) {
  sgl_sched_stats st = s->stats;
  st.n_running = st.n_waiting = 0;
  st.pool_pages = sgl_model_pool_pages(s->insts[0].model);
  for (int i = 0; i < s->n_reqs; i++) {
    if (s->reqs[i].state == SGL_REQ_RUNNING) st.n_running++;
    else if (s->reqs[i].state == SGL_REQ_WAITING) st.n_waiting++;
  }
  return st;
}

/* ---- 多 LoRA 实例池（方案 B）---- */

#ifdef _WIN32
#define sgl_setenv(k, v) _putenv_s(k, v)
#else
#define sgl_setenv(k, v) setenv(k, v, 1)
#endif

int sgl_sched_find_lora(const sgl_scheduler *s, const char *name) {
  if (!s) return -1;
  if (!name || !*name) return 0; /* base */
  for (int i = 1; i < s->n_insts; i++)
    if (s->insts[i].name && strcmp(s->insts[i].name, name) == 0) return i;
  return -1;
}

int sgl_sched_add_lora(sgl_scheduler *s, const char *name, const char *model_dir,
                       const char *lora_dir) {
  if (!s || !name || !*name || !model_dir || !lora_dir || !*lora_dir) return -1;
  if (sgl_sched_find_lora(s, name) >= 0) {
    SGL_ERROR("add_lora: duplicate name '%s'", name);
    return -1;
  }
  /* 临时设 SGL_LORA 复用现有加载+合并路径（llama_load 内合并，量化前），
   * 完成后恢复原环境，避免影响后续行为 */
  const char *old = getenv("SGL_LORA");
  char *old_copy = (old && *old) ? (char *)sgl_xmalloc(strlen(old) + 1) : NULL;
  if (old_copy) strcpy(old_copy, old);
  sgl_setenv("SGL_LORA", lora_dir);
  sgl_model *m = sgl_model_load(model_dir);
  if (old_copy) {
    sgl_setenv("SGL_LORA", old_copy);
    sgl_xfree(old_copy);
  } else {
    sgl_setenv("SGL_LORA", "");
  }
  if (!m) { SGL_ERROR("add_lora '%s': model load failed", name); return -1; }
  sgl_radix *rx = sgl_radix_new(m);
  if (!rx) { sgl_model_free(m); return -1; }
  if (s->n_insts == s->cap_insts) {
    s->cap_insts *= 2;
    s->insts = (sgl_inst *)sgl_xrealloc(s->insts, sizeof(sgl_inst) * (size_t)s->cap_insts);
  }
  sgl_inst *in = &s->insts[s->n_insts++];
  in->name = (char *)sgl_xmalloc(strlen(name) + 1);
  strcpy(in->name, name);
  in->model = m;
  in->radix = rx;
  SGL_INFO("lora instance '%s' ready (adapter=%s, pool=%d pages)", name, lora_dir,
           sgl_model_pool_pages(m));
  return s->n_insts - 1;
}

static int submit_impl(sgl_scheduler *s, const char *text, int max_new,
                       const sgl_sampling_params *sp, const char *lora, int streaming) {
  int inst = sgl_sched_find_lora(s, lora);
  if (inst < 0) {
    SGL_ERROR("submit: unknown lora '%s'", lora ? lora : "");
    return -1;
  }
  sgl_req *r = req_new(s, text, max_new, sp, inst);
  if (!r) return -1;
  if (streaming) r->streaming = 1;
  int need = r->n_prompt + r->max_new + 1;
  if (need > sgl_model_pool_pages(s->insts[inst].model)) {
    SGL_ERROR("submit: need %d pages > pool %d (lora=%s)", need,
              sgl_model_pool_pages(s->insts[inst].model), lora ? lora : "base");
    r->state = SGL_REQ_DONE;
    r->n_gen = -1; /* 失败标记：poll 跳过 */
    return -1;
  }
  return r->id;
}

int sgl_sched_submit_lora(sgl_scheduler *s, const char *text, int max_new,
                          const sgl_sampling_params *sp, const char *lora) {
  return submit_impl(s, text, max_new, sp, lora, 0);
}

int sgl_sched_submit_stream_lora(sgl_scheduler *s, const char *text, int max_new,
                                 const sgl_sampling_params *sp, const char *lora) {
  return submit_impl(s, text, max_new, sp, lora, 1);
}
