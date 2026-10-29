/* sgl scheduler 实现：连续批处理调度器 + RadixCache 前缀复用 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/scheduler.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/log.h"
#include "sgl/grammar.h"
#include "sgl/model.h"
#include "sgl/radix.h"

typedef enum { SGL_REQ_WAITING = 0, SGL_REQ_RUNNING, SGL_REQ_DONE } sgl_req_state;

/* P1: grammar 词表字节缓存（1 MiB；超出部分截断）*/
#define SGL_VH_CAP (1 << 20)

/* 多 LoRA 实例池的一个实例：base（insts[0]，name=NULL）或某个 adapter 的
 * 「base+合并 delta」完整副本（独立 KV 池 + 独立 RadixCache）*/
typedef struct {
  char *name;      /* NULL = base 副本（含 DP 多副本）；否则路由名 */
  sgl_model *model;
  sgl_radix *radix;
  int dead;        /* 1 = 已被 /unload_lora_adapter 卸载（下标不复用，避开悬垂请求）*/
} sgl_inst;

typedef struct {
  int id;
  sgl_req_state state;
  int priority;    /* 优先级（大者先 prefill；默认 0）*/
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
  float *feats;   /* G8: 视觉 token embedding [n_img][E_lm]（sgl_xmalloc，prefill 后释放）*/
  int n_img;      /* G8: 视觉 token 数（0 = 纯文本）*/
  /* G21：多图。img_at[] 是所有 `<image>` 占位符的位置（按出现顺序），
   * n_place 是有几个占位符，per_img 是每张图占的视觉 token 数。*/
  int img_at[8];
  int n_place;
  int per_img;
  /* G12: `<image>` 占位符在 prompt 里的位置。>=0 → 视觉 token 插在这里（原占位符
   * 那个 token 丢掉，位置 = img_at），其后的文本 token 顺延；-1 → 旧行为（接在
   * prompt 末尾，仅在没有占位符 token 时用）。 */
  /* P0-3/P0-4: 采样 logprob 与 chunked prefill 游标 */
  float *tok_lp;  /* 每个生成 token 的 logprob */
  int n_gen_lp;
  sgl_grammar *gram; /* P1: 约束解码（NULL = 不约束）*/
  int gram_done;     /* 已到完整合法输出（可以提前结束）*/
  int *gram_ids;     /* [0..V) 候选 token id 缓存 */
  unsigned char *gram_ok; /* 掩码结果 */
  int gram_vocab;
  int gram_alloc;
  int prefill_pos; /* 已 prefill 到的 prompt 位置 */
  int prefilling;  /* 1 = 页已分配，正在分块 prefill */
} sgl_req;

struct sgl_scheduler {
  sgl_inst *insts;   /* insts[0] = base */
  int n_insts, cap_insts;
  sgl_tokenizer *tok;
  int eos_id;
  int img_tok;       /* G12: "<image>" 占位符的 token id（-1 = 词表里没有）*/
  sgl_req *reqs;
  int n_reqs, cap_reqs;
  int next_id;
  /* step 用的批缓冲 */
  int *b_tokens, *b_lens;
  int **b_pages;
  float *b_logits; /* [SGL_GPT2_MAX_BATCH][vocab] */
  int prefill_chunk; /* P0-4: 每步 prefill 的 token 上限（SGL_PREFILL_CHUNK）*/
  /* P1: grammar 词表缓存（token id → UTF-8 字节）*/
  char *vh_buf;
  const char **vh_ptr;
  int *vh_len;
  int vh_vocab;
  int vh_ready;
  int paused;      /* 1 = /pause_generation 生效中，step 不推进 */
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
  /* G12: llava 的 "<image>" 是词表里的 added token（单 token）；有它就能把视觉 token
   * 插到占位符位置，而不是老式的“接在 prompt 末尾”。*/
  s->img_tok = sgl_tokenizer_piece_id(s->tok, "<image>");
  s->next_id = 1;
  /* P1: grammar 词表（token id → UTF-8 字节）延后到第一次约束请求时构建 */
  s->vh_buf = (char *)sgl_xmalloc(SGL_VH_CAP);
  s->vh_ptr = (const char **)sgl_xmalloc(sizeof(char *) * (size_t)sgl_tokenizer_vocab_size(s->tok));
  s->vh_len = (int *)sgl_xmalloc(sizeof(int) * (size_t)sgl_tokenizer_vocab_size(s->tok));
  s->cap_reqs = 16;
  s->reqs = (sgl_req *)sgl_xmalloc(sizeof(sgl_req) * s->cap_reqs);
  int mb = SGL_GPT2_MAX_BATCH;
  s->b_tokens = (int *)sgl_xmalloc(sizeof(int) * mb);
  s->b_pages = (int **)sgl_xmalloc(sizeof(int *) * mb);
  s->b_lens = (int *)sgl_xmalloc(sizeof(int) * mb);
  s->b_logits = (float *)sgl_xmalloc(sizeof(float) * mb * sgl_model_vocab(s->insts[0].model));
  /* P0-4: chunked prefill 块大小（默认 512；0/非法 → 4096 = 近似整段）*/
  {
    const char *ck = getenv("SGL_PREFILL_CHUNK");
    int v = ck ? atoi(ck) : 512;
    s->prefill_chunk = v > 0 ? v : 4096;
  }
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
    sgl_xfree(s->reqs[i].feats); /* G8: 未走到 prefill 就结束的请求在这回收 */
    sgl_xfree(s->reqs[i].tok_lp); /* P0-3 */
    sgl_xfree(s->reqs[i].gram_ids);
    sgl_xfree(s->reqs[i].gram_ok);
    sgl_grammar_free(s->reqs[i].gram); /* P1 */
  }
  sgl_xfree(s->reqs);
  sgl_xfree(s->vh_buf); sgl_xfree(s->vh_ptr); sgl_xfree(s->vh_len);
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

static sgl_req *req_new(sgl_scheduler *s, const char *text, const int *ids, int n_ids, int max_new,
                        const sgl_sampling_params *sp, int inst, int priority) {
  if (s->n_reqs == s->cap_reqs) {
    s->cap_reqs *= 2;
    s->reqs = (sgl_req *)sgl_xrealloc(s->reqs, sizeof(sgl_req) * s->cap_reqs);
  }
  sgl_req *r = &s->reqs[s->n_reqs++];
  memset(r, 0, sizeof(*r));
  r->id = s->next_id++;
  r->state = SGL_REQ_WAITING;
  r->prompt = ids ? (int *)sgl_xmalloc(sizeof(int) * (size_t)(n_ids > 0 ? n_ids : 1))
                  : sgl_tokenizer_encode(s->tok, text, strlen(text), &r->n_prompt);
  if (ids) {
    /* P0-3: 直接给 token id（OpenAI 的 prompt:[ids] / sglang 的 input_ids）*/
    r->n_prompt = n_ids;
    if (n_ids > 0) memcpy(r->prompt, ids, sizeof(int) * (size_t)n_ids);
  }
  if (!r->prompt || r->n_prompt <= 0) { r->prompt = NULL; return NULL; }
  r->max_new = max_new > 0 ? max_new : 16;
  r->sp = sp ? *sp : (sgl_sampling_params){0};
  r->rng = mix64((uint64_t)r->id * 0x9E3779B97F4A7C15ULL + 1);
  r->inst = inst;
  r->priority = priority;
  r->cap_gen = r->max_new + 1;
  r->gen = (int *)sgl_xmalloc(sizeof(int) * (size_t)r->cap_gen);
  r->tok_lp = (float *)sgl_xmalloc(sizeof(float) * (size_t)r->cap_gen); /* P0-3 */
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

/* P1: 构建 grammar 词表（token id → 真实 UTF-8 字节）。只做一次。*/
static void sched_vocab_ready(sgl_scheduler *s) {
  if (s->vh_ready) return;
  int V = sgl_tokenizer_vocab_size(s->tok);
  s->vh_vocab = V;
  size_t off = 0;
  for (int i = 0; i < V; i++) {
    char tmp[256];
    int n = sgl_tokenizer_piece_bytes(s->tok, i, tmp, (int)sizeof(tmp));
    if (n <= 0 || off + (size_t)n + 1 > SGL_VH_CAP) {
      s->vh_ptr[i] = NULL;
      s->vh_len[i] = 0;
      continue;
    }
    memcpy(s->vh_buf + off, tmp, (size_t)n);
    s->vh_buf[off + (size_t)n] = 0;
    s->vh_ptr[i] = s->vh_buf + off;
    s->vh_len[i] = n;
    off += (size_t)n + 1;
  }
  s->vh_ready = 1;
}

/* P1: 约束采样——把会把输出带出合法前缀的 token 的 logits 置 -inf。
 * 返回 1 = 当前已是完整合法输出（调用方可停）。*/
static int req_apply_grammar(sgl_scheduler *s, sgl_req *r, float *logits) {
  if (!r->gram) return 0;
  sched_vocab_ready(s);
  sgl_grammar_set_vocab(r->gram, s->vh_ptr, s->vh_len, s->vh_vocab);
  int V = r->gram_vocab;
  if (V <= 0) V = s->vh_vocab;
  if (r->gram_ids == NULL || r->gram_alloc < V) {
    sgl_xfree(r->gram_ids);
    sgl_xfree(r->gram_ok);
    r->gram_ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)V);
    r->gram_ok = (unsigned char *)sgl_xmalloc((size_t)V);
    for (int i = 0; i < V; i++) r->gram_ids[i] = i;
    r->gram_alloc = V;
  }
  sgl_grammar_mask(r->gram, r->gram_ids, V, r->gram_ok);
  int allowed = 0;
  for (int i = 0; i < V && i < sgl_model_vocab(s->insts[r->inst].model); i++) {
    if (r->gram_ok[i]) {
      allowed++;
    } else {
      logits[i] = -INFINITY;
    }
  }
  if (allowed == 0) {
    /* 理论上不应该发生（文法卡死）：不约束，免得整批退化为 NaN */
    SGL_ERROR("grammar: no allowed token for req %d, falling back to unconstrained", r->id);
    return 0;
  }
  return sgl_grammar_can_finish(r->gram);
}

/* 采样并把 token 追加进 r->gen；返回 1 表示请求应结束（eos/max_new/满）*/
static int req_sample_append(sgl_scheduler *s, sgl_req *r, const float *logits) {
  sgl_model *m = s->insts[r->inst].model;
  float lp = 0.0f;
  int V = sgl_model_vocab(m);
  int tok = sgl_sample_ex(logits, V, &r->sp, &r->rng, r->gen, r->n_gen, &lp);
  if (tok == s->eos_id && !r->sp.ignore_eos) return 1; /* eos：不追加 */
  if (sgl_sampling_is_stop(&r->sp, tok)) return 1;      /* P0-3: stop_token_ids */
  r->gen[r->n_gen++] = tok;
  if (r->n_gen_lp < r->cap_gen) {
    r->tok_lp[r->n_gen_lp++] = lp; /* P0-3 */
  }
  if (r->gram) {
    if (sgl_grammar_accept(r->gram, tok) != 0 && r->sp.repetition_penalty >= 0.0f) {
      /* 理论上 mask 过不会非法；真非法就标死，别静默生成脏输出 */
      SGL_ERROR("grammar: rejected token %d for req %d (%s)", tok, r->id, sgl_grammar_error(r->gram));
    }
  }
  s->stats.decode_tokens++;
  if (r->n_gen >= r->max_new) return 1;
  if (r->gram && sgl_grammar_can_finish(r->gram)) return 1; /* 结构化输出完整即停 */
  if (r->kv_len + 1 >= sgl_model_ctx(m)) return 1; /* ctx 满 */
  return 0;
}

/* G12: 把视觉 token 插到 `<image>` 占位符位置的关键一步。
 * 设题（占位符在 prompt 内、位置 img_at）：
 *   [0, img_at)             文本已算过
 *   [img_at, img_at+n_img)  视觉 emb（丢掉占位符那个 token）
 *   [img_at+n_img, ...)     占位符之后的文本 token（位置顺延，RoPE 跟着变）
 * 无占位符（img_at < 0）时保持旧行为：尾巴为空，视觉 token 接在 prompt 末尾。*/
typedef struct { int tail_from, tail_pos, vis_pos; } img_plan;
/* G21：第 j 张（共 n_place 张）图在「占位符序列」上的插入点位。
 * 前一张图占掉 per_img 个位置，所以之后的文本位置整体后移 j*per_img。*/
static img_plan plan_img_j(const sgl_req *r, int j) {
  img_plan p;
  int shift = j * r->per_img;
  if (j < r->n_place) {
    p.vis_pos = r->img_at[j] + shift;
    p.tail_from = r->img_at[j] + 1;
    p.tail_pos = p.vis_pos + r->per_img;
  } else {
    p.vis_pos = r->n_prompt + shift;
    p.tail_from = r->n_prompt; /* 尾巴为空 */
    p.tail_pos = p.vis_pos + r->per_img;
  }
  return p;
}

/* P0-4: prefill 第一阶段——RadixCache 匹配 + 页分配（失败返回 1 需等，-1 错误）*/
static int prefill_begin(sgl_scheduler *s, sgl_req *r) {
  sgl_inst *in = &s->insts[r->inst];
  sgl_model *m = in->model;
  int need_total = r->n_prompt + r->n_img + r->max_new + 1;
  r->cap_pages = need_total;
  r->pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)need_total);

  /* G8: 带视觉 token 的请求不走 RadixCache（前缀语义不同：KV 里混了图片 emb）*/
  int matched = r->n_img > 0 ? 0 : sgl_radix_match(in->radix, r->prompt, r->n_prompt, r->pages);
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

  r->prefill_pos = matched;
  r->kv_len = matched;
  r->prefilling = 1;
  /* 全命中：还差最后一个 prompt token 才有 logits */
  if (r->prefill_pos >= r->n_prompt && r->n_prompt > 0) r->prefill_pos = r->n_prompt - 1;
  return 0;
}

/* P0-4: 推进一块 prefill（每块 s->prefill_chunk 个 token）。
 * 图片请求不分块（视觉 token 要插在占位符位置，一次算完）。
 * 返回 0 正常（可能还没算完），-1 前向错误。*/
static int prefill_step(sgl_scheduler *s, sgl_req *r) {
  sgl_inst *in = &s->insts[r->inst];
  sgl_model *m = in->model;
  float *lg = s->b_logits;
  if (r->n_img > 0) {
    /* G8/G12: 视觉 token 进同一页表；G12 插到占位符位置，并把占位符后的文本尾巴
     * 在视觉 token 之后接着算（位置 = img_at + n_img 起），这样 RoPE 位置与
     * llava「把 <image> 换成一个视觉 token 块」完全一致。*/
    int from = r->prefill_pos;
    for (int j = 0; j < r->n_place; j++) {
      int n_suf = r->img_at[j] - from;
      if (n_suf > 0) {
        int base = from + j * r->per_img;
        if (sgl_model_prefill(m, r->prompt + from, n_suf, base, r->pages, lg) != 0) return -1;
        s->stats.prefill_tokens += n_suf;
        from = r->img_at[j];
      }
      img_plan pl = plan_img_j(r, j);
      if (sgl_model_prefill_emb(m, r->feats + (long)j * r->per_img * sgl_model_hidden(m),
                                r->per_img, pl.vis_pos, r->pages, lg) != 0)
        return -1;
      s->stats.prefill_tokens += r->per_img;
      r->kv_len = pl.vis_pos + r->per_img;
      from = pl.tail_from;
    }
    if (from < r->n_prompt) {
      int ntail = r->n_prompt - from;
      int base = from + r->n_place * r->per_img;
      if (sgl_model_prefill(m, r->prompt + from, ntail, base, r->pages, lg) != 0) return -1;
      s->stats.prefill_tokens += ntail;
      r->kv_len = base + ntail;
    }
    sgl_xfree(r->feats);
    r->feats = NULL;
    r->n_img = 0;
  } else {
    int remain = r->n_prompt - r->prefill_pos;
    int n = remain > s->prefill_chunk ? s->prefill_chunk : remain;
    if (n > 0) {
      if (sgl_model_prefill(m, r->prompt + r->prefill_pos, n, r->prefill_pos, r->pages, lg) != 0)
        return -1;
      s->stats.prefill_tokens += n;
      r->prefill_pos += n;
      r->kv_len = r->prefill_pos;
    }
    if (r->prefill_pos < r->n_prompt) return 0; /* 还没算完，下一步继续 */
  }
  r->state = SGL_REQ_RUNNING;
  r->prefilling = 0;
  if (r->gram) req_apply_grammar(s, r, lg); /* P1 */
  if (req_sample_append(s, r, lg)) req_finish(s, r);
  return 0;
}

/* P0-4: 抢占一个运行中的非流式请求（释放 KV → 回到等待队列重算）。
 * 优先级低的先被抢；同优先级抢生成得最少的（丢弃的进度最少）。0 = 没抢到。*/
static int preempt_one(sgl_scheduler *s) {
  sgl_req *best = NULL;
  for (int i = 0; i < s->n_reqs; i++) {
    sgl_req *r = &s->reqs[i];
    if (r->state != SGL_REQ_RUNNING || r->streaming || !r->pages) continue;
    if (!best || r->priority < best->priority ||
        (r->priority == best->priority && r->n_gen < best->n_gen))
      best = r;
  }
  if (!best) return 0;
  sgl_model *m = s->insts[best->inst].model;
  sgl_model_pages_unref(m, best->pages, best->cap_pages);
  sgl_xfree(best->pages);
  best->pages = NULL;
  best->n_gen = 0;
  best->n_gen_lp = 0;
  sgl_grammar_reset(best->gram); /* P1: 重算时文法也回到初态 */
  best->kv_len = 0;
  best->prefill_pos = 0;
  best->hit_len = 0;
  best->state = SGL_REQ_WAITING;
  s->stats.n_preempt++;
  return 1;
}

int sgl_sched_step(sgl_scheduler *s) {
  if (!s || s->paused) return 0; /* /pause_generation：请求留在队列，KV 保留 */
  /* 1) 推进正在进行的 chunked prefill（P0-4）：每步一块，decode 不被长 prompt 堵死 */
  for (int i = 0; i < s->n_reqs; i++) {
    sgl_req *r = &s->reqs[i];
    if (r->state != SGL_REQ_WAITING || !r->prefilling) continue;
    int rc = prefill_step(s, r);
    if (rc < 0) { r->state = SGL_REQ_DONE; r->n_gen = -1; } /* 前向致命错 */
    return 0;
  }

  /* 2) 没有进行中的，就取优先级最高（同优先级取最早）的等待请求开始 prefill */
  {
    int best = -1;
    for (int i = 0; i < s->n_reqs; i++) {
      sgl_req *r = &s->reqs[i];
      if (r->state != SGL_REQ_WAITING) continue;
      if (best < 0 || r->priority > s->reqs[best].priority) best = i;
    }
    if (best < 0) goto decode_phase;
    sgl_req *r = &s->reqs[best];
    int rc = prefill_begin(s, r);
    if (rc < 0) { r->state = SGL_REQ_DONE; r->n_gen = -1; return 0; }
    if (rc == 1) {
      /* KV 池不足：先抢一个运行中请求重算（原版 preempt-recompute 策略）*/
      if (!preempt_one(s)) return 0; /* 无可抢占：等运行中的请求腾出页 */
      rc = prefill_begin(s, r);
      if (rc != 0) return 0;
    }
    int rc2 = prefill_step(s, r);
    if (rc2 < 0) { r->state = SGL_REQ_DONE; r->n_gen = -1; }
    return 0;
  }

decode_phase:

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
      float *lg = s->b_logits + (size_t)bi * sgl_model_vocab(m);
      if (r->gram) req_apply_grammar(s, r, lg); /* P1: 约束解码掩码 */
      if (req_sample_append(s, r, lg)) {
        req_finish(s, r);
        finished++;
      }
      bi++;
    }
  }
  return finished;
}

int sgl_sched_poll_logprobs(sgl_scheduler *s, int req_id, float **out, int *n) {
  if (!s) return -1;
  for (int i = 0; i < s->n_reqs; i++) {
    sgl_req *r = &s->reqs[i];
    if (r->id != req_id) continue;
    if (!r->tok_lp || r->n_gen_lp <= 0) return -1;
    int k = r->n_gen_lp;
    if (r->n_gen >= 0 && r->n_gen < k) k = r->n_gen; /* 被取走后 n_gen=-1 */
    *out = (float *)sgl_xmalloc(sizeof(float) * (size_t)k);
    memcpy(*out, r->tok_lp, sizeof(float) * (size_t)k);
    if (n) *n = k;
    return 0;
  }
  return -1;
}

int sgl_sched_abort(sgl_scheduler *s, int req_id) {
  if (!s) return -1;
  for (int i = 0; i < s->n_reqs; i++) {
    sgl_req *r = &s->reqs[i];
    if (r->id != req_id || r->state == SGL_REQ_DONE) continue;
    if (r->pages) {
      sgl_model_pages_unref(s->insts[r->inst].model, r->pages, r->cap_pages);
      sgl_xfree(r->pages);
      r->pages = NULL;
    }
    if (r->feats) {
      sgl_xfree(r->feats);
      r->feats = NULL;
      r->n_img = 0;
    }
    r->state = SGL_REQ_DONE;
    r->n_gen = -1; /* 与「已取走」同义：poll 不会返回它 */
    return 0;
  }
  return -1;
}

void sgl_sched_flush_cache(sgl_scheduler *s) {
  if (!s) return;
  for (int i = 0; i < s->n_insts; i++) {
    sgl_radix_free(s->insts[i].radix);
    s->insts[i].radix = sgl_radix_new(s->insts[i].model);
  }
  SGL_INFO("radix cache flushed");
}

int sgl_sched_poll(sgl_scheduler *s, int *req_id, char **text, int **ids,
                   int *n_ids, int *n_prompt, int *n_hit) {
  return sgl_sched_poll_lp(s, req_id, text, ids, n_ids, n_prompt, n_hit, NULL, NULL);
}

int sgl_sched_poll_lp(sgl_scheduler *s, int *req_id, char **text, int **ids, int *n_ids,
                      int *n_prompt, int *n_hit, float **lp_out, int *n_lp) {
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
    /* P0-3: logprob 在清掉 n_gen 之前取，不额外占调用方一次查找 */
    if (lp_out) {
      int k = r->n_gen_lp < r->n_gen ? r->n_gen_lp : r->n_gen;
      if (k > 0) {
        *lp_out = (float *)sgl_xmalloc(sizeof(float) * (size_t)k);
        memcpy(*lp_out, r->tok_lp, sizeof(float) * (size_t)k);
      } else {
        *lp_out = NULL;
      }
      if (n_lp) *n_lp = k > 0 ? k : 0;
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
  /* P1 DP：池容量与空闲页按所有 base 副本累加 */
  st.pool_pages = 0;
  st.pool_free = 0;
  st.n_replicas = 0;
  for (int k = 0; k < s->n_insts; k++) {
    if (s->insts[k].name || s->insts[k].dead) continue;
    st.n_replicas++;
    st.pool_pages += sgl_model_pool_pages(s->insts[k].model);
    st.pool_free += sgl_model_nfree(s->insts[k].model);
  }
  if (st.n_replicas == 0) { /* 只有 lora 实例的极端情况：回退到 0 号 */
    st.pool_pages = sgl_model_pool_pages(s->insts[0].model);
    st.pool_free = sgl_model_nfree(s->insts[0].model);
  }
  st.prefill_chunk = s->prefill_chunk;
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
  if (!name || !*name) {
    /* P1 DP：在 base 副本里挑当前运行请求最少的（--dp N 时有多份）*/
    int best = 0, best_load = 1 << 30;
    for (int k = 0; k < s->n_insts; k++) {
      if (s->insts[k].name || s->insts[k].dead) continue;
      int load = 0;
      for (int i = 0; i < s->n_reqs; i++)
        if (s->reqs[i].state != SGL_REQ_DONE && s->reqs[i].inst == k) load++;
      if (load < best_load) {
        best_load = load;
        best = k;
      }
    }
    return best;
  }
  for (int i = 1; i < s->n_insts; i++)
    if (s->insts[i].name && !s->insts[i].dead && strcmp(s->insts[i].name, name) == 0) return i;
  return -1;
}

/* P1: 加一份 base 副本（数据并行：--dp N 起 N 份完整模型，各自独立 KV 池）。
 * 返回副本数量（含第一份）；失败返回 -1。*/
int sgl_sched_add_replica(sgl_scheduler *s, const char *model_dir) {
  if (!s || !model_dir) return -1;
  if (s->n_insts >= s->cap_insts) {
    s->cap_insts *= 2;
    s->insts = (sgl_inst *)sgl_xrealloc(s->insts, sizeof(sgl_inst) * (size_t)s->cap_insts);
    memset(s->insts + s->n_insts, 0, sizeof(sgl_inst) * (size_t)(s->cap_insts - s->n_insts));
  }
  sgl_model *m = sgl_model_load(model_dir);
  if (!m) {
    SGL_ERROR("add_replica: model load failed (%s)", model_dir);
    return -1;
  }
  int k = s->n_insts++;
  memset(&s->insts[k], 0, sizeof(s->insts[k]));
  s->insts[k].name = NULL;
  s->insts[k].model = m;
  s->insts[k].radix = sgl_radix_new(m);
  SGL_INFO("data-parallel replica #%d ready (%s)", k, model_dir);
  return k + 1;
}

/* P1: 热更新权重（/update_weights_from_disk）。inst = -1 表示所有 base 副本。
 * 需要该实例当前没有未完成的请求（有则返回 -2，让调用方稍后重试）。*/
int sgl_sched_reload_weights(sgl_scheduler *s, int inst, const char *model_dir) {
  if (!s || !model_dir) return -1;
  int lo = 0, hi = s->n_insts;
  if (inst >= 0) {
    if (inst >= s->n_insts) return -1;
    lo = inst;
    hi = inst + 1;
  }
  /* 先检查忙闲（避免加载到一半才发现要回滚）*/
  for (int k = lo; k < hi; k++) {
    for (int i = 0; i < s->n_reqs; i++)
      if (s->reqs[i].state != SGL_REQ_DONE && s->reqs[i].inst == k) return -2;
  }
  for (int k = lo; k < hi; k++) {
    sgl_model *nm = sgl_model_load(model_dir);
    if (!nm) {
      SGL_ERROR("reload_weights: load failed (%s), instance %d unchanged", model_dir, k);
      return -1;
    }
    sgl_model_free(s->insts[k].model);
    s->insts[k].model = nm;
    sgl_radix_free(s->insts[k].radix); /* 旧 KV 页属于旧模型的池 */
    s->insts[k].radix = sgl_radix_new(nm);
    SGL_INFO("weights reloaded on instance %d (%s)", k, model_dir);
  }
  return 0;
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

static int submit_impl(sgl_scheduler *s, const char *text, const int *ids, int n_ids, int max_new,
                       const sgl_sampling_params *sp, const char *lora, int streaming, int priority) {
  int inst = sgl_sched_find_lora(s, lora);
  if (inst < 0) {
    SGL_ERROR("submit: unknown lora '%s'", lora ? lora : "");
    return -1;
  }
  sgl_req *r = req_new(s, text, ids, n_ids, max_new, sp, inst, priority);
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
  return submit_impl(s, text, NULL, 0, max_new, sp, lora, 0, 0);
}

int sgl_sched_submit_prio(sgl_scheduler *s, const char *text, int max_new,
                          const sgl_sampling_params *sp, const char *lora, int priority) {
  return submit_impl(s, text, NULL, 0, max_new, sp, lora, 0, priority);
}

/* P0-3: 直接给 token id 提交（不经 tokenizer）*/
int sgl_sched_submit_ids(sgl_scheduler *s, const int *ids, int n_ids, int max_new,
                         const sgl_sampling_params *sp, const char *lora) {
  if (!s || !ids || n_ids <= 0) return -1;
  return submit_impl(s, NULL, ids, n_ids, max_new, sp, lora, 0, 0);
}

/* G8: 带视觉特征的提交。feats [n_img][E_lm] 的所有权转给调度器（sgl_xmalloc 分配，
 * prefill 用完 sgl_xfree）。失败（模型不支 emb / 页不够）返回 -1 并接管释放 feats。 */
int sgl_sched_submit_img(sgl_scheduler *s, const char *text, int max_new,
                         const sgl_sampling_params *sp, float *feats, int n_img) {
  if (!s || !feats || n_img <= 0) { sgl_xfree(feats); return -1; }
  int inst = 0;
  sgl_req *r = req_new(s, text, NULL, 0, max_new, sp, inst, 0);
  if (!r) { sgl_xfree(feats); return -1; }
  r->feats = feats;
  r->n_img = n_img;
  /* G21（多图）：扫出所有 `<image>` 占位符，按出现顺序把图片依次插进去。
   * 之前只取第一个占位符，多图全撑到同一个位置（README 的 Known limitations）。
   * 每张图的 token 数必须一致（同一视觉塔、同一分辨率），否则退回单占位符行为。*/
  r->n_place = 0;
  r->per_img = n_img;
  if (s->img_tok >= 0) {
    for (int i = 0; i < r->n_prompt && r->n_place < 8; i++)
      if (r->prompt[i] == s->img_tok) r->img_at[r->n_place++] = i;
  }
  if (r->n_place > 0) {
    if (n_img % r->n_place == 0) {
      r->per_img = n_img / r->n_place;
    } else {
      /* 图片数与占位符数对不上：只用第一个占位符（旧行为）*/
      r->n_place = 1;
      r->per_img = n_img;
    }
  }
  if (getenv("SGL_IMG_DEBUG"))
    SGL_INFO("img insert: img_tok=%d n_prompt=%d places=%d per_img=%d n_img=%d", s->img_tok,
             r->n_prompt, r->n_place, r->per_img, n_img);
  int need = r->n_prompt + n_img + r->max_new + 1;
  if (need > sgl_model_pool_pages(s->insts[inst].model)) {
    SGL_ERROR("submit_img: need %d pages > pool %d", need,
              sgl_model_pool_pages(s->insts[inst].model));
    r->state = SGL_REQ_DONE;
    r->n_gen = -1;
    return -1;
  }
  return r->id;
}

/* P1: 带 grammar 提交（不支持流式/图片；gram 所有权转给调度器）*/
int sgl_sched_submit_grammar(sgl_scheduler *s, const char *text, int max_new,
                             const sgl_sampling_params *sp, const char *lora, sgl_grammar *gram) {
  int id = submit_impl(s, text, NULL, 0, max_new, sp, lora, 0, 0);
  if (id < 0) {
    sgl_grammar_free(gram);
    return -1;
  }
  for (int i = 0; i < s->n_reqs; i++) {
    if (s->reqs[i].id != id) continue;
    s->reqs[i].gram = gram;
    s->reqs[i].gram_vocab = sgl_tokenizer_vocab_size(s->tok);
    break;
  }
  return id;
}

int sgl_sched_submit_stream_lora(sgl_scheduler *s, const char *text, int max_new,
                                 const sgl_sampling_params *sp, const char *lora) {
  return submit_impl(s, text, NULL, 0, max_new, sp, lora, 1, 0);
}

int sgl_sched_submit_stream_prio(sgl_scheduler *s, const char *text, int max_new,
                                 const sgl_sampling_params *sp, const char *lora, int priority) {
  return submit_impl(s, text, NULL, 0, max_new, sp, lora, 1, priority);
}

/* ---- 生命周期：/pause_generation /continue_generation ---- */

void sgl_sched_set_paused(sgl_scheduler *s, int paused) {
  if (s) s->paused = paused ? 1 : 0;
}

int sgl_sched_is_paused(const sgl_scheduler *s) { return s ? s->paused : 0; }

/* ---- tokenizer 直通（/tokenize /detokenize）---- */

int *sgl_sched_encode(sgl_scheduler *s, const char *text, size_t len, int *n_out) {
  if (!s || !text) return NULL;
  int n = 0;
  int *ids = sgl_tokenizer_encode(s->tok, text, len, &n);
  if (n_out) *n_out = n;
  return ids;
}

char *sgl_sched_decode(sgl_scheduler *s, const int *ids, int n) {
  if (!s || !ids || n <= 0) return NULL;
  return sgl_tokenizer_decode(s->tok, ids, n);
}

int sgl_sched_vocab_size(const sgl_scheduler *s) {
  return s ? sgl_tokenizer_vocab_size(s->tok) : 0;
}

int sgl_sched_piece_id(const sgl_scheduler *s, const char *piece) {
  return s ? sgl_tokenizer_piece_id(s->tok, piece) : -1;
}

/* ---- 运行时卸载 LoRA（/unload_lora_adapter）----
 * 实例下标不复用（运行中/已完成未取走的请求还指着它），只把模型和 radix 释放并标 dead。*/
int sgl_sched_remove_lora(sgl_scheduler *s, const char *name) {
  if (!s || !name || !*name) return -1;
  int k = sgl_sched_find_lora(s, name);
  if (k <= 0) return -1;
  for (int i = 0; i < s->n_reqs; i++)
    if (s->reqs[i].state != SGL_REQ_DONE && s->reqs[i].inst == k) return -2;
  sgl_radix_free(s->insts[k].radix);
  sgl_model_free(s->insts[k].model);
  s->insts[k].radix = NULL;
  s->insts[k].model = NULL;
  s->insts[k].dead = 1;
  SGL_INFO("lora instance '%s' unloaded", name);
  sgl_xfree(s->insts[k].name);
  s->insts[k].name = NULL;
  return 0;
}
