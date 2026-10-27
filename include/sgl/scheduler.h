/* sgl/scheduler.h — 多请求调度器：连续批处理（prefill 优先 + 批量 decode）
 * 对应 sglang srt/managers/scheduler.py 的单机同步核心路径：
 *   - 到达的请求进等待队列（FIFO）
 *   - 每步优先 prefill 一个等待请求（对应 sglang 默认 prefill 优先策略）
 *   - 无等待请求时，把所有运行中请求组成一个 batch 一次 decode 前向 */
#ifndef SGL_SCHEDULER_H
#define SGL_SCHEDULER_H

#include <stdint.h>
#include "sgl/grammar.h"

#include "sgl/bpe.h"
#include "sgl/gpt2.h"
#include "sgl/sampling.h"

typedef struct sgl_scheduler sgl_scheduler;

/* 统计（对应 sglang 的 decode throughput 日志字段）*/
typedef struct {
  long prefill_tokens;   /* 已 prefill 的 prompt token 总数（不含缓存命中部分）*/
  long cache_hit_tokens; /* RadixCache 命中复用的 token 总数 */
  long decode_tokens;    /* 已生成的 token 总数 */
  int  n_finished;       /* 已完成请求数 */
  int  n_running;        /* 当前运行中 */
  int  n_waiting;        /* 当前等待中 */
  int  pool_pages;       /* KV 池总页数 */
  int  pool_free;        /* KV 池空闲页数 */
  int  n_replicas;       /* P1 DP：base 副本数（含主副本）*/
  long n_preempt;        /* 被抢占（重算）的请求次数 */
  int  prefill_chunk;    /* chunked prefill 的块大小（token）*/
} sgl_sched_stats;

sgl_scheduler *sgl_sched_new(const char *model_dir, uint64_t seed);
void sgl_sched_free(sgl_scheduler *s);

/* 提交请求（文本会立刻 tokenize），返回请求 id（从 1 递增）；失败 -1 */
int sgl_sched_submit(sgl_scheduler *s, const char *text, int max_new,
                     const sgl_sampling_params *sp);

/* 推进一步：优先 prefill 一个等待请求，否则批量 decode 全部运行中请求。
 * 返回这一步完成的请求数（0 = 无进展或仍在跑）*/
int sgl_sched_step(sgl_scheduler *s);

/* 取一个已完成请求的产物。返回 1 且填充输出（text/ids 由 malloc 分配，
 * 调用方负责 free(ids) 和 sgl_xfree(text)；n_prompt/n_hit 可为 NULL（n_hit 为
 * RadixCache 命中的前缀长度）；没有已完成的返回 0）*/
int sgl_sched_poll(sgl_scheduler *s, int *req_id, char **text, int **ids,
                   int *n_ids, int *n_prompt, int *n_hit);

/* 运行中 + 等待中的请求总数（=0 表示全部做完）*/
int sgl_sched_nactive(sgl_scheduler *s);

/* P0-4: 中止一个请求（运行中/等待中均可），释放其 KV 页；0 = 成功。 */
int sgl_sched_abort(sgl_scheduler *s, int req_id);

/* P1: 清空所有实例的 RadixCache（对应 /flush_cache）*/
void sgl_sched_flush_cache(sgl_scheduler *s);

/* P0-3: 取某个已完成请求每个生成 token 的 logprob（拷贝，调用方 sgl_xfree）。
 * 需在 poll 之后、请求仍留在调度器里时调用。*/
int sgl_sched_poll_logprobs(sgl_scheduler *s, int req_id, float **out, int *n);

/* 同 sgl_sched_poll，但额外返回每个生成 token 的 logprob（P0-3）*/
int sgl_sched_poll_lp(sgl_scheduler *s, int *req_id, char **text, int **ids, int *n_ids,
                      int *n_prompt, int *n_hit, float **lp_out, int *n_lp);

sgl_sched_stats sgl_sched_get_stats(const sgl_scheduler *s);

/* ---- 流式（SSE 用）---- */
typedef enum {
  SGL_STREAM_MORE = 0,  /* 运行中，暂无新 token */
  SGL_STREAM_TOKEN,     /* *piece（malloc，sgl_xfree 释放），*tok_id 是新 token */
  SGL_STREAM_DONE,      /* 结束（新 token 已在前几次全部交出）*/
} sgl_stream_ev;

/* 非阻塞取一个新 token；同一请求可反复调用直到 DONE（失败的请求直接 DONE）*/
sgl_stream_ev sgl_sched_stream_next(sgl_scheduler *s, int req_id,
                                    char **piece, int *tok_id);

/* 同 submit，但标记为流式：poll 不回收，由 stream_next 取尽后 reclaim */
int sgl_sched_submit_stream(sgl_scheduler *s, const char *text, int max_new,
                            const sgl_sampling_params *sp);

/* 回收一个已流式取尽的请求（与 poll 同语义地标记已取走）*/
void sgl_sched_reclaim_stream(sgl_scheduler *s, int req_id);

/* ---- 多 LoRA 实例池（方案 B）----
 * 每个 adapter 一份「base+合并 delta」的完整 sgl_model 副本（独立 KV 池），
 * 请求按 lora 名路由到对应实例；lora=NULL 走 base 实例（下标 0）。
 * 复用现有加载+合并代码路径（sgl_model_load 读 SGL_LORA 环境变量）。*/

/* 加一个 LoRA 实例：内部临时设 SGL_LORA=lora_dir 再 sgl_model_load(model_dir)，
 * 完成后恢复原环境。成功返回实例下标（≥1），失败 -1。 */
int sgl_sched_add_lora(sgl_scheduler *s, const char *name, const char *model_dir,
                       const char *lora_dir);

/* lora 名 → 实例下标；NULL/"" → 0（base）；未知名 → -1 */
int sgl_sched_find_lora(const sgl_scheduler *s, const char *name);

/* 同 submit / submit_stream，但按 lora 名路由实例；未知名返回 -1 */
/* P0-3: 直接给 token id 提交（不经 tokenizer）*/
int sgl_sched_submit_ids(sgl_scheduler *s, const int *ids, int n_ids, int max_new,
                         const sgl_sampling_params *sp, const char *lora);

/* P1: 按 grammar 约束提交（gram 所有权转给调度器；失败时自行释放）*/
int sgl_sched_submit_grammar(sgl_scheduler *s, const char *text, int max_new,
                            const sgl_sampling_params *sp, const char *lora, sgl_grammar *gram);

/* P1: 加一份 base 副本（数据并行）*/
int sgl_sched_add_replica(sgl_scheduler *s, const char *model_dir);

/* P1: 热更新权重（对应 /update_weights_from_disk）：把模型 dir 重新加载到指定实例（-1 = 所有
 * base 副本）。该实例还有未完成的请求时返回 -2；成功返回 0。KV 池与 Radix 一并换新。*/
int sgl_sched_reload_weights(sgl_scheduler *s, int inst, const char *model_dir);

int sgl_sched_submit_lora(sgl_scheduler *s, const char *text, int max_new,
                          const sgl_sampling_params *sp, const char *lora);
int sgl_sched_submit_stream_lora(sgl_scheduler *s, const char *text, int max_new,
                                 const sgl_sampling_params *sp, const char *lora);

/* ---- G8 多模态（LLaVA 风格 embedding 级融合，CPU 路径）----
 * feats: [n_img][E_lm] 视觉 token（由调用方的视觉塔+projector 算好，sgl_xmalloc 分配），
 * 接在 prompt token 后面进同一页表；首 token 的 logits 取最后一个视觉 token。
 * 所有权转给调度器（prefill 完即释放）；失败也接管释放。仅 llama 后端支持。
 * 返回请求 id，失败 -1。 */
int sgl_sched_submit_img(sgl_scheduler *s, const char *text, int max_new,
                         const sgl_sampling_params *sp, float *feats, int n_img);

#endif
