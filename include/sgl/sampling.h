/* sgl/sampling.h — 采样（对应 sglang srt/sampling/sampling_batch_info.py 的单条路径）*/
#ifndef SGL_SAMPLING_H
#define SGL_SAMPLING_H

#include <stdint.h>

/* P0-3: 补齐 OpenAI / sglang 的采样参数面。
 * 语义与 OpenAI 一致：repetition_penalty=0 或 1 视为关闭；
 * frequency/presence_penalty=0 视为关闭；min_p=0 视为关闭。 */
#define SGL_MAX_STOP_TOKENS 8

typedef struct {
  float temperature; /* <=0 退化为 greedy */
  int top_k;         /* <=0 不启用 */
  float top_p;       /* (0,1] 外不启用 */
  float min_p;       /* >0：过滤 p < min_p * max_p 的候选 */
  float repetition_penalty; /* HF 语义：>0 且 !=1 生效 */
  float frequency_penalty;  /* logit -= f * count(tok) */
  float presence_penalty;   /* logit -= p * (count(tok) > 0) */
  int ignore_eos;           /* 1 = 不因 eos 结束 */
  int stop_ids[SGL_MAX_STOP_TOKENS]; /* 额外结束 token（命中即停）*/
  int n_stop_ids;
  uint64_t seed;
} sgl_sampling_params;

uint64_t sgl_rng_next(uint64_t *state); /* xorshift64* */

/* greedy：返回 argmax */
int sgl_sample_greedy(const float *logits, int n);

/* temperature/top-k/top-p/min_p/penalty 采样。
 * history/n_history 用于 penalty（可为 NULL/0）。
 * out_logprob 非 NULL 时写入选中 token 的 logprob（log_softmax，greedy 路径也支持）。*/
int sgl_sample_ex(const float *logits, int n, const sgl_sampling_params *p, uint64_t *rng,
                  const int *history, int n_history, float *out_logprob);

/* 兼容旧接口（无历史、不取 logprob） */
int sgl_sample(const float *logits, int n, const sgl_sampling_params *p, uint64_t *rng);

/* token = stop_ids 里的一员？ */
int sgl_sampling_is_stop(const sgl_sampling_params *p, int token);

/* [0,1) 均匀随机数——sgl_sample_ex 内部同一口径；投机采样的接受/重采样
 * 检验也用它，保证两处对 RNG 的消费方式一致。*/
double sgl_rng_uniform(uint64_t *state);

/* 只做分布变换、不抽牌：把 logits 按 p 的 penalty → temperature → min_p →
 * top_k → top_p 规则变成「实际抽样分布」写入 out_probs[n]（被截断的位置为
 * 0；temperature<=0 退化为 argmax 的 one-hot）。sgl_sample_ex 抽的就是这个
 * 分布（同一构建代码）。投机采样（spec.c/cuda_spec.c）用它拿 target/draft
 * 两边的 p̃、q̃ 做接受检验与修正重采样。*/
void sgl_dist_transform(const float *logits, int n, const sgl_sampling_params *p,
                         const int *history, int n_history, float *out_probs);

#endif