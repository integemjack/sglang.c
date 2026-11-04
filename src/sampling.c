/* sgl sampling 实现 */
#include "sgl/sampling.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"

uint64_t sgl_rng_next(uint64_t *state) {
  uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 0x2545F4914F6CDD1DULL;
}

double sgl_rng_uniform(uint64_t *state) {
  return (double)(sgl_rng_next(state) >> 11) / 9007199254740992.0; /* [0,1) */
}

int sgl_sample_greedy(const float *logits, int n) {
  int best = 0;
  for (int i = 1; i < n; i++)
    if (logits[i] > logits[best]) best = i;
  return best;
}

int sgl_sampling_is_stop(const sgl_sampling_params *p, int token) {
  if (!p) return 0;
  for (int i = 0; i < p->n_stop_ids && i < SGL_MAX_STOP_TOKENS; i++)
    if (p->stop_ids[i] == token) return 1;
  return 0;
}

/* log_softmax(logits)[idx]（数值稳定）*/
static float log_softmax_at(const float *logits, int n, int idx) {
  float mx = logits[0];
  for (int i = 1; i < n; i++)
    if (logits[i] > mx) mx = logits[i];
  double sum = 0;
  for (int i = 0; i < n; i++) sum += exp((double)(logits[i] - mx));
  return (float)((double)(logits[idx] - mx) - log(sum));
}

/* penalty：把 logits 拷到 tmp 后按 OpenAI 语义改写 */
static void apply_penalties(const float *logits, int n, const sgl_sampling_params *p,
                            const int *history, int n_history, float *tmp) {
  memcpy(tmp, logits, sizeof(float) * (size_t)n);
  if (n_history <= 0 || !history) return;
  int rep = (p->repetition_penalty > 0.0f && p->repetition_penalty != 1.0f);
  int fq = (p->frequency_penalty != 0.0f);
  int pr = (p->presence_penalty != 0.0f);
  if (!rep && !fq && !pr) return;
  /* 统计历史上每个 token 的出现次数（只统计出现过的，避免整表清零）*/
  int *seen = (int *)malloc(sizeof(int) * (size_t)n_history);
  int *cnt = (int *)malloc(sizeof(int) * (size_t)n_history);
  if (!seen || !cnt) {
    free(seen);
    free(cnt);
    return;
  }
  int ns = 0;
  for (int i = 0; i < n_history; i++) {
    int t = history[i];
    if (t < 0 || t >= n) continue;
    int k = 0;
    for (; k < ns; k++)
      if (seen[k] == t) {
        cnt[k]++;
        break;
      }
    if (k == ns) {
      seen[ns] = t;
      cnt[ns] = 1;
      ns++;
    }
  }
  for (int k = 0; k < ns; k++) {
    int t = seen[k];
    float v = tmp[t];
    if (rep) v = (v > 0) ? v / p->repetition_penalty : v * p->repetition_penalty;
    if (fq) v -= p->frequency_penalty * (float)cnt[k];
    if (pr) v -= p->presence_penalty;
    tmp[t] = v;
  }
  free(seen);
  free(cnt);
}

typedef struct {
  int idx;
  float p;
} cand;

static int cmp_desc(const void *a, const void *b) {
  float x = ((const cand *)a)->p, y = ((const cand *)b)->p;
  return (x < y) - (x > y);
}

/* penalty 开启且带历史时，返回改写后的 logits（写进 *tmp_out，调用方 free）；
 * 否则原样返回。malloc 失败退回原始 logits（与旧版内联写法一致）。*/
static const float *penalized(const float *logits, int n, const sgl_sampling_params *p,
                              const int *history, int n_history, float **tmp_out) {
  *tmp_out = NULL;
  if (!p || !(p->repetition_penalty > 0.0f && p->repetition_penalty != 1.0f ||
              p->frequency_penalty != 0.0f || p->presence_penalty != 0.0f) ||
      n_history <= 0 || !history)
    return logits;
  float *tmp = (float *)malloc(sizeof(float) * (size_t)n);
  if (!tmp) return logits;
  apply_penalties(logits, n, p, history, n_history, tmp);
  *tmp_out = tmp;
  return tmp;
}

/* 候选构建（单一来源）：softmax(logits/T) → min_p → top_k → top_p 截断。
 * 温度路径：*cs_out = 按概率降序的候选数组（调用方 sgl_xfree），*m_out = 保留数，
 *   *renorm_out = 保留集概率和；greedy（temperature<=0）：cs=NULL、m=0、
 *   renorm=1，返回 argmax。out_probs 非 NULL 时写完整抽样分布（保留位 =
 *   p/renorm，其余 0；greedy 写 one-hot）。抽牌不在这里——分布与谁怎么抽
 *   无关，这正是修正采样（spec.c/cuda_spec.c）能复用它的原因。*/
static int build_cands(const float *lg, int n, const sgl_sampling_params *p,
                       cand **cs_out, int *m_out, double *renorm_out, float *out_probs) {
  int max_i = sgl_sample_greedy(lg, n);
  *cs_out = NULL;
  *m_out = 0;
  *renorm_out = 1.0;
  if (out_probs) memset(out_probs, 0, sizeof(float) * (size_t)n);
  if (!p || p->temperature <= 0.0f) {
    if (out_probs) out_probs[max_i] = 1.0f;
    return max_i;
  }

  /* softmax(logits/T) */
  float *probs = (float *)sgl_xmalloc(sizeof(float) * (size_t)n);
  double T = p->temperature;
  double sum = 0;
  for (int i = 0; i < n; i++) {
    probs[i] = (float)exp((lg[i] - lg[max_i]) / T);
    sum += probs[i];
  }
  for (int i = 0; i < n; i++) probs[i] /= (float)sum;

  /* 候选集：min_p → top-k → top-p */
  cand *cs = (cand *)sgl_xmalloc(sizeof(cand) * (size_t)n);
  for (int i = 0; i < n; i++) cs[i] = (cand){i, probs[i]};
  qsort(cs, (size_t)n, sizeof(cand), cmp_desc);

  int m = n;
  if (p->min_p > 0.0f && p->min_p < 1.0f) {
    float thresh = cs[0].p * p->min_p;
    int cut = 1;
    while (cut < m && cs[cut].p >= thresh) cut++;
    m = cut;
  }
  if (p->top_k > 0 && p->top_k < m) m = p->top_k;
  if (p->top_p > 0.0f && p->top_p < 1.0f) {
    double cum = 0;
    int cut = m;
    for (int i = 0; i < m; i++) {
      cum += cs[i].p;
      if (cum >= p->top_p) {
        cut = i + 1;
        break;
      }
    }
    m = cut < 1 ? 1 : cut;
  }

  double renorm = 0;
  for (int i = 0; i < m; i++) renorm += cs[i].p;
  if (out_probs)
    for (int i = 0; i < m; i++) out_probs[cs[i].idx] = (float)(cs[i].p / renorm);
  sgl_xfree(probs);
  *cs_out = cs;
  *m_out = m;
  *renorm_out = renorm;
  return max_i;
}

int sgl_sample_ex(const float *logits, int n, const sgl_sampling_params *p, uint64_t *rng,
                  const int *history, int n_history, float *out_logprob) {
  float *tmp = NULL;
  const float *lg = penalized(logits, n, p, history, n_history, &tmp);
  cand *cs = NULL;
  int m = 0;
  double renorm = 1;
  int max_i = build_cands(lg, n, p, &cs, &m, &renorm, NULL);
  int pick;
  if (!cs) {
    pick = max_i; /* greedy */
  } else {
    double r = sgl_rng_uniform(rng); /* [0,1) */
    double acc = 0;
    pick = cs[0].idx;
    for (int i = 0; i < m; i++) {
      acc += cs[i].p / renorm;
      if (r < acc) {
        pick = cs[i].idx;
        break;
      }
    }
  }
  if (out_logprob) {
    /* 选中的 token 若被截断掉，返回它的真实 logprob（与 HF 一致：报告模型分布下的值）*/
    *out_logprob = log_softmax_at(lg, n, pick);
  }
  sgl_xfree(cs);
  free(tmp);
  return pick;
}

void sgl_dist_transform(const float *logits, int n, const sgl_sampling_params *p,
                        const int *history, int n_history, float *out_probs) {
  if (!out_probs || n <= 0) return;
  float *tmp = NULL;
  const float *lg = penalized(logits, n, p, history, n_history, &tmp);
  cand *cs = NULL;
  int m = 0;
  double renorm = 1;
  build_cands(lg, n, p, &cs, &m, &renorm, out_probs);
  sgl_xfree(cs);
  free(tmp);
}

int sgl_sample(const float *logits, int n, const sgl_sampling_params *p, uint64_t *rng) {
  return sgl_sample_ex(logits, n, p, rng, NULL, 0, NULL);
}