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

int sgl_sample_ex(const float *logits, int n, const sgl_sampling_params *p, uint64_t *rng,
                  const int *history, int n_history, float *out_logprob) {
  float *tmp = NULL;
  const float *lg = logits;
  if (p && (p->repetition_penalty > 0.0f && p->repetition_penalty != 1.0f ||
            p->frequency_penalty != 0.0f || p->presence_penalty != 0.0f) &&
      n_history > 0 && history) {
    tmp = (float *)malloc(sizeof(float) * (size_t)n);
    if (tmp) {
      apply_penalties(logits, n, p, history, n_history, tmp);
      lg = tmp;
    }
  }

  if (!p || p->temperature <= 0.0f) {
    int pick = sgl_sample_greedy(lg, n);
    if (out_logprob) *out_logprob = log_softmax_at(lg, n, pick);
    free(tmp);
    return pick;
  }

  /* softmax(logits/T) */
  float *probs = (float *)sgl_xmalloc(sizeof(float) * (size_t)n);
  int max_i = sgl_sample_greedy(lg, n);
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
  double r = (double)(sgl_rng_next(rng) >> 11) / 9007199254740992.0; /* [0,1) */
  double acc = 0;
  int pick = cs[0].idx;
  for (int i = 0; i < m; i++) {
    acc += cs[i].p / renorm;
    if (r < acc) {
      pick = cs[i].idx;
      break;
    }
  }
  if (out_logprob) {
    /* 选中的 token 若被截断掉，返回它的真实 logprob（与 HF 一致：报告模型分布下的值）*/
    *out_logprob = log_softmax_at(lg, n, pick);
  }
  sgl_xfree(probs);
  sgl_xfree(cs);
  free(tmp);
  return pick;
}

int sgl_sample(const float *logits, int n, const sgl_sampling_params *p, uint64_t *rng) {
  return sgl_sample_ex(logits, n, p, rng, NULL, 0, NULL);
}