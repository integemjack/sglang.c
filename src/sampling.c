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

typedef struct {
  int idx;
  float p;
} cand;

static int cmp_desc(const void *a, const void *b) {
  float x = ((const cand *)a)->p, y = ((const cand *)b)->p;
  return (x < y) - (x > y);
}

int sgl_sample(const float *logits, int n, const sgl_sampling_params *p,
               uint64_t *rng) {
  if (p->temperature <= 0.0f) return sgl_sample_greedy(logits, n);

  /* softmax(logits/T)：数值稳定 */
  float *probs = (float *)sgl_xmalloc(sizeof(float) * (size_t)n);
  int max_i = sgl_sample_greedy(logits, n);
  double T = p->temperature;
  double sum = 0;
  for (int i = 0; i < n; i++) {
    probs[i] = (float)exp((logits[i] - logits[max_i]) / T);
    sum += probs[i];
  }
  for (int i = 0; i < n; i++) probs[i] /= (float)sum;

  /* 候选集：top-k 截断 → top-p 截断 */
  cand *cs = (cand *)sgl_xmalloc(sizeof(cand) * (size_t)n);
  for (int i = 0; i < n; i++) cs[i] = (cand){i, probs[i]};
  qsort(cs, (size_t)n, sizeof(cand), cmp_desc);

  int m = n;
  if (p->top_k > 0 && p->top_k < n) m = p->top_k;
  if (p->top_p > 0.0f && p->top_p < 1.0f) {
    double cum = 0;
    int cut = m;
    for (int i = 0; i < m; i++) {
      cum += cs[i].p;
      if (cum >= p->top_p) { cut = i + 1; break; }
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
    if (r < acc) { pick = cs[i].idx; break; }
  }
  sgl_xfree(probs);
  sgl_xfree(cs);
  return pick;
}
