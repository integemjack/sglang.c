/* sgl/sampling.h — 采样（对应 sglang srt/sampling/sampling_batch_info.py 的单条路径）*/
#ifndef SGL_SAMPLING_H
#define SGL_SAMPLING_H

#include <stdint.h>

typedef struct {
  float temperature; /* <=0 退化为 greedy */
  int top_k;         /* <=0 不启用 */
  float top_p;       /* (0,1] 外不启用 */
  uint64_t seed;
} sgl_sampling_params;

uint64_t sgl_rng_next(uint64_t *state); /* xorshift64* */

/* greedy：返回 argmax */
int sgl_sample_greedy(const float *logits, int n);

/* temperature/top-k/top-p 采样 */
int sgl_sample(const float *logits, int n, const sgl_sampling_params *p,
               uint64_t *rng);

#endif
