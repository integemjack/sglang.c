/* test_clip_real.c — G8: 真实 CLIP 权重分层对照
 * 参考：tests/clip_real_ref.py 用 torch + transformers 的真实 CLIPVisionModel
 * 逐层导出（pre_ln 之后、每个 encoder layer 之后、last_hidden_state、pooled、
 * image_embeds），本测试用同一 pixel_values 跑 C 视觉塔，逐层比 maxerr。
 * 阈值：逐层 fp32 < 1e-3；最终投影输出 < 1e-2。
 * 用法：test_clip_real <clip_dir>   （clip_dir 下需有 ref/*.npy） */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/clip.h"

static int FAILURES = 0;
#define CHECK(cond, ...)                                                        \
  do {                                                                          \
    if (!(cond)) {                                                              \
      FAILURES++;                                                               \
      printf("FAIL %s:%d: ", __FILE__, __LINE__);                               \
      printf(__VA_ARGS__);                                                      \
      printf("\n");                                                             \
    }                                                                           \
  } while (0)

/* 读 float32 C-contiguous .npy（v1.0 头）；*n_el = 元素个数。malloc/free 配对 */
static float *npy_load(const char *path, long *n_el) {
  FILE *f = NULL;
  fopen_s(&f, path, "rb");
  if (!f) { printf("npy open fail: %s\n", path); return NULL; }
  fseek(f, 0, SEEK_END);
  long fsz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *nb = (char *)malloc((size_t)fsz);
  if (!nb || fread(nb, 1, (size_t)fsz, f) != (size_t)fsz) { fclose(f); free(nb); return NULL; }
  fclose(f);
  if (fsz < 12 || memcmp(nb, "\x93" "NUMPY", 6) != 0) { free(nb); return NULL; }
  unsigned short hlen = *(unsigned short *)(nb + 8);
  long dstart = 10 + (long)hlen;
  long cnt = (fsz - dstart) / 4;
  float *out = (float *)malloc(sizeof(float) * (size_t)cnt);
  if (!out) { free(nb); return NULL; }
  memcpy(out, nb + dstart, sizeof(float) * (size_t)cnt);
  free(nb);
  if (n_el) *n_el = cnt;
  return out;
}

static double maxdiff(const float *a, const float *r, long n, long skip_rows, long row_len) {
  double e = 0;
  for (long i = skip_rows * row_len; i < n; i++) {
    double d = fabs((double)a[i] - r[i]);
    if (d > e) e = d;
  }
  return e;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_CLIP_REAL");
  if (!dir || !*dir) { printf("test_clip_real: SKIP（没给 clip 目录）\n"); return 0; }

  char path[1024];
  long nel = 0;
  snprintf(path, sizeof(path), "%s/ref/pv.npy", dir);
  float *pv = npy_load(path, &nel);
  CHECK(pv != NULL, "ref/pv.npy 缺失（先跑 tests/clip_real_ref.py）");
  if (!pv) return 1;

  sgl_clip *c = sgl_clip_load(dir);
  CHECK(c != NULL, "clip load %s", dir);
  if (!c) return 1;
  int E = c->E, NP = c->n_patch, rows = NP + 1, L = c->L;
  printf("[clip_real] E=%d H=%d L=%d P=%d img=%d I=%d NP=%d act=%d eps=%g img_proj=%d\n",
         c->E, c->H, c->L, c->P, c->img, c->I, NP, c->act, (double)c->eps, c->img_proj_dim);
  CHECK((long)3 * c->img * c->img == nel, "pv.npy numel %ld != %d", nel, 3 * c->img * c->img);
  CHECK(E == 768 && L == 12 && NP == 49, "不是 clip-vit-base-patch32 结构（E=%d L=%d NP=%d）", E, L, NP);

  float *dump = (float *)sgl_xmalloc(sizeof(float) * (size_t)(L + 1) * rows * E);
  float *patch = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * E);
  float *pooled = (float *)sgl_xmalloc(sizeof(float) * (size_t)E);
  float *emb = (float *)sgl_xmalloc(sizeof(float) * (size_t)(c->img_proj_dim > 0 ? c->img_proj_dim : 1));
  CHECK(sgl_clip_encode_full(c, pv, patch, pooled, emb, dump) == 0, "encode_full");

  /* 逐层 */
  double worst = 0;
  for (int b = 0; b <= L; b++) {
    char nm[64];
    if (b == 0) snprintf(nm, sizeof(nm), "preln.npy");
    else snprintf(nm, sizeof(nm), "layer%02d.npy", b - 1);
    snprintf(path, sizeof(path), "%s/ref/%s", dir, nm);
    long rn = 0;
    float *r = npy_load(path, &rn);
    CHECK(r != NULL && rn == (long)rows * E, "%s 读取（n=%ld 期望 %d）", nm, rn, rows * E);
    if (!r) continue;
    double e = maxdiff(dump + (size_t)b * rows * E, r, (long)rows * E, 0, E);
    printf("[layer %2d] %-14s maxerr = %.3g%s\n", b, nm, e, b == 0 ? "  (encoder 输入)" : "");
    if (e > worst) worst = e;
    CHECK(e < 1e-3, "layer %d 逐层 fp32 阈值 1e-3 超：%.3g", b, e);
    free(r);
  }
  printf("[clip_real] 逐层最大 maxerr = %.3g（阈值 1e-3）\n", worst);

  /* last_hidden_state 的 patch 行（LLaVA 取的视觉 token）*/
  {
    snprintf(path, sizeof(path), "%s/ref/last.npy", dir);
    long rn = 0;
    float *r = npy_load(path, &rn);
    CHECK(r != NULL && rn == (long)rows * E, "last.npy 读取");
    if (r) {
      /* patch[i] 对应 r 的第 i+1 行（r 的第 0 行是 class）*/
      double e = 0;
      for (long i = E; i < (long)rows * E; i++) {
        double d = fabs((double)patch[i - E] - r[i]);
        if (d > e) e = d;
      }
      printf("[final ] last_hidden_state[1:] maxerr = %.3g\n", e);
      CHECK(e < 1e-3, "last_hidden_state 超阈值：%.3g", e);
      free(r);
    }
  }
  /* pooled + image_embeds（最终投影输出）*/
  {
    snprintf(path, sizeof(path), "%s/ref/pooled.npy", dir);
    long rn = 0;
    float *r = npy_load(path, &rn);
    CHECK(r != NULL && rn == (long)E, "pooled.npy 读取");
    if (r) {
      double e = maxdiff(pooled, r, (long)E, 0, E);
      printf("[pooled] post_layernorm(cls) maxerr = %.3g\n", e);
      CHECK(e < 1e-3, "pooled 超阈值：%.3g", e);
      free(r);
    }
    snprintf(path, sizeof(path), "%s/ref/image_embeds.npy", dir);
    r = npy_load(path, &rn);
    CHECK(r != NULL && rn == (long)c->img_proj_dim, "image_embeds.npy 读取");
    if (r) {
      double e = maxdiff(emb, r, (long)rn, 0, 1);
      double base = 0;
      for (long i = 0; i < rn; i++) { double a = fabs(r[i]); if (a > base) base = a; }
      printf("[proj  ] image_embeds maxerr = %.3g（|ref|max = %.3g）\n", e, base);
      CHECK(e < 1e-2, "image_embeds（最终投影）超阈值：%.3g", e);
      free(r);
    }
  }

  sgl_xfree(dump); sgl_xfree(patch); sgl_xfree(pooled); sgl_xfree(emb);
  free(pv); /* npy_load 用 malloc，必须用 free（不能进 sgl_xfree）*/
  sgl_clip_free(c);
  if (FAILURES) { printf("test_clip_real: %d FAILURES\n", FAILURES); return 1; }
  printf("test_clip_real: all OK\n");
  return 0;
}
