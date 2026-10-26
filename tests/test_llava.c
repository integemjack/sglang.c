/* test_llava.c — t10 LLaVA 多模态路径验证（随机权重结构对照）
 * 1. CLIP encode（conv patch embed + transformer + projection）与 numpy 全等
 * 2. mm_projector（单层 linear，对应 llava-1.5 的两层 MLP 简化）与 numpy 全等
 * 3. 融合路径：<image> 展开 NP 个视觉 token 后的 embedding 前向可用（数值由 1+2 保证） */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/clip.h"
#include "sgl/llama.h"
#include "sgl/log.h"
#include "sgl/ops.h"
#include "sgl/safetensors.h"

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

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_LLAVA");
  if (!dir || !*dir) dir = "models/llava-rand";
  const char *ref = argc > 2 ? argv[2] : "ref_llava.json";

  sgl_clip *c = sgl_clip_load(dir);
  CHECK(c != NULL, "clip load");
  if (!c) return 1;
  printf("[clip] E=%d H=%d L=%d P=%d img=%d NP=%d proj=%d\n",
         c->E, c->H, c->L, c->P, c->img, c->n_patch, c->proj_dim);

  /* load image.raw [3*img*img] CHW */
  char path[1024];
  snprintf(path, sizeof(path), "%s/image.raw", dir);
  FILE *f = NULL;
  fopen_s(&f, path, "rb");
  CHECK(f != NULL, "image.raw open");
  if (!f) return 1;
  size_t npx = (size_t)3 * c->img * c->img;
  float *img = (float *)sgl_xmalloc(sizeof(float) * npx);
  CHECK(fread(img, sizeof(float), npx, f) == npx, "image read");
  fclose(f);

  int NP = c->n_patch, PROJ = c->proj_dim;
  float *feats = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * PROJ);
  CHECK(sgl_clip_encode(c, img, feats) == 0, "clip encode");
  printf("[dbg] C feats[0][:4] = %g %g %g %g  feats[NP-1][0] = %g\n",
         feats[0], feats[1], feats[2], feats[3], feats[(size_t)(NP - 1) * PROJ]);

  /* projector: linear_1 [LM_E, PROJ] */
  snprintf(path, sizeof(path), "%s/projector.safetensors", dir);
  sgl_st_ctx *pst = sgl_st_open(path);
  CHECK(pst != NULL, "projector open");
  sgl_tensor w1 = sgl_st_get(pst, "linear_1.weight", 2, (int64_t[]){64, PROJ});
  sgl_tensor b1 = sgl_st_get(pst, "linear_1.bias", 1, (int64_t[]){64});
  CHECK(w1.data && b1.data, "projector tensors");
  float *proj_c = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * 64);
  sgl_matmul_nt(feats, w1.data, b1.data, proj_c, NP, 64, PROJ);

  /* compare against numpy .npy (row-major float32 [NP, 64]) */
  snprintf(path, sizeof(path), "%s/_ref_proj.npy", dir);
  f = NULL;
  fopen_s(&f, path, "rb");
  CHECK(f != NULL, "ref npy open (run ref_llava.py first)");
  if (!f) return 1;
  fseek(f, 0, SEEK_END);
  long fsz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *nb = (char *)sgl_xmalloc((size_t)fsz);
  if (fread(nb, 1, (size_t)fsz, f) != (size_t)fsz) return 1;
  fclose(f);
  /* npy v1 header: \x93NUMPY... — find data start (header len aligned to 64) */
  unsigned short hlen = *(unsigned short *)(nb + 8);
  const float *rproj = (const float *)(nb + 10 + hlen);
  printf("[dbg] npy hlen=%u first8 = %g %g %g %g\n", hlen, rproj[0], rproj[1], rproj[2], rproj[3]);
  double maxerr = 0;
  for (size_t i = 0; i < (size_t)NP * 64; i++) {
    double e = fabs((double)proj_c[i] - rproj[i]);
    if (e > maxerr) maxerr = e;
  }
  printf("[llava] CLIP+projector vs numpy: maxerr = %.3g over %d x 64\n", maxerr, NP);
  CHECK(maxerr < 2e-4, "llava numerics diverge: %g", maxerr);

  /* ---------- 3) 融合路径：emb 版前向 == token 版前向（同一层循环） ----------
   * 用 mistral-rand 的 wte 行当 embedding 序列——数值路径一致性由全等断言保证。 */
  {
    sgl_llama *lm = sgl_llama_load("models/mistral-rand");
    if (lm) {
      int ids5[5] = {9707, 1879, 4321, 100, 999};
      int *pg5 = (int *)sgl_xmalloc(sizeof(int) * 5);
      CHECK(sgl_llama_pages_alloc(lm, pg5, 5) == 0, "fusion pages");
      float *lg_t = (float *)sgl_xmalloc(sizeof(float) * (size_t)lm->vocab);
      float *lg_e = (float *)sgl_xmalloc(sizeof(float) * (size_t)lm->vocab);
      CHECK(sgl_llama_forward(lm, ids5, 5, 0, pg5, lg_t) == 0, "fusion token fwd");
      /* emb 序列：直接从 wte 拼行 */
      float *emb5 = (float *)sgl_xmalloc(sizeof(float) * 5 * (size_t)lm->n_embd);
      for (int i = 0; i < 5; i++)
        memcpy(emb5 + (size_t)i * lm->n_embd,
               lm->wte.data + (size_t)ids5[i] * lm->n_embd,
               sizeof(float) * (size_t)lm->n_embd);
      CHECK(sgl_llama_forward_emb(lm, emb5, 5, 0, pg5, lg_e) == 0, "fusion emb fwd");
      double ferr = 0;
      for (int v2 = 0; v2 < lm->vocab; v2++) {
        double e2 = fabs((double)lg_t[v2] - lg_e[v2]);
        if (e2 > ferr) ferr = e2;
      }
      printf("[fusion] forward_emb(wte rows) == forward(tokens): maxerr = %.3g\n", ferr);
      CHECK(ferr < 1e-5, "fusion path diverges: %g", ferr);
      sgl_xfree(emb5); sgl_xfree(lg_t); sgl_xfree(lg_e); sgl_xfree(pg5);
      sgl_llama_free(lm);
    } else {
      printf("[fusion] models/mistral-rand missing, skipped\n");
    }
  }

  sgl_xfree(nb);
  sgl_xfree(proj_c); sgl_xfree(feats); sgl_xfree(img);
  sgl_st_close(pst);
  sgl_clip_free(c);
  if (FAILURES) { printf("test_llava: %d FAILURES\n", FAILURES); return 1; }
  printf("test_llava: all OK\n");
  return 0;
}
