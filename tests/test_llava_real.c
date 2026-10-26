/* G12 验收（真实权重，分段对照）：真实 CLIP-L/14-336（openai/clip-vit-large-patch14-336）
 * + 真实 llava-1.5 的 mm_projector（mlp2x_gelu）→ 与 HF transformers 的 CPU 参考逐层对照。
 *
 * 为什么是「分段」：llava-1.5-7b 的 LM 是 fp16 13.5GB，本机剩余磁盘放不下（见
 * docs/NOTES_g8.md G12 节），任务里也允许「只抽 projector + CLIP 做分段对照」。
 *
 * 参考由 tests/llava_real_ref.py 生成（torch + transformers，同一张图）：
 *   _ref_pv.npy         CLIPImageProcessor 归一化后的 CHW [3,336,336]
 *   _ref_vis.npy        hidden_states[mm_vision_select_layer][:,1:,:] → [576,1024]
 *   _ref_real_proj.npy  mm_projector 输出 [576,4096]
 *
 * 用法: test_llava_real <clip_dir> <proj_dir>
 *      （默认 $SGL_TEST_LLAVA_REAL_CLIP / $SGL_TEST_LLAVA_REAL_PROJ）
 */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/image.h"

static int FAILURES = 0;
#define CHECK(cond, ...) do { if (!(cond)) { FAILURES++; printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                              printf(__VA_ARGS__); printf("\n"); } } while (0)

/* 读 .npy（v1.0，<f4，C 序）：返回 malloc 的 float，*n 写元素数 */
static float *npy_load(const char *path, long *n) {
  FILE *f = NULL;
#ifdef _MSC_VER
  if (fopen_s(&f, path, "rb") != 0) f = NULL;
#else
  f = fopen(path, "rb");
#endif
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)malloc((size_t)sz);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
  fclose(f);
  unsigned short hlen = *(unsigned short *)(buf + 8);
  const float *data = (const float *)(buf + 10 + hlen);
  long nl = (sz - 10 - hlen) / 4;
  float *out = (float *)malloc(sizeof(float) * (size_t)nl);
  memcpy(out, data, sizeof(float) * (size_t)nl);
  free(buf);
  if (n) *n = nl;
  return out;
}

static void cmp(const char *what, const float *a, const float *b, long n) {
  double e = 0, mx = 0;
  for (long i = 0; i < n; i++) {
    double d = fabs((double)a[i] - (double)b[i]);
    if (d > e) e = d;
    if (fabs((double)b[i]) > mx) mx = fabs((double)b[i]);
  }
  double rel = mx > 0 ? e / mx : e;
  printf("[real] %-22s maxerr = %.3e (ref max |v| = %.3g, relative %.2e)\n", what, e, mx, rel);
  CHECK(rel < 1e-3, "%s relative error %.2e too large (abs %.3e)", what, rel, e);
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *cdir = argc > 1 ? argv[1] : getenv("SGL_TEST_LLAVA_REAL_CLIP");
  const char *pdir = argc > 2 ? argv[2] : getenv("SGL_TEST_LLAVA_REAL_PROJ");
  if (!cdir || !*cdir) cdir = "models/clip-vit-large-patch14-336";
  if (!pdir || !*pdir) pdir = "models/llava-real";
  char path[1024];
  snprintf(path, sizeof(path), "%s/_ref_vis.npy", cdir);
  FILE *probe = fopen(path, "rb");
  if (!probe) {
    printf("SKIP: 缺 %s（先跑 tests/llava_real_ref.py 和 convert_real_llava.py）\n", path);
    return 0;
  }
  fclose(probe);

  /* ---- 1) 预处理：PNG → 归一化 CHW，vs CLIPImageProcessor ---- */
  snprintf(path, sizeof(path), "%s/test336.png", cdir);
  FILE *f = NULL;
  f = fopen(path, "rb");
  CHECK(f != NULL, "test336.png");
  if (!f) return 1;
  fseek(f, 0, SEEK_END);
  long pn = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *png = (unsigned char *)malloc((size_t)pn);
  if (fread(png, 1, (size_t)pn, f) != (size_t)pn) { fclose(f); return 1; }
  fclose(f);

  sgl_clip *c = sgl_clip_load(cdir);
  CHECK(c != NULL, "clip load（真实 CLIP-L/14-336）");
  if (!c) return 1;
  printf("[real] CLIP: E=%d L=%d P=%d img=%d NP=%d select_layer=%d act=%d\n",
         c->E, c->L, c->P, c->img, c->n_patch, c->sel_layer, c->act);
  CHECK(c->img == 336 && c->P == 14 && c->E == 1024 && c->L == 24, "真实 CLIP-L/14-336 形状");

  unsigned char *rgb = NULL;
  int w = 0, h = 0;
  CHECK(sgl_image_decode_rgb(png, (size_t)pn, &rgb, &w, &h) == 0, "decode test336.png");
  CHECK(w == 336 && h == 336, "图片尺寸 %dx%d", w, h);
  float *chw = (float *)sgl_xmalloc(sizeof(float) * 3 * 336 * 336);
  sgl_clip_preprocess(rgb, w, h, 336, NULL, NULL, chw);
  sgl_xfree(rgb);

  long npv = 0;
  snprintf(path, sizeof(path), "%s/_ref_pv.npy", cdir);
  float *ref_pv = npy_load(path, &npv);
  CHECK(ref_pv && npv == 3L * 336 * 336, "_ref_pv.npy");
  if (ref_pv) cmp("preprocess (pv)", chw, ref_pv, npv);

  /* ---- 2) 视觉塔 patch 行 vs HF hidden_states[-2][:,1:,:] ---- */
  int NP = c->n_patch, E = c->E;
  float *feats = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * (size_t)E);
  CHECK(sgl_clip_encode(c, chw, feats) == 0, "clip encode");
  long nvis = 0;
  snprintf(path, sizeof(path), "%s/_ref_vis.npy", cdir);
  float *ref_vis = npy_load(path, &nvis);
  CHECK(ref_vis && nvis == (long)NP * E, "_ref_vis.npy（%ld vs %d）", nvis, NP * E);
  if (ref_vis) cmp("CLIP patch feats", feats, ref_vis, nvis);

  /* ---- 3) 真实 mm_projector（mlp2x_gelu，1024→4096→4096）---- */
  snprintf(path, sizeof(path), "%s/projector.safetensors", pdir);
  sgl_st_ctx *pst = sgl_st_open(path);
  CHECK(pst != NULL, "projector.safetensors（真实 llava mm_projector）");
  if (!pst) return 1;
  sgl_tensor w1 = sgl_st_get(pst, "linear_1.weight", -1, NULL);
  sgl_tensor b1 = sgl_st_get(pst, "linear_1.bias", -1, NULL);
  sgl_tensor w2 = sgl_st_get(pst, "linear_2.weight", -1, NULL);
  sgl_tensor b2 = sgl_st_get(pst, "linear_2.bias", -1, NULL);
  CHECK(w1.data && b1.data && w2.data && b2.data, "projector 张量");
  CHECK(w1.shape[0] == 4096 && w1.shape[1] == 1024 && w2.shape[0] == 4096 &&
        w2.shape[1] == 4096, "projector 形状应是 1024->4096->4096");
  sgl_projector pj;
  CHECK(sgl_projector_init(&pj, w1.data, b1.data, w1.shape[0], w1.shape[1],
                           w2.data, b2.data, w2.shape[0], w2.shape[1]) == 0, "projector init");
  printf("[real] projector: %d -> %d -> %d (act=%d gelu)\n", pj.in, pj.mid, pj.out, pj.act);

  float *mine = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * (size_t)pj.out);
  sgl_projector_apply(&pj, feats, NP, mine);
  long npr = 0;
  snprintf(path, sizeof(path), "%s/_ref_real_proj.npy", pdir);
  float *ref_pr = npy_load(path, &npr);
  CHECK(ref_pr && npr == (long)NP * pj.out, "_ref_real_proj.npy（%ld vs %d）", npr, NP * pj.out);
  if (ref_pr) cmp("mm_projector out", mine, ref_pr, npr);

  free(png);
  sgl_xfree(chw);
  sgl_xfree(feats);
  sgl_xfree(mine);
  free(ref_pv);
  free(ref_vis);
  free(ref_pr);
  sgl_st_close(pst);
  sgl_clip_free(c);
  if (FAILURES) { printf("test_llava_real: %d FAILURES\n", FAILURES); return 1; }
  printf("test_llava_real: all OK\n");
  return 0;
}