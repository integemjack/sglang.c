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
#include "sgl/image.h"
#include "sgl/llama.h"
#include "sgl/log.h"
#include "sgl/ops.h"
#include "sgl/port.h"
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

/* 读一个 float32 C-contiguous .npy（v1.0 头，2 字节 header len）；失败 NULL。
 * 返回 malloc 的数组（调用方 free），*n_el = 元素个数。 */
static float *npy_load(const char *dir, const char *name, long *n_el) {
  char tmp[1024];
  snprintf(tmp, sizeof(tmp), "%s/%s", dir, name);
  FILE *f = NULL;
  fopen_s(&f, tmp, "rb");
  if (!f) { printf("npy open fail: %s\n", tmp); return NULL; }
  fseek(f, 0, SEEK_END);
  long fsz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *nb = (char *)malloc((size_t)fsz);
  if (!nb || fread(nb, 1, (size_t)fsz, f) != (size_t)fsz) { fclose(f); free(nb); return NULL; }
  fclose(f);
  if (fsz < 12 || memcmp(nb, "\x93" "NUMPY", 6) != 0) { free(nb); return NULL; }
  unsigned short hlen = *(unsigned short *)(nb + 8);
  long dstart = 10 + (long)hlen;
  /* 从 header 里的 shape 算元素个数："shape': (16, 64), " 或 (64,) */
  const char *sp = strstr(nb + 10, "shape");
  long n = 1;
  if (sp) {
    sp = strchr(sp, '(') + 1;
    while (*sp && *sp != ')') {
      if (*sp >= '0' && *sp <= '9') {
        long v = 0;
        while (*sp >= '0' && *sp <= '9') { v = v * 10 + (*sp - '0'); sp++; }
        n *= v;
      } else sp++;
    }
  }
  long cnt = (fsz - dstart) / 4;
  if (n <= 0) n = cnt;
  float *out = (float *)malloc(sizeof(float) * (size_t)cnt);
  if (!out) { free(nb); return NULL; }
  memcpy(out, nb + dstart, sizeof(float) * (size_t)cnt);
  free(nb);
  if (n_el) *n_el = n;
  return out;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_LLAVA");
  if (!dir || !*dir) dir = "models/llava-rand";
  const char *ref = argc > 2 ? argv[2] : "ref_llava.json";
  (void)ref; /* G12 起比对的是目录里的 _ref_{feats,proj,imgemb}.npy，这个参数只为兼容旧调用 */

  /* CI / fresh clone 里没有随机多模态模型（gen_random_llava.py 生成，不入库）：
   * 缺模型就 SKIP，别把「模型没生成」变成「测试失败」——t9/t10 时代 CI 就
   * 一直红在这一条（v0.2.0 那次 5 个 job 因此失败，release 被 skipped）。
   * 其他 test_*（test_llama / test_qwen / test_beam / test_bpe）早就是这种写法。 */
  {
    char probe[1024];
    snprintf(probe, sizeof(probe), "%s/clip_model.safetensors", dir);
    FILE *pf = NULL;
    fopen_s(&pf, probe, "rb");
    if (!pf) {
      printf("test_llava: SKIP（无 %s；跑 tests/gen_random_llava.py 生成）\n", dir);
      return 0;
    }
    fclose(pf);
    snprintf(probe, sizeof(probe), "%s/image.raw", dir);
    fopen_s(&pf, probe, "rb");
    if (!pf) {
      printf("test_llava: SKIP（无 %s；跑 tests/gen_random_llava.py 生成）\n", probe);
      return 0;
    }
    fclose(pf);
  }

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

  int NP = c->n_patch, PROJ = c->proj_dim;   /* 真实语义：PROJ == c->E（encode 行宽）*/
  float *feats = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * PROJ);
  CHECK(sgl_clip_encode(c, img, feats) == 0, "clip encode");
  printf("[dbg] C feats[0][:4] = %g %g %g %g  feats[NP-1][0] = %g\n",
         feats[0], feats[1], feats[2], feats[3], feats[(size_t)(NP - 1) * PROJ]);

  /* projector: G12 自动判层数（linear_1 可能是 [mid, PROJ]，两层 MLP 再加 linear_2）*/
  snprintf(path, sizeof(path), "%s/projector.safetensors", dir);
  sgl_st_ctx *pst = sgl_st_open(path);
  CHECK(pst != NULL, "projector open");
  sgl_tensor w1 = sgl_st_get(pst, "linear_1.weight", -1, NULL);
  sgl_tensor b1 = sgl_st_get(pst, "linear_1.bias", -1, NULL);
  sgl_tensor w2 = sgl_st_get(pst, "linear_2.weight", -1, NULL);
  sgl_tensor b2 = w2.data ? sgl_st_get(pst, "linear_2.bias", -1, NULL) : sgl_tensor_null();
  CHECK(w1.data && b1.data, "projector tensors");
  int LM_E = w2.data ? (int)w2.shape[0] : (int)w1.shape[0];
  sgl_projector pj;
  CHECK(sgl_projector_init(&pj, w1.data, b1.data, w1.shape[0], PROJ,
                           w2.data, b2.data, w2.data ? w2.shape[0] : 0,
                           w2.data ? w2.shape[1] : 0) == 0, "projector init");
  float *proj_c = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * (size_t)LM_E);
  sgl_projector_apply(&pj, feats, NP, proj_c);

  /* 分层对照：feats（patch 行）与 numpy 全等 */
  {
    long nel = 0;
    float *rf = npy_load(dir, "_ref_feats.npy", &nel);
    CHECK(rf != NULL && nel == (long)NP * PROJ, "ref feats npy (run ref_llava.py first)");
    if (rf) {
      double e = 0;
      for (long i = 0; i < (long)NP * PROJ; i++) {
        double d = fabs((double)feats[i] - rf[i]);
        if (d > e) e = d;
      }
      printf("[llava] CLIP patch feats vs ref(_ref_feats.npy; numpy 或 MLX): maxerr = %.3g\n", e);
      CHECK(e < 2e-4, "feats diverge: %g", e);
      free(rf);
    }
    /* pooled cls + image_embeds（visual_projection 只作用 pooled）*/
    float *pooled = (float *)sgl_xmalloc(sizeof(float) * (size_t)c->E);
    float *emb = (float *)sgl_xmalloc(sizeof(float) * (size_t)(c->img_proj_dim > 0 ? c->img_proj_dim : 1));
    CHECK(sgl_clip_encode_full(c, img, NULL, pooled, emb, NULL) == 0, "clip encode_full");
    nel = 0;
    float *ri = npy_load(dir, "_ref_imgemb.npy", &nel);
    CHECK(ri != NULL && nel == (long)c->img_proj_dim, "ref imgemb npy");
    if (ri) {
      double e = 0;
      for (long i = 0; i < nel; i++) {
        double d = fabs((double)emb[i] - ri[i]);
        if (d > e) e = d;
      }
      printf("[llava] image_embeds(pooled->visual_projection) vs ref: maxerr = %.3g\n", e);
      CHECK(e < 2e-4, "image_embeds diverge: %g", e);
      free(ri);
    }
    sgl_xfree(pooled); sgl_xfree(emb);
  }

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
  for (size_t i = 0; i < (size_t)NP * (size_t)LM_E; i++) {
    double e = fabs((double)proj_c[i] - rproj[i]);
    if (e > maxerr) maxerr = e;
  }
  printf("[llava] CLIP+projector(%s) vs ref(_ref_proj.npy; numpy 或 MLX): maxerr = %.3g over %d x %d\n",
         w2.data ? "2-layer MLP" : "1-layer linear", maxerr, NP, LM_E);
  CHECK(maxerr < 2e-4, "llava numerics diverge: %g", maxerr);

  /* ---------- G8 图片解码：真实 JPEG/PNG 字节 → 解码 → 与 PIL 期望像素对照 ----------
   * 素材由 tests/clip_real_ref.py 生成（test.png / test.jpg / ref/expect_norm.npy，
   * 期望值 = PIL BICUBIC resize 224×224 → /255 → (x-mean)/std）。目录不存在则跳过。 */
  {
    const char *cdir = getenv("SGL_TEST_CLIP_REAL");
    if (cdir && *cdir) {
      long en0 = 0;
      float *probe = npy_load(cdir, "ref/expect_norm_png.npy", &en0);
      if (probe) {
        int sz = c->img; /* 随机模型是 32；真实目录里是 224，按 ref 尺寸推 */
        long side = (long)sqrt((double)(en0 / 3));
        if (side * side * 3 == en0) sz = (int)side;
        free(probe);
        float *chw = (float *)sgl_xmalloc(sizeof(float) * 3 * (size_t)sz * (size_t)sz);
        const char *names[2] = {"test.png", "test.jpg"};
        const char *rgbref[2] = {"ref/png_rgb.npy", "ref/jpg_rgb.npy"};
        const char *normref[2] = {"ref/expect_norm_png.npy", "ref/expect_norm_jpg.npy"};
        for (int it = 0; it < 2; it++) {
          char fp[1024];
          snprintf(fp, sizeof(fp), "%s/%s", cdir, names[it]);
          FILE *im = NULL;
          fopen_s(&im, fp, "rb");
          if (!im) { printf("[image] %s 缺失，跳过\n", names[it]); continue; }
          fseek(im, 0, SEEK_END);
          long isz = ftell(im);
          fseek(im, 0, SEEK_SET);
          unsigned char *bytes = (unsigned char *)malloc((size_t)isz);
          size_t rd = fread(bytes, 1, (size_t)isz, im);
          fclose(im);
          int iw = 0, ih = 0;
          unsigned char *rgb = NULL;
          CHECK(rd == (size_t)isz, "%s 读取", names[it]);
          CHECK(sgl_image_decode_rgb(bytes, rd, &rgb, &iw, &ih) == 0, "%s 解码", names[it]);
          if (rgb) {
            /* a) 解码结果直接对 PIL：PNG 无损，必须逐字节全等；
             *    JPEG 比 stb 与 libjpeg（PIL 用的）的解码差，只报数不卡死 */
            long rn = 0;
            float *rr = npy_load(cdir, rgbref[it], &rn);
            CHECK(rr != NULL && rn == (long)iw * ih * 3, "%s 的 PIL 解码参考", names[it]);
            if (rr) {
              double e = 0;
              for (long i = 0; i < rn; i++) {
                double d = fabs((double)rgb[i] - rr[i]);
                if (d > e) e = d;
              }
              printf("[image] %s 解码 vs PIL（%dx%d）：maxdiff = %g 像素级\n",
                     names[it], iw, ih, e);
              if (it == 0) CHECK(e == 0, "PNG 解码应与 PIL 逐字节全等：%g", e);
              free(rr);
            }
            /* b) 走完整预处理链后与 PIL+HF processor 的归一化张量对 */
            long en = 0;
            float *expect = npy_load(cdir, normref[it], &en);
            CHECK(expect != NULL && en == (long)3 * sz * sz, "%s 的归一化参考", names[it]);
            if (expect) {
            sgl_clip_preprocess(rgb, iw, ih, sz, NULL, NULL, chw);
            double e = 0;
            for (long i = 0; i < en; i++) {
              double d = fabs((double)chw[i] - expect[i]);
              if (d > e) e = d;
            }
            /* 阈值：1 个像素级 = 1/255/std ≈ 0.015；PNG 只允许 resize 舍入差，
             * JPEG 还叠加两个解码器的色度重建差（实测见 NOTES）*/
            double thr = it == 0 ? 0.03 : 0.30;
            printf("[image] %s (%dx%d) → %d×%d CHW vs PIL: maxerr = %.3g（阈值 %g）\n",
                   names[it], iw, ih, sz, sz, e, thr);
            CHECK(e < thr, "%s 解码/预处理与 PIL 差太多：%.3g", names[it], e);
            free(expect);
            }
            sgl_xfree(rgb);
          }
          free(bytes);
        }
        /* base64 往返：把 test.png 的字节 base64 编码再解回来，应与原字节全等 */
        {
          char fp[1024];
          snprintf(fp, sizeof(fp), "%s/test.png", cdir);
          FILE *im = NULL;
          fopen_s(&im, fp, "rb");
          if (im) {
            fseek(im, 0, SEEK_END);
            long isz = ftell(im);
            fseek(im, 0, SEEK_SET);
            unsigned char *bytes = (unsigned char *)malloc((size_t)isz);
            if (fread(bytes, 1, (size_t)isz, im) == (size_t)isz) {
              static const char TAB[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
              size_t bn = 4 * ((size_t)isz + 2) / 3;
              char *b64 = (char *)malloc(bn + 1);
              size_t o = 0;
              for (long i = 0; i < isz; i += 3) {
                unsigned v = (unsigned)bytes[i] << 16;
                if (i + 1 < isz) v |= bytes[i + 1] << 8;
                if (i + 2 < isz) v |= bytes[i + 2];
                b64[o++] = TAB[(v >> 18) & 63];
                b64[o++] = TAB[(v >> 12) & 63];
                b64[o++] = (i + 1 < isz) ? TAB[(v >> 6) & 63] : '=';
                b64[o++] = (i + 2 < isz) ? TAB[v & 63] : '=';
              }
              b64[o] = 0;
              unsigned char *dec = NULL;
              size_t dn = 0;
              CHECK(sgl_base64_decode(b64, o, &dec, &dn) == 0, "base64 解码");
              CHECK(dn == (size_t)isz && memcmp(dec, bytes, (size_t)isz) == 0,
                    "base64 往返不等（%zu vs %ld）", dn, isz);
              printf("[image] base64 往返 %ld 字节：%s\n", isz,
                     (dn == (size_t)isz && !memcmp(dec, bytes, (size_t)isz)) ? "全等 ✓" : "✗");
              sgl_xfree(dec);
            }
            free(bytes);
            fclose(im);
          }
        }
        sgl_xfree(chw);
      } else {
        printf("[image] ref/expect_norm_png.npy 缺失（先跑 clip_real_ref.py），跳过解码测试\n");
      }
    } else {
      printf("[image] SGL_TEST_CLIP_REAL 未设，跳过解码测试\n");
    }
  }

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
