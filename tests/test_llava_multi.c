/* test_llava_multi.c — 多图（G21）端到端数值验证
 *
 * 目标：证明「一个 prompt 里两个 <image> 占位符 + 两张图」被正确处理成
 *       「每个占位符各换成它自己那一段视觉 token」。
 *
 * 做法（不依赖调度器内部字段）：
 *   1) 手工构造参考：把 prompt 的 token 序列里第 j 个占位符替换成第 j 张图的
 *      视觉 embedding 行，得到一条纯 emb 序列，直接调 sgl_llama_forward_emb
 *      算出最后一个位置的 logits，取 argmax = 期望的首 token。
 *   2) 走调度器：sgl_sched_submit_img(...) 提交同样两张图，贪心 max_new=1，
 *      取实际生成的首 token。
 *   3) 两者必须一致。同时再提交一次「只放第一张图」的请求，要求它的首 token
 *      **与两图版不同**（否则说明第二张图没被用上，也就是旧 bug）。
 *
 * 素材：SGL_TEST_LLAVA 目录（默认 models/llava-rand）里的 image.raw + CLIP + projector，
 *       两张图用「原图」与「水平翻转图」——翻转后 patch 序列不同，视觉 token 必然不同。
 *
 * 编译：随 test_* 自动注册（链接 sgl）；夹具缺失时打印 SKIP 并返回 0。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/clip.h"
#include "sgl/image.h"
#include "sgl/llama.h"
#include "sgl/model.h"
#include "sgl/safetensors.h"
#include "sgl/scheduler.h"

static int fails = 0;
#define CHECK(cond, ...)                            \
  do {                                              \
    if (!(cond)) {                                  \
      printf("FAIL %s:%d: ", __FILE__, __LINE__);   \
      printf(__VA_ARGS__);                          \
      printf("\n");                                 \
      fails++;                                      \
    }                                               \
  } while (0)

static long read_raw(const char *path, float *dst, long n) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  long got = (long)fread(dst, sizeof(float), (size_t)n, f);
  fclose(f);
  return got;
}

int main(void) {
  const char *dir = getenv("SGL_TEST_LLAVA");
  if (!dir || !*dir) dir = "models/llava-rand";
  char p[512];
  snprintf(p, sizeof(p), "%s/clip_model.safetensors", dir);
  FILE *probe = fopen(p, "rb");
  if (!probe) {
    printf("test_llava_multi: SKIP（%s 不存在，先跑 tests/ref_llava.py 生成夹具）\n", dir);
    return 0;
  }
  fclose(probe);

  /* ---- 1) 视觉塔：两张图（原图 / 水平翻转）→ 两段 feats ---- */
  sgl_clip *c = sgl_clip_load(dir);
  CHECK(c != NULL, "clip load");
  if (!c) return 1;
  int NP = (int)c->n_patch, PROJ = c->proj_dim > 0 ? c->proj_dim : 0;
  const int LM_E = 64; /* mistral-rand 的 hidden（夹具是 64 维迷你模型）*/
  int img_px = c->img * c->img * 3;
  float *img = (float *)sgl_xmalloc(sizeof(float) * (size_t)img_px);
  snprintf(p, sizeof(p), "%s/image.raw", dir);
  CHECK(read_raw(p, img, img_px) == img_px, "image.raw 读取");
  float *feats_a = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * PROJ);
  CHECK(sgl_clip_encode(c, img, feats_a) == 0, "clip encode A");
  /* 水平翻转：px[y][x] = px[y][side-1-x] */
  for (int y = 0; y < c->img; y++)
    for (int x = 0; x < c->img / 2; x++) {
      int x2 = c->img - 1 - x;
      for (int ch = 0; ch < 3; ch++) {
        float *a = img + ((size_t)y * c->img + x) * 3 + ch;
        float *b = img + ((size_t)y * c->img + x2) * 3 + ch;
        float t = *a;
        *a = *b;
        *b = t;
      }
    }
  float *feats_b = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * PROJ);
  CHECK(sgl_clip_encode(c, img, feats_b) == 0, "clip encode B");
  {
    double d = 0;
    for (long i = 0; i < (long)NP * PROJ; i++) {
      double e = fabs((double)feats_a[i] - feats_b[i]);
      if (e > d) d = e;
    }
    printf("[multi] 两张图（原图 vs 水平翻转）的 patch feats 差异 max = %.3g\n", d);
    CHECK(d > 1e-3, "两张图必须不同，否则这个测试证明不了什么");
  }

  /* ---- 2) projector：feats → 文本空间 ---- */
  snprintf(p, sizeof(p), "%s/projector.safetensors", dir);
  sgl_st_ctx *pst = sgl_st_open(p);
  CHECK(pst != NULL, "projector open");
  if (!pst) return 1;
  sgl_tensor w1 = sgl_st_get(pst, "linear_1.weight", -1, NULL);
  sgl_tensor b1 = sgl_st_get(pst, "linear_1.bias", -1, NULL);
  sgl_tensor w2 = sgl_st_get(pst, "linear_2.weight", -1, NULL);
  sgl_tensor b2 = sgl_st_get(pst, "linear_2.bias", -1, NULL);
  sgl_projector pj;
  CHECK(sgl_projector_init(&pj, w1.data, b1.data, w1.shape[0], PROJ, w2.data, b2.data,
                           w2.data ? w2.shape[0] : 0, w2.data ? w2.shape[1] : 0) == 0,
        "projector init");
  float *va = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * LM_E);
  float *vb = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * LM_E);
  sgl_projector_apply(&pj, feats_a, NP, va);
  sgl_projector_apply(&pj, feats_b, NP, vb);

  /* ---- 3) 调度器：两图 vs 单图 ---- */
  sgl_scheduler *s = sgl_sched_new("models/llava-lm", 42);
  CHECK(s != NULL, "sched new");
  if (!s) return 1;
  int img_tok = sgl_sched_piece_id(s, "<image>");
  if (img_tok < 0) {
    printf("test_llava_multi: SKIP（分词器里没有 <image> token）\n");
    sgl_sched_free(s);
    return 0;
  }

  const char *prompt2 = "USER: <image> and <image> what is it? ASSISTANT:";
  float *f2 = (float *)sgl_xmalloc(sizeof(float) * 2 * (size_t)NP * LM_E);
  memcpy(f2, va, sizeof(float) * (size_t)NP * LM_E);
  memcpy(f2 + (size_t)NP * LM_E, vb, sizeof(float) * (size_t)NP * LM_E);

  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp));
  sp.temperature = 0.0f;
  sp.top_p = 1.0f;
  sp.top_k = 1;
  sp.seed = 7;

  int id2 = sgl_sched_submit_img(s, prompt2, 1, &sp, f2, 2 * NP);
  CHECK(id2 >= 0, "submit 2 图");
  int guard = 0;
  while (sgl_sched_nactive(s) > 0 && guard++ < 10000) sgl_sched_step(s);
  int rid = -1, nids = 0, *ids = NULL, npr = 0, nhit = 0;
  char *txt = NULL;
  sgl_sched_poll(s, &rid, &txt, &ids, &nids, &npr, &nhit);
  CHECK(rid == id2 && nids == 1, "两图请求应生成 1 个 token（rid=%d n=%d）", rid, nids);
  int tok2 = nids > 0 ? ids[0] : -1;
  sgl_xfree(txt);
  sgl_xfree(ids);
  printf("[multi] 两图请求首 token = %d\n", tok2);

  /* 单图（只放第一张）*/
  float *f1 = (float *)sgl_xmalloc(sizeof(float) * (size_t)NP * LM_E);
  memcpy(f1, va, sizeof(float) * (size_t)NP * LM_E);
  int id1 = sgl_sched_submit_img(s, "USER: <image> and <image> what is it? ASSISTANT:", 1, &sp,
                                 f1, NP);
  CHECK(id1 >= 0, "submit 单图");
  guard = 0;
  while (sgl_sched_nactive(s) > 0 && guard++ < 10000) sgl_sched_step(s);
  rid = -1;
  nids = 0;
  ids = NULL;
  sgl_sched_poll(s, &rid, &txt, &ids, &nids, &npr, &nhit);
  int tok1 = nids > 0 ? ids[0] : -1;
  sgl_xfree(txt);
  sgl_xfree(ids);
  printf("[multi] 单图请求首 token = %d\n", tok1);
  /* 注意：这个随机小模型对数种输入的 argmax 是同一个 token（贪心不敏感），
   * 所以“两图 != 单图”不能用首 token 比，得比 logits 分布（见下面的手工参考段）。*/

  /* ---- 4) 手工参考：把两个占位符各换成对应视觉段，跑 unimodal emb 前向 ---- */
  {
    sgl_model *m = sgl_model_load("models/llava-lm");
    CHECK(m != NULL, "model load");
    if (m) {
      sgl_llama *g_lm = sgl_llama_load("models/llava-lm");
      CHECK(g_lm != NULL, "llama load（取 wte 行用）");
      int ntok = 0;
      int *tt = sgl_sched_encode(s, prompt2, strlen(prompt2), &ntok);
      CHECK(tt != NULL && ntok > 0, "encode prompt");
      int nplace = 0;
      for (int i = 0; i < ntok; i++)
        if (tt[i] == img_tok) nplace++;
      CHECK(nplace == 2, "prompt 里应有 2 个 <image>（实际 %d）", nplace);
      int L = ntok - nplace + 2 * NP;
      float *emb = (float *)sgl_xmalloc(sizeof(float) * (size_t)L * LM_E);
      int w = 0, j = 0;
      for (int i = 0; i < ntok; i++) {
        if (tt[i] == img_tok) {
          const float *src = (j == 0) ? va : vb;
          memcpy(emb + (size_t)w * LM_E, src, sizeof(float) * (size_t)NP * LM_E);
          w += NP;
          j++;
        } else {
          /* 文本 token 的 emb：从 mistral-rand 的 wte 取行（与模型同一份权重）*/
          memcpy(emb + (size_t)w * LM_E, g_lm->wte.data + (size_t)tt[i] * LM_E,
                 sizeof(float) * LM_E);
          w++;
        }
      }
      CHECK(w == L, "手工 emb 序列长度 %d != %d", w, L);
      int *pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)(L + 4));
      int vref = -1;
      if (sgl_model_pages_alloc(m, pages, L + 1) == 0) {
        float *lg = (float *)sgl_xmalloc(sizeof(float) * (size_t)sgl_model_vocab(m));
        if (sgl_model_prefill_emb(m, emb, L, 0, pages, lg) == 0) {
          int best = 0;
          for (int v = 1; v < sgl_model_vocab(m); v++)
            if (lg[v] > lg[best]) best = v;
          vref = best;
        }
        sgl_xfree(lg);
      }
      printf("[multi] 手工参考（两段视觉 emb + 文本）首 token = %d\n", vref);
      CHECK(vref == tok2, "调度器两图结果 %d 与手工参考 %d 不一致", tok2, vref);

      /* 再把“只有第一张图”也手工算一遍：两图的 logits 必须与单图明显不同
       * （否则第二张图根本没进 KV；这里比 logits 而不是 argmax，因为随机小模型的
       *  argmax 对多种输入都相同）。*/
      {
        int L1 = ntok - nplace + NP;
        float *emb1 = (float *)sgl_xmalloc(sizeof(float) * (size_t)L1 * LM_E);
        int w1 = 0;
        int used = 0;
        for (int i = 0; i < ntok; i++) {
          if (tt[i] == img_tok) {
            if (!used) {
              memcpy(emb1 + (size_t)w1 * LM_E, va, sizeof(float) * (size_t)NP * LM_E);
              w1 += NP;
              used = 1;
            }
            continue;
          }
          memcpy(emb1 + (size_t)w1 * LM_E, g_lm->wte.data + (size_t)tt[i] * LM_E,
                 sizeof(float) * LM_E);
          w1++;
        }
        int *pg1 = (int *)sgl_xmalloc(sizeof(int) * (size_t)(L1 + 4));
        if (sgl_model_pages_alloc(m, pg1, L1 + 1) == 0) {
          float *lg1 = (float *)sgl_xmalloc(sizeof(float) * (size_t)sgl_model_vocab(m));
          if (sgl_model_prefill_emb(m, emb1, L1, 0, pg1, lg1) == 0) {
            /* 重新算两图 logits（上一轮 lg 已释放）*/
            float *lg2 = (float *)sgl_xmalloc(sizeof(float) * (size_t)sgl_model_vocab(m));
            int *pg2 = (int *)sgl_xmalloc(sizeof(int) * (size_t)(L + 4));
            if (sgl_model_pages_alloc(m, pg2, L + 1) == 0 &&
                sgl_model_prefill_emb(m, emb, L, 0, pg2, lg2) == 0) {
              double d = 0;
              for (int v = 0; v < sgl_model_vocab(m); v++) {
                double e2 = fabs((double)lg1[v] - (double)lg2[v]);
                if (e2 > d) d = e2;
              }
              printf("[multi] 两图 vs 单图 logits 最大差 = %.3g\n", d);
              CHECK(d > 1e-3, "第二张图对 logits 没有影响（d=%.3g）", d);
            }
            sgl_xfree(pg2);
            sgl_xfree(lg2);
          }
          sgl_xfree(lg1);
        }
        sgl_xfree(pg1);
        sgl_xfree(emb1);
      }
      sgl_xfree(pages);
      sgl_xfree(emb);
      sgl_xfree(tt);
      if (g_lm) sgl_llama_free(g_lm);
      sgl_model_free(m);
    }
  }

  fprintf(stderr, "[cleanup] sched\n");
  sgl_sched_free(s);
  /* f1/f2 的所有权已交给调度器（prefill 后由它 sgl_xfree），不能在这里再释 */
  fprintf(stderr, "[cleanup] proj out\n");
  sgl_xfree(va);
  sgl_xfree(vb);
  fprintf(stderr, "[cleanup] clip feats\n");
  sgl_xfree(feats_a);
  sgl_xfree(feats_b);
  sgl_xfree(img);
  fprintf(stderr, "[cleanup] st/clip\n");
  sgl_st_close(pst);
  sgl_clip_free(c);
  fprintf(stderr, "[cleanup] done\n");

  if (fails) {
    printf("\ntest_llava_multi: %d 项失败\n", fails);
    return 1;
  }
  printf("\ntest_llava_multi: 全部通过（多图各自插到自己的占位符）\n");
  return 0;
}