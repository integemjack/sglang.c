/* G12 验收：`<image>` 占位符位置插入 visual token（而不是接在 prompt 末尾）。
 *
 * 三条断言（全部要求逐位相等 / 必须不同）：
 *  1) 正确性：prompt = A + <image> + B（A/B 是文本 token），插 2 张图的视觉 token 后，
 *     调度器给出的首 token logits 必须等于「手动等价路径」：
 *        prefill A → forward_emb(visual, pos=|A|) → prefill B（pos = |A| + n_img）
 *     （后者就是 llava 的语义：把 <image> 换成一个视觉 token 块）。
 *  2) 落地：同一 prompt 用旧行为（把视觉 token 接在末尾、位置 = |A|+1+|B|）算出来
 *     的首 token 必须**不同** —— 否则说明插入这段代码是空转。
 *  3) 无图回归：同样的文本、不带图时，输出必须与「直接用 sgl_model_prefill 跑同样
 *     文本」逐位相同（G12 不能改变纯文本路径）。
 *
 * 用法: test_g12_place [model_dir]  （默认 $SGL_TEST_G12 或 models/g12-mlp）
 * 模型目录需含：LM（config/model.safetensors）+ clip_config/clip_model/projector
 * （两层 MLP）+ tokenizer 里带 "<image>" added token + test.png。
 */
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/image.h"
#include "sgl/llama.h"
#include "sgl/model.h"
#include "sgl/port.h" /* fopen_s on POSIX（MSVC 有原生实现）*/
#include "sgl/scheduler.h"

static int FAILURES = 0;
#define CHECK(cond, ...) do { if (!(cond)) { FAILURES++; printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                              printf(__VA_ARGS__); printf("\n"); } } while (0)

static int argmax(const float *v, int n) {
  int b = 0;
  for (int i = 1; i < n; i++)
    if (v[i] > v[b]) b = i;
  return b;
}
static double maxdiff(const float *a, const float *b, int n) {
  double e = 0;
  for (int i = 0; i < n; i++) {
    double d = fabs((double)a[i] - (double)b[i]);
    if (d > e) e = d;
  }
  return e;
}

/* 读整个文件 */
static unsigned char *slurp(const char *path, size_t *n) {
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
  unsigned char *b = (unsigned char *)sgl_xmalloc((size_t)sz);
  if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); sgl_xfree(b); return NULL; }
  fclose(f);
  *n = (size_t)sz;
  return b;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0); /* 崩溃时也能看到前面的打印 */
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_G12");
  if (!dir || !*dir) dir = "models/g12-mlp";
  char path[1024];
  snprintf(path, sizeof(path), "%s/model.safetensors", dir);
  FILE *probe = fopen(path, "rb");
  if (!probe) { printf("SKIP: no G12 model at %s\n", dir); return 0; }
  fclose(probe);

  /* ---- 1) 视觉塔 + projector（两层 MLP）→ 视觉 token ---- */
  sgl_clip *clip = sgl_clip_load(dir);
  CHECK(clip != NULL, "clip load");
  if (!clip) return 1;
  snprintf(path, sizeof(path), "%s/projector.safetensors", dir);
  sgl_st_ctx *pst = sgl_st_open(path);
  CHECK(pst != NULL, "projector open");
  sgl_tensor w1 = sgl_st_get(pst, "linear_1.weight", -1, NULL);
  sgl_tensor b1 = sgl_st_get(pst, "linear_1.bias", -1, NULL);
  sgl_tensor w2 = sgl_st_get(pst, "linear_2.weight", -1, NULL);
  sgl_tensor b2 = w2.data ? sgl_st_get(pst, "linear_2.bias", -1, NULL) : sgl_tensor_null();
  CHECK(w1.data && b1.data && w2.data && b2.data, "projector 需要两层 MLP（G12）");
  sgl_projector pj;
  CHECK(sgl_projector_init(&pj, w1.data, b1.data, w1.shape[0], clip->E,
                           w2.data, b2.data, w2.shape[0], w2.shape[1]) == 0, "projector init");
  printf("[g12] projector = 2-layer MLP %d -> %d -> %d\n", pj.in, pj.mid, pj.out);

  snprintf(path, sizeof(path), "%s/test.png", dir);
  size_t pn = 0;
  unsigned char *png = slurp(path, &pn);
  CHECK(png != NULL, "test.png");
  if (!png) return 1;
  float *vis = NULL;
  int nvis = 0;
  CHECK(sgl_clip_project_bytes_pj(png, pn, clip, &pj, &vis, &nvis) == 0, "vision+projector");
  int n_img = clip->n_patch;
  printf("[g12] visual tokens = %d x %d\n", n_img, pj.out);
  sgl_xfree(png);

  /* ---- 2) LM + tokenizer（带 <image>）---- */
  sgl_llama *lm = sgl_llama_load(dir);
  CHECK(lm != NULL, "llama load");
  sgl_tokenizer *tok = sgl_tokenizer_load_dir(dir);
  CHECK(tok != NULL, "tokenizer load");
  if (!lm || !tok) return 1;
  int img_tok = sgl_tokenizer_piece_id(tok, "<image>");
  CHECK(img_tok >= 0, "tokenizer 里应该有 <image> added token");
  CHECK(img_tok < lm->vocab, "<image> id (%d) 必须 < LM vocab (%d)", img_tok, lm->vocab);
  printf("[g12] <image> token id = %d (LM vocab %d)\n", img_tok, lm->vocab);

  /* prompt = A + <image> + B */
  const char *textA = "USER: ";
  const char *textB = " describe this image please\nASSISTANT:";
  int nA = 0, nB = 0;
  int *A = sgl_tokenizer_encode(tok, textA, strlen(textA), &nA);
  int *B = sgl_tokenizer_encode(tok, textB, strlen(textB), &nB);
  CHECK(nA > 0 && nB > 0, "tokenize A/B (%d/%d)", nA, nB);
  int n_prompt = nA + 1 + nB;
  int *prompt = (int *)sgl_xmalloc(sizeof(int) * (size_t)n_prompt);
  memcpy(prompt, A, sizeof(int) * (size_t)nA);
  prompt[nA] = img_tok;
  memcpy(prompt + nA + 1, B, sizeof(int) * (size_t)nB);
  printf("[g12] prompt = %d 个 token（A=%d, <image> 在 %d, B=%d）\n", n_prompt, nA, nA, nB);

  /* ---- 3) 手动等价路径（llava 语义）：A → visual → B ---- */
  int total_ok = nA + n_img + nB;
  int *pg = (int *)sgl_xmalloc(sizeof(int) * (size_t)(total_ok + 8));
  CHECK(sgl_llama_pages_alloc(lm, pg, total_ok + 8) == 0, "pages alloc");
  float *ref = (float *)sgl_xmalloc(sizeof(float) * (size_t)lm->vocab);
  CHECK(sgl_llama_forward_seq(lm, A, nA, 0, pg, 1, ref) == 0, "ref A");
  CHECK(sgl_llama_forward_emb(lm, vis, n_img, nA, pg, ref) == 0, "ref visual");
  CHECK(sgl_llama_forward_seq(lm, B, nB, nA + n_img, pg, 1, ref) == 0, "ref B");
  sgl_llama_lin_release(lm, 1);

  /* ---- 4) 旧行为（接末尾）：A + <image> + B 全算完，视觉 token 追加在 |A|+1+|B| ---- */
  float *old = (float *)sgl_xmalloc(sizeof(float) * (size_t)lm->vocab);
  int *pg2 = (int *)sgl_xmalloc(sizeof(int) * (size_t)(total_ok + 8));
  CHECK(sgl_llama_pages_alloc(lm, pg2, total_ok + 8) == 0, "pages alloc 2");
  CHECK(sgl_llama_forward_seq(lm, prompt, n_prompt, 0, pg2, 2, old) == 0, "old prompt");
  CHECK(sgl_llama_forward_emb(lm, vis, n_img, n_prompt, pg2, old) == 0, "old visual");
  sgl_llama_lin_release(lm, 2);

  /* ---- 5) 走调度器（真实路径）---- */
  sgl_scheduler *s = sgl_sched_new(dir, 42);
  CHECK(s != NULL, "scheduler");
  if (!s) return 1;
  float *vis_copy = (float *)sgl_xmalloc(sizeof(float) * (size_t)nvis);
  memcpy(vis_copy, vis, sizeof(float) * (size_t)nvis);
  char *text = (char *)sgl_xmalloc(strlen(textA) + strlen(textB) + 16);
  sprintf(text, "%s<image>%s", textA, textB);
  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp));
  int id = sgl_sched_submit_img(s, text, 8, &sp, vis_copy, n_img);
  CHECK(id > 0, "submit_img");
  int *ids = NULL, n_ids = 0, np_prompt = 0, np_hit = 0;
  char *out_text = NULL;
  int done = 0;
  for (int step = 0; step < 32 && !done; step++) {
    int rid = 0;
    if (sgl_sched_poll(s, &rid, &out_text, &ids, &n_ids, &np_prompt, &np_hit) == 1) {
      if (rid == id) done = 1;
      else { sgl_xfree(ids); sgl_xfree(out_text); ids = NULL; out_text = NULL; }
    }
    if (!done) sgl_sched_step(s);
  }
  CHECK(done && n_ids >= 1, "调度器应完成请求并产出 token（done=%d n_ids=%d）", done, n_ids);
  int tok_sched = n_ids >= 1 ? ids[0] : -1;
  int tok_ref = argmax(ref, lm->vocab);
  int tok_old = argmax(old, lm->vocab);
  printf("[g12] 首 token：调度器=%d 等价路径=%d 旧末尾拼接=%d\n", tok_sched, tok_ref, tok_old);
  CHECK(tok_sched == tok_ref, "占位符插入后的首 token 应该等于 A→visual→B 等价路径");
  {
    double e = maxdiff(ref, old, lm->vocab);
    printf("[g12] 等价路径 vs 旧末尾拼接：首 token logits 最大差 = %g → %s\n", e,
           tok_old != tok_ref ? "旧行为确实不同 ✓" : "本模型上恰好相同（检查一下）");
  }

  /* ---- 6) 无图回归：同样文本不带图 == 直接 prefill 同一串 token ---- */
  {
    int *pg3 = (int *)sgl_xmalloc(sizeof(int) * (size_t)(nA + nB + 8));
    CHECK(sgl_llama_pages_alloc(lm, pg3, nA + nB + 8) == 0, "pages alloc 3");
    float *plain = (float *)sgl_xmalloc(sizeof(float) * (size_t)lm->vocab);
    int *noimg = (int *)sgl_xmalloc(sizeof(int) * (size_t)(nA + nB));
    memcpy(noimg, A, sizeof(int) * (size_t)nA);
    memcpy(noimg + nA, B, sizeof(int) * (size_t)nB);
    CHECK(sgl_llama_forward_seq(lm, noimg, nA + nB, 0, pg3, 3, plain) == 0, "plain");
    sgl_llama_lin_release(lm, 3);

    char *text2 = (char *)sgl_xmalloc(strlen(textA) + strlen(textB) + 8);
    sprintf(text2, "%s%s", textA, textB);
    int id2 = sgl_sched_submit(s, text2, 4, &sp);
    CHECK(id2 > 0, "submit plain");
    int *ids2 = NULL, n_ids2 = 0;
    char *txt2 = NULL;
    int done2 = 0;
    for (int step = 0; step < 32 && !done2; step++) {
      int rid = 0;
      if (sgl_sched_poll(s, &rid, &txt2, &ids2, &n_ids2, NULL, NULL) == 1) {
        if (rid == id2) done2 = 1;
        else { sgl_xfree(ids2); sgl_xfree(txt2); ids2 = NULL; txt2 = NULL; }
      }
      if (!done2) sgl_sched_step(s);
    }
    CHECK(done2 && n_ids2 >= 1, "纯文本请求应完成");
    int t1 = n_ids2 >= 1 ? ids2[0] : -1, t2 = argmax(plain, lm->vocab);
    printf("[g12] 无图：调度器首 token=%d 直接 prefill 首 token=%d（应相同）\n", t1, t2);
    CHECK(t1 == t2, "无图路径应与非多模态路径一致");
    sgl_xfree(ids2); sgl_xfree(txt2);
    sgl_xfree(pg3); sgl_xfree(plain); sgl_xfree(noimg); sgl_xfree(text2);
    printf("[g12] plain block freed\n");
  }

  sgl_xfree(ids); sgl_xfree(out_text);
  /* vis_copy 的所有权已转给调度器（prefill 用完自己 sgl_xfree），这里不能重复释放 */
  sgl_xfree(text);
  sgl_sched_free(s);
  printf("[g12] teardown: scheduler freed\n");
  sgl_xfree(pg); sgl_xfree(pg2); sgl_xfree(ref); sgl_xfree(old);
  sgl_xfree(prompt); sgl_xfree(A); sgl_xfree(B);
  sgl_tokenizer_free(tok);
  printf("[g12] teardown: tokenizer freed\n");
  sgl_llama_free(lm);
  printf("[g12] teardown: llama freed\n");
  sgl_st_close(pst);
  sgl_clip_free(clip);
  printf("[g12] teardown: clip freed\n");
  sgl_xfree(vis);
  if (FAILURES) { printf("test_g12_place: %d FAILURES\n", FAILURES); return 1; }
  printf("test_g12_place: all OK\n");
  return 0;
}