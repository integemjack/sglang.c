/* sgl model 实现：gpt2 / llama 后端分发 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/model.h"

#include <stdio.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/deepseek.h"
#include "sgl/gpt2.h"
#include "sgl/json.h"
#include "sgl/llama.h"
#include "sgl/log.h"

struct sgl_model {
  int kind; /* 0=gpt2, 1=llama, 2=deepseek(V2/V3) */
  void *impl;
};

/* 读 config.json 判型：DeepSeek（MLA+MoE）→ 2，llama 用 hidden_size → 1，gpt2 用 n_embd → 0 */
static int detect_kind(const char *dir) {
  if (sgl_deepseek_detect(dir) == 1) return 2;
  char path[1024];
  snprintf(path, sizeof(path), "%s/config.json", dir);
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); sgl_xfree(buf); return -1; }
  fclose(f);
  buf[sz] = 0;
  sgl_json *j = sgl_json_parse(buf, (size_t)sz);
  sgl_xfree(buf);
  if (!j) return -1;
  int kind = -1;
  if (sgl_json_get(j, "hidden_size")) kind = 1;
  else if (sgl_json_get(j, "n_embd")) kind = 0;
  sgl_json_free(j);
  return kind;
}

sgl_model *sgl_model_load(const char *model_dir) {
  int kind = detect_kind(model_dir);
  if (kind < 0) {
    SGL_ERROR("model %s: config.json 无法识别（需 hidden_size 或 n_embd）", model_dir);
    return NULL;
  }
  sgl_model *m = (sgl_model *)sgl_xmalloc(sizeof(*m));
  m->kind = kind;
  if (kind == 0) {
    m->impl = sgl_gpt2_load(model_dir);
    if (!m->impl) { sgl_xfree(m); return NULL; }
    SGL_INFO("model backend: gpt2");
  } else if (kind == 2) {
    m->impl = sgl_deepseek_load(model_dir);
    if (!m->impl) { sgl_xfree(m); return NULL; }
    SGL_INFO("model backend: deepseek (MLA+MoE)");
  } else {
    m->impl = sgl_llama_load(model_dir);
    if (!m->impl) { sgl_xfree(m); return NULL; }
    SGL_INFO("model backend: llama");
  }
  return m;
}

void sgl_model_free(sgl_model *m) {
  if (!m) return;
  if (m->kind == 0) sgl_gpt2_free((sgl_gpt2 *)m->impl);
  else if (m->kind == 2) sgl_deepseek_free((sgl_deepseek *)m->impl);
  else sgl_llama_free((sgl_llama *)m->impl);
  sgl_xfree(m);
}

int sgl_model_vocab(const sgl_model *m) {
  if (m->kind == 0) return ((sgl_gpt2 *)m->impl)->vocab;
  if (m->kind == 2) return sgl_deepseek_vocab((sgl_deepseek *)m->impl);
  return ((sgl_llama *)m->impl)->vocab;
}

int sgl_model_ctx(const sgl_model *m) {
  if (m->kind == 0) return ((sgl_gpt2 *)m->impl)->n_ctx;
  if (m->kind == 2) return sgl_deepseek_ctx((sgl_deepseek *)m->impl);
  return ((sgl_llama *)m->impl)->n_ctx;
}

int sgl_model_pages_alloc(sgl_model *m, int *idx, int n) {
  if (m->kind == 0) return sgl_gpt2_pages_alloc((sgl_gpt2 *)m->impl, idx, n);
  if (m->kind == 2) return sgl_deepseek_pages_alloc((sgl_deepseek *)m->impl, idx, n);
  return sgl_llama_pages_alloc((sgl_llama *)m->impl, idx, n);
}

void sgl_model_pages_unref(sgl_model *m, const int *idx, int n) {
  if (m->kind == 0) sgl_gpt2_pages_unref((sgl_gpt2 *)m->impl, idx, n);
  else if (m->kind == 2) sgl_deepseek_pages_unref((sgl_deepseek *)m->impl, idx, n);
  else sgl_llama_pages_unref((sgl_llama *)m->impl, idx, n);
}

void sgl_model_page_ref(sgl_model *m, int page, int delta) {
  if (m->kind == 0) sgl_gpt2_page_ref((sgl_gpt2 *)m->impl, page, delta);
  else if (m->kind == 2) sgl_deepseek_page_ref((sgl_deepseek *)m->impl, page, delta);
  else sgl_llama_page_ref((sgl_llama *)m->impl, page, delta);
}

int sgl_model_nfree(const sgl_model *m) {
  if (m->kind == 0) return sgl_gpt2_nfree((sgl_gpt2 *)m->impl);
  if (m->kind == 2) return sgl_deepseek_nfree((sgl_deepseek *)m->impl);
  return sgl_llama_nfree((sgl_llama *)m->impl);
}

int sgl_model_pool_pages(const sgl_model *m) {
  if (m->kind == 0) return ((sgl_gpt2 *)m->impl)->n_pool;
  if (m->kind == 2) return sgl_deepseek_pool_pages((sgl_deepseek *)m->impl);
  return ((sgl_llama *)m->impl)->n_pool;
}

int sgl_model_page_refs(const sgl_model *m, int page) {
  if (m->kind == 2) return sgl_deepseek_page_refs((sgl_deepseek *)m->impl, page);
  if (m->kind == 0) {
    const sgl_gpt2 *g = (const sgl_gpt2 *)m->impl;
    return (page >= 0 && page < g->n_pool) ? g->page_refs[page] : 0;
  }
  const sgl_llama *l = (const sgl_llama *)m->impl;
  return (page >= 0 && page < l->n_pool) ? l->page_refs[page] : 0;
}

int sgl_model_prefill(sgl_model *m, const int *tokens, int n, int pos0,
                      const int *idx, float *logits_out) {
  if (m->kind == 0) return sgl_gpt2_prefill((sgl_gpt2 *)m->impl, tokens, n, pos0, idx, logits_out);
  if (m->kind == 2)
    return sgl_deepseek_prefill((sgl_deepseek *)m->impl, tokens, n, pos0, idx, logits_out);
  return sgl_llama_forward((sgl_llama *)m->impl, tokens, n, pos0, idx, logits_out);
}

int sgl_model_prefill_emb(sgl_model *m, const float *emb, int n, int pos0,
                          const int *idx, float *logits_out) {
  if (m->kind == 2)
    return sgl_deepseek_prefill_emb((sgl_deepseek *)m->impl, emb, n, pos0, idx, logits_out);
  if (m->kind == 1)
    return sgl_llama_forward_emb((sgl_llama *)m->impl, emb, n, pos0, idx, logits_out);
  SGL_ERROR("model: emb prefill（多模态）仅 llama / deepseek 后端支持（kind=%d）", m->kind);
  return -1;
}

int sgl_model_decode_batch(sgl_model *m, const int *tokens, int *const *idx,
                           const int *seq_lens, int B, float *logits_out) {
  if (m->kind == 0)
    return sgl_gpt2_decode_batch((sgl_gpt2 *)m->impl, tokens, idx, seq_lens, B, logits_out);
  if (m->kind == 2)
    return sgl_deepseek_decode_batch((sgl_deepseek *)m->impl, tokens, idx, seq_lens, B, logits_out);
  return sgl_llama_decode_batch((sgl_llama *)m->impl, tokens, idx, seq_lens, B, logits_out);
}
