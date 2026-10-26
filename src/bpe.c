/* sgl bpe 实现：GPT-2 字节级 BPE（HF tokenizer.json 格式）*/
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/bpe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/log.h"

/* ---------- 简单字符串 hash 表（open addressing） ---------- */

typedef struct {
  char **keys;
  int *vals;
  int cap; /* 2 的幂 */
  int count;
} strmap;

/* 与 sgl_xfree 配套的字符串复制（_strdup 是 CRT malloc，不能用 _aligned_free 释放）*/
static char *xstrdup(const char *s) {
  size_t n = strlen(s) + 1;
  char *p = (char *)sgl_xmalloc(n);
  memcpy(p, s, n);
  return p;
}

static uint64_t fnv1a(const char *s, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= (unsigned char)s[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static void sm_init(strmap *m, int expected) {
  int cap = 16;
  while (cap < expected * 2) cap <<= 1;
  m->keys = (char **)sgl_xcalloc(cap, sizeof(char *));
  m->vals = (int *)sgl_xcalloc(cap, sizeof(int));
  m->cap = cap;
  m->count = 0;
}

static void sm_put(strmap *m, const char *key, int val) {
  uint64_t h = fnv1a(key, strlen(key));
  int i = (int)(h & (uint64_t)(m->cap - 1));
  while (m->keys[i]) {
    if (strcmp(m->keys[i], key) == 0) { m->vals[i] = val; return; }
    i = (i + 1) & (m->cap - 1);
  }
  m->keys[i] = xstrdup(key);
  m->vals[i] = val;
  m->count++;
}

static int sm_get(const strmap *m, const char *key) {
  if (!m->keys) return -1;
  uint64_t h = fnv1a(key, strlen(key));
  int i = (int)(h & (uint64_t)(m->cap - 1));
  while (m->keys[i]) {
    if (strcmp(m->keys[i], key) == 0) return m->vals[i];
    i = (i + 1) & (m->cap - 1);
  }
  return -1;
}

static void sm_free(strmap *m) {
  for (int i = 0; i < m->cap; i++) sgl_xfree(m->keys[i]);
  sgl_xfree(m->keys);
  sgl_xfree(m->vals);
  memset(m, 0, sizeof(*m));
}

/* ---------- tokenizer 状态 ---------- */

struct sgl_tokenizer {
  strmap vocab;         /* token 字符串（unicode 空间）→ id */
  char **id2tok;        /* id → token 字符串（指向 strdup 副本或 NULL）*/
  int vocab_size;
  strmap merges;        /* "left right" → rank */
  int byte_to_cp[256];  /* 原始字节 → unicode codepoint */
  int cp_to_byte[512];  /* codepoint → 原始字节；-1 无效（映射最大 0x143）*/
  char **added_tokens;  /* 特殊 token 原文（如 <|endoftext|>）*/
  int *added_ids;
  int n_added;
  int eos_id;
  /* Llama/SentencePiece 路径（style=1）*/
  int style;            /* 0 = GPT-2 byte-level，1 = Llama metaspace */
  int bos_id;
  int byte_tok_ids[256]; /* "<0xXX>" token id；-1 不存在 */
};

/* GPT-2 的 bytes_to_unicode() */
static void build_byte_tables(sgl_tokenizer *t) {
  int bs[256], cs[256], n_bs = 0;
  for (int b = 0x21; b <= 0x7E; b++) bs[n_bs++] = b;
  for (int b = 0xA1; b <= 0xAC; b++) bs[n_bs++] = b;
  for (int b = 0xAE; b <= 0xFF; b++) bs[n_bs++] = b;
  memcpy(cs, bs, sizeof(bs));
  int n = 0;
  for (int b = 0; b < 256; b++) {
    int found = 0;
    for (int i = 0; i < n_bs; i++)
      if (bs[i] == b) { found = 1; break; }
    if (!found) { bs[n_bs] = b; cs[n_bs] = 256 + n; n_bs++; n++; }
  }
  memset(t->cp_to_byte, -1, sizeof(t->cp_to_byte));
  for (int i = 0; i < n_bs; i++) {
    t->byte_to_cp[bs[i]] = cs[i];
    if (cs[i] < 512) t->cp_to_byte[cs[i]] = bs[i];
  }
}

static char *read_file_all(const char *path, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  if (!f) { SGL_ERROR("open %s failed", path); return NULL; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    fclose(f); sgl_xfree(buf); return NULL;
  }
  fclose(f);
  buf[sz] = 0;
  *out_len = (size_t)sz;
  return buf;
}

/* UTF-8 codepoint 迭代：取下一个 codepoint，宽度写 *width */
static unsigned utf8_next(const unsigned char *p, size_t len, int *width) {
  if (len == 0 || p[0] < 0x80) { *width = 1; return len ? p[0] : 0; }
  if ((p[0] & 0xE0) == 0xC0 && len >= 2) { *width = 2; return ((unsigned)p[0] & 0x1F) << 6 | (p[1] & 0x3F); }
  if ((p[0] & 0xF0) == 0xE0 && len >= 3) { *width = 3; return ((unsigned)p[0] & 0x0F) << 12 | ((unsigned)p[1] & 0x3F) << 6 | (p[2] & 0x3F); }
  if ((p[0] & 0xF8) == 0xF0 && len >= 4) { *width = 4; return ((unsigned)p[0] & 0x07) << 18 | ((unsigned)p[1] & 0x3F) << 12 | ((unsigned)p[2] & 0x3F) << 6 | (p[3] & 0x3F); }
  *width = 1;
  return 0xFFFD;
}

static void utf8_encode(unsigned cp, char *out, int *n) {
  if (cp < 0x80) { out[0] = (char)cp; *n = 1; }
  else if (cp < 0x800) {
    out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); *n = 2;
  } else if (cp < 0x10000) {
    out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F)); *n = 3;
  } else {
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F)); *n = 4;
  }
}

sgl_tokenizer *sgl_tokenizer_load(const char *path) {
  size_t len = 0;
  char *text = read_file_all(path, &len);
  if (!text) return NULL;
  sgl_json *root = sgl_json_parse(text, len);
  sgl_xfree(text);
  if (!root) return NULL;

  const sgl_json *model = sgl_json_get(root, "model");
  const sgl_json *vocab = model ? sgl_json_get(model, "vocab") : NULL;
  if (!vocab || vocab->type != SGL_JSON_OBJECT) {
    SGL_ERROR("tokenizer.json: model.vocab 缺失（本项目先支持 GPT-2 系 BPE）");
    sgl_json_free(root);
    return NULL;
  }
  /* 注：HF 官方 gpt2 的 tokenizer.json 里 model 没有 type 字段（隐含 ByteLevel BPE）；
   * 其他导出器可能写 "type":"BPE"，两者都接受。 */
  sgl_tokenizer *t = (sgl_tokenizer *)sgl_xcalloc(1, sizeof(sgl_tokenizer));
  build_byte_tables(t);

  sm_init(&t->vocab, vocab->count);
  t->vocab_size = vocab->count;
  t->id2tok = (char **)sgl_xcalloc(t->vocab_size, sizeof(char *));
  for (int i = 0; i < vocab->count; i++) {
    const char *tok = vocab->keys[i];
    int id = (int)vocab->items[i]->num;
    sm_put(&t->vocab, tok, id);
    if (id >= 0 && id < t->vocab_size && !t->id2tok[id]) t->id2tok[id] = xstrdup(tok);
  }

  const sgl_json *merges = sgl_json_get(model, "merges");
  sm_init(&t->merges, merges ? merges->count : 16);
  if (merges)
    for (int i = 0; i < merges->count; i++) {
      const sgl_json *e = merges->items[i];
      if (e->type == SGL_JSON_STRING) {
        /* HF 官方格式："Ġ t"（左 空格 右，正好就是合并 key）*/
        sm_put(&t->merges, e->str, i);
      } else if (e->type == SGL_JSON_ARRAY && e->count == 2 &&
                 e->items[0]->type == SGL_JSON_STRING &&
                 e->items[1]->type == SGL_JSON_STRING) {
        /* 另一种导出格式：["Ġ", "t"] */
        char key[512];
        snprintf(key, sizeof(key), "%s %s", e->items[0]->str, e->items[1]->str);
        sm_put(&t->merges, key, i);
      }
    }

  /* added_tokens（<|endoftext|> 等）*/
  const sgl_json *added = sgl_json_get(root, "added_tokens");
  t->n_added = added ? added->count : 0;
  t->added_tokens = (char **)sgl_xcalloc(t->n_added + 1, sizeof(char *));
  t->added_ids = (int *)sgl_xcalloc(t->n_added + 1, sizeof(int));
  for (int i = 0; i < t->n_added; i++) {
    const sgl_json *e = added->items[i];
    const char *c = sgl_json_get_str(e, "content", "");
    t->added_tokens[i] = xstrdup(c);
    t->added_ids[i] = sgl_json_get_int(e, "id", -1);
    if (strcmp(c, "<|endoftext|>") == 0 && t->added_ids[i] >= 0) t->eos_id = t->added_ids[i];
  }
  if (!t->eos_id) {
    int e = sm_get(&t->vocab, "<|endoftext|>");
    t->eos_id = e >= 0 ? e : t->vocab_size - 1; /* GPT-2 惯例兜底 */
  }

  int n_merges = merges ? merges->count : 0; /* 先存下：root 马上释放 */

  /* 风格判定：Metaspace 预分词（Llama 系）或存在 <s>/<0x00> 特殊 token */
  const sgl_json *pre = sgl_json_get(root, "pre_tokenizer");
  const char *pretype = pre ? sgl_json_get_str(pre, "type", "") : "";
  if (strcmp(pretype, "Metaspace") == 0) t->style = 1;
  if (sm_get(&t->vocab, "<s>") >= 0 && sm_get(&t->vocab, "<0x41>") >= 0) t->style = 1;
  t->bos_id = sm_get(&t->vocab, "<s>");
  t->eos_id = sm_get(&t->vocab, "</s>");
  if (t->eos_id < 0) t->eos_id = sm_get(&t->vocab, "<|endoftext|>");
  for (int b = 0; b < 256; b++) {
    char kname[8];
    snprintf(kname, sizeof(kname), "<0x%02X>", b);
    t->byte_tok_ids[b] = sm_get(&t->vocab, kname);
  }

  sgl_json_free(root);
  SGL_INFO("tokenizer loaded: style=%s vocab=%d merges=%d added=%d bos=%d eos=%d",
           t->style ? "llama/SP" : "gpt2/ByteLevel",
           t->vocab_size, n_merges, t->n_added, t->bos_id, t->eos_id);
  return t;
}

/* 按目录加载：tokenizer.json 为主；若同目录 tokenizer_config.json 给了
 * eos_token_id/bos_token_id（如 Qwen 的 <|im_end|>），用它覆盖。 */
sgl_tokenizer *sgl_tokenizer_load_dir(const char *dir) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
  sgl_tokenizer *t = sgl_tokenizer_load(path);
  if (!t) return NULL;
  snprintf(path, sizeof(path), "%s/tokenizer_config.json", dir);
  FILE *f = fopen(path, "rb");
  if (!f) return t;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); sgl_xfree(buf); return t; }
  fclose(f);
  buf[sz] = 0;
  sgl_json *j = sgl_json_parse(buf, (size_t)sz);
  sgl_xfree(buf);
  if (!j) return t;
  int eos = sgl_json_get_int(j, "eos_token_id", -1);
  sgl_json_free(j);
  if (eos >= 0 && eos != t->eos_id) {
    SGL_INFO("tokenizer_config: eos %d -> %d", t->eos_id, eos);
    t->eos_id = eos;
  }
  /* 注意：不覆盖 bos。Qwen 的 bos_token_id=<|endoftext|>（占位，HF 编码并不加 BOS），
   * 若采纳会在句首多一个 token；Llama 系 bos 已由 vocab 的 <s> 判定。 */
  return t;
}

void sgl_tokenizer_free(sgl_tokenizer *t) {
  if (!t) return;
  sm_free(&t->vocab);
  sm_free(&t->merges);
  for (int i = 0; i < t->n_added; i++) sgl_xfree(t->added_tokens[i]);
  sgl_xfree(t->added_tokens);
  sgl_xfree(t->added_ids);
  if (t->id2tok)
    for (int i = 0; i < t->vocab_size; i++) sgl_xfree(t->id2tok[i]);
  sgl_xfree(t->id2tok);
  sgl_xfree(t);
}

int sgl_tokenizer_vocab_size(const sgl_tokenizer *t) { return t->vocab_size; }
int sgl_tokenizer_eos_id(const sgl_tokenizer *t) { return t->eos_id; }

/* G12: 字面量 token 查 id（如 llava 的 "<image>"）*/
int sgl_tokenizer_piece_id(const sgl_tokenizer *t, const char *piece) {
  if (!t || !piece) return -1;
  for (int i = 0; i < t->n_added; i++)
    if (t->added_tokens[i] && strcmp(t->added_tokens[i], piece) == 0) return t->added_ids[i];
  return -1;
}
int sgl_tokenizer_bos_id(const sgl_tokenizer *t) { return t->bos_id; }
int sgl_tokenizer_style(const sgl_tokenizer *t) { return t->style; }

/* ---------- pre-tokenize：GPT-2 正则的手写扫描器 ----------
 * 's|'t|'re|'ve|'m|'ll|'d| ?字母+| ?数字+| ?符号+|\s+(?!\S)|\s+
 * 严格按 alternation 顺序在段首位置匹配。
 * 已知简化：>0x7F 的 UTF-8 序列归「符号」类（\p{L} 语义有差异，
 * 英文输入与 HF 完全一致；中文等多字节文本可能得到不同但合法的切分）。 */

typedef struct {
  const unsigned char *p;
  size_t len;
  int pos;
} scanner;

static int is_ascii_letter(unsigned c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static int is_digit(unsigned c) { return c >= '0' && c <= '9'; }
static int is_space_b(unsigned c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

static int match_abbr(const unsigned char *p, int i, int len) {
  static const char *ab[] = {"s", "t", "re", "ve", "m", "ll", "d", NULL};
  for (int a = 0; ab[a]; a++) {
    int l = (int)strlen(ab[a]);
    if (i + 1 + l <= len && memcmp(p + i + 1, ab[a], (size_t)l) == 0) return 1 + l;
  }
  return 0;
}

/* 返回段 [start,end)，无更多段时返回 0 */
static int scan_next(scanner *s, int *start, int *end) {
  if (s->pos >= (int)s->len) return 0;
  *start = s->pos;
  const unsigned char *p = s->p;
  int i = s->pos;
  int len = (int)s->len;

  /* 分支 1：缩写词 's|'t|'re|'ve|'m|'ll|'d */
  if (p[i] == '\'' && i + 1 < len) {
    int m = match_abbr(p, i, len);
    if (m) {
      s->pos = i + m;
      *end = s->pos;
      return 1;
    }
  }

  /* 分支 2-4：可选单个前导空格 + 字母 / 数字 / 符号 */
  int j = i + (p[i] == ' ' ? 1 : 0);
  if (j < len && is_ascii_letter(p[j])) {
    while (j < len && is_ascii_letter(p[j])) j++;
    s->pos = j; *end = j; return 1;
  }
  if (j < len && is_digit(p[j])) {
    while (j < len && is_digit(p[j])) j++;
    s->pos = j; *end = j; return 1;
  }
  if (j < len && !is_space_b(p[j])) {
    /* 符号类：非空白、非 ASCII 字母数字（' 也属于此类，与正则一致）；
     * >0x7F 的 UTF-8 序列按码点整体并入 */
    while (j < len && !is_space_b(p[j]) &&
           (p[j] > 0x7F || (!is_ascii_letter(p[j]) && !is_digit(p[j])))) {
      if (p[j] > 0x7F) { int w; utf8_next(p + j, (size_t)(len - j), &w); j += w; }
      else j++;
    }
    s->pos = j; *end = j; return 1;
  }

  /* 分支 5/6：纯空白段；后跟非空白时留一个空格给下一段作前导空格（\s+(?!\S) 语义） */
  j = i;
  while (j < len && is_space_b(p[j])) j++;
  if (j < len && j - i > 1) j--;
  if (j == i) j++; /* 防死循环 */
  s->pos = j; *end = j;
  return 1;
}

/* ---------- BPE 合并：word 序列逐对合并，中间串走 arena ---------- */

typedef struct {
  char *buf;
  size_t len, cap;
} arena;

static char *arena_add(arena *a, const char *s, size_t n) {
  if (a->len + n + 1 > a->cap) {
    a->cap = (a->cap * 2) + n + 1 + (1 << 16);
    a->buf = (char *)sgl_xrealloc(a->buf, a->cap);
  }
  char *out = a->buf + a->len;
  memcpy(out, s, n);
  out[n] = 0;
  a->len += n + 1;
  return out;
}

static int bpe_word(const sgl_tokenizer *t, char **word, int nword,
                    int *out_ids, int max_out, arena *a) {
  int n = nword;
  while (n > 1) {
    /* 找 rank 最小的相邻对 */
    int best_rank = -1, best_i = -1;
    for (int i = 0; i + 1 < n; i++) {
      size_t la = strlen(word[i]), lb = strlen(word[i + 1]);
      char key[600];
      if (la + lb + 1 >= sizeof(key)) continue;
      memcpy(key, word[i], la);
      key[la] = ' ';
      memcpy(key + la + 1, word[i + 1], lb);
      key[la + lb + 1] = 0;
      int r = sm_get(&t->merges, key);
      if (r >= 0 && (best_rank < 0 || r < best_rank)) { best_rank = r; best_i = i; }
    }
    if (best_rank < 0) break;
    /* 合并所有出现的 best 对（GPT-2 官方实现语义）*/
    const char *L = word[best_i];
    const char *R = word[best_i + 1];
    size_t ll = strlen(L), lr = strlen(R);
    char cat[600];
    memcpy(cat, L, ll);
    memcpy(cat + ll, R, lr);
    char *merged = arena_add(a, cat, ll + lr);
    int w = 0;
    for (int i = 0; i < n; i++) {
      if (i + 1 < n && word[i] == L && word[i + 1] == R) {
        word[w++] = merged;
        i++; /* 跳过被合并的右元 */
      } else {
        word[w++] = word[i];
      }
    }
    n = w;
  }
  int nout = 0;
  for (int i = 0; i < n; i++) {
    int id = sm_get(&t->vocab, word[i]);
    if (id < 0) {
      if (t->style == 1) {
        /* SP 字节回退：该符号串的每个字节查 "<0xXX>" */
        const unsigned char *bs = (const unsigned char *)word[i];
        for (size_t bi = 0; bs[bi]; bi++) {
          int bid = t->byte_tok_ids[bs[bi]];
          if (bid >= 0 && nout < max_out) out_ids[nout++] = bid;
        }
        continue;
      }
      id = 0; /* unk 兜底：byte-level 理论上不发生 */
    }
    if (nout < max_out) out_ids[nout++] = id;
  }
  return nout;
}

/* Llama/SP 路径：一个 piece（如 "▁Hello"）→ token ids */
static int sp_encode_piece(const sgl_tokenizer *t, const char *piece, size_t plen,
                           int *out, int cap) {
  if (plen == 0) return 0;
  arena a = {0};
  char **word = (char **)sgl_xmalloc(sizeof(char *) * (plen + 1));
  int nword = 0;
  size_t i = 0;
  while (i < plen) {
    int w;
    utf8_next((const unsigned char *)piece + i, plen - i, &w);
    if (w <= 0) w = 1;
    word[nword++] = arena_add(&a, piece + i, (size_t)w);
    i += (size_t)w;
  }
  int n = bpe_word(t, word, nword, out, cap, &a);
  sgl_xfree(word);
  sgl_xfree(a.buf);
  return n;
}

/* Llama/SP 主编码：空格→"▁"，按空白分段，首段与空格后段加 "▁" 前缀 */
static int *sp_encode(const sgl_tokenizer *t, const char *text, size_t len,
                      int *out_n) {
  int max_tokens = (int)len * 2 + 16;
  int *ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)max_tokens);
  int nids = 0;
  /* HF 默认 add_bos（Llama 系）：句首加 <s>；与 tokenizers 参考一致 */
  if (t->bos_id >= 0) ids[nids++] = t->bos_id;
  size_t i = 0;
  int first = 1;
  while (i < len) {
    /* added token 字面量优先 */
    int matched = 0;
    for (int a = 0; a < t->n_added; a++) {
      size_t l = strlen(t->added_tokens[a]);
      if (l && i + l <= len && memcmp(text + i, t->added_tokens[a], l) == 0) {
        ids[nids++] = t->added_ids[a];
        i += l;
        matched = 1;
        break;
      }
    }
    if (matched) { first = 0; continue; }

    size_t ws = i;
    while (ws < len && (text[ws] == ' ' || text[ws] == '\t' || text[ws] == '\n' || text[ws] == '\r')) ws++;
    if (ws >= len) break; /* 尾部纯空白不产生 token */
    int add_us = first || ws > i;
    first = 0;
    i = ws;
    char *pbuf = (char *)sgl_xmalloc(len - i + 8);
    size_t plen = 0;
    if (add_us) { /* "▁" U+2581 */
      pbuf[plen++] = (char)0xE2; pbuf[plen++] = (char)0x96; pbuf[plen++] = (char)0x81;
    }
    while (i < len && text[i] != ' ' && text[i] != '\t' && text[i] != '\n' && text[i] != '\r')
      pbuf[plen++] = text[i++];
    pbuf[plen] = 0;
    nids += sp_encode_piece(t, pbuf, plen, ids + nids, max_tokens - nids);
    sgl_xfree(pbuf);
  }
  *out_n = nids;
  return ids;
}

/* G12: 纯 BPE 编码（不处理 added token 字面量）。原 sgl_tokenizer_encode 的主体，
 * 提出来是为了先按 added token 切文本再逐段编码 —— HF 是在 pre-tokenize 之前就
 * 切出特殊 token 的，所以在段中间出现的 "<image>" 也必须是一个 token。
 * （以前的实现只在「段首」匹配 added token，于是 "USER: <image> ..." 里的
 *  "<image>" 被 "符号类" 段吃进去，变成 2 个普通 token。）*/
static void gpt2_encode_plain(const sgl_tokenizer *t, const char *text, size_t len,
                              int *ids, int max_tokens, int *nids_io) {
  int nids = *nids_io;
  size_t i = 0;
  while (i < len) {
    scanner sc = {(const unsigned char *)text, len, (int)i};
    int start, end;
    if (!scan_next(&sc, &start, &end)) break;
    i = (size_t)sc.pos;

    /* 段内：bytes → unicode 空间的单字符串序列，再 BPE 合并 */
    arena a = {0};
    int seg_len = end - start;
    char **word = (char **)sgl_xmalloc(sizeof(char *) * (size_t)(seg_len + 1));
    int nword = 0;
    for (int b = start; b < end; b++) {
      int cp = t->byte_to_cp[(unsigned char)text[b]];
      char utf[8];
      int w;
      utf8_encode((unsigned)cp, utf, &w);
      word[nword++] = arena_add(&a, utf, (size_t)w);
    }
    nids += bpe_word(t, word, nword, ids + nids, max_tokens - nids, &a);
    sgl_xfree(word);
    sgl_xfree(a.buf);
  }
  *nids_io = nids;
}

int *sgl_tokenizer_encode(const sgl_tokenizer *t, const char *text, size_t len,
                          int *out_n) {
  if (t->style == 1) return sp_encode(t, text, len, out_n);
  int max_tokens = (int)len * 2 + 16;
  int *ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)max_tokens);
  int nids = 0;

  size_t i = 0;
  while (i < len) {
    /* G12: 找从 i 开始最早出现的 added token 字面量（任何位置，不限段首）*/
    size_t best = len;
    int best_a = -1;
    for (int a = 0; a < t->n_added; a++) {
      size_t l = strlen(t->added_tokens[a]);
      if (!l) continue;
      for (size_t p = i; p + l <= len; p++) {
        if (memcmp(text + p, t->added_tokens[a], l) == 0) {
          if (p < best) { best = p; best_a = a; }
          break;
        }
      }
    }
    if (best_a < 0) {
      gpt2_encode_plain(t, text + i, len - i, ids, max_tokens, &nids);
      break;
    }
    if (best > i) gpt2_encode_plain(t, text + i, best - i, ids, max_tokens, &nids);
    ids[nids++] = t->added_ids[best_a];
    i = best + strlen(t->added_tokens[best_a]);
  }
  *out_n = nids;
  return ids;
}

char *sgl_tokenizer_decode(const sgl_tokenizer *t, const int *ids, int n) {
  if (t->style == 1) {
    /* Llama：token 字符串原样输出，"▁"→空格，"<0xXX>"→字节 */
    size_t cap = 4096, len = 0;
    char *out = (char *)sgl_xmalloc(cap);
    for (int k = 0; k < n; k++) {
      const char *tok = (ids[k] >= 0 && ids[k] < t->vocab_size) ? t->id2tok[ids[k]] : NULL;
      if (!tok) continue;
      if (tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && strlen(tok) == 6) {
        int byte = (int)strtol(tok + 3, NULL, 16);
        if (len + 2 > cap) { cap *= 2; out = (char *)sgl_xrealloc(out, cap); }
        out[len++] = (char)byte;
        continue;
      }
      const unsigned char *p = (const unsigned char *)tok;
      while (*p) {
        if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) { /* ▁ */
          if (len + 2 > cap) { cap *= 2; out = (char *)sgl_xrealloc(out, cap); }
          out[len++] = ' ';
          p += 3;
        } else {
          if (len + 2 > cap) { cap *= 2; out = (char *)sgl_xrealloc(out, cap); }
          out[len++] = (char)*p++;
        }
      }
    }
    out[len] = 0;
    return out;
  }
  size_t cap = 4096, len = 0;
  char *out = (char *)sgl_xmalloc(cap);
  for (int k = 0; k < n; k++) {
    const char *tok = (ids[k] >= 0 && ids[k] < t->vocab_size) ? t->id2tok[ids[k]] : NULL;
    if (!tok) continue;
    const unsigned char *p = (const unsigned char *)tok;
    while (*p) {
      int w;
      unsigned cp = utf8_next(p, 4, &w);
      int byte = (cp < 512) ? t->cp_to_byte[cp] : -1;
      if (byte < 0) byte = '?';
      if (len + 2 > cap) {
        cap *= 2;
        out = (char *)sgl_xrealloc(out, cap);
      }
      out[len++] = (char)byte;
      p += w;
    }
  }
  out[len] = 0;
  return out;
}
