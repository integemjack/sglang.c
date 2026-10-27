/* sgl wordpiece.c — BERT 系 WordPiece tokenizer（HF tokenizer.json）
 * 实现 chain: BertNormalizer(clean_text / handle_chinese_chars / lowercase /
 * strip_accents) -> BertPreTokenizer(空白 / 标点 / CJK 切分) -> WordPiece 贪心最长匹配。
 * 对应 sglang 里 HF tokenizers 的 BertTokenizer(Fast)。
 *
 * 已知限制（与 HF 的差异见文件末）：
 *   - lowercase 只做 ASCII A-Z->a-z，非 ASCII 的大小写折叠（İ/Σ 等）不做；
 *   - strip_accents 只丢掉组合符号 codepoint，不做 NFD 分解（é 不会被拆成 e+́）；
 *   - 不做 BPE / SentencePiece（RoBERTa 的 GPT-2 式 BPE 词表走 sgl/bpe.h）。
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/bert.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/log.h"

/* ---------- 词表 hash 表（open addressing，key 为 UTF-8 字符串） ---------- */

typedef struct {
  char **keys;
  int *vals;
  int cap; /* 2 的幂 */
  int count;
} wp_map;

static char *wp_strdup(const char *s) {
  size_t n = strlen(s) + 1;
  char *p = (char *)sgl_xmalloc(n);
  memcpy(p, s, n);
  return p;
}

static uint64_t wp_fnv1a(const char *s, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= (unsigned char)s[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static void wm_init(wp_map *m, int expected) {
  int cap = 16;
  while (cap < expected * 2) cap <<= 1;
  m->keys = (char **)sgl_xcalloc((size_t)cap, sizeof(char *));
  m->vals = (int *)sgl_xcalloc((size_t)cap, sizeof(int));
  m->cap = cap;
  m->count = 0;
}

static void wm_put(wp_map *m, const char *key, int val) {
  uint64_t h = wp_fnv1a(key, strlen(key));
  int i = (int)(h & (uint64_t)(m->cap - 1));
  while (m->keys[i]) {
    if (strcmp(m->keys[i], key) == 0) { m->vals[i] = val; return; }
    i = (i + 1) & (m->cap - 1);
  }
  m->keys[i] = wp_strdup(key);
  m->vals[i] = val;
  m->count++;
}

static int wm_get(const wp_map *m, const char *key) {
  if (!m->keys) return -1;
  uint64_t h = wp_fnv1a(key, strlen(key));
  int i = (int)(h & (uint64_t)(m->cap - 1));
  while (m->keys[i]) {
    if (strcmp(m->keys[i], key) == 0) return m->vals[i];
    i = (i + 1) & (m->cap - 1);
  }
  return -1;
}

static void wm_free(wp_map *m) {
  for (int i = 0; i < m->cap; i++) sgl_xfree(m->keys[i]);
  sgl_xfree(m->keys);
  sgl_xfree(m->vals);
  memset(m, 0, sizeof(*m));
}

/* ---------- tokenizer 状态 ---------- */

struct sgl_wp {
  wp_map vocab;
  int vocab_size;
  int unk_id, cls_id, sep_id, pad_id, mask_id;
  int lowercase, strip_accents, handle_chinese, clean_text;
  int max_chars;            /* model.max_input_chars_per_word */
  int max_len;              /* truncation.max_length */
  char cont_prefix[16];     /* model.continuing_subword_prefix，通常 "##" */
};

/* ---------- UTF-8 ---------- */

static unsigned wp_utf8_next(const unsigned char *p, size_t len, int *width) {
  if (len == 0 || p[0] < 0x80) { *width = 1; return len ? p[0] : 0; }
  if ((p[0] & 0xE0) == 0xC0 && len >= 2) {
    *width = 2; return ((unsigned)p[0] & 0x1F) << 6 | (p[1] & 0x3F);
  }
  if ((p[0] & 0xF0) == 0xE0 && len >= 3) {
    *width = 3;
    return ((unsigned)p[0] & 0x0F) << 12 | ((unsigned)p[1] & 0x3F) << 6 | (p[2] & 0x3F);
  }
  if ((p[0] & 0xF8) == 0xF0 && len >= 4) {
    *width = 4;
    return ((unsigned)p[0] & 0x07) << 18 | ((unsigned)p[1] & 0x3F) << 12 |
           ((unsigned)p[2] & 0x3F) << 6 | (p[3] & 0x3F);
  }
  *width = 1;
  return 0xFFFD;
}

static void wp_utf8_encode(unsigned cp, char *out, int *n) {
  if (cp < 0x80) { out[0] = (char)cp; *n = 1; }
  else if (cp < 0x800) {
    out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); *n = 2;
  } else if (cp < 0x10000) {
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F)); *n = 3;
  } else {
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F)); *n = 4;
  }
}

/* ---------- 字符类别（与 HF BertPreTokenizer 的 _is_* 一致） ---------- */

static int wp_is_space(unsigned cp) {
  if (cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D ||
      cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 || cp == 0x2028 ||
      cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000)
    return 1;
  return cp >= 0x2000 && cp <= 0x200A;
}

static int wp_is_control(unsigned cp) {
  if (cp == 0x09 || cp == 0x0A || cp == 0x0D) return 0;
  return cp <= 0x1F || (cp >= 0x7F && cp <= 0x9F);
}

static int wp_is_cjk(unsigned cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
         (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
         (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F) ||
         (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0x3040 && cp <= 0x30FF) ||
         (cp >= 0x31F0 && cp <= 0x31FF) || (cp >= 0xFF00 && cp <= 0xFFEF) ||
         (cp >= 0xAC00 && cp <= 0xD7AF);
}

static int wp_is_punct(unsigned cp) {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
      (cp >= 123 && cp <= 126))
    return 1;
  if (cp < 0x80) return 0;
  static const unsigned r[][2] = {
      {0x2000, 0x206F}, {0x2E00, 0x2E7F}, {0x3001, 0x303F}, {0xFE10, 0xFE19},
      {0xFE30, 0xFE6F}, {0xFF01, 0xFF0F}, {0xFF1A, 0xFF20}, {0xFF3B, 0xFF40},
      {0xFF5B, 0xFF65},
  };
  for (size_t i = 0; i < sizeof(r) / sizeof(r[0]); i++)
    if (cp >= r[i][0] && cp <= r[i][1]) return 1;
  return 0;
}

static int wp_is_combining(unsigned cp) {
  return (cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x1AB0 && cp <= 0x1AFF) ||
         (cp >= 0x1DC0 && cp <= 0x1DFF) || (cp >= 0x20D0 && cp <= 0x20FF) ||
         (cp >= 0xFE20 && cp <= 0xFE2F);
}

/* ---------- JSON 小工具 ---------- */

static char *wp_read_file(const char *path, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  if (sz < 0) { fclose(f); return NULL; }
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    fclose(f); sgl_xfree(buf); return NULL;
  }
  fclose(f);
  buf[sz] = 0;
  if (out_len) *out_len = (size_t)sz;
  return buf;
}

/* 取 JSON 布尔（缺省/类型不对返回 def）*/
static int wp_bool(const sgl_json *o, const char *key, int def) {
  const sgl_json *v = o ? sgl_json_get(o, key) : NULL;
  if (!v) return def;
  if (v->type == SGL_JSON_BOOL) return v->boolean;
  if (v->type == SGL_JSON_NULL) return def;
  return def;
}

/* normalizer 可能是 {type:BertNormalizer, lowercase:bool, strip_accents:bool|null}
 * / {type:Lowercase} / {type:Sequence, normalizers:[…]}：递归读出两个标志，
 * 未明确指定时用 -1 表示「沿用推断」。 */
static void wp_scan_normalizer(const sgl_json *n, int *lower, int *strip) {
  if (!n || n->type != SGL_JSON_OBJECT) return;
  const char *ty = sgl_json_get_str(n, "type", "");
  if (strcmp(ty, "Lowercase") == 0) *lower = 1;
  if (strcmp(ty, "BertNormalizer") == 0) {
    if (*lower < 0) *lower = wp_bool(n, "lowercase", 1);
    const sgl_json *sa = sgl_json_get(n, "strip_accents");
    if (sa && sa->type == SGL_JSON_BOOL) *strip = sa->boolean;
  }
  const sgl_json *sub = sgl_json_get(n, "normalizers");
  if (sub && sub->type == SGL_JSON_ARRAY)
    for (int i = 0; i < sub->count; i++) wp_scan_normalizer(sub->items[i], lower, strip);
}

/* ---------- 加载 ---------- */

sgl_wp *sgl_wp_load_dir(const char *dir) {
  char path[1200];
  snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
  size_t len = 0;
  char *text = wp_read_file(path, &len);
  if (!text) {
    SGL_ERROR("bert tokenizer: 读不到 %s", path);
    return NULL;
  }
  sgl_json *root = sgl_json_parse(text, len);
  sgl_xfree(text);
  if (!root) { SGL_ERROR("bert tokenizer: %s JSON 解析失败", path); return NULL; }

  const sgl_json *model = sgl_json_get(root, "model");
  const sgl_json *vocab = model ? sgl_json_get(model, "vocab") : NULL;
  if (!vocab || vocab->type != SGL_JSON_OBJECT) {
    SGL_ERROR("bert tokenizer: %s 里 model.vocab 缺失（需要 WordPiece 词表）", path);
    sgl_json_free(root);
    return NULL;
  }
  sgl_wp *t = (sgl_wp *)sgl_xcalloc(1, sizeof(sgl_wp));
  wm_init(&t->vocab, vocab->count);
  t->vocab_size = 0;
  for (int i = 0; i < vocab->count; i++) {
    int id = (int)vocab->items[i]->num;
    wm_put(&t->vocab, vocab->keys[i], id);
    if (id + 1 > t->vocab_size) t->vocab_size = id + 1;
  }
  /* 词表就位后从 unk_token 读 UNK id */

  const char *unk = sgl_json_get_str(model, "unk_token", "[UNK]");
  const char *cont = sgl_json_get_str(model, "continuing_subword_prefix", "##");
  snprintf(t->cont_prefix, sizeof(t->cont_prefix), "%s", cont);
  t->max_chars = sgl_json_get_int(model, "max_input_chars_per_word", 100);
  t->clean_text = 1;
  t->handle_chinese = 1;

  int lower = -1, strip = -1;
  wp_scan_normalizer(sgl_json_get(root, "normalizer"), &lower, &strip);
  if (lower < 0) {
    /* 回退：tokenizer_config.json 的 do_lower_case */
    char cfgp[1200];
    snprintf(cfgp, sizeof(cfgp), "%s/tokenizer_config.json", dir);
    size_t clen = 0;
    char *ctext = wp_read_file(cfgp, &clen);
    if (ctext) {
      sgl_json *cc = sgl_json_parse(ctext, clen);
      sgl_xfree(ctext);
      if (cc) {
        lower = wp_bool(cc, "do_lower_case", 0);
        sgl_json_free(cc);
      }
    }
  }
  if (lower < 0) lower = 0;
  if (strip < 0) strip = lower; /* BertNormalizer 的 strip_accents:null == do_lower_case */
  t->lowercase = lower;
  t->strip_accents = strip;

  const sgl_json *tr = sgl_json_get(root, "truncation");
  t->max_len = tr ? sgl_json_get_int(tr, "max_length", 512) : 512;
  if (t->max_len <= 0) t->max_len = 512;

  /* 特殊 token id：先按 BERT 名，再按 RoBERTa 名 */
  struct { const char *name; int *slot; } sp[] = {
      {"[UNK]", &t->unk_id}, {"[CLS]", &t->cls_id}, {"[SEP]", &t->sep_id},
      {"[PAD]", &t->pad_id}, {"[MASK]", &t->mask_id},
  };
  for (size_t i = 0; i < sizeof(sp) / sizeof(sp[0]); i++) *sp[i].slot = wm_get(&t->vocab, sp[i].name);
  struct { const char *alt; int cur; int *slot; } alt[] = {
      {"<unk>", t->unk_id, &t->unk_id}, {"<s>", t->cls_id, &t->cls_id},
      {"</s>", t->sep_id, &t->sep_id},  {"<pad>", t->pad_id, &t->pad_id},
      {"<mask>", t->mask_id, &t->mask_id},
  };
  for (size_t i = 0; i < sizeof(alt) / sizeof(alt[0]); i++)
    if (*alt[i].slot < 0) *alt[i].slot = wm_get(&t->vocab, alt[i].alt);

  if (t->unk_id < 0) {
    SGL_ERROR("bert tokenizer: 词表里没有 unk token '%s'", unk);
    sgl_wp_free(t);
    sgl_json_free(root);
    return NULL;
  }
  if (t->cls_id < 0) t->cls_id = t->unk_id, SGL_WARN("bert tokenizer: 无 [CLS]，退化为 UNK");
  if (t->sep_id < 0) t->sep_id = t->unk_id, SGL_WARN("bert tokenizer: 无 [SEP]，退化为 UNK");
  if (t->pad_id < 0) t->pad_id = 0;

  SGL_INFO("bert tokenizer: %s vocab=%d lowercase=%d strip_accents=%d max_len=%d cont='%s'",
           path, t->vocab_size, t->lowercase, t->strip_accents, t->max_len,
           t->cont_prefix);
  sgl_json_free(root);
  return t;
}

void sgl_wp_free(sgl_wp *t) {
  if (!t) return;
  wm_free(&t->vocab);
  sgl_xfree(t);
}

int sgl_wp_vocab_size(const sgl_wp *t) { return t ? t->vocab_size : 0; }
int sgl_wp_id(const sgl_wp *t, const char *piece) {
  return (t && piece) ? wm_get(&t->vocab, piece) : -1;
}
int sgl_wp_cls_id(const sgl_wp *t) { return t ? t->cls_id : -1; }
int sgl_wp_sep_id(const sgl_wp *t) { return t ? t->sep_id : -1; }
int sgl_wp_max_len(const sgl_wp *t) { return t ? t->max_len : 0; }

/* ---------- WordPiece 贪心最长匹配 ---------- */

/* 把一个「基本词」拆成 wordpiece id；拼不出合法序列时整词 -> UNK（HF 语义）*/
static int wp_word_ids(const sgl_wp *t, const char *w, int *ids, int n, int cap) {
  int wlen = (int)strlen(w);
  if (wlen == 0) return n;
  int *offs = (int *)sgl_xmalloc(sizeof(int) * (size_t)(wlen + 1));
  int nchars = 0;
  for (int i = 0; i < wlen;) {
    int wd = 1;
    wp_utf8_next((const unsigned char *)w + i, (size_t)(wlen - i), &wd);
    offs[nchars++] = i;
    i += wd;
  }
  offs[nchars] = wlen;
  if (nchars > t->max_chars) {
    if (n < cap) ids[n++] = t->unk_id;
    sgl_xfree(offs);
    return n;
  }
  size_t plen = (size_t)wlen + strlen(t->cont_prefix) + 1;
  char *piece = (char *)sgl_xmalloc(plen);
  int s = 0;
  while (s < nchars && n < cap) {
    int best = -1, bestid = -1;
    for (int e = nchars; e > s; e--) {
      int st = offs[s], en = offs[e];
      if (s == 0) snprintf(piece, plen, "%.*s", en - st, w + st);
      else snprintf(piece, plen, "%s%.*s", t->cont_prefix, en - st, w + st);
      int id = wm_get(&t->vocab, piece);
      if (id >= 0) { best = e; bestid = id; break; }
    }
    if (best < 0) { /* 该词无法 WordPiece 化 */
      ids[n++] = t->unk_id;
      sgl_xfree(piece);
      sgl_xfree(offs);
      return n;
    }
    ids[n++] = bestid;
    s = best;
  }
  sgl_xfree(piece);
  sgl_xfree(offs);
  return n;
}

int sgl_wp_encode_plain(const sgl_wp *t, const char *text, int *ids, int cap) {
  if (!t || !text || !ids || cap <= 0) return 0;
  size_t len = strlen(text);
  char *wb = (char *)sgl_xmalloc(len + 8); /* 归一化后不会比原文更长 */
  int wn = 0, n = 0;
  size_t off = 0;
  const unsigned char *p = (const unsigned char *)text;
  while (off < len && n < cap) {
    int width = 1;
    unsigned cp = wp_utf8_next(p + off, len - off, &width);
    off += (size_t)width;
    if (t->clean_text && (cp == 0 || cp == 0xFFFD || wp_is_control(cp))) continue;
    if (wp_is_space(cp)) {
      if (wn) { wb[wn] = 0; n = wp_word_ids(t, wb, ids, n, cap); wn = 0; }
      continue;
    }
    if (t->lowercase && cp >= 'A' && cp <= 'Z') cp += 32;
    if (t->strip_accents && wp_is_combining(cp)) continue;
    if ((t->handle_chinese && wp_is_cjk(cp)) || wp_is_punct(cp)) {
      if (wn) { wb[wn] = 0; n = wp_word_ids(t, wb, ids, n, cap); wn = 0; }
      if (n < cap) {
        char one[8];
        int w2 = 0;
        wp_utf8_encode(cp, one, &w2);
        one[w2] = 0;
        n = wp_word_ids(t, one, ids, n, cap);
      }
      continue;
    }
    int w2 = 0;
    wp_utf8_encode(cp, wb + wn, &w2);
    wn += w2;
  }
  if (wn && n < cap) { wb[wn] = 0; n = wp_word_ids(t, wb, ids, n, cap); }
  sgl_xfree(wb);
  return n;
}

int sgl_wp_encode_ids(const sgl_wp *t, const char *text, int *ids, int cap) {
  if (!t || !text || !ids || cap < 2) return 0;
  int limit = t->max_len < cap ? t->max_len : cap;
  if (limit < 2) return 0;
  int n = 0;
  ids[n++] = t->cls_id;
  n += sgl_wp_encode_plain(t, text, ids + n, limit - n); /* 留一格给 [SEP] */
  if (n > limit - 1) n = limit - 1;
  ids[n++] = t->sep_id;
  return n;
}

/* ---------- 按目录缓存（sgl_wordpiece_encode 用） ---------- */

static struct {
  char *dir;
  sgl_wp *tok;
} g_wp_cache[4];

int sgl_wordpiece_encode(const char *dir, const char *text, int *ids, int cap) {
  if (!dir || !text || !ids || cap <= 0) return -1;
  sgl_wp *t = NULL;
  int slot = -1, empty = -1;
  for (int i = 0; i < 4; i++) {
    if (g_wp_cache[i].dir && strcmp(g_wp_cache[i].dir, dir) == 0) { t = g_wp_cache[i].tok; break; }
    if (!g_wp_cache[i].dir && empty < 0) empty = i;
  }
  if (!t) {
    t = sgl_wp_load_dir(dir);
    if (!t) return -1;
    slot = empty >= 0 ? empty : 0;
    sgl_xfree(g_wp_cache[slot].dir);
    g_wp_cache[slot].dir = wp_strdup(dir);
    g_wp_cache[slot].tok = t;
  }
  return sgl_wp_encode_ids(t, text, ids, cap);
}