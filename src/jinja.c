/* sgl jinja 实现：Jinja2 子集解释器（chat_template 专用）
 *
 * 三个阶段：词法切分（文本 / {{ }} / {% %}，含 {{- -}} 空白控制）→ AST → 渲染。
 * 值模型 jval 支持 undefined/null/bool/number/string/list/dict/namespace，
 * 全部由 arena 分配，渲染结束一次性释放（可重入：AST 只读，render 各自建 arena）。
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/jinja.h"

#include "sgl/alloc.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/log.h"
#include "sgl/port.h"

/* ============================ arena ============================ */
typedef struct ablk {
  struct ablk *next;
  size_t used, cap;
  char *data;
} ablk;

typedef struct {
  ablk *head;
  int oom;
} arena;

#define A_CHUNK (256 * 1024)

static void *a_alloc(arena *a, size_t n) {
  n = (n + 15) & ~(size_t)15;
  if (!a->head || a->head->used + n > a->head->cap) {
    size_t cap = n > A_CHUNK ? n : A_CHUNK;
    ablk *b = (ablk *)malloc(sizeof(ablk));
    if (!b) {
      a->oom = 1;
      return NULL;
    }
    b->data = (char *)malloc(cap);
    if (!b->data) {
      free(b);
      a->oom = 1;
      return NULL;
    }
    b->used = 0;
    b->cap = cap;
    b->next = a->head;
    a->head = b;
  }
  void *p = a->head->data + a->head->used;
  a->head->used += n;
  memset(p, 0, n);
  return p;
}

static char *a_strndup(arena *a, const char *s, size_t n) {
  char *d = (char *)a_alloc(a, n + 1);
  if (!d) return NULL;
  if (n) memcpy(d, s, n);
  d[n] = 0;
  return d;
}

static char *a_strdup(arena *a, const char *s) { return a_strndup(a, s, strlen(s)); }

static const char *a_printf(arena *a, const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n < 0) return "";
  if ((size_t)n < sizeof(buf)) return a_strndup(a, buf, (size_t)n);
  char *big = (char *)a_alloc(a, (size_t)n + 1);
  if (!big) return "";
  va_start(ap, fmt);
  vsnprintf(big, (size_t)n + 1, fmt, ap);
  va_end(ap);
  return big;
}

static void a_reset(arena *a) {
  ablk *b = a->head;
  while (b) {
    ablk *nx = b->next;
    free(b->data);
    free(b);
    b = nx;
  }
  a->head = NULL;
  a->oom = 0;
}

/* ============================ string builder ============================ */
typedef struct {
  char *p;
  size_t len, cap;
  int oom;
} sb;

static void sb_init(sb *s) { memset(s, 0, sizeof(*s)); }

static void sb_put(sb *s, const char *d, size_t n) {
  if (!n || s->oom) return;
  if (s->len + n + 1 > s->cap) {
    size_t cap = s->cap ? s->cap : 256;
    while (cap < s->len + n + 1) cap *= 2;
    char *np = (char *)realloc(s->p, cap);
    if (!np) {
      s->oom = 1;
      return;
    }
    s->p = np;
    s->cap = cap;
  }
  memcpy(s->p + s->len, d, n);
  s->len += n;
  s->p[s->len] = 0;
}

static void sb_puts(sb *s, const char *str) { sb_put(s, str, strlen(str)); }

static void sb_free(sb *s) {
  free(s->p);
  s->p = NULL;
  s->len = s->cap = 0;
}

/* ============================ 值 ============================ */
enum { V_UNDEF = 0, V_NULL, V_BOOL, V_NUM, V_STR, V_LIST, V_DICT, V_NS };

typedef struct jval jval;
typedef struct {
  const char *k;
  jval *v;
} jkv;

struct jval {
  int t;
  int b;
  double num;
  const char *s;
  jval **it;
  int n, cap;
  jkv *kv;
  int nkv, capkv;
};

static jval *v_new(arena *a, int t) {
  jval *v = (jval *)a_alloc(a, sizeof(jval));
  if (!v) return NULL;
  v->t = t;
  return v;
}

static jval *v_undef(arena *a) { return v_new(a, V_UNDEF); }
static jval *v_null(arena *a) { return v_new(a, V_NULL); }

static jval *v_bool(arena *a, int b) {
  jval *v = v_new(a, V_BOOL);
  if (v) v->b = b ? 1 : 0;
  return v;
}

static jval *v_num(arena *a, double d) {
  jval *v = v_new(a, V_NUM);
  if (v) v->num = d;
  return v;
}

static jval *v_str(arena *a, const char *s) {
  jval *v = v_new(a, V_STR);
  if (v) v->s = s ? s : "";
  return v;
}

static jval *v_list(arena *a) { return v_new(a, V_LIST); }
static jval *v_dict(arena *a, int is_ns) { return v_new(a, is_ns ? V_NS : V_DICT); }

static jval *v_get(const jval *d, const char *k) {
  if (!d || (d->t != V_DICT && d->t != V_NS) || !k) return NULL;
  for (int i = 0; i < d->nkv; i++)
    if (strcmp(d->kv[i].k, k) == 0) return d->kv[i].v;
  return NULL;
}

static void v_set(arena *a, jval *d, const char *k, jval *v) {
  if (!d || (d->t != V_DICT && d->t != V_NS) || !v || !k) return;
  for (int i = 0; i < d->nkv; i++)
    if (strcmp(d->kv[i].k, k) == 0) {
      d->kv[i].v = v;
      return;
    }
  if (d->nkv == d->capkv) {
    int nc = d->capkv ? d->capkv * 2 : 8;
    jkv *nk = (jkv *)a_alloc(a, sizeof(jkv) * (size_t)nc);
    if (!nk) return;
    if (d->kv) memcpy(nk, d->kv, sizeof(jkv) * (size_t)d->nkv);
    d->kv = nk;
    d->capkv = nc;
  }
  d->kv[d->nkv].k = a_strdup(a, k);
  d->kv[d->nkv].v = v;
  d->nkv++;
}

static void v_push(arena *a, jval *l, jval *v) {
  if (!l || l->t != V_LIST || !v) return;
  if (l->n == l->cap) {
    int nc = l->cap ? l->cap * 2 : 8;
    jval **ni = (jval **)a_alloc(a, sizeof(jval *) * (size_t)nc);
    if (!ni) return;
    if (l->it) memcpy(ni, l->it, sizeof(jval *) * (size_t)l->n);
    l->it = ni;
    l->cap = nc;
  }
  l->it[l->n++] = v;
}

static int v_truthy(const jval *v) {
  if (!v) return 0;
  switch (v->t) {
    case V_UNDEF:
    case V_NULL: return 0;
    case V_BOOL: return v->b;
    case V_NUM: return v->num != 0.0;
    case V_STR: return v->s && v->s[0];
    case V_LIST: return v->n > 0;
    case V_DICT:
    case V_NS: return v->nkv > 0;
    default: return 0;
  }
}

static int v_equal(const jval *x, const jval *y) {
  if (!x || !y) return 0;
  if (x->t == V_NUM && y->t == V_NUM) return x->num == y->num;
  if (x->t == V_BOOL && y->t == V_BOOL) return x->b == y->b;
  if (x->t == V_UNDEF || y->t == V_UNDEF) return x->t == y->t;
  if (x->t == V_NULL || y->t == V_NULL) return x->t == y->t;
  if (x->t == V_STR && y->t == V_STR) return strcmp(x->s, y->s) == 0;
  if (x->t == V_LIST && y->t == V_LIST) {
    if (x->n != y->n) return 0;
    for (int i = 0; i < x->n; i++)
      if (!v_equal(x->it[i], y->it[i])) return 0;
    return 1;
  }
  if ((x->t == V_DICT || x->t == V_NS) && (y->t == V_DICT || y->t == V_NS)) {
    if (x->nkv != y->nkv) return 0;
    for (int i = 0; i < x->nkv; i++) {
      jval *b = v_get(y, x->kv[i].k);
      if (!b || !v_equal(x->kv[i].v, b)) return 0;
    }
    return 1;
  }
  return 0;
}

/* Jinja str() 语义：True/False/None/整数不带小数 */
static const char *v_tostr(arena *a, const jval *v) {
  if (!v) return "";
  switch (v->t) {
    case V_UNDEF: return "";
    case V_NULL: return "None";
    case V_BOOL: return v->b ? "True" : "False";
    case V_NUM: {
      double d = v->num;
      if (d == (double)(long long)d && fabs(d) < 1e15) return a_printf(a, "%lld", (long long)d);
      return a_printf(a, "%.17g", d);
    }
    case V_STR: return v->s ? v->s : "";
    case V_LIST: {
      sb s;
      sb_init(&s);
      sb_puts(&s, "[");
      for (int i = 0; i < v->n; i++) {
        if (i) sb_puts(&s, ", ");
        sb_puts(&s, v_tostr(a, v->it[i]));
      }
      sb_puts(&s, "]");
      char *r = a_strndup(a, s.p ? s.p : "[]", s.len);
      sb_free(&s);
      return r;
    }
    case V_DICT:
    case V_NS: {
      sb s;
      sb_init(&s);
      sb_puts(&s, "{");
      for (int i = 0; i < v->nkv; i++) {
        if (i) sb_puts(&s, ", ");
        if (!(v->kv[i].v && v->kv[i].v->t == V_STR)) {
          sb_puts(&s, "'");
          sb_puts(&s, v->kv[i].k);
          sb_puts(&s, "': ");
        }
        sb_puts(&s, v_tostr(a, v->kv[i].v));
      }
      sb_puts(&s, "}");
      char *r = a_strndup(a, s.p ? s.p : "{}", s.len);
      sb_free(&s);
      return r;
    }
    default: return "";
  }
}

/* json.dumps 风格（tojson）：分隔符 ", " / ": "，非 ASCII -> \uXXXX */
static void v_json(arena *a, sb *out, const jval *v) {
  if (!v) {
    sb_puts(out, "null");
    return;
  }
  switch (v->t) {
    case V_UNDEF:
    case V_NULL: sb_puts(out, "null"); break;
    case V_BOOL: sb_puts(out, v->b ? "true" : "false"); break;
    case V_NUM: sb_puts(out, a_printf(a, "%.17g", v->num)); break;
    case V_STR: {
      sb_puts(out, "\"");
      for (const unsigned char *p = (const unsigned char *)v->s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
          case '"': sb_puts(out, "\\\""); break;
          case '\\': sb_puts(out, "\\\\"); break;
          case '\n': sb_puts(out, "\\n"); break;
          case '\r': sb_puts(out, "\\r"); break;
          case '\t': sb_puts(out, "\\t"); break;
          case '\b': sb_puts(out, "\\b"); break;
          case '\f': sb_puts(out, "\\f"); break;
          default:
            if (c < 0x20 || c >= 0x7F) sb_puts(out, a_printf(a, "\\u%04x", c));
            else {
              char ch = (char)c;
              sb_put(out, &ch, 1);
            }
        }
      }
      sb_puts(out, "\"");
      break;
    }
    case V_LIST: {
      sb_puts(out, "[");
      for (int i = 0; i < v->n; i++) {
        if (i) sb_puts(out, ", ");
        v_json(a, out, v->it[i]);
      }
      sb_puts(out, "]");
      break;
    }
    case V_DICT:
    case V_NS: {
      /* Jinja 的 tojson 默认 sort_keys=True */
      int idx[64];
      int nk = v->nkv < 64 ? v->nkv : 64;
      for (int i = 0; i < nk; i++) idx[i] = i;
      for (int i = 1; i < nk; i++) {
        int cur = idx[i], j = i - 1;
        while (j >= 0 && strcmp(v->kv[idx[j]].k, v->kv[cur].k) > 0) {
          idx[j + 1] = idx[j];
          j--;
        }
        idx[j + 1] = cur;
      }
      sb_puts(out, "{");
      for (int i = 0; i < nk; i++) {
        if (i) sb_puts(out, ", ");
        jval key;
        memset(&key, 0, sizeof(key));
        key.t = V_STR;
        key.s = v->kv[idx[i]].k;
        v_json(a, out, &key);
        sb_puts(out, ": ");
        v_json(a, out, v->kv[idx[i]].v);
      }
      sb_puts(out, "}");
      break;
    }
  }
}

static jval *json_to_val(arena *a, const sgl_json *j) {
  if (!j) return v_null(a);
  switch (j->type) {
    case SGL_JSON_NULL: return v_null(a);
    case SGL_JSON_BOOL: return v_bool(a, j->boolean);
    case SGL_JSON_NUMBER: return v_num(a, j->num);
    case SGL_JSON_STRING: return v_str(a, a_strdup(a, j->str ? j->str : ""));
    case SGL_JSON_ARRAY: {
      jval *l = v_list(a);
      for (int i = 0; i < j->count; i++) v_push(a, l, json_to_val(a, j->items[i]));
      return l;
    }
    case SGL_JSON_OBJECT: {
      jval *d = v_dict(a, 0);
      for (int i = 0; i < j->count; i++) v_set(a, d, j->keys[i], json_to_val(a, j->items[i]));
      return d;
    }
  }
  return v_null(a);
}

/* ============================ 词法 ============================ */
enum { T_TEXT, T_OUT, T_STMT };
typedef struct {
  int k;
  const char *s;
  size_t n;
} mtok;

static size_t rstrip_ws(const char *s, size_t n) {
  while (n > 0 && isspace((unsigned char)s[n - 1])) n--;
  return n;
}

static int lex(arena *a, const char *src, size_t len, mtok **out, int *n_out) {
  mtok *toks = NULL;
  int n = 0, cap = 0;
  const char *p = src, *end = src + len;
  const char *text_start = src;
  int text_valid = 1;

#define PUSH_TOK(KIND, START, LEN)                 \
  do {                                             \
    if (n == cap) {                                \
      cap = cap ? cap * 2 : 32;                    \
      mtok *nt = (mtok *)a_alloc(a, sizeof(mtok) * (size_t)cap); \
      if (!nt) return -1;                          \
      if (n) memcpy(nt, toks, sizeof(mtok) * (size_t)n);         \
      toks = nt;                                   \
    }                                              \
    toks[n].k = (KIND);                            \
    toks[n].s = (START);                           \
    toks[n].n = (LEN);                             \
    n++;                                           \
  } while (0)

  for (;;) {
    const char *open = NULL;
    int kind = 0;
    for (const char *q = p; q + 1 < end; q++)
      if (q[0] == '{' && (q[1] == '{' || q[1] == '%' || q[1] == '#')) {
        open = q;
        kind = q[1];
        break;
      }
    if (!open) break;

    if (text_valid && open > text_start) PUSH_TOK(T_TEXT, text_start, (size_t)(open - text_start));

    int left_trim = (open + 2 < end && open[2] == '-');
    if (left_trim && n > 0 && toks[n - 1].k == T_TEXT)
      toks[n - 1].n = rstrip_ws(toks[n - 1].s, toks[n - 1].n);

    if (kind == '#') {
      const char *close = NULL;
      for (const char *q = open + 2; q + 1 < end; q++)
        if (q[0] == '#' && q[1] == '}') {
          close = q;
          break;
        }
      if (!close) return -1;
      p = close + 2;
      text_start = p;
      text_valid = 0;
      continue;
    }

    const char *body = open + 2 + (left_trim ? 1 : 0);
    char close0 = (kind == '{') ? '}' : '%';
    const char *close = NULL;
    for (const char *q = body; q + 1 < end; q++)
      if (q[0] == close0 && q[1] == '}') {
        close = q;
        break;
      }
    if (!close) return -1;
    int right_trim = (close > body && close[-1] == '-');
    size_t bn = (size_t)(close - body) - (size_t)(right_trim ? 1 : 0);
    PUSH_TOK((kind == '{') ? T_OUT : T_STMT, body, bn);

    p = close + 2;
    text_start = p;
    text_valid = 1;
    if (right_trim) {
      while (p < end && isspace((unsigned char)*p)) p++;
      text_start = p;
    }
  }
  if (text_valid && end > text_start) PUSH_TOK(T_TEXT, text_start, (size_t)(end - text_start));
#undef PUSH_TOK
  *out = toks;
  *n_out = n;
  return 0;
}

/* ============================ AST ============================ */
enum { N_TEXT, N_OUT, N_SET, N_FOR, N_IF };
typedef struct jnode jnode;
typedef struct {
  const char *cond;
  size_t clen;
  jnode **body;
  int nbody;
} jbranch;

struct jnode {
  int k;
  const char *text;
  size_t tlen;
  const char *expr;
  size_t elen;
  const char *var;
  const char *lhs;
  size_t llen;
  jnode **body;
  int nbody;
  jnode **ebody;
  int nebody;
  jbranch *br;
  int nbr;
};

typedef struct {
  mtok *t;
  int n, i;
  arena *a;
} tps;

static void stmt_kw(const mtok *m, const char **kw, size_t *kwlen) {
  size_t i = 0;
  while (i < m->n && isspace((unsigned char)m->s[i])) i++;
  size_t j = i;
  while (j < m->n && !isspace((unsigned char)m->s[j]) && m->s[j] != '\0') j++;
  *kw = m->s + i;
  *kwlen = j - i;
}

static int kw_is(const char *kw, size_t len, const char *want) {
  return len == strlen(want) && strncmp(kw, want, len) == 0;
}

static jnode *node_new(tps *p, int k) {
  jnode *nd = (jnode *)a_alloc(p->a, sizeof(jnode));
  if (nd) nd->k = k;
  return nd;
}

static int at_stop(tps *p, const char *const *stops, int nstops) {
  if (p->i >= p->n) return 0;
  mtok *m = &p->t[p->i];
  if (m->k != T_STMT) return 0;
  const char *kw;
  size_t kwlen;
  stmt_kw(m, &kw, &kwlen);
  for (int s = 0; s < nstops; s++)
    if (kw_is(kw, kwlen, stops[s])) return 1;
  return 0;
}

static int list_push(arena *a, jnode ***list, int *n, int *cap, jnode *nd) {
  if (*n == *cap) {
    int nc = *cap ? *cap * 2 : 16;
    jnode **nl = (jnode **)a_alloc(a, sizeof(jnode *) * (size_t)nc);
    if (!nl) return -1;
    if (*n) memcpy(nl, *list, sizeof(jnode *) * (size_t)*n);
    *list = nl;
    *cap = nc;
  }
  (*list)[(*n)++] = nd;
  return 0;
}

static jnode **parse_nodes(tps *p, const char *const *stops, int nstops, int *n_out) {
  jnode **list = NULL;
  int n = 0, cap = 0;
  while (p->i < p->n) {
    if (at_stop(p, stops, nstops)) break;
    mtok *m = &p->t[p->i];
    jnode *nd = NULL;

    if (m->k == T_TEXT) {
      nd = node_new(p, N_TEXT);
      if (!nd) return NULL;
      nd->text = m->s;
      nd->tlen = m->n;
      p->i++;
    } else if (m->k == T_OUT) {
      nd = node_new(p, N_OUT);
      if (!nd) return NULL;
      nd->expr = m->s;
      nd->elen = m->n;
      p->i++;
    } else {
      const char *kw;
      size_t kwlen;
      stmt_kw(m, &kw, &kwlen);
      if (kw_is(kw, kwlen, "if")) {
        nd = node_new(p, N_IF);
        if (!nd) return NULL;
        for (int guard = 0; guard < 1000; guard++) {
          mtok *cur = &p->t[p->i];
          const char *k2;
          size_t l2;
          stmt_kw(cur, &k2, &l2);
          const char *cond = NULL;
          size_t clen = 0;
          if (kw_is(k2, l2, "if") || kw_is(k2, l2, "elif")) {
            cond = k2 + l2;
            clen = (size_t)(cur->s + cur->n - cond);
          }
          p->i++;
          static const char *st[] = {"elif", "else", "endif"};
          int nbody = 0;
          jnode **body = parse_nodes(p, st, 3, &nbody);
          if (!body && nbody < 0) return NULL;
          if (nd->nbr == 0) {
            nd->br = (jbranch *)a_alloc(p->a, sizeof(jbranch) * 8);
            if (!nd->br) return NULL;
          } else if (nd->nbr % 8 == 0) {
            jbranch *nb = (jbranch *)a_alloc(p->a, sizeof(jbranch) * (size_t)(nd->nbr + 8));
            if (!nb) return NULL;
            memcpy(nb, nd->br, sizeof(jbranch) * (size_t)nd->nbr);
            nd->br = nb;
          }
          nd->br[nd->nbr].cond = cond;
          nd->br[nd->nbr].clen = clen;
          nd->br[nd->nbr].body = body;
          nd->br[nd->nbr].nbody = nbody;
          nd->nbr++;
          if (p->i >= p->n) return NULL;
          mtok *nx = &p->t[p->i];
          const char *k3;
          size_t l3;
          stmt_kw(nx, &k3, &l3);
          if (kw_is(k3, l3, "elif")) continue;
          if (kw_is(k3, l3, "else")) {
            p->i++;
            static const char *st2[] = {"endif"};
            int ne = 0;
            jnode **eb = parse_nodes(p, st2, 1, &ne);
            if (!eb && ne < 0) return NULL;
            nd->ebody = eb;
            nd->nebody = ne;
            if (p->i >= p->n) return NULL;
            p->i++; /* endif */
            break;
          }
          p->i++; /* endif */
          break;
        }
      } else if (kw_is(kw, kwlen, "for")) {
        nd = node_new(p, N_FOR);
        if (!nd) return NULL;
        const char *q = kw + kwlen;
        const char *me = m->s + m->n;
        while (q < me && isspace((unsigned char)*q)) q++;
        const char *vs = q;
        while (q < me && (isalnum((unsigned char)*q) || *q == '_')) q++;
        nd->var = a_strndup(p->a, vs, (size_t)(q - vs));
        while (q < me && isspace((unsigned char)*q)) q++;
        if (q + 2 <= me && strncmp(q, "in", 2) == 0) q += 2;
        while (q < me && isspace((unsigned char)*q)) q++;
        nd->expr = q;
        nd->elen = (size_t)(me - q);
        p->i++;
        static const char *st[] = {"else", "endfor"};
        int nbody = 0;
        jnode **body = parse_nodes(p, st, 2, &nbody);
        if (!body && nbody < 0) return NULL;
        nd->body = body;
        nd->nbody = nbody;
        if (p->i < p->n) {
          mtok *nx = &p->t[p->i];
          const char *k2;
          size_t l2;
          stmt_kw(nx, &k2, &l2);
          if (kw_is(k2, l2, "else")) {
            p->i++;
            static const char *st2[] = {"endfor"};
            int ne = 0;
            jnode **eb = parse_nodes(p, st2, 1, &ne);
            if (!eb && ne < 0) return NULL;
            nd->ebody = eb;
            nd->nebody = ne;
          }
          if (p->i < p->n) p->i++; /* endfor */
        }
      } else if (kw_is(kw, kwlen, "set")) {
        nd = node_new(p, N_SET);
        if (!nd) return NULL;
        const char *q = kw + kwlen;
        const char *me = m->s + m->n;
        while (q < me && isspace((unsigned char)*q)) q++;
        const char *eq = NULL;
        for (const char *r = q; r < me; r++)
          if (*r == '=') {
            eq = r;
            break;
          }
        if (!eq) return NULL;
        const char *le = eq;
        while (le > q && isspace((unsigned char)le[-1])) le--;
        nd->lhs = a_strndup(p->a, q, (size_t)(le - q));
        nd->llen = (size_t)(le - q);
        const char *vs = eq + 1;
        while (vs < me && isspace((unsigned char)*vs)) vs++;
        nd->expr = vs;
        nd->elen = (size_t)(me - vs);
        p->i++;
      } else {
        p->i++; /* 未知语句忽略 */
        continue;
      }
    }
    if (list_push(p->a, &list, &n, &cap, nd) != 0) {
      *n_out = -1;
      return NULL;
    }
  }
  *n_out = n;
  return list;
}

/* ============================ 作用域 ============================ */
typedef struct sc {
  const char *name;
  jval *v;
  struct sc *next;
} sc;

typedef struct {
  arena *a;
  sc *vars;
  sb *out;
  int err;
  char msg[256];
} rctx;

static void scope_push(rctx *c, const char *name, jval *v) {
  sc *s = (sc *)a_alloc(c->a, sizeof(sc));
  if (!s) return;
  s->name = name;
  s->v = v;
  s->next = c->vars;
  c->vars = s;
}

static jval *scope_get(rctx *c, const char *name) {
  for (sc *s = c->vars; s; s = s->next)
    if (strcmp(s->name, name) == 0) return s->v;
  return NULL;
}

/* ============================ 表达式求值 ============================ */
typedef struct {
  const char *p, *end;
  rctx *c;
  int err;
  char msg[256];
} ex;

static void ex_ws(ex *e) {
  while (e->p < e->end && isspace((unsigned char)*e->p)) e->p++;
}

static void ex_fail(ex *e, const char *msg) {
  if (!e->err) {
    e->err = 1;
    snprintf(e->msg, sizeof(e->msg), "%s", msg);
  }
}

static int peek_word(const ex *e, const char *w) {
  const char *p = e->p;
  while (p < e->end && isspace((unsigned char)*p)) p++;
  size_t n = strlen(w);
  if ((size_t)(e->end - p) < n) return 0;
  if (strncmp(p, w, n) != 0) return 0;
  if (p + n < e->end && (isalnum((unsigned char)p[n]) || p[n] == '_')) return 0;
  return 1;
}

static int ex_word(ex *e, const char *w) {
  size_t n = strlen(w);
  if ((size_t)(e->end - e->p) < n) return 0;
  if (strncmp(e->p, w, n) != 0) return 0;
  if (e->p + n < e->end && (isalnum((unsigned char)e->p[n]) || e->p[n] == '_')) return 0;
  e->p += n;
  return 1;
}

static int ex_op(ex *e, const char *op) {
  size_t n = strlen(op);
  ex_ws(e);
  if ((size_t)(e->end - e->p) < n || strncmp(e->p, op, n) != 0) return 0;
  e->p += n;
  return 1;
}

static const char *ex_strlit(ex *e) {
  ex_ws(e);
  if (e->p >= e->end) return NULL;
  char q = *e->p;
  if (q != '"' && q != '\'') return NULL;
  e->p++;
  sb s;
  sb_init(&s);
  while (e->p < e->end && *e->p != q) {
    char c = *e->p++;
    if (c == '\\' && e->p < e->end) {
      char x = *e->p++;
      switch (x) {
        case 'n': c = '\n'; break;
        case 't': c = '\t'; break;
        case 'r': c = '\r'; break;
        case '0': c = '\0'; break;
        case '\\': c = '\\'; break;
        case '\'': c = '\''; break;
        case '"': c = '"'; break;
        case 'u': { /* 只处理 BMP 的 \uXXXX，输出 UTF-8 */
          if (e->end - e->p < 4) { c = 'u'; break; }
          unsigned cp = 0;
          for (int i = 0; i < 4; i++) {
            char h = e->p[i];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
          }
          e->p += 4;
          char u[4];
          int un = 0;
          if (cp < 0x80) u[un++] = (char)cp;
          else if (cp < 0x800) {
            u[un++] = (char)(0xC0 | (cp >> 6));
            u[un++] = (char)(0x80 | (cp & 0x3F));
          } else {
            u[un++] = (char)(0xE0 | (cp >> 12));
            u[un++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            u[un++] = (char)(0x80 | (cp & 0x3F));
          }
          sb_put(&s, u, (size_t)un);
          continue;
        }
        default: c = x; break;
      }
    }
    sb_put(&s, &c, 1);
  }
  if (e->p < e->end) e->p++; /* 结尾引号 */
  char *r = a_strndup(e->c->a, s.p ? s.p : "", s.len);
  sb_free(&s);
  return r;
}

static int ex_ident(ex *e, char *buf, size_t cap) {
  ex_ws(e);
  if (e->p >= e->end) return 0;
  if (!(isalpha((unsigned char)*e->p) || *e->p == '_')) return 0;
  size_t n = 0;
  while (e->p < e->end && (isalnum((unsigned char)*e->p) || *e->p == '_')) {
    if (n + 1 < cap) buf[n++] = *e->p;
    e->p++;
  }
  buf[n] = 0;
  return 1;
}

static jval *ex_expr(ex *e);

static const char *val_str_of(ex *e, jval *v) { return v_tostr(e->c->a, v); }

/* 字符串方法调用 */
static jval *call_method(ex *e, jval *obj, const char *name, jval **args, int nargs) {
  arena *a = e->c->a;
  const char *s = v_tostr(a, obj);
  if (!strcmp(name, "startswith"))
    return v_bool(a, (nargs >= 1 && s && args[0]) ? (strncmp(s, val_str_of(e, args[0]), strlen(val_str_of(e, args[0]))) == 0) : 0);
  if (!strcmp(name, "endswith")) {
    if (nargs < 1 || !s) return v_bool(a, 0);
    const char *sfx = val_str_of(e, args[0]);
    size_t ls = strlen(s), lx = strlen(sfx);
    return v_bool(a, ls >= lx && strcmp(s + ls - lx, sfx) == 0);
  }
  if (!strcmp(name, "split")) {
    jval *l = v_list(a);
    if (nargs < 1 || !args[0] || args[0]->t == V_UNDEF || args[0]->t == V_NULL || !args[0]->s[0]) {
      /* 无分隔符：按空白切分（去掉空串）*/
      const char *p = s;
      while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        const char *q = p;
        while (*q && !isspace((unsigned char)*q)) q++;
        v_push(a, l, v_str(a, a_strndup(a, p, (size_t)(q - p))));
        p = q;
      }
      return l;
    }
    const char *sep = val_str_of(e, args[0]);
    size_t sl = strlen(sep);
    const char *p = s;
    for (;;) {
      const char *hit = strstr(p, sep);
      if (!hit) {
        v_push(a, l, v_str(a, a_strdup(a, p)));
        break;
      }
      v_push(a, l, v_str(a, a_strndup(a, p, (size_t)(hit - p))));
      p = hit + sl;
    }
    return l;
  }
  if (!strcmp(name, "strip") || !strcmp(name, "lstrip") || !strcmp(name, "rstrip")) {
    const char *chars = NULL;
    if (nargs >= 1 && args[0] && args[0]->t == V_STR && args[0]->s[0]) chars = args[0]->s;
    const char *b = s, *en2 = s + strlen(s);
    if (!strcmp(name, "strip") || !strcmp(name, "lstrip")) {
      while (*b && (chars ? strchr(chars, *b) != NULL : isspace((unsigned char)*b))) b++;
    }
    if (!strcmp(name, "strip") || !strcmp(name, "rstrip")) {
      while (en2 > b && (chars ? strchr(chars, en2[-1]) != NULL : isspace((unsigned char)en2[-1]))) en2--;
    }
    return v_str(a, a_strndup(a, b, (size_t)(en2 - b)));
  }
  if (!strcmp(name, "lower")) {
    char *d = a_strdup(a, s);
    for (char *q = d; *q; q++) *q = (char)tolower((unsigned char)*q);
    return v_str(a, d);
  }
  if (!strcmp(name, "upper")) {
    char *d = a_strdup(a, s);
    for (char *q = d; *q; q++) *q = (char)toupper((unsigned char)*q);
    return v_str(a, d);
  }
  if (!strcmp(name, "replace") && nargs >= 2) {
    const char *from = val_str_of(e, args[0]), *to = val_str_of(e, args[1]);
    sb o;
    sb_init(&o);
    const char *p = s;
    size_t fl = strlen(from);
    while (fl && (p = strstr(p, from)) != NULL) {
      sb_put(&o, s, (size_t)(p - s));
      sb_puts(&o, to);
      s = p + fl;
      p = s;
    }
    sb_puts(&o, s);
    char *r = a_strndup(a, o.p ? o.p : "", o.len);
    sb_free(&o);
    return v_str(a, r);
  }
  if (!strcmp(name, "format")) return v_str(a, s); /* 简化：不替换占位 */
  if (!strcmp(name, "join") && nargs >= 1) {
    const char *sep = val_str_of(e, args[0]);
    sb o;
    sb_init(&o);
    if (obj && obj->t == V_LIST)
      for (int i = 0; i < obj->n; i++) {
        if (i) sb_puts(&o, sep);
        sb_puts(&o, v_tostr(a, obj->it[i]));
      }
    char *r = a_strndup(a, o.p ? o.p : "", o.len);
    sb_free(&o);
    return v_str(a, r);
  }
  ex_fail(e, "unknown string method");
  return v_undef(a);
}

static jval *apply_filter(ex *e, jval *v, const char *name, jval **args, int nargs) {
  arena *a = e->c->a;
  if (!strcmp(name, "tojson")) {
    sb o;
    sb_init(&o);
    v_json(a, &o, v);
    char *r = a_strndup(a, o.p ? o.p : "", o.len);
    sb_free(&o);
    return v_str(a, r);
  }
  if (!strcmp(name, "length") || !strcmp(name, "count")) {
    if (!v) return v_num(a, 0);
    if (v->t == V_STR) return v_num(a, (double)strlen(v->s));
    if (v->t == V_LIST) return v_num(a, (double)v->n);
    if (v->t == V_DICT || v->t == V_NS) return v_num(a, (double)v->nkv);
    return v_num(a, 0);
  }
  if (!strcmp(name, "default") || !strcmp(name, "d")) {
    if (v && v->t != V_UNDEF) return v;
    return nargs >= 1 ? args[0] : v_str(a, "");
  }
  if (!strcmp(name, "trim")) {
    const char *s = v_tostr(a, v);
    const char *b = s, *en2 = s + strlen(s);
    while (*b && isspace((unsigned char)*b)) b++;
    while (en2 > b && isspace((unsigned char)en2[-1])) en2--;
    return v_str(a, a_strndup(a, b, (size_t)(en2 - b)));
  }
  if (!strcmp(name, "string")) return v_str(a, v_tostr(a, v));
  if (!strcmp(name, "int")) {
    const char *s = v_tostr(a, v);
    return v_num(a, strtod(s, NULL));
  }
  if (!strcmp(name, "float")) return v_num(a, strtod(v_tostr(a, v), NULL));
  if (!strcmp(name, "first")) {
    if (v && v->t == V_LIST && v->n) return v->it[0];
    if (v && v->t == V_STR && v->s[0]) return v_str(a, a_strndup(a, v->s, 1));
    return v_undef(a);
  }
  if (!strcmp(name, "last")) {
    if (v && v->t == V_LIST && v->n) return v->it[v->n - 1];
    if (v && v->t == V_STR && v->s[0]) return v_str(a, a_strndup(a, v->s + strlen(v->s) - 1, 1));
    return v_undef(a);
  }
  if (!strcmp(name, "join")) {
    const char *sep = nargs >= 1 ? val_str_of(e, args[0]) : "";
    sb o;
    sb_init(&o);
    if (v && v->t == V_LIST)
      for (int i = 0; i < v->n; i++) {
        if (i) sb_puts(&o, sep);
        sb_puts(&o, v_tostr(a, v->it[i]));
      }
    char *r = a_strndup(a, o.p ? o.p : "", o.len);
    sb_free(&o);
    return v_str(a, r);
  }
  if (!strcmp(name, "upper")) {
    char *d = a_strdup(a, v_tostr(a, v));
    for (char *q = d; *q; q++) *q = (char)toupper((unsigned char)*q);
    return v_str(a, d);
  }
  if (!strcmp(name, "lower")) {
    char *d = a_strdup(a, v_tostr(a, v));
    for (char *q = d; *q; q++) *q = (char)tolower((unsigned char)*q);
    return v_str(a, d);
  }
  if (!strcmp(name, "list")) {
    if (v && v->t == V_LIST) return v;
    jval *l = v_list(a);
    const char *s = v_tostr(a, v);
    for (const char *p = s; *p; p++) v_push(a, l, v_str(a, a_strndup(a, p, 1)));
    return l;
  }
  if (!strcmp(name, "safe") || !strcmp(name, "e") || !strcmp(name, "escape") || !strcmp(name, "forceescape"))
    return v;
  ex_fail(e, "unknown filter");
  return v_undef(a);
}

static jval *ex_primary(ex *e) {
  arena *a = e->c->a;
  ex_ws(e);
  if (e->p >= e->end) {
    ex_fail(e, "unexpected end of expression");
    return v_undef(a);
  }
  char c = *e->p;
  if (c == '(') {
    e->p++;
    jval *v = ex_expr(e);
    ex_ws(e);
    if (e->p < e->end && *e->p == ')') e->p++;
    return v;
  }
  if (c == '[') { /* list 字面量 */
    e->p++;
    jval *l = v_list(a);
    ex_ws(e);
    if (e->p < e->end && *e->p == ']') {
      e->p++;
      return l;
    }
    for (;;) {
      v_push(a, l, ex_expr(e));
      ex_ws(e);
      if (e->p < e->end && *e->p == ',') {
        e->p++;
        continue;
      }
      if (e->p < e->end && *e->p == ']') e->p++;
      break;
    }
    return l;
  }
  if (c == '"' || c == '\'') {
    const char *s = ex_strlit(e);
    return v_str(a, s ? s : "");
  }
  if (isdigit((unsigned char)c) || (c == '-' && e->p + 1 < e->end && isdigit((unsigned char)e->p[1]))) {
    char *endp = NULL;
    double d = strtod(e->p, &endp);
    if (endp == e->p) {
      ex_fail(e, "bad number");
      return v_undef(a);
    }
    e->p = endp;
    return v_num(a, d);
  }
  char name[128];
  if (!ex_ident(e, name, sizeof(name))) {
    ex_fail(e, "expected identifier");
    return v_undef(a);
  }
  ex_ws(e);
  if (e->p < e->end && *e->p == '(') {
    /* 函数调用 */
    e->p++;
    jval *args[16];
    const char *kw_names[16];
    int nargs = 0;
    ex_ws(e);
    if (e->p < e->end && *e->p == ')') {
      e->p++;
    } else {
      for (;;) {
        ex_ws(e);
        /* name=value 形式的参数 */
        const char *save = e->p;
        char an[128];
        if (ex_ident(e, an, sizeof(an))) {
          ex_ws(e);
          if (e->p < e->end && *e->p == '=' && (e->p + 1 >= e->end || e->p[1] != '=')) {
            e->p++;
            if (nargs < 16) {
              args[nargs] = ex_expr(e);
              kw_names[nargs] = a_strdup(a, an);
              nargs++;
            }
            ex_ws(e);
            if (e->p < e->end && *e->p == ',') {
              e->p++;
              continue;
            }
            if (e->p < e->end && *e->p == ')') e->p++;
            break;
          }
        }
        e->p = save;
        if (nargs < 16) {
          args[nargs] = ex_expr(e);
          kw_names[nargs] = NULL;
          nargs++;
        }
        ex_ws(e);
        if (e->p < e->end && *e->p == ',') {
          e->p++;
          continue;
        }
        if (e->p < e->end && *e->p == ')') e->p++;
        break;
      }
    }
    if (!strcmp(name, "namespace")) {
      jval *ns = v_dict(a, 1);
      for (int i = 0; i < nargs; i++)
        if (kw_names[i]) v_set(a, ns, kw_names[i], args[i]);
      return ns;
    }
    if (!strcmp(name, "raise_exception")) {
      const char *m = nargs >= 1 ? v_tostr(a, args[0]) : "raise_exception";
      ex_fail(e, m);
      return v_undef(a);
    }
    if (!strcmp(name, "range")) {
      jval *l = v_list(a);
      long b = 0, en2 = 0, st = 1;
      if (nargs == 1) en2 = (long)args[0]->num;
      else if (nargs >= 2) {
        b = (long)args[0]->num;
        en2 = (long)args[1]->num;
      }
      if (nargs >= 3) st = (long)args[2]->num;
      if (st == 0) st = 1;
      for (long i = b; (st > 0 ? i < en2 : i > en2); i += st) v_push(a, l, v_num(a, (double)i));
      return l;
    }
    /* 未知函数：按无参返回 undef */
    return v_undef(a);
  }
  if (!strcmp(name, "true") || !strcmp(name, "True")) return v_bool(a, 1);
  if (!strcmp(name, "false") || !strcmp(name, "False")) return v_bool(a, 0);
  if (!strcmp(name, "none") || !strcmp(name, "None")) return v_null(a);
  jval *v = scope_get(e->c, name);
  if (!v) return v_undef(a);
  return v;
}

static jval *ex_postfix(ex *e) {
  arena *a = e->c->a;
  jval *v = ex_primary(e);
  if (e->err) return v;
  for (;;) {
    ex_ws(e);
    if (e->p >= e->end) break;
    char c = *e->p;
    if (c == '.') {
      e->p++;
      char name[128];
      if (!ex_ident(e, name, sizeof(name))) break;
      ex_ws(e);
      if (e->p < e->end && *e->p == '(') {
        e->p++;
        jval *args[8];
        int nargs = 0;
        ex_ws(e);
        if (e->p < e->end && *e->p == ')') e->p++;
        else {
          for (;;) {
            if (nargs < 8) args[nargs++] = ex_expr(e);
            ex_ws(e);
            if (e->p < e->end && *e->p == ',') {
              e->p++;
              continue;
            }
            if (e->p < e->end && *e->p == ')') e->p++;
            break;
          }
        }
        v = call_method(e, v, name, args, nargs);
      } else {
        jval *got = v_get(v, name);
        v = got ? got : v_undef(a);
      }
    } else if (c == '[') {
      e->p++;
      ex_ws(e);
      jval *first = NULL;
      int is_slice = 0;
      if (e->p < e->end && *e->p == ':') {
        is_slice = 1;
      } else {
        first = ex_expr(e);
        ex_ws(e);
        if (e->p < e->end && *e->p == ':') is_slice = 1;
      }
      if (is_slice) {
        if (e->p < e->end && *e->p == ':') e->p++;
        ex_ws(e);
        jval *second = NULL;
        if (e->p < e->end && *e->p != ']' && *e->p != ':') second = ex_expr(e);
        ex_ws(e);
        long step = 1;
        if (e->p < e->end && *e->p == ':') {
          e->p++;
          ex_ws(e);
          if (e->p < e->end && *e->p != ']') {
            jval *st = ex_expr(e);
            step = st ? (long)st->num : 1;
          }
          ex_ws(e);
        }
        if (e->p < e->end && *e->p == ']') e->p++;
        /* 只支持 list / str 的切片，step 允许 1 / -1 */
        long n = 0;
        if (v && v->t == V_LIST) n = v->n;
        else if (v && v->t == V_STR) n = (long)strlen(v->s);
        if (step == 0) step = 1;
        long b = first ? (long)(first->num < 0 ? first->num + n : first->num) : (step > 0 ? 0 : n - 1);
        long en2 = second ? (long)(second->num < 0 ? second->num + n : second->num) : (step > 0 ? n : -1);
        if (b < 0) b = 0;
        if (en2 > n) en2 = n;
        if (v && v->t == V_LIST) {
          jval *l = v_list(a);
          for (long i = b; step > 0 ? i < en2 : i > en2; i += step)
            if (i >= 0 && i < n) v_push(a, l, v->it[i]);
          v = l;
        } else if (v && v->t == V_STR) {
          sb o;
          sb_init(&o);
          for (long i = b; step > 0 ? i < en2 : i > en2; i += step)
            if (i >= 0 && i < n) sb_put(&o, v->s + i, 1);
          char *r = a_strndup(a, o.p ? o.p : "", o.len);
          sb_free(&o);
          v = v_str(a, r);
        } else {
          v = v_undef(a);
        }
      } else {
        if (e->p < e->end && *e->p == ']') e->p++;
        jval *idx = first;
        if (v && v->t == V_LIST && idx && idx->t == V_NUM) {
          long i = (long)idx->num;
          if (i < 0) i += v->n;
          v = (i >= 0 && i < v->n) ? v->it[i] : v_undef(a);
        } else if (v && v->t == V_STR && idx && idx->t == V_NUM) {
          long i = (long)idx->num;
          long len = (long)strlen(v->s);
          if (i < 0) i += len;
          v = (i >= 0 && i < len) ? v_str(a, a_strndup(a, v->s + i, 1)) : v_undef(a);
        } else if ((v && (v->t == V_DICT || v->t == V_NS)) && idx) {
          jval *got = (idx->t == V_STR) ? v_get(v, idx->s) : NULL;
          v = got ? got : v_undef(a);
        } else {
          v = v_undef(a);
        }
      }
    } else if (c == '|') {
      e->p++;
      char name[128];
      if (!ex_ident(e, name, sizeof(name))) break;
      jval *args[8];
      int nargs = 0;
      ex_ws(e);
      if (e->p < e->end && *e->p == '(') {
        e->p++;
        ex_ws(e);
        if (e->p < e->end && *e->p == ')') e->p++;
        else {
          for (;;) {
            if (nargs < 8) args[nargs++] = ex_expr(e);
            ex_ws(e);
            if (e->p < e->end && *e->p == ',') {
              e->p++;
              continue;
            }
            if (e->p < e->end && *e->p == ')') e->p++;
            break;
          }
        }
      }
      v = apply_filter(e, v, name, args, nargs);
      if (e->err) return v;
    } else {
      break;
    }
  }
  return v;
}

static jval *ex_unary(ex *e) {
  arena *a = e->c->a;
  ex_ws(e);
  if (e->p < e->end && *e->p == '-') {
    e->p++;
    jval *v = ex_unary(e);
    return v_num(a, -(v && v->t == V_NUM ? v->num : 0));
  }
  if (e->p < e->end && *e->p == '+') {
    e->p++;
    return ex_unary(e);
  }
  return ex_postfix(e);
}

static jval *ex_mul(ex *e) {
  arena *a = e->c->a;
  jval *v = ex_unary(e);
  for (;;) {
    ex_ws(e);
    if (e->p >= e->end) break;
    if (*e->p == '*') {
      e->p++;
      jval *r = ex_unary(e);
      v = v_num(a, (v ? v->num : 0) * (r ? r->num : 0));
    } else if (*e->p == '/') {
      e->p++;
      jval *r = ex_unary(e);
      double d = r ? r->num : 0;
      v = v_num(a, d == 0 ? 0 : (v ? v->num : 0) / d);
    } else if (*e->p == '%') {
      e->p++;
      jval *r = ex_unary(e);
      double d = r ? r->num : 0;
      v = v_num(a, d == 0 ? 0 : fmod(v ? v->num : 0, d));
    } else break;
  }
  return v;
}

static jval *ex_add(ex *e) {
  arena *a = e->c->a;
  jval *v = ex_mul(e);
  for (;;) {
    ex_ws(e);
    if (e->p >= e->end) break;
    if (*e->p == '+') {
      e->p++;
      jval *r = ex_mul(e);
      if (v && r && v->t == V_NUM && r->t == V_NUM) v = v_num(a, v->num + r->num);
      else {
        const char *ls = v_tostr(a, v), *rs = v_tostr(a, r);
        size_t n = strlen(ls) + strlen(rs) + 1;
        char *buf = (char *)a_alloc(a, n);
        if (buf) {
          snprintf(buf, n, "%s%s", ls, rs);
          v = v_str(a, buf);
        }
      }
    } else if (*e->p == '-' && e->p + 1 < e->end &&
               (e->p[1] == ' ' || e->p[1] == '(' || isdigit((unsigned char)e->p[1]) || e->p[1] == '_' ||
                isalpha((unsigned char)e->p[1]))) {
      e->p++;
      jval *r = ex_mul(e);
      v = v_num(a, (v ? v->num : 0) - (r ? r->num : 0));
    } else break;
  }
  return v;
}

static jval *ex_not(ex *e) {
  arena *a = e->c->a;
  ex_ws(e);
  if (ex_word(e, "not")) {
    jval *v = ex_not(e);
    return v_bool(a, !v_truthy(v));
  }
  return ex_add(e);
}

static jval *ex_cmp(ex *e) {
  arena *a = e->c->a;
  jval *v = ex_not(e);
  for (;;) {
    ex_ws(e);
    if (e->p >= e->end) break;
    if (ex_op(e, "==")) {
      jval *r = ex_not(e);
      v = v_bool(a, v_equal(v, r));
    } else if (ex_op(e, "!=")) {
      jval *r = ex_not(e);
      v = v_bool(a, !v_equal(v, r));
    } else if (ex_op(e, "<=")) {
      jval *r = ex_not(e);
      v = v_bool(a, (v ? v->num : 0) <= (r ? r->num : 0));
    } else if (ex_op(e, ">=")) {
      jval *r = ex_not(e);
      v = v_bool(a, (v ? v->num : 0) >= (r ? r->num : 0));
    } else if (ex_op(e, "<")) {
      jval *r = ex_not(e);
      v = v_bool(a, (v ? v->num : 0) < (r ? r->num : 0));
    } else if (ex_op(e, ">")) {
      jval *r = ex_not(e);
      v = v_bool(a, (v ? v->num : 0) > (r ? r->num : 0));
    } else if (peek_word(e, "not")) {
      const char *save = e->p;
      ex_ws(e);
      ex_word(e, "not");
      ex_ws(e);
      if (!ex_word(e, "in")) {
        e->p = save;
        break;
      }
      jval *r = ex_not(e);
      int in = 0;
      if (r && r->t == V_LIST) {
        for (int i = 0; i < r->n; i++)
          if (v_equal(v, r->it[i])) {
            in = 1;
            break;
          }
      } else if (r && r->t == V_STR) {
        const char *needle = v_tostr(a, v);
        in = needle[0] && strstr(r->s, needle) != NULL;
      } else if (r && (r->t == V_DICT || r->t == V_NS) && v && v->t == V_STR) {
        in = v_get(r, v->s) != NULL;
      }
      v = v_bool(a, !in);
    } else if (ex_word(e, "in")) {
      jval *r = ex_not(e);
      int in = 0;
      if (r && r->t == V_LIST) {
        for (int i = 0; i < r->n; i++)
          if (v_equal(v, r->it[i])) {
            in = 1;
            break;
          }
      } else if (r && r->t == V_STR) {
        const char *needle = v_tostr(a, v);
        in = needle[0] && strstr(r->s, needle) != NULL;
      } else if (r && (r->t == V_DICT || r->t == V_NS) && v && v->t == V_STR) {
        in = v_get(r, v->s) != NULL;
      }
      v = v_bool(a, in);
    } else if (ex_word(e, "is")) {
      int neg = 0;
      ex_ws(e);
      if (ex_word(e, "not")) neg = 1;
      char tname[64];
      if (!ex_ident(e, tname, sizeof(tname))) break;
      int res = 0;
      int vt = v ? v->t : V_UNDEF;
      if (!strcmp(tname, "defined")) res = (vt != V_UNDEF);
      else if (!strcmp(tname, "undefined")) res = (vt == V_UNDEF);
      else if (!strcmp(tname, "string")) res = (vt == V_STR);
      else if (!strcmp(tname, "number")) res = (vt == V_NUM);
      else if (!strcmp(tname, "integer")) res = (vt == V_NUM);
      else if (!strcmp(tname, "boolean")) res = (vt == V_BOOL);
      else if (!strcmp(tname, "true")) res = (vt == V_BOOL && v->b);
      else if (!strcmp(tname, "false")) res = (vt == V_BOOL && !v->b);
      else if (!strcmp(tname, "none")) res = (vt == V_NULL);
      else if (!strcmp(tname, "mapping")) res = (vt == V_DICT || vt == V_NS);
      else if (!strcmp(tname, "sequence") || !strcmp(tname, "iterable"))
        res = (vt == V_LIST || vt == V_STR || vt == V_DICT || vt == V_NS);
      else res = 0;
      v = v_bool(a, neg ? !res : res);
    } else {
      break;
    }
  }
  return v;
}

static jval *ex_and(ex *e) {
  jval *v = ex_cmp(e);
  for (;;) {
    ex_ws(e);
    if (!ex_word(e, "and")) break;
    jval *r = ex_cmp(e);
    if (v_truthy(v)) v = r;
  }
  return v;
}

static jval *ex_expr(ex *e) {
  arena *a = e->c->a;
  jval *v = ex_and(e);
  for (;;) {
    ex_ws(e);
    if (!ex_word(e, "or")) break;
    jval *r = ex_and(e);
    if (!v_truthy(v)) v = r;
  }
  (void)a;
  return v;
}

/* 在一段表达式文本上求值 */
static jval *eval_text(rctx *c, const char *s, size_t len) {
  ex e;
  memset(&e, 0, sizeof(e));
  e.p = s;
  e.end = s + len;
  e.c = c;
  jval *v = ex_expr(&e);
  if (e.err && !c->err) {
    c->err = 1;
    snprintf(c->msg, sizeof(c->msg), "%s", e.msg);
  }
  return v;
}

/* ============================ 渲染 ============================ */
static void run_nodes(rctx *c, jnode **ns, int n);

static jval *make_loop(arena *a, int i, int n) {
  jval *l = v_dict(a, 0);
  v_set(a, l, "index0", v_num(a, i));
  v_set(a, l, "index", v_num(a, i + 1));
  v_set(a, l, "first", v_bool(a, i == 0));
  v_set(a, l, "last", v_bool(a, i == n - 1));
  v_set(a, l, "length", v_num(a, n));
  v_set(a, l, "revindex", v_num(a, n - i));
  v_set(a, l, "revindex0", v_num(a, n - i - 1));
  return l;
}

static void run_node(rctx *c, jnode *nd) {
  if (!nd || c->err) return;
  arena *a = c->a;
  switch (nd->k) {
    case N_TEXT: sb_put(c->out, nd->text, nd->tlen); break;
    case N_OUT: {
      jval *v = eval_text(c, nd->expr, nd->elen);
      if (!c->err) sb_puts(c->out, v_tostr(a, v));
      break;
    }
    case N_SET: {
      jval *v = eval_text(c, nd->expr, nd->elen);
      if (c->err) return;
      const char *dot = NULL;
      for (size_t i = 0; i < nd->llen; i++)
        if (nd->lhs[i] == '.') {
          dot = nd->lhs + i;
          break;
        }
      if (!dot) {
        scope_push(c, nd->lhs, v);
      } else {
        char root[128];
        size_t rl = (size_t)(dot - nd->lhs);
        if (rl >= sizeof(root)) rl = sizeof(root) - 1;
        memcpy(root, nd->lhs, rl);
        root[rl] = 0;
        jval *obj = scope_get(c, root);
        const char *attr = dot + 1;
        if (obj && (obj->t == V_NS || obj->t == V_DICT)) v_set(a, obj, attr, v);
      }
      break;
    }
    case N_IF: {
      for (int i = 0; i < nd->nbr; i++) {
        if (!nd->br[i].cond) {
          run_nodes(c, nd->br[i].body, nd->br[i].nbody);
          return;
        }
        jval *cond = eval_text(c, nd->br[i].cond, nd->br[i].clen);
        if (c->err) return;
        if (v_truthy(cond)) {
          run_nodes(c, nd->br[i].body, nd->br[i].nbody);
          return;
        }
      }
      run_nodes(c, nd->ebody, nd->nebody);
      break;
    }
    case N_FOR: {
      jval *it = eval_text(c, nd->expr, nd->elen);
      if (c->err) return;
      sc *saved = c->vars;
      int iterated = 0;
      if (it && it->t == V_LIST) {
        for (int i = 0; i < it->n && !c->err; i++) {
          scope_push(c, nd->var, it->it[i]);
          scope_push(c, "loop", make_loop(a, i, it->n));
          run_nodes(c, nd->body, nd->nbody);
          c->vars = saved;
          iterated = 1;
        }
      } else if (it && (it->t == V_DICT || it->t == V_NS)) {
        for (int i = 0; i < it->nkv && !c->err; i++) {
          scope_push(c, nd->var, v_str(a, it->kv[i].k));
          scope_push(c, "loop", make_loop(a, i, it->nkv));
          run_nodes(c, nd->body, nd->nbody);
          c->vars = saved;
          iterated = 1;
        }
      } else if (it && it->t == V_STR && it->s && it->s[0]) {
        int len = (int)strlen(it->s);
        for (int i = 0; i < len && !c->err; i++) {
          scope_push(c, nd->var, v_str(a, a_strndup(a, it->s + i, 1)));
          scope_push(c, "loop", make_loop(a, i, len));
          run_nodes(c, nd->body, nd->nbody);
          c->vars = saved;
          iterated = 1;
        }
      }
      if (!iterated) run_nodes(c, nd->ebody, nd->nebody);
      break;
    }
  }
}

static void run_nodes(rctx *c, jnode **ns, int n) {
  for (int i = 0; i < n && !c->err; i++) run_node(c, ns[i]);
}

/* ============================ 公共 API ============================ */
struct sgl_jinja {
  arena a;
  char *src;
  size_t src_len;
  jnode **nodes;
  int n;
  const char *bos;
  const char *eos;
};

sgl_jinja *sgl_jinja_from_text(const char *tpl, size_t len) {
  sgl_jinja *j = (sgl_jinja *)calloc(1, sizeof(sgl_jinja));
  if (!j) return NULL;
  /* Jinja2 默认 keep_trailing_newline=False：去掉模板末尾的一个换行 */
  if (len >= 2 && tpl[len - 2] == '\r' && tpl[len - 1] == '\n') len -= 2;
  else if (len >= 1 && tpl[len - 1] == '\n') len -= 1;
  j->src = a_strndup(&j->a, tpl, len);
  j->src_len = len;
  mtok *toks = NULL;
  int ntoks = 0;
  if (lex(&j->a, j->src, len, &toks, &ntoks) != 0) {
    SGL_ERROR("jinja: unbalanced {{ }} or {%% %%}");
    sgl_jinja_free(j);
    return NULL;
  }
  tps p = {toks, ntoks, 0, &j->a};
  int nn = 0;
  j->nodes = parse_nodes(&p, NULL, 0, &nn);
  if (!j->nodes && nn < 0) {
    SGL_ERROR("jinja: parse failed");
    sgl_jinja_free(j);
    return NULL;
  }
  j->n = nn;
  if (j->a.oom) {
    sgl_jinja_free(j);
    return NULL;
  }
  return j;
}

sgl_jinja *sgl_jinja_load_dir(const char *dir) {
  char p[1200];
  snprintf(p, sizeof(p), "%s/tokenizer_config.json", dir);
  FILE *f = NULL;
  if (fopen_s(&f, p, "rb") != 0 || !f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n <= 0 || n > (1 << 22)) {
    fclose(f);
    return NULL;
  }
  char *buf = (char *)malloc((size_t)n + 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  size_t got = fread(buf, 1, (size_t)n, f);
  fclose(f);
  buf[got] = 0;
  sgl_json *cfg = sgl_json_parse(buf, got);
  free(buf);
  if (!cfg) return NULL;
  const sgl_json *ct = sgl_json_get(cfg, "chat_template");
  const char *tpl = NULL;
  if (ct && ct->type == SGL_JSON_STRING) {
    tpl = ct->str;
  } else if (ct && ct->type == SGL_JSON_ARRAY && ct->count > 0) {
    const sgl_json *first = ct->items[0];
    const char *t2 = sgl_json_get_str(first, "template", NULL);
    if (!t2) t2 = sgl_json_get_str(first, "chat_template", NULL);
    tpl = t2;
  }
  sgl_jinja *j = NULL;
  if (tpl) {
    j = sgl_jinja_from_text(tpl, strlen(tpl));
    if (j) SGL_INFO("jinja: chat_template loaded (%llu bytes) from %s",
                    (unsigned long long)strlen(tpl), p);
  }
  if (j) {
    /* bos/eos：可能是字符串，也可能是 {"content": "..."} */
    const sgl_json *bt = sgl_json_get(cfg, "bos_token");
    const sgl_json *et = sgl_json_get(cfg, "eos_token");
    const char *bs = NULL, *es = NULL;
    if (bt && bt->type == SGL_JSON_STRING) bs = bt->str;
    else if (bt && bt->type == SGL_JSON_OBJECT) bs = sgl_json_get_str(bt, "content", NULL);
    if (et && et->type == SGL_JSON_STRING) es = et->str;
    else if (et && et->type == SGL_JSON_OBJECT) es = sgl_json_get_str(et, "content", NULL);
    j->bos = a_strdup(&j->a, bs ? bs : "");
    j->eos = a_strdup(&j->a, es ? es : "");
  }
  sgl_json_free(cfg);
  return j;
}

void sgl_jinja_free(sgl_jinja *j) {
  if (!j) return;
  a_reset(&j->a);
  free(j);
}

const char *sgl_jinja_source(const sgl_jinja *j) { return j ? (j->src ? j->src : "") : ""; }
const char *sgl_jinja_bos(const sgl_jinja *j) { return j ? j->bos : NULL; }
const char *sgl_jinja_eos(const sgl_jinja *j) { return j ? j->eos : NULL; }

int sgl_jinja_render(const sgl_jinja *j, const sgl_json *messages, int add_generation_prompt,
                     const sgl_json *kwargs, char **out, size_t *out_len) {
  if (!j || !out) return -1;
  *out = NULL;
  if (out_len) *out_len = 0;
  arena a;
  memset(&a, 0, sizeof(a));
  sb o;
  sb_init(&o);
  rctx c;
  memset(&c, 0, sizeof(c));
  c.a = &a;
  c.out = &o;

  jval *msgs = messages ? json_to_val(&a, messages) : v_list(&a);
  scope_push(&c, "messages", msgs);
  scope_push(&c, "add_generation_prompt", v_bool(&a, add_generation_prompt));
  /* HF tokenizers 常用的其余变量 */
  scope_push(&c, "bos_token",
             v_str(&a, sgl_json_get_str(kwargs, "bos_token", (j->bos && j->bos[0]) ? j->bos : "<s>")));
  scope_push(&c, "eos_token",
             v_str(&a, sgl_json_get_str(kwargs, "eos_token", (j->eos && j->eos[0]) ? j->eos : "</s>")));
  scope_push(&c, "pad_token", v_str(&a, sgl_json_get_str(kwargs, "pad_token", "")));
  if (kwargs && kwargs->type == SGL_JSON_OBJECT)
    for (int i = 0; i < kwargs->count; i++)
      scope_push(&c, kwargs->keys[i], json_to_val(&a, kwargs->items[i]));

  run_nodes(&c, j->nodes, j->n);
  int rc = 0;
  if (c.err || o.oom || a.oom) {
    SGL_ERROR("jinja render failed: %s", c.err ? c.msg : "out of memory");
    rc = -1;
  } else {
    /* 交给调用方的文本用 sgl 分配器（调用方一律 sgl_xfree；内部 sb 是 malloc，
     * 两者在 Windows 上不兼容——_aligned_free/plain malloc 混用会堆损坏）*/
    char *res = (char *)sgl_xmalloc(o.len + 1);
    if (o.len) memcpy(res, o.p, o.len);
    res[o.len] = 0;
    *out = res;
    if (out_len) *out_len = o.len;
  }
  sb_free(&o);
  a_reset(&a);
  return rc;
}