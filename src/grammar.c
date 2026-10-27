/* sgl/grammar.c — 约束解码（constrained decoding）引擎
 *
 * 设计
 * ====
 * 编译期（sgl_grammar_from_*）：
 *   JSON Schema / 正则 → 不可变的字节级节点树（gnode）。节点种类：
 *     N_LIT   字面字节串（"{" / "null" / 枚举字面量 …）
 *     N_CLASS 单字节字符类 + 重复次数区间 [rmin, rmax]（"a-z 重复 2~4 次"）
 *     N_UTF8  一个 UTF-8 字符（ASCII 走 set，0xC2~0xF4 走多字节续字节规则）
 *     N_SEQ   顺序拼接            N_ALT  分支（anyOf / type 数组 / enum）
 *     N_REP   重复 child [rmin, rmax]
 *     N_OBJ   JSON 对象（键名集合 + required 位集 + 每个键的值节点 / 任意键模式）
 *     N_NUM   JSON 数字（整数/小数、符号、整数部分位数、minimum/maximum）
 *     N_ANY   任意 JSON 值（递归节点，运行期展开，编译期不生递归）
 *
 * 运行期：状态是一个「帧栈」（gframe 数组），逗号上方下标大的一端是栈顶。栈顶帧描述
 *   "下一个字节可以是什么"，每消费一个字节就在栈上原地推进；某个构造完成时对应帧弹出，
 *   弹出后若新栈顶是 REP 帧就累计一次重复。因为所有帧都是 POD，复制栈 == 复制状态，
 *   所以「非法字节不改变状态」用 "复制到临时栈、成功才提交" 实现，token 级掩码同样复用这条路径。
 *
 * 关键点：
 *   - REP / CLASS / NUM 是“可以继续也可以结束”的模糊点，这里采用贪心 ——
 *     字节能被当前帧吞下就吞，吞不下且该帧已达最小次数就弹栈交给父节点重试；
 *   - ALT（enum / anyOf / type 数组）若多个分支都能吃下当前字节，按真并集语义
 *     逐个回溯：快照 frames[0..alt_idx]（父层帧也在快照里）→ 试一个分支 → 失败就恢复。
 *     所以 enum:["x","y"] 这类同首字节的分支能正确区分；回溯预算 G_FEED_BUDGET 步。
 *
 * 复杂度：单字节推进 O(栈深)（只有子节点展开/弹出会动栈深）；
 *   token 掩码 = 每个候选 token 一次 O(栈深) 的 memcpy + O(字节数) 推进。
 *   栈深只随 JSON 嵌套层数增长（数组/对象成员复用同一个 REP / OBJ 帧，不随元素个数增长）。
 */
#include "sgl/grammar.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/json.h"

#define G_INF 0x3fffffff          /* 重复次数上界（够大且不会溢出） */
#define G_MAX_DEPTH 40            /* schema 嵌套层数上限 */
#define G_MAX_KEYS 60             /* 单个对象的属性数上限（键位集用 uint64_t） */
#define G_MAX_INT_DIGITS 18       /* 整数部分位数上限 */
#define G_SETTLE_GUARD 200000     /* 病态文法的收敛保护 */
#define G_FEED_BUDGET 400000      /* 单次喂字节（含 ALT 回溯）的步数上限，防指数爆炸 */

/* ---------------- 节点 ---------------- */
enum { N_LIT = 1, N_CLASS, N_UTF8, N_SEQ, N_ALT, N_REP, N_OBJ, N_NUM, N_ANY };

typedef struct gnode gnode;

struct gnode {
  int type;
  /* N_LIT */
  char *lit;
  int lit_len;
  /* N_CLASS / N_UTF8 */
  uint8_t set[32];
  int rmin, rmax;
  int multi; /* N_UTF8：允许 0xC2~0xF4 起始的多字节 UTF-8 */
  /* N_SEQ / N_ALT */
  gnode **kids;
  int nkids;
  /* N_REP */
  gnode *child;
  /* N_OBJ */
  char **keys;
  int *key_lens;
  gnode **vals;
  int nkeys;
  uint64_t reqmask; /* required 键位集 */
  int addl;         /* 1 = additionalProperties 非 false：任意键名 + 任意值 */
  gnode *member_any;
  /* N_NUM */
  int is_int, allow_minus, allow_frac, allow_exp;
  int has_min, has_max;
  double vmin, vmax;
};

/* ---------------- 帧 ---------------- */
enum { FK_LIT = 1, FK_CLASS, FK_UTF8, FK_ALT, FK_REP, FK_OBJ, FK_NUM };

/* FK_OBJ 阶段 */
enum {
  PH_LBRACE = 0,
  PH_KEY_OR_END, /* 对象起始/成员之间：键 或 '}' */
  PH_KEY,        /* 逗号后：必须是键 */
  PH_KEY_MATCH,  /* 正在按键名逐字节匹配 */
  PH_VALUE,      /* 值节点在栈上 */
  PH_SEP,        /* 值之后：',' 或 '}' */
  PH_DONE        /* 收尾 '}' 已消费 */
};

/* FK_NUM 阶段 */
enum { NP_INT = 0, NP_FRAC, NP_EXP_SIGN, NP_EXP_DIGIT };

typedef struct {
  int kind;
  int flag;   /* FK_REP：子节点已展开；FK_NUM：负号 */
  int a, b, c; /* LIT:pos | CLASS:次数 | ALT:分支 | REP:次数 | OBJ:phase | NUM:phase */
  uint64_t used, cand;
  const gnode *n;
  double dv; /* FK_NUM：整数部分绝对值 */
} gframe;

typedef struct {
  gframe *frames;
  int num, cap;
} gstack;

struct sgl_grammar {
  char err[256];
  gnode **arena;
  int narena, caparena;
  gnode *root;
  gnode *colon_lit;
  gnode *any_node; /* N_ANY */
  gnode *any_alt;  /* N_ANY 的展开（N_ALT） */
  gstack st;
  const char *const *pieces;
  const int *plens;
  int npieces;
};

/* 工厂函数失败时（还没有文法对象）把原因留在这里，sgl_grammar_error(NULL) 可取回。
 * 注意：进程级、非线程安全（只在创建阶段用）。 */
static char g_last_err[256];

static void set_last_err(const char *msg) {
  snprintf(g_last_err, sizeof(g_last_err), "%s", msg ? msg : "");
}

static void g_err(sgl_grammar *g, const char *msg) {
  set_last_err(msg);
  if (g) snprintf(g->err, sizeof(g->err), "%s", msg ? msg : "");
}

/* ---------------- 字节集合 ---------------- */
static void set_clear(uint8_t m[32]) { memset(m, 0, 32); }
static void set_add(uint8_t m[32], int b) {
  if (b >= 0 && b < 256) m[b >> 3] |= (uint8_t)(1u << (b & 7));
}
static int set_has(const uint8_t m[32], int b) {
  if (b < 0 || b > 255) return 0;
  return (m[b >> 3] >> (b & 7)) & 1;
}
static void set_or(uint8_t d[32], const uint8_t s[32]) {
  for (int i = 0; i < 32; i++) d[i] = (uint8_t)(d[i] | s[i]);
}
static void set_range(uint8_t m[32], int lo, int hi) {
  for (int b = lo; b <= hi; b++) set_add(m, b);
}
static int set_count(const uint8_t m[32]) {
  int c = 0;
  for (int i = 0; i < 32; i++)
    for (int k = 0; k < 8; k++) c += (m[i] >> k) & 1;
  return c;
}
static int set_first(const uint8_t m[32]) {
  for (int i = 0; i < 32; i++)
    for (int k = 0; k < 8; k++)
      if ((m[i] >> k) & 1) return i * 8 + k;
  return -1;
}

/* ---------------- 节点池 ---------------- */
static gnode *nnew(sgl_grammar *g, int type) {
  gnode *n = (gnode *)calloc(1, sizeof(gnode));
  if (!n) {
    g_err(g, "grammar: out of memory");
    return NULL;
  }
  n->type = type;
  n->rmax = G_INF;
  if (g->narena == g->caparena) {
    int cap = g->caparena ? g->caparena * 2 : 32;
    gnode **a = (gnode **)realloc(g->arena, (size_t)cap * sizeof(gnode *));
    if (!a) {
      free(n);
      g_err(g, "grammar: out of memory");
      return NULL;
    }
    g->arena = a;
    g->caparena = cap;
  }
  g->arena[g->narena++] = n;
  return n;
}

static gnode *n_lit(sgl_grammar *g, const char *s, int len) {
  gnode *n = nnew(g, N_LIT);
  if (!n) return NULL;
  if (len < 0) len = s ? (int)strlen(s) : 0;
  n->lit = (char *)malloc((size_t)len + 1);
  if (!n->lit) {
    g_err(g, "grammar: out of memory");
    return NULL;
  }
  if (len && s) memcpy(n->lit, s, (size_t)len);
  n->lit[len] = 0;
  n->lit_len = len;
  return n;
}

static gnode *n_class(sgl_grammar *g, const uint8_t set[32], int rmin, int rmax) {
  gnode *n = nnew(g, N_CLASS);
  if (!n) return NULL;
  memcpy(n->set, set, 32);
  n->rmin = rmin;
  n->rmax = rmax;
  return n;
}

static gnode *n_seq(sgl_grammar *g, const gnode *const *kids, int nk) {
  gnode *n = nnew(g, N_SEQ);
  if (!n) return NULL;
  if (nk <= 0) return n;
  n->kids = (gnode **)malloc((size_t)nk * sizeof(gnode *));
  if (!n->kids) {
    g_err(g, "grammar: out of memory");
    return NULL;
  }
  for (int i = 0; i < nk; i++) n->kids[i] = (gnode *)kids[i];
  n->nkids = nk;
  return n;
}

static gnode *n_alt(sgl_grammar *g, const gnode *const *kids, int nk) {
  gnode *n = n_seq(g, kids, nk);
  if (n) n->type = N_ALT;
  return n;
}

static gnode *n_rep(sgl_grammar *g, const gnode *child, int rmin, int rmax) {
  gnode *n = nnew(g, N_REP);
  if (!n) return NULL;
  n->child = (gnode *)child;
  n->rmin = rmin;
  n->rmax = rmax;
  return n;
}

static void node_free(gnode *n) {
  if (!n) return;
  free(n->lit);
  free(n->kids);
  if (n->keys) {
    for (int i = 0; i < n->nkeys; i++) free(n->keys[i]);
    free(n->keys);
  }
  free(n->key_lens);
  free(n->vals);
  free(n);
}

/* ---------------- 帧栈 ---------------- */
static int st_reserve(gstack *s, int extra) {
  if (s->num + extra <= s->cap) return 0;
  int cap = s->cap ? s->cap * 2 : 64;
  while (cap < s->num + extra) cap *= 2;
  gframe *f = (gframe *)realloc(s->frames, (size_t)cap * sizeof(gframe));
  if (!f) return -1;
  s->frames = f;
  s->cap = cap;
  return 0;
}

static gframe *st_push_raw(gstack *s) {
  if (st_reserve(s, 1)) return NULL;
  gframe *f = &s->frames[s->num++];
  memset(f, 0, sizeof(*f));
  return f;
}

static void pop_top(gstack *s) {
  if (s->num <= 0) return;
  s->num--;
  if (s->num <= 0) return;
  gframe *p = &s->frames[s->num - 1];
  if (p->kind == FK_REP && p->flag) {
    p->a++; /* 一次重复结束 */
    p->flag = 0;
  } else if (p->kind == FK_OBJ && p->a == PH_VALUE) {
    p->a = PH_SEP; /* 值子树全部出栈 */
  }
}

/* 前向声明（节点展开与掩码互相递归） */
static int push_node(const sgl_grammar *g, gstack *s, const gnode *n);
static void mask_node_first(const sgl_grammar *g, const gnode *n, uint8_t m[32]);

/* 把节点展开成帧；N_SEQ 一次全部压入（逆序，使首个孩子位于栈顶）。 */
static int push_node(const sgl_grammar *g, gstack *s, const gnode *n) {
  if (!n) return -1;
  switch (n->type) {
  case N_LIT: {
    gframe *f = st_push_raw(s);
    if (!f) return -1;
    f->kind = FK_LIT;
    f->n = n;
    return 0;
  }
  case N_CLASS: {
    gframe *f = st_push_raw(s);
    if (!f) return -1;
    f->kind = FK_CLASS;
    f->n = n;
    return 0;
  }
  case N_UTF8: {
    gframe *f = st_push_raw(s);
    if (!f) return -1;
    f->kind = FK_UTF8;
    f->n = n;
    f->a = -1; /* 尚未见到起始字节 */
    return 0;
  }
  case N_ALT: {
    gframe *f = st_push_raw(s);
    if (!f) return -1;
    f->kind = FK_ALT;
    f->n = n;
    f->a = -1; /* 尚未选分支 */
    return 0;
  }
  case N_REP: {
    gframe *f = st_push_raw(s);
    if (!f) return -1;
    f->kind = FK_REP;
    f->n = n;
    return 0;
  }
  case N_OBJ: {
    gframe *f = st_push_raw(s);
    if (!f) return -1;
    f->kind = FK_OBJ;
    f->n = n;
    f->a = PH_LBRACE;
    return 0;
  }
  case N_NUM: {
    gframe *f = st_push_raw(s);
    if (!f) return -1;
    f->kind = FK_NUM;
    f->n = n;
    return 0;
  }
  case N_ANY:
    return g->any_alt ? push_node(g, s, g->any_alt) : -1;
  case N_SEQ:
    for (int i = n->nkids - 1; i >= 0; i--)
      if (push_node(g, s, n->kids[i])) return -1;
    return 0;
  default:
    return -1;
  }
}

/* 把所有"已经结束但还留在栈上"的帧弹掉，并补上 REP 的强制重复。 */
static void settle(const sgl_grammar *g, gstack *s) {
  int guard = 0;
  for (;;) {
    if (s->num <= 0) return;
    if (++guard > G_SETTLE_GUARD) return;
    gframe *f = &s->frames[s->num - 1];
    const gnode *n = f->n;
    switch (f->kind) {
    case FK_LIT:
      if (f->a >= n->lit_len) {
        pop_top(s);
        continue;
      }
      return;
    case FK_CLASS:
      if (f->a >= n->rmax) {
        pop_top(s);
        continue;
      }
      return;
    case FK_UTF8:
      if (f->a == 0) {
        pop_top(s);
        continue;
      }
      return;
    case FK_ALT:
      if (f->a >= 0) { /* 分支已跑完 */
        pop_top(s);
        continue;
      }
      return;
    case FK_REP:
      if (f->flag) return; /* 子节点还在栈上 */
      if (f->a >= n->rmax) {
        pop_top(s);
        continue;
      }
      if (f->a < n->rmin) { /* 强制再重复一次 */
        f->flag = 1;
        if (push_node(g, s, n->child)) return;
        continue;
      }
      return;
    case FK_OBJ:
      if (f->a == PH_DONE) {
        pop_top(s);
        continue;
      }
      return;
    default:
      return;
    }
  }
}

/* ---------------- 首字节集合 / 接受态判定 ---------------- */
static void mask_node_first(const sgl_grammar *g, const gnode *n, uint8_t m[32]) {
  if (!n) return;
  switch (n->type) {
  case N_LIT:
    if (n->lit_len > 0) set_add(m, (uint8_t)n->lit[0]);
    return;
  case N_CLASS:
    if (n->rmax > 0) set_or(m, n->set);
    return;
  case N_UTF8:
    set_or(m, n->set);
    if (n->multi) set_range(m, 0xC2, 0xF4);
    return;
  case N_SEQ:
    /* 顺序拼接：第一个能产生字节的孩子决定首字节；空孩子（如空 LIT）跳过 */
    for (int i = 0; i < n->nkids; i++) {
      uint8_t t[32];
      set_clear(t);
      mask_node_first(g, n->kids[i], t);
      if (set_count(t) > 0) {
        set_or(m, t);
        return;
      }
    }
    return;
  case N_ALT:
    for (int i = 0; i < n->nkids; i++) mask_node_first(g, n->kids[i], m);
    return;
  case N_REP:
    if (n->rmax > 0) mask_node_first(g, n->child, m);
    return;
  case N_OBJ:
    set_add(m, '{');
    return;
  case N_NUM:
    if (n->allow_minus) set_add(m, '-');
    set_range(m, '0', '9');
    return;
  case N_ANY:
    if (g->any_alt) mask_node_first(g, g->any_alt, m);
    return;
  default:
    return;
  }
}

static int finish_frame(const sgl_grammar *g, const gstack *s, int depth);

/* ---------------- 数字约束 ---------------- */
static int num_stop_ok(const gnode *n, double v, int neg) {
  if (!n->has_min && !n->has_max) return 1;
  double x = neg ? -v : v;
  if (n->has_min && x < n->vmin) return 0;
  if (n->has_max && x > n->vmax) return 0;
  return 1;
}

/* 整数部分前缀 v（位数 k）是否还能通过继续补数字落进 [min,max]。
 * 整数：补 j 位得到 [v*10^j, v*10^j + 10^j - 1]；小数：区间放宽一位（允许小数部分）。 */
static int num_prefix_ok(const gnode *n, double v, int k, int neg) {
  if (!n->has_min && !n->has_max) return 1;
  if (k <= 0) return 1;
  double lo = n->has_min ? n->vmin : -1e300;
  double hi = n->has_max ? n->vmax : 1e300;
  double p = 1.0;
  for (int j = 0; j < 19; j++) {
    double a = v * p;
    double b = a + (n->is_int ? (p - 1.0) : p);
    double slog = neg ? -b : a;
    double shig = neg ? -a : b;
    if (shig >= lo && slog <= hi) return 1;
    if (k == 1 && v == 0) break; /* "0" 之后再接数字是前导零，非法 */
    p *= 10.0;
  }
  return 0;
}

static int num_consume(gframe *f, uint8_t b) {
  const gnode *n = f->n;
  switch (f->a) {
  case NP_INT:
    if (f->b == 0 && !f->c && b == '-' && n->allow_minus) {
      f->flag = 1;
      f->c = 1;
      return 1; /* TC_CONSUMED */
    }
    if (b >= '0' && b <= '9') {
      if (f->b >= G_MAX_INT_DIGITS) return -1;
      if (f->b == 1 && f->dv == 0) return -1; /* 前导零 */
      double nv = f->dv * 10.0 + (double)(b - '0');
      if (!num_prefix_ok(n, nv, f->b + 1, f->flag)) return -1;
      f->dv = nv;
      f->b++;
      return 1;
    }
    if (f->b >= 1 && n->allow_frac && b == '.' && num_prefix_ok(n, f->dv, f->b, f->flag)) {
      f->a = NP_FRAC;
      f->c = 0;
      return 1;
    }
    if (f->b >= 1 && n->allow_exp && (b == 'e' || b == 'E') &&
        num_prefix_ok(n, f->dv, f->b, f->flag)) {
      f->a = NP_EXP_SIGN;
      f->c = 0;
      return 1;
    }
    if (f->b >= 1 && num_stop_ok(n, f->dv, f->flag)) return 0; /* TC_COMPLETE：数字到此为止 */
    return -1;
  case NP_FRAC:
    if (b >= '0' && b <= '9') {
      f->c++;
      if (f->c > 30) return -1;
      return 1;
    }
    if (f->c >= 1) {
      if (n->allow_exp && (b == 'e' || b == 'E')) {
        f->a = NP_EXP_SIGN;
        f->c = 0;
        return 1;
      }
      return 0;
    }
    return -1;
  case NP_EXP_SIGN:
    if (b == '+' || b == '-') {
      f->a = NP_EXP_DIGIT;
      f->c = 0;
      return 1;
    }
    if (b >= '0' && b <= '9') {
      f->a = NP_EXP_DIGIT;
      f->c = 1;
      return 1;
    }
    return -1;
  case NP_EXP_DIGIT:
    if (b >= '0' && b <= '9') {
      f->c++;
      if (f->c > 8) return -1;
      return 1;
    }
    return (f->c >= 1) ? 0 : -1;
  default:
    return -1;
  }
}

/* ---------------- 对象 ---------------- */
static uint64_t obj_all_keys(const gnode *n) {
  if (n->nkeys <= 0) return 0;
  if (n->nkeys >= 64) return ~0ull;
  return (1ull << n->nkeys) - 1;
}
static uint64_t obj_avail(const gframe *f, const gnode *n) {
  return obj_all_keys(n) & ~f->used;
}
static int obj_req_ok(const gframe *f, const gnode *n) {
  return (f->used & n->reqmask) == n->reqmask;
}

/* 返回 1=吃掉字节 0=本帧已完成（可弹栈）-1=非法 2=已改状态但没吃字节（重试） */
static int obj_consume(const sgl_grammar *g, gstack *s, gframe *f, uint8_t b) {
  const gnode *n = f->n;
  switch (f->a) {
  case PH_LBRACE:
    if (b != '{') return -1;
    f->a = PH_KEY_OR_END;
    return 1;

  case PH_KEY_OR_END:
  case PH_KEY: {
    if (b == '}' && f->a == PH_KEY_OR_END && obj_req_ok(f, n)) {
      f->a = PH_DONE;
      return 1;
    }
    if (n->addl) { /* 任意键名 */
      if (b != '"') return -1;
      f->a = PH_VALUE; /* 成员子树出栈后由 pop_top 翻成 PH_SEP */
      if (push_node(g, s, n->member_any)) return -1;
      return 2;
    }
    uint64_t avail = obj_avail(f, n);
    if (b != '"' || avail == 0) return -1;
    uint64_t cand = 0;
    for (int i = 0; i < n->nkeys; i++)
      if ((avail >> i) & 1)
        if (n->key_lens[i] > 0 && (uint8_t)n->keys[i][0] == b) cand |= (1ull << i);
    if (!cand) return -1;
    f->cand = cand;
    f->b = 1;
    f->a = PH_KEY_MATCH;
    return 1;
  }

  case PH_KEY_MATCH: {
    if (b == '"') { /* 键名收尾：挑出"下一个字节就是收尾引号"的候选 */
      int chosen = -1;
      for (int i = 0; i < n->nkeys; i++)
        if (((f->cand >> i) & 1) && n->key_lens[i] == f->b + 1) {
          chosen = i;
          break;
        }
      if (chosen < 0) return -1;
      f->used |= (1ull << chosen);
      f->a = PH_VALUE;
      /* 栈顶先执行：先压值、再压 ':'，使 ':' 位于栈顶 */
      if (push_node(g, s, n->vals[chosen])) return -1;
      if (push_node(g, s, g->colon_lit)) return -1;
      return 1; /* 收尾引号属于键名字节，已消费 */
    }
    uint64_t cand = 0;
    for (int i = 0; i < n->nkeys; i++)
      if (((f->cand >> i) & 1) && f->b < n->key_lens[i] && (uint8_t)n->keys[i][f->b] == b)
        cand |= (1ull << i);
    if (!cand) return -1;
    f->cand = cand;
    f->b++;
    return 1;
  }

  case PH_SEP:
    if (b == ',') {
      if (!n->addl && obj_avail(f, n) == 0) return -1; /* 没有剩余键可填 */
      f->a = PH_KEY;
      f->b = 0;
      return 1;
    }
    if (b == '}' && obj_req_ok(f, n)) {
      f->a = PH_DONE;
      return 1;
    }
    return -1;

  case PH_DONE:
    return 0;

  default: /* PH_VALUE：值子树在栈上，本帧不该是栈顶 */
    return -1;
  }
}

/* ---------------- 字节推进（ALT 分支带回溯） ---------------- */
/* 把 bytes[pos..len) 依次喂进栈，0=全部合法。
 * 栈顶帧的处理结果 rc：1=吃掉一个字节 0=本帧完成（可弹栈）-1=非法 2=状态已变、没吃字节、重试。
 * FK_ALT 若多个分支都能吃下当前字节，按真并集语义逐个回溯（这是 enum / anyOf 正确性的关键）。 */
static int feed_rec(const sgl_grammar *g, gstack *s, const char *bytes, int pos, int len,
                    int *budget) {
  for (;;) {
    if (pos >= len) return 0;
    if (--(*budget) < 0) return -1;
    if (s->num <= 0) return -1;
    int b = (uint8_t)bytes[pos];
    gframe *f = &s->frames[s->num - 1];
    const gnode *n = f->n;
    int rc;
    switch (f->kind) {
    case FK_LIT:
      if (f->a >= n->lit_len) {
        rc = 0;
      } else if ((uint8_t)n->lit[f->a] != b) {
        return -1;
      } else {
        f->a++;
        rc = 1;
      }
      break;
    case FK_CLASS:
      if (f->a < n->rmax && set_has(n->set, b)) {
        f->a++;
        rc = 1;
      } else {
        rc = (f->a >= n->rmin) ? 0 : -1;
      }
      break;
    case FK_UTF8:
      if (f->a < 0) {
        if (b < 0x80) {
          if (!set_has(n->set, b)) return -1;
          f->a = 0;
        } else {
          if (!n->multi || b < 0xC2 || b > 0xF4) return -1;
          f->a = (b < 0xE0) ? 1 : ((b < 0xF0) ? 2 : 3);
        }
        rc = 1;
      } else if (f->a == 0) {
        rc = 0;
      } else {
        if ((b & 0xC0) != 0x80) return -1;
        f->a--;
        rc = 1;
      }
      break;
    case FK_ALT: {
      if (f->a >= 0) { /* 分支已跑完 */
        rc = 0;
        break;
      }
      int nc = 0, first = -1;
      for (int i = 0; i < n->nkids; i++) {
        uint8_t m[32];
        set_clear(m);
        mask_node_first(g, n->kids[i], m);
        if (set_has(m, b)) {
          nc++;
          if (first < 0) first = i;
        }
      }
      if (nc == 0) return -1;
      int alt_idx = s->num - 1;
      if (nc == 1) { /* 唯一候选：不需要快照 */
        s->frames[alt_idx].a = first;
        if (push_node(g, s, n->kids[first])) return -1;
        rc = 2;
        break;
      }
      /* 多候选：收集候选 → 逐个回溯（先快照 frames[0..alt_idx]，父层帧也在猫像里） */
      int *cand = (int *)malloc((size_t)nc * sizeof(int));
      gframe *snap = (gframe *)malloc((size_t)(alt_idx + 1) * sizeof(gframe));
      if (!cand || !snap) {
        free(cand);
        free(snap);
        return -1;
      }
      int j = 0;
      for (int i = 0; i < n->nkids; i++) {
        uint8_t m[32];
        set_clear(m);
        mask_node_first(g, n->kids[i], m);
        if (set_has(m, b)) cand[j++] = i;
      }
      memcpy(snap, s->frames, (size_t)(alt_idx + 1) * sizeof(gframe));
      for (int k = 0; k < nc; k++) {
        s->frames[alt_idx].a = cand[k];
        if (push_node(g, s, n->kids[cand[k]])) break;
        if (feed_rec(g, s, bytes, pos, len, budget) == 0) {
          free(cand);
          free(snap);
          return 0;
        }
        memcpy(s->frames, snap, (size_t)(alt_idx + 1) * sizeof(gframe));
        s->num = alt_idx + 1;
      }
      free(cand);
      free(snap);
      return -1;
    }
    case FK_REP:
      if (f->flag) return -1; /* 子节点在栈上时本帧不可能是栈顶 */
      if (f->a < n->rmax) {
        uint8_t m[32];
        set_clear(m);
        mask_node_first(g, n->child, m);
        if (set_has(m, b)) {
          f->flag = 1;
          if (push_node(g, s, n->child)) return -1;
          rc = 2;
          break;
        }
      }
      rc = (f->a >= n->rmin) ? 0 : -1;
      break;
    case FK_OBJ:
      rc = obj_consume(g, s, f, b);
      break;
    case FK_NUM:
      rc = num_consume(f, b);
      break;
    default:
      return -1;
    }
    if (rc < 0) return -1;
    if (rc == 1) { /* 吃掉了一个字节 */
      settle(g, s);
      pos++;
      continue;
    }
    if (rc == 0) { /* 本帧结束 → 弹栈后重试同一个字节 */
      pop_top(s);
      continue;
    }
    /* rc == 2：状态已变但没吃字节，继续重试 */
  }
}

static int gs_accept_bytes(const sgl_grammar *g, gstack *s, const char *bytes, int len) {
  int budget = G_FEED_BUDGET;
  return feed_rec(g, s, bytes, 0, len, &budget);
}

static int finish_frame(const sgl_grammar *g, const gstack *s, int depth) {
  if (depth < 0) return 1;
  const gframe *f = &s->frames[depth];
  const gnode *n = f->n;
  switch (f->kind) {
  case FK_LIT:
    return (f->a >= n->lit_len) ? finish_frame(g, s, depth - 1) : 0;
  case FK_CLASS:
    return (f->a >= n->rmin) ? finish_frame(g, s, depth - 1) : 0;
  case FK_UTF8:
    return (f->a == 0) ? finish_frame(g, s, depth - 1) : 0;
  case FK_ALT:
    return (f->a >= 0) ? finish_frame(g, s, depth - 1) : 0;
  case FK_REP:
    return (!f->flag && f->a >= n->rmin) ? finish_frame(g, s, depth - 1) : 0;
  case FK_OBJ:
    return (f->a == PH_DONE) ? finish_frame(g, s, depth - 1) : 0;
  case FK_NUM:
    if (f->a == NP_INT) return (f->b >= 1 && num_stop_ok(n, f->dv, f->flag)) ? finish_frame(g, s, depth - 1) : 0;
    if (f->a == NP_FRAC) return (f->c >= 1) ? finish_frame(g, s, depth - 1) : 0;
    if (f->a == NP_EXP_DIGIT) return (f->c >= 1) ? finish_frame(g, s, depth - 1) : 0;
    return 0;
  default:
    return 0;
  }
}

/* ---------------- 字符串缓冲（渲染 JSON 字面量用） ---------------- */
typedef struct {
  char *p;
  size_t n, cap;
} sbuf;

static int sb_reserve(sbuf *b, size_t extra) {
  if (b->n + extra + 1 <= b->cap) return 0;
  size_t cap = b->cap ? b->cap * 2 : 64;
  while (cap < b->n + extra + 1) cap *= 2;
  char *p = (char *)realloc(b->p, cap);
  if (!p) return -1;
  b->p = p;
  b->cap = cap;
  return 0;
}
static int sb_putc(sbuf *b, char c) {
  if (sb_reserve(b, 1)) return -1;
  b->p[b->n++] = c;
  b->p[b->n] = 0;
  return 0;
}
static int sb_put(sbuf *b, const char *s, size_t l) {
  if (sb_reserve(b, l)) return -1;
  memcpy(b->p + b->n, s, l);
  b->n += l;
  b->p[b->n] = 0;
  return 0;
}
static void sb_free(sbuf *b) {
  free(b->p);
  b->p = NULL;
  b->n = b->cap = 0;
}

static int sb_quoted(sbuf *b, const char *s, size_t len) {
  if (sb_putc(b, '"')) return -1;
  for (size_t i = 0; i < len; i++) {
    unsigned char c = (unsigned char)s[i];
    switch (c) {
    case '"': if (sb_put(b, "\\\"", 2)) return -1; break;
    case '\\': if (sb_put(b, "\\\\", 2)) return -1; break;
    case '\b': if (sb_put(b, "\\b", 2)) return -1; break;
    case '\f': if (sb_put(b, "\\f", 2)) return -1; break;
    case '\n': if (sb_put(b, "\\n", 2)) return -1; break;
    case '\r': if (sb_put(b, "\\r", 2)) return -1; break;
    case '\t': if (sb_put(b, "\\t", 2)) return -1; break;
    default:
      if (c < 0x20) {
        char tmp[7];
        snprintf(tmp, sizeof(tmp), "\\u%04x", (unsigned)c);
        if (sb_put(b, tmp, 6)) return -1;
      } else if (sb_putc(b, (char)c)) {
        return -1;
      }
    }
  }
  return sb_putc(b, '"');
}

static int sb_render_json(sbuf *b, const sgl_json *j) {
  if (!j) return -1;
  char tmp[64];
  switch (j->type) {
  case SGL_JSON_NULL: return sb_put(b, "null", 4);
  case SGL_JSON_BOOL: return sb_put(b, j->boolean ? "true" : "false", j->boolean ? 4 : 5);
  case SGL_JSON_NUMBER: {
    int n;
    if (j->num == floor(j->num) && fabs(j->num) < 1e15)
      n = snprintf(tmp, sizeof(tmp), "%.0f", j->num);
    else
      n = snprintf(tmp, sizeof(tmp), "%.17g", j->num);
    if (n < 0) return -1;
    return sb_put(b, tmp, (size_t)n);
  }
  case SGL_JSON_STRING: return sb_quoted(b, j->str ? j->str : "", j->str ? strlen(j->str) : 0);
  case SGL_JSON_ARRAY:
    if (sb_putc(b, '[')) return -1;
    for (int i = 0; i < j->count; i++) {
      if (i && sb_putc(b, ',')) return -1;
      if (sb_render_json(b, j->items[i])) return -1;
    }
    return sb_putc(b, ']');
  case SGL_JSON_OBJECT:
    if (sb_putc(b, '{')) return -1;
    for (int i = 0; i < j->count; i++) {
      if (i && sb_putc(b, ',')) return -1;
      if (sb_quoted(b, j->keys[i] ? j->keys[i] : "", j->keys[i] ? strlen(j->keys[i]) : 0)) return -1;
      if (sb_putc(b, ':')) return -1;
      if (sb_render_json(b, j->items[i])) return -1;
    }
    return sb_putc(b, '}');
  default:
    return -1;
  }
}

static gnode *lit_from_json(sgl_grammar *g, const sgl_json *j) {
  sbuf b;
  memset(&b, 0, sizeof(b));
  if (sb_render_json(&b, j)) {
    sb_free(&b);
    g_err(g, "grammar: cannot render enum/const value");
    return NULL;
  }
  gnode *n = n_lit(g, b.p ? b.p : "", (int)b.n);
  sb_free(&b);
  return n;
}

static gnode *quoted_key(sgl_grammar *g, const char *s) {
  sbuf b;
  memset(&b, 0, sizeof(b));
  if (sb_quoted(&b, s ? s : "", s ? strlen(s) : 0)) {
    sb_free(&b);
    g_err(g, "grammar: out of memory");
    return NULL;
  }
  gnode *n = n_lit(g, b.p ? b.p : "", (int)b.n);
  sb_free(&b);
  return n;
}

/* ---------------- 正则编译器 ---------------- */
typedef struct {
  sgl_grammar *g;
  const char *p;
  int len;
  int i;
  int ok;
} grex;

typedef struct {
  gnode **v;
  int n, cap;
} nlist;

static int nl_push(nlist *l, gnode *n) {
  if (l->n >= l->cap) {
    int cap = l->cap ? l->cap * 2 : 8;
    gnode **v = (gnode **)realloc(l->v, (size_t)cap * sizeof(gnode *));
    if (!v) return -1;
    l->v = v;
    l->cap = cap;
  }
  l->v[l->n++] = n;
  return 0;
}

static int node_bit(const gnode *n, int *c) {
  if (n->type == N_LIT && n->lit_len == 1) {
    *c = (uint8_t)n->lit[0];
    return 1;
  }
  if (n->type == N_CLASS && n->rmin == 1 && n->rmax == 1 && set_count(n->set) == 1) {
    *c = set_first(n->set);
    return 1;
  }
  return 0;
}

/* 合并相邻的单字节原子成 N_LIT（减少帧数，也让键名/结构字符的匹配快一拍） */
static gnode *seq_build(sgl_grammar *g, nlist *l) {
  if (l->n == 0) {
    free(l->v);
    l->v = NULL;
    l->n = l->cap = 0;
    return n_lit(g, "", 0);
  }
  nlist out;
  memset(&out, 0, sizeof(out));
  for (int i = 0; i < l->n; i++) {
    gnode *a = l->v[i];
    int c;
    if (node_bit(a, &c)) {
      gnode *last = out.n ? out.v[out.n - 1] : NULL;
      if (last && last->type == N_LIT) {
        char *s = (char *)realloc(last->lit, (size_t)last->lit_len + 2);
        if (!s) {
          g_err(g, "grammar: out of memory");
          return NULL;
        }
        last->lit = s;
        last->lit[last->lit_len] = (char)c;
        last->lit_len++;
        last->lit[last->lit_len] = 0;
      } else {
        char buf[2];
        buf[0] = (char)c;
        buf[1] = 0;
        if (nl_push(&out, n_lit(g, buf, 1))) {
          g_err(g, "grammar: out of memory");
          return NULL;
        }
      }
    } else if (nl_push(&out, a)) {
      g_err(g, "grammar: out of memory");
      return NULL;
    }
  }
  free(l->v);
  l->v = NULL;
  l->n = l->cap = 0;
  if (out.n == 1) {
    gnode *r = out.v[0];
    free(out.v);
    return r;
  }
  gnode *r = n_seq(g, (const gnode *const *)out.v, out.n);
  free(out.v);
  return r;
}

static gnode *rx_alt(grex *r);

static gnode *rx_char(sgl_grammar *g, int c) {
  char buf[2];
  buf[0] = (char)c;
  buf[1] = 0;
  return n_lit(g, buf, 1);
}

static gnode *rx_set_node(grex *r, const uint8_t set[32]) {
  return n_class(r->g, set, 1, 1);
}

/* 读一个"单元"：普通字符返回 0 并给 *ch；字符类转义（\d 等）返回 1 并填 set。 */
static int rx_unit(grex *r, int *ch, uint8_t set[32]) {
  if (r->i >= r->len) {
    r->ok = 0;
    return -1;
  }
  int c = (uint8_t)r->p[r->i];
  if (c != '\\') {
    r->i++;
    *ch = c;
    return 0;
  }
  r->i++;
  if (r->i >= r->len) {
    r->ok = 0;
    return -1;
  }
  int e = (uint8_t)r->p[r->i++];
  switch (e) {
  case 'd': set_clear(set); set_range(set, '0', '9'); return 1;
  case 'D': set_clear(set); set_range(set, 0, 255); set_range(set, '0', '9'); goto negate;
  case 'w':
    set_clear(set);
    set_range(set, 'a', 'z');
    set_range(set, 'A', 'Z');
    set_range(set, '0', '9');
    set_add(set, '_');
    return 1;
  case 'W':
    set_clear(set);
    set_range(set, 0, 255);
    set_range(set, 'a', 'z');
    set_range(set, 'A', 'Z');
    set_range(set, '0', '9');
    set_add(set, '_');
    goto negate;
  case 's':
    set_clear(set);
    set_add(set, ' ');
    set_add(set, '\t');
    set_add(set, '\n');
    set_add(set, '\r');
    set_add(set, '\f');
    set_add(set, '\v');
    return 1;
  case 'S':
    set_clear(set);
    set_range(set, 0, 255);
    set_add(set, ' ');
    set_add(set, '\t');
    set_add(set, '\n');
    set_add(set, '\r');
    set_add(set, '\f');
    set_add(set, '\v');
    goto negate;
  case 'n': *ch = '\n'; return 0;
  case 'r': *ch = '\r'; return 0;
  case 't': *ch = '\t'; return 0;
  case 'f': *ch = '\f'; return 0;
  case 'v': *ch = '\v'; return 0;
  case '0': *ch = 0; return 0;
  default: *ch = e; return 0;
  }
negate:
  for (int k = 0; k < 32; k++) set[k] = (uint8_t)~set[k];
  return 1;
}

static gnode *rx_class(grex *r) {
  sgl_grammar *g = r->g;
  r->i++; /* '[' */
  uint8_t set[32];
  set_clear(set);
  int neg = 0, first = 1, closed = 0;
  if (r->i < r->len && r->p[r->i] == '^') {
    neg = 1;
    r->i++;
  }
  while (r->i < r->len) {
    if (r->p[r->i] == ']' && !first) {
      r->i++;
      closed = 1;
      break;
    }
    first = 0;
    int ch = 0;
    uint8_t tmp[32];
    set_clear(tmp);
    int rc = rx_unit(r, &ch, tmp);
    if (rc < 0) return NULL;
    if (rc == 1) { /* 字符类转义：直接并入 */
      set_or(set, tmp);
      continue;
    }
    if (r->i + 1 < r->len && r->p[r->i] == '-' && r->p[r->i + 1] != ']') {
      r->i++;
      int hi = 0;
      uint8_t t2[32];
      set_clear(t2);
      if (rx_unit(r, &hi, t2) != 0) {
        r->ok = 0;
        return NULL;
      }
      for (int b = ch; b <= hi; b++) set_add(set, b);
    } else {
      set_add(set, ch);
    }
  }
  if (!closed) {
    r->ok = 0;
    g_err(g, "grammar: unterminated '[' in regex");
    return NULL;
  }
  if (neg)
    for (int k = 0; k < 32; k++) set[k] = (uint8_t)(~set[k] & 0xFF);
  return rx_set_node(r, set);
}

static gnode *rx_atom(grex *r) {
  sgl_grammar *g = r->g;
  if (r->i >= r->len) return n_lit(g, "", 0);
  int c = (uint8_t)r->p[r->i];
  if (c == '(') {
    r->i++;
    gnode *x = rx_alt(r);
    if (r->i < r->len && r->p[r->i] == ')')
      r->i++;
    else
      r->ok = 0;
    return x;
  }
  if (c == '[') return rx_class(r);
  if (c == '.') {
    uint8_t set[32];
    set_clear(set);
    set_range(set, 0, 255);
    set['\n' >> 3] &= (uint8_t)~(1u << ('\n' & 7)); /* . 不匹配换行 */
    r->i++;
    return rx_set_node(r, set);
  }
  if (c == '^' || c == '$') { /* 锚点：整串语义下忽略 */
    r->i++;
    return n_lit(g, "", 0);
  }
  if (c == '\\') {
    int ch = 0;
    uint8_t set[32];
    set_clear(set);
    int rc = rx_unit(r, &ch, set);
    if (rc < 0) return NULL;
    if (rc == 1) return rx_set_node(r, set);
    return rx_char(g, ch);
  }
  r->i++;
  return rx_char(g, c);
}

static gnode *rx_repeat(grex *r) {
  gnode *a = rx_atom(r);
  if (!a || !r->ok) return a;
  int mn = 1, mx = 1, has = 0;
  while (r->i < r->len) {
    int c = (uint8_t)r->p[r->i];
    if (c == '*') {
      mn = 0;
      mx = G_INF;
      has = 1;
      r->i++;
    } else if (c == '+') {
      mn = 1;
      mx = G_INF;
      has = 1;
      r->i++;
    } else if (c == '?') {
      mn = 0;
      mx = 1;
      has = 1;
      r->i++;
    } else if (c == '{') {
      int j = r->i + 1, lo = 0, hi = -1, got = 0;
      while (j < r->len && r->p[j] >= '0' && r->p[j] <= '9') {
        lo = lo * 10 + (r->p[j] - '0');
        j++;
        got = 1;
      }
      if (!got) break;
      if (j < r->len && r->p[j] == ',') {
        j++;
        if (j < r->len && r->p[j] >= '0' && r->p[j] <= '9') {
          hi = 0;
          while (j < r->len && r->p[j] >= '0' && r->p[j] <= '9') {
            hi = hi * 10 + (r->p[j] - '0');
            j++;
          }
        }
      } else {
        hi = lo;
      }
      if (j >= r->len || r->p[j] != '}') break;
      r->i = j + 1;
      mn = lo;
      mx = (hi < 0) ? G_INF : hi;
      if (mx < mn) mx = mn;
      has = 1;
    } else {
      break;
    }
  }
  if (!has) return a;
  if (a->type == N_CLASS && a->rmin == 1 && a->rmax == 1) {
    a->rmin = mn;
    a->rmax = mx;
    return a;
  }
  return n_rep(r->g, a, mn, mx);
}

static gnode *rx_concat(grex *r) {
  nlist l;
  memset(&l, 0, sizeof(l));
  while (r->i < r->len && r->p[r->i] != '|' && r->p[r->i] != ')') {
    gnode *a = rx_repeat(r);
    if (!a || !r->ok) {
      free(l.v);
      return NULL;
    }
    if (nl_push(&l, a)) {
      free(l.v);
      g_err(r->g, "grammar: out of memory");
      return NULL;
    }
  }
  return seq_build(r->g, &l);
}

static gnode *rx_alt(grex *r) {
  nlist l;
  memset(&l, 0, sizeof(l));
  gnode *first = rx_concat(r);
  if (!first || !r->ok) {
    free(l.v);
    return NULL;
  }
  if (nl_push(&l, first)) {
    free(l.v);
    return NULL;
  }
  while (r->i < r->len && r->p[r->i] == '|') {
    r->i++;
    gnode *a = rx_concat(r);
    if (!a || !r->ok) {
      free(l.v);
      return NULL;
    }
    if (nl_push(&l, a)) {
      free(l.v);
      return NULL;
    }
  }
  if (l.n == 1) {
    gnode *x = l.v[0];
    free(l.v);
    return x;
  }
  gnode *x = n_alt(r->g, (const gnode *const *)l.v, l.n);
  free(l.v);
  return x;
}

static gnode *compile_regex(sgl_grammar *g, const char *pat) {
  grex r;
  memset(&r, 0, sizeof(r));
  r.g = g;
  r.p = pat;
  r.len = pat ? (int)strlen(pat) : 0;
  r.ok = 1;
  gnode *n = rx_alt(&r);
  if (!n || !r.ok || r.i != r.len) {
    if (!g->err[0]) g_err(g, "grammar: invalid regex");
    return NULL;
  }
  return n;
}

/* ---------------- JSON Schema 编译器 ---------------- */
static gnode *compile_schema(sgl_grammar *g, const sgl_json *s, int depth);

static gnode *string_unit(sgl_grammar *g) {
  /* 一个"字符"：普通字符（UTF-8 一个码点）或一个转义序列 */
  gnode *utf8 = nnew(g, N_UTF8);
  if (!utf8) return NULL;
  set_clear(utf8->set);
  set_range(utf8->set, 0x20, 0x7E);
  utf8->set['"' >> 3] &= (uint8_t)~(1u << ('"' & 7));
  utf8->set['\\' >> 3] &= (uint8_t)~(1u << ('\\' & 7));
  utf8->multi = 1;

  uint8_t se[32];
  set_clear(se);
  const char *simple = "\"\\/bfnrt";
  for (const char *p = simple; *p; p++) set_add(se, (uint8_t)*p);
  const gnode *esc_kids[2] = {n_lit(g, "\\", 1), n_class(g, se, 1, 1)};
  gnode *esc = n_seq(g, esc_kids, 2);

  uint8_t hex[32];
  set_clear(hex);
  set_range(hex, '0', '9');
  set_range(hex, 'a', 'f');
  set_range(hex, 'A', 'F');
  const gnode *uni_kids[2] = {n_lit(g, "\\u", 2), n_class(g, hex, 4, 4)};
  gnode *uni = n_seq(g, uni_kids, 2);

  const gnode *alts[3] = {utf8, esc, uni};
  return n_alt(g, alts, 3);
}

static gnode *compile_string(sgl_grammar *g, const sgl_json *s) {
  const char *pat = sgl_json_get_str(s, "pattern", NULL);
  if (pat) { /* pattern 匹配的是字符串内容，前后引号仍必需 */
    gnode *rx = compile_regex(g, pat);
    if (!rx) return NULL;
    const gnode *pk[3] = {n_lit(g, "\"", 1), rx, n_lit(g, "\"", 1)};
    return n_seq(g, pk, 3);
  }
  int mn = (int)sgl_json_get_num(s, "minLength", 0);
  double mxv = sgl_json_get_num(s, "maxLength", (double)G_INF);
  int mx = (mxv >= (double)G_INF) ? G_INF : (int)mxv;
  if (mn < 0) mn = 0;
  if (mx < mn) mx = mn;
  gnode *unit = string_unit(g);
  if (!unit) return NULL;
  gnode *content = n_rep(g, unit, mn, mx);
  const gnode *kids[3] = {n_lit(g, "\"", 1), content, n_lit(g, "\"", 1)};
  return n_seq(g, kids, 3);
}

static gnode *compile_number_node(sgl_grammar *g, int is_int, const sgl_json *s) {
  gnode *n = nnew(g, N_NUM);
  if (!n) return NULL;
  n->is_int = is_int;
  n->allow_frac = !is_int;
  n->allow_exp = !is_int;
  n->vmin = -1.0;
  n->vmax = 1.0;
  if (s) {
    const sgl_json *mn = sgl_json_get(s, "minimum");
    const sgl_json *mx = sgl_json_get(s, "maximum");
    if (mn && mn->type == SGL_JSON_NUMBER) {
      n->has_min = 1;
      n->vmin = mn->num;
    }
    if (mx && mx->type == SGL_JSON_NUMBER) {
      n->has_max = 1;
      n->vmax = mx->num;
    }
  }
  n->allow_minus = (!n->has_min || n->vmin < 0);
  if (n->has_min && n->has_max && n->vmin > n->vmax) {
    g_err(g, "grammar: minimum > maximum");
    return NULL;
  }
  return n;
}

static gnode *any_json(sgl_grammar *g);

static gnode *compile_object(sgl_grammar *g, const sgl_json *s, int depth) {
  const sgl_json *props = sgl_json_get(s, "properties");
  const sgl_json *req = sgl_json_get(s, "required");
  const sgl_json *addl = sgl_json_get(s, "additionalProperties");
  int nk = (props && props->type == SGL_JSON_OBJECT) ? props->count : 0;
  gnode *n = nnew(g, N_OBJ);
  if (!n) return NULL;
  n->addl = !(addl && addl->type == SGL_JSON_BOOL && addl->boolean == 0);
  /* 有 required 但没写 additionalProperties 时按严格模式处理（additionalProperties:false）：
   * 任意键名模式下无法跟踪哪些 required 键已出现，与其静默不强制，不如收紧到
   * 「只允许 properties 里列出的键」—— 结构化输出场景下这正是调用方想要的语义。*/
  if (n->addl && addl == NULL && req && req->type == SGL_JSON_ARRAY && req->count > 0)
    n->addl = 0;
  if (nk > 0) {
    if (nk > G_MAX_KEYS) {
      g_err(g, "grammar: too many properties (limit 60)");
      return NULL;
    }
    n->keys = (char **)calloc((size_t)nk, sizeof(char *));
    n->key_lens = (int *)calloc((size_t)nk, sizeof(int));
    n->vals = (gnode **)calloc((size_t)nk, sizeof(gnode *));
    if (!n->keys || !n->key_lens || !n->vals) {
      g_err(g, "grammar: out of memory");
      return NULL;
    }
    n->nkeys = nk;
    for (int i = 0; i < nk; i++) {
      gnode *k = quoted_key(g, props->keys[i]);
      if (!k) return NULL;
      n->keys[i] = k->lit;
      n->key_lens[i] = k->lit_len;
      k->lit = NULL; /* 所有权转给对象节点 */
      n->vals[i] = compile_schema(g, props->items[i], depth + 1);
      if (!n->vals[i]) return NULL;
    }
  }
  if (req && req->type == SGL_JSON_ARRAY) {
    if (n->addl) {
      /* 任意键名模式下无法跟踪 required（键名不由文法生成），只能不强制 */
      set_last_err("grammar: additionalProperties != false -> required is not enforced");
    } else {
      for (int i = 0; i < req->count; i++) {
        if (req->items[i]->type != SGL_JSON_STRING) continue;
        int found = -1;
        for (int j = 0; j < nk; j++)
          if (props && props->keys && props->keys[j] && req->items[i]->str &&
              strcmp(props->keys[j], req->items[i]->str) == 0) {
            found = j;
            break;
          }
        if (found < 0) {
          g_err(g, "grammar: required key not in properties (unsatisfiable)");
          return NULL;
        }
        n->reqmask |= (1ull << found);
      }
    }
  }
  if (n->addl) {
    /* 退化：任意键名 + 任意 JSON 值（已知属性也不再用各自的 schema 收紧） */
    gnode *key = compile_string(g, NULL);
    gnode *any = any_json(g);
    if (!key || !any) return NULL;
    const gnode *kids[3] = {key, g->colon_lit, any};
    n->member_any = n_seq(g, kids, 3);
    if (!n->member_any) return NULL;
  }
  return n;
}

static gnode *compile_array(sgl_grammar *g, const sgl_json *s, int depth) {
  const sgl_json *it = sgl_json_get(s, "items");
  gnode *item = it ? compile_schema(g, it, depth + 1) : any_json(g);
  if (!item) return NULL;
  int mn = (int)sgl_json_get_num(s, "minItems", 0);
  double mxv = sgl_json_get_num(s, "maxItems", (double)G_INF);
  int mx = (mxv >= (double)G_INF) ? G_INF : (int)mxv;
  if (mn < 0) mn = 0;
  if (mx < mn) mx = mn;

  /* 逗号分隔：第 2..mx 个元素 */
  const gnode *sep_kids[2] = {n_lit(g, ",", 1), item};
  gnode *sep = n_seq(g, sep_kids, 2);
  int sep_min = (mn >= 1) ? mn - 1 : 0;
  int sep_max = (mx >= 1) ? mx - 1 : 0;
  gnode *rest = n_rep(g, sep, sep_min, sep_max);
  const gnode *inner_kids[2] = {item, rest};
  gnode *inner = n_seq(g, inner_kids, 2);
  gnode *head = n_rep(g, inner, mn > 0 ? 1 : 0, mx > 0 ? 1 : 0);

  const gnode *arr_kids[3] = {n_lit(g, "[", 1), head, n_lit(g, "]", 1)};
  return n_seq(g, arr_kids, 3);
}

static gnode *any_json(sgl_grammar *g) {
  if (g->any_node) return g->any_node;
  gnode *any = nnew(g, N_ANY);
  if (!any) return NULL;
  g->any_node = any;
  gnode *strn = compile_string(g, NULL);
  gnode *numn = compile_number_node(g, 0, NULL);
  if (!strn || !numn) return NULL;
  /* 任意对象：{"<key>": <any>} */
  gnode *obj = nnew(g, N_OBJ);
  if (!obj) return NULL;
  obj->addl = 1;
  const gnode *mem_kids[3] = {strn, g->colon_lit, any};
  obj->member_any = n_seq(g, mem_kids, 3);
  if (!obj->member_any) return NULL;
  /* 任意数组：[<any>{,<any>}] */
  const gnode *sep_kids[2] = {n_lit(g, ",", 1), any};
  gnode *sep = n_seq(g, sep_kids, 2);
  gnode *rest = n_rep(g, sep, 0, G_INF);
  const gnode *inner_kids[2] = {any, rest};
  gnode *inner = n_seq(g, inner_kids, 2);
  gnode *head = n_rep(g, inner, 0, 1);
  const gnode *arr_kids[3] = {n_lit(g, "[", 1), head, n_lit(g, "]", 1)};
  gnode *arr = n_seq(g, arr_kids, 3);
  if (!arr) return NULL;
  const gnode *alts[7] = {obj,          arr,          strn, n_lit(g, "true", 4),
                          n_lit(g, "false", 5), n_lit(g, "null", 4), numn};
  g->any_alt = n_alt(g, alts, 7);
  if (!g->any_alt) return NULL;
  return any;
}

static gnode *compile_type(sgl_grammar *g, const sgl_json *s, const char *name, int depth) {
  if (!name) return any_json(g);
  if (strcmp(name, "object") == 0) return compile_object(g, s, depth);
  if (strcmp(name, "array") == 0) return compile_array(g, s, depth);
  if (strcmp(name, "string") == 0) return compile_string(g, s);
  if (strcmp(name, "number") == 0) return compile_number_node(g, 0, s);
  if (strcmp(name, "integer") == 0) return compile_number_node(g, 1, s);
  if (strcmp(name, "boolean") == 0) {
    const gnode *kids[2] = {n_lit(g, "true", 4), n_lit(g, "false", 5)};
    return n_alt(g, kids, 2);
  }
  if (strcmp(name, "null") == 0) return n_lit(g, "null", 4);
  return any_json(g);
}

static gnode *compile_schema(sgl_grammar *g, const sgl_json *s, int depth) {
  if (!s) return any_json(g);
  if (depth > G_MAX_DEPTH) {
    g_err(g, "grammar: schema nesting too deep");
    return NULL;
  }
  if (s->type == SGL_JSON_BOOL) {
    if (s->boolean) return any_json(g);
    g_err(g, "grammar: schema is false (unsatisfiable)");
    return NULL;
  }
  if (s->type != SGL_JSON_OBJECT) {
    g_err(g, "grammar: schema must be an object or boolean");
    return NULL;
  }
  if (sgl_json_get(s, "allOf")) {
    g_err(g, "grammar: allOf is not supported (use anyOf or flat schema)");
    return NULL;
  }
  const sgl_json *cst = sgl_json_get(s, "const");
  if (cst) return lit_from_json(g, cst);
  const sgl_json *en = sgl_json_get(s, "enum");
  if (en && en->type == SGL_JSON_ARRAY) {
    if (en->count <= 0) {
      g_err(g, "grammar: empty enum");
      return NULL;
    }
    if (en->count == 1) return lit_from_json(g, en->items[0]);
    gnode **kids = (gnode **)calloc((size_t)en->count, sizeof(gnode *));
    if (!kids) {
      g_err(g, "grammar: out of memory");
      return NULL;
    }
    for (int i = 0; i < en->count; i++) {
      kids[i] = lit_from_json(g, en->items[i]);
      if (!kids[i]) {
        free(kids);
        return NULL;
      }
    }
    gnode *n = n_alt(g, (const gnode *const *)kids, en->count);
    free(kids);
    return n;
  }
  const sgl_json *sel = sgl_json_get(s, "anyOf");
  if (!sel) sel = sgl_json_get(s, "oneOf");
  if (sel && sel->type == SGL_JSON_ARRAY) {
    if (sel->count <= 0) {
      g_err(g, "grammar: empty anyOf");
      return NULL;
    }
    gnode **kids = (gnode **)calloc((size_t)sel->count, sizeof(gnode *));
    if (!kids) {
      g_err(g, "grammar: out of memory");
      return NULL;
    }
    for (int i = 0; i < sel->count; i++) {
      kids[i] = compile_schema(g, sel->items[i], depth + 1);
      if (!kids[i]) {
        free(kids);
        return NULL;
      }
    }
    gnode *n = n_alt(g, (const gnode *const *)kids, sel->count);
    free(kids);
    return n;
  }
  const sgl_json *tv = sgl_json_get(s, "type");
  if (tv) {
    if (tv->type == SGL_JSON_STRING) return compile_type(g, s, tv->str, depth);
    if (tv->type == SGL_JSON_ARRAY) {
      if (tv->count <= 0) {
        g_err(g, "grammar: empty type array");
        return NULL;
      }
      if (tv->count == 1)
        return compile_type(g, s, tv->items[0]->type == SGL_JSON_STRING ? tv->items[0]->str : NULL, depth);
      gnode **kids = (gnode **)calloc((size_t)tv->count, sizeof(gnode *));
      if (!kids) {
        g_err(g, "grammar: out of memory");
        return NULL;
      }
      for (int i = 0; i < tv->count; i++) {
        kids[i] = compile_type(g, s, tv->items[i]->type == SGL_JSON_STRING ? tv->items[i]->str : NULL,
                               depth);
        if (!kids[i]) {
          free(kids);
          return NULL;
        }
      }
      gnode *n = n_alt(g, (const gnode *const *)kids, tv->count);
      free(kids);
      return n;
    }
  }
  /* 没写 type：按出现的关键字推断 */
  if (sgl_json_get(s, "properties") || sgl_json_get(s, "required") ||
      sgl_json_get(s, "additionalProperties"))
    return compile_object(g, s, depth);
  if (sgl_json_get(s, "items") || sgl_json_get(s, "minItems") || sgl_json_get(s, "maxItems"))
    return compile_array(g, s, depth);
  if (sgl_json_get(s, "pattern") || sgl_json_get(s, "minLength") || sgl_json_get(s, "maxLength"))
    return compile_string(g, s);
  if (sgl_json_get(s, "minimum") || sgl_json_get(s, "maximum")) return compile_number_node(g, 0, s);
  return any_json(g);
}

/* ---------------- 生命周期 ---------------- */
static sgl_grammar *grammar_new(void) {
  sgl_grammar *g = (sgl_grammar *)calloc(1, sizeof(sgl_grammar));
  if (!g) return NULL;
  g->colon_lit = n_lit(g, ":", 1);
  if (!g->colon_lit) {
    sgl_grammar_free(g);
    return NULL;
  }
  return g;
}

static int grammar_start(sgl_grammar *g) {
  g->st.num = 0;
  if (push_node(g, &g->st, g->root)) {
    g_err(g, "grammar: cannot expand root");
    return -1;
  }
  settle(g, &g->st);
  return 0;
}

sgl_grammar *sgl_grammar_from_json_schema(const char *schema_text, size_t len) {
  set_last_err("");
  if (!schema_text || len == 0) {
    set_last_err("grammar: empty schema text");
    return NULL;
  }
  sgl_grammar *g = grammar_new();
  if (!g) {
    set_last_err("grammar: out of memory");
    return NULL;
  }
  sgl_json *j = sgl_json_parse(schema_text, len);
  if (!j) {
    g_err(g, "grammar: schema is not valid JSON");
    sgl_grammar_free(g);
    return NULL;
  }
  g->root = compile_schema(g, j, 0);
  sgl_json_free(j);
  if (!g->root || g->err[0]) {
    if (!g->err[0]) g_err(g, "grammar: cannot compile schema");
    set_last_err(g->err);
    sgl_grammar_free(g);
    return NULL;
  }
  if (grammar_start(g)) {
    set_last_err(g->err);
    sgl_grammar_free(g);
    return NULL;
  }
  return g;
}

sgl_grammar *sgl_grammar_from_regex(const char *pattern) {
  set_last_err("");
  if (!pattern) {
    set_last_err("grammar: null pattern");
    return NULL;
  }
  sgl_grammar *g = grammar_new();
  if (!g) {
    set_last_err("grammar: out of memory");
    return NULL;
  }
  g->root = compile_regex(g, pattern);
  if (!g->root || grammar_start(g)) {
    if (!g->err[0]) g_err(g, "grammar: cannot compile regex");
    set_last_err(g->err);
    sgl_grammar_free(g);
    return NULL;
  }
  return g;
}

void sgl_grammar_free(sgl_grammar *g) {
  if (!g) return;
  for (int i = 0; i < g->narena; i++) node_free(g->arena[i]);
  free(g->arena);
  free(g->st.frames);
  free(g);
}

void sgl_grammar_reset(sgl_grammar *g) {
  if (!g) return;
  g->err[0] = 0;
  grammar_start(g);
}

void sgl_grammar_set_vocab(sgl_grammar *g, const char *const *piece_bytes, const int *piece_len,
                           int n_pieces) {
  if (!g) return;
  g->pieces = piece_bytes;
  g->plens = piece_len;
  g->npieces = n_pieces > 0 ? n_pieces : 0;
}

const char *sgl_grammar_error(const sgl_grammar *g) {
  if (!g) return g_last_err;
  return g->err;
}

/* token id → 字节片段；0 表示该 token 不可用 */
static int vocab_piece(const sgl_grammar *g, int id, const char **p, int *len) {
  if (id < 0 || id >= g->npieces || !g->pieces) return 0;
  const char *s = g->pieces[id];
  if (!s) return 0;
  int l = g->plens ? g->plens[id] : (int)strlen(s);
  if (l < 0) return 0;
  *p = s;
  *len = l;
  return 1;
}

int sgl_grammar_accept_bytes(sgl_grammar *g, const char *bytes, int len) {
  if (!g) return -1;
  if (len <= 0) return 0;
  if (!bytes) return -1;
  g->err[0] = 0;
  gstack tmp;
  memset(&tmp, 0, sizeof(tmp));
  if (g->st.num > 0) {
    tmp.frames = (gframe *)malloc((size_t)g->st.num * sizeof(gframe));
    if (!tmp.frames) {
      g_err(g, "grammar: out of memory");
      return -1;
    }
    memcpy(tmp.frames, g->st.frames, (size_t)g->st.num * sizeof(gframe));
  }
  tmp.num = g->st.num;
  tmp.cap = g->st.num;
  int rc = gs_accept_bytes(g, &tmp, bytes, len);
  if (rc == 0) {
    free(g->st.frames);
    g->st.frames = tmp.frames;
    g->st.num = tmp.num;
    g->st.cap = tmp.cap;
    return 0;
  }
  free(tmp.frames);
  g_err(g, "grammar: illegal byte sequence at input");
  return -1;
}

int sgl_grammar_accept(sgl_grammar *g, int token_id) {
  if (!g) return -1;
  const char *p = NULL;
  int len = 0;
  if (!vocab_piece(g, token_id, &p, &len)) {
    g_err(g, "grammar: token not available in vocabulary");
    return -1;
  }
  return sgl_grammar_accept_bytes(g, p, len);
}

void sgl_grammar_mask(const sgl_grammar *g, const int *token_ids, int n, unsigned char *allowed) {
  if (!allowed || n <= 0) return;
  for (int i = 0; i < n; i++) allowed[i] = 0;
  if (!g || !token_ids) return;
  /* 基准状态复制一份，再对每个候选 token 在临时栈上跑一遍 */
  gstack base;
  memset(&base, 0, sizeof(base));
  if (g->st.num > 0) {
    base.frames = (gframe *)malloc((size_t)g->st.num * sizeof(gframe));
    if (!base.frames) return;
    memcpy(base.frames, g->st.frames, (size_t)g->st.num * sizeof(gframe));
  }
  base.num = g->st.num;
  base.cap = g->st.num;
  gstack tmp;
  memset(&tmp, 0, sizeof(tmp));
  tmp.frames = (gframe *)malloc((size_t)(g->st.num ? g->st.num : 1) * sizeof(gframe));
  if (!tmp.frames) {
    free(base.frames);
    return;
  }
  tmp.cap = g->st.num ? g->st.num : 1;
  for (int i = 0; i < n; i++) {
    const char *p = NULL;
    int len = 0;
    if (!vocab_piece(g, token_ids[i], &p, &len)) continue;
    if (len == 0) {
      allowed[i] = 1; /* 空片段不改变状态 */
      continue;
    }
    if (base.num > 0) memcpy(tmp.frames, base.frames, (size_t)base.num * sizeof(gframe));
    tmp.num = base.num;
    allowed[i] = (gs_accept_bytes(g, &tmp, p, len) == 0) ? 1 : 0;
  }
  free(tmp.frames);
  free(base.frames);
}

int sgl_grammar_can_finish(const sgl_grammar *g) {
  if (!g) return 0;
  return finish_frame(g, &g->st, g->st.num - 1) ? 1 : 0;
}