/* sgl radix 实现：RadixAttention 前缀树（对照 sglang radix_cache.py 的 insert/match/evict）*/
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/radix.h"

#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/log.h"
#include "sgl/model.h"

typedef struct rnode {
  struct rnode *parent, *child, *sibling; /* first-child / next-sibling */
  int *tokens;  /* 本节点 token 段 */
  int *pages;   /* 与 tokens 等长的 KV 页号 */
  int n;
  uint64_t last_access;
} rnode;

struct sgl_radix {
  sgl_model *model;
  rnode *root;      /* 虚拟根（n=0）*/
  uint64_t clock;
  int n_nodes;
  int n_pages;      /* 树持有的页数（含共享）*/
};

sgl_radix *sgl_radix_new(sgl_model *model) {
  sgl_radix *r = (sgl_radix *)sgl_xcalloc(1, sizeof(*r));
  r->model = model;
  r->root = (rnode *)sgl_xcalloc(1, sizeof(rnode));
  return r;
}

static void node_free_recur(sgl_radix *r, rnode *nd) {
  if (!nd) return;
  node_free_recur(r, nd->child);
  rnode *nx;
  for (rnode *c = nd->child; c; c = nx) { nx = c->sibling; } /* 已由递归释放 */
  /* 释放本节点页引用 */
  for (int i = 0; i < nd->n; i++) sgl_model_page_ref(r->model, nd->pages[i], -1);
  r->n_pages -= nd->n;
  r->n_nodes--;
  sgl_xfree(nd->tokens);
  sgl_xfree(nd->pages);
  sgl_xfree(nd);
}

void sgl_radix_free(sgl_radix *r) {
  if (!r) return;
  /* 后序释放全部子树 */
  rnode *stack[1024];
  int sp = 0;
  if (r->root->child) stack[sp++] = r->root->child;
  while (sp > 0) {
    rnode *nd = stack[--sp];
    if (nd->child) stack[sp++] = nd->child;
    if (nd->sibling) stack[sp++] = nd->sibling;
    if (sp > 1000) { SGL_ERROR("radix tree too deep"); break; }
    for (int i = 0; i < nd->n; i++) sgl_model_page_ref(r->model, nd->pages[i], -1);
    r->n_pages -= nd->n;
    r->n_nodes--;
    sgl_xfree(nd->tokens);
    sgl_xfree(nd->pages);
    sgl_xfree(nd);
  }
  sgl_xfree(r->root);
  sgl_xfree(r);
}

/* 在 cur 的孩子里找首 token 匹配的节点 */
static rnode *find_child(rnode *cur, int tok) {
  for (rnode *c = cur->child; c; c = c->sibling)
    if (c->tokens[0] == tok) return c;
  return NULL;
}

static rnode *node_new(rnode *parent, const int *tokens, const int *pages, int n) {
  rnode *nd = (rnode *)sgl_xcalloc(1, sizeof(rnode));
  nd->tokens = (int *)sgl_xmalloc(sizeof(int) * (size_t)(n ? n : 1));
  nd->pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)(n ? n : 1));
  memcpy(nd->tokens, tokens, sizeof(int) * (size_t)n);
  memcpy(nd->pages, pages, sizeof(int) * (size_t)n);
  nd->n = n;
  nd->parent = parent;
  /* 挂到 parent 的孩子链头 */
  nd->sibling = parent->child;
  parent->child = nd;
  return nd;
}

/* 把 nd 的前 k 个 token/页拆出新父节点，nd 变为它的孩子（保留剩余部分） */
static rnode *node_split(rnode *nd, int k) {
  rnode *mid = (rnode *)sgl_xcalloc(1, sizeof(rnode));
  mid->tokens = (int *)sgl_xmalloc(sizeof(int) * (size_t)k);
  mid->pages = (int *)sgl_xmalloc(sizeof(int) * (size_t)k);
  memcpy(mid->tokens, nd->tokens, sizeof(int) * (size_t)k);
  memcpy(mid->pages, nd->pages, sizeof(int) * (size_t)k);
  mid->n = k;
  mid->last_access = nd->last_access;
  /* nd 保留后半 */
  memmove(nd->tokens, nd->tokens + k, sizeof(int) * (size_t)(nd->n - k));
  memmove(nd->pages, nd->pages + k, sizeof(int) * (size_t)(nd->n - k));
  nd->n -= k;
  /* 挂链：mid 替换 nd 在父节点的位置，nd 变 mid 的独子 */
  rnode *p = nd->parent;
  mid->parent = p;
  if (p->child == nd) p->child = mid;
  else {
    rnode *c = p->child;
    while (c->sibling != nd) c = c->sibling;
    c->sibling = mid;
  }
  mid->sibling = nd->sibling;
  nd->sibling = NULL;
  nd->parent = mid;
  mid->child = nd;
  return mid;
}

int sgl_radix_match(sgl_radix *r, const int *tokens, int n, int *out_pages) {
  r->clock++;
  int pos = 0;
  rnode *cur = r->root;
  cur->last_access = r->clock;
  while (pos < n) {
    rnode *c = find_child(cur, tokens[pos]);
    if (!c) break;
    int k = 0;
    while (k < c->n && pos + k < n && c->tokens[k] == tokens[pos + k]) k++;
    if (k == 0) break;
    memcpy(out_pages + pos, c->pages, sizeof(int) * (size_t)k);
    c->last_access = r->clock;
    pos += k;
    cur = c;
    if (k < c->n) break; /* 部分命中：页已取，但不能继续深入 */
  }
  return pos;
}

void sgl_radix_insert(sgl_radix *r, const int *tokens, const int *pages, int n) {
  r->clock++;
  if (n <= 0) return;
  rnode *cur = r->root;
  cur->last_access = r->clock;
  int pos = 0;
  while (pos < n) {
    rnode *c = find_child(cur, tokens[pos]);
    if (!c) break;
    int k = 0;
    while (k < c->n && pos + k < n && c->tokens[k] == tokens[pos + k]) k++;
    if (k < c->n) {
      /* 分裂：c 的前 k 个与新序列一致，剩余部分下放 */
      node_split(c, k);
      /* 继续在 mid（即 c，现在只剩前缀 k）下走 */
    }
    c->last_access = r->clock;
    pos += k;
    cur = c;
    if (k == 0) break; /* 不应发生（首 token 已匹配）*/
  }
  if (pos < n) {
    node_new(cur, tokens + pos, pages + pos, n - pos);
    r->n_nodes++;
  }
  /* 树成为全部页的共同持有者 */
  for (int i = 0; i < n; i++) sgl_model_page_ref(r->model, pages[i], +1);
  r->n_pages += n;
}

/* 收集可驱逐叶子（无孩子且其所有页只被树自身引用）*/
static int node_evictable(sgl_radix *r, rnode *nd) {
  if (nd->child) return 0;
  for (int i = 0; i < nd->n; i++)
    if (sgl_model_page_refs(r->model, nd->pages[i]) != 1) return 0; /* 有运行请求借用 */
  return 1;
}

static int remove_node(sgl_radix *r, rnode *nd) {
  rnode *p = nd->parent;
  if (p->child == nd) p->child = nd->sibling;
  else {
    rnode *c = p->child;
    while (c && c->sibling != nd) c = c->sibling;
    if (c) c->sibling = nd->sibling;
  }
  int freed = nd->n;
  for (int i = 0; i < nd->n; i++) sgl_model_page_ref(r->model, nd->pages[i], -1);
  r->n_pages -= nd->n;
  r->n_nodes--;
  sgl_xfree(nd->tokens);
  sgl_xfree(nd->pages);
  sgl_xfree(nd);
  return freed;
}

int sgl_radix_evict(sgl_radix *r, int need) {
  int freed = 0;
  while (freed < need) {
    /* LRU：找 last_access 最小的可驱逐叶子 */
    rnode *best = NULL;
    rnode *stack[512];
    int sp = 0;
    if (r->root->child) stack[sp++] = r->root->child;
    while (sp > 0) {
      rnode *nd = stack[--sp];
      if (node_evictable(r, nd) && (!best || nd->last_access < best->last_access)) best = nd;
      if (nd->child) stack[sp++] = nd->child;
      if (nd->sibling) stack[sp++] = nd->sibling;
      if (sp > 500) { SGL_ERROR("radix evict: tree too wide"); return freed; }
    }
    if (!best) break; /* 没有可驱逐的了 */
    freed += remove_node(r, best);
  }
  return freed;
}

int sgl_radix_n_nodes(const sgl_radix *r) { return r->n_nodes; }
int sgl_radix_n_pages(const sgl_radix *r) { return r->n_pages; }
