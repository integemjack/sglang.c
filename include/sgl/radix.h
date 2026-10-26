/* sgl/radix.h — RadixAttention 前缀树缓存（对应 sglang srt/mem_cache/radix_cache.py）
 * 节点持有 token 段及其 KV 页号；语义：
 *   - match：最长前缀匹配，返回命中长度并借用其页（引用 +1 由调用方完成）
 *   - insert：完整序列入树（树成为页的共同持有者，页引用 +1）；
 *     已有前缀自动复用，分歧处自动分裂节点
 *   - evict：LRU 驱逐「无运行请求引用」的叶子节点，直到释放出足够页 */
#ifndef SGL_RADIX_H
#define SGL_RADIX_H

#include "sgl/model.h"

typedef struct sgl_radix sgl_radix;

sgl_radix *sgl_radix_new(sgl_model *model);
void sgl_radix_free(sgl_radix *r);

/* 最长前缀匹配：tokens[0..n) → 返回命中长度；out_pages 收集命中部分的页号
 * （容量须 ≥ n）。同时刷新路径 LRU。 */
int sgl_radix_match(sgl_radix *r, const int *tokens, int n, int *out_pages);

/* 完整序列入树：pages[i] 为位置 i 的页号（长度 n）。树对 [0,n) 的页各 +1 引用；
 * 调用方此后仍持有自己的引用，自行释放。 */
void sgl_radix_insert(sgl_radix *r, const int *tokens, const int *pages, int n);

/* LRU 驱逐至释放 ≥ need 页（被运行请求引用的节点跳过）。返回实际释放页数。 */
int sgl_radix_evict(sgl_radix *r, int need);

/* 统计 */
int sgl_radix_n_nodes(const sgl_radix *r);
int sgl_radix_n_pages(const sgl_radix *r); /* 树持有的页总数 */

#endif
