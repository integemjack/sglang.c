/* sgl/http.h — 极简 HTTP/1.1 推理服务（对应 sglang 的 http_server.py 子集）
 * 端点：
 *   GET  /health          → {"ok":true}
 *   POST /generate        → sglang 原生格式
 *        {"text": "...", "sampling_params": {"max_new_tokens":N, "temperature":t,
 *         "top_k":k, "top_p":p, "seed":s}}
 *        ← {"text": prompt+生成, "completion": 仅生成, "meta_info": {...}}
 *   POST /v1/completions  → OpenAI 兼容子集
 *        {"model","prompt","max_tokens","temperature",...}
 *        ← {"choices":[{"text":...,"finish_reason":"stop"}],"usage":{...}}
 * 并发：每连接一线程；后台 worker 线程驱动 sgl_scheduler（互斥锁串行化），
 * 因此并发请求会被连续批处理合并（与 sglang 语义一致）。
 * 已知简化：无 keep-alive（Connection: close）、无流式、不支持 ignore_eos。 */
#ifndef SGL_HTTP_H
#define SGL_HTTP_H

#include "sgl/scheduler.h"

typedef struct sgl_http_server sgl_http_server;

/* 启动服务（acceptor + worker 线程已在跑），失败返回 NULL */
sgl_http_server *sgl_http_serve(sgl_scheduler *sched, const char *host, int port);

/* ---- G8 多模态入口（CPU 路径）----
 * 服务端注册一个「图片字节 → LM 空间视觉 token」回调：
 * /v1/chat/completions 的 content[] 里出现
 *   {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
 * 时，http.c 先 base64 解码，再把图片字节交给回调；回调返
 * sgl_xmalloc 的 [n_img][E_lm]（所有权转给 http.c，最终交给调度器），*n_img 回填。
 * 未注册回调时遇到 image_url 直接 400。 */
typedef float *(*sgl_http_vision_fn)(void *ud, const unsigned char *bytes,
                                     size_t n, int *n_img, int *e_lm);
void sgl_http_set_vision(sgl_http_server *srv, sgl_http_vision_fn fn, void *ud);

/* 停止并释放（断开监听，等待线程退出）*/
void sgl_http_stop(sgl_http_server *srv);

#endif
