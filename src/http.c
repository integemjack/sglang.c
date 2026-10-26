/* sgl http 实现：winsock2 + 每连接一线程 + worker 驱动调度器 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/http.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/log.h"

#define SGL_HTTP_BODY_MAX (4 << 20) /* 4 MiB */

typedef struct done_item {
  int req_id;
  char *text;   /* 生成的文本（不含 prompt）*/
  int *ids;
  int n_ids;
  int prompt_tokens;
  struct done_item *next;
} done_item;

struct sgl_http_server {
  sgl_scheduler *sched;
  SOCKET listen_fd;
  pthread_mutex_t lock;
  pthread_cond_t cv_work;  /* 有新请求提交 / 需要推进 */
  pthread_cond_t cv_done;  /* 有请求完成 */
  pthread_t worker, acceptor;
  int stop;            /* worker 停止标志 */
  int active_conns;    /* 挡 acceptor 退出时的悬挂连接 */
  pthread_cond_t cv_conns_zero;
  done_item *done_head, *done_tail;
  long req_counter;    /* OpenAI id 生成 */
};

/* ---------- 响应构造：动态字符串 + JSON 转义 ---------- */

typedef struct { char *buf; size_t len, cap; } sbuf;

static void sb_init(sbuf *s) {
  s->cap = 256; s->len = 0;
  s->buf = (char *)sgl_xmalloc(s->cap);
  s->buf[0] = 0;
}
static void sb_putn(sbuf *s, const char *p, size_t n) {
  if (s->len + n + 1 > s->cap) {
    while (s->len + n + 1 > s->cap) s->cap *= 2;
    s->buf = (char *)sgl_xrealloc(s->buf, s->cap);
  }
  memcpy(s->buf + s->len, p, n);
  s->len += n;
  s->buf[s->len] = 0;
}
static void sb_puts(sbuf *s, const char *p) { sb_putn(s, p, strlen(p)); }
static void sb_putjson(sbuf *s, const char *p) { /* UTF-8 原样，转义控制字符 */
  sb_puts(s, "\"");
  for (const unsigned char *q = (const unsigned char *)p; *q; q++) {
    switch (*q) {
      case '"': sb_puts(s, "\\\""); break;
      case '\\': sb_puts(s, "\\\\"); break;
      case '\n': sb_puts(s, "\\n"); break;
      case '\r': sb_puts(s, "\\r"); break;
      case '\t': sb_puts(s, "\\t"); break;
      default:
        if (*q < 0x20) { char tmp[8]; snprintf(tmp, sizeof(tmp), "\\u%04x", *q); sb_puts(s, tmp); }
        else sb_putn(s, (const char *)q, 1);
    }
  }
  sb_puts(s, "\"");
}

static void send_all(SOCKET fd, const char *p, size_t n) {
  while (n > 0) {
    int k = send(fd, p, (int)n, 0);
    if (k <= 0) return;
    p += k; n -= (size_t)k;
  }
}

static void respond(SOCKET fd, int code, const char *status, const char *body) {
  sbuf r;
  sb_init(&r);
  char head[256];
  snprintf(head, sizeof(head), "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
           code, status, strlen(body));
  sb_puts(&r, head);
  sb_puts(&r, body);
  send_all(fd, r.buf, r.len);
  sgl_xfree(r.buf);
}

static void respond_error(SOCKET fd, int code, const char *msg) {
  sbuf b; sb_init(&b);
  sb_puts(&b, "{\"error\":{\"message\":");
  sb_putjson(&b, msg);
  sb_puts(&b, "}}");
  respond(fd, code, code == 404 ? "Not Found" : "Bad Request", b.buf);
  sgl_xfree(b.buf);
}

/* ---------- 请求读取 ---------- */

typedef struct {
  char method[8];
  char path[256];
  char *body;   /* malloc，调用方 sgl_xfree */
  size_t body_len;
} http_req;

static int read_request(SOCKET fd, http_req *req) {
  size_t cap = 8192, len = 0;
  char *buf = (char *)sgl_xmalloc(cap);
  size_t hdr_end = 0;
  memset(req, 0, sizeof(*req));

  for (;;) {
    /* 朴素搜索 header 结束标记 \r\n\r\n（body 上限 4MB，可接受）*/
    if (len >= 4) {
      for (size_t i = 0; i + 3 < len; i++) {
        if (buf[i] == '\r' && buf[i+1] == '\n' && buf[i+2] == '\r' && buf[i+3] == '\n') { hdr_end = i + 4; break; }
      }
    }
    if (hdr_end) break;
    if (len + 1 >= cap) {
      if (cap > (SGL_HTTP_BODY_MAX + 8192)) { sgl_xfree(buf); return -1; }
      cap *= 2;
      buf = (char *)sgl_xrealloc(buf, cap);
    }
    int k = recv(fd, buf + len, (int)(cap - len - 1), 0);
    if (k <= 0) { sgl_xfree(buf); return -1; }
    len += (size_t)k;
    buf[len] = 0;
  }

  /* 请求行 */
  if (sscanf(buf, "%7s %255s", req->method, req->path) != 2) { sgl_xfree(buf); return -1; }

  /* Content-Length */
  size_t clen = 0;
  char *cl = NULL;
  for (char *p = buf; p < buf + hdr_end; ) {
    char *eol = strchr(p, '\n');
    if (!eol) break;
    if (_strnicmp(p, "content-length:", 15) == 0) { clen = (size_t)strtoul(p + 15, NULL, 10); cl = p; }
    if (_strnicmp(p, "expect:", 7) == 0 && strstr(p, "100-continue")) {
      send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
    }
    p = eol + 1;
  }
  (void)cl;

  if (clen > SGL_HTTP_BODY_MAX) { sgl_xfree(buf); return -1; }

  /* 读满 body */
  while (len - hdr_end < clen) {
    if (len + 1 >= cap) {
      cap *= 2;
      buf = (char *)sgl_xrealloc(buf, cap);
    }
    int k = recv(fd, buf + len, (int)(cap - len - 1), 0);
    if (k <= 0) break;
    len += (size_t)k;
  }
  req->body_len = len - hdr_end < clen ? len - hdr_end : clen;
  req->body = (char *)sgl_xmalloc(req->body_len + 1);
  memcpy(req->body, buf + hdr_end, req->body_len);
  req->body[req->body_len] = 0;
  sgl_xfree(buf);
  return 0;
}

/* ---------- 完成队列 ---------- */

static void done_push(sgl_http_server *srv, int id, char *text, int *ids, int n, int ptok) {
  done_item *d = (done_item *)sgl_xmalloc(sizeof(*d));
  d->req_id = id; d->text = text; d->ids = ids; d->n_ids = n; d->prompt_tokens = ptok; d->next = NULL;
  if (srv->done_tail) srv->done_tail->next = d; else srv->done_head = d;
  srv->done_tail = d;
}

static done_item *done_pop(sgl_http_server *srv, int id) {
  done_item **pp = &srv->done_head;
  for (; *pp; pp = &(*pp)->next) {
    if ((*pp)->req_id == id) {
      done_item *d = *pp;
      *pp = d->next;
      if (srv->done_tail == d) {
        srv->done_tail = NULL;
        for (done_item *t = srv->done_head; t; t = t->next) srv->done_tail = t;
      }
      return d;
    }
  }
  return NULL;
}

/* ---------- worker：驱动调度循环 ---------- */

static void *worker_fn(void *arg) {
  sgl_http_server *srv = (sgl_http_server *)arg;
  for (;;) {
    pthread_mutex_lock(&srv->lock);
    if (srv->stop) { pthread_mutex_unlock(&srv->lock); break; }
    if (sgl_sched_nactive(srv->sched) == 0) {
      pthread_cond_wait(&srv->cv_work, &srv->lock);
      pthread_mutex_unlock(&srv->lock);
      continue;
    }
    sgl_sched_step(srv->sched);
    int id, n, npt, nhit; char *text; int *ids;
    while (sgl_sched_poll(srv->sched, &id, &text, &ids, &n, &npt, &nhit))
      done_push(srv, id, text, ids, n, npt);
    pthread_cond_broadcast(&srv->cv_done);
    pthread_mutex_unlock(&srv->lock);
  }
  return NULL;
}

/* ---------- 等待指定请求完成（须持有锁进入语义由内部处理）---------- */

static done_item *wait_done(sgl_http_server *srv, int id) {
  pthread_mutex_lock(&srv->lock);
  for (;;) {
    done_item *d = done_pop(srv, id);
    if (d) { pthread_mutex_unlock(&srv->lock); return d; }
    pthread_cond_wait(&srv->cv_done, &srv->lock);
  }
}

/* ---------- 路由 ---------- */

static int json_bool(const sgl_json *j, const char *key) {
  const sgl_json *v = sgl_json_get(j, key);
  return v && v->type == SGL_JSON_BOOL && v->boolean;
}

/* OpenAI stop：string 或 string[]（最多 4 个，各限 128 字节）*/
static void parse_stops(const sgl_json *j, char stops[4][128], int *n_stops) {
  *n_stops = 0;
  const sgl_json *v = sgl_json_get(j, "stop");
  if (!v) return;
  if (v->type == SGL_JSON_STRING) {
    snprintf(stops[0], 128, "%s", v->str);
    *n_stops = 1;
  } else if (v->type == SGL_JSON_ARRAY) {
    for (int i = 0; i < v->count && *n_stops < 4; i++)
      if (v->items[i]->type == SGL_JSON_STRING)
        snprintf(stops[(*n_stops)++], 128, "%s", v->items[i]->str);
  }
}

/* 截断到最早出现的 stop（不含 stop 本身，OpenAI 语义）；返回是否命中 */
static int apply_stops(char stops[4][128], int n_stops, char *text) {
  if (n_stops <= 0) return 0;
  char *earliest = NULL;
  for (int i = 0; i < n_stops; i++) {
    char *hit = strstr(text, stops[i]);
    if (hit && (!earliest || hit < earliest)) earliest = hit;
  }
  if (earliest) { *earliest = 0; return 1; }
  return 0;
}

/* 流式响应头（close 分帧：无 Content-Length，连接关闭即结束）*/
static void respond_open_stream(SOCKET fd) {
  const char *hdr =
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/event-stream\r\n"
      "Cache-Control: no-cache\r\n"
      "Connection: close\r\n"
      "\r\n";
  send(fd, hdr, (int)strlen(hdr), 0);
}

static void route(sgl_http_server *srv, SOCKET fd, const http_req *req) {
  if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/health") == 0) {
    respond(fd, 200, "OK", "{\"ok\":true}");
    return;
  }
  if (strcmp(req->method, "POST") != 0 ||
      (strcmp(req->path, "/generate") != 0 && strcmp(req->path, "/v1/completions") != 0 &&
       strcmp(req->path, "/v1/chat/completions") != 0)) {
    respond_error(fd, 404, "only GET /health, POST /generate, /v1/completions, /v1/chat/completions");
    return;
  }
  int chat = strcmp(req->path, "/v1/chat/completions") == 0;
  int is_stream = 0;
  char stops[4][128];
  int n_stops = 0;
  int n_choices = 1;

  sgl_json *j = sgl_json_parse(req->body, req->body_len);
  if (!j) { respond_error(fd, 400, "invalid JSON body"); return; }

  const char *text = NULL;
  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp));
  int max_new = 16;
  int openai = strcmp(req->path, "/v1/completions") == 0;
  char *chatml = NULL; /* chat 模式下渲染后的 prompt */

  if (chat) {
    /* OpenAI chat：messages[] → ChatML（Qwen/Llama-3 等主流 instruct 模板）*/
    const sgl_json *msgs = sgl_json_get(j, "messages");
    if (!msgs || msgs->type != SGL_JSON_ARRAY || msgs->count == 0) {
      sgl_json_free(j);
      respond_error(fd, 400, "\"messages\" array required");
      return;
    }
    sbuf ml;
    sb_init(&ml);
    for (int i = 0; i < msgs->count; i++) {
      const sgl_json *m = msgs->items[i];
      const char *role = sgl_json_get_str(m, "role", "user");
      const char *content = sgl_json_get_str(m, "content", "");
      sb_puts(&ml, "<|im_start|>");
      sb_puts(&ml, role);
      sb_puts(&ml, "\n");
      sb_puts(&ml, content);
      sb_puts(&ml, "<|im_end|>\n");
    }
    sb_puts(&ml, "<|im_start|>assistant\n");
    chatml = ml.buf;
    max_new = sgl_json_get_int(j, "max_tokens", 256);
    sp.temperature = (float)sgl_json_get_num(j, "temperature", 0.0);
    sp.top_p = (float)sgl_json_get_num(j, "top_p", 0.0);
    sp.seed = (uint64_t)sgl_json_get_int(j, "seed", 0);
    is_stream = chat && json_bool(j, "stream");
    parse_stops(j, stops, &n_stops);
    n_choices = sgl_json_get_int(j, "n", 1);
    if (is_stream && n_choices != 1) {
      sgl_json_free(j);
      respond_error(fd, 400, "stream only supports n=1");
      return;
    }
    if (n_choices < 1) n_choices = 1;
    if (n_choices > 4) n_choices = 4;
    if (max_new <= 0 || max_new > 4096) max_new = 256;
    sgl_json_free(j);
    text = chatml;
  } else if (openai) {
    text = sgl_json_get_str(j, "prompt", NULL);
    max_new = sgl_json_get_int(j, "max_tokens", 16);
    sp.temperature = (float)sgl_json_get_num(j, "temperature", 0.0);
    sp.top_p = (float)sgl_json_get_num(j, "top_p", 0.0);
    sp.top_k = sgl_json_get_int(j, "top_k", 0); /* OpenAI 无 top_k，兼容透传 */
    sp.seed = (uint64_t)sgl_json_get_int(j, "seed", 0);
    parse_stops(j, stops, &n_stops);
    n_choices = sgl_json_get_int(j, "n", 1);
    if (n_choices < 1) n_choices = 1;
    if (n_choices > 4) n_choices = 4;
    if (max_new <= 0 || max_new > 1024) max_new = 16;
  } else {
    text = sgl_json_get_str(j, "text", NULL);
    const sgl_json *spj = sgl_json_get(j, "sampling_params");
    if (spj) {
      max_new = sgl_json_get_int(spj, "max_new_tokens", 16);
      sp.temperature = (float)sgl_json_get_num(spj, "temperature", 0.0);
      sp.top_p = (float)sgl_json_get_num(spj, "top_p", 0.0);
      sp.top_k = sgl_json_get_int(spj, "top_k", 0);
      sp.seed = (uint64_t)sgl_json_get_int(spj, "seed", 0);
    }
    if (max_new <= 0 || max_new > 1024) max_new = 16;
  }
  if (!text || !*text) {
    sgl_json_free(j);
    respond_error(fd, 400, openai ? "\"prompt\"" : "\"text\" is required");
    return;
  }
  /* text 指向 json 树内部：立刻复制，避免下面的 json_free 变成悬垂指针；
   * chat 路径的 chatml 已是自有内存且 j 已释放。*/
  char *prompt_copy;
  if (chat) {
    prompt_copy = chatml;
  } else {
    size_t plen = strlen(text) + 1;
    char *pc = (char *)sgl_xmalloc(plen);
    memcpy(pc, text, plen);
    prompt_copy = pc;
    sgl_json_free(j);
  }
  text = prompt_copy;

  /* n 个请求（n>1 时各自加 seed 偏移，temperature>0 才有差异）*/
  int ids[4] = {-1, -1, -1, -1};
  pthread_mutex_lock(&srv->lock);
  for (int i = 0; i < n_choices; i++) {
    sgl_sampling_params spi = sp;
    if (i > 0) spi.seed = sp.seed ? sp.seed + (uint64_t)i : (uint64_t)(0x1234 + i);
    ids[i] = is_stream ? sgl_sched_submit_stream(srv->sched, text, max_new, &spi)
                       : sgl_sched_submit(srv->sched, text, max_new, &spi);
    if (ids[i] < 0) break;
  }
  pthread_mutex_unlock(&srv->lock);
  if (ids[0] < 0) {
    sgl_xfree(prompt_copy);
    respond_error(fd, 500, "submit failed (pool exhausted?)");
    return;
  }
  pthread_cond_signal(&srv->cv_work);

  /* ---- 流式分支（仅 ids[0]；n=1 已限制）---- */
  if (is_stream) {
    int id = ids[0];
    respond_open_stream(fd);
    pthread_mutex_lock(&srv->lock);
    long seq = ++srv->req_counter;
    pthread_mutex_unlock(&srv->lock);
    char idbuf[64];
    snprintf(idbuf, sizeof(idbuf), "chatcmpl-%ld", seq);
    long created = (long)time(NULL);
    for (;;) {
      char *piece = NULL;
      int tid;
      pthread_mutex_lock(&srv->lock);
      sgl_stream_ev ev = sgl_sched_stream_next(srv->sched, id, &piece, &tid);
      pthread_mutex_unlock(&srv->lock);
      if (ev == SGL_STREAM_TOKEN) {
        sbuf b;
        sb_init(&b);
        sb_puts(&b, "data: {\"id\":\"");
        sb_puts(&b, idbuf);
        char head[128];
        snprintf(head, sizeof(head), "\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"sglang-c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":", created);
        sb_puts(&b, head);
        sb_putjson(&b, piece ? piece : "");
        sb_puts(&b, "},\"finish_reason\":null}]}\n\n");
        send(fd, b.buf, (int)b.len, 0);
        sgl_xfree(b.buf);
        sgl_xfree(piece);
      } else if (ev == SGL_STREAM_DONE) {
        sbuf b;
        sb_init(&b);
        sb_puts(&b, "data: {\"id\":\"");
        sb_puts(&b, idbuf);
        char tail[192];
        snprintf(tail, sizeof(tail), "\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"sglang-c\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", created);
        sb_puts(&b, tail);
        send(fd, b.buf, (int)b.len, 0);
        sgl_xfree(b.buf);
        pthread_mutex_lock(&srv->lock);
        sgl_sched_reclaim_stream(srv->sched, id);
        pthread_mutex_unlock(&srv->lock);
        break;
      } else {
        Sleep(25); /* MORE：等下一批 */
      }
    }
    sgl_xfree(prompt_copy);
    return;
  }

  /* ---- 非流式：等全部 n 个完成 ---- */
  done_item *ds[4] = {0};
  int n_done = 0;
  for (int i = 0; i < n_choices && ids[i] >= 0; i++) {
    ds[i] = wait_done(srv, ids[i]);
    if (ds[i]) n_done++;
  }
  done_item *d = ds[0];
  int ptok = d->prompt_tokens;

  /* stop 截断（各 choice 独立）+ finish_reason */
  int hit_stop[4] = {0};
  for (int i = 0; i < n_done; i++)
    if (ds[i]->text) hit_stop[i] = apply_stops(stops, n_stops, ds[i]->text);

  sbuf b;
  sb_init(&b);
  long created = (long)time(NULL);
  if (chat) {
    char idbuf[64];
    pthread_mutex_lock(&srv->lock);
    long seq = ++srv->req_counter;
    pthread_mutex_unlock(&srv->lock);
    snprintf(idbuf, sizeof(idbuf), "chatcmpl-%ld", seq);
    char head[160];
    snprintf(head, sizeof(head), "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"sglang-c\",\"choices\":[", idbuf, created);
    sb_puts(&b, head);
    for (int i = 0; i < n_done; i++) {
      if (i) sb_puts(&b, ",");
      sb_puts(&b, "{\"index\":");
      char ib[16];
      snprintf(ib, sizeof(ib), "%d", i);
      sb_puts(&b, ib);
      sb_puts(&b, ",\"message\":{\"role\":\"assistant\",\"content\":");
      sb_putjson(&b, ds[i]->text ? ds[i]->text : "");
      sb_puts(&b, hit_stop[i] ? "},\"finish_reason\":\"stop\"}" : "},\"finish_reason\":\"length\"}");
    }
    char tail[160];
    snprintf(tail, sizeof(tail), "],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
             ptok, d->n_ids, ptok + d->n_ids);
    sb_puts(&b, tail);
  } else if (openai) {
    char idbuf[64];
    pthread_mutex_lock(&srv->lock);
    long seq = ++srv->req_counter;
    pthread_mutex_unlock(&srv->lock);
    snprintf(idbuf, sizeof(idbuf), "cmpl-%ld", seq);
    char head[192];
    snprintf(head, sizeof(head), "{\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":\"sglang-c-gpt2\",\"choices\":[", idbuf, created);
    sb_puts(&b, head);
    for (int i = 0; i < n_done; i++) {
      if (i) sb_puts(&b, ",");
      sb_puts(&b, "{\"text\":");
      sb_putjson(&b, ds[i]->text ? ds[i]->text : "");
      char ct[96];
      snprintf(ct, sizeof(ct), ",\"index\":%d,\"finish_reason\":\"%s\"}", i, hit_stop[i] ? "stop" : "length");
      sb_puts(&b, ct);
    }
    char tail[192];
    snprintf(tail, sizeof(tail), "],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
             ptok, d->n_ids, ptok + d->n_ids);
    sb_puts(&b, tail);
  } else {
    sb_puts(&b, "{\"text\":");
    sb_putjson(&b, text); /* sglang 原生 /generate 返回 prompt+生成 */
    sb_puts(&b, ",\"completion\":");
    sb_putjson(&b, d->text ? d->text : "");
    char meta[192];
    snprintf(meta, sizeof(meta), ",\"meta_info\":{\"finish_reason\":\"%s\",\"completion_tokens\":%d}}",
             hit_stop[0] ? "stop" : "length", d->n_ids);
    sb_puts(&b, meta);
  }
  respond(fd, 200, "OK", b.buf);
  sgl_xfree(b.buf);
  sgl_xfree(prompt_copy);
  for (int i = 0; i < n_done; i++) {
    sgl_xfree(ds[i]->text);
    sgl_xfree(ds[i]->ids);
    sgl_xfree(ds[i]);
  }
}

/* ---------- 连接线程 ---------- */

typedef struct {
  sgl_http_server *srv;
  SOCKET fd;
} conn_ctx;

static void *conn_fn(void *arg) {
  conn_ctx *ctx = (conn_ctx *)arg;
  http_req req;
  if (read_request(ctx->fd, &req) == 0) {
    route(ctx->srv, ctx->fd, &req);
    sgl_xfree(req.body);
  }
  closesocket(ctx->fd);
  pthread_mutex_lock(&ctx->srv->lock);
  if (--ctx->srv->active_conns == 0) pthread_cond_broadcast(&ctx->srv->cv_conns_zero);
  pthread_mutex_unlock(&ctx->srv->lock);
  sgl_xfree(ctx);
  return NULL;
}

static void *acceptor_fn(void *arg) {
  sgl_http_server *srv = (sgl_http_server *)arg;
  for (;;) {
    struct sockaddr_in peer;
    int plen = sizeof(peer);
    SOCKET c = accept(srv->listen_fd, (struct sockaddr *)&peer, &plen);
    if (c == INVALID_SOCKET) break; /* listen socket 关闭（stop）*/
    if (srv->stop) { closesocket(c); break; }
    pthread_mutex_lock(&srv->lock);
    srv->active_conns++;
    pthread_mutex_unlock(&srv->lock);
    conn_ctx *ctx = (conn_ctx *)sgl_xmalloc(sizeof(*ctx));
    ctx->srv = srv; ctx->fd = c;
    pthread_t t;
    if (pthread_create(&t, NULL, conn_fn, ctx) != 0) {
      closesocket(c);
      pthread_mutex_lock(&srv->lock);
      srv->active_conns--;
      pthread_mutex_unlock(&srv->lock);
      sgl_xfree(ctx);
    } else {
      pthread_detach(t);
    }
  }
  return NULL;
}

/* ---------- 生命周期 ---------- */

sgl_http_server *sgl_http_serve(sgl_scheduler *sched, const char *host, int port) {
  WSADATA wd;
  if (WSAStartup(MAKEWORD(2, 2), &wd) != 0) { SGL_ERROR("WSAStartup failed"); return NULL; }

  sgl_http_server *srv = (sgl_http_server *)sgl_xcalloc(1, sizeof(*srv));
  srv->sched = sched;
  srv->listen_fd = INVALID_SOCKET;
  pthread_mutex_init(&srv->lock, NULL);
  pthread_cond_init(&srv->cv_work, NULL);
  pthread_cond_init(&srv->cv_done, NULL);
  pthread_cond_init(&srv->cv_conns_zero, NULL);

  srv->listen_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (srv->listen_fd == INVALID_SOCKET) { SGL_ERROR("socket() failed"); goto fail; }
  int yes = 1;
  setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons((unsigned short)port);
  if (!host || !strcmp(host, "0.0.0.0")) addr.sin_addr.s_addr = htonl(INADDR_ANY);
  else if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

  if (bind(srv->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
    SGL_ERROR("bind %s:%d failed (WSAGetLastError=%d)", host ? host : "127.0.0.1", port, WSAGetLastError());
    goto fail;
  }
  if (listen(srv->listen_fd, 64) != 0) { SGL_ERROR("listen failed"); goto fail; }

  if (pthread_create(&srv->worker, NULL, worker_fn, srv) != 0) { SGL_ERROR("worker thread"); goto fail; }
  if (pthread_create(&srv->acceptor, NULL, acceptor_fn, srv) != 0) {
    pthread_mutex_lock(&srv->lock); srv->stop = 1; pthread_cond_broadcast(&srv->cv_work); pthread_mutex_unlock(&srv->lock);
    pthread_join(srv->worker, NULL);
    goto fail;
  }
  SGL_INFO("http serving on %s:%d (/health /generate /v1/completions)",
           host ? host : "127.0.0.1", port);
  return srv;

fail:
  if (srv->listen_fd != INVALID_SOCKET) closesocket(srv->listen_fd);
  pthread_cond_destroy(&srv->cv_work); pthread_cond_destroy(&srv->cv_done); pthread_cond_destroy(&srv->cv_conns_zero);
  pthread_mutex_destroy(&srv->lock);
  sgl_xfree(srv);
  WSACleanup();
  return NULL;
}

void sgl_http_stop(sgl_http_server *srv) {
  if (!srv) return;
  pthread_mutex_lock(&srv->lock);
  srv->stop = 1;
  pthread_cond_broadcast(&srv->cv_work);
  pthread_cond_broadcast(&srv->cv_done);
  while (srv->active_conns > 0) {
    pthread_mutex_unlock(&srv->lock);
    closesocket(srv->listen_fd); /* 令 accept 退出 */
    Sleep(100);
    pthread_mutex_lock(&srv->lock);
  }
  pthread_mutex_unlock(&srv->lock);
  closesocket(srv->listen_fd);
  pthread_join(srv->acceptor, NULL);
  pthread_join(srv->worker, NULL);
  while (srv->done_head) {
    done_item *d = srv->done_head;
    srv->done_head = d->next;
    sgl_xfree(d->text); sgl_xfree(d->ids); sgl_xfree(d);
  }
  pthread_cond_destroy(&srv->cv_work); pthread_cond_destroy(&srv->cv_done); pthread_cond_destroy(&srv->cv_conns_zero);
  pthread_mutex_destroy(&srv->lock);
  sgl_xfree(srv);
  WSACleanup();
}
