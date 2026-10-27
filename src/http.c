/* sgl http 实现：winsock2 + 每连接一线程 + worker 驱动调度器 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/http.h"

#include "sgl/image.h"
#include "sgl/log.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include "sgl/port.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include "sgl/thread_compat.h" /* MinGW = winpthreads；MSVC = Win32 兼容层。
                               * 必须排在 winsock2.h 之后（它会 include <windows.h>）*/
static void sgl_sock_init(void) {
  WSADATA wd;
  if (WSAStartup(MAKEWORD(2, 2), &wd) != 0) SGL_ERROR("WSAStartup failed");
}
static void sgl_sleep_ms(int ms) { Sleep(ms); }
#else
#include "sgl/port.h"
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h> /* TCP_NODELAY（t11 加的 ND 开关；Linux/macOS 不在这里声明） */
#include <arpa/inet.h>
#include <unistd.h>
#include "sgl/thread_compat.h" /* POSIX = 真 pthread */
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define closesocket(s) close(s)
#define WSAGetLastError() (errno)
#define WSACleanup() ((void)0)
#define sgl_sock_init() ((void)0)
static void sgl_sleep_ms(int ms) {
  struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
  nanosleep(&ts, NULL);
}
#endif

#include "sgl/alloc.h"
#include "sgl/bert.h"
#include "sgl/grammar.h"
#include "sgl/json.h"
#include "sgl/parser.h"
#include "sgl/log.h"

#define SGL_HTTP_BODY_MAX (4 << 20) /* 4 MiB */

typedef struct done_item {
  int req_id;
  char *text;   /* 生成的文本（不含 prompt）*/
  int *ids;
  int n_ids;
  float *lp;    /* P0-3: 每个生成 token 的 logprob（可为 NULL）*/
  int n_lp;
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
  /* G4: active_conns 以前挂在 srv->lock 上，而 worker 每推进一步都持有
   * srv->lock 做前向计算（CPU 上一个 step 可达百毫秒）。高并发时 acceptor
   * 抢不到登记锁 → 内核已完成 TCP 握手、连接却长时间不建线程 → 客户端
   * “连上但没响应”（实测 CPU 版 12 并发下 6.7% 的 /health 10s 无首字节，
   * 而 POST /generate 同时刻正常）。现在单独一把 conns_lock，acceptor 不
   * 再排在计算锁后面。 */
  pthread_mutex_t conns_lock;
  int active_conns;    /* 挡 acceptor 退出时的悬挂连接 */
  pthread_cond_t cv_conns_zero;
  done_item *done_head, *done_tail;
  long req_counter;    /* OpenAI id 生成 */
  sgl_http_vision_fn vision;   /* G8: 图片→视觉 token 回调（NULL = 不支持图片）*/
  void *vision_ud;
  sgl_jinja *jinja;            /* P0-2: 模型自带 chat_template（NULL = 回退 ChatML）*/
  sgl_bert *embed;             /* P2: embedding / rerank 模型（NULL = 未加载）*/
  char model_name[256];        /* 响应体里的 model 字段（模型目录 basename）*/
};

void sgl_http_set_vision(sgl_http_server *srv, sgl_http_vision_fn fn, void *ud) {
  if (!srv) return;
  srv->vision = fn;
  srv->vision_ud = ud;
}

void sgl_http_set_embedding(sgl_http_server *srv, sgl_bert *m) { srv->embed = m; }

void sgl_http_set_chat_template(sgl_http_server *srv, const char *model_dir) {
  if (!srv || !model_dir) return;
  srv->jinja = sgl_jinja_load_dir(model_dir);
  /* 模型名：取目录末段 */
  const char *b = model_dir;
  for (const char *p = model_dir; *p; p++)
    if (*p == '/' || *p == '\\') b = p + 1;
  snprintf(srv->model_name, sizeof(srv->model_name), "%s", b);
  if (!srv->jinja) SGL_INFO("http: no chat_template in %s, falling back to ChatML", model_dir);
}

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

/* text/plain（Prometheus /metrics）*/
static void respond_text(SOCKET fd, int code, const char *status, const char *body) {
  sbuf r;
  sb_init(&r);
  char head[256];
  snprintf(head, sizeof(head),
           "HTTP/1.1 %d %s\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
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

static void done_push(sgl_http_server *srv, int id, char *text, int *ids, int n, int ptok,
                      float *lp, int n_lp) {
  done_item *d = (done_item *)sgl_xmalloc(sizeof(*d));
  d->req_id = id; d->text = text; d->ids = ids; d->n_ids = n; d->prompt_tokens = ptok; d->next = NULL;
  d->lp = lp;
  d->n_lp = n_lp;
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
    int id, n, npt, nhit; char *text; int *ids; float *lp = NULL; int nlp = 0;
    while (sgl_sched_poll_lp(srv->sched, &id, &text, &ids, &n, &npt, &nhit, &lp, &nlp))
      done_push(srv, id, text, ids, n, npt, lp, nlp);
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

/* P0-3: OpenAI / sglang 采样参数（三个端点共用）*/
static void parse_sampling_json(const sgl_json *j, sgl_sampling_params *sp, int *max_new, int def_max,
                                int max_cap) {
  if (!j || !sp) return;
  if (max_new) {
    *max_new = sgl_json_get_int(j, "max_tokens", sgl_json_get_int(j, "max_new_tokens", def_max));
    if (*max_new <= 0 || *max_new > max_cap) *max_new = def_max;
  }
  sp->temperature = (float)sgl_json_get_num(j, "temperature", 0.0);
  sp->top_p = (float)sgl_json_get_num(j, "top_p", 0.0);
  sp->top_k = sgl_json_get_int(j, "top_k", 0);
  sp->min_p = (float)sgl_json_get_num(j, "min_p", 0.0);
  sp->repetition_penalty = (float)sgl_json_get_num(j, "repetition_penalty", 1.0);
  sp->frequency_penalty = (float)sgl_json_get_num(j, "frequency_penalty", 0.0);
  sp->presence_penalty = (float)sgl_json_get_num(j, "presence_penalty", 0.0);
  sp->ignore_eos = json_bool(j, "ignore_eos");
  sp->seed = (uint64_t)sgl_json_get_int(j, "seed", 0);
  sp->n_stop_ids = 0;
  const sgl_json *st = sgl_json_get(j, "stop_token_ids");
  if (st && st->type == SGL_JSON_ARRAY)
    for (int i = 0; i < st->count && sp->n_stop_ids < SGL_MAX_STOP_TOKENS; i++)
      if (st->items[i]->type == SGL_JSON_NUMBER)
        sp->stop_ids[sp->n_stop_ids++] = (int)st->items[i]->num;
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

/* ---------- P0-2: Jinja chat template ---------- */

static void dump_json(sbuf *o, const sgl_json *j) {
  if (!j) {
    sb_puts(o, "null");
    return;
  }
  switch (j->type) {
    case SGL_JSON_NULL: sb_puts(o, "null"); break;
    case SGL_JSON_BOOL: sb_puts(o, j->boolean ? "true" : "false"); break;
    case SGL_JSON_NUMBER: {
      char b[64];
      snprintf(b, sizeof(b), "%.17g", j->num);
      sb_puts(o, b);
      break;
    }
    case SGL_JSON_STRING: sb_putjson(o, j->str ? j->str : ""); break;
    case SGL_JSON_ARRAY:
      sb_puts(o, "[");
      for (int i = 0; i < j->count; i++) {
        if (i) sb_puts(o, ",");
        dump_json(o, j->items[i]);
      }
      sb_puts(o, "]");
      break;
    case SGL_JSON_OBJECT:
      sb_puts(o, "{");
      for (int i = 0; i < j->count; i++) {
        if (i) sb_puts(o, ",");
        sb_putjson(o, j->keys[i]);
        sb_puts(o, ":");
        dump_json(o, j->items[i]);
      }
      sb_puts(o, "}");
      break;
  }
}

/* 用模型自带的 chat_template 渲染 messages。
 * content[] 里的 image_url 先编码成视觉特征（与旧 ChatML 路径一致），
 * 文本里留 <image> 占位；tool_calls 等其余字段原样交给模板。
 * 失败返回 NULL，*err 给原因。 */
static char *chat_render_template(sgl_http_server *srv, const sgl_json *reqj, const sgl_json *msgs,
                                  float **feats_out, long *n_feats_out, int *feat_row_out,
                                  int *n_img_tok_out, const char **err) {
  sbuf t;
  sb_init(&t);
  sb_puts(&t, "[");
  float *feats = NULL;
  long n_feats = 0;
  int feat_row = 0, n_img_tok = 0;
  const char *myerr = NULL;

  for (int i = 0; i < msgs->count; i++) {
    const sgl_json *m = msgs->items[i];
    const char *role = sgl_json_get_str(m, "role", "user");
    const sgl_json *cj = sgl_json_get(m, "content");
    if (i) sb_puts(&t, ",");
    sb_puts(&t, "{\"role\":");
    sb_putjson(&t, role);
    sb_puts(&t, ",\"content\":");
    if (cj && cj->type == SGL_JSON_ARRAY) {
      sbuf txt;
      sb_init(&txt);
      for (int k = 0; k < cj->count; k++) {
        const sgl_json *part = cj->items[k];
        const char *ptype = sgl_json_get_str(part, "type", "text");
        if (strcmp(ptype, "image_url") != 0) {
          sb_puts(&txt, sgl_json_get_str(part, "text", ""));
          continue;
        }
        sb_puts(&txt, "<image>");
        if (myerr) continue;
        if (!srv->vision) {
          myerr = "server has no vision tower (use --vision)";
          continue;
        }
        const sgl_json *iu = sgl_json_get(part, "image_url");
        const char *url = iu ? sgl_json_get_str(iu, "url", "") : "";
        const char *b64 = url;
        const char *cm = strchr(url, ',');
        if (!strncmp(url, "data:", 5) && cm) b64 = cm + 1;
        unsigned char *bytes = NULL;
        size_t bn = 0;
        if (sgl_base64_decode(b64, strlen(b64), &bytes, &bn) != 0) {
          myerr = "image_url base64 decode failed";
          continue;
        }
        int ni = 0, el = 0;
        float *f = srv->vision(srv->vision_ud, bytes, bn, &ni, &el);
        sgl_xfree(bytes);
        if (!f || ni <= 0 || el <= 0) {
          sgl_xfree(f);
          myerr = "image decode / vision encode failed";
          continue;
        }
        if (feat_row == 0) feat_row = el;
        if (feat_row != el) {
          sgl_xfree(f);
          myerr = "images with different feature width";
          continue;
        }
        feats = (float *)sgl_xrealloc(feats, sizeof(float) * (size_t)(n_feats + (long)ni * el));
        memcpy(feats + n_feats, f, sizeof(float) * (size_t)ni * el);
        n_feats += (long)ni * el;
        n_img_tok += ni;
        sgl_xfree(f);
      }
      sb_putjson(&t, txt.buf ? txt.buf : "");
      sgl_xfree(txt.buf);
    } else if (cj && cj->type == SGL_JSON_STRING) {
      sb_putjson(&t, cj->str ? cj->str : "");
    } else {
      sb_putjson(&t, "");
    }
    /* 其余字段（tool_calls / name / tool_call_id …）原样保留 */
    for (int k = 0; k < m->count; k++) {
      const char *kn = m->keys[k];
      if (!strcmp(kn, "role") || !strcmp(kn, "content")) continue;
      sb_puts(&t, ",");
      sb_putjson(&t, kn);
      sb_puts(&t, ":");
      dump_json(&t, m->items[k]);
    }
    sb_puts(&t, "}");
  }
  sb_puts(&t, "]");

  *feats_out = feats;
  *n_feats_out = n_feats;
  *feat_row_out = feat_row;
  *n_img_tok_out = n_img_tok;
  *err = myerr;
  if (myerr) {
    sgl_xfree(t.buf);
    return NULL;
  }

  /* kwargs：chat_template_kwargs 展开到顶层 + tools（Qwen3 的 enable_thinking 等走这里）*/
  sbuf kw;
  sb_init(&kw);
  sb_puts(&kw, "{");
  int first = 1;
  const sgl_json *ck = sgl_json_get(reqj, "chat_template_kwargs");
  if (ck && ck->type == SGL_JSON_OBJECT)
    for (int i = 0; i < ck->count; i++) {
      if (!first) sb_puts(&kw, ",");
      first = 0;
      sb_putjson(&kw, ck->keys[i]);
      sb_puts(&kw, ":");
      dump_json(&kw, ck->items[i]);
    }
  const sgl_json *tools = sgl_json_get(reqj, "tools");
  if (tools) {
    if (!first) sb_puts(&kw, ",");
    sb_puts(&kw, "\"tools\":");
    dump_json(&kw, tools);
  }
  sb_puts(&kw, "}");

  sgl_json *m2 = sgl_json_parse(t.buf, t.len);
  sgl_json *kwj = sgl_json_parse(kw.buf, kw.len);
  char *out = NULL;
  size_t outlen = 0;
  int rc = (m2 && srv->jinja) ? sgl_jinja_render(srv->jinja, m2, 1, kwj, &out, &outlen) : -1;
  if (rc != 0) {
    *err = "chat template render failed";
    out = NULL;
  }
  if (m2) sgl_json_free(m2);
  if (kwj) sgl_json_free(kwj);
  sgl_xfree(kw.buf);
  sgl_xfree(t.buf);
  return out;
}

/* ---------- P2: /v1/embeddings /v1/rerank（BERT 类编码器）---------- */

/* 提取 embedding 输入：string 或 string[]（也接受 [ [1,2] ] 的 token 数组，按空格拼接走文本）*/
static void handle_embeddings(sgl_http_server *srv, SOCKET fd, const sgl_json *j) {
  if (!srv->embed) {
    respond_error(fd, 400, "no embedding model loaded (start with --embedding-model DIR)");
    return;
  }
  const sgl_json *in = sgl_json_get(j, "input");
  if (!in) {
    respond_error(fd, 400, "\"input\" required");
    return;
  }
  int n = (in->type == SGL_JSON_ARRAY) ? in->count : 1;
  int E = sgl_bert_hidden(srv->embed);
  float *buf = (float *)sgl_xmalloc(sizeof(float) * (size_t)(E > 0 ? E : 1));
  sbuf b;
  sb_init(&b);
  sb_puts(&b, "{\"object\":\"list\",\"data\":[");
  int emitted = 0;
  for (int i = 0; i < n; i++) {
    const sgl_json *it = (in->type == SGL_JSON_ARRAY) ? in->items[i] : in;
    const char *txt = NULL;
    if (it->type == SGL_JSON_STRING) {
      txt = it->str;
    } else {
      continue; /* token 数组：BERT 词表不通用，直接忽略 */
    }
    if (sgl_bert_encode(srv->embed, txt ? txt : "", SGL_BERT_POOL_MEAN, buf, E) != 0) {
      sgl_xfree(buf);
      sgl_xfree(b.buf);
      respond_error(fd, 500, "embedding forward failed");
      return;
    }
    if (emitted) sb_puts(&b, ",");
    emitted++;
    char head[96];
    snprintf(head, sizeof(head), "{\"object\":\"embedding\",\"index\":%d,\"embedding\":[", i);
    sb_puts(&b, head);
    for (int k = 0; k < E; k++) {
      char t2[40];
      snprintf(t2, sizeof(t2), "%s%.6f", k ? "," : "", (double)buf[k]);
      sb_puts(&b, t2);
    }
    sb_puts(&b, "]}");
  }
  sb_puts(&b, "],\"model\":\"");
  sb_puts(&b, srv->model_name[0] ? srv->model_name : "sglang-c");
  sb_puts(&b, "\",\"usage\":{\"prompt_tokens\":0,\"total_tokens\":0}}");
  respond(fd, 200, "OK", b.buf);
  sgl_xfree(b.buf);
  sgl_xfree(buf);
}

static void handle_rerank(sgl_http_server *srv, SOCKET fd, const sgl_json *j) {
  if (!srv->embed) {
    respond_error(fd, 400, "no rerank model loaded (start with --embedding-model DIR)");
    return;
  }
  const char *q = sgl_json_get_str(j, "query", sgl_json_get_str(j, "text_1", NULL));
  const sgl_json *docs = sgl_json_get(j, "documents");
  const char *d1 = sgl_json_get_str(j, "text_2", NULL);
  if (!q || (!docs && !d1)) {
    respond_error(fd, 400, "\"query\" and \"documents\" required");
    return;
  }
  int n = docs ? (docs->type == SGL_JSON_ARRAY ? docs->count : 1) : 1;
  sbuf b;
  sb_init(&b);
  sb_puts(&b, "{\"object\":\"list\",\"results\":[");
  for (int i = 0; i < n; i++) {
    const char *d = NULL;
    if (d1) d = d1;
    else {
      const sgl_json *it = docs->type == SGL_JSON_ARRAY ? docs->items[i] : docs;
      if (it->type == SGL_JSON_STRING) d = it->str;
    }
    if (!d) continue;
    float sc = sgl_bert_rerank(srv->embed, q, d);
    if (i) sb_puts(&b, ",");
    char t2[160];
    snprintf(t2, sizeof(t2), "{\"index\":%d,\"relevance_score\":%.6f}", i, (double)sc);
    sb_puts(&b, t2);
  }
  sb_puts(&b, "],\"model\":\"");
  sb_puts(&b, srv->model_name[0] ? srv->model_name : "sglang-c");
  sb_puts(&b, "\",\"usage\":{\"prompt_tokens\":0,\"total_tokens\":0}}");
  respond(fd, 200, "OK", b.buf);
  sgl_xfree(b.buf);
}

/* P2: /v1/score — 两段文本的 embedding 余弦相似度（对应上游 /v1/score）*/
static void handle_score(sgl_http_server *srv, SOCKET fd, const sgl_json *j) {
  if (!srv->embed) {
    respond_error(fd, 400, "no embedding model loaded (start with --embedding-model DIR)");
    return;
  }
  const char *t1 = sgl_json_get_str(j, "text_1", sgl_json_get_str(j, "query", NULL));
  const char *t2 = sgl_json_get_str(j, "text_2", sgl_json_get_str(j, "document", NULL));
  if (!t1 || !t2) {
    respond_error(fd, 400, "\"text_1\" and \"text_2\" required");
    return;
  }
  int E = sgl_bert_hidden(srv->embed);
  float *a = (float *)sgl_xmalloc(sizeof(float) * (size_t)E);
  float *b2 = (float *)sgl_xmalloc(sizeof(float) * (size_t)E);
  if (sgl_bert_encode(srv->embed, t1, SGL_BERT_POOL_MEAN, a, E) != 0 ||
      sgl_bert_encode(srv->embed, t2, SGL_BERT_POOL_MEAN, b2, E) != 0) {
    sgl_xfree(a);
    sgl_xfree(b2);
    respond_error(fd, 500, "embedding forward failed");
    return;
  }
  double dot = 0, na = 0, nb = 0;
  for (int k = 0; k < E; k++) {
    dot += (double)a[k] * b2[k];
    na += (double)a[k] * a[k];
    nb += (double)b2[k] * b2[k];
  }
  double cos = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
  sbuf b;
  sb_init(&b);
  char t[256];
  snprintf(t, sizeof(t),
           "{\"object\":\"list\",\"data\":[{\"index\":0,\"score\":%.6f}],\"usage\":{"
           "\"prompt_tokens\":0,\"total_tokens\":0}}",
           cos);
  sb_puts(&b, t);
  respond(fd, 200, "OK", b.buf);
  sgl_xfree(b.buf);
  sgl_xfree(a);
  sgl_xfree(b2);
}

static void route(sgl_http_server *srv, SOCKET fd, const http_req *req) {
  /* ---------- 只读/管理端点（P1，与上游 sglang 同名）---------- */
  if (strcmp(req->method, "GET") == 0) {
    if (!strcmp(req->path, "/health")) {
      sgl_sched_stats st = sgl_sched_get_stats(srv->sched);
      sbuf b;
      sb_init(&b);
      char t[512];
      snprintf(t, sizeof(t),
               "{\"ok\":true,\"model\":\"%s\",\"n_running\":%d,\"n_waiting\":%d,"
               "\"prefill_tokens\":%ld,\"decode_tokens\":%ld,\"cache_hit_tokens\":%ld}",
               srv->model_name[0] ? srv->model_name : "sglang-c", st.n_running, st.n_waiting,
               st.prefill_tokens, st.decode_tokens, st.cache_hit_tokens);
      sb_puts(&b, t);
      respond(fd, 200, "OK", b.buf);
      sgl_xfree(b.buf);
      return;
    }
    if (!strcmp(req->path, "/ready") || !strcmp(req->path, "/health_generate") ||
        !strcmp(req->path, "/ping")) {
      respond(fd, 200, "OK", "{\"ok\":true}");
      return;
    }
    if (!strcmp(req->path, "/v1/models")) {
      sbuf b;
      sb_init(&b);
      sb_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":\"");
      sb_puts(&b, srv->model_name[0] ? srv->model_name : "sglang-c");
      sb_puts(&b, "\",\"object\":\"model\",\"owned_by\":\"sglang-c\"}]}");
      respond(fd, 200, "OK", b.buf);
      sgl_xfree(b.buf);
      return;
    }
    if (!strcmp(req->path, "/get_server_info") || !strcmp(req->path, "/server_info") ||
        !strcmp(req->path, "/get_model_info")) {
      sgl_sched_stats st = sgl_sched_get_stats(srv->sched);
      sbuf b;
      sb_init(&b);
      char t[768];
      snprintf(t, sizeof(t),
               "{\"model\":\"%s\",\"prefill_chunk\":%d,\"pool_pages\":%d,\"pool_free\":%d,"
               "\"n_running\":%d,\"n_waiting\":%d,\"prefill_tokens\":%ld,\"decode_tokens\":%ld,"
               "\"cache_hit_tokens\":%ld,\"n_finished\":%d,\"n_preempt\":%ld,\"dp_replicas\":%d}",
               srv->model_name[0] ? srv->model_name : "sglang-c", st.prefill_chunk, st.pool_pages,
               st.pool_free, st.n_running, st.n_waiting, st.prefill_tokens, st.decode_tokens,
               st.cache_hit_tokens, st.n_finished, st.n_preempt, st.n_replicas);
      sb_puts(&b, t);
      respond(fd, 200, "OK", b.buf);
      sgl_xfree(b.buf);
      return;
    }
    if (!strcmp(req->path, "/metrics")) {
      sgl_sched_stats st = sgl_sched_get_stats(srv->sched);
      sbuf b;
      sb_init(&b);
      char t[1024];
      snprintf(t, sizeof(t),
               "# TYPE sgl_prefill_tokens_total counter\nsgl_prefill_tokens_total %ld\n"
               "# TYPE sgl_decode_tokens_total counter\nsgl_decode_tokens_total %ld\n"
               "# TYPE sgl_cache_hit_tokens_total counter\nsgl_cache_hit_tokens_total %ld\n"
               "# TYPE sgl_requests_finished_total counter\nsgl_requests_finished_total %d\n"
               "# TYPE sgl_requests_running gauge\nsgl_requests_running %d\n"
               "# TYPE sgl_requests_waiting gauge\nsgl_requests_waiting %d\n"
               "# TYPE sgl_kv_pool_pages gauge\nsgl_kv_pool_pages %d\n"
               "# TYPE sgl_kv_pool_free gauge\nsgl_kv_pool_free %d\n"
               "# TYPE sgl_preemptions_total counter\nsgl_preemptions_total %ld\n",
               st.prefill_tokens, st.decode_tokens, st.cache_hit_tokens, st.n_finished, st.n_running,
               st.n_waiting, st.pool_pages, st.pool_free, st.n_preempt);
      sb_puts(&b, t);
      respond_text(fd, 200, "OK", b.buf);
      sgl_xfree(b.buf);
      return;
    }
  }
  if (strcmp(req->method, "POST") == 0 &&
      (!strcmp(req->path, "/v1/embeddings") || !strcmp(req->path, "/v1/rerank") ||
       !strcmp(req->path, "/v1/score"))) {
    sgl_json *ej = sgl_json_parse(req->body, req->body_len);
    if (!ej) {
      respond_error(fd, 400, "invalid JSON body");
      return;
    }
    if (!strcmp(req->path, "/v1/embeddings")) handle_embeddings(srv, fd, ej);
    else if (!strcmp(req->path, "/v1/rerank")) handle_rerank(srv, fd, ej);
    else handle_score(srv, fd, ej);
    sgl_json_free(ej);
    return;
  }
  if (strcmp(req->method, "POST") == 0 && !strcmp(req->path, "/update_weights_from_disk")) {
    sgl_json *uj = sgl_json_parse(req->body, req->body_len);
    const char *mp = uj ? sgl_json_get_str(uj, "model_path", NULL) : NULL;
    int inst = uj ? sgl_json_get_int(uj, "instance", -1) : -1;
    int rc = mp ? sgl_sched_reload_weights(srv->sched, inst, mp) : -1;
    if (uj) sgl_json_free(uj);
    if (rc == 0) respond(fd, 200, "OK", "{\"ok\":true}");
    else if (rc == -2) respond_error(fd, 409, "instance busy (wait for in-flight requests)");
    else respond_error(fd, 400, "reload failed (bad model_path or load error)");
    return;
  }
  if (strcmp(req->method, "POST") == 0 && !strcmp(req->path, "/flush_cache")) {
    sgl_sched_flush_cache(srv->sched);
    respond(fd, 200, "OK", "{\"ok\":true}");
    return;
  }
  if (strcmp(req->method, "POST") == 0 && !strcmp(req->path, "/abort_request")) {
    sgl_json *aj = sgl_json_parse(req->body, req->body_len);
    int rid = aj ? sgl_json_get_int(aj, "rid", sgl_json_get_int(aj, "request_id", -1)) : -1;
    int rc = sgl_sched_abort(srv->sched, rid);
    if (aj) sgl_json_free(aj);
    if (rc == 0) respond(fd, 200, "OK", "{\"ok\":true}");
    else respond_error(fd, 404, "no such rid");
    return;
  }
  if (strcmp(req->method, "POST") != 0 ||
      (strcmp(req->path, "/generate") != 0 && strcmp(req->path, "/v1/completions") != 0 &&
       strcmp(req->path, "/v1/chat/completions") != 0)) {
    respond_error(fd, 404,
                  "only GET /health /ready /v1/models /get_server_info /metrics, POST /generate "
                  "/v1/completions /v1/chat/completions /v1/embeddings /v1/rerank /v1/score "
                  "/flush_cache /abort_request /update_weights_from_disk");
    return;
  }
  int chat = strcmp(req->path, "/v1/chat/completions") == 0;
  int is_stream = 0;
  char stops[4][128];
  int n_stops = 0;
  int n_choices = 1;
  /* G8: 多模态（仅 chat 路径会填）*/
  float *feats = NULL;
  long n_feats = 0; /* 已拼的 float 个数（只为日志）*/
  int feat_row = 0, n_img_tok = 0;
  const char *img_err = NULL;
  (void)n_feats;

  sgl_json *j = sgl_json_parse(req->body, req->body_len);
  if (!j) { respond_error(fd, 400, "invalid JSON body"); return; }

  /* 可选顶层字段 "lora":"name"：路由到启动时 --lora 加载的 adapter 实例
   * （/generate 与 /v1/chat/completions 及 /v1/completions 均支持；未知名 → 400）*/
  char lora_name[128] = "";
  const char *lora_req = sgl_json_get_str(j, "lora", NULL);
  if (lora_req && *lora_req) {
    if (sgl_sched_find_lora(srv->sched, lora_req) < 0) {
      char msg[192];
      snprintf(msg, sizeof(msg),
               "unknown lora '%s' (load adapters at startup with --lora name=dir)", lora_req);
      sgl_json_free(j);
      respond_error(fd, 400, msg);
      return;
    }
    snprintf(lora_name, sizeof(lora_name), "%s", lora_req);
  }
  const char *lor = lora_name[0] ? lora_name : NULL;

  const char *text = NULL;
  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp));
  int max_new = 16;
  int want_lp = 0; /* P0-3: logprobs */
  int *prompt_ids = NULL; /* P0-3: prompt/input_ids 直接给 token */
  int n_prompt_ids = 0;
  sgl_grammar *gram = NULL; /* P1: 约束解码 */
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
    if (srv->jinja) {
      /* P0-2: 模型自带 Jinja 模板（含 tools / chat_template_kwargs）*/
      chatml = chat_render_template(srv, j, msgs, &feats, &n_feats, &feat_row, &n_img_tok, &img_err);
      if (!chatml) {
        sgl_json_free(j);
        sgl_xfree(feats);
        respond_error(fd, 400, img_err ? img_err : "chat template render failed");
        return;
      }
    } else {
    /* 回退：内置 ChatML（Qwen 系默认格式）*/
    sbuf ml;
    sb_init(&ml);
    /* G8: content[] 里的 image_url → 视觉 token（多张图按顺序拼接）
     * feats/n_img_tok/img_err 在函数开头声明，提交阶段要用 */
    for (int i = 0; i < msgs->count; i++) {
      const sgl_json *m = msgs->items[i];
      const char *role = sgl_json_get_str(m, "role", "user");
      const sgl_json *cj = sgl_json_get(m, "content");
      int is_arr = cj && cj->type == SGL_JSON_ARRAY;
      sb_puts(&ml, "<|im_start|>");
      sb_puts(&ml, role);
      sb_puts(&ml, "\n");
      if (is_arr) {
        for (int k = 0; k < cj->count; k++) {
          const sgl_json *part = cj->items[k];
          const char *ptype = sgl_json_get_str(part, "type", "text");
          if (strcmp(ptype, "image_url") != 0) {
            sb_puts(&ml, sgl_json_get_str(part, "text", ""));
            continue;
          }
          sb_puts(&ml, "<image>"); /* 文本里放占位，真融合走 emb */
          if (img_err) continue;
          if (!srv->vision) { img_err = "server has no vision tower (use --vision)"; continue; }
          const sgl_json *iu = sgl_json_get(part, "image_url");
          const char *url = iu ? sgl_json_get_str(iu, "url", "") : "";
          const char *b64 = url;
          const char *cm = strchr(url, ',');
          if (!strncmp(url, "data:", 5) && cm) b64 = cm + 1; /* data:image/x;base64, */
          unsigned char *bytes = NULL;
          size_t bn = 0;
          if (sgl_base64_decode(b64, strlen(b64), &bytes, &bn) != 0) {
            img_err = "image_url base64 decode failed";
            continue;
          }
          int ni = 0, el = 0;
          float *f = srv->vision(srv->vision_ud, bytes, bn, &ni, &el);
          sgl_xfree(bytes);
          if (!f || ni <= 0 || el <= 0) {
            sgl_xfree(f);
            img_err = "image decode / vision encode failed";
            continue;
          }
          if (feat_row == 0) feat_row = el;
          if (feat_row != el) {
            sgl_xfree(f);
            img_err = "images with different feature width";
            continue;
          }
          feats = (float *)sgl_xrealloc(feats, sizeof(float) * (size_t)(n_feats + (long)ni * el));
          memcpy(feats + n_feats, f, sizeof(float) * (size_t)ni * el);
          n_feats += (long)ni * el;
          n_img_tok += ni;
          sgl_xfree(f);
        }
      } else {
        sb_puts(&ml, sgl_json_get_str(m, "content", ""));
      }
      sb_puts(&ml, "<|im_end|>\n");
    }
    sb_puts(&ml, "<|im_start|>assistant\n");
    chatml = ml.buf;
    }
    parse_sampling_json(j, &sp, &max_new, 256, 4096);
    want_lp = json_bool(j, "logprobs") || sgl_json_get_int(j, "top_logprobs", 0) > 0;
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
    /* P0-3: prompt 也可以是 token id 数组 */
    const sgl_json *pj = sgl_json_get(j, "prompt");
    if (pj && pj->type == SGL_JSON_ARRAY) {
      prompt_ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)(pj->count > 0 ? pj->count : 1));
      for (int k = 0; k < pj->count; k++)
        if (pj->items[k]->type == SGL_JSON_NUMBER) prompt_ids[n_prompt_ids++] = (int)pj->items[k]->num;
      if (n_prompt_ids > 0) text = "";
    }
    parse_sampling_json(j, &sp, &max_new, 16, 1024);
    want_lp = json_bool(j, "logprobs") || sgl_json_get_int(j, "logprobs", 0) > 0;
    parse_stops(j, stops, &n_stops);
    n_choices = sgl_json_get_int(j, "n", 1);
    if (n_choices < 1) n_choices = 1;
    if (n_choices > 4) n_choices = 4;
    if (max_new <= 0 || max_new > 1024) max_new = 16;
  } else {
    text = sgl_json_get_str(j, "text", NULL);
    /* P0-3: sglang 原生 /generate 支持 input_ids 直接给 token */
    const sgl_json *ij = sgl_json_get(j, "input_ids");
    if (ij && ij->type == SGL_JSON_ARRAY) {
      prompt_ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)(ij->count > 0 ? ij->count : 1));
      for (int k = 0; k < ij->count; k++)
        if (ij->items[k]->type == SGL_JSON_NUMBER) prompt_ids[n_prompt_ids++] = (int)ij->items[k]->num;
      if (n_prompt_ids > 0) text = "";
    }
    const sgl_json *spj = sgl_json_get(j, "sampling_params");
    if (spj) parse_sampling_json(spj, &sp, &max_new, 16, 1024);
    if (max_new <= 0 || max_new > 1024) max_new = 16;
  }
  if ((!text || !*text) && n_prompt_ids <= 0) {
    sgl_json_free(j);
    respond_error(fd, 400, openai ? "\"prompt\"" : "\"text\" is required");
    return;
  }
  /* text 指向 json 树内部：立刻复制，避免下面的 json_free 变成悬垂指针；
   * chat 路径的 chatml 已是自有内存且 j 已释放。*/
  char *prompt_copy;
  if (chat) {
    prompt_copy = chatml;
  } else if (n_prompt_ids > 0) {
      prompt_copy = NULL; /* 不再需要文本 */
      sgl_json_free(j);
  } else {
      size_t plen = strlen(text) + 1;
      char *pc = (char *)sgl_xmalloc(plen);
      memcpy(pc, text, plen);
      prompt_copy = pc;
      sgl_json_free(j);
  }
  text = prompt_copy;

  /* P1: 结构化输出约束（response_format / grammar）*/
  if (chat) {
    const sgl_json *rf = sgl_json_get(j, "response_format");
    const sgl_json *gs = NULL;
    int want = 0;
    if (rf && rf->type == SGL_JSON_OBJECT) {
      const char *rt = sgl_json_get_str(rf, "type", "text");
      if (!strcmp(rt, "json_object")) {
        want = 1;
        gram = sgl_grammar_from_json_schema("{\"type\":\"object\"}", 19);
      } else if (!strcmp(rt, "json_schema")) {
        const sgl_json *js = sgl_json_get(rf, "json_schema");
        gs = js ? sgl_json_get(js, "schema") : NULL;
        want = 1;
        if (gs) {
          sbuf sj;
          sb_init(&sj);
          dump_json(&sj, gs);
          gram = sgl_grammar_from_json_schema(sj.buf, sj.len);
          sgl_xfree(sj.buf);
        }
      } else if (!strcmp(rt, "structural_tag") || !strcmp(rt, "regex")) {
        const char *pat = sgl_json_get_str(rf, "regex", sgl_json_get_str(rf, "pattern", NULL));
        want = 1;
        if (pat) gram = sgl_grammar_from_regex(pat);
      }
    }
    if (want && !gram) {
      sgl_xfree(prompt_copy);
      sgl_xfree(feats);
      respond_error(fd, 400, "response_format not supported by grammar engine");
      return;
    }
  }

  /* t11: same pathological-prompt guard as cuda_http — a huge single-char
   * body parks BPE encode for minutes on the CPU stack too. */
  if (text && strlen(text) > 65536) {
    sgl_xfree(prompt_copy);
    sgl_xfree(feats);
    respond_error(fd, 503, "prompt too long (>64KB)");
    return;
  }

  /* G8: 图片请求——emb 级融合只在非流式、n=1、base 实例下接线（其余组合明确拒）*/
  if (img_err) {
    sgl_xfree(prompt_copy);
    sgl_xfree(feats);
    respond_error(fd, 400, img_err);
    return;
  }
  if (feats && (is_stream || n_choices > 1 || (lor && *lor))) {
    sgl_xfree(prompt_copy);
    sgl_xfree(feats);
    respond_error(fd, 400, "image_url supports only non-streaming n=1 on the base instance");
    return;
  }

  /* n 个请求（n>1 时各自加 seed 偏移，temperature>0 才有差异）*/
  int ids[4] = {-1, -1, -1, -1};
  pthread_mutex_lock(&srv->lock);
  for (int i = 0; i < n_choices; i++) {
    sgl_sampling_params spi = sp;
    if (i > 0) spi.seed = sp.seed ? sp.seed + (uint64_t)i : (uint64_t)(0x1234 + i);
    ids[i] = is_stream
        ? sgl_sched_submit_stream_lora(srv->sched, text, max_new, &spi, lor)
        : feats ? sgl_sched_submit_img(srv->sched, text, max_new, &spi, feats, n_img_tok)
        : gram ? sgl_sched_submit_grammar(srv->sched, text, max_new, &spi, lor, gram)
        : n_prompt_ids > 0 ? sgl_sched_submit_ids(srv->sched, prompt_ids, n_prompt_ids, max_new, &spi, lor)
                : sgl_sched_submit_lora(srv->sched, text, max_new, &spi, lor);
    if (gram) gram = NULL; /* 所有权已转给调度器 */
    if (feats) feats = NULL; /* 所有权已转给调度器（失败时 submit_img 自己回收）*/
    if (ids[i] < 0) break;
  }
  pthread_mutex_unlock(&srv->lock);
  sgl_xfree(prompt_ids);
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
        sgl_sleep_ms(25); /* MORE：等下一批 */
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
      sb_puts(&b, ",\"message\":{\"role\":\"assistant\"");
      /* P2: reasoning 分离（DeepSeek-R1 / Qwen3），正文里如果有 <tool_call> 就解成 tool_calls */
      {
        char *rtext = NULL;
        const char *body = NULL;
        int had_reason = sgl_parse_reasoning(ds[i]->text ? ds[i]->text : "", &rtext, &body, NULL);
        int n_calls = 0;
        char *calls = sgl_parse_tool_calls(body ? body : "", &n_calls);
        const char *visible = body ? body : "";
        if (n_calls > 0) {
          /* 去掉 <tool_call> 段，只留前面的叙述文字（OpenAI 的行为）*/
          const char *p = strstr(visible, "<tool_call>");
          size_t keep = p ? (size_t)(p - visible) : strlen(visible);
          char *trimmed = (char *)sgl_xmalloc(keep + 1);
          memcpy(trimmed, visible, keep);
          trimmed[keep] = 0;
          sb_puts(&b, ",\"content\":");
          sb_putjson(&b, trimmed[0] ? trimmed : "");
          sgl_xfree(trimmed);
        } else {
          sb_puts(&b, ",\"content\":");
          sb_putjson(&b, visible);
        }
        if (had_reason && rtext) {
          sb_puts(&b, ",\"reasoning_content\":");
          sb_putjson(&b, rtext);
        }
        if (n_calls > 0) {
          /* OpenAI tool_calls 形状：id/type/function{name,arguments}；这里把解析出的
           * JSON 对象原样作为 function.arguments（它对模型输出是权威形式）。*/
          sb_puts(&b, ",\"tool_calls\":[");
          sgl_json *arr = sgl_json_parse(calls, strlen(calls));
          int nc = (arr && arr->type == SGL_JSON_ARRAY) ? arr->count : 0;
          for (int k = 0; k < nc; k++) {
            const sgl_json *o = arr->items[k];
            const char *nm = o->type == SGL_JSON_OBJECT ? sgl_json_get_str(o, "name", "") : "";
            const sgl_json *args = o->type == SGL_JSON_OBJECT ? sgl_json_get(o, "arguments") : NULL;
            if (k) sb_puts(&b, ",");
            char th[160];
            snprintf(th, sizeof(th),
                     "{\"id\":\"call_%d_%d\",\"type\":\"function\",\"function\":{\"name\":", i, k);
            sb_puts(&b, th);
            sb_putjson(&b, nm);
            sb_puts(&b, ",\"arguments\":");
            if (args) {
              sbuf ab;
              sb_init(&ab);
              dump_json(&ab, args);
              sb_putjson(&b, ab.buf ? ab.buf : "{}");
              sgl_xfree(ab.buf);
            } else {
              sb_putjson(&b, "{}");
            }
            sb_puts(&b, "}}");
          }
          if (arr) sgl_json_free(arr);
          sb_puts(&b, "]");
        }
        sgl_xfree(calls);
        sgl_xfree(rtext);
        /* finish_reason 统一由下面（logprobs 之后）输出；有 tool_call 时要报 tool_calls */
        if (n_calls > 0) hit_stop[i] = 2;
        sb_puts(&b, "}");
      }
    choice_done:;
      /* P0-3: logprobs（OpenAI chat 格式：choices[].logprobs.content[]）*/
      if (want_lp && ds[i]->lp && ds[i]->n_lp > 0) {
        sb_puts(&b, ",\"logprobs\":{\"content\":[");
        for (int k = 0; k < ds[i]->n_lp; k++) {
          if (k) sb_puts(&b, ",");
          char t2[160];
          snprintf(t2, sizeof(t2), "{\"token\":\"\",\"logprob\":%.6f}", (double)ds[i]->lp[k]);
          sb_puts(&b, t2);
        }
        sb_puts(&b, "]}");
      }
      sb_puts(&b, hit_stop[i] == 2 ? ",\"finish_reason\":\"tool_calls\"}"
                                   : hit_stop[i] ? ",\"finish_reason\":\"stop\"}"
                                                : ",\"finish_reason\":\"length\"}");
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
    snprintf(head, sizeof(head),
             "{\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":\"%s\",\"choices\":[",
             idbuf, created, srv->model_name[0] ? srv->model_name : "sglang-c");
    sb_puts(&b, head);
    for (int i = 0; i < n_done; i++) {
      if (i) sb_puts(&b, ",");
      sb_puts(&b, "{\"text\":");
      sb_putjson(&b, ds[i]->text ? ds[i]->text : "");
      char ct[96];
      snprintf(ct, sizeof(ct), ",\"index\":%d", i);
      sb_puts(&b, ct);
      if (want_lp && ds[i]->lp && ds[i]->n_lp > 0) {
        /* P0-3: OpenAI completions 的 logprobs（token_logprobs 数组）*/
        sb_puts(&b, ",\"logprobs\":{\"tokens\":[],\"token_logprobs\":[");
        for (int k = 0; k < ds[i]->n_lp; k++) {
          char t2[48];
          snprintf(t2, sizeof(t2), "%s%.6f", k ? "," : "", (double)ds[i]->lp[k]);
          sb_puts(&b, t2);
        }
        sb_puts(&b, "]}");
      }
      snprintf(ct, sizeof(ct), ",\"finish_reason\":\"%s\"}", hit_stop[i] ? "stop" : "length");
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
    sgl_xfree(ds[i]->lp);
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
  /* t11: disable Nagle — head+body split sends interact with delayed-ACK
   * into a fixed ~21-24s stall per response (2x RTO) on loopback. */
  {
    int nd = 1;
    setsockopt(ctx->fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&nd, sizeof(nd));
#ifdef _WIN32
    DWORD rto = 30000;
#else
    struct timeval rto = {30, 0};
#endif
    setsockopt(ctx->fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&rto, sizeof(rto));
    setsockopt(ctx->fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&rto, sizeof(rto));
  }
  if (read_request(ctx->fd, &req) == 0) {
    route(ctx->srv, ctx->fd, &req);
    sgl_xfree(req.body);
  }
  closesocket(ctx->fd);
  pthread_mutex_lock(&ctx->srv->conns_lock);
  if (--ctx->srv->active_conns == 0) pthread_cond_broadcast(&ctx->srv->cv_conns_zero);
  pthread_mutex_unlock(&ctx->srv->conns_lock);
  sgl_xfree(ctx);
  return NULL;
}

static void *acceptor_fn(void *arg) {
  sgl_http_server *srv = (sgl_http_server *)arg;
  for (;;) {
    struct sockaddr_in peer;
    int plen = sizeof(peer);
    SOCKET c = accept(srv->listen_fd, (struct sockaddr *)&peer, &plen);
    if (c == INVALID_SOCKET) {
      /* t11: transient accept errors (burst) used to kill the acceptor */
#ifdef _WIN32
      int e = WSAGetLastError();
      if (e == WSAECONNABORTED || e == WSAEINTR || e == WSAEMFILE) continue;
#else
      if (errno == ECONNABORTED || errno == EINTR) continue;
#endif
      break; /* listen socket 关闭（stop）*/
    }
    if (srv->stop) { closesocket(c); break; }
    pthread_mutex_lock(&srv->conns_lock);
    srv->active_conns++;
    pthread_mutex_unlock(&srv->conns_lock);
    conn_ctx *ctx = (conn_ctx *)sgl_xmalloc(sizeof(*ctx));
    ctx->srv = srv; ctx->fd = c;
    pthread_t t;
    if (pthread_create(&t, NULL, conn_fn, ctx) != 0) {
      closesocket(c);
      pthread_mutex_lock(&srv->conns_lock);
      srv->active_conns--;
      pthread_mutex_unlock(&srv->conns_lock);
      sgl_xfree(ctx);
    } else {
      pthread_detach(t);
    }
  }
  return NULL;
}

/* ---------- 生命周期 ---------- */

sgl_http_server *sgl_http_serve(sgl_scheduler *sched, const char *host, int port) {
  sgl_sock_init();

  sgl_http_server *srv = (sgl_http_server *)sgl_xcalloc(1, sizeof(*srv));
  srv->sched = sched;
  srv->listen_fd = INVALID_SOCKET;
  pthread_mutex_init(&srv->lock, NULL);
  pthread_mutex_init(&srv->conns_lock, NULL); /* G4 */
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
  pthread_mutex_destroy(&srv->lock); pthread_mutex_destroy(&srv->conns_lock);
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
    sgl_sleep_ms(100);
    pthread_mutex_lock(&srv->lock);
    /* G4: 计数归 conns_lock 管，隔着 srv->lock 看不算数 */
    pthread_mutex_lock(&srv->conns_lock);
    int still = srv->active_conns;
    pthread_mutex_unlock(&srv->conns_lock);
    if (still == 0) break;
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
  pthread_mutex_destroy(&srv->lock); pthread_mutex_destroy(&srv->conns_lock);
  sgl_xfree(srv);
  WSACleanup();
}
