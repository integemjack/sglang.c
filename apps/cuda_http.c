/* sgl cuda_http.c - GPU HTTP inference server (MSVC build, links sgl_cuda)
 * endpoints: GET /health, POST /generate, POST /v1/completions,
 *            POST /v1/chat/completions (chatml template; stream via SSE)
 * engine: KV-segment scheduler on GPU - each request gets a segment of
 *         g->ctx (seg = ctx / MAX_CONC), worker batches decode via
 *         sgl_cuda_decode_batch; prefill via sgl_cuda_forward_seg.
 * usage: cuda_http.exe <model_dir> [port]
 */
#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/cuda.h"
#include "sgl/json.h"
#include "sgl/llama.h"
#include "sgl/sampling.h"

#pragma comment(lib, "ws2_32.lib")

#define MAX_CONC 8
#define MAX_NEW_CAP 512

static int ACTIVE_CONC = 8; /* effective concurrency, SGL_HTTP_CONC overrides */
typedef enum { R_WAIT = 0, R_RUN, R_DONE, R_FAIL, R_REAPED } rstate;

typedef struct request {
  int id;
  rstate state;
  int *prompt;
  int n_prompt;
  int gen[MAX_NEW_CAP];
  int n_gen;
  int max_new;
  sgl_sampling_params sp;
  uint64_t rng;
  int seg;             /* kv segment index, -1 until assigned */
  int seg_len;         /* segment capacity in positions */
  float *logits;       /* host buffer vocab floats (last step) */
  int has_logits;
  char *text_out;      /* decoded completion (malloc) */
  char finish[12];     /* "stop" | "length" (t5: honest finish_reason) */
  int streaming;
  int n_streamed;
  struct request *next_done;
  struct request *next_q; /* t5: wait-queue link */
} request;

typedef struct {
  sgl_llama *cpu;
  sgl_cuda_model *gpu;
  sgl_tokenizer *tok;
  int vocab;
  int eos;
  int seg_len; /* ctx / MAX_CONC */
  request *reqs[MAX_CONC];
  int n_active;
  request *waitq_head, *waitq_tail; /* t5: queue beyond ACTIVE_CONC */
  int n_waitq;
  int next_id;
  CRITICAL_SECTION lock;
  CONDITION_VARIABLE cv_work;   /* worker waits for requests */
  CONDITION_VARIABLE cv_step;   /* http threads wait for progress */
  volatile int stop;
} engine;

static engine E;

/* ---------------- request lifecycle ---------------- */

static request *req_new(const char *text, int max_new, const sgl_sampling_params *sp) {
  if (max_new <= 0 || max_new > MAX_NEW_CAP) max_new = 64;
  request *r = (request *)calloc(1, sizeof(request));
  if (!r) return NULL;
  r->prompt = sgl_tokenizer_encode(E.tok, text, strlen(text), &r->n_prompt);
  if (!r->prompt || r->n_prompt <= 0) { free(r); return NULL; }
  r->max_new = max_new;
  r->sp = *sp;
  r->rng = sp->seed ? sp->seed : 0x9E3779B97F4A7C15ULL ^ (uint64_t)GetTickCount64();
  r->seg = -1;
  r->logits = (float *)malloc(sizeof(float) * (size_t)E.vocab);
  return r;
}

static void req_free(request *r) {
  if (!r) return;
  sgl_xfree(r->prompt);
  free(r->logits);
  sgl_xfree(r->text_out); /* decode_gen 返回 sgl_xmalloc（对齐）——绝不能 free() */
  free(r);
}

static int submit(const char *text, int max_new, const sgl_sampling_params *sp) {
  /* t11: cap prompt length before the O(n^2)-ish BPE merge — a pathological
   * 200KB single-char body used to park encode for minutes (server hang). */
  if (!text || strlen(text) > 65536) return -1;
  /* t5: no more 503 when full — park in a wait queue; the worker drains it
   * into freed slots every pass. Hard cap only guards against unbounded
   * queue growth (128 waiting). */
  EnterCriticalSection(&E.lock);
  if (E.n_active + E.n_waitq >= 128) { LeaveCriticalSection(&E.lock); return -1; }
  request *r = req_new(text, max_new, sp);
  if (!r) { LeaveCriticalSection(&E.lock); return -1; }
  r->id = ++E.next_id;
  r->state = R_WAIT;
  int placed = 0;
  if (E.n_active < ACTIVE_CONC) {
    for (int i = 0; i < MAX_CONC; i++)
      if (!E.reqs[i]) { E.reqs[i] = r; placed = 1; break; }
    E.n_active++;
  }
  if (!placed) {
    r->next_q = NULL;
    if (E.waitq_tail) E.waitq_tail->next_q = r;
    else E.waitq_head = r;
    E.waitq_tail = r;
    E.n_waitq++;
  }
  LeaveCriticalSection(&E.lock);
  WakeConditionVariable(&E.cv_work);
  return r->id;
}

/* ---------------- decode text of r->gen ---------------- */

static char *decode_gen(request *r) {
  return sgl_tokenizer_decode(E.tok, r->gen, r->n_gen);
}

/* ---------------- worker: prefill + batched decode ---------------- */

static DWORD WINAPI worker(LPVOID unused) {
  (void)unused;
  float *batch_logits = (float *)malloc(sizeof(float) * (size_t)MAX_CONC * E.vocab);
  for (;;) {
    EnterCriticalSection(&E.lock);
    if (E.stop) { LeaveCriticalSection(&E.lock); break; }
    /* reap: free requests marked by their http thread (safe: worker no longer
     * references them after this pass begins) */
    for (int i = 0; i < MAX_CONC; i++) {
      request *r = E.reqs[i];
      if (r && r->state == R_REAPED) {
        req_free(r);
        E.reqs[i] = NULL;
        E.n_active--;
      }
    }
    /* t5: drain the wait queue into freed slots (FIFO) */
    while (E.n_active < ACTIVE_CONC && E.waitq_head) {
      request *w = E.waitq_head;
      E.waitq_head = w->next_q;
      if (!E.waitq_head) E.waitq_tail = NULL;
      E.n_waitq--;
      int placed = 0;
      for (int i = 0; i < MAX_CONC; i++)
        if (!E.reqs[i]) { E.reqs[i] = w; placed = 1; break; }
      if (placed) E.n_active++;
      else { /* slots all taken by not-yet-reaped reqs: push back front */
        w->next_q = E.waitq_head;
        E.waitq_head = w;
        if (!E.waitq_tail) E.waitq_tail = w;
        E.n_waitq++;
        break;
      }
    }
    int any = 0;
    for (int i = 0; i < MAX_CONC; i++) if (E.reqs[i]) any = 1;
    if (E.n_waitq > 0) any = 1;
    if (!any) {
      SleepConditionVariableCS(&E.cv_work, &E.lock, 50);
      LeaveCriticalSection(&E.lock);
      continue;
    }
    /* 1) prefill waiting requests (assign segments). M26a: chunked prefill —
     * forward at most seg_len tokens per call (KV writes land at base+pos,
     * so chunks just advance pos0) and reject prompts that cannot fit their
     * KV segment at all (they would overwrite the neighbour segment). */
    for (int i = 0; i < MAX_CONC; i++) {
      request *r = E.reqs[i];
      if (!r || r->state != R_WAIT) continue;
      r->seg = i;
      r->seg_len = E.seg_len;
      int base = i * E.seg_len;
      LeaveCriticalSection(&E.lock);
      int rc;
      if (r->n_prompt > r->seg_len) {
        rc = -2; /* prompt exceeds its KV segment capacity */
      } else {
        rc = 0;
        int off = 0;
        while (off < r->n_prompt && rc == 0) {
          int chunk = r->n_prompt - off;
          /* only the last chunk needs the lm_head pass + logits D2H */
          float *dst = (off + chunk >= r->n_prompt) ? r->logits : NULL;
          rc = sgl_cuda_forward_seg(E.gpu, r->prompt + off, chunk, off, base, dst);
          off += chunk;
        }
      }
      EnterCriticalSection(&E.lock);
      if (rc != 0) { r->state = R_FAIL; }
      else { r->state = R_RUN; r->has_logits = 1; }
      any = 1;
    }
    /* 2) batched decode of running requests
     * semantics per pass: sample from r->logits (available), append to gen,
     * then batch-forward the LAST token of each running seq to refresh logits. */
    request *run[MAX_CONC];
    int B = 0;
    for (int i = 0; i < MAX_CONC; i++) {
      request *r = E.reqs[i];
      if (r && r->state == R_RUN) run[B++] = r;
    }
    if (B > 0) {
      /* 2a) sample + append for those with logits (first pass after prefill) */
      for (int b = 0; b < B; b++) {
        request *r = run[b];
        if (!r->has_logits || r->state != R_RUN) continue;
        if (r->n_gen >= r->max_new) {
          r->state = R_DONE;
          strcpy_s(r->finish, sizeof(r->finish), "length");
          r->text_out = decode_gen(r);
          continue;
        }
        int t = r->sp.temperature > 0 ? sgl_sample(r->logits, E.vocab, &r->sp, &r->rng)
                                      : sgl_sample_greedy(r->logits, E.vocab);
        if (t == E.eos) {
          r->state = R_DONE;
          strcpy_s(r->finish, sizeof(r->finish), "stop");
          r->text_out = decode_gen(r);
          continue;
        }
        r->gen[r->n_gen++] = t;
      }
      /* 2b) batch forward last token of each still-running seq */
      int bt[MAX_CONC], bb[MAX_CONC], bl[MAX_CONC];
      int nb = 0;
      request *br[MAX_CONC];
      for (int b = 0; b < B; b++) {
        request *r = run[b];
        if (r->state != R_RUN) continue;
        if (r->n_prompt + r->n_gen >= r->seg_len) {
          r->state = R_DONE;
          strcpy_s(r->finish, sizeof(r->finish), "length");
          r->text_out = decode_gen(r);
          continue;
        }
        bt[nb] = r->gen[r->n_gen - 1];
        bb[nb] = r->seg * E.seg_len;
        bl[nb] = r->n_prompt + r->n_gen - 1;
        br[nb] = r;
        nb++;
      }
      if (nb > 0) {
        int rc = sgl_cuda_decode_batch(E.gpu, bt, bb, bl, nb, batch_logits);
        if (rc != 0) {
          for (int b = 0; b < nb; b++) br[b]->state = R_FAIL;
        } else {
          for (int b = 0; b < nb; b++) {
            memcpy(br[b]->logits, batch_logits + (size_t)b * E.vocab,
                   sizeof(float) * E.vocab);
            br[b]->has_logits = 1;
          }
        }
      }
      WakeAllConditionVariable(&E.cv_step);
    } else if (any) {
      /* nothing runnable: brief sleep WITHOUT holding lock */
      LeaveCriticalSection(&E.lock);
      Sleep(5);
      continue;
    }
    /* end of pass: release lock before looping (Enter at top) */
    LeaveCriticalSection(&E.lock);
  }
  free(batch_logits);
  return 0;
}

/* ---------------- http plumbing ---------------- */

typedef struct { char *body; size_t body_len; char method[8]; char path[128]; } http_req;

static int read_request(SOCKET fd, http_req *req) {
  size_t cap = 8192, len = 0;
  char *buf = (char *)malloc(cap);
  size_t hdr_end = 0;
  memset(req, 0, sizeof(*req));
  for (;;) {
    if (len >= 4)
      for (size_t i = 0; i + 3 < len; i++)
        if (buf[i] == '\r' && buf[i+1] == '\n' && buf[i+2] == '\r' && buf[i+3] == '\n') { hdr_end = i + 4; break; }
    if (hdr_end) break;
    if (len + 1024 > cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
    int n = recv(fd, buf + len, 1024, 0);
    if (n <= 0) { free(buf); return -1; }
    len += (size_t)n;
    if (len > 4 * 1024 * 1024) { free(buf); return -1; }
  }
  /* method + path */
  if (sscanf(buf, "%7s %127s", req->method, req->path) != 2) { free(buf); return -1; }
  /* content-length body */
  long cl = 0;
  const char *clh = strstr(buf, "Content-Length:");
  if (!clh) clh = strstr(buf, "content-length:");
  if (clh) cl = atol(clh + 15);
  while ((long)(len - hdr_end) < cl) {
    if (len + 1024 > cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
    int n = recv(fd, buf + len, 1024, 0);
    if (n <= 0) break;
    len += (size_t)n;
  }
  req->body = (char *)malloc(len - hdr_end + 1);
  memcpy(req->body, buf + hdr_end, len - hdr_end);
  req->body[len - hdr_end] = 0;
  req->body_len = len - hdr_end;
  free(buf);
  return 0;
}

static void json_escape(const char *s, char *out, size_t cap) {
  size_t o = 0;
  for (const unsigned char *p = (const unsigned char *)s; *p && o + 7 < cap; p++) {
    if (*p == '"' || *p == '\\') { out[o++] = '\\'; out[o++] = (char)*p; }
    else if (*p == '\n') { out[o++] = '\\'; out[o++] = 'n'; }
    else if (*p == '\r') { out[o++] = '\\'; out[o++] = 'r'; }
    else if (*p == '\t') { out[o++] = '\\'; out[o++] = 't'; }
    else if (*p < 0x20) { o += snprintf(out + o, cap - o, "\\u%04x", *p); }
    else out[o++] = (char)*p;
  }
  out[o] = 0;
}

/* t5 fix: send() on these sockets intermittently returns -1/WSAEWOULDBLOCK
 * mid-response (seen after burst load; the client then waits forever for the
 * missing body bytes). Retry short sends; bail on hard errors. */
static int send_all(SOCKET fd, const char *buf, int len) {
  int off = 0;
  int tries = 0;
  while (off < len) {
    int n = send(fd, buf + off, len - off, 0);
    if (n > 0) { off += n; tries = 0; continue; }
    int e = WSAGetLastError();
    if (e == WSAEWOULDBLOCK || e == WSAENOBUFS || e == WSAEINTR) {
      if (++tries > 2000) return -1; /* ~2s max wait in 1ms slices */
      Sleep(1);
      continue;
    }
    return -1;
  }
  return off;
}

static void respond(SOCKET fd, int code, const char *ctype, const char *body) {
  char head[256];
  int hl = snprintf(head, sizeof(head),
                    "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
                    code, code == 200 ? "OK" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" : "Error",
                    ctype, strlen(body));
  /* t5: single buffered write — a split head/body send hit intermittent
   * WOULDBLOCK between the two calls and left the client with a truncated
   * response (headers advertising bytes that never arrive) */
  int bl = (int)strlen(body);
  char *pkt = (char *)malloc((size_t)hl + (size_t)bl);
  if (pkt) {
    memcpy(pkt, head, (size_t)hl);
    memcpy(pkt + hl, body, (size_t)bl);
    send_all(fd, pkt, hl + bl);
    free(pkt);
  } else {
    send_all(fd, head, hl);
    send_all(fd, body, bl);
  }
}

static void respond_error(SOCKET fd, int code, const char *msg) {
  char body[256];
  char esc[192];
  json_escape(msg, esc, sizeof(esc));
  snprintf(body, sizeof(body), "{\"error\":{\"message\":\"%s\"}}", esc);
  respond(fd, code, "application/json", body);
}

/* build chatml prompt from messages array; returns malloc'd string */
static char *build_chatml(const sgl_json *j) {
  const sgl_json *msgs = sgl_json_get(j, "messages");
  if (!msgs || msgs->type != SGL_JSON_ARRAY || msgs->count == 0) return NULL;
  size_t cap = 1024, len = 0;
  char *s = (char *)malloc(cap);
  if (!s) return NULL;
  s[0] = 0;
#define APPEND(str)                                                    \
  do {                                                                 \
    size_t _n = strlen(str);                                           \
    while (len + _n + 1 > cap) {                                       \
      cap *= 2;                                                        \
      char *_s2 = (char *)realloc(s, cap);                             \
      if (!_s2) { free(s); return NULL; }                              \
      s = _s2;                                                         \
    }                                                                  \
    memcpy(s + len, str, _n);                                          \
    len += _n;                                                          \
    s[len] = 0;                                                        \
  } while (0)
  for (int i = 0; i < msgs->count; i++) {
    const sgl_json *m = msgs->items[i];
    const char *role = sgl_json_get_str(m, "role", "user");
    const char *content = sgl_json_get_str(m, "content", "");
    APPEND("<|im_start|>");
    APPEND(role);
    APPEND("\n");
    APPEND(content);
    APPEND("<|im_end|>\n");
  }
  APPEND("<|im_start|>assistant\n");
#undef APPEND
  return s;
}

/* find request by id (slots + t5 wait queue) */
static request *find_req(int id) {
  for (int i = 0; i < MAX_CONC; i++)
    if (E.reqs[i] && E.reqs[i]->id == id) return E.reqs[i];
  for (request *w = E.waitq_head; w; w = w->next_q)
    if (w->id == id) return w;
  return NULL;
}

/* wait until request id reaches DONE/FAIL; optionally stream SSE meanwhile */
static int wait_done(SOCKET fd, int id, char **text, request **out_r) {
  for (;;) {
    EnterCriticalSection(&E.lock);
    request *r = find_req(id);
    if (!r) { LeaveCriticalSection(&E.lock); return -1; }
    if (r->state == R_DONE) {
      /* t5 fix: fast requests reach DONE before wait_done ever entered its
       * streaming loop — flush any un-streamed tokens first so the SSE client
       * gets the full content, not just the finish chunk. */
      if (r->streaming) {
        while (r->n_streamed < r->n_gen) {
          int t = r->gen[r->n_streamed++];
          char *piece = sgl_tokenizer_decode(E.tok, &t, 1);
          char esc[512];
          json_escape(piece ? piece : "", esc, sizeof(esc));
          char chunk[1024];
          int n = snprintf(chunk, sizeof(chunk),
                           "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                           esc);
          LeaveCriticalSection(&E.lock);
          send_all(fd, chunk, n);
          EnterCriticalSection(&E.lock);
          r = find_req(id);
          if (!r) { LeaveCriticalSection(&E.lock); return -1; }
          sgl_xfree(piece);
        }
      }
      *text = r->text_out ? _strdup(r->text_out) : _strdup("");
      if (out_r) *out_r = r;
      LeaveCriticalSection(&E.lock);
      return 0;
    }
    if (r->state == R_FAIL) { LeaveCriticalSection(&E.lock); return -2; }
    /* stream out new tokens while running */
    if (r->streaming) {
      while (r->n_streamed < r->n_gen) {
        int t = r->gen[r->n_streamed++];
        char *piece = sgl_tokenizer_decode(E.tok, &t, 1);
        char esc[512];
        json_escape(piece ? piece : "", esc, sizeof(esc));
        char chunk[1024];
        int n = snprintf(chunk, sizeof(chunk),
                         "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                         esc);
        LeaveCriticalSection(&E.lock);
        send_all(fd, chunk, n);
        EnterCriticalSection(&E.lock);
        r = find_req(id);
        if (!r) { LeaveCriticalSection(&E.lock); return -1; }
        sgl_xfree(piece);
      }
    }
    /* sleep waiting for worker progress: SleepCS releases lock while waiting
     * and RE-ACQUIRES on wake - so we must Leave exactly once after it. */
    SleepConditionVariableCS(&E.cv_step, &E.lock, 2000);
    LeaveCriticalSection(&E.lock);
  }
}

static void reclaim(int id) {
  /* only marks: actual free happens in worker after it stops referencing
   * the request (worker owns lifecycle; avoids use-after-free from its run[]
   * array). Set state to R_REAPED if DONE/FAIL; worker frees next pass. */
  EnterCriticalSection(&E.lock);
  for (int i = 0; i < MAX_CONC; i++)
    if (E.reqs[i] && E.reqs[i]->id == id &&
        (E.reqs[i]->state == R_DONE || E.reqs[i]->state == R_FAIL)) {
      E.reqs[i]->state = R_REAPED;
      break;
    }
  LeaveCriticalSection(&E.lock);
  WakeConditionVariable(&E.cv_work);
}

static int route(SOCKET fd, const http_req *req) {
  /* t5: GET works normally but under burst load a tiny no-body request can
   * intermittently park its conn_thread (see NOTES_t5). Monitoring should use
   * `POST /health -d x` — body-carrying requests verified stable. */
  if (strcmp(req->path, "/health") == 0 &&
      (strcmp(req->method, "GET") == 0 || strcmp(req->method, "POST") == 0)) {
    respond(fd, 200, "application/json", "{\"ok\":true}");
    return 0;
  }
  if (strcmp(req->method, "POST") != 0) { respond_error(fd, 404, "not found"); return 0; }
  int chat = strcmp(req->path, "/v1/chat/completions") == 0;
  int openai = strcmp(req->path, "/v1/completions") == 0;
  int gen = strcmp(req->path, "/generate") == 0;
  if (!chat && !openai && !gen) { respond_error(fd, 404, "unknown path"); return 0; }

  sgl_json *j = sgl_json_parse(req->body, req->body_len);
  if (!j) { respond_error(fd, 400, "invalid JSON"); return 0; }

  char *prompt = NULL;
  int max_new = 64;
  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp));
  int is_stream = 0;
  if (chat) {
    prompt = build_chatml(j);
    if (!prompt) { sgl_json_free(j); respond_error(fd, 400, "messages required"); return 0; }
    max_new = sgl_json_get_int(j, "max_tokens", 256);
    sp.temperature = (float)sgl_json_get_num(j, "temperature", 0.0);
    sp.top_p = (float)sgl_json_get_num(j, "top_p", 0.0);
    const sgl_json *sv = sgl_json_get(j, "stream");
    is_stream = sv && sv->type == SGL_JSON_BOOL && sv->boolean;
  } else {
    const char *key = openai ? "prompt" : "text";
    const char *text = sgl_json_get_str(j, key, NULL);
    if (!text || !*text) { sgl_json_free(j); respond_error(fd, 400, "prompt required"); return 0; }
    size_t n = strlen(text) + 1;
    prompt = (char *)malloc(n);
    memcpy(prompt, text, n);
    max_new = sgl_json_get_int(j, "max_tokens", 64);
    sp.temperature = (float)sgl_json_get_num(j, "temperature", 0.0);
    sp.top_p = (float)sgl_json_get_num(j, "top_p", 0.0);
    const sgl_json *sv = sgl_json_get(j, "stream");
    is_stream = sv && sv->type == SGL_JSON_BOOL && sv->boolean;
  }
  sgl_json_free(j);
  if (max_new <= 0 || max_new > MAX_NEW_CAP) max_new = 256;

  int id = submit(prompt, max_new, &sp);
  free(prompt);
  if (id < 0) { respond_error(fd, 503, "prompt too long (>64KB) or server busy"); return 0; }

  if (is_stream) {
    /* mark streaming then wait_done emits SSE chunks */
    EnterCriticalSection(&E.lock);
    for (int i = 0; i < MAX_CONC; i++)
      if (E.reqs[i] && E.reqs[i]->id == id) { E.reqs[i]->streaming = 1; break; }
    LeaveCriticalSection(&E.lock);
    const char *hdr =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    send_all(fd, hdr, (int)strlen(hdr));
    char *text = NULL;
    request *r = NULL;
    int rc = wait_done(fd, id, &text, &r);
    if (rc == 0 && text) {
      char esc[8192];
      json_escape(text, esc, sizeof(esc));
      (void)esc;
      const char *fin = r ? (r->finish[0] ? r->finish : "stop") : "stop";
      char chunk[256];
      int n = snprintf(chunk, sizeof(chunk),
                       "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\ndata: [DONE]\n\n",
                       fin);
      send_all(fd, chunk, n);
      free(text);
    }
    reclaim(id);
    return 0;
  }

  char *text = NULL;
  request *r = NULL;
  int rc = wait_done(fd, id, &text, &r);
  if (rc == -2) { reclaim(id); respond_error(fd, 500, "generation failed"); return 0; }
  if (rc != 0) { reclaim(id); respond_error(fd, 500, "internal"); return 0; }
  int ptok = r ? r->n_prompt : 0;
  int ngen = r ? r->n_gen : 0;
  const char *fin = r ? (r->finish[0] ? r->finish : "stop") : "stop";
  char esc[8192];
  json_escape(text ? text : "", esc, sizeof(esc));
  char body[8600];
  if (chat) {
    snprintf(body, sizeof(body),
             "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"%s\"}],"
             "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
             esc, fin, ptok, ngen, ptok + ngen);
  } else if (openai) {
    snprintf(body, sizeof(body),
             "{\"choices\":[{\"text\":\"%s\",\"index\":0,\"finish_reason\":\"%s\"}],"
             "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
             esc, fin, ptok, ngen, ptok + ngen);
  } else {
    snprintf(body, sizeof(body),
             "{\"text\":\"\",\"completion\":\"%s\",\"meta_info\":{\"finish_reason\":\"%s\",\"completion_tokens\":%d}}",
             esc, fin, ngen);
  }
  free(text);
  reclaim(id);
  respond(fd, 200, "application/json", body);
  return 0;
}

typedef struct { SOCKET fd; } conn_ctx;

static DWORD WINAPI conn_thread(LPVOID arg) {
  conn_ctx *c = (conn_ctx *)arg;
  http_req req;
  /* t5 robustness: a stalled client (or a winsock internal hiccup seen under
   * burst load) could park recv() forever and leak the thread. Cap it. */
  DWORD rto = 30000;
  setsockopt(c->fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&rto, sizeof(rto));
  DWORD sto = 5000; /* t5: a blocked send() must not park the thread forever */
  setsockopt(c->fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&sto, sizeof(sto));
  setsockopt(c->fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&(int){1}, sizeof(int));
  if (read_request(c->fd, &req) == 0) {
    route(c->fd, &req);
    free(req.body);
  } else {
    fprintf(stderr, "[http] read_request failed (wsa=%d)\n", WSAGetLastError());
  }
  closesocket(c->fd);
  free(c);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model_dir> [port=30001]\n", argv[0]);
    return 1;
  }
  const char *dir = argv[1];
  int port = argc > 2 ? atoi(argv[2]) : 30001;

  WSADATA wsa;
  WSAStartup(MAKEWORD(2, 2), &wsa);

  memset(&E, 0, sizeof(E));
  InitializeCriticalSection(&E.lock);
  InitializeConditionVariable(&E.cv_work);
  InitializeConditionVariable(&E.cv_step);

  double t0 = (double)clock() / CLOCKS_PER_SEC;
  E.cpu = sgl_llama_load(dir);
  if (!E.cpu) return 2;
  E.gpu = sgl_cuda_from_llama(E.cpu);
  if (!E.gpu) return 2;
  char tpath[1024];
  snprintf(tpath, sizeof(tpath), "%s/tokenizer.json", dir);
  E.tok = sgl_tokenizer_load(tpath);
  if (!E.tok) return 2;
  E.vocab = E.cpu->vocab;
  E.eos = sgl_tokenizer_eos_id(E.tok);
  {
    const char *ec = getenv("SGL_HTTP_CONC");
    if (ec) {
      int c = atoi(ec);
      if (c >= 1 && c <= MAX_CONC) ACTIVE_CONC = c;
    }
  }
  E.seg_len = sgl_cuda_ctx(E.gpu) / ACTIVE_CONC;
  /* t5: decode CUDA graphs are unstable under this server's dynamic batch
   * sizes (intermittent sticky CUDA errors after B changes + buffer grows;
   * single-batch cuda_test replays are fine). Default OFF here; set
   * SGL_CUDA_GRAPH=1 explicitly to experiment. */
  if (!getenv("SGL_CUDA_GRAPH")) _putenv_s("SGL_CUDA_GRAPH", "0");
  printf("[engine] loaded in %.1fs; ctx=%d seg=%d conc=%d eos=%d\n",
         (double)clock() / CLOCKS_PER_SEC - t0, sgl_cuda_ctx(E.gpu), E.seg_len, ACTIVE_CONC, E.eos);
  fflush(stdout);

  HANDLE w = CreateThread(NULL, 0, worker, NULL, 0, NULL);

  SOCKET lst = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons((u_short)port);
  addr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
  if (bind(lst, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
    fprintf(stderr, "bind failed: %d\n", WSAGetLastError());
    return 2;
  }
  listen(lst, 16);
  printf("[http] serving on 127.0.0.1:%d (/health /generate /v1/completions /v1/chat/completions)\n", port);
  fflush(stdout);
  for (;;) {
    SOCKET c = accept(lst, NULL, NULL);
    if (c == INVALID_SOCKET) {
      /* t5 fix: transient accept failures (connection aborted mid-handshake
       * under burst load) used to kill the accept loop entirely — the server
       * kept its listening socket but served no one. Retry instead. */
      int e = WSAGetLastError();
      if (e == WSAECONNABORTED || e == WSAEINTR || e == WSAEMFILE) continue;
      fprintf(stderr, "[http] accept fatal: %d\n", e);
      break;
    }
    conn_ctx *ctx = (conn_ctx *)malloc(sizeof(*ctx));
    ctx->fd = c;
    HANDLE t = CreateThread(NULL, 0, conn_thread, ctx, 0, NULL);
    if (t) CloseHandle(t);
  }
  E.stop = 1;
  WakeAllConditionVariable(&E.cv_work);
  WaitForSingleObject(w, INFINITE);
  return 0;
}
