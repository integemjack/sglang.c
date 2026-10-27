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
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#else
/* POSIX（Linux/macOS）：winsock -> BSD socket，Win32 同步原语 -> pthread。
 * 下面所有平台差异都收在 #else 分文里，Windows 编译路径一字不改。 */
#include "sgl/port.h" /* fopen_s / _strnicmp / strcpy_s 垫片 */
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>

/* ---- winsock -> POSIX 符号映射 ---- */
#define SOCKET int
#define INVALID_SOCKET (-1)
#define closesocket(s) close(s)
#define WSAGetLastError() (errno)
#define WSAEWOULDBLOCK EAGAIN
#define WSAENOBUFS ENOBUFS
#define WSAEINTR EINTR
#define WSAECONNABORTED ECONNABORTED
#define WSAEMFILE EMFILE
#define _strdup strdup
#ifdef MSG_NOSIGNAL
#define SGL_SEND_FLAGS MSG_NOSIGNAL /* 客户端先断开时 send 得 EPIPE，而不是 SIGPIPE 杀进程 */
#else
#define SGL_SEND_FLAGS 0 /* macOS 等没有 MSG_NOSIGNAL（靠 SO_NOSIGPIPE / signal）*/
#endif

/* ---- Win32 临界区 / 条件变量 -> pthread ---- */
#define CRITICAL_SECTION pthread_mutex_t
#define InitializeCriticalSection(m) pthread_mutex_init((m), NULL)
#define EnterCriticalSection(m) pthread_mutex_lock(m)
#define LeaveCriticalSection(m) pthread_mutex_unlock(m)
#define CONDITION_VARIABLE pthread_cond_t
#define InitializeConditionVariable(c) pthread_cond_init((c), NULL)
#define WakeConditionVariable(c) pthread_cond_signal(c)
#define WakeAllConditionVariable(c) pthread_cond_broadcast(c)
/* SleepConditionVariableCS 等价：ms 超时（相对时间 -> CLOCK_REALTIME 绝对时间）;
 * 与 Win32 一样：等待期间释放锁，醒来重新持锁。 */
static int sgl_sleep_cs(pthread_cond_t *cv, pthread_mutex_t *cs, int ms) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += ms / 1000;
  ts.tv_nsec += (long)(ms % 1000) * 1000000L;
  if (ts.tv_nsec >= 1000000000L) { ts.tv_sec += 1; ts.tv_nsec -= 1000000000L; }
  return pthread_cond_timedwait(cv, cs, &ts);
}
#define SleepConditionVariableCS(cv, cs, ms) sgl_sleep_cs((cv), (cs), (ms))
static void sgl_sleep_ms(int ms) {
  struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
  nanosleep(&ts, NULL);
}
#define Sleep(ms) sgl_sleep_ms(ms)
/* GetTickCount64：单调毫秒（Windows 上是开机毫秒，只用于打时间戳/造种子） */
static unsigned long long sgl_tick_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (unsigned long long)ts.tv_sec * 1000ULL + (unsigned long long)(ts.tv_nsec / 1000000L);
}
#define GetTickCount64() sgl_tick_ms()
/* 线程入口签名：Win32 = DWORD WINAPI f(LPVOID)，POSIX = void *f(void *) */
#define SGL_THREAD_SIG(name) static void *name(void *arg)
#define SGL_THREAD_RETURN return NULL
#endif

#ifdef _WIN32
#define SGL_SEND_FLAGS 0 /* winsock 无 MSG_NOSIGNAL */
#define SGL_THREAD_SIG(name) static DWORD WINAPI name(LPVOID arg)
#define SGL_THREAD_RETURN return 0
#endif

#include <math.h> /* P1: grammar 掩码用 -INFINITY */

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/clip.h"
#include "sgl/cuda.h"
#include "sgl/grammar.h" /* P1: 约束解码 */
#include "sgl/image.h" /* G11: PNG/JPEG 解码 + CLIP 预处理（与 CPU 版同一条路径）*/
#include "sgl/jinja.h"  /* P0-2: 模型自带 chat_template */
#include "sgl/json.h"
#include "sgl/llama.h"
#include "sgl/log.h" /* P1: SGL_INFO/SGL_ERROR（grammar 词表与拒 token 日志）*/
#include "sgl/ops.h"
#include "sgl/parser.h" /* P2: reasoning / tool_call 解析 */
#include "sgl/safetensors.h"
#include "sgl/sampling.h"

#define MAX_CONC 8
#define MAX_NEW_CAP 512

static int ACTIVE_CONC = 8; /* effective concurrency, SGL_HTTP_CONC overrides */
typedef enum { R_WAIT = 0, R_RUN, R_DONE, R_FAIL, R_REAPED } rstate;

typedef struct request {
  int id;
  rstate state;
  int *prompt;
  int n_prompt;
  float *img_feats; /* t10: projector output [n_img][E_lm], appended after prompt */
  int n_img;
  int gen[MAX_NEW_CAP];
  int n_gen;
  int max_new;
  sgl_sampling_params sp;
  uint64_t rng;
  int seg;             /* kv segment index, -1 until assigned */
  int seg_len;         /* segment capacity in positions */
  int reached_run;     /* G6: prefill 跑成功过（才能登记前缀租让） */
  float *logits;       /* host buffer vocab floats (last step) */
  int has_logits;
  char *text_out;      /* decoded completion (malloc) */
  char finish[12];     /* "stop" | "length" (t5: honest finish_reason) */
  /* P0-3 / P1：与 CPU 版对齐的采样面 */
  sgl_grammar *gram;   /* 约束解码（response_format），NULL = 不约束 */
  float *lp;           /* [max_new] 每个生成 token 的 logprob（want_lp 时才有）*/
  int n_lp;
  int want_lp;
  int choices_done;    /* 1 = tool_calls 结束（finish_reason 用）*/
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
  sgl_clip *clip;      /* t10: vision tower, NULL when model dir has no clip_config.json */
  sgl_jinja *jinja;    /* P0-2: 模型自带 chat_template（NULL = 回退 ChatML）*/
  char model_name[256];/* 响应体里的 model 字段 */
  /* P1 /metrics 计数器 */
  long tot_prefill, tot_decode, tot_finished, tot_prompt_tokens;
  float *pj_w, *pj_b;  /* mm_projector linear_1 [mid][proj_in] + [mid] */
  float *pj_w2, *pj_b2; /* G12: linear_2 [E_lm][mid] + [E_lm]（NULL = 单层 linear）*/
  int pj_in, pj_mid;
  sgl_projector pj;    /* G12: 统一走这个（单层/两层自动）*/
  request *reqs[MAX_CONC];
  int n_active;
  request *waitq_head, *waitq_tail; /* t5: queue beyond ACTIVE_CONC */
  int n_waitq;
  int next_id;
  CRITICAL_SECTION lock;
  CONDITION_VARIABLE cv_work;   /* worker waits for requests */
  CONDITION_VARIABLE cv_step;   /* http threads wait for progress */
  volatile int stop;
  char gpu_err[256];  /* G5: why the GPU engine is unavailable (degraded mode) */
  int degraded;       /* G5: 1 = serving without GPU; every generate gets 503 */
  int last_prompt_tokens; /* P1: 最近一次因单段容量被拒的 prompt token 数（给错误文案用）*/
} engine;

static engine E;

/* P1: grammar 词表缓存（token id → 真实字节）。只有第一次用到约束解码时才建，
 * 因为要遍历整个词表（~15 万条）。与 CPU 调度器同一套做法。*/
#define SGL_VH_CAP (1 << 20)
static char *g_vh_buf;
static const char **g_vh_ptr;
static int *g_vh_len;
static int g_vh_vocab;
static int g_vh_ready;
static int *g_gram_ids;
static unsigned char *g_gram_ok;
static int g_gram_cap;

static void sched_vocab_ensure(void) {
  if (g_vh_ready || !E.tok) return;
  int V = sgl_tokenizer_vocab_size(E.tok);
  g_vh_vocab = V;
  g_vh_buf = (char *)malloc(SGL_VH_CAP);
  g_vh_ptr = (const char **)malloc(sizeof(char *) * (size_t)(V > 0 ? V : 1));
  g_vh_len = (int *)malloc(sizeof(int) * (size_t)(V > 0 ? V : 1));
  size_t off = 0;
  for (int i = 0; i < V; i++) {
    char tmp[256];
    int n = sgl_tokenizer_piece_bytes(E.tok, i, tmp, (int)sizeof(tmp));
    if (n <= 0 || off + (size_t)n + 1 > SGL_VH_CAP) {
      g_vh_ptr[i] = NULL;
      g_vh_len[i] = 0;
      continue;
    }
    memcpy(g_vh_buf + off, tmp, (size_t)n);
    g_vh_buf[off + (size_t)n] = 0;
    g_vh_ptr[i] = g_vh_buf + off;
    g_vh_len[i] = n;
    off += (size_t)n + 1;
  }
  g_vh_ready = 1;
  SGL_INFO("grammar vocab ready: %d tokens, %llu bytes", V, (unsigned long long)off);
}

/* ---- G6: GPU 前缀缓存（段租让） ----
 * 同一个 system prompt 被多个请求共用时，第二个请求不必重算 prefill：把已经
 * 算好的前缀 KV 从「租让中」的段拷到自己的段头 len 行，然后从 len 起接着
 * prefill。租让 = 请求结束后不回收它的 KV 段，留给前缀相同的后续请求；表满
 * 按 LRU 驱逐（驱逐后那个段才能给别人用）。
 * 正确性：前缀必须都从位置 0 开始（RoPE 把位置编进了 K），服务端每条序列在
 * 自己段内 pos0 都从 0 起，成立；搬来的行与重算的行逐位相等（同一套 kernel、
 * 同样的计算顺序），所以命中与否输出一致（cuda_test [prefix] 逐 token 对拍）。 */
#define PFX_MAX 16
#define PFX_MIN 32 /* 短于这个的前缀不值得占一个段 */
/* G15: 租让 = 请求结束后不回收的 KV 行。slot.len 是「这个段的 [0,len) 行全部
 * 有效且与 toks[] 逐 token 对应」的长度，所以它不止 prompt：段里后面的生成 token
 * 的行也是好的（decode_batch 用同一套 kernel 写的），多轮对话 / 续写场景下
 * 下一个请求的 prompt 恰好以「上一轮的 prompt+生成结果」开头，就能整段复用。
 * 带图的请求只登记 text 部分（视觉行依赖图片，位置在 text 之后，后续谁都能接）。*/
typedef struct {
  int used;
  int seg;    /* 租让的 KV 段号（base = seg * seg_len） */
  int len;    /* 有效行数：prompt（+ 生成 token，非视觉时） */
  int *toks;  /* 与 [0,len) 行一一对应的 token 副本（精确比对，不靠 hash） */
  unsigned long long use; /* LRU 时钟 */
} pfx_slot;
static pfx_slot g_pfx[PFX_MAX];
static int g_pfx_of[MAX_CONC];             /* 段 -> 表项下标+1；0 = 未租让 */
static unsigned long long g_pfx_clock;
static long g_pfx_hits, g_pfx_misses, g_pfx_hit_rows, g_pfx_leased_rows;
static long g_pfx_flushes;
/* G15(b): 运维杆 —— SGL_PFX_MAX=0 彻底关掉前缀缓存（显存/排错极端情况用），
 * 也可以给小值限住租让表。*/ 
static int PFX_MAX_EFF = PFX_MAX;
static void pfx_drop(int idx); /* 定义在下面（flush 要用）*/

/* 把全部租让丢掉（段里的 KV 会在后续请求里被覆盖）。
 * 注意：本服务的段是按并发数**静态划分**的，租让不额外占显存（/health 的
 * leased_rows 只是“这些行还能复用”的计数，不是额外分配）；所以这个函数
 * 是给「请求超容量（rc=-2）/ 关缓存」这类需要干净现场的场景用的，
 * 而不是真的释放显存。*/ 
static void pfx_flush_all(void) {
  for (int i = 0; i < PFX_MAX; i++) pfx_drop(i);
  g_pfx_flushes++;
}

static void pfx_drop(int idx) {
  if (idx < 0 || idx >= PFX_MAX || !g_pfx[idx].used) return;
  g_pfx_of[g_pfx[idx].seg] = 0;
  g_pfx_leased_rows -= g_pfx[idx].len; /* 这是 gauge（当前可复用行数），不是累计 */
  free(g_pfx[idx].toks);
  memset(&g_pfx[idx], 0, sizeof(g_pfx[idx]));
}

/* 最长公共前缀：返回长度（< PFX_MIN 视为没命中），*out_seg 给出来源段 */
static int pfx_match(const int *toks, int n, int *out_seg) {
  int best = 0, bseg = -1;
  for (int i = 0; i < PFX_MAX_EFF; i++) {
    if (!g_pfx[i].used) continue;
    int m = g_pfx[i].len < n ? g_pfx[i].len : n;
    int l = 0;
    while (l < m && g_pfx[i].toks[l] == toks[l]) l++;
    if (l > best) { best = l; bseg = g_pfx[i].seg; }
  }
  if (best < PFX_MIN) { *out_seg = -1; return 0; }
  *out_seg = bseg;
  for (int i = 0; i < PFX_MAX_EFF; i++)
    if (g_pfx[i].used && g_pfx[i].seg == bseg) g_pfx[i].use = ++g_pfx_clock;
  return best;
}

/* 请求结束后把它的段登记为前缀租让。
 * toks/n_tok：这个段 [0,n_tok) 行对应的 token（非视觉时为 prompt+生成，
 * 视觉时为 prompt 文本部分）；slot.len 就是 n_tok。 */
static void pfx_register(int seg, const int *toks, int n_tok) {
  int len = n_tok;
  if (PFX_MAX_EFF <= 0) return;
  if (len < PFX_MIN || seg < 0 || seg >= MAX_CONC) return;
  int slot = -1, lru = 0;
  unsigned long long oldest = ~0ull;
  for (int i = 0; i < PFX_MAX_EFF; i++) {
    if (!g_pfx[i].used) { slot = i; break; }
    if (g_pfx[i].seg == seg) { slot = i; break; } /* 同段直接覆盖旧登记 */
    if (g_pfx[i].use < oldest) { oldest = g_pfx[i].use; lru = i; }
  }
  if (slot < 0) { pfx_drop(lru); slot = lru; } /* 表满：LRU 驱逐（那个段从此可被复用） */
  int *cp = (int *)malloc(sizeof(int) * (size_t)len);
  if (!cp) return;
  memcpy(cp, toks, sizeof(int) * (size_t)len);
  free(g_pfx[slot].toks);
  g_pfx[slot].toks = cp;
  g_pfx[slot].used = 1;
  g_pfx[slot].seg = seg;
  g_pfx[slot].len = len;
  g_pfx[slot].use = ++g_pfx_clock;
  g_pfx_of[seg] = slot + 1;
  g_pfx_leased_rows += len;
}

/* G15：把一次已完成请求能租让的行拼出来。
 * 非视觉：prompt + 生成（末尾那个生成 token 的 KV 行不一定被写过 ——
 *   decode 循环在 n_gen==max_new 或 eos 时就直接收了 —— 所以保守地只算到
 *   n_gen-1 行）。
 * 视觉：只到 prompt 末（后面的视觉行依赖图片，不归这段文本管）。
 * 返回实际登记的行数（0 = 不登记）。 */
static int pfx_register_req(int seg, const request *r) {
  if (!r || r->n_gen < 0) return -1;
  if (r->n_img > 0) {
    pfx_register(seg, r->prompt, r->n_prompt);
    return r->n_prompt;
  }
  int n_gen_ok = r->n_gen > 0 ? r->n_gen - 1 : 0;
  int total = r->n_prompt + n_gen_ok;
  if (n_gen_ok == 0) { pfx_register(seg, r->prompt, r->n_prompt); return r->n_prompt; }
  int *buf = (int *)malloc(sizeof(int) * (size_t)total);
  if (!buf) { pfx_register(seg, r->prompt, r->n_prompt); return r->n_prompt; }
  memcpy(buf, r->prompt, sizeof(int) * (size_t)r->n_prompt);
  memcpy(buf + r->n_prompt, r->gen, sizeof(int) * (size_t)n_gen_ok);
  pfx_register(seg, buf, total);
  free(buf);
  return total;
}

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
  E.tot_prompt_tokens += r->n_prompt;
  return r;
}

static void req_free(request *r) {
  if (!r) return;
  sgl_xfree(r->prompt);
  sgl_xfree(r->img_feats); /* t10: projector output (sgl_xmalloc aligned) */
  sgl_xfree(r->lp);
  sgl_grammar_free(r->gram);
  free(r->logits);
  sgl_xfree(r->text_out); /* decode_gen 返回 sgl_xmalloc（对齐）——绝不能 free() */
  free(r);
}

/* t10/G11: image URL -> CLIP encode -> projector -> [n_img][E_lm] features
 * (sgl_xmalloc'd). Returns NULL on any mismatch. 两种载荷格式：
 *  - data:image/{png,jpeg};base64,... （G11）：stb_image 解码 + PIL 语义
 *    resize/crop → 走的正是 CPU 版那条 sgl_clip_project_bytes（真实客户端发的
 *    就是 JPEG/PNG，以前这里只认 raw float，等于真实图片全用不了）；
 *  - 裸 base64 的 raw float32 [3*img*img] CHW（t10 兼容分支，端到端老脚本在用）。*/
static float *vision_feats(const char *url, int *n_out) {
  *n_out = 0;
  if (!E.clip || !url) return NULL;
  /* strip the data:...;base64, prefix if present */
  const char *comma = strrchr(url, ',');
  const char *p = comma ? comma + 1 : url;
  /* G11: 按 data URL 的 mime 分派（image/png、image/jpeg、image/jpg）*/
  if (!strncmp(url, "data:image/png", 14) || !strncmp(url, "data:image/jpeg", 15) ||
      !strncmp(url, "data:image/jpg", 14)) {
    unsigned char *bytes = NULL;
    size_t bn = 0;
    if (sgl_base64_decode(p, strlen(p), &bytes, &bn) != 0) return NULL;
    float *out = NULL;
    int nn = 0;
    int rc = sgl_clip_project_bytes_pj(bytes, bn, E.clip, &E.pj, &out, &nn);
    sgl_xfree(bytes);
    if (rc != 0 || !out) {
      sgl_xfree(out);
      return NULL;
    }
    *n_out = E.clip->n_patch;
    return out;
  }
  size_t cap = strlen(p) * 3 / 4 + 4;
  unsigned char *raw = (unsigned char *)sgl_xmalloc(cap);
  size_t nraw = 0;
  int acc = 0, bits = 0;
  for (; *p; p++) {
    if (*p == '=' || *p == '\n' || *p == '\r') continue;
    int c = (unsigned char)*p, v;
    if (c >= 'A' && c <= 'Z') v = c - 'A';
    else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
    else if (c >= '0' && c <= '9') v = c - '0' + 52;
    else if (c == '+') v = 62;
    else if (c == '/') v = 63;
    else { sgl_xfree(raw); return NULL; } /* invalid b64 char */
    acc = (acc << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      raw[nraw++] = (unsigned char)((acc >> bits) & 0xFF);
    }
  }
  size_t want = (size_t)3 * E.clip->img * E.clip->img * sizeof(float);
  if (nraw != want) { sgl_xfree(raw); return NULL; }
  float *feats_proj = (float *)sgl_xmalloc(
      sizeof(float) * (size_t)E.clip->n_patch * E.cpu->n_embd);
  float *feats = (float *)sgl_xmalloc(
      sizeof(float) * (size_t)E.clip->n_patch * E.clip->proj_dim);
  if (sgl_clip_encode(E.clip, (const float *)raw, feats) != 0) {
    sgl_xfree(raw); sgl_xfree(feats); sgl_xfree(feats_proj);
    return NULL;
  }
  sgl_xfree(raw);
  sgl_matmul_nt(feats, E.pj_w, E.pj_b, feats_proj, E.clip->n_patch,
                E.cpu->n_embd, E.clip->proj_dim);
  sgl_xfree(feats);
  *n_out = E.clip->n_patch;
  return feats_proj;
}

static int submit2(const char *text, int max_new, const sgl_sampling_params *sp,
                   float *img_feats, int n_img) {
  /* t11: cap prompt length before the O(n^2)-ish BPE merge — a pathological
   * 200KB single-char body used to park encode for minutes (server hang). */
  if (!text || strlen(text) > 65536) { sgl_xfree(img_feats); return -1; }
  /* t5: no more 503 when full — park in a wait queue; the worker drains it
   * into freed slots every pass. Hard cap only guards against unbounded
   * queue growth (128 waiting). */
  EnterCriticalSection(&E.lock);
  if (E.n_active + E.n_waitq >= 128) { LeaveCriticalSection(&E.lock); sgl_xfree(img_feats); return -1; }
  request *r = req_new(text, max_new, sp);
  if (!r) { LeaveCriticalSection(&E.lock); sgl_xfree(img_feats); return -1; }
  /* P1: 单段容量检查——prompt 比一个 KV 段还长时永远算不出来，以前会让请求进 R_FAIL，
   * 调用方只看到 “generation failed”。这里提前拒掉，并让 route 给明确提示。*/
  if (E.seg_len > 0 && r->n_prompt + n_img > E.seg_len) {
    int need = r->n_prompt + n_img;
    LeaveCriticalSection(&E.lock);
    sgl_xfree(img_feats);
    req_free(r);
    E.last_prompt_tokens = need;
    return -2;
  }
  r->img_feats = img_feats;
  r->n_img = n_img;
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

SGL_THREAD_SIG(worker) {
  (void)arg;
  float *batch_logits = (float *)malloc(sizeof(float) * (size_t)MAX_CONC * E.vocab);
  for (;;) {
    EnterCriticalSection(&E.lock);
    if (E.stop) { LeaveCriticalSection(&E.lock); break; }
    /* reap: free requests marked by their http thread (safe: worker no longer
     * references them after this pass begins) */
    for (int i = 0; i < MAX_CONC; i++) {
      request *r = E.reqs[i];
      if (r && r->state == R_REAPED) {
        /* G6/G15: 跑过 prefill 的请求，把它的段登记为前缀租让（后面的请求就能复用）。
         * G15: ① 行数不再只算 prompt，而是「这个段里确实有有效 KV 的行」（含生成 token，
         * 多轮对话/续写直接受益）；② 带图的请求也登记了（只登到 text 末：视觉行依赖
         * 图片，位置在 text 之后，不归这段登记管）。*/
        if (E.gpu && r->reached_run && r->seg == i && r->n_prompt >= PFX_MIN)
          pfx_register_req(i, r);
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
      /* G6/G15: 前缀命中 -> 把租让段里的 KV 行搬进本段，prefill 从命中处起。
       * 全命中时留最后一个 token 重算一遍（那一行 KV 值不变），否则本轮得
       * 不到 logits。带图的请求也参与（能命中 text 部分，视觉行照旧算）。*/
      int off = 0;
      if (E.gpu && r->n_prompt >= PFX_MIN) {
        int src_seg = -1;
        int hit = pfx_match(r->prompt, r->n_prompt, &src_seg);
        if (hit >= r->n_prompt) hit = r->n_prompt - 1;
        /* 本段自己如果持有租让：新请求要写它的 KV 行了。除非这段租让恰好就是
         * 我们要原样保留的那段（src==本段 且 hit==租让长度），否则必须撕掉这条租让，
         * 不然下一个请求会匹配到已被覆盖的垃圾 KV。 */
        if (hit > 0 && g_pfx_of[i]) {
          int li = g_pfx_of[i] - 1;
          if (!(src_seg == i && hit == g_pfx[li].len)) pfx_drop(li);
        }
        if (hit > 0) {
          if (src_seg != i) {
            if (g_pfx_of[i]) pfx_drop(g_pfx_of[i] - 1); /* 本段旧租让要被覆盖了 */
            if (sgl_cuda_copy_kv(E.gpu, src_seg * E.seg_len, base, hit) != 0) hit = 0;
          }
          if (hit > 0) { g_pfx_hits++; g_pfx_hit_rows += hit; off = hit; }
          else g_pfx_misses++;
        }
      } else if (E.gpu) {
        g_pfx_misses++;
      }
      if (r->n_prompt + r->n_img > r->seg_len) {
        rc = -2; /* prompt (+visual tokens) exceeds its KV segment capacity */
        pfx_flush_all(); /* G15(b): 现场已经不干净（本请求要的比一段还长），
                          * 把租让全丢掉，别让后续请求命中半截。*/ 
      } else {
        rc = 0;
        while (off < r->n_prompt && rc == 0) {
          int chunk = r->n_prompt - off;
          /* only the last chunk needs the lm_head pass + logits D2H */
          float *dst = (off + chunk >= r->n_prompt && r->n_img == 0) ? r->logits : NULL;
          rc = sgl_cuda_forward_seg(E.gpu, r->prompt + off, chunk, off, base, dst);
      E.tot_prefill += chunk;
          off += chunk;
        }
        /* t10: visual tokens go AFTER the text prompt, writing KV at
         * [n_prompt, n_prompt+n_img) via the embedding-level forward; their
         * last position produces the logits for the first sampled token. */
        if (rc == 0 && r->n_img > 0) {
          rc = sgl_cuda_forward_seg_emb(E.gpu, r->img_feats, r->n_img,
                                        r->n_prompt, base, r->logits);
        }
      }
      EnterCriticalSection(&E.lock);
      if (rc != 0) { r->state = R_FAIL; }
      else { r->state = R_RUN; r->has_logits = 1; r->reached_run = 1; }
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
          E.tot_finished++;
          continue;
        }
        /* P1: 约束解码——把会把输出带出合法前缀的 token 置 -inf */
        if (r->gram) {
          sched_vocab_ensure();
          sgl_grammar_set_vocab(r->gram, g_vh_ptr, g_vh_len, g_vh_vocab);
          int V = E.vocab < g_vh_vocab ? E.vocab : g_vh_vocab;
          if (g_gram_ids == NULL || g_gram_cap < g_vh_vocab) {
            free(g_gram_ids);
            free(g_gram_ok);
            g_gram_ids = (int *)malloc(sizeof(int) * (size_t)g_vh_vocab);
            g_gram_ok = (unsigned char *)malloc((size_t)g_vh_vocab);
            for (int i = 0; i < g_vh_vocab; i++) g_gram_ids[i] = i;
            g_gram_cap = g_vh_vocab;
          }
          sgl_grammar_mask(r->gram, g_gram_ids, g_vh_vocab, g_gram_ok);
          int allowed = 0;
          for (int i = 0; i < V; i++) {
            if (g_gram_ok[i]) allowed++;
            else r->logits[i] = -INFINITY;
          }
          if (allowed == 0) { /* 文法卡死：不约束，宁可输出脏也不出 NaN */
            for (int i = 0; i < E.vocab; i++) r->logits[i] = r->logits[i] == -INFINITY ? 0.0f : r->logits[i];
            sgl_grammar_free(r->gram);
            r->gram = NULL;
          }
        }
        float lp = 0.0f;
        int t = sgl_sample_ex(r->logits, E.vocab, &r->sp, &r->rng, r->gen, r->n_gen, &lp);
        int stop = 0;
        if (t == E.eos && !r->sp.ignore_eos) stop = 1;
        if (sgl_sampling_is_stop(&r->sp, t)) stop = 1;
        if (stop) {
          r->state = R_DONE;
          strcpy_s(r->finish, sizeof(r->finish), "stop");
          r->text_out = decode_gen(r);
          E.tot_finished++;
          continue;
        }
        r->gen[r->n_gen++] = t;
        if (r->want_lp && r->lp && r->n_lp < r->max_new) r->lp[r->n_lp++] = lp;
        if (r->gram) {
          if (sgl_grammar_accept(r->gram, t) != 0)
            SGL_ERROR("grammar: rejected token %d (%s)", t, sgl_grammar_error(r->gram));
          if (sgl_grammar_can_finish(r->gram)) { /* 完整合法输出即停 */
            r->state = R_DONE;
            strcpy_s(r->finish, sizeof(r->finish), "stop");
            r->text_out = decode_gen(r);
            E.tot_finished++;
            continue;
          }
        }
        E.tot_decode++;
      }
      /* 2b) batch forward last token of each still-running seq */
      int bt[MAX_CONC], bb[MAX_CONC], bl[MAX_CONC];
      int nb = 0;
      request *br[MAX_CONC];
      for (int b = 0; b < B; b++) {
        request *r = run[b];
        if (r->state != R_RUN) continue;
        if (r->n_prompt + r->n_img + r->n_gen >= r->seg_len) {
          r->state = R_DONE;
          strcpy_s(r->finish, sizeof(r->finish), "length");
          r->text_out = decode_gen(r);
          E.tot_finished++;
          continue;
        }
        bt[nb] = r->gen[r->n_gen - 1];
        bb[nb] = r->seg * E.seg_len;
        bl[nb] = r->n_prompt + r->n_img + r->n_gen - 1;
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
  SGL_THREAD_RETURN;
}

/* ---------------- http plumbing ---------------- */

typedef struct { char *body; size_t body_len; char method[8]; char path[128]; } http_req;

static int trace_on(void); /* G17 诊断开关（定义在 conn_thread 上方）*/

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
    if (len + 1 >= cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
    int n = recv(fd, buf + len, (int)(cap - len - 1), 0);
    if (trace_on()) fprintf(stderr, "[trace] recv fd=%llu n=%d len=%zu\n",
                           (unsigned long long)fd, n, len);
    if (n <= 0) { free(buf); return -1; }
    len += (size_t)n;
    buf[len] = 0; /* G17 修复：必须 NUL 截断，否则下面的 sscanf/strstr 会跑到
                   * len 之后的未初始化（或回收重用）堆内存里去。*/
    if (len > 4 * 1024 * 1024) { free(buf); return -1; }
  }
  buf[len] = 0;
  if (trace_on()) fprintf(stderr, "[trace] hdr fd=%llu len=%zu hdr_end=%zu\n",
                         (unsigned long long)fd, len, hdr_end);
  /* method + path */
  if (sscanf(buf, "%7s %127s", req->method, req->path) != 2) { free(buf); return -1; }
  /* G17 根因修复：Content-Length 只能在**本请求的 header 段**里找。
   * 以前用 strstr(buf, ...)：buf 不是 C 字符串，负载一大就会撞上前一个 POST 请求
   * 在堆里残留的 "Content-Length: 105" 并把 cl 当成 105，然后 body 循环一直等
   * 永不会来的字节，导致 GET /health 挂到客户端超时断开为止（实测 g4_probe：
   * 12 并发 load 下 100% timeout_read，且挂的时长恰好等于客户端超时；空闲时
   * 因为堆是新拿的、恰好为 0 才看不出来）。CPU 版 src/http.c 早就是有界搜索。*/
  long cl = 0;
  for (char *p = buf; p < buf + hdr_end; ) {
    char *eol = strchr(p, '\n');
    if (!eol || eol >= buf + hdr_end) break;
    if (_strnicmp(p, "content-length:", 15) == 0) cl = atol(p + 15);
    p = eol + 1;
  }
  while ((long)(len - hdr_end) < cl) {
    if (len + 1 >= cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
    int n = recv(fd, buf + len, (int)(cap - len - 1), 0);
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

/* P0-3: 布尔字段（"ignore_eos": true）也可能是数字 */
static int json_true(const sgl_json *j, const char *key) {
  const sgl_json *v = sgl_json_get(j, key);
  if (!v) return 0;
  if (v->type == SGL_JSON_BOOL) return v->boolean;
  if (v->type == SGL_JSON_NUMBER) return v->num != 0.0;
  return 0;
}

/* 把一个 sgl_json 子对象序列化成 JSON 文本（grammar 需要 schema 的字节）*/
static int json_dump_into(const sgl_json *j, char *out, size_t cap, size_t *outlen);

static int json_dump_into(const sgl_json *j, char *out, size_t cap, size_t *outlen) {
  size_t o = 0;
  /* 递归序列化：用显式栈太繁，这里用简单递归（schema 嵌套不超过几十层）*/
  struct frame {
    const sgl_json *j;
    int idx;
  };
  struct frame st[64];
  int sp = 0;
  const sgl_json *cur = j;
#define PUT(ch)                       \
  do {                                \
    if (o + 2 > cap) return 0;        \
    out[o++] = (char)(ch);            \
  } while (0)
  for (;;) {
    if (!cur) return 0;
    switch (cur->type) {
      case SGL_JSON_NULL: { const char *s = "null"; while (*s) PUT(*s++); break; }
      case SGL_JSON_BOOL: { const char *s = cur->boolean ? "true" : "false"; while (*s) PUT(*s++); break; }
      case SGL_JSON_NUMBER: {
        char t[64];
        int n = snprintf(t, sizeof(t), "%.17g", cur->num);
        for (int i = 0; i < n; i++) PUT(t[i]);
        break;
      }
      case SGL_JSON_STRING: {
        PUT('"');
        for (const unsigned char *p = (const unsigned char *)(cur->str ? cur->str : ""); *p; p++) {
          if (*p == '"' || *p == '\\') { PUT('\\'); PUT(*p); }
          else if (*p < 0x20) { PUT('\\'); PUT('u'); PUT('0'); PUT('0'); PUT("0123456789abcdef"[*p >> 4]); PUT("0123456789abcdef"[*p & 15]); }
          else PUT(*p);
        }
        PUT('"');
        break;
      }
      case SGL_JSON_ARRAY:
      case SGL_JSON_OBJECT: {
        int is_obj = cur->type == SGL_JSON_OBJECT;
        if (cur->count == 0) { PUT(is_obj ? '{' : '['); PUT(is_obj ? '}' : ']'); break; }
        PUT(is_obj ? '{' : '[');
        st[sp].j = cur;
        st[sp].idx = 0;
        sp++;
        if (is_obj) {
          PUT('"');
          for (const char *p = cur->keys[0]; *p; p++) PUT(*p);
          PUT('"');
          PUT(':');
        }
        cur = cur->items[0];
        continue;
      }
    }
    /* 一个值写完了：看栈顶容器还有没有下一个 */
    for (;;) {
      if (sp == 0) { PUT(0); /* 收尾 NUL（PUT 已保证容量）*/
        *outlen = o - 1;
        if (*outlen < cap) out[*outlen] = 0;
        return 1;
      }
      struct frame *f = &st[sp - 1];
      int is_obj = f->j->type == SGL_JSON_OBJECT;
      f->idx++;
      if (f->idx >= f->j->count) {
        PUT(is_obj ? '}' : ']');
        sp--;
        continue;
      }
      PUT(',');
      if (is_obj) {
        PUT('"');
        for (const char *p = f->j->keys[f->idx]; *p; p++) PUT(*p);
        PUT('"');
        PUT(':');
      }
      cur = f->j->items[f->idx];
      break;
    }
  }
#undef PUT
}

/* t5 fix: send() on these sockets intermittently returns -1/WSAEWOULDBLOCK
 * mid-response (seen after burst load; the client then waits forever for the
 * missing body bytes). Retry short sends; bail on hard errors. */
static int send_all(SOCKET fd, const char *buf, int len) {
  int off = 0;
  int tries = 0;
  while (off < len) {
    int n = send(fd, buf + off, len - off, SGL_SEND_FLAGS);
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
    if (trace_on()) fprintf(stderr, "[trace] sent fd=%llu bytes=%d t=%llu\n", (unsigned long long)fd,
                           hl + bl, (unsigned long long)GetTickCount64());
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

/* shared tail: stream (SSE) or plain response for request id (t10 refactor). */
static int finish_request(SOCKET fd, int id, int is_stream, int chat, int openai);

/* build chatml prompt from messages array; returns malloc'd string.
 * t10: content may be an ARRAY of parts —
 *   [{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"data:..."}}]
 * Text parts append in order; the first image part's url is stashed (malloc'd)
 * into *img_url_out and "<image>" is appended as a marker (visual tokens are
 * inserted by the worker right after the whole prompt, LLaVA-style). */
static char *build_chatml(const sgl_json *j, char **img_url_out) {
  const sgl_json *msgs = sgl_json_get(j, "messages");
  if (img_url_out) *img_url_out = NULL;
  if (!msgs || msgs->type != SGL_JSON_ARRAY || msgs->count == 0) return NULL;
  /* P0-2: 模型自带 chat_template 时优先用它（字节级与 HF apply_chat_template 一致）*/
  if (E.jinja) {
    char *out = NULL;
    size_t outlen = 0;
    sgl_json *kw = NULL;
    /* chat_template_kwargs 展开到顶层 + tools 原样传（Qwen3 的 enable_thinking 走这里）*/
    const sgl_json *ck = sgl_json_get(j, "chat_template_kwargs");
    const sgl_json *tl = sgl_json_get(j, "tools");
    if ((ck && ck->type == SGL_JSON_OBJECT && ck->count > 0) ||
        (tl && tl->type == SGL_JSON_ARRAY)) {
      size_t cap = 1 << 16;
      char *b = (char *)malloc(cap);
      size_t len = 0;
      b[0] = 0;
      b[len++] = '{';
      b[len] = 0;
      int first = 1;
      if (ck && ck->type == SGL_JSON_OBJECT) {
        for (int i = 0; i < ck->count; i++) {
          size_t vl = 0;
          char *v = (char *)malloc(cap);
          if (!v) continue;
          if (!json_dump_into(ck->items[i], v, cap, &vl)) {
            free(v);
            continue;
          }
          size_t need = strlen(ck->keys[i]) + vl + 8;
          while (len + need > cap) {
            char *nb = (char *)realloc(b, cap * 2);
            if (!nb) break;
            b = nb;
            cap *= 2;
          }
          len += (size_t)snprintf(b + len, cap - len, "%s\"%s\":%s", first ? "" : ",", ck->keys[i], v);
          first = 0;
          free(v);
        }
      }
      if (tl && tl->type == SGL_JSON_ARRAY) {
        size_t vl = 0;
        char *v = (char *)malloc(cap);
        if (v && json_dump_into(tl, v, cap, &vl)) {
          size_t need = vl + 16;
          while (len + need > cap) {
            char *nb = (char *)realloc(b, cap * 2);
            if (!nb) break;
            b = nb;
            cap *= 2;
          }
          len += (size_t)snprintf(b + len, cap - len, "%s\"tools\":%s", first ? "" : ",", v);
          first = 0;
        }
        free(v);
      }
      b[len++] = '}';
      b[len] = 0;
      kw = sgl_json_parse(b, len);
      free(b);
    }
    int rc = sgl_jinja_render(E.jinja, msgs, 1, kw, &out, &outlen);
    if (kw) sgl_json_free(kw);
    if (rc == 0 && out && outlen > 0) {
      /* 带图时把 image_url 抄出来（worker 在后面插视觉 token）*/
      for (int i = 0; i < msgs->count && img_url_out; i++) {
        const sgl_json *c = sgl_json_get(msgs->items[i], "content");
        if (!c || c->type != SGL_JSON_ARRAY) continue;
        for (int k = 0; k < c->count; k++) {
          const sgl_json *iu = sgl_json_get(c->items[k], "image_url");
          const char *url = iu ? sgl_json_get_str(iu, "url", NULL) : NULL;
          if (url && !*img_url_out) {
            size_t ul = strlen(url) + 1;
            char *cp = (char *)malloc(ul);
            if (cp) memcpy(cp, url, ul);
            *img_url_out = cp;
          }
        }
      }
      /* 调用方（route）用 free() 释放 prompt，所以这里转成普通 malloc 缓冲 */
      char *dup = (char *)malloc(outlen + 1);
      if (dup) {
        memcpy(dup, out, outlen);
        dup[outlen] = 0;
      }
      sgl_xfree(out);
      return dup;
    }
    sgl_xfree(out);
    SGL_WARN("chat_template render failed, falling back to ChatML");
  }
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
    APPEND("<|im_start|>");
    APPEND(role);
    APPEND("\n");
    const sgl_json *content = sgl_json_get(m, "content");
    if (content && content->type == SGL_JSON_ARRAY) {
      /* t10 multimodal content parts */
      for (int k = 0; k < content->count; k++) {
        const sgl_json *part = content->items[k];
        const char *ty = sgl_json_get_str(part, "type", "");
        if (!strcmp(ty, "text")) {
          APPEND(sgl_json_get_str(part, "text", ""));
        } else if (!strcmp(ty, "image_url")) {
          const sgl_json *iu = sgl_json_get(part, "image_url");
          const char *url = iu ? sgl_json_get_str(iu, "url", NULL) : NULL;
          if (url && img_url_out && !*img_url_out) {
            size_t ul = strlen(url) + 1;
            *img_url_out = (char *)malloc(ul);
            memcpy(*img_url_out, url, ul);
          }
          APPEND("<image>");
        }
      }
    } else {
      APPEND(sgl_json_get_str(m, "content", ""));
    }
    APPEND("<|im_end|>\n");
  }
  APPEND("<|im_start|>assistant\n");
#undef APPEND
  return s;
}

/* P1: 单段容量不足的明确提示（以前只会看到 "generation failed"）*/
static const char *seg_full_msg(void) {
  static char msg[256];
  snprintf(msg, sizeof(msg),
           "prompt is %d tokens but one KV segment holds %d (SGL_CUDA_CTX / concurrency): "
           "raise SGL_CUDA_CTX or lower SGL_HTTP_CONC",
           E.last_prompt_tokens, E.seg_len);
  return msg;
}

/* P0-3: OpenAI / sglang 采样参数（与 CPU 版同一套语义）*/
static void parse_sampling(const sgl_json *j, sgl_sampling_params *sp) {
  sp->temperature = (float)sgl_json_get_num(j, "temperature", 0.0);
  sp->top_p = (float)sgl_json_get_num(j, "top_p", 0.0);
  sp->top_k = sgl_json_get_int(j, "top_k", 0);
  sp->min_p = (float)sgl_json_get_num(j, "min_p", 0.0);
  sp->repetition_penalty = (float)sgl_json_get_num(j, "repetition_penalty", 1.0);
  sp->frequency_penalty = (float)sgl_json_get_num(j, "frequency_penalty", 0.0);
  sp->presence_penalty = (float)sgl_json_get_num(j, "presence_penalty", 0.0);
  sp->ignore_eos = json_true(j, "ignore_eos");
  sp->seed = (uint64_t)sgl_json_get_int(j, "seed", 0);
  const sgl_json *st = sgl_json_get(j, "stop_token_ids");
  sp->n_stop_ids = 0;
  if (st && st->type == SGL_JSON_ARRAY)
    for (int i = 0; i < st->count && sp->n_stop_ids < SGL_MAX_STOP_TOKENS; i++)
      if (st->items[i]->type == SGL_JSON_NUMBER) sp->stop_ids[sp->n_stop_ids++] = (int)st->items[i]->num;
}

/* P1: response_format → grammar（json_object / json_schema / regex）*/
static sgl_grammar *make_grammar(const sgl_json *j) {
  const sgl_json *rf = sgl_json_get(j, "response_format");
  if (!rf || rf->type != SGL_JSON_OBJECT) return NULL;
  const char *rt = sgl_json_get_str(rf, "type", "text");
  if (!strcmp(rt, "json_object")) return sgl_grammar_from_json_schema("{\"type\":\"object\"}", 19);
  if (!strcmp(rt, "regex") || !strcmp(rt, "structural_tag")) {
    const char *pat = sgl_json_get_str(rf, "regex", sgl_json_get_str(rf, "pattern", NULL));
    return pat ? sgl_grammar_from_regex(pat) : NULL;
  }
  if (!strcmp(rt, "json_schema")) {
    const sgl_json *js = sgl_json_get(rf, "json_schema");
    const sgl_json *sc = js ? sgl_json_get(js, "schema") : NULL;
    if (!sc) return NULL;
    /* schema 子对象 → 文本：只支持标量/数组的短 schema；超大 schema 放弃约束 */
    size_t cap = 1 << 16;
    char *b = (char *)malloc(cap);
    if (!b) return NULL;
    size_t len = 0;
    int ok = json_dump_into(sc, b, cap, &len);
    sgl_grammar *g = ok ? sgl_grammar_from_json_schema(b, len) : NULL;
    free(b);
    return g;
  }
  return NULL;
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
    /* G5: health 说真话——降级（显存不够/上传失败）时 gpu:false + 原因，
     * 监控能直接看到，而不是只看到端口连不上。 */
    char body[512];
    if (E.degraded)
      snprintf(body, sizeof(body),
               "{\"ok\":false,\"gpu\":false,\"degraded\":true,\"reason\":\"%s\"}", E.gpu_err);
    else {
      /* G15: 命中率 / 省下的 prefill 行数一起暴露（命中率的分母是参与前缀比对的请求数，
       * 即命中+未命中；短于 PFX_MIN 的请求不计入，否则短的聊天请求会把命中率拉低）。*/
      long tot = g_pfx_hits + g_pfx_misses;
      snprintf(body, sizeof(body),
               "{\"ok\":true,\"gpu\":true,\"ctx\":%d,\"seg\":%d,\"conc\":%d,"
               "\"prefix\":{\"slots\":%d,\"hits\":%ld,\"misses\":%ld,\"hit_rate\":%.3f,"
               "\"hit_rows\":%ld,\"leased_rows\":%ld,\"flushes\":%ld}}",
               sgl_cuda_ctx(E.gpu), E.seg_len, ACTIVE_CONC,
               PFX_MAX_EFF, g_pfx_hits, g_pfx_misses,
               tot ? (double)g_pfx_hits / (double)tot : 0.0,
               g_pfx_hit_rows, g_pfx_leased_rows, g_pfx_flushes);
    }
    respond(fd, 200, "application/json", body);
    return 0;
  }
  if (strcmp(req->method, "POST") != 0) {
    /* P1: 只读监控端点（与 CPU 版同名）*/
    if (strcmp(req->method, "GET") == 0) {
      EnterCriticalSection(&E.lock);
      long pf = E.tot_prefill, dc = E.tot_decode, fin = E.tot_finished, pt = E.tot_prompt_tokens;
      int nact = E.n_active, nwq = E.n_waitq;
      LeaveCriticalSection(&E.lock);
      char b[1024];
      const char *mn = E.model_name[0] ? E.model_name : "sglang-c";
      if (!strcmp(req->path, "/v1/models")) {
        snprintf(b, sizeof(b),
                 "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
                 "\"owned_by\":\"sglang-c\"}]}",
                 mn);
      } else if (!strcmp(req->path, "/ready") || !strcmp(req->path, "/health_generate") ||
                 !strcmp(req->path, "/ping")) {
        snprintf(b, sizeof(b), "{\"ok\":true}");
      } else if (!strcmp(req->path, "/get_server_info") || !strcmp(req->path, "/server_info") ||
                 !strcmp(req->path, "/get_model_info")) {
        snprintf(b, sizeof(b),
                 "{\"model\":\"%s\",\"device\":\"cuda\",\"ctx\":%d,\"n_running\":%d,"
                 "\"n_waiting\":%d,\"prefill_tokens\":%ld,\"decode_tokens\":%ld,"
                 "\"n_finished\":%ld,\"gpu\":%s}",
                 mn, E.gpu ? sgl_cuda_ctx(E.gpu) : 0, nact, nwq, pf, dc, fin,
                 E.degraded ? "false" : "true");
      } else if (!strcmp(req->path, "/metrics")) {
        snprintf(b, sizeof(b),
                 "# TYPE sgl_prefill_tokens_total counter\nsgl_prefill_tokens_total %ld\n"
                 "# TYPE sgl_decode_tokens_total counter\nsgl_decode_tokens_total %ld\n"
                 "# TYPE sgl_requests_finished_total counter\nsgl_requests_finished_total %ld\n"
                 "# TYPE sgl_prompt_tokens_total counter\nsgl_prompt_tokens_total %ld\n"
                 "# TYPE sgl_requests_running gauge\nsgl_requests_running %d\n"
                 "# TYPE sgl_requests_waiting gauge\nsgl_requests_waiting %d\n"
                 "# TYPE sgl_prefix_hit_rows_total counter\nsgl_prefix_hit_rows_total %ld\n",
                 pf, dc, fin, pt, nact, nwq, g_pfx_hit_rows);
      } else {
        respond_error(fd, 404, "not found");
        return 0;
      }
      respond(fd, 200, "application/json", b);
      return 0;
    }
    respond_error(fd, 404, "not found");
    return 0;
  }
  int chat = strcmp(req->path, "/v1/chat/completions") == 0;
  int openai = strcmp(req->path, "/v1/completions") == 0;
  int gen = strcmp(req->path, "/generate") == 0;
  /* P1: 管理端点 */
  if (!strcmp(req->path, "/flush_cache")) {
    EnterCriticalSection(&E.lock);
    pfx_flush_all();
    LeaveCriticalSection(&E.lock);
    respond(fd, 200, "application/json", "{\"ok\":true}");
    return 0;
  }
  if (!strcmp(req->path, "/abort_request")) {
    sgl_json *aj = sgl_json_parse(req->body, req->body_len);
    int rid = aj ? sgl_json_get_int(aj, "rid", sgl_json_get_int(aj, "request_id", -1)) : -1;
    if (aj) sgl_json_free(aj);
    EnterCriticalSection(&E.lock);
    request *ar = find_req(rid);
    int ok = 0;
    if (ar && (ar->state == R_WAIT || ar->state == R_RUN)) {
      ar->state = R_DONE;
      strcpy_s(ar->finish, sizeof(ar->finish), "stop");
      ar->text_out = decode_gen(ar);
      ok = 1;
    }
    LeaveCriticalSection(&E.lock);
    if (!ok) { respond_error(fd, 404, "no such rid"); return 0; }
    respond(fd, 200, "application/json", "{\"ok\":true}");
    return 0;
  }
  if (!strcmp(req->path, "/update_weights_from_disk")) {
    /* GPU 引擎的权重在显存里，热更新要重建 CUDA 模型与所有段；这里明确拒绝，
     * 不假装成功（CPU 引擎的 /update_weights_from_disk 是可用实现）。*/
    respond_error(fd, 501, "reload is not supported by the cuda engine (restart sglc)");
    return 0;
  }
  if (!chat && !openai && !gen) { respond_error(fd, 404, "unknown path"); return 0; }
  /* G5: 降级模式——不排队、不假装有结果，直接 503 + 原因 */
  if (E.degraded || !E.gpu) {
    respond_error(fd, 503, E.gpu_err[0] ? E.gpu_err : "gpu engine unavailable");
    return 0;
  }

  sgl_json *j = sgl_json_parse(req->body, req->body_len);
  if (!j) { respond_error(fd, 400, "invalid JSON"); return 0; }

  char *prompt = NULL;
  int max_new = 64;
  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp));
  sp.repetition_penalty = 1.0f;
  int is_stream = 0;
  int want_lp = 0;
  sgl_grammar *gram = NULL;
  char *img_url = NULL;
  if (chat) {
    prompt = build_chatml(j, &img_url);
    if (!prompt) { sgl_json_free(j); respond_error(fd, 400, "messages required"); return 0; }
    if (img_url) {
      /* t10 multimodal: resolve the image now (outside the engine lock) */
      if (!E.clip) {
        free(prompt); free(img_url); sgl_json_free(j);
        respond_error(fd, 400, "model has no vision tower (clip_config.json missing)");
        return 0;
      }
      int n_img = 0;
      float *feats = vision_feats(img_url, &n_img);
      free(img_url);
      if (!feats) {
        free(prompt); sgl_json_free(j);
        respond_error(fd, 400, "bad image (need data:image/png|jpeg;base64,... or raw float32 CHW)");
        return 0;
      }
      int id2 = submit2(prompt, max_new, &sp, feats, n_img);
      free(prompt);
      sgl_json_free(j);
      if (id2 == -2) {
        respond_error(fd, 400, seg_full_msg());
        return 0;
      }
      if (id2 < 0) { respond_error(fd, 503, "server busy"); return 0; }
      return finish_request(fd, id2, is_stream, 1, 0);
    }
    max_new = sgl_json_get_int(j, "max_tokens", 256);
    parse_sampling(j, &sp);
    want_lp = json_true(j, "logprobs") || sgl_json_get_int(j, "top_logprobs", 0) > 0;
    gram = make_grammar(j);
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
    parse_sampling(j, &sp);
    want_lp = json_true(j, "logprobs") || sgl_json_get_int(j, "logprobs", 0) > 0;
    const sgl_json *sv = sgl_json_get(j, "stream");
    is_stream = sv && sv->type == SGL_JSON_BOOL && sv->boolean;
  }
  sgl_json_free(j);
  if (max_new <= 0 || max_new > MAX_NEW_CAP) max_new = 256;

  int id = submit2(prompt, max_new, &sp, NULL, 0);
  free(prompt);
  if (id == -2) {
    sgl_grammar_free(gram);
    respond_error(fd, 400, seg_full_msg());
    return 0;
  }
  if (id < 0) {
    sgl_grammar_free(gram);
    respond_error(fd, 503, "prompt too long (>64KB) or server busy");
    return 0;
  }
  /* 把 grammar / logprobs 挂到刚提交的请求上 */
  EnterCriticalSection(&E.lock);
  request *nr = find_req(id);
  if (nr) {
    nr->gram = gram;
    gram = NULL;
    nr->want_lp = want_lp;
    if (want_lp) nr->lp = (float *)sgl_xmalloc(sizeof(float) * (size_t)nr->max_new);
  }
  LeaveCriticalSection(&E.lock);
  sgl_grammar_free(gram);
  return finish_request(fd, id, is_stream, chat, openai);
}

/* shared tail: stream (SSE) or plain response for request id (t10 refactor). */
static int finish_request(SOCKET fd, int id, int is_stream, int chat, int openai) {

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
  (void)esc;
  /* P2: reasoning 拆分 + tool_call 提取（与 CPU 版同一套处理器）*/
  char *rtext = NULL;
  const char *body_txt = text ? text : "";
  int had_reason = sgl_parse_reasoning(body_txt, &rtext, &body_txt, NULL);
  int n_calls = 0;
  char *calls = sgl_parse_tool_calls(body_txt, &n_calls);
  char esc_body[8192], esc_reason[8192], esc_calls[8192];
  json_escape(body_txt, esc_body, sizeof(esc_body));
  if (had_reason && rtext) json_escape(rtext, esc_reason, sizeof(esc_reason));
  if (n_calls > 0 && calls) json_escape(calls, esc_calls, sizeof(esc_calls));
  const char *fin2 = n_calls > 0 ? "tool_calls" : fin;
  /* logprobs（OpenAI chat 形状）*/
  char lpbuf[8192];
  lpbuf[0] = 0;
  if (r && r->want_lp && r->lp && r->n_lp > 0) {
    size_t o = 0;
    o += (size_t)snprintf(lpbuf + o, sizeof(lpbuf) - o, ",\"logprobs\":{\"content\":[");
    for (int i = 0; i < r->n_lp && o + 64 < sizeof(lpbuf); i++)
      o += (size_t)snprintf(lpbuf + o, sizeof(lpbuf) - o, "%s{\"token\":\"\",\"logprob\":%.6f}",
                            i ? "," : "", (double)r->lp[i]);
    o += (size_t)snprintf(lpbuf + o, sizeof(lpbuf) - o, "]}");
  }
  size_t bcap = strlen(esc_body) + strlen(esc_reason) + strlen(esc_calls) + sizeof(lpbuf) + 1024;
  char *body = (char *)malloc(bcap);
  if (!body) { sgl_xfree(calls); sgl_xfree(rtext); free(text); reclaim(id); respond_error(fd, 500, "oom"); return 0; }
  if (chat) {
    /* message 字段逐段拼：content、可选 reasoning_content、可选 tool_calls */
    size_t o = 0;
    o += (size_t)snprintf(body + o, bcap - o, "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                                                    "\"content\":\"%s\"", esc_body);
    if (had_reason && rtext)
      o += (size_t)snprintf(body + o, bcap - o, ",\"reasoning_content\":\"%s\"", esc_reason);
    if (n_calls > 0 && calls)
      o += (size_t)snprintf(body + o, bcap - o, ",\"tool_calls\":%s", esc_calls);
    snprintf(body + o, bcap - o,
             "}%s,\"finish_reason\":\"%s\"}],"
             "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
             lpbuf, fin2, ptok, ngen, ptok + ngen);
  } else if (openai) {
    snprintf(body, bcap,
             "{\"choices\":[{\"text\":\"%s\",\"index\":0%s,\"finish_reason\":\"%s\"}],"
             "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
             esc_body, lpbuf, fin2, ptok, ngen, ptok + ngen);
  } else {
    snprintf(body, bcap,
             "{\"text\":\"\",\"completion\":\"%s\",\"meta_info\":{\"finish_reason\":\"%s\","
             "\"completion_tokens\":%d}}",
             esc_body, fin2, ngen);
  }
  sgl_xfree(calls);
  sgl_xfree(rtext);
  free(text);
  reclaim(id);
  respond(fd, 200, "application/json", body);
  free(body);
  return 0;
}

typedef struct { SOCKET fd; } conn_ctx;

/* G17 诊断：SGL_HTTP_TRACE=1 时把每个连接的进/出打一行。
 * 用来区分「/health 偶发不响应」到底是 conn_thread 没建起来（只有 accept 行、
 * 没有 conn enter），还是它卡在 read_request/route 里（有 enter 没 done）。*/
static int trace_on(void) {
  static int on = -1;
  if (on < 0) on = getenv("SGL_HTTP_TRACE") ? 1 : 0;
  return on;
}

SGL_THREAD_SIG(conn_thread) {
  conn_ctx *c = (conn_ctx *)arg;
  http_req req;
  /* t5 robustness: a stalled client (or a winsock internal hiccup seen under
   * burst load) could park recv() forever and leak the thread. Cap it. */
#ifdef _WIN32
  DWORD rto = 30000;
  setsockopt(c->fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&rto, sizeof(rto));
  DWORD sto = 5000; /* t5: a blocked send() must not park the thread forever */
  setsockopt(c->fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&sto, sizeof(sto));
  setsockopt(c->fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&(int){1}, sizeof(int));
#else
  /* POSIX：SO_RCVTIMEO/SO_SNDTIMEO 收的是 struct timeval（不是 DWORD 毫秒）*/
  struct timeval rto = {30, 0};
  setsockopt(c->fd, SOL_SOCKET, SO_RCVTIMEO, &rto, sizeof(rto));
  struct timeval sto = {5, 0};
  setsockopt(c->fd, SOL_SOCKET, SO_SNDTIMEO, &sto, sizeof(sto));
  int nd = 1;
  setsockopt(c->fd, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));
#endif
  if (trace_on()) fprintf(stderr, "[trace] conn enter fd=%llu t=%llu\n", (unsigned long long)c->fd,
                         (unsigned long long)GetTickCount64());
  if (read_request(c->fd, &req) == 0) {
    route(c->fd, &req);
    free(req.body);
  } else {
    fprintf(stderr, "[http] read_request failed (wsa=%d)\n", WSAGetLastError());
  }
  if (trace_on()) fprintf(stderr, "[trace] conn done  fd=%llu t=%llu\n", (unsigned long long)c->fd,
                         (unsigned long long)GetTickCount64());
  closesocket(c->fd);
  free(c);
  SGL_THREAD_RETURN;
}

/* 引擎入口：sglc.exe 内嵌调用（编宏 SGL_EMBED_CUDA_HTTP）与独立 cuda_http.exe 共用。
 * host 语义同 http.c：NULL / "127.0.0.1" / "localhost" 只听本机；
 * "0.0.0.0" / "any" / "*" 听全部网卡；其他按 IPv4 字面量解析（失败回退 loopback）。 */
int sgl_cuda_http_serve(const char *dir, int port, const char *host) {

#ifdef _WIN32
  WSADATA wsa;
  WSAStartup(MAKEWORD(2, 2), &wsa);
#else
  signal(SIGPIPE, SIG_IGN); /* 连接到已关闭的连接上 send()：不要因 SIGPIPE 被杀 */
#endif

  memset(&E, 0, sizeof(E));
  InitializeCriticalSection(&E.lock);
  InitializeConditionVariable(&E.cv_work);
  InitializeConditionVariable(&E.cv_step);

  double t0 = (double)clock() / CLOCKS_PER_SEC;
  E.cpu = sgl_llama_load(dir);
  if (!E.cpu) return 2;
  /* G5: 上传前预检。以前显存不够 = 进程直接死（cudaMalloc 失败 → return 2），
   * 监控/调用方只能看到连接被拒。现在先算账（权重 + KV(ctx) + 工作区 + 上下文
   * 开销），不够就不上传，服务照常起，/health 报 degraded，generate 返 503 + 原因。
   * Windows 上还有第二种更阴的情况：WDDM 把超出部分溢到系统内存，不报错但慢 3-20x
   * （实测 8K prefill 2.8s → 10.4s），所以预检留 192MB 余量。 */
  {
    const char *ce = getenv("SGL_CUDA_CTX");
    int cap = (ce && atoi(ce) > 0) ? atoi(ce) : 16384;
    int want = E.cpu->n_ctx < cap ? E.cpu->n_ctx : cap;
    const char *ec = getenv("SGL_HTTP_CONC");
    int conc = ACTIVE_CONC;
    if (ec) { int c = atoi(ec); if (c >= 1 && c <= MAX_CONC) conc = c; }
    int rows = want / conc;
    long need = sgl_cuda_estimate_vram_mb(E.cpu, want, rows);
    long avail = sgl_cuda_avail_vram_mb();
    printf("[engine] vram precheck: need ~%ldMB, free %ldMB (ctx=%d rows=%d)\n", need, avail, want, rows);
    fflush(stdout);
    if (need > 0 && avail > 0 && avail < need + 192) {
      snprintf(E.gpu_err, sizeof(E.gpu_err),
               "insufficient VRAM: model needs ~%ldMB, %ldMB free (lower SGL_CUDA_CTX or use SGL_CUDA_FP16=1/SGL_CUDA_W8=1)",
               need, avail);
      E.degraded = 1;
      fprintf(stderr, "[engine] DEGRADED: %s\n", E.gpu_err);
    }
  }
  if (!E.degraded) {
    E.gpu = sgl_cuda_from_llama(E.cpu);
    if (!E.gpu) {
      snprintf(E.gpu_err, sizeof(E.gpu_err),
               "GPU upload failed (out of VRAM); free %ldMB, try SGL_CUDA_CTX=8192 or SGL_CUDA_W8=1",
               sgl_cuda_avail_vram_mb());
      E.degraded = 1;
      fprintf(stderr, "[engine] DEGRADED: %s\n", E.gpu_err);
    }
  }
  if (E.degraded) E.seg_len = 0;
  char tpath[1024];
  snprintf(tpath, sizeof(tpath), "%s/tokenizer.json", dir);
  E.tok = sgl_tokenizer_load(tpath);
  if (!E.tok) return 2;
  E.vocab = E.cpu->vocab;
  E.eos = sgl_tokenizer_eos_id(E.tok);
  /* P0-2: 模型自带 chat_template（tokenizer_config.json）；P1: 响应体里的 model 字段 */
  E.jinja = sgl_jinja_load_dir(dir);
  {
    const char *b = dir;
    for (const char *p = dir; *p; p++)
      if (*p == '/' || *p == '\\') b = p + 1;
    snprintf(E.model_name, sizeof(E.model_name), "%s", b);
  }
  if (!E.jinja)
    SGL_INFO("cuda http: no chat_template in %s, falling back to ChatML", dir);
  /* t10: optional vision tower — <model_dir>/clip_config.json enables the
 * multimodal path (/v1/chat/completions with image_url parts). */
 {
   char cpath[1024];
   snprintf(cpath, sizeof(cpath), "%s/clip_config.json", dir);
   FILE *cf = NULL;
   fopen_s(&cf, cpath, "rb");
   if (cf) {
     fclose(cf);
     E.clip = sgl_clip_load(dir);
     if (E.clip) {
       snprintf(cpath, sizeof(cpath), "%s/projector.safetensors", dir);
       sgl_st_ctx *pst = sgl_st_open(cpath);
       sgl_tensor w1 = {0}, b1 = {0}, w2 = {0}, b2 = {0};
       if (pst) {
         /* G12: 自动判层数 —— 有 linear_2 就是两层 MLP（linear_1→GELU→linear_2）。
          * llava-1.5 的真实 mm_projector 就是两层：[4096,4096] + [4096,4096]。*/
         int64_t lm_e = E.cpu->n_embd;
         w2 = sgl_st_get(pst, "linear_2.weight", -1, NULL);
         if (w2.data && w2.ndim == 2) {
           int64_t mid = w2.shape[0];
           w1 = sgl_st_get(pst, "linear_1.weight", 2, (int64_t[]){mid, E.clip->proj_dim});
           b1 = sgl_st_get(pst, "linear_1.bias", 1, (int64_t[]){mid});
           b2 = sgl_st_get(pst, "linear_2.bias", 1, (int64_t[]){lm_e});
         } else {
           w2 = (sgl_tensor){0};
           w1 = sgl_st_get(pst, "linear_1.weight", 2, (int64_t[]){lm_e, E.clip->proj_dim});
           b1 = sgl_st_get(pst, "linear_1.bias", 1, (int64_t[]){lm_e});
         }
       }
       if (!w1.data || !b1.data) {
         fprintf(stderr, "[mm] projector.safetensors missing/mismatched — vision disabled\n");
         sgl_clip_free(E.clip); E.clip = NULL;
       } else {
         E.pj_w = w1.data; /* mmap-backed, lives as long as we keep pst open
                              (never closed — server lifetime) */
         E.pj_b = b1.data;
         E.pj_in = E.clip->proj_dim;
         if (w2.data) {
           E.pj_w2 = w2.data;
           E.pj_b2 = b2.data;
           E.pj_mid = (int)w2.shape[0];
         }
         if (sgl_projector_init(&E.pj, E.pj_w, E.pj_b, w2.data ? E.pj_mid : E.cpu->n_embd,
                                E.clip->proj_dim, E.pj_w2, E.pj_b2,
                                w2.data ? E.cpu->n_embd : 0, E.pj_mid) != 0) {
           fprintf(stderr, "[mm] projector shape mismatch — vision disabled\n");
           sgl_clip_free(E.clip); E.clip = NULL;
         } else {
           printf("[mm] vision tower ON: CLIP E=%d L=%d patch=%d NP=%d proj=%d -> lm E=%d "
                  "(%s projector)\n",
                  E.clip->E, E.clip->L, E.clip->P, E.clip->n_patch, E.clip->proj_dim,
                  E.cpu->n_embd, w2.data ? "2-layer MLP" : "1-layer linear");
         }
       }
     }
   }
 }
  {
    const char *ec = getenv("SGL_HTTP_CONC");
    if (ec) {
      int c = atoi(ec);
      if (c >= 1 && c <= MAX_CONC) ACTIVE_CONC = c;
    }
  }
  E.seg_len = E.gpu ? sgl_cuda_ctx(E.gpu) / ACTIVE_CONC : 0;
  /* G15(b): 运维杆 SGL_PFX_MAX —— 0 = 关掉前缀缓存，1..16 = 限租让表大小。 */
  {
    const char *pm = getenv("SGL_PFX_MAX");
    if (pm) {
      int v = atoi(pm);
      PFX_MAX_EFF = v < 0 ? 0 : (v > PFX_MAX ? PFX_MAX : v);
      printf("[engine] prefix cache: %s (SGL_PFX_MAX=%d)\n",
             PFX_MAX_EFF ? "on" : "OFF", PFX_MAX_EFF);
    }
  }
  /* G3: decode graphs 现在默认开。t5 当时强制关掉是因为：
   *   (1) 捕获失败会把一个进程级 static 置为“graphs off”，一个 B 的偶然失败
   *       就让所有 batch 永远走 plain（比 graph 慢 3-4x）—— 改成按 B 记录失败；
   *   (2) 服务端 ws_cap 随最长 prompt 增长 → invalidate 全部 graph → 在下一次
   *       decode 里重新捕获（实测每张 ~0.5s，夹在请求中间）—— 起服务前 prewarm
   *       把 ws_cap / batch 镜像一次撑到位，之后 graph 缓存全程有效；
   *   (3) 捕获时才开始 lazy-load 内核 / cublas 选算法，在 capture 里做这些事会
   *       失败 —— capture 前先跑一次 plain step 预热。
   * SGL_CUDA_GRAPH=0 仍可显式关闭。 */
  if (E.gpu) {
    int warmed = sgl_cuda_prewarm(E.gpu, E.seg_len, ACTIVE_CONC);
    printf("[engine] prewarm: ws rows=%d, decode graphs %d/%d\n", E.seg_len, warmed, ACTIVE_CONC);
  }
  printf("[engine] loaded in %.1fs; ctx=%d seg=%d conc=%d eos=%d%s\n",
         (double)clock() / CLOCKS_PER_SEC - t0, E.gpu ? sgl_cuda_ctx(E.gpu) : 0,
         E.seg_len, ACTIVE_CONC, E.eos, E.degraded ? " [DEGRADED]" : "");
  fflush(stdout);

#ifdef _WIN32
  HANDLE w = CreateThread(NULL, 0, worker, NULL, 0, NULL);
#else
  pthread_t w;
  pthread_create(&w, NULL, worker, NULL);
#endif

  SOCKET lst = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (lst == INVALID_SOCKET) {
    fprintf(stderr, "socket failed: %d\n", WSAGetLastError());
    return 2;
  }
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons((unsigned short)port);
  /* host → 监听地址（与 http.c 的 sgl_http_serve 保持一致）*/
  if (!host || !strcmp(host, "127.0.0.1") || !strcmp(host, "localhost"))
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  else if (!strcmp(host, "0.0.0.0") || !strcmp(host, "any") || !strcmp(host, "*"))
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  else if (inet_pton(AF_INET, host, &addr.sin_addr) != 1)
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
#ifndef _WIN32
  /* 重启服务时不被 TIME_WAIT 卡住（Windows 的 SO_REUSEADDR 语义不同，不动它）*/
  setsockopt(lst, SOL_SOCKET, SO_REUSEADDR, &(int){1}, sizeof(int));
#endif
  if (bind(lst, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
    fprintf(stderr, "bind failed: %d\n", WSAGetLastError());
    return 2;
  }
  listen(lst, 16);
  printf("[http] serving on %s:%d (/health /generate /v1/completions /v1/chat/completions)\n",
         host ? host : "127.0.0.1", port);
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
    if (trace_on()) fprintf(stderr, "[trace] accept fd=%llu t=%llu\n", (unsigned long long)c,
                           (unsigned long long)GetTickCount64());
#ifdef _WIN32
    HANDLE t = CreateThread(NULL, 0, conn_thread, ctx, 0, NULL);
    if (t) CloseHandle(t);
    else fprintf(stderr, "[http] CreateThread failed (fd=%llu, wsa=%d)\n",
                 (unsigned long long)c, WSAGetLastError());
#else
    pthread_t t;
    if (pthread_create(&t, NULL, conn_thread, ctx) == 0) pthread_detach(t);
    else fprintf(stderr, "[http] pthread_create failed (fd=%llu, errno=%d)\n",
                 (unsigned long long)c, errno);
#endif
  }
  E.stop = 1;
  WakeAllConditionVariable(&E.cv_work);
#ifdef _WIN32
  WaitForSingleObject(w, INFINITE);
#else
  pthread_join(w, NULL);
#endif
  return 0;
}

#ifndef SGL_EMBED_CUDA_HTTP
int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model_dir> [port=30001]\n", argv[0]);
    return 1;
  }
  return sgl_cuda_http_serve(argv[1], argc > 2 ? atoi(argv[2]) : 30001, argc > 3 ? argv[3] : NULL);
}
#endif
