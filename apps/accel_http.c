/* sgl accel_http.c - 后端无关的设备推理 HTTP 服务（W1：原 apps/cuda_http.c）
 * endpoints: GET /health, POST /generate, POST /v1/completions,
 *            POST /v1/chat/completions (chatml template; stream via SSE)
 * engine: KV-segment scheduler - each request gets a segment of
 *         acc_ctx() (seg = ctx / MAX_CONC), worker batches decode via
 *         acc_decode_batch; prefill via acc_forward_seg.
 * W1：设备侧调用全部经 include/sgl/accel.h 的 sgl_accel_api 函数表（下面的 A），
 * CUDA / MLX 各提供一张；本文件不含任何后端专有代码，main 在后端薄壳里。
 * usage: 由 apps/cuda_http.c（或 apps/mlx_http.c）传入表并起服务
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
/* Sleep((DWORD)d) 这类调用点用的是 Win32 的类型名；POSIX 下补一个同宽别名，
 * 免得每个调用点都要 #ifdef（本文件只在 Windows 上编过，MLX 后端首次在 macOS 编）。*/
typedef unsigned long DWORD;
/* GetTickCount64：单调毫秒（Windows 上是开机毫秒，只用于打时间戳/造种子） */
static unsigned long long sgl_tick_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (unsigned long long)ts.tv_sec * 1000ULL + (unsigned long long)(ts.tv_nsec / 1000000L);
}
#define GetTickCount64() sgl_tick_ms()
/* 单调微秒，给性能诊断用 */
static double sgl_now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}
/* 线程入口签名：Win32 = DWORD WINAPI f(LPVOID)，POSIX = void *f(void *) */
#define SGL_THREAD_SIG(name) static void *name(void *arg)
#define SGL_THREAD_RETURN return NULL
#endif

#ifdef _WIN32
#define SGL_SEND_FLAGS 0 /* winsock 无 MSG_NOSIGNAL */
#define SGL_THREAD_SIG(name) static DWORD WINAPI name(LPVOID arg)
#define SGL_THREAD_RETURN return 0
/* 单调微秒，给性能诊断用（POSIX 版在上面的 #else 里）*/
static double sgl_now_us(void) {
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart * 1e6 / (double)f.QuadPart;
}
#endif

#include <math.h> /* P1: grammar 掩码用 -INFINITY */

#include "sgl/alloc.h"
#include "sgl/bert.h" /* GPU 服务也能直接服务 embedding/rerank 模型 */
#include "sgl/deepseek.h" /* W3: DeepSeek（MLA+MoE）走设备独立路径 */
#include "sgl/bpe.h"
#include "sgl/clip.h"
#include "sgl/accel.h" /* W1: 后端函数表（原 sgl/cuda.h）*/
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

/* ================= W1：后端入口 =================
 * 所有设备侧调用都经 A（活动后端的函数表）。可选入口为 NULL 时在这里统一降级，
 * 调用点不必各自判空：前缀缓存关、投机关、图像请求拒、hybrid 模型拒。
 * 包装函数名与原来的后端专有名字一一对应，便于对照 git 历史。*/
static const sgl_accel_api *A;

#define ACC_OPT(fn, ret) \
  do { if (!A || !A->fn) return (ret); } while (0)

int sgl_accel_api_check(const sgl_accel_api *api, char *err, int errlen) {
#define ACC_REQ(f) \
  if (!api->f) { if (err && errlen > 0) snprintf(err, (size_t)errlen, "accel: missing required entry \"" #f "\""); return -1; }
  if (!api) { if (err && errlen > 0) snprintf(err, (size_t)errlen, "accel: NULL api"); return -1; }
  if (api->abi != SGL_ACCEL_ABI) {
    if (err && errlen > 0)
      snprintf(err, (size_t)errlen, "accel: abi mismatch (backend %d, server %d)", api->abi, SGL_ACCEL_ABI);
    return -1;
  }
  ACC_REQ(name) ACC_REQ(from_llama) ACC_REQ(free) ACC_REQ(ctx) ACC_REQ(forward_seg)
  ACC_REQ(decode_batch) ACC_REQ(estimate_mem_mb) ACC_REQ(avail_mem_mb)
#undef ACC_REQ
  return 0;
}

static sgl_accel *acc_from_llama(const void *m) { return A->from_llama(m); }
static int acc_ctx(const sgl_accel *m) { return A->ctx(m); }
static int acc_forward_seg(sgl_accel *m, const int *t, int n, int p0, int kb, float *lg) {
  return A->forward_seg(m, t, n, p0, kb, lg);
}
static int acc_forward_seg_emb(sgl_accel *m, const float *e, int n, int p0, int kb, float *lg) {
  ACC_OPT(forward_seg_emb, -1);
  return A->forward_seg_emb(m, e, n, p0, kb, lg);
}
static int acc_decode_batch(sgl_accel *m, const int *t, const int *bp, const int *l, int B, float *lg) {
  return A->decode_batch(m, t, bp, l, B, lg);
}
static int acc_copy_kv(sgl_accel *m, int s, int d, int n) {
  ACC_OPT(copy_kv, -1);
  return A->copy_kv(m, s, d, n);
}
static long acc_estimate_mem_mb(const void *m, int ctx, int rows) {
  ACC_OPT(estimate_mem_mb, -1);
  return A->estimate_mem_mb(m, ctx, rows);
}
static long acc_avail_mem_mb(void) {
  ACC_OPT(avail_mem_mb, -1);
  return A->avail_mem_mb();
}
static int acc_prewarm(sgl_accel *m, int rows, int max_b) {
  ACC_OPT(prewarm, -1);
  return A->prewarm(m, rows, max_b);
}
/* 递归状态槽：没有这个能力的后端返回 0 槽 → 上层据此打开前缀缓存 */
static int acc_lin_slots(const sgl_accel *m) { return A->lin_slots ? A->lin_slots(m) : 0; }
static void acc_set_lin_slot(sgl_accel *m, int s) { if (A->set_lin_slot) A->set_lin_slot(m, s); }
static void acc_set_decode_slots(sgl_accel *m, const int *s, int B) {
  if (A->set_decode_slots) A->set_decode_slots(m, s, B);
}
/* MTP / 投机解码：整组可选，任一为 NULL 就当特性关闭 */
static int acc_mtp_load(sgl_accel *m, int kv_ctx) {
  ACC_OPT(mtp_load, -1);
  return A->mtp_load(m, kv_ctx);
}
static int acc_mtp_ready(const sgl_accel *m) { return A->mtp_ready ? A->mtp_ready(m) : 0; }
static void acc_mtp_set_base(sgl_accel *m, int b) { if (A->mtp_set_base) A->mtp_set_base(m, b); }
static int acc_mtp_set_hidden(sgl_accel *m, int row) {
  ACC_OPT(mtp_set_hidden, -1);
  return A->mtp_set_hidden(m, row);
}
static int acc_mtp_drafts(sgl_accel *m, int t0, int pos, int W, float minp, int *out, float *pr) {
  ACC_OPT(mtp_drafts, 0);
  return A->mtp_drafts(m, t0, pos, W, minp, out, pr);
}
static const float *acc_mtp_hprev(const sgl_accel *m) { return A->mtp_hprev ? A->mtp_hprev(m) : NULL; }
static const float *acc_mtp_hlast(const sgl_accel *m) { return A->mtp_hlast ? A->mtp_hlast(m) : NULL; }
static int acc_mtp_fill_kv(sgl_accel *m, const int *t, int n, int p0, const float *hp) {
  ACC_OPT(mtp_fill_kv, -1);
  return A->mtp_fill_kv(m, t, n, p0, hp);
}
static int acc_spec_begin(sgl_accel *m, int W) {
  ACC_OPT(spec_begin, -1);
  return A->spec_begin(m, W);
}
static int acc_spec_verify(sgl_accel *m, const int *t, int n, int p0, int kb, float *rows, int *k) {
  ACC_OPT(spec_verify, -1);
  return A->spec_verify(m, t, n, p0, kb, rows, k);
}
static int acc_spec_commit(sgl_accel *m, int slot, int k) {
  ACC_OPT(spec_commit, -1);
  return A->spec_commit(m, slot, k);
}
static int acc_mtp_accept_test(sgl_accel *m, const int *p, int np, int ng, double *acc, int *nok) {
  ACC_OPT(mtp_accept_test, -1);
  return A->mtp_accept_test(m, p, np, ng, acc, nok);
}
/* ================= W1：后端入口结束 ================= */

/* M0: 从 8 提到 32 —— batch GDN 已经把 hybrid 模型的 B>1 decode 真正并行化
 * （实测聚合 36.8 → 141.6 tok/s @B=8），并发上限不再由 kernel 串行决定。
 * 需配合 SGL_LIN_SLOTS >= 并发数（每个请求一个 GDN 递归状态槽）。*/
#define MAX_CONC 32
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
  int stream_text;     /* 1 = /v1/completions：SSE 块用 OpenAI text_completion 形状（choices[0].text）*/
  int n_streamed;
  int prio;            /* W2: 优先级（大的先跑；同优先级 FIFO）*/
  int spec_ready;      /* M3: 已把 prefill 末位 hidden 存进 mtp.hlast，可开投机 */
  struct request *next_done;
  struct request *next_q; /* t5: wait-queue link */
} request;

typedef struct {
  sgl_llama *cpu;
  sgl_accel *gpu;      /* W1: 后端实例（CUDA 侧即 sgl_cuda_model），只经 A 这张函数表调用 */
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
  int pfx_ok; /* t11：hybrid 模型关闭前缀复用（见启动处）*/
  volatile int paused;   /* W2: /pause_generation——worker 不推进 decode，请求与 KV 都保留 */
  int step_delay_ms;     /* W2: /slow_down 注入的人工延时（每 step 前睡这么多毫秒）*/
  int profiling;         /* W2: /start_profile 的状态位（本引擎无 kernel 级 profiler，只记状态）*/
  sgl_bert *bert;        /* 模型目录是 BERT/RoBERTa 时走 embedding 服务（无 KV/无设备）*/
  char model_dir[1024];  /* W2: 模型目录（/model_info、WordPiece 分词用）*/
  /* W3: DeepSeek（MLA + MoE）——设备侧是另一套结构，单请求阻塞生成 */
  void *ds_cpu;          /* sgl_deepseek*（CPU 侧加载的权重，设备实例从中上传）*/
  void *ds;              /* sgl_cuda_ds*（设备实例）*/
  CRITICAL_SECTION ds_lock; /* 单一序列：一次只跑一个生成 */
  float *ds_logits;      /* [vocab] */
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
                   float *img_feats, int n_img, int prio) {
  /* t11: cap prompt length before the O(n^2)-ish BPE merge — a pathological
   * 200KB single-char body used to park encode for minutes (server hang). */
  /* t11: 上限是【字节】不是 token。之前 64KB 对中文太小——1 个汉字 3 字节，
   * 64KB 只装得下约 2.2 万汉字（≈1.4 万 token），一到大上下文就提前 503。
   * 真正的隐患是病态输入的长 BPE 合并（海量单字符），所以保留一个字节上限，
   * 但放到 1MB（≈ 33 万汉字），足够 256K 上下文的普通文本。*/
  if (!text || strlen(text) > (size_t)(1u << 20)) { sgl_xfree(img_feats); return -1; }
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
  r->prio = prio;
  r->state = R_WAIT;
  int placed = 0;
  if (E.n_active < ACTIVE_CONC) {
    for (int i = 0; i < MAX_CONC; i++)
      if (!E.reqs[i]) { E.reqs[i] = r; placed = 1; break; }
    E.n_active++;
  }
  if (!placed) {
    r->next_q = NULL;
    /* W2: 按优先级插队（大的在前；同优先级保持 FIFO，插在最后一个同优先级之后）*/
    if (!E.waitq_head) {
      E.waitq_head = E.waitq_tail = r;
    } else if (E.waitq_tail->prio >= r->prio) {
      E.waitq_tail->next_q = r;
      E.waitq_tail = r;
    } else {
      request *prev = NULL, *cur = E.waitq_head;
      while (cur && cur->prio >= r->prio) { prev = cur; cur = cur->next_q; }
      r->next_q = cur;
      if (prev) prev->next_q = r;
      else E.waitq_head = r;
      if (!cur) E.waitq_tail = r;
    }
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

static int spec_try_step(request *r, int W); /* M3，定义在 worker 之后 */

/* SGL_SPEC_W（1..8 才算开；0 = 关）。prefill 据此决定要不要回填 MTP 的 KV */
static int spec_w_env(void) {
  static int w = -1;
  if (w < 0) {
    const char *sw = getenv("SGL_SPEC_W");
    w = sw ? atoi(sw) : 0;
    if (w < 0 || w > 8) w = 0;
  }
  return w;
}

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
        if (E.gpu && E.pfx_ok && r->reached_run && r->seg == i && r->n_prompt >= PFX_MIN)
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
    /* W2: /pause_generation——不推进 prefill/decode，请求与已算好的 KV 全部留着；
     * /continue_generation 唤醒后从原位继续。 */
    if (E.paused) {
      SleepConditionVariableCS(&E.cv_work, &E.lock, 50);
      LeaveCriticalSection(&E.lock);
      continue;
    }
    /* W2: /slow_down 注入的人工延时（压力测试 / 让出带宽用）*/
    if (E.step_delay_ms > 0) {
      int d = E.step_delay_ms;
      LeaveCriticalSection(&E.lock);
      Sleep((DWORD)d); /* Windows 是 API；POSIX 由文件头的 #define 转成 nanosleep */
      EnterCriticalSection(&E.lock);
      if (E.stop) { LeaveCriticalSection(&E.lock); break; }
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
      if (E.gpu && r->n_prompt >= PFX_MIN && E.pfx_ok) {
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
            if (acc_copy_kv(E.gpu, src_seg * E.seg_len, base, hit) != 0) hit = 0;
          }
          if (hit > 0) { g_pfx_hits++; g_pfx_hit_rows += hit; off = hit; }
          else g_pfx_misses++;
        }
      } else if (E.gpu) {
        g_pfx_misses++;
      }
      /* t11：告诉引擎本序列用哪个线性注意力递归状态槽（= 段号）。
       * 递归状态不能像 KV 那样按位置重建，不设就会全用槽 0 而互相覆盖。*/
      acc_set_lin_slot(E.gpu, i);
      acc_mtp_set_base(E.gpu, base); /* MTP 自有 KV 与主 KV 同段 */
      if (getenv("SGL_HTTP_DBG"))
        fprintf(stderr, "[pf] id=%d seg=%d slot=%d nprompt=%d off=%d\n",
                r->id, i, i, r->n_prompt, off);
      if (r->n_prompt + r->n_img > r->seg_len) {
        rc = -2; /* prompt (+visual tokens) exceeds its KV segment capacity */
        pfx_flush_all(); /* G15(b): 现场已经不干净（本请求要的比一段还长），
                          * 把租让全丢掉，别让后续请求命中半截。*/ 
      } else {
        rc = 0;
        /* t11：真正分块 prefill。原来 chunk = 整段剩余 prompt，激活工作区要
         * 按「整段的同时行数」分配（约 405 KiB/行）——想要 32K 上下文就要
         * 13 GiB 工作区，必然 OOM。现在每次最多 SGL_PREFILL_CHUNK 行（默认
         * 1024，约 405 MiB 工作区），KV 按 base+pos0 累加、递归状态不清零
         * （只有 pos0==0 的那一块才清），结果与一次算完等价。*/
        static int pch = 0;
        if (pch <= 0) {
          const char *ce = getenv("SGL_PREFILL_CHUNK");
          pch = (ce && atoi(ce) > 0) ? atoi(ce) : 1024;
          if (pch > 8192) pch = 8192;
        }
        /* M3：投机要用 MTP 自有 KV，prompt 段得逐块回填（只算 K/V，开销远小于 prefill）。
         * 前缀命中时 [0, off) 没有主模型 hidden 可用，那几行沿用旧值，只影响草稿质量。*/
        int mtp_fill = spec_w_env() > 0 && acc_mtp_ready(E.gpu);
        int first_chunk = 1;
        while (off < r->n_prompt && rc == 0) {
          int chunk = r->n_prompt - off;
          if (chunk > pch) chunk = pch;
          /* only the last chunk needs the lm_head pass + logits D2H */
          float *dst = (off + chunk >= r->n_prompt && r->n_img == 0) ? r->logits : NULL;
          rc = acc_forward_seg(E.gpu, r->prompt + off, chunk, off, base, dst);
          if (rc == 0 && mtp_fill &&
              acc_mtp_fill_kv(E.gpu, r->prompt + off, chunk, off,
                                   first_chunk ? NULL : acc_mtp_hprev(E.gpu)) != 0)
            mtp_fill = 0;
          first_chunk = 0;
      E.tot_prefill += chunk;
          off += chunk;
        }
        /* t10: visual tokens go AFTER the text prompt, writing KV at
         * [n_prompt, n_prompt+n_img) via the embedding-level forward; their
         * last position produces the logits for the first sampled token. */
        if (rc == 0 && r->n_img > 0) {
          rc = acc_forward_seg_emb(E.gpu, r->img_feats, r->n_img,
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
      int bt[MAX_CONC], bb[MAX_CONC], bl[MAX_CONC], bs[MAX_CONC];
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
        bs[nb] = r->seg;   /* t11：本行的线性注意力状态槽 */
        br[nb] = r;
        nb++;
      }
      if (nb > 0) {
        /* M3：单请求 + 贪心 → 投机解码。其它情况走原路（批量/采样都不受影响）。*/
        int did_spec = 0;
        {
          static int spec_on = -1, spec_W = 0;
          if (spec_on < 0) {
            const char *sw = getenv("SGL_SPEC_W");
            spec_W = sw ? atoi(sw) : 0;
            spec_on = 0;
            if (spec_W > 0 && spec_W <= 8) {
              if (acc_spec_begin(E.gpu, spec_W) == 0) spec_on = 1;
              else { fprintf(stderr, "[spec] snapshot alloc failed -> spec off\n"); spec_W = 0; }
            } else spec_W = 0;
            if (spec_W > 0) fprintf(stderr, "[spec] W=%d chain=%s\n", spec_W,
                                    getenv("SGL_MTP_CHAIN") ? getenv("SGL_MTP_CHAIN") : "post");
          }
          request *r0 = br[0];
          if (spec_on && nb == 1 && acc_mtp_ready(E.gpu) &&
              r0->sp.temperature <= 0.0f && r0->sp.top_k <= 1 && !r0->gram &&
              !r0->want_lp && r0->n_gen > 0)
            did_spec = spec_try_step(r0, spec_W);
        }
        if (!did_spec) {
        acc_set_decode_slots(E.gpu, bs, nb);
        if (getenv("SGL_HTTP_DBG")) {
          fprintf(stderr, "[dec] nb=%d", nb);
          for (int q = 0; q < nb; q++)
            fprintf(stderr, " | tok=%d base=%d len=%d slot=%d", bt[q], bb[q], bl[q], bs[q]);
          fprintf(stderr, "\n");
        }
        /* P3 诊断（SGL_HTTP_PROF=1）：把每个 token 的时间拆成「前向」和「前向之外」
         * 两段——前向外的 gap 包含采样、logits 拷贝、以及为流式公平性加的
         * 200μs nanosleep。不先量出这个，就会把时间花在错的地方。*/
        static int hp_on = -1;
        static double hp_prev = 0.0, hp_dec = 0.0, hp_gap = 0.0;
        static long hp_n = 0;
        if (hp_on < 0) { const char *e2 = getenv("SGL_HTTP_PROF"); hp_on = e2 ? atoi(e2) : 0; }
        double hp_t0 = hp_on ? sgl_now_us() : 0.0;
        if (hp_on && hp_prev > 0.0) hp_gap += hp_t0 - hp_prev;
        int rc = acc_decode_batch(E.gpu, bt, bb, bl, nb, batch_logits);
        if (hp_on) {
          double hp_t1 = sgl_now_us();
          hp_dec += hp_t1 - hp_t0;
          hp_prev = hp_t1;
          if (++hp_n % 16 == 0) {
            fprintf(stderr, "[hprof] n=%ld  dec %.2f ms  gap %.2f ms (per token)\n",
                    hp_n, hp_dec / 16000.0, hp_gap / 16000.0);
            hp_dec = hp_gap = 0.0;
          }
        }
        if (rc != 0) {
          for (int b = 0; b < nb; b++) br[b]->state = R_FAIL;
        } else {
          for (int b = 0; b < nb; b++) {
            memcpy(br[b]->logits, batch_logits + (size_t)b * E.vocab,
                   sizeof(float) * E.vocab);
            br[b]->has_logits = 1;
          }
        }
        } /* !did_spec */
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
    /* t11：让出 CPU。worker 每轮 decode 只前进 1 个 token，释放锁后立刻重抢，
     * 等待中的 HTTP 线程（wait_done）几乎永远抢不到锁 → SSE 流式变成
     * “生成完毕才一次性吐完”（实测首字延迟 = 全部生成时间）。这里给等待者
     * 一个 200μs 窗口（33 tok/s 下每 token 多 0.2ms，可忽略）。*/
#ifndef _WIN32
    { struct timespec _ts = {0, 200000}; nanosleep(&_ts, NULL); }
#else
    Sleep(0);
#endif
  }
  free(batch_logits);
  SGL_THREAD_RETURN;
}

/* ---------------- http plumbing ---------------- */

typedef struct { char *body; size_t body_len; char method[8]; char path[128]; } http_req;

static int trace_on(void); /* G17 诊断开关（定义在 conn_thread 上方）*/

/* M3：一次投机迭代。调用时机 = 普通 decode 前向的位置（2b）。
 *
 * 约定与普通路径完全一致，只是把「一次前向出 1 个 token」变成「一次前向出
 * k+1 个」：
 *   - 进入时 r->gen 末位是已经采样好的 t0，r->logits 是 t0 之后的预测；
 *   - 草稿 d1..dW 由 MTP 链式拟出；（hidden 取最后一个落定位置）
 *   - 一次 verify 前向（n=W+1 行，带因果掩码）拿到每行 logits；
 *   - 取最长匹配前缀 k，追加 d1..dk，把 GDN 状态回滚到 k+1 行；
 *   - r->logits 换成第 k 行（= 最后一个落定位置之后的预测），下一轮继续。
 *
 * 返回 1 = 已推进（调用方跳过普通 decode）；0 = 没做（降级到原路）。*/
static int spec_try_step(request *r, int W) {
  static float *rows = NULL;
  static int rows_cap = 0;
  int i, k = 0, nd, room, pos, stop_at = -1;
  int drafts[16], toks[16];
  if (!r || r->n_gen <= 0 || !r->logits) return 0;
  if (!r->spec_ready) {
    if (acc_mtp_set_hidden(E.gpu, -1) != 0) return 0;
    r->spec_ready = 1;
  }
  pos = r->n_prompt + r->n_img + r->n_gen - 1; /* 最后一个已落定 token 的位置 */
  room = r->seg_len - pos - 1;                 /* 还能写几个位置 */
  if (room < 1) return 0;
  if (W > room - 1) W = room - 1;
  /* 草稿全收后 n_gen 不能超过 max_new（gen[] 是定长数组；普通路径到 max_new 即停）*/
  if (W > r->max_new - r->n_gen) W = r->max_new - r->n_gen;
  if (W > 15) W = 15;
  if (W < 1) return 0;
  if (rows_cap < (W + 1) * E.vocab) {
    float *nr = (float *)realloc(rows, sizeof(float) * (size_t)(W + 1) * E.vocab);
    if (!nr) return 0;
    rows = nr;
    rows_cap = (W + 1) * E.vocab;
  }
  /* 验收用：SGL_SPEC_CORRUPT=1 每步把第 (step % nd) 个草稿改坏，逼出各个 k 上的
   * 拒绝 + 回滚；与正常投机应在相同 token 处（off 的近 tie 处）与 off 分叉。
   * SGL_SPEC_STATS=1 逐步打印 k、各草稿概率与耗时。
   * SGL_SPEC_MINP：草稿链整体置信度低于它就提前停（0 = 固定拟 W 个，W 此时是上限）。
   * 27B 实测 W=6/0.3 比 W=4/0 每个 prompt 都慢 3–9%（MTP 置信度对接受与否预测力弱），默认关。*/
  static int spec_corrupt = -1, spec_stats = -1;
  static float spec_minp = 0.0f;
  static long spec_steps = 0, spec_toks = 0;
  float probs[16];
  double t_d0, t_d1, t_v1, t_c1;
  if (spec_corrupt < 0) {
    spec_corrupt = getenv("SGL_SPEC_CORRUPT") ? atoi(getenv("SGL_SPEC_CORRUPT")) : 0;
    spec_stats = getenv("SGL_SPEC_STATS") ? atoi(getenv("SGL_SPEC_STATS")) : 0;
    spec_minp = getenv("SGL_SPEC_MINP") ? (float)atof(getenv("SGL_SPEC_MINP")) : 0.0f;
  }
  /* 并发时别的请求的 prefill 会改 MTP 段基址，每步按本请求的段重设 */
  acc_mtp_set_base(E.gpu, r->seg * E.seg_len);
  toks[0] = r->gen[r->n_gen - 1];
  t_d0 = spec_stats ? sgl_now_us() : 0.0;
  nd = acc_mtp_drafts(E.gpu, toks[0], pos, W, spec_minp, drafts, spec_stats ? probs : NULL);
  if (nd < 1) return 0;
  t_d1 = spec_stats ? sgl_now_us() : 0.0;
  if (spec_corrupt) {
    int c = (int)(spec_steps % nd);
    drafts[c] = (drafts[c] + 1) % E.vocab;
  }
  for (i = 0; i < nd; i++) toks[i + 1] = drafts[i];
  if (acc_spec_verify(E.gpu, toks, nd + 1, pos, r->seg * E.seg_len, rows, &k) != 0)
    return 0;
  t_v1 = spec_stats ? sgl_now_us() : 0.0;
  if (k > nd) k = nd;
  /* 停词：出现在第 i 个草稿就只收前 i 个（与普通路径一致，EOS 本身不入 gen）*/
  for (i = 0; i < k; i++) {
    if ((drafts[i] == E.eos && !r->sp.ignore_eos) || sgl_sampling_is_stop(&r->sp, drafts[i])) {
      stop_at = i;
      break;
    }
  }
  if (stop_at >= 0) k = stop_at;
  /* 被接受的草稿行，MTP KV 里是链式 hidden 算的；用 verify 的真 hidden 重写
   *（commit 会改 hlast，所以要在它之前：此刻 hlast 仍是 hidden[pos-1]）*/
  if (k > 0) acc_mtp_fill_kv(E.gpu, toks, k + 1, pos, acc_mtp_hlast(E.gpu));
  if (acc_spec_commit(E.gpu, r->seg, k) != 0) return 0;
  spec_steps++;
  spec_toks += k + 1;
  if (spec_stats) {
    char pb[160];
    int o = 0;
    t_c1 = sgl_now_us();
    for (i = 0; i < nd && o < (int)sizeof(pb) - 8; i++)
      o += snprintf(pb + o, sizeof(pb) - (size_t)o, "%s%.3f", i ? "," : "", probs[i]);
    fprintf(stderr, "[spec] step=%ld pos=%d nd=%d k=%d tok/step=%.3f p=%s ms=%.2f/%.2f/%.2f\n",
            spec_steps, pos, nd, k, (double)spec_toks / (double)spec_steps, pb,
            (t_d1 - t_d0) / 1000.0, (t_v1 - t_d1) / 1000.0, (t_c1 - t_v1) / 1000.0);
  }
  for (i = 0; i < k; i++) r->gen[r->n_gen++] = drafts[i];
  memcpy(r->logits, rows + (size_t)k * E.vocab, sizeof(float) * E.vocab);
  r->has_logits = 1;
  E.tot_decode += (long)(k + 1);
  if (stop_at >= 0) {
    r->state = R_DONE;
    strcpy_s(r->finish, sizeof(r->finish), "stop");
    r->text_out = decode_gen(r);
    E.tot_finished++;
  }
  return 1;
}

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

/* 一个 token 的 SSE 块：/v1/completions 用 text_completion 形状，chat 与 /generate 用 delta */
static int sse_token_chunk(const request *r, const char *esc, char *chunk, size_t cap) {
  if (r->stream_text)
    return snprintf(chunk, cap,
                    "data: {\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"%s\","
                    "\"finish_reason\":null}]}\n\n",
                    esc);
  return snprintf(chunk, cap,
                  "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                  esc);
}

/* wait until request id reaches DONE/FAIL; optionally stream SSE meanwhile */
static int wait_done(SOCKET fd, int id, char **text, request **out_r) {
  static int dbgw = -1;
  unsigned long long t_w0 = GetTickCount64();
  int nwake = 0;
  for (;;) {
    EnterCriticalSection(&E.lock);
    if (dbgw < 0) { const char *e = getenv("SGL_HTTP_DBG"); dbgw = e ? atoi(e) : 0; }
    request *r = find_req(id);
    if (!r) { LeaveCriticalSection(&E.lock); return -1; }
    if (dbgw > 0 && (nwake < 8 || (r->state == R_DONE)))
      fprintf(stderr, "[wait] t=%llums #%d id=%d state=%d gen=%d streamed=%d streaming=%d\n",
              GetTickCount64() - t_w0, nwake, id, r->state, r->n_gen, r->n_streamed,
              r->streaming);
    nwake++;
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
          int n = sse_token_chunk(r, esc, chunk, sizeof(chunk));
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
        int n = sse_token_chunk(r, esc, chunk, sizeof(chunk));
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

/* ================= W2: 扩展端点（与 CPU 版 src/http.c handle_extended 同形状）=================
 *
 * 同一个二进制、同一批客户端能直连任一后端。后端能力差异如实反映：
 *   - 生成类端点走本引擎的 submit2/wait_done（段租约 + 批量 decode）；
 *   - 权重 tensor 同步 / hicache / audio / reward / expert 分布等本引擎没有后端的，明确 501；
 *   - 运行时 LoRA：设备侧权重在 from_llama() 时已合并，无法运行时卸载 → 501。
 */
typedef struct {
  char *p;
  size_t len, cap;
} ob;

static void ob_init(ob *b) {
  b->cap = 1024;
  b->len = 0;
  b->p = (char *)malloc(b->cap);
  if (b->p) b->p[0] = 0;
}
static void ob_put(ob *b, const char *s) {
  if (!b->p || !s) return;
  size_t n = strlen(s);
  if (b->len + n + 1 > b->cap) {
    size_t c = b->cap;
    while (b->len + n + 1 > c) c *= 2;
    char *np = (char *)realloc(b->p, c);
    if (!np) return;
    b->p = np;
    b->cap = c;
  }
  memcpy(b->p + b->len, s, n + 1);
  b->len += n;
}
static void ob_int(ob *b, long v) {
  char t[32];
  snprintf(t, sizeof(t), "%ld", v);
  ob_put(b, t);
}
static void ob_json_str(ob *b, const char *s) {
  char esc[8192];
  json_escape(s ? s : "", esc, sizeof(esc));
  ob_put(b, esc);
}
static void ob_free(ob *b) {
  free(b->p);
  b->p = NULL;
  b->len = b->cap = 0;
}

/* 从 body 里按顺序试几个 key 取文本（上游同一语义有 prompt/text/input 多种叫法）*/
static const char *ext_text(const sgl_json *j, const char *const *keys, int nk) {
  for (int i = 0; i < nk; i++) {
    const sgl_json *v = sgl_json_get(j, keys[i]);
    if (v && v->type == SGL_JSON_STRING) return v->str;
  }
  return NULL;
}

/* 会话登记表（与 CPU 版一致：只登记名字，不持久化 KV）*/
#define EXT_MAX_SESS 32
static char g_sess[EXT_MAX_SESS][80];
static int g_nsess;
static int sess_add(const char *id) {
  if (g_nsess >= EXT_MAX_SESS) return 0;
  snprintf(g_sess[g_nsess], sizeof(g_sess[0]), "%s", id && *id ? id : "session");
  return ++g_nsess;
}
static int sess_del(const char *id) {
  for (int i = 0; i < g_nsess; i++)
    if (!strcmp(g_sess[i], id)) {
      memmove(g_sess[i], g_sess[g_nsess - 1], sizeof(g_sess[0]));
      g_nsess--;
      return 1;
    }
  return 0;
}

/* SSE 数据块（chat_shape=1 → OpenAI chat delta；0 → text_completion）*/
static int sb_chunk(int chat_shape, const char *esc, char *chunk, size_t cap);

/* ================= W3: DeepSeek（MLA + MoE）生成路径 =================
 *
 * 设备侧是独立结构（sgl_cuda_ds）：KV 行是 [latent | k_pe]，没有段租让/批量 decode。
 * 所以这里用最简单的正确做法：一次一个请求、单序列阻塞生成。
 * 采样与 CPU 引擎同一套（sgl_sample_ex），所以同一 prompt 的输出文本应与 CPU 一致。
 * 流式：stream_fd != INVALID_SOCKET 时按 token 发 SSE（shape 由 chat 决定）。*/
static int ds_gen(const char *prompt, int max_new, SOCKET stream_fd, int chat_shape,
                  char **out_text, int *npt, int *ngen, char *fin, int fincap) {
  if (!E.ds) return -1;
  int np = 0;
  int *ids = sgl_tokenizer_encode(E.tok, prompt, strlen(prompt), &np);
  if (!ids || np <= 0) {
    sgl_xfree(ids);
    return -1;
  }
  int ctx = A->ds_ctx ? A->ds_ctx(E.ds) : 0;
  if (ctx > 0 && np + max_new > ctx) {
    if (np >= ctx) { sgl_xfree(ids); return -1; }
    if (max_new > ctx - np) max_new = ctx - np;
  }
  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp));
  sp.repetition_penalty = 1.0f;
  /* 默认贪心（与 /generate 的默认一致）；调用方要采样就在外面拼好参数——这里只认贪心 */
  EnterCriticalSection(&E.ds_lock);
  A->ds_reset(E.ds);
  int rc = A->ds_forward(E.ds, ids, np, 0, E.ds_logits);
  sgl_xfree(ids);
  int n_gen = 0;
  char finbuf[16];
  snprintf(finbuf, sizeof(finbuf), "%s", "length");
  int *gen = (int *)sgl_xmalloc(sizeof(int) * (size_t)(max_new > 0 ? max_new : 1));
  uint64_t rng = 0x9E3779B97F4A7C15ULL ^ (uint64_t)GetTickCount64();
  if (rc == 0) {
    for (int i = 0; i < max_new; i++) {
      float lp = 0.0f;
      int t = sgl_sample_ex(E.ds_logits, E.vocab, &sp, &rng, gen, n_gen, &lp);
      if (t == E.eos && !sp.ignore_eos) {
        snprintf(finbuf, sizeof(finbuf), "%s", "stop");
        break;
      }
      if (sgl_sampling_is_stop(&sp, t)) {
        snprintf(finbuf, sizeof(finbuf), "%s", "stop");
        break;
      }
      gen[n_gen++] = t;
      if (stream_fd != INVALID_SOCKET) {
        char *piece = sgl_tokenizer_decode(E.tok, &t, 1);
        char esc[512];
        json_escape(piece ? piece : "", esc, sizeof(esc));
        char chunk[1024];
        int m = sb_chunk(chat_shape, esc, chunk, sizeof(chunk));
        send_all(stream_fd, chunk, m);
        sgl_xfree(piece);
      }
      if (n_gen >= max_new) break;
      if (A->ds_decode(E.ds, t, np + n_gen - 1, E.ds_logits) != 0) {
        rc = -1;
        break;
      }
    }
  }
  LeaveCriticalSection(&E.ds_lock);
  char *text = sgl_tokenizer_decode(E.tok, gen, n_gen);
  sgl_xfree(gen);
  if (npt) *npt = np;
  if (ngen) *ngen = n_gen;
  if (fin) snprintf(fin, (size_t)fincap, "%s", finbuf);
  if (rc != 0) {
    sgl_xfree(text);
    return -2;
  }
  if (out_text) *out_text = text;
  else sgl_xfree(text);
  return 0;
}

/* SSE 数据块（chat_shape=1 → OpenAI chat delta；0 → text_completion）*/
static int sb_chunk(int chat_shape, const char *esc, char *chunk, size_t cap) {
  if (chat_shape)
    return snprintf(chunk, cap,
                    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},"
                    "\"finish_reason\":null}]}\n\n",
                    esc);
  return snprintf(chunk, cap,
                  "data: {\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"%s\","
                  "\"finish_reason\":null}]}\n\n",
                  esc);
}

/* 提交一次生成并等它跑完（不流式）。返回 0 成功，-2 失败，-1 忙。 */
static int ext_gen(const char *prompt, int max_new, int prio, char **out_text, int *npt,
                   int *ngen, char *fin, int fincap) {
  if (E.ds) return ds_gen(prompt, max_new, INVALID_SOCKET, 0, out_text, npt, ngen, fin, fincap);
  if (!E.gpu || E.degraded) return -1;
  sgl_sampling_params sp;
  memset(&sp, 0, sizeof(sp));
  sp.repetition_penalty = 1.0f;
  int id = submit2(prompt, max_new, &sp, NULL, 0, prio);
  if (id < 0) return id;
  char *t = NULL;
  request *r = NULL;
  int rc = wait_done(INVALID_SOCKET, id, &t, &r);
  if (rc != 0) {
    free(t); /* wait_done 用 _strdup（malloc），不能用 sgl_xfree */
    reclaim(id);
    return -2;
  }
  if (out_text) *out_text = t;
  else free(t);
  if (npt) *npt = r ? r->n_prompt : 0;
  if (ngen) *ngen = r ? r->n_gen : 0;
  if (fin) snprintf(fin, (size_t)fincap, "%s", r && r->finish[0] ? r->finish : "stop");
  reclaim(id);
  return 0;
}

/* 把 "input"/"messages"/"system" 归一化成 messages JSON 再交给 build_chatml。
 * 支持：字符串、字符串数组、[{role,content}]（content 可为字符串或 part 数组）。*/
static char *ext_prompt(const sgl_json *j, const char *key, const char *system) {
  const sgl_json *in = sgl_json_get(j, key);
  if (!in) return NULL;
  ob m;
  ob_init(&m);
  ob_put(&m, "{\"messages\":[");
  int first = 1;
#define EXT_MSG(role, text)                                          \
  do {                                                               \
    ob_put(&m, first ? "" : ",");                                   \
    first = 0;                                                       \
    ob_put(&m, "{\"role\":\"");                                   \
    ob_put(&m, role);                                                \
    ob_put(&m, "\",\"content\":\"");                              \
    ob_json_str(&m, text);                                           \
    ob_put(&m, "\"}");                                             \
  } while (0)
  if (system && *system) EXT_MSG("system", system);
  if (in->type == SGL_JSON_STRING) {
    EXT_MSG("user", in->str);
  } else if (in->type == SGL_JSON_ARRAY) {
    for (int i = 0; i < in->count; i++) {
      const sgl_json *it = in->items[i];
      if (it->type == SGL_JSON_STRING) {
        EXT_MSG("user", it->str);
        continue;
      }
      if (it->type != SGL_JSON_OBJECT) continue;
      const char *role = sgl_json_get_str(it, "role", "user");
      const sgl_json *c = sgl_json_get(it, "content");
      ob t;
      ob_init(&t);
      if (c && c->type == SGL_JSON_ARRAY) {
        for (int k = 0; k < c->count; k++) {
          const sgl_json *p = c->items[k];
          const char *tx = sgl_json_get_str(p, "text", NULL);
          if (tx) ob_put(&t, tx);
        }
      } else if (c && c->type == SGL_JSON_STRING) {
        ob_put(&t, c->str);
      }
      EXT_MSG(role, t.p ? t.p : "");
      ob_free(&t);
    }
  } else {
    ob_free(&m);
    return NULL;
  }
#undef EXT_MSG
  ob_put(&m, "]}");
  sgl_json *mj = sgl_json_parse(m.p, m.len);
  ob_free(&m);
  if (!mj) return NULL;
  char *prompt = build_chatml(mj, NULL);
  sgl_json_free(mj);
  return prompt;
}

static const char *stop_reason_of(const char *fin) {
  if (!strcmp(fin, "length")) return "max_tokens";
  if (!strcmp(fin, "tool_calls")) return "tool_use";
  return "end_turn";
}

/* Anthropic /v1/messages 流式：逐 token 发 content_block_delta（真流式，不是凑一块）*/
static int anth_stream(SOCKET fd, int id, const char *model) {
  char b[1024];
  int n = snprintf(b, sizeof(b),
                   "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_%d\","
                   "\"type\":\"message\",\"role\":\"assistant\",\"model\":\"%s\",\"content\":[],"
                   "\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":0,"
                   "\"output_tokens\":0}}}\n\n",
                   id, model);
  send_all(fd, b, n);
  n = snprintf(b, sizeof(b),
               "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,"
               "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n");
  send_all(fd, b, n);
  int npt = 0, ngen = 0;
  char fin[16] = "stop";
  for (;;) {
    EnterCriticalSection(&E.lock);
    request *r = find_req(id);
    if (!r) { LeaveCriticalSection(&E.lock); return -1; }
    if (r->state == R_FAIL) { LeaveCriticalSection(&E.lock); return -2; }
    while (r->n_streamed < r->n_gen) {
      int tk = r->gen[r->n_streamed++];
      char *piece = sgl_tokenizer_decode(E.tok, &tk, 1);
      char esc[512];
      json_escape(piece ? piece : "", esc, sizeof(esc));
      char chunk[1024];
      int m = snprintf(chunk, sizeof(chunk),
                       "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,"
                       "\"delta\":{\"type\":\"text_delta\",\"text\":\"%s\"}}\n\n",
                       esc);
      LeaveCriticalSection(&E.lock);
      send_all(fd, chunk, m);
      EnterCriticalSection(&E.lock);
      r = find_req(id);
      sgl_xfree(piece);
      if (!r) { LeaveCriticalSection(&E.lock); return -1; }
    }
    if (r->state == R_DONE) {
      npt = r->n_prompt;
      ngen = r->n_gen;
      snprintf(fin, sizeof(fin), "%s", r->finish[0] ? r->finish : "stop");
      LeaveCriticalSection(&E.lock);
      break;
    }
    SleepConditionVariableCS(&E.cv_step, &E.lock, 2000);
    LeaveCriticalSection(&E.lock);
  }
  send_all(fd, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
           (int)strlen("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"));
  n = snprintf(b, sizeof(b),
               "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"%s\","
               "\"stop_sequence\":null},\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}}\n\n",
               stop_reason_of(fin), npt, ngen);
  send_all(fd, b, n);
  send_all(fd, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
           (int)strlen("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
  reclaim(id);
  return 0;
}

/* 本引擎没有后端的端点：明确 501 + 原因（不假装成功，也不 404 让调用方瞎猜）*/
typedef struct { const char *path; const char *why; } ext_ni;
static const ext_ni g_ni[] = {
    {"/v1/classify", "no reward/classification head backend on this engine"},
    {"/v1/decisions", "no reward-model backend on this engine"},
    {"/v1/audio/transcriptions", "no audio backend compiled in"},
    {"/v1/audio/speech", "no audio backend compiled in"},
    {"/hicache/storage-backend", "no hierarchical cache backend (single-tier radix only)"},
    {"/hicache/storage-backend/check", "no hierarchical cache backend"},
    {"/hicache/storage-hook", "no hierarchical cache backend"},
    {"/add_external_corpus", "no multimodal external-corpus backend"},
    {"/remove_external_corpus", "no multimodal external-corpus backend"},
    {"/get_external_corpus", "no multimodal external-corpus backend"},
    {"/update_weights_from_tensor", "device weights cannot be replaced in place; restart"},
    {"/update_weights_from_distributed", "no distributed weight transport (no NCCL)"},
    {"/update_weights_from_ipc", "no IPC weight transport"},
    {"/init_weights_update_group", "no distributed weight transport (no NCCL)"},
    {"/destroy_weights_update_group", "no distributed weight transport (no NCCL)"},
    {"/init_weights_send_group_for_remote_instance", "no distributed weight transport"},
    {"/get_weights_by_name", "device weights are opaque after quantization"},
    {"/dump_expert_distribution_record", "no MoE expert-distribution recorder"},
    {"/expert_distribution", "no MoE expert-distribution recorder"},
    {"/freeze_gc", "no GC in C"},
    {"/set_trace_level", "no torch profiler in C"},
    {"/configure_logging", "log level is fixed at build time (SGL_LOG)"},
    {"/set_internal_state", "no internal-state inspector in C"},
    {"/dumper/start", "no torch dump backend"},
    {"/dumper/stop", "no torch dump backend"},
    {"/pd_role_switch", "no prefill/decode disaggregation"},
    {"/vertex_generate", "no Vertex AI adapter"},
    {"/weights_checker", "no weight checker for device weights"},
    {"/get_load", "no per-GPU load accounting (single GPU)"},
    {"/clear_hicache_storage_backend", "no hierarchical cache backend"},
    {"/get_tokenizer_load", "no async tokenizer loading"},
};

/* 递增序号：响应体里的 id（resp_/msg_ 后缀）*/
static long ext_seq(void) {
  static volatile long n;
#ifdef _WIN32
  return InterlockedIncrement(&n);
#else
  return __sync_add_and_fetch(&n, 1);
#endif
}

/* 取任意文本字段（tokenize 系列能接受多种叫法；messages 就拼各段 text）*/
static const char *ext_any_text(const sgl_json *j, char *buf, size_t cap) {
  static const char *const keys[] = {"text", "prompt", "input", "content"};
  for (int i = 0; i < (int)(sizeof(keys) / sizeof(keys[0])); i++) {
    const sgl_json *v = sgl_json_get(j, keys[i]);
    if (v && v->type == SGL_JSON_STRING) return v->str;
  }
  const sgl_json *m = sgl_json_get(j, "messages");
  if (m && m->type == SGL_JSON_ARRAY) {
    ob t;
    ob_init(&t);
    for (int i = 0; i < m->count; i++) {
      const sgl_json *c = sgl_json_get(m->items[i], "content");
      if (c && c->type == SGL_JSON_STRING) ob_put(&t, c->str);
      else if (c && c->type == SGL_JSON_ARRAY)
        for (int k = 0; k < c->count; k++) {
          const char *tx = sgl_json_get_str(c->items[k], "text", NULL);
          if (tx) ob_put(&t, tx);
        }
    }
    if (t.p && t.len) {
      snprintf(buf, cap, "%s", t.p);
      ob_free(&t);
      return buf;
    }
    ob_free(&t);
  }
  return NULL;
}

/* 目录里的 encoder 模型（BERT / RoBERTa）：同一个二进制直接当 embedding 服务用 */
static int dir_is_encoder(const char *dir) {
  char path[1100];
  snprintf(path, sizeof(path), "%s/config.json", dir);
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  static char buf[1 << 16];
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = 0;
  sgl_json *j = sgl_json_parse(buf, n);
  if (!j) return 0;
  const char *mt = sgl_json_get_str(j, "model_type", "");
  int yes = !strcmp(mt, "bert") || !strcmp(mt, "roberta") || !strcmp(mt, "xlm-roberta") ||
            !strcmp(mt, "xlm_roberta");
  sgl_json_free(j);
  return yes;
}

static int route_ext(SOCKET fd, const http_req *req) {
  const char *p = req->path;
  int post = !strcmp(req->method, "POST");

  /* ---------- 分词 ---------- */
  if (post && (!strcmp(p, "/tokenize") || !strcmp(p, "/v1/tokenize") || !strcmp(p, "/encode") ||
               !strcmp(p, "/v1/messages/count_tokens"))) {
    sgl_json *j = sgl_json_parse(req->body, req->body_len);
    if (!j) { respond_error(fd, 400, "invalid JSON"); return 1; }
    char tmp[1 << 15];
    const char *text = ext_any_text(j, tmp, sizeof(tmp));
    if (!text) { sgl_json_free(j); respond_error(fd, 400, "text/input required"); return 1; }
    int n = 0, *ids = NULL;
    if (E.tok) {
      ids = sgl_tokenizer_encode(E.tok, text, strlen(text), &n);
    } else if (E.bert) {
      ids = (int *)sgl_xmalloc(sizeof(int) * (strlen(text) + 8));
      n = sgl_wordpiece_encode(E.model_dir, text, ids, (int)strlen(text) + 8);
      if (n < 0) n = 0;
    }
    if (!strcmp(p, "/v1/messages/count_tokens")) {
      char out[96];
      snprintf(out, sizeof(out), "{\"input_tokens\":%d}", n);
      respond(fd, 200, "application/json", out);
    } else if (!strcmp(p, "/encode")) {
      /* 与 CPU 版同形状：input_ids + length */
      ob b;
      ob_init(&b);
      ob_put(&b, "{\"input_ids\":[");
      for (int i = 0; i < n; i++) {
        char t2[16];
        snprintf(t2, sizeof(t2), "%s%d", i ? "," : "", ids ? ids[i] : 0);
        ob_put(&b, t2);
      }
      char tail[48];
      snprintf(tail, sizeof(tail), "],\"length\":%d}", n);
      ob_put(&b, tail);
      respond(fd, 200, "application/json", b.p);
      ob_free(&b);
    } else {
      ob b;
      ob_init(&b);
      ob_put(&b, "{\"tokens\":[");
      for (int i = 0; i < n; i++) {
        char t2[16];
        snprintf(t2, sizeof(t2), "%s%d", i ? "," : "", ids ? ids[i] : 0);
        ob_put(&b, t2);
      }
      ob_put(&b, "],\"count\":");
      ob_int(&b, n);
      ob_put(&b, "}");
      respond(fd, 200, "application/json", b.p);
      ob_free(&b);
    }
    sgl_xfree(ids);
    sgl_json_free(j);
    return 1;
  }
  if (post && (!strcmp(p, "/detokenize") || !strcmp(p, "/v1/detokenize"))) {
    sgl_json *j = sgl_json_parse(req->body, req->body_len);
    if (!j) { respond_error(fd, 400, "invalid JSON"); return 1; }
    const sgl_json *tv = sgl_json_get(j, "tokens");
    if (!tv) tv = sgl_json_get(j, "input_ids");
    if (!tv) tv = sgl_json_get(j, "ids");
    if (!E.tok || !tv || tv->type != SGL_JSON_ARRAY) {
      sgl_json_free(j);
      respond_error(fd, 400, "\"tokens\" array required (and a tokenizer)");
      return 1;
    }
    int n = tv->count;
    int *ids = (int *)sgl_xmalloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
    for (int i = 0; i < n; i++) ids[i] = (int)tv->items[i]->num;
    char *txt = sgl_tokenizer_decode(E.tok, ids, n);
    ob b;
    ob_init(&b);
    ob_put(&b, "{\"text\":\"");
    ob_json_str(&b, txt ? txt : "");
    ob_put(&b, "\"}");
    respond(fd, 200, "application/json", b.p);
    ob_free(&b);
    sgl_xfree(txt);
    sgl_xfree(ids);
    sgl_json_free(j);
    return 1;
  }

  /* ---------- 运维 / 生命周期 ---------- */
  if (post && (!strcmp(p, "/pause_generation") || !strcmp(p, "/continue_generation"))) {
    int pause = !strcmp(p, "/pause_generation");
    E.paused = pause;
    if (!pause) WakeAllConditionVariable(&E.cv_work);
    respond(fd, 200, "application/json",
            pause ? "{\"success\":true,\"paused\":true}" : "{\"success\":true,\"paused\":false}");
    return 1;
  }
  if (post && !strcmp(p, "/slow_down")) {
    sgl_json *j = sgl_json_parse(req->body, req->body_len);
    int ms = j ? sgl_json_get_int(j, "delay_ms", sgl_json_get_int(j, "ms", 0)) : 0;
    if (j) sgl_json_free(j);
    if (ms < 0) ms = 0;
    if (ms > 10000) ms = 10000;
    E.step_delay_ms = ms;
    char out[96];
    snprintf(out, sizeof(out), "{\"success\":true,\"delay_ms\":%d}", ms);
    respond(fd, 200, "application/json", out);
    return 1;
  }
  if (post && (!strcmp(p, "/start_profile") || !strcmp(p, "/stop_profile"))) {
    E.profiling = !strcmp(p, "/start_profile");
    char out[96];
    snprintf(out, sizeof(out), "{\"success\":true,\"profiling\":%s}",
             E.profiling ? "true" : "false");
    respond(fd, 200, "application/json", out);
    return 1;
  }
  if (post && !strcmp(p, "/release_memory_occupation")) {
    EnterCriticalSection(&E.lock);
    pfx_flush_all();
    LeaveCriticalSection(&E.lock);
    respond(fd, 200, "application/json",
            "{\"success\":true,\"released\":[\"prefix_cache\"]}");
    return 1;
  }
  if (post && !strcmp(p, "/resume_memory_occupation")) {
    respond(fd, 200, "application/json", "{\"success\":true}");
    return 1;
  }
  if (post && !strcmp(p, "/open_session")) {
    sgl_json *j = sgl_json_parse(req->body, req->body_len);
    /* 与 CPU 版同键名：body.session（兼容 session_id）*/
    const char *sid = j ? sgl_json_get_str(j, "session", sgl_json_get_str(j, "session_id", "session"))
                        : "session";
    char idbuf[80];
    snprintf(idbuf, sizeof(idbuf), "%s", sid && *sid ? sid : "session");
    int ok = sess_add(idbuf);
    if (j) sgl_json_free(j);
    if (!ok) { respond_error(fd, 503, "too many sessions"); return 1; }
    ob b;
    ob_init(&b);
    ob_put(&b, "{\"success\":true,\"session\":true,\"session_id\":\"");
    ob_json_str(&b, idbuf);
    ob_put(&b, "\",\"note\":\"registration only: this engine keeps no cross-request KV\"}");
    respond(fd, 200, "application/json", b.p);
    ob_free(&b);
    return 1;
  }
  if (post && !strcmp(p, "/close_session")) {
    sgl_json *j = sgl_json_parse(req->body, req->body_len);
    const char *sid = j ? sgl_json_get_str(j, "session", sgl_json_get_str(j, "session_id", "")) : "";
    sess_del(sid);
    if (j) sgl_json_free(j);
    respond(fd, 200, "application/json", "{\"success\":true}");
    return 1;
  }
  if (post && (!strcmp(p, "/load_lora_adapter") || !strcmp(p, "/unload_lora_adapter"))) {
    respond_error(fd, 501, "device weights are merged at load time; restart with --lora to change adapters");
    return 1;
  }

  /* ---------- BERT：embedding / rerank / score ---------- */
  if (post && (!strcmp(p, "/v1/embeddings") || !strcmp(p, "/v1/rerank") || !strcmp(p, "/v1/score"))) {
    if (!E.bert) { respond_error(fd, 501, "embedding model not loaded (model dir is not BERT/RoBERTa)"); return 1; }
    sgl_json *j = sgl_json_parse(req->body, req->body_len);
    if (!j) { respond_error(fd, 400, "invalid JSON"); return 1; }
    int dim = sgl_bert_hidden(E.bert);
    float *vec = (float *)sgl_xmalloc(sizeof(float) * (size_t)(dim > 0 ? dim : 1));
    if (!strcmp(p, "/v1/embeddings")) {
      const sgl_json *in = sgl_json_get(j, "input");
      ob b;
      ob_init(&b);
      ob_put(&b, "{\"object\":\"list\",\"data\":[");
      int idx = 0;
      for (int k = 0; k < (in && in->type == SGL_JSON_ARRAY ? in->count : 1); k++) {
        const char *tx = (in && in->type == SGL_JSON_ARRAY)
                             ? (in->items[k]->type == SGL_JSON_STRING ? in->items[k]->str : "")
                             : sgl_json_get_str(j, "input", "");
        if (sgl_bert_encode(E.bert, tx, SGL_BERT_POOL_MEAN, vec, dim) != 0) continue;
        char head[128];
        snprintf(head, sizeof(head), "%s{\"object\":\"embedding\",\"index\":%d,\"embedding\":[",
                 idx ? "," : "", idx);
        ob_put(&b, head);
        for (int d = 0; d < dim; d++) {
          char nb[40];
          snprintf(nb, sizeof(nb), "%s%.6f", d ? "," : "", (double)vec[d]);
          ob_put(&b, nb);
        }
        ob_put(&b, "]}");
        idx++;
      }
      char tail[160];
      snprintf(tail, sizeof(tail), "],\"model\":\"%s\",\"usage\":{\"prompt_tokens\":0,\"total_tokens\":0}}",
               E.model_name);
      ob_put(&b, tail);
      respond(fd, 200, "application/json", b.p);
      ob_free(&b);
    } else if (!strcmp(p, "/v1/rerank")) {
      const char *q = sgl_json_get_str(j, "query", "");
      const sgl_json *docs = sgl_json_get(j, "documents");
      ob b;
      ob_init(&b);
      ob_put(&b, "{\"object\":\"list\",\"results\":[");
      int idx = 0;
      for (int k = 0; docs && docs->type == SGL_JSON_ARRAY && k < docs->count; k++) {
        const char *d = docs->items[k]->type == SGL_JSON_STRING ? docs->items[k]->str : "";
        float sc = sgl_bert_rerank(E.bert, q, d);
        if (!(sc == sc)) continue; /* NaN */
        char e[256];
        snprintf(e, sizeof(e),
                 "%s{\"index\":%d,\"relevance_score\":%.6f,\"document\":{\"text\":\"\"}}",
                 idx ? "," : "", idx, (double)sc);
        ob_put(&b, e);
        idx++;
      }
      ob_put(&b, "]}");
      respond(fd, 200, "application/json", b.p);
      ob_free(&b);
    } else {
      const char *t1 = sgl_json_get_str(j, "text_1", sgl_json_get_str(j, "text1", ""));
      const char *t2 = sgl_json_get_str(j, "text_2", sgl_json_get_str(j, "text2", ""));
      float *v2 = (float *)sgl_xmalloc(sizeof(float) * (size_t)(dim > 0 ? dim : 1));
      double dot = 0;
      if (sgl_bert_encode(E.bert, t1, SGL_BERT_POOL_MEAN, vec, dim) == 0 &&
          sgl_bert_encode(E.bert, t2, SGL_BERT_POOL_MEAN, v2, dim) == 0)
        for (int d = 0; d < dim; d++) dot += (double)vec[d] * (double)v2[d];
      sgl_xfree(v2);
      char out[160];
      snprintf(out, sizeof(out),
               "{\"object\":\"list\",\"data\":[{\"index\":0,\"score\":%.6f}],"
               "\"usage\":{\"prompt_tokens\":0,\"total_tokens\":0}}",
               dot);
      respond(fd, 200, "application/json", out);
    }
    sgl_xfree(vec);
    sgl_json_free(j);
    return 1;
  }

  /* ---------- 解析工具 ---------- */
  if (post && !strcmp(p, "/separate_reasoning")) {
    sgl_json *j = sgl_json_parse(req->body, req->body_len);
    const char *text = j ? sgl_json_get_str(j, "text", "") : "";
    char *reason = NULL;
    const char *content = text;
    int had = sgl_parse_reasoning(text, &reason, &content, NULL);
    ob b;
    ob_init(&b);
    ob_put(&b, "{\"reasoning_content\":\"");
    ob_json_str(&b, had && reason ? reason : "");
    ob_put(&b, "\",\"content\":\"");
    ob_json_str(&b, content ? content : "");
    ob_put(&b, "\"}");
    respond(fd, 200, "application/json", b.p);
    ob_free(&b);
    sgl_xfree(reason);
    if (j) sgl_json_free(j);
    return 1;
  }
  if (post && !strcmp(p, "/parse_function_call")) {
    sgl_json *j = sgl_json_parse(req->body, req->body_len);
    const char *text = j ? sgl_json_get_str(j, "text", "") : "";
    int n = 0;
    char *calls = sgl_parse_tool_calls(text, &n);
    ob b;
    ob_init(&b);
    ob_put(&b, "{\"tool_calls\":");
    ob_put(&b, (n > 0 && calls) ? calls : "[]");
    char t[48];
    snprintf(t, sizeof(t), ",\"count\":%d}", n);
    ob_put(&b, t);
    respond(fd, 200, "application/json", b.p);
    ob_free(&b);
    sgl_xfree(calls);
    if (j) sgl_json_free(j);
    return 1;
  }

  /* ---------- 生成类扩展端点 ---------- */
  int is_resp = !strcmp(p, "/v1/responses");
  int is_msg = !strcmp(p, "/v1/messages");
  int is_oll_show = !strcmp(p, "/api/show");
  int is_oll_gen = !strcmp(p, "/api/generate");
  int is_oll_chat = !strcmp(p, "/api/chat");
  int is_inv = !strcmp(p, "/invocations");
  if (post && (is_resp || is_msg || is_oll_show || is_oll_gen || is_oll_chat || is_inv)) {
    sgl_json *j = sgl_json_parse(req->body, req->body_len);
    if (!j) { respond_error(fd, 400, "invalid JSON"); return 1; }
    if (is_oll_show) {
      ob b;
      ob_init(&b);
      ob_put(&b, "{\"details\":{\"parent_model\":\"\",\"format\":\"sguf\",\"family\":\"sglang-c\","
                "\"families\":[\"sglang-c\"],\"parameter_size\":\"unknown\",\"quantization_level\":\"unknown\"},"
                "\"modelfile\":\"# sglang-c\\nFROM ");
      ob_json_str(&b, E.model_dir);
      ob_put(&b, "\",\"parameters\":\"\",\"template\":\"{{ .Prompt }}\","
                "\"model_info\":{\"general.architecture\":\"sglang-c\",\"general.basename\":\"");
      ob_json_str(&b, E.model_name);
      ob_put(&b, "\"},\"capabilities\":[\"completion\",\"tools\"]}");
      respond(fd, 200, "application/json", b.p);
      ob_free(&b);
      sgl_json_free(j);
      return 1;
    }
    /* prompts */
    char *prompt = NULL;
    if (is_resp) {
      const char *sys = sgl_json_get_str(j, "instructions", NULL);
      prompt = ext_prompt(j, "input", sys);
    } else if (is_msg) {
      const sgl_json *sv = sgl_json_get(j, "system");
      ob sb;
      ob_init(&sb);
      if (sv && sv->type == SGL_JSON_STRING) ob_put(&sb, sv->str);
      else if (sv && sv->type == SGL_JSON_ARRAY)
        for (int k = 0; k < sv->count; k++) {
          const char *tx = sgl_json_get_str(sv->items[k], "text", NULL);
          if (tx) ob_put(&sb, tx);
        }
      prompt = ext_prompt(j, "messages", sb.p ? sb.p : NULL);
      ob_free(&sb);
    } else if (is_oll_chat) {
      prompt = ext_prompt(j, "messages", sgl_json_get_str(j, "system", NULL));
    } else {
      const char *tx = sgl_json_get_str(j, is_inv ? "inputs" : "prompt", NULL);
      if (!tx) tx = sgl_json_get_str(j, "prompt", "");
      size_t ln = strlen(tx) + 1;
      prompt = (char *)malloc(ln);
      if (prompt) memcpy(prompt, tx, ln);
    }
    if (!prompt) { sgl_json_free(j); respond_error(fd, 400, "prompt/input/messages required"); return 1; }
    if (!E.gpu || E.degraded) {
      free(prompt);
      sgl_json_free(j);
      respond_error(fd, 503, E.gpu_err[0] ? E.gpu_err : "generation backend unavailable");
      return 1;
    }
    int max_new = sgl_json_get_int(j, "max_output_tokens", 0);
    if (max_new <= 0) max_new = sgl_json_get_int(j, "max_tokens", 0);
    if (max_new <= 0) max_new = 256;
    if (max_new > MAX_NEW_CAP) max_new = MAX_NEW_CAP;
    int prio = sgl_json_get_int(j, "priority", 0);
    const sgl_json *streamv = sgl_json_get(j, "stream");
    int stream = streamv && streamv->type == SGL_JSON_BOOL && streamv->boolean;
    sgl_json_free(j);

    if (is_msg && stream) {
      /* Anthropic 真流式 */
      sgl_sampling_params sp;
      memset(&sp, 0, sizeof(sp));
      sp.repetition_penalty = 1.0f;
      int id = submit2(prompt, max_new, &sp, NULL, 0, prio);
      free(prompt);
      if (id == -2) {
        respond_error(fd, 400, seg_full_msg());
        return 1;
      }
      if (id < 0) { respond_error(fd, 503, "server busy"); return 1; }
      const char *hdr =
          "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
          "Connection: close\r\n\r\n";
      send_all(fd, hdr, (int)strlen(hdr));
      int rc = anth_stream(fd, id, E.model_name);
      if (rc != 0) reclaim(id);
      return 1;
    }
    char *text = NULL;
    int npt = 0, ngen = 0;
    char fin[16] = "stop";
    int rc = ext_gen(prompt, max_new, prio, &text, &npt, &ngen, fin, (int)sizeof(fin));
    free(prompt);
    if (rc == -2) { respond_error(fd, 500, "generation failed"); return 1; }
    if (rc < 0) { respond_error(fd, 503, "server busy or prompt too long"); return 1; }
    ob b;
    ob_init(&b);
    if (is_resp) {
      char head[256];
      time_t now = time(NULL);
      snprintf(head, sizeof(head),
               "{\"id\":\"resp_%ld\",\"object\":\"response\",\"created_at\":%ld,"
               "\"status\":\"completed\",\"model\":\"%s\",\"output\":[{\"id\":\"msg_%ld\","
               "\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{"
               "\"type\":\"output_text\",\"text\":\"",
               ext_seq(), (long)now, E.model_name, ext_seq());
      ob_put(&b, head);
      ob_json_str(&b, text ? text : "");
      ob_put(&b, "\"}]}],\"output_text\":\"");
      ob_json_str(&b, text ? text : "");
      char tail[256];
      snprintf(tail, sizeof(tail),
               "\",\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d,\"total_tokens\":%d}}", npt,
               ngen, npt + ngen);
      ob_put(&b, tail);
      respond(fd, 200, "application/json", b.p);
    } else if (is_msg) {
      char head[256];
      snprintf(head, sizeof(head),
               "{\"id\":\"msg_%ld\",\"type\":\"message\",\"role\":\"assistant\","
               "\"model\":\"%s\",\"content\":[{\"type\":\"text\",\"text\":\"",
               ext_seq(), E.model_name);
      ob_put(&b, head);
      ob_json_str(&b, text ? text : "");
      char tail[256];
      snprintf(tail, sizeof(tail),
               "\"}],\"stop_reason\":\"%s\",\"stop_sequence\":null,"
               "\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}}",
               stop_reason_of(fin), npt, ngen);
      ob_put(&b, tail);
      respond(fd, 200, "application/json", b.p);
    } else if (is_oll_gen) {
      char head[160];
      snprintf(head, sizeof(head), "{\"model\":\"%s\",\"created_at\":\"", E.model_name);
      ob_put(&b, head);
      ob_put(&b, "1970-01-01T00:00:00Z\",\"response\":\"");
      ob_json_str(&b, text ? text : "");
      char tail[256];
      snprintf(tail, sizeof(tail),
               "\",\"done\":true,\"done_reason\":\"%s\",\"context\":[],"
               "\"prompt_eval_count\":%d,\"eval_count\":%d}",
               fin, npt, ngen);
      ob_put(&b, tail);
      respond(fd, 200, "application/json", b.p);
    } else if (is_oll_chat) {
      char head[160];
      snprintf(head, sizeof(head),
               "{\"model\":\"%s\",\"created_at\":\"1970-01-01T00:00:00Z\",\"message\":{"
               "\"role\":\"assistant\",\"content\":\"",
               E.model_name);
      ob_put(&b, head);
      ob_json_str(&b, text ? text : "");
      char tail[256];
      snprintf(tail, sizeof(tail),
               "\"},\"done\":true,\"done_reason\":\"%s\",\"prompt_eval_count\":%d,\"eval_count\":%d}",
               "stop", npt, ngen);
      ob_put(&b, tail);
      respond(fd, 200, "application/json", b.p);
    } else { /* /invocations（SageMaker）*/
      ob_put(&b, "{\"generated_text\":\"");
      ob_json_str(&b, text ? text : "");
      ob_put(&b, "\"}");
      respond(fd, 200, "application/json", b.p);
    }
    ob_free(&b);
    free(text); /* _strdup（malloc）*/
    return 1;
  }

  /* ---------- GET ---------- */
  if (!post) {
    if (!strcmp(p, "/api/tags")) {
      ob b;
      ob_init(&b);
      ob_put(&b, "{\"models\":[{\"name\":\"");
      ob_json_str(&b, E.model_name);
      ob_put(&b, "\",\"model\":\"");
      ob_json_str(&b, E.model_name);
      ob_put(&b, "\",\"modified_at\":\"1970-01-01T00:00:00Z\",\"size\":0,\"digest\":\"\","
                "\"details\":{\"family\":\"sglang-c\"}}]}");
      respond(fd, 200, "application/json", b.p);
      ob_free(&b);
      return 1;
    }
    if (!strcmp(p, "/model_info")) {
      char path[1100];
      snprintf(path, sizeof(path), "%s/config.json", E.model_dir);
      FILE *f = fopen(path, "rb");
      if (!f) { respond_error(fd, 404, "config.json not found"); return 1; }
      ob b;
      ob_init(&b);
      ob_put(&b, "{\"model_name\":\"");
      ob_json_str(&b, E.model_name);
      ob_put(&b, "\",\"config\":");
      char buf[4096];
      size_t n;
      int wrote = 0;
      while ((n = fread(buf, 1, sizeof(buf) - 1, f)) > 0) {
        buf[n] = 0;
        ob_put(&b, buf);
        wrote = 1;
      }
      fclose(f);
      if (!wrote) ob_put(&b, "{}");
      ob_put(&b, "}");
      respond(fd, 200, "application/json", b.p);
      ob_free(&b);
      return 1;
    }
  }

  /* ---------- 明确 501 的端点 ---------- */
  if (post || !strcmp(p, "/health")) {
    for (int i = 0; i < (int)(sizeof(g_ni) / sizeof(g_ni[0])); i++)
      if (!strcmp(p, g_ni[i].path)) {
        respond_error(fd, 501, g_ni[i].why);
        return 1;
      }
  }
  return 0;
}

/* W3: DeepSeek 设备路径的路由（不依赖段租让/批量调度的那部分端点）*/
static int route_ds(SOCKET fd, const http_req *req) {
  const char *p = req->path;
  int post = !strcmp(req->method, "POST");
  int chat = !strcmp(p, "/v1/chat/completions");
  int sgl_native = !strcmp(p, "/generate"); /* sglang 原生形状：text + meta_info */
  int comp = !strcmp(p, "/v1/completions") || !strcmp(p, "/completions") || sgl_native;
  if (post && !strcmp(p, "/health")) {
    char body[256];
    snprintf(body, sizeof(body),
             "{\"ok\":true,\"gpu\":true,\"backend\":\"deepseek-mla\",\"ctx\":%d,\"vocab\":%d}",
             A->ds_ctx ? A->ds_ctx(E.ds) : 0, E.vocab);
    respond(fd, 200, "application/json", body);
    return 0;
  }
  if (!strcmp(p, "/health") || !strcmp(p, "/ping") || !strcmp(p, "/health_generate")) {
    char body[256];
    snprintf(body, sizeof(body),
             "{\"ok\":true,\"gpu\":true,\"backend\":\"deepseek-mla\",\"ctx\":%d,\"vocab\":%d}",
             A->ds_ctx ? A->ds_ctx(E.ds) : 0, E.vocab);
    respond(fd, 200, "application/json", body);
    return 0;
  }
  if (!post) { respond_error(fd, 404, "not found"); return 0; }
  if (!chat && !comp) {
    respond_error(fd, 503,
                  "this model runs on the DeepSeek(MLA+MoE) device path: only /generate, "
                  "/v1/completions and /v1/chat/completions are available");
    return 0;
  }
  sgl_json *j = sgl_json_parse(req->body, req->body_len);
  if (!j) { respond_error(fd, 400, "invalid JSON"); return 0; }
  char *prompt = NULL;
  char *img_url = NULL;
  if (chat) {
    prompt = build_chatml(j, &img_url);
  } else {
    const sgl_json *tv = sgl_json_get(j, "text");
    if (!tv) tv = sgl_json_get(j, "prompt");
    if (!tv) tv = sgl_json_get(j, "input");
    const char *tx = (tv && tv->type == SGL_JSON_STRING) ? tv->str : NULL;
    if (tx) {
      size_t ln = strlen(tx) + 1;
      prompt = (char *)malloc(ln);
      if (prompt) memcpy(prompt, tx, ln);
    }
  }
  int max_new = sgl_json_get_int(j, "max_new_tokens", 0);
  if (max_new <= 0) max_new = sgl_json_get_int(j, "max_tokens", 0);
  /* sglang 原生把采样参数包在 sampling_params 里 */
  const sgl_json *spj = sgl_json_get(j, "sampling_params");
  if (spj && max_new <= 0)
    max_new = sgl_json_get_int(spj, "max_new_tokens", sgl_json_get_int(spj, "max_tokens", 0));
  if (max_new <= 0) max_new = 128;
  if (max_new > MAX_NEW_CAP) max_new = MAX_NEW_CAP;
  const sgl_json *streamv = sgl_json_get(j, "stream");
  int stream = streamv && streamv->type == SGL_JSON_BOOL && streamv->boolean;
  sgl_json_free(j);
  if (img_url) sgl_xfree(img_url);
  if (!prompt) { respond_error(fd, 400, "prompt/text required"); return 0; }
  if (stream) {
    const char *hdr =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n";
    send_all(fd, hdr, (int)strlen(hdr));
    int rc = ds_gen(prompt, max_new, fd, chat, NULL, NULL, NULL, NULL, 0);
    free(prompt);
    const char *done = chat ? "data: [DONE]\n\n" : "data: [DONE]\n\n";
    send_all(fd, done, (int)strlen(done));
    return rc == 0 ? 0 : -1;
  }
  char *text = NULL;
  int np = 0, ng = 0;
  char fin[16] = "stop";
  char *saved_prompt = NULL;
  if (sgl_native) {
    size_t ln = strlen(prompt) + 1;
    saved_prompt = (char *)malloc(ln);
    if (saved_prompt) memcpy(saved_prompt, prompt, ln);
  }
  int rc = ds_gen(prompt, max_new, INVALID_SOCKET, chat, &text, &np, &ng, fin, (int)sizeof(fin));
  free(prompt);
  if (rc != 0) { respond_error(fd, 500, "deepseek generation failed"); return 0; }
  ob b;
  ob_init(&b);
  if (chat) {
    char head[256];
    snprintf(head, sizeof(head),
             "{\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion\",\"created\":%ld,"
             "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
             "\"content\":\"",
             ext_seq(), (long)time(NULL), E.model_name);
    ob_put(&b, head);
    ob_json_str(&b, text ? text : "");
    char tail[256];
    snprintf(tail, sizeof(tail),
             "\"},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,"
             "\"completion_tokens\":%d,\"total_tokens\":%d}}",
             fin, np, ng, np + ng);
    ob_put(&b, tail);
  } else if (sgl_native) {
    /* 与 CPU 版同形状：{"text": <prompt 回显>, "completion": <生成>, "meta_info":{...}} */
    ob_put(&b, "{\"text\":\"");
    ob_json_str(&b, saved_prompt ? saved_prompt : "");
    ob_put(&b, "\",\"completion\":\"");
    ob_json_str(&b, text ? text : "");
    char tail[256];
    snprintf(tail, sizeof(tail),
             "\",\"meta_info\":{\"finish_reason\":\"%s\",\"completion_tokens\":%d,"
             "\"prompt_tokens\":%d}}",
             fin, ng, np);
    ob_put(&b, tail);
  } else {
    char head[256];
    snprintf(head, sizeof(head),
             "{\"id\":\"cmpl-%ld\",\"object\":\"text_completion\",\"created\":%ld,"
             "\"model\":\"%s\",\"choices\":[{\"index\":0,\"text\":\"",
             ext_seq(), (long)time(NULL), E.model_name);
    ob_put(&b, head);
    ob_json_str(&b, text ? text : "");
    char tail[256];
    snprintf(tail, sizeof(tail),
             "\",\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,"
             "\"completion_tokens\":%d,\"total_tokens\":%d}}",
             fin, np, ng, np + ng);
    ob_put(&b, tail);
  }
  respond(fd, 200, "application/json", b.p);
  ob_free(&b);
  sgl_xfree(text);
  free(saved_prompt);
  return 0;
}

static int route(SOCKET fd, const http_req *req) {
  if (route_ext(fd, req)) return 0;
  /* W3: DeepSeek（MLA+MoE）设备路径：chat/completions/generate 走单序列阻塞生成，
   * 其余依赖段租让/批量的端点（embeddings/rerank 等）不适用 → 503。 */
  if (E.ds) return route_ds(fd, req);

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
               acc_ctx(E.gpu), E.seg_len, ACTIVE_CONC,
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
                 mn, E.gpu ? acc_ctx(E.gpu) : 0, nact, nwq, pf, dc, fin,
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
  /* W2: 优先级（与 CPU 版同名同义，大的先跑）*/
  int prio = sgl_json_get_int(j, "priority", 0);

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
      int id2 = submit2(prompt, max_new, &sp, feats, n_img, prio);
      free(prompt);
      sgl_json_free(j);
      if (id2 == -2) {
        respond_error(fd, 400, seg_full_msg());
        return 0;
      }
      if (id2 < 0) { respond_error(fd, 503, "server busy"); return 0; }
      return finish_request(fd, id2, is_stream, 1, 0);
    }
    max_new = sgl_json_get_int(j, "max_tokens", 0);
    /* sglang 原生把生成长度包在 sampling_params 里（与 CPU 版同语义）*/
    const sgl_json *spj = sgl_json_get(j, "sampling_params");
    if (max_new <= 0 && spj)
      max_new = sgl_json_get_int(spj, "max_new_tokens", sgl_json_get_int(spj, "max_tokens", 0));
    if (max_new <= 0) max_new = sgl_json_get_int(j, "max_new_tokens", 0);
    if (max_new <= 0) max_new = 256;
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
    max_new = sgl_json_get_int(j, "max_tokens", 0);
    const sgl_json *spj2 = sgl_json_get(j, "sampling_params");
    if (max_new <= 0 && spj2)
      max_new = sgl_json_get_int(spj2, "max_new_tokens", sgl_json_get_int(spj2, "max_tokens", 0));
    if (max_new <= 0) max_new = sgl_json_get_int(j, "max_new_tokens", 0);
    if (max_new <= 0) max_new = 64;
    parse_sampling(j, &sp);
    want_lp = json_true(j, "logprobs") || sgl_json_get_int(j, "logprobs", 0) > 0;
    const sgl_json *sv = sgl_json_get(j, "stream");
    is_stream = sv && sv->type == SGL_JSON_BOOL && sv->boolean;
  }
  sgl_json_free(j);
  if (max_new <= 0 || max_new > MAX_NEW_CAP) max_new = 256;

  int id = submit2(prompt, max_new, &sp, NULL, 0, prio);
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
    /* mark streaming then wait_done emits SSE chunks（find_req 也覆盖排队中的请求）*/
    EnterCriticalSection(&E.lock);
    request *sr = find_req(id);
    if (sr) {
      sr->streaming = 1;
      sr->stream_text = openai && !chat;
    }
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
      int n = (r && r->stream_text)
                  ? snprintf(chunk, sizeof(chunk),
                             "data: {\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"\","
                             "\"finish_reason\":\"%s\"}]}\n\ndata: [DONE]\n\n",
                             fin)
                  : snprintf(chunk, sizeof(chunk),
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

/* 引擎入口（W1：后端无关）。后端薄壳（apps/cuda_http.c 的 sgl_cuda_http_serve、
 * 以后的 apps/mlx_http.c）与 sglc.exe 内嵌调用共用，api = 选中的后端函数表。
 * host 语义同 http.c：NULL / "127.0.0.1" / "localhost" 只听本机；
 * "0.0.0.0" / "any" / "*" 听全部网卡；其他按 IPv4 字面量解析（失败回退 loopback）。 */
int sgl_accel_http_serve(const sgl_accel_api *api, const char *dir, int port, const char *host) {
  {
    char aerr[256];
    if (sgl_accel_api_check(api, aerr, (int)sizeof(aerr)) != 0) {
      fprintf(stderr, "[engine] accel backend unusable: %s\n", aerr);
      return 2;
    }
    A = api;
    printf("[engine] accel backend: %s\n", A->name);
    fflush(stdout);
  }

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
  snprintf(E.model_dir, sizeof(E.model_dir), "%s", dir);
  InitializeCriticalSection(&E.ds_lock);

  /* W3: DeepSeek（MLA + MoE）——设备侧是独立结构，不走 llama 初始化那条路 */
  if (api->ds_load && sgl_deepseek_detect(dir) == 1) {
    E.ds_cpu = sgl_deepseek_load(dir);
    if (!E.ds_cpu) {
      fprintf(stderr, "[engine] deepseek load failed: %s\n", dir);
      return 2;
    }
    E.ds = api->ds_load(E.ds_cpu);
    if (!E.ds) {
      fprintf(stderr, "[engine] deepseek upload to device failed\n");
      sgl_deepseek_free((sgl_deepseek *)E.ds_cpu);
      return 2;
    }
    E.vocab = api->ds_vocab ? api->ds_vocab(E.ds) : sgl_deepseek_vocab((sgl_deepseek *)E.ds_cpu);
    E.ds_logits = (float *)sgl_xmalloc(sizeof(float) * (size_t)(E.vocab > 0 ? E.vocab : 1));
    E.tok = sgl_tokenizer_load_dir(dir);
    if (!E.tok) {
      fprintf(stderr, "[engine] deepseek tokenizer missing in %s\n", dir);
      return 2;
    }
    E.eos = sgl_tokenizer_eos_id(E.tok);
    snprintf(E.model_name, sizeof(E.model_name), "%s", dir);
    { size_t bl = strlen(E.model_name); while (bl > 0 && (E.model_name[bl-1]=='/'||E.model_name[bl-1]=='\\')) E.model_name[--bl]=0;
      for (size_t k = 0; k < bl; k++) if (E.model_name[k]=='/'||E.model_name[k]=='\\') { memmove(E.model_name, E.model_name+k+1, bl-k); E.model_name[bl-k-1]=0; break; } }
    printf("[engine] deepseek device path: %s (vocab=%d ctx=%d) — 单序列生成\n", dir, E.vocab,
           api->ds_ctx ? api->ds_ctx(E.ds) : sgl_deepseek_ctx((sgl_deepseek *)E.ds_cpu));
    fflush(stdout);
    goto serve_loop;
  }

  /* W2: 模型目录是 BERT/RoBERTa 时，同一个二进制直接当 embedding 服务用
   * （encoder 无 KV、不需设备）；生成类端点全 503，不假装。*/
  if (dir_is_encoder(dir)) {
    E.bert = sgl_bert_load(dir);
    if (E.bert) {
      E.vocab = sgl_bert_hidden(E.bert);
      E.eos = -1;
      E.tok = sgl_tokenizer_load_dir(dir); /* WordPiece 目录没有 byte-level 词表 → NULL */
      snprintf(E.model_name, sizeof(E.model_name), "%s", dir);
      printf("[engine] encoder mode: %s (hidden=%d) — /v1/embeddings /v1/rerank /v1/score\n", dir,
             E.vocab);
      fflush(stdout);
      goto serve_loop;
    }
    fprintf(stderr, "[engine] bert load failed, falling back to causal LM path\n");
  }

  double t0 = (double)clock() / CLOCKS_PER_SEC;
  int cfg_only = 0;
  E.cpu = sgl_llama_load(dir);
  if (!E.cpu) {
    /* CPU 侧读不了权重。典型情况：mlx-community 的 4-bit 检查点把稠密矩阵存成
     * `<base>.weight`(U32 打包) + `.scales`/`.biases`(F16)，sgl_st_get 只认浮点，
     * 于是 wte 拿不到、sgl_llama_load 返回 NULL。
     * 但设备后端可能自己能读（MLX 走 sgl_st_get_raw + quantized_matmul），
     * 所以退一步：只拿 config 与 m->st，剩下的交给 from_llama。
     * 注意：cfg_only 模型只有几何字段，没有权重 —— 只有“能自己从 m->st 读权重”
     * 的后端（目前只有 MLX）能接住；CUDA 的 from_llama 会读 m->layers，必需的是
     * 完整加载的模型。 */
    E.cpu = sgl_llama_load_full(dir, /*config_only=*/1);
    if (!E.cpu) return 2;
    cfg_only = 1;
    SGL_INFO("engine: CPU 侧读不了权重（量化检查点？），已退到 config-only");
  }
  if (cfg_only && strcmp(A->name, "mlx") != 0) {
    fprintf(stderr,
            "[engine] 这个检查点的权重 CPU 侧读不了（典型：mlx-community 的 4-bit），"
            "而后端 '%s' 不支持自读权重。请用 --device mlx。\n",
            A->name);
    return 2;
  }
  /* G5: 上传前预检。以前显存不够 = 进程直接死（cudaMalloc 失败 → return 2），
   * 监控/调用方只能看到连接被拒。现在先算账（权重 + KV(ctx) + 工作区 + 上下文
   * 开销），不够就不上传，服务照常起，/health 报 degraded，generate 返 503 + 原因。
   * Windows 上还有第二种更阴的情况：WDDM 把超出部分溢到系统内存，不报错但慢 3-20x
   * （实测 8K prefill 2.8s → 10.4s），所以预检留 192MB 余量。 */
  {
    const char *ce = getenv("SGL_CUDA_CTX");
    /* M2：量化 KV 下 17 KiB/token，16384 这个旧上限（为 FP32 KV 设的）没意义了 */
    const char *kqe0 = getenv("SGL_KVQ");
    int kvq_on0 = (kqe0 && atoi(kqe0)) ? 1 : 0;
    int cap = ce ? atoi(ce) : (kvq_on0 ? (1 << 23) : 16384);
    if (cap <= 0) cap = 1 << 23;
    int want = E.cpu->n_ctx < cap ? E.cpu->n_ctx : cap;
    const char *ec = getenv("SGL_HTTP_CONC");
    int conc = ACTIVE_CONC;
    if (ec) { int c = atoi(ec); if (c >= 1 && c <= MAX_CONC) conc = c; }
    int rows = want / conc;
    /* t11：工作区不再按「整段」预留——prefill 已分块（SGL_PREFILL_CHUNK），
     * 峰值行数就是块大小；用 want/conc 会估出 26GB 工作区把合法的
     * 大上下文配置错判成显存不足（ctx=65536 时 need 61GB）。*/
    {
      const char *pc = getenv("SGL_PREFILL_CHUNK");
      int pch = (pc && atoi(pc) > 0) ? atoi(pc) : 1024;
      if (pch > 8192) pch = 8192;
      if (pch < 256) pch = 256;
      if (rows > pch) rows = pch;
    }
    long need = acc_estimate_mem_mb(E.cpu, want, rows);
    long avail = acc_avail_mem_mb();
    printf("[engine] vram precheck: need ~%ldMB, free %ldMB (ctx=%d rows=%d)\n", need, avail, want, rows);
    fflush(stdout);
    if (need > 0 && avail > 0 && avail + 3072 < need) {
      /* t11：估算口径偏保守（实测 ctx=4096 时估 ~32.4GB、实际 29.6GB）。
       * Linux 上 cudaMalloc 失败是干净的（不像 Windows WDDM 会静默溢到
       * 系统内存），所以只在明显不够（差 3GB 以上）时提前降级，其余交给
       * 真实分配判定——否则 48K/64K 这种合法大上下文配置会被误杀。*/
      snprintf(E.gpu_err, sizeof(E.gpu_err),
               "insufficient VRAM: model needs ~%ldMB, %ldMB free (lower SGL_CUDA_CTX or use SGL_CUDA_FP16=1/SGL_CUDA_W8=1)",
               need, avail);
      E.degraded = 1;
      fprintf(stderr, "[engine] DEGRADED: %s\n", E.gpu_err);
    }
  }
  if (!E.degraded) {
    E.gpu = acc_from_llama(E.cpu);
    if (!E.gpu) {
      snprintf(E.gpu_err, sizeof(E.gpu_err),
               "GPU upload failed (out of VRAM); free %ldMB, try SGL_CUDA_CTX=8192 or SGL_CUDA_W8=1",
               acc_avail_mem_mb());
      E.degraded = 1;
      fprintf(stderr, "[engine] DEGRADED: %s\n", E.gpu_err);
    }
  }
  if (E.degraded) E.seg_len = 0;
  /* M3: MTP 草稿头。默认关（SGL_MTP=1 开），因为要先测出接受率再决定。
   * 权重在同一个检查点的 mtp.safetensors 里，没有就静默降级。*/
  if (E.gpu && !E.degraded) {
    const char *me = getenv("SGL_MTP");
    if (me && atoi(me) > 0) {
      int mctx = acc_ctx(E.gpu);
      if (acc_mtp_load(E.gpu, mctx) != 0)
        SGL_INFO("cuda http: MTP requested but not available, continuing without");
    }
  }
  char tpath[1024];
  snprintf(tpath, sizeof(tpath), "%s/tokenizer.json", dir);
  /* t11：必须走 load_dir——它会在 tokenizer_config.json 里解析 eos_token（字符串/
   * 对象形式，如 Qwen 的 "<|im_end|>"）并覆盖。直接 load 则 eos 恒为 -1，
   * 导致生成永远不停止（“Hello!!!!!!!!” 循环）。*/
  E.tok = sgl_tokenizer_load_dir(dir);
  if (!E.tok) E.tok = sgl_tokenizer_load(tpath); /* 退路：无 tokenizer_config */
  if (!E.tok) return 2;
  E.vocab = E.cpu->vocab;
  E.eos = sgl_tokenizer_eos_id(E.tok);
  /* t11：线性注意力（hybrid）模型不能做前缀复用——KV 能按位置拷贝，但递归状态
   * （gated-deltanet 的 state + conv 历史）没有中间快照，命中前缀后跳过那段 prefill
   * 就等于用错的 state 继续。多轮对话/固定 system prompt 时必然失真，所以禁用。*/
  E.pfx_ok = (acc_lin_slots(E.gpu) <= 0);
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
  E.seg_len = E.gpu ? acc_ctx(E.gpu) / ACTIVE_CONC : 0;
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
    /* t11：prewarm 只为「预先把工作区撑到位、避免 decode graph 反复失效」，
     * 它的行数应跟 prefill 分块大小一致，而不是 seg_len。
     * 之前传 seg_len（大上下文时 18432）→ 要 12GB 工作区直接 OOM，
     * 把显存吃干后长 prompt 的 prefill 全部失败（503）。*/
    const char *pc = getenv("SGL_PREFILL_CHUNK");
    int pch = (pc && atoi(pc) > 0) ? atoi(pc) : 1024;
    if (pch > 8192) pch = 8192;
    int warm_rows = E.seg_len < pch ? E.seg_len : pch;
    int warmed = acc_prewarm(E.gpu, warm_rows, ACTIVE_CONC);
    printf("[engine] prewarm: ws rows=%d (seg=%d chunk=%d), decode graphs %d/%d\n",
           warm_rows, E.seg_len, pch, warmed, ACTIVE_CONC);
  }
  printf("[engine] loaded in %.1fs; ctx=%d seg=%d conc=%d eos=%d%s\n",
         (double)clock() / CLOCKS_PER_SEC - t0, E.gpu ? acc_ctx(E.gpu) : 0,
         E.seg_len, ACTIVE_CONC, E.eos, E.degraded ? " [DEGRADED]" : "");
  fflush(stdout);
  /* M3 自验证：SGL_MTP_TEST=1 时跑一遍 teacher-forced 接受率然后退出。
   * 这是判断 MTP 实现对不对的唯一判据（对：0.6–0.85，错：0–0.05）。*/
  {
    const char *mt = getenv("SGL_MTP_TEST");
    if (mt && atoi(mt) > 0 && E.gpu && acc_mtp_ready(E.gpu)) {
      const char *pr = getenv("SGL_MTP_PROMPT");
      if (!pr) pr = "The capital of France is";
      int ng = 24;
      const char *ge = getenv("SGL_MTP_NGEN");
      if (ge && atoi(ge) > 0) ng = atoi(ge);
      int ids[512];
      int np = 0;
      if (getenv("SGL_MTP_DBG")) fprintf(stderr, "[mtp] stage: tokenize\n");
      /* SGL_MTP_IDS=1,2,3 直接给 token id，绕开分词器（排错用）*/
      const char *idsenv = getenv("SGL_MTP_IDS");
      if (idsenv && *idsenv) {
        const char *q = idsenv;
        while (*q && np < 512) {
          ids[np++] = (int)strtol(q, (char **)&q, 10);
          if (*q == ',') q++;
        }
      } else {
        int *enc = sgl_tokenizer_encode(E.tok, pr, strlen(pr), &np);
        if (enc && np > 0) {
          if (np > 512) np = 512;
          memcpy(ids, enc, sizeof(int) * (size_t)np);
        }
        sgl_xfree(enc); /* 必须 sgl_xfree：sgl_xmalloc 在 ptr[-8] 存原始地址，
                         * 用 free() 会直接堆损坏（启动时 “double free or
                         * corruption (out)” 就是这个），见 TODO_PROMPTS.md:66 */
      }
      if (getenv("SGL_MTP_DBG")) fprintf(stderr, "[mtp] stage: tokenized np=%d\n", np);
      if (np <= 0) {
        fprintf(stderr, "[mtp] tokenize failed\n");
      } else {
        double acc = 0;
        int nok = 0;
        if (acc_mtp_accept_test(E.gpu, ids, np, ng, &acc, &nok) == 0)
          printf("[mtp] RESULT prompt=%s  n=%d  accept=%.3f (%d)\n", pr, np, acc, nok);
        fflush(stdout);
      }
      exit(0);
    }
  }

#ifdef _WIN32
  HANDLE w = CreateThread(NULL, 0, worker, NULL, 0, NULL);
#else
  pthread_t w;
  pthread_create(&w, NULL, worker, NULL);
#endif

serve_loop:
  {
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
  } /* serve_loop 块（encoder 模式跳过了设备初始化）*/
  E.stop = 1;
  WakeAllConditionVariable(&E.cv_work);
#ifdef _WIN32
  WaitForSingleObject(w, INFINITE);
#else
  pthread_join(w, NULL);
#endif
  return 0;
}

/* main 在后端薄壳里（apps/cuda_http.c / apps/mlx_http.c）：本文件后端无关，
 * 不知道自己该配哪张函数表。 */
