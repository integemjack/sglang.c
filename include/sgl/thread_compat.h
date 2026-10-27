/* sgl/thread_compat.h — pthread 的最小 Windows/MSVC 兼容层
 *
 * 为什么需要：src/http.c（HTTP 服务端：acceptor / worker / 每连接一线程）是用
 * pthread 写的，以前只有 MinGW（winpthreads）能编，MSVC 构建干脆把 http.c 排除，
 * 于是 `sglc --serve` 在 Windows 上被 #ifdef _MSC_VER 直接拒绝，逼用户另起一个
 * cuda_http.exe。这里把 http.c 用到的那几个 pthread 原语映射到 Win32 等价物
 * （CRITICAL_SECTION / CONDITION_VARIABLE / _beginthreadex），让同一个 sglc.exe
 * 在 MSVC 下也能起服务。
 *
 * 只覆盖 http.c 实际用到的子集：mutex init/lock/unlock/destroy、
 * cond init/wait/signal/broadcast/destroy、thread create/join/detach。
 * 语义与 pthread 一致（条件变量等待会自动释放互斥量，唤醒后重新获取）。
 */
#ifndef SGL_THREAD_COMPAT_H
#define SGL_THREAD_COMPAT_H

#if defined(_WIN32) && defined(_MSC_VER)

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600 /* CONDITION_VARIABLE 需要 Vista+ */
#endif
#include <windows.h>
#include <process.h>
#include <stdlib.h>

typedef CRITICAL_SECTION pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef HANDLE pthread_t;

#define pthread_mutex_init(m, a) InitializeCriticalSection(m)
#define pthread_mutex_lock(m) EnterCriticalSection(m)
#define pthread_mutex_unlock(m) LeaveCriticalSection(m)
#define pthread_mutex_destroy(m) DeleteCriticalSection(m)
#define pthread_mutex_trylock(m) (TryEnterCriticalSection(m) ? 0 : 16 /* EBUSY */)

/* SleepConditionVariableCS 语义等同 pthread_cond_wait：原子地释放临界区并等待，
 * 唤醒后重新进入临界区。SleepConditionVariableCS 也必须以 EnterCriticalSection
 * 进入临界区为前提，与 pthread_cond_wait 要求持有 mutex 一致。 */
#define pthread_cond_init(c, a) InitializeConditionVariable(c)
#define pthread_cond_wait(c, m) SleepConditionVariableCS((c), (m), INFINITE)
#define pthread_cond_signal(c) WakeConditionVariable(c)
#define pthread_cond_broadcast(c) WakeAllConditionVariable(c)
#define pthread_cond_destroy(c) ((void)0)

typedef struct sgl_thr_start {
  void *(*fn)(void *);
  void *arg;
} sgl_thr_start;

static unsigned __stdcall sgl_thr_tramp(void *p) {
  sgl_thr_start s = *(sgl_thr_start *)p;
  free(p);
  s.fn(s.arg);
  return 0;
}

static int sgl_pthread_create(pthread_t *t, void *(*fn)(void *), void *arg) {
  sgl_thr_start *s = (sgl_thr_start *)malloc(sizeof(*s));
  if (!s) return -1;
  s->fn = fn;
  s->arg = arg;
  uintptr_t h = _beginthreadex(NULL, 0, sgl_thr_tramp, s, 0, NULL);
  if (!h) { free(s); return -1; }
  *t = (HANDLE)h;
  return 0;
}

#define pthread_create(t, attr, fn, arg) sgl_pthread_create((t), (fn), (arg))
#define pthread_join(t, ret) do { WaitForSingleObject((t), INFINITE); CloseHandle(t); } while (0)
#define pthread_detach(t) CloseHandle(t)
#define pthread_self() GetCurrentThreadId()

#else
#include <pthread.h>
#endif

#endif /* SGL_THREAD_COMPAT_H */