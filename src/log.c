/* sgl log 实现（Windows / POSIX 双分支）*/
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/log.h"

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  static CRITICAL_SECTION g_cs;
  static int g_cs_init = 0;
  static void ensure_init(void) {
    if (!g_cs_init) {
      InitializeCriticalSection(&g_cs);
      g_cs_init = 1;
    }
  }
#else
  #include <pthread.h>
  #include <sys/time.h>
  static pthread_mutex_t g_cs = PTHREAD_MUTEX_INITIALIZER;
  static void ensure_init(void) {}
#endif

static sgl_log_level g_level = SGL_LOG_INFO;

void sgl_log_set_level(sgl_log_level level) { g_level = level; }
sgl_log_level sgl_log_get_level(void) { return g_level; }

static const char *level_name(sgl_log_level lvl) {
  switch (lvl) {
  case SGL_LOG_DEBUG: return "DEBUG";
  case SGL_LOG_INFO: return "INFO";
  case SGL_LOG_WARN: return "WARN";
  default: return "ERROR";
  }
}

void sgl_log(sgl_log_level lvl, const char *file, int line, const char *fmt, ...) {
  if (lvl < g_level) return;
  ensure_init();
  const char *base = strrchr(file, '\\');
  if (!base) base = strrchr(file, '/');
  base = base ? base + 1 : file;

  char body[2048];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(body, sizeof(body), fmt, ap);
  va_end(ap);

#ifdef _WIN32
  SYSTEMTIME st;
  GetLocalTime(&st);
  EnterCriticalSection(&g_cs);
  fprintf(stderr, "%02d:%02d:%02d.%03d %s %s:%d] %s\n", st.wHour, st.wMinute,
          st.wSecond, st.wMilliseconds, level_name(lvl), base, line, body);
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  struct tm tmv;
  localtime_r(&tv.tv_sec, &tmv);
  pthread_mutex_lock(&g_cs);
  fprintf(stderr, "%02d:%02d:%02d.%03d %s %s:%d] %s\n", tmv.tm_hour, tmv.tm_min,
          tmv.tm_sec, (int)(tv.tv_usec / 1000), level_name(lvl), base, line, body);
#endif
  fflush(stderr);
#ifdef _WIN32
  LeaveCriticalSection(&g_cs);
#else
  pthread_mutex_unlock(&g_cs);
#endif
}
