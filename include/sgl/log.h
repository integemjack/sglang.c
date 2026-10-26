/* sgl/log.h — 日志（对应 Python logging 的最小子集） */
#ifndef SGL_LOG_H
#define SGL_LOG_H

#include <stdarg.h>

typedef enum {
  SGL_LOG_DEBUG = 0,
  SGL_LOG_INFO = 1,
  SGL_LOG_WARN = 2,
  SGL_LOG_ERROR = 3,
} sgl_log_level;

void sgl_log_set_level(sgl_log_level level);
sgl_log_level sgl_log_get_level(void);

/* 线程安全：内部用锁保证多行不交错（OpenMP 多线程 matmul 时打点用） */
void sgl_log(sgl_log_level lvl, const char *file, int line, const char *fmt, ...)
#ifndef _MSC_VER
    __attribute__((format(printf, 4, 5)))
#endif
;

#define SGL_DEBUG(...) sgl_log(SGL_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define SGL_INFO(...) sgl_log(SGL_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define SGL_WARN(...) sgl_log(SGL_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define SGL_ERROR(...) sgl_log(SGL_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

#endif
