/* sgl/port.h — 跨平台垫片（MSVC / MinGW / POSIX）
 * Windows 原生 API 保持不变；POSIX 侧把 MSVC/C11-K 风格调用映射到 POSIX 等价物。 */
#ifndef SGL_PORT_H
#define SGL_PORT_H

#ifdef _WIN32
  #include <stdlib.h>  /* _putenv_s */
  #include <stdio.h>   /* fopen_s */
  #include <string.h>  /* _strnicmp */
#else
  #include <stdlib.h>  /* setenv / putenv */
  #include <stdio.h>   /* fopen */
  #include <string.h>
  #include <strings.h> /* strncasecmp */

  #define _putenv(k)           putenv((char *)(k))
  #define _putenv_s(k, v)      setenv((k), (v), 1)
  /* MSVC/C11-K 的 fopen_s：返回 0 = 成功、非 0 = 失败（errno 风格），并总是写 *pf。
   * 早先版本写成 `(*(pf) = fopen(...))`，返回的是 FILE*：成功时非 0，于是所有
   * `if (fopen_s(&f,...) != 0 || !f)` 在 POSIX 上都会把打开成功当失败（测试静默 SKIP）。*/
  static inline int sgl_fopen_s(FILE **pf, const char *path, const char *mode) {
    if (!pf) return 22; /* EINVAL */
    *pf = fopen(path, mode);
    return *pf ? 0 : 2; /* ENOENT 级：非 0 即失败，与 MSVC 的判法一致 */
  }
  #define fopen_s(pf, path, mode) sgl_fopen_s((pf), (path), (mode))
  #define _strnicmp(s1, s2, n) strncasecmp((s1), (s2), (n))

  /* MSVC 的 strcpy_s（Annex K 边界检查拷贝）：glibc 不提供。返回 0 = 成功，
   * 非 0 = 失败（目标置空串），与 MSVC 的行为关心的部分一致。*/
  static inline int sgl_strcpy_s(char *dst, size_t cap, const char *src) {
    size_t n = src ? strlen(src) : 0;
    if (!dst || cap == 0) return 22; /* EINVAL */
    if (n + 1 > cap) { dst[0] = 0; return 34; } /* ERANGE */
    memcpy(dst, src, n + 1);
    return 0;
  }
  #define strcpy_s(dst, cap, src) sgl_strcpy_s((dst), (size_t)(cap), (src))
#endif

#endif /* SGL_PORT_H */
