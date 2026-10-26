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
  #define fopen_s(pf, path, mode) (*(pf) = fopen((path), (mode)))
  #define _strnicmp(s1, s2, n) strncasecmp((s1), (s2), (n))
#endif

#endif /* SGL_PORT_H */
