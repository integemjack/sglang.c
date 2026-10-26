/* sgl alloc 实现：对齐分配 + 只读文件映射（Windows / POSIX 双分支）*/
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/alloc.h"
#include "sgl/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #include <malloc.h>
#else
  #include <unistd.h>
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <sys/mman.h>
#endif

#ifdef _WIN32

void *sgl_xmalloc(size_t size) {
  void *p = _aligned_malloc(size ? size : 1, 64);
  if (!p) {
    SGL_ERROR("out of memory: %zu bytes", size);
    abort();
  }
  return p;
}

void *sgl_xrealloc(void *ptr, size_t size) {
  void *p = _aligned_realloc(ptr, size ? size : 1, 64);
  if (!p) {
    SGL_ERROR("out of memory (realloc): %zu bytes", size);
    abort();
  }
  return p;
}

void sgl_xfree(void *ptr) { _aligned_free(ptr); }

#else /* POSIX: 64 字节手动对齐；realloc 需要旧 size，块头记录 (raw, size) */

void *sgl_xmalloc(size_t size) {
  size = size ? size : 1;
  void *raw = malloc(size + 64 + 16);
  if (!raw) {
    SGL_ERROR("out of memory: %zu bytes", size);
    abort();
  }
  uintptr_t a = ((uintptr_t)raw + 16 + 63) & ~(uintptr_t)63;
  ((void **)a)[-1] = raw;
  ((size_t *)a)[-2] = size;
  return (void *)a;
}

void *sgl_xrealloc(void *ptr, size_t size) {
  size = size ? size : 1;
  if (!ptr) return sgl_xmalloc(size);
  void *raw = ((void **)ptr)[-1];
  size_t old = ((size_t *)ptr)[-2];
  void *n = sgl_xmalloc(size); /* abort on OOM inside */
  memcpy(n, ptr, old < size ? old : size);
  free(raw);
  return n;
}

void sgl_xfree(void *ptr) { if (ptr) free(((void **)ptr)[-1]); }

#endif

void *sgl_xcalloc(size_t count, size_t size) {
  void *p = sgl_xmalloc(count * size);
  memset(p, 0, count * size);
  return p;
}

int sgl_mmap_open(sgl_mmap *m, const char *path) {
  memset(m, 0, sizeof(*m));
#ifdef _WIN32
  HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (file == INVALID_HANDLE_VALUE) return -1;
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(file, &sz)) { CloseHandle(file); return -1; }
  HANDLE map = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
  if (!map) { CloseHandle(file); return -1; }
  void *data = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
  if (!data) { CloseHandle(map); CloseHandle(file); return -1; }
  m->data = data;
  m->size = (size_t)sz.QuadPart;
  m->_handle = file; /* map 句柄可以关掉，view 保持有效直到 UnmapViewOfFile */
  CloseHandle(map);
  return 0;
#else
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;
  struct stat st;
  if (fstat(fd, &st) != 0) { close(fd); return -1; }
  void *data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd); /* mmap 建立后 fd 可关 */
  if (data == MAP_FAILED) return -1;
  m->data = data;
  m->size = (size_t)st.st_size;
  return 0;
#endif
}

const void *sgl_mmap_at(const sgl_mmap *m, size_t offset, size_t len) {
  if (offset + len > m->size) {
    SGL_ERROR("mmap range out of file: offset=%zu len=%zu size=%zu", offset, len, m->size);
    return NULL;
  }
  return (const char *)m->data + offset;
}

void sgl_mmap_close(sgl_mmap *m) {
#ifdef _WIN32
  if (m->data) UnmapViewOfFile(m->data);
  if (m->_handle) CloseHandle((HANDLE)m->_handle);
#else
  if (m->data && m->size) munmap((void *)m->data, m->size);
#endif
  memset(m, 0, sizeof(*m));
}
