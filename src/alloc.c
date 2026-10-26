/* sgl alloc 实现：对齐分配 + Windows 只读文件映射 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/alloc.h"
#include "sgl/log.h"

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

void *sgl_xmalloc(size_t size) {
  void *p = _aligned_malloc(size ? size : 1, 64);
  if (!p) {
    SGL_ERROR("out of memory: %zu bytes", size);
    abort();
  }
  return p;
}

void *sgl_xcalloc(size_t count, size_t size) {
  void *p = sgl_xmalloc(count * size);
  memset(p, 0, count * size);
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

int sgl_mmap_open(sgl_mmap *m, const char *path) {
  memset(m, 0, sizeof(*m));
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
}

const void *sgl_mmap_at(const sgl_mmap *m, size_t offset, size_t len) {
  if (offset + len > m->size) {
    SGL_ERROR("mmap range out of file: offset=%zu len=%zu size=%zu", offset, len, m->size);
    return NULL;
  }
  return (const char *)m->data + offset;
}

void sgl_mmap_close(sgl_mmap *m) {
  if (m->data) UnmapViewOfFile(m->data);
  if (m->_handle) CloseHandle((HANDLE)m->_handle);
  memset(m, 0, sizeof(*m));
}
