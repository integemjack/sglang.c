/* test_st_index — P0-1 分片 safetensors：自造两个 shard + index.json，验证
 *   sgl_st_open_dir 按 weight_map 跨分片取张量（F32 零拷贝 / F16 转 fp32），
 *   单文件 sgl_st_open 语义不变，缺失名字返回 null。*/
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "sgl/alloc.h"
#include "sgl/port.h"
#include "sgl/safetensors.h"

static int fails;

static void check(int cond, const char *what) {
  if (!cond) {
    printf("FAIL: %s\n", what);
    fails++;
  } else {
    printf("ok  : %s\n", what);
  }
}

static void write_file_(const char *path, const void *head, size_t hlen, const void *data,
                        size_t dlen) {
  FILE *f = NULL;
  if (fopen_s(&f, path, "wb") != 0 || !f) {
    printf("FAIL: cannot create %s\n", path);
    fails++;
    return;
  }
  if (head && hlen) fwrite(head, 1, hlen, f);
  if (data && dlen) fwrite(data, 1, dlen, f);
  fclose(f);
}

/* safetensors = <u64 header_len><header><data> */
static void write_shard(const char *path, const char *header_json, const void *data, size_t len) {
  unsigned long long h = (unsigned long long)strlen(header_json);
  FILE *f = NULL;
  if (fopen_s(&f, path, "wb") != 0 || !f) {
    printf("FAIL: cannot create %s\n", path);
    fails++;
    return;
  }
  fwrite(&h, 8, 1, f);
  fwrite(header_json, 1, (size_t)h, f);
  if (data && len) fwrite(data, 1, len, f);
  fclose(f);
}

int main(void) {
  const char *dir = "sgl_st_index_tmp";
#ifdef _WIN32
  _mkdir(dir);
#else
  mkdir(dir, 0755);
#endif

  /* shard 1：[2,2] F32 值 1..4 */
  float a[4] = {1.f, 2.f, 3.f, 4.f};
  write_shard("sgl_st_index_tmp/model-00001-of-00002.safetensors",
              "{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}", a, sizeof(a));

  /* shard 2：[3] F32 值 5,6,7 + [2] F16 值 1.0 / -2.0 */
  struct {
    float b[3];
    unsigned short c[2];
  } s2 = {{5.f, 6.f, 7.f}, {0x3C00u, 0xC000u}};
  write_shard("sgl_st_index_tmp/model-00002-of-00002.safetensors",
              "{\"b\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,12]},"
              "\"c\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[12,16]}}",
              &s2, sizeof(s2));

  const char *index =
      "{\"metadata\":{\"total_size\":28},\"weight_map\":{"
      "\"a\":\"model-00001-of-00002.safetensors\","
      "\"b\":\"model-00002-of-00002.safetensors\","
      "\"c\":\"model-00002-of-00002.safetensors\"}}";
  write_file_("sgl_st_index_tmp/model.safetensors.index.json", index, strlen(index), NULL, 0);

  sgl_st_ctx *c = sgl_st_open_dir(dir);
  check(c != NULL, "open_dir with index");
  if (!c) {
    printf("test_st_index: %d failure(s)\n", fails);
    return 1;
  }
  check(sgl_st_shard_count(c) == 2, "shard count == 2");

  sgl_tensor ta = sgl_st_get(c, "a", 2, (int64_t[]){2, 2});
  check(!sgl_tensor_is_null(ta), "tensor a found in shard 1");
  check(ta.data && ta.data[0] == 1.f && ta.data[3] == 4.f, "tensor a values");

  sgl_tensor tb = sgl_st_get(c, "b", 1, (int64_t[]){3});
  check(!sgl_tensor_is_null(tb), "tensor b found in shard 2");
  check(tb.data && tb.data[0] == 5.f && tb.data[2] == 7.f, "tensor b values");

  sgl_tensor tc = sgl_st_get(c, "c", 1, (int64_t[]){2});
  check(!sgl_tensor_is_null(tc), "tensor c (F16) converted");
  check(tc.data && tc.data[0] == 1.f && tc.data[1] == -2.f, "tensor c fp16 -> fp32 values");

  sgl_tensor miss = sgl_st_get(c, "nope", 1, NULL);
  check(sgl_tensor_is_null(miss), "missing tensor -> null");

  const char **names = NULL;
  int n = sgl_st_names(c, &names);
  check(n == 3, "weight_map lists 3 tensors");

  sgl_st_close(c);

  /* 单文件模式语义不变 */
  sgl_st_ctx *one = sgl_st_open("sgl_st_index_tmp/model-00001-of-00002.safetensors");
  check(one != NULL, "single-file open still works");
  if (one) {
    check(sgl_st_shard_count(one) == 1, "single-file shard count == 1");
    sgl_tensor t = sgl_st_get(one, "a", 2, NULL);
    check(!sgl_tensor_is_null(t) && t.data[1] == 2.f, "single-file tensor a");
    sgl_st_close(one);
  }

  printf(fails ? "test_st_index: %d failure(s)\n" : "test_st_index: all OK\n", fails);
  return fails ? 1 : 0;
}