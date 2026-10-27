/* test_fp8 — P1：safetensors 里的 FP8（F8_E4M3 / F8_E5M2）载入后解成 f32。
 * 手搓两个 shard：一个 E4M3 张量、一个 E5M2 张量，逐字节写入已知位模式，
 * 和 IEEE 定义的手算值比（含 subnormal / ±0 / 最大值 / NaN）。*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "sgl/port.h"
#include "sgl/safetensors.h"

static int fails;
static void check(int cond, const char *msg) {
  if (!cond) {
    fails++;
    printf("FAIL: %s\n", msg);
  }
}

/* 写一个只有单个 F32/F16/FP8 张量的 safetensors 文件（形状 [n]）*/
static void write_st(const char *path, const char *dtype, const unsigned char *bytes, int n, int esz) {
  char hdr[512];
  int hl = snprintf(hdr, sizeof(hdr),
                    "{\"t\":{\"dtype\":\"%s\",\"shape\":[%d],\"data_offsets\":[0,%d]}}", dtype, n, n * esz);
  unsigned long long hlen = (unsigned long long)hl;
  FILE *f = NULL;
  fopen_s(&f, path, "wb");
  if (!f) {
    printf("FAIL: cannot write %s\n", path);
    fails++;
    return;
  }
  fwrite(&hlen, 8, 1, f);
  fwrite(hdr, 1, (size_t)hl, f);
  fwrite(bytes, 1, (size_t)(n * esz), f);
  fclose(f);
}

static void mkdir_one(const char *p) {
#ifdef _WIN32
  _mkdir(p);
#else
  mkdir(p, 0755);
#endif
}

int main(void) {
  const char *dir = "sgl_fp8_tmp";
  mkdir_one(dir);

  /* E4M3FN 位模式 → 期望值（IEEE e4m3fn）*/
  struct {
    unsigned char b;
    float want;
    const char *name;
  } e4[] = {
      {0x00, 0.0f, "0"},
      {0x80, 0.0f, "-0"},
      {0x38, 1.0f, "1.0 (e=7,m=0)"},
      {0xB8, -1.0f, "-1.0"},
      {0x3C, 1.5f, "1.5"},
      {0x40, 2.0f, "2.0"},
      {0x01, 1.0f / 512.0f, "min subnormal 2^-9"},
      {0x7E, 448.0f, "max 448"},
      {0x08, 1.0f / 64.0f, "min normal 2^-6"},
  };
  int n4 = (int)(sizeof(e4) / sizeof(e4[0]));
  unsigned char b4[16];
  for (int i = 0; i < n4; i++) b4[i] = e4[i].b;
  char p4[256];
  snprintf(p4, sizeof(p4), "%s/e4.safetensors", dir);
  write_st(p4, "F8_E4M3", b4, n4, 1);

  /* E5M2 位模式 */
  struct {
    unsigned char b;
    float want;
    const char *name;
  } e5[] = {
      {0x00, 0.0f, "0"},
      {0x3C, 1.0f, "1.0"},
      {0xBE, -1.5f, "-1.5"},
      {0x01, 1.0f / 65536.0f, "min subnormal 2^-16"},
      {0x7B, 57344.0f, "max 57344"},
      {0x04, 1.0f / 16384.0f, "min normal 2^-14"},
  };
  int n5 = (int)(sizeof(e5) / sizeof(e5[0]));
  unsigned char b5[16];
  for (int i = 0; i < n5; i++) b5[i] = e5[i].b;
  char p5[256];
  snprintf(p5, sizeof(p5), "%s/e5.safetensors", dir);
  write_st(p5, "F8_E5M2", b5, n5, 1);

  sgl_st_ctx *c4 = sgl_st_open(p4);
  check(c4 != NULL, "open e4m3 shard");
  if (c4) {
    sgl_tensor t = sgl_st_get(c4, "t", 1, (int64_t[]){n4});
    check(t.data != NULL, "e4m3 tensor load");
    if (t.data) {
      for (int i = 0; i < n4; i++) {
        float got = t.data[i];
        if (!(got == e4[i].want)) {
          printf("FAIL: e4m3 %s: got %.9g want %.9g\n", e4[i].name, (double)got, (double)e4[i].want);
          fails++;
        }
      }
    }
    sgl_st_close(c4);
  }

  sgl_st_ctx *c5 = sgl_st_open(p5);
  check(c5 != NULL, "open e5m2 shard");
  if (c5) {
    sgl_tensor t = sgl_st_get(c5, "t", 1, (int64_t[]){n5});
    check(t.data != NULL, "e5m2 tensor load");
    if (t.data) {
      for (int i = 0; i < n5; i++) {
        float got = t.data[i];
        if (!(got == e5[i].want)) {
          printf("FAIL: e5m2 %s: got %.9g want %.9g\n", e5[i].name, (double)got, (double)e5[i].want);
          fails++;
        }
      }
    }
    sgl_st_close(c5);
  }

  /* F16 次正规数：值 = frac * 2^-24。这条曾是真 bug（直接 frac<<13 会变成
   * f32 的次正规，差 2^-125），由 GGUF 的量化块 d 撞上才暴露。*/
  struct {
    unsigned short h;
    float want;
    const char *name;
  } f16[] = {
      {0x0000, 0.0f, "+0"},
      {0x8000, -0.0f, "-0"},
      {0x3C00, 1.0f, "1.0"},
      {0xBC00, -1.0f, "-1.0"},
      {0x0001, 5.9604644775390625e-08f, "min subnormal 2^-24"},
      {0x8001, -5.9604644775390625e-08f, "-min subnormal"},
      {0x8398, -5.4836273193359375e-05f, "subnormal -5.4836e-05 (量化块 d 量级)"},
      {0x03FF, 6.097555160522461e-05f, "max subnormal"},
      {0x0400, 6.103515625e-05f, "min normal 2^-14"},
      {0x7BFF, 65504.0f, "max 65504"},
  };
  int n16 = (int)(sizeof(f16) / sizeof(f16[0]));
  unsigned char b16[64];
  for (int i = 0; i < n16; i++) {
    b16[2 * i] = (unsigned char)(f16[i].h & 0xFF);
    b16[2 * i + 1] = (unsigned char)(f16[i].h >> 8);
  }
  char p16[256];
  snprintf(p16, sizeof(p16), "%s/f16.safetensors", dir);
  write_st(p16, "F16", b16, n16, 2); /* F16：每元素 2 字节 */

  sgl_st_ctx *c16 = sgl_st_open(p16);
  check(c16 != NULL, "open f16 shard");
  if (c16) {
    sgl_tensor t = sgl_st_get(c16, "t", 1, (int64_t[]){n16});
    check(t.data != NULL, "f16 tensor load");
    if (t.data) {
      for (int i = 0; i < n16; i++) {
        float got = t.data[i];
        int same = (got == f16[i].want) ||
                   (f16[i].want == 0.0f && got == 0.0f); /* ±0 视为相等 */
        if (!same) {
          printf("FAIL: f16 %s: got %.9g want %.9g\n", f16[i].name, (double)got,
                 (double)f16[i].want);
          fails++;
        }
      }
    }
    sgl_st_close(c16);
  }

  if (fails) {
    printf("test_fp8: %d failure(s)\n", fails);
    return 1;
  }
  printf("test_fp8: all OK (%d e4m3 + %d e5m2 + %d f16 patterns)\n", n4, n5, n16);
  return 0;
}