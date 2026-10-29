// probe5: 哪种「从裸字节建 uint32 array」的写法在 M1 + MLX 0.29.3 上能work？
// 每个 case 单独跑（崩了不影响别的 case）：
//   ./probe5 0 | 1 | 2 | 3 | 4
#include <mlx/mlx.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mx = mlx::core;

static void* aligned(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, (bytes + 63) & ~(size_t)63) != 0) return nullptr;
  return p;
}

int main(int argc, char** argv) {
  int c = argc > 1 ? atoi(argv[1]) : 0;
  const int N = 151936, KP = 112;
  const size_t n = (size_t)N * KP;   // 17,016,832 个 uint32 = 68 MB
  const size_t bytes = n * 4;

  static std::vector<uint32_t> src;
  src.resize(n);
  for (size_t i = 0; i < n; i++) src[i] = (uint32_t)(i * 2654435761u);
  printf("case %d: n=%zu bytes=%zu\n", c, n, bytes);
  fflush(stdout);

  if (c == 0) { // 小 shape 对照
    std::vector<uint32_t> s(112 * 112, 7u);
    mx::array a(s.data(), mx::Shape{112, 112}, mx::uint32);
    a.eval();
    printf("  ok small uint32, sum=%.0f\n", mx::sum(mx::astype(a, mx::float32)).item<float>());
  } else if (c == 1) { // 大 shape + vector.data()
    mx::array a(src.data(), mx::Shape{N, KP}, mx::uint32);
    a.eval();
    printf("  ok big uint32 from vector\n");
  } else if (c == 2) { // 大 shape + 64B 对齐缓冲
    void* buf = aligned(bytes);
    memcpy(buf, src.data(), bytes);
    mx::array a((const uint32_t*)buf, mx::Shape{N, KP}, mx::uint32);
    a.eval();
    free(buf);
    printf("  ok big uint32 from aligned buf\n");
  } else if (c == 3) { // 大 shape float32（对照：float 路径从没出过问题）
    std::vector<float> f(n);
    for (size_t i = 0; i < n; i++) f[i] = 1.0f;
    mx::array a(f.data(), mx::Shape{N, KP}, mx::float32);
    a.eval();
    printf("  ok big float32, sum=%.0f\n", mx::sum(a).item<float>());
  } else if (c == 4) { // float32 构造后 view 成 uint32（绕过 uint32 的 init 分支）
    std::vector<float> f(n);
    memcpy(f.data(), src.data(), bytes);
    mx::array a(f.data(), mx::Shape{N, KP}, mx::float32);
    a.eval();
    mx::array w = mx::view(a, mx::uint32);
    w.eval();
    printf("  ok view->uint32, size=%zu\n", w.size());
  }
  fflush(stdout);
  return 0;
}