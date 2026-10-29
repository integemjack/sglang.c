// probe3: rope 输出到底是什么（probe2 里读到 1e18 量级，先诊断 dtype/shape/前几个值）
#include <mlx/fast.h>
#include <mlx/mlx.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace mx = mlx::core;

static void dump(const char *tag, const mx::array &a) {
  printf("%-14s dtype=%s ndim=%d shape=[", tag, a.dtype() == mx::float32 ? "f32"
                                       : a.dtype() == mx::float16 ? "f16"
                                       : a.dtype() == mx::bfloat16 ? "bf16" : "?",
         (int)a.ndim());
  for (size_t i = 0; i < a.shape().size(); i++) printf("%d%s", a.shape()[i], i + 1 < a.shape().size() ? "," : "");
  printf("] size=%d  head:", (int)a.size());
  if (a.dtype() == mx::float32) {
    for (int i = 0; i < 8 && i < (int)a.size(); i++) printf(" %.5f", a.data<float>()[i]);
  } else {
    auto f = mx::astype(a, mx::float32);
    f.eval();
    for (int i = 0; i < 8 && i < (int)a.size(); i++) printf(" %.5f", f.data<float>()[i]);
  }
  printf("\n");
}

int main() {
  const int B = 1, S = 2, H = 3, D = 8;
  std::vector<float> q((size_t)B * S * H * D);
  for (size_t i = 0; i < q.size(); i++) q[i] = (float)(((i * 37) % 23) - 11) * 0.05f;

  auto qa = mx::array(q.data(), mx::Shape{B, S, H, D}, mx::float32);
  qa.eval();
  dump("input", qa);
  printf("host q head:");
  for (int i = 0; i < 8; i++) printf(" %.5f", q[i]);
  printf("\n");

  auto r1 = mx::fast::rope(qa, D, false, 10000.0f, 1.0f, 0);
  r1.eval();
  dump("rope(int off)", r1);

  auto r2 = mx::fast::rope(qa, D, false, 10000.0f, 1.0f, mx::array(0));
  r2.eval();
  dump("rope(arr off)", r2);

  // 位置 0 应当是恒等（cos=1,sin=0）→ 用 offset=0 且只看 S=0 那一层
  // 再试 3D 输入 [S,H,D]
  auto q3 = mx::reshape(qa, mx::Shape{S, H, D});
  auto r3 = mx::fast::rope(q3, D, false, 10000.0f, 1.0f, 0);
  r3.eval();
  dump("rope 3D", r3);

  // traditional
  auto r4 = mx::fast::rope(qa, D, true, 10000.0f, 1.0f, 0);
  r4.eval();
  dump("rope trad", r4);

  // 只用 2 个头、位置 0：期望 == 输入
  printf("\n期望：offset=0 时 S=0 的那 H*D 个值应等于输入前 %d 个\n", H * D);
  return 0;
}
