// tools/mlx_smoke/probe2.cpp — 写 src/mlx_backend.cpp 之前必须钉死的 7 个 API 语义
//
// 每一项都跟主机参考实现对拍，打印 PASS/FAIL；全过打 PROBE2_PASS。
//   1) mx::array(const float*, Shape, Dtype) 是否拷贝并转换 dtype
//   2) mx::fast::rope 的「位置轴」是 S 还是 H（决定 q reshape 成 [B,S,H,D] 还是 [B,H,S,D]），
//      以及 traditional=false 是否等于本仓库 CPU 的 rotate_half 配对 (i, i+D/2)
//   3) rope 的 freqs / scale / 数组 offset 语义（对应 layer_invf / rope_pos_scale / 批量 decode）
//   4) mx::slice_update 的行区间写入语义（KV 池靠它）
//   5) mx::softmax 的轴语义
//   6) mx::matmul 在 4D 上是否按前两维批处理（注意力靠它）
//   7) mx::take 查表（embedding）
#include <mlx/fast.h>
#include <mlx/mlx.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace mx = mlx::core;

static int FAILS = 0;
static void chk(const char *name, bool ok, const char *fmt = "", double v = 0.0) {
  printf("%-54s %s", name, ok ? "PASS" : "FAIL");
  if (fmt[0] && !ok) {
    printf("  ");
    printf(fmt, v);
  }
  printf("\n");
  if (!ok) FAILS++;
}
static bool shape_is(const mx::Shape &s, std::vector<int> want) {
  if (s.size() != want.size()) return false;
  for (size_t i = 0; i < want.size(); i++)
    if (s[i] != want[i]) return false;
  return true;
}

// 本仓库 CPU 的 rope（src/ops.c sgl_rope_inplace / llama.c rope_layer）：
// 配对 (i, i+D/2)，ang = pos * invf[i]，c/s 再乘 attn_factor。
static void host_rope(float *head, int D, double pos, const double *invf, double sc) {
  int half = D / 2;
  for (int i = 0; i < half; i++) {
    double ang = pos * invf[i];
    float c = (float)(std::cos(ang) * sc), s = (float)(std::sin(ang) * sc);
    float a = head[i], b = head[i + half];
    head[i] = a * c - b * s;
    head[i + half] = a * s + b * c;
  }
}

int main() {
  printf("== mlx probe2 ==\n");

  // ---- 1) 主机指针 + dtype 转换 ----
  {
    float h[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    auto a = mx::array(h, mx::Shape{2, 3}, mx::float16);
    a.eval();
    bool dt_ok = (a.dtype() == mx::float16);
    auto b = mx::astype(a, mx::float32);
    b.eval();
    double err = 0;
    for (int i = 0; i < 6; i++) err = std::fmax(err, std::fabs(b.data<float>()[i] - h[i]));
    chk("1) array(float*, Shape, float16) 拷贝+转换", dt_ok && err < 1e-3, "err=%g", err);
    h[0] = 999.f; // 改主机内存不应影响已建的数组（证明是拷贝不是视图）
    auto c = mx::astype(a, mx::float32);
    c.eval();
    chk("1b) 是拷贝而非视图", std::fabs(c.data<float>()[0] - 1.0f) < 1e-6, "got=%g",
        c.data<float>()[0]);
  }

  // ---- 2/3) rope 的轴语义 + traditional + freqs + scale + 数组 offset ----
  {
    const int B = 1, S = 2, H = 3, D = 8, half = D / 2;
    const double base = 10000.0;
    double invf[4];
    for (int i = 0; i < half; i++) invf[i] = std::pow(base, -(double)(2 * i) / D);
    std::vector<float> freqs_h(half);
    for (int i = 0; i < half; i++) freqs_h[i] = (float)invf[i];

    // 实测（第一轮）：MLX 的 rope 位置轴是 -2，不是 -3。用 [B,S,H,D] 时命中的是
    // 「位置沿 H」；所以 q 必须排成 [B,H,S,D]（与 mlx-lm 一致：reshape 后 transpose(0,2,1,3)）。
    std::vector<float> q((size_t)B * H * S * D);
    for (size_t i = 0; i < q.size(); i++) q[i] = (float)((int)((i * 37) % 23) - 11) * 0.05f;

    // 参考 A：位置沿 S（[B,H,S,D] 的 axis -2）
    std::vector<float> refA = q;
    for (int h = 0; h < H; h++)
      for (int s = 0; s < S; s++)
        host_rope(refA.data() + ((size_t)h * S + s) * D, D, (double)s, invf, 1.0);
    // 参考 B：位置沿 H（axis -3）—— 用来证明没搞反
    std::vector<float> refB = q;
    for (int h = 0; h < H; h++)
      for (int s = 0; s < S; s++)
        host_rope(refB.data() + ((size_t)h * S + s) * D, D, (double)h, invf, 1.0);

    auto qa = mx::array(q.data(), mx::Shape{B, H, S, D}, mx::float32);
    auto freqs = mx::array(freqs_h.data(), mx::Shape{(int)half}, mx::float32);
    auto r1 = mx::fast::rope(qa, D, false, (float)base, 1.0f, 0);
    auto r2 = mx::fast::rope(qa, D, false, std::nullopt, 1.0f, 0, freqs);
    r1.eval();
    r2.eval();

    auto cmp = [&](const mx::array &r, const std::vector<float> &ref) {
      double e = 0;
      for (size_t i = 0; i < ref.size(); i++)
        e = std::fmax(e, std::fabs((double)r.data<float>()[i] - ref[i]));
      return e;
    };
    double e1A = cmp(r1, refA), e1B = cmp(r1, refB);
    double e2A = cmp(r2, refA);
    printf("   rope(base)  vs 位置沿S(axis -2)=%.3g  沿H(axis -3)=%.3g\n", e1A, e1B);
    printf("   rope(freqs) vs 位置沿S=%.3g\n", e2A);
    chk("2) rope traditional=false 命中 CPU 的 rotate_half", e1A < 1e-4 || e1B < 1e-4,
        "min err=%g", std::fmin(e1A, e1B));
    chk("2b) 位置轴是 -2（q 要排成 [B,H,S,D]）", e1A < 1e-4, "errS=%g", e1A);
    // 3) freqs 参数【不】等价于本仓库的 layer_invf 表（实测 err≈1.07）。
    // 结论：不用 freqs，改用下面 3e 的手写 rope —— 它还能覆盖 attn_factor /
    // 每层不同 invf / partial rotary，这些 fast::rope 都做不到。仅记录，不计失败。
    printf("3) 传 freqs vs 传 base：err=%.3g → 不等价，本后端不用 freqs（见 3e）\n", e2A);

    // 3e) 手写 rope（mx 原生算子）：角度在主机用 double 算（与 CPU 逐字一致），
    // cos/sin 传上设备；x 排成 [B,H,S,D]，只旋转每头前 rot 维（partial rotary）。
    {
      const int rot = D; // 先测全维；partial 由 rot<D 分支覆盖
      std::vector<float> cs((size_t)S * half), sn((size_t)S * half);
      for (int s = 0; s < S; s++)
        for (int i = 0; i < half; i++) {
          double ang = (double)s * invf[i];
          cs[(size_t)s * half + i] = (float)std::cos(ang);
          sn[(size_t)s * half + i] = (float)std::sin(ang);
        }
      auto C = mx::array(cs.data(), mx::Shape{S, half}, mx::float32);
      auto Sn = mx::array(sn.data(), mx::Shape{S, half}, mx::float32);
      auto x1 = mx::slice(qa, mx::Shape{0, 0, 0, 0}, mx::Shape{B, H, S, half});
      auto x2 = mx::slice(qa, mx::Shape{0, 0, 0, half}, mx::Shape{B, H, S, rot});
      auto o1 = mx::subtract(mx::multiply(x1, C), mx::multiply(x2, Sn));
      auto o2 = mx::add(mx::multiply(x1, Sn), mx::multiply(x2, C));
      auto rm = mx::concatenate(std::vector<mx::array>{o1, o2}, -1);
      rm.eval();
      (void)rot;
      chk("3e) 手写 rope（slice+concat）等价于 CPU 公式", cmp(rm, refA) < 1e-5,
          "err=%g", cmp(rm, refA));
    }

    auto r3 = mx::fast::rope(qa, D, true, (float)base, 1.0f, 0);
    r3.eval();
    chk("3b) traditional=true 结果不同（false 才是我们要的）", cmp(r3, refA) > 1e-3,
        "err=%g", cmp(r3, refA));

    // 数组 offset：批量 decode 时每行用自己的位置
    const int BB = 2;
    std::vector<float> qd((size_t)BB * H * 1 * D);
    for (size_t i = 0; i < qd.size(); i++) qd[i] = (float)((int)((i * 11) % 19) - 9) * 0.07f;
    int pos[BB] = {3, 7};
    std::vector<float> refD = qd;
    for (int b = 0; b < BB; b++)
      for (int h = 0; h < H; h++)
        host_rope(refD.data() + ((size_t)b * H + h) * D, D, (double)pos[b], invf, 1.0);
    auto qda = mx::array(qd.data(), mx::Shape{BB, H, 1, D}, mx::float32);
    auto pa = mx::array(pos, mx::Shape{BB}, mx::int32);
    auto rd = mx::fast::rope(qda, D, false, (float)base, 1.0f, pa);
    rd.eval();
    chk("3c) 数组 offset [B] 对 [B,H,1,D] 逐行生效", cmp(rd, refD) < 1e-4, "err=%g",
        cmp(rd, refD));

    // scale 乘在位置上（对应 rope_pos_scale）
    std::vector<float> refS = q;
    for (int h = 0; h < H; h++)
      for (int s = 0; s < S; s++)
        host_rope(refS.data() + ((size_t)h * S + s) * D, D, (double)s * 0.5, invf, 1.0);
    auto rs = mx::fast::rope(qa, D, false, (float)base, 0.5f, 0);
    rs.eval();
    chk("3d) scale 乘在位置上（= rope_pos_scale）", cmp(rs, refS) < 1e-4, "err=%g",
        cmp(rs, refS));
  }

  // ---- 4) slice_update ----
  {
    const int ctx = 8, KVH = 2, D = 4, kb = 3, n = 2;
    std::vector<float> pool((size_t)ctx * KVH * D, 0.f);
    std::vector<float> upd((size_t)n * KVH * D);
    for (size_t i = 0; i < upd.size(); i++) upd[i] = (float)(i + 1);
    auto P = mx::array(pool.data(), mx::Shape{ctx, KVH, D}, mx::float32);
    auto U = mx::array(upd.data(), mx::Shape{n, KVH, D}, mx::float32);
    auto Q = mx::slice_update(P, U, mx::Shape{kb, 0, 0}, mx::Shape{kb + n, KVH, D});
    Q.eval();
    bool ok = true;
    for (int r = 0; r < ctx && ok; r++)
      for (int c = 0; c < KVH * D && ok; c++) {
        float want = 0.f;
        if (r >= kb && r < kb + n) want = upd[(size_t)(r - kb) * KVH * D + c];
        if (std::fabs(Q.data<float>()[(size_t)r * KVH * D + c] - want) > 1e-6f) ok = false;
      }
    chk("4) slice_update 只改 [kb,kb+n) 行", ok);
    P.eval();
    printf("   （原数组未被就地改=%s → 每次调用产生新数组，KV 要按层拆开）\n",
           std::fabs(P.data<float>()[(size_t)kb * KVH * D]) < 1e-6f ? "是" : "否");
  }

  // ---- 5) softmax 轴 ----
  {
    float h[6] = {1.f, 2.f, 3.f, 1.f, 1.f, 1.f};
    auto a = mx::array(h, mx::Shape{2, 3}, mx::float32);
    auto s = mx::softmax(a, std::vector<int>{-1}, true);
    s.eval();
    double r0 = s.data<float>()[0] + s.data<float>()[1] + s.data<float>()[2];
    double r1 = s.data<float>()[3] + s.data<float>()[4] + s.data<float>()[5];
    chk("5) softmax(a,{-1},true) 按最后一轴归一",
        std::fabs(r0 - 1.0) < 1e-5 && std::fabs(r1 - 1.0) < 1e-5, "row0=%g", r0);
  }

  // ---- 6) 4D matmul 批处理 ----
  {
    std::vector<float> A(1 * 2 * 2 * 3), Bv(1 * 2 * 3 * 2);
    for (size_t i = 0; i < A.size(); i++) A[i] = (float)(i % 5) * 0.1f;
    for (size_t i = 0; i < Bv.size(); i++) Bv[i] = (float)(i % 3) * 0.2f;
    auto a = mx::array(A.data(), mx::Shape{1, 2, 2, 3}, mx::float32);
    auto b = mx::array(Bv.data(), mx::Shape{1, 2, 3, 2}, mx::float32);
    auto c = mx::matmul(a, b);
    c.eval();
    bool shape_ok = shape_is(c.shape(), {1, 2, 2, 2});
    double err = 0;
    for (int bh = 0; bh < 2; bh++)
      for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
          float acc = 0;
          for (int k = 0; k < 3; k++)
            acc += A[((size_t)bh * 2 + i) * 3 + k] * Bv[((size_t)bh * 3 + k) * 2 + j];
          err = std::fmax(err, std::fabs((double)c.data<float>()[((size_t)bh * 2 + i) * 2 + j] - acc));
        }
    chk("6) matmul 4D 按前两维批处理", shape_ok && err < 1e-5, "err=%g", err);
  }

  // ---- 7) take 查表 ----
  {
    float w[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; // [4,3]
    int ids[2] = {3, 1};
    auto W = mx::array(w, mx::Shape{4, 3}, mx::float32);
    auto I = mx::array(ids, mx::Shape{2}, mx::int32);
    auto E = mx::take(W, I, 0);
    E.eval();
    bool ok = shape_is(E.shape(), {2, 3}) && std::fabs(E.data<float>()[0] - 9.f) < 1e-6f &&
              std::fabs(E.data<float>()[3] - 3.f) < 1e-6f;
    chk("7) take(W, int32 indices, axis=0)", ok);
  }

  printf("\n%s (%d 项失败)\n", FAILS == 0 ? "PROBE2_PASS" : "PROBE2_FAIL", FAILS);
  return FAILS == 0 ? 0 : 1;
}
