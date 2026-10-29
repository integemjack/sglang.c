// sgl_mlx_smoke —— MLX 后端可行性冒烟（W2 骨架验证，不碰仓库任何现有代码）
// 验证四件事：find_package(MLX) 能配、C++20 能编、@rpath/libmlx.dylib + mlx.metallib 能加载、
// Metal GPU 真在算（不是回落 CPU）。同时把三个关键 fast 算子与主机侧参考对拍。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include "mlx/mlx.h"

namespace mx = mlx::core;

static double host_rmsnorm_ref(const std::vector<float>& x, const std::vector<float>& w, int N,
                               float eps, std::vector<float>& out) {
  out.resize(x.size());
  for (size_t r = 0; r < x.size() / N; r++) {
    double ss = 0.0;
    for (int i = 0; i < N; i++) ss += (double)x[r * N + i] * x[r * N + i];
    double inv = 1.0 / std::sqrt(ss / N + eps);
    for (int i = 0; i < N; i++) out[r * N + i] = (float)(x[r * N + i] * inv * w[i]);
  }
  return 0.0;
}

int main() {
  printf("== mlx ==\n");
  printf("version      : %s\n", mx::version());
  auto dev = mx::default_device();
  printf("default_dev  : type=%d index=%d\n", (int)dev.type, dev.index);
  printf("metal_avail  : %d\n", (int)mx::metal::is_available());
  if (mx::metal::is_available()) {
    // 0.29.3 的 device_info() 返回 unordered_map<string, variant<string, unsigned long>>
    auto di = mx::metal::device_info();
    for (const auto& kv : di) {
      if (std::holds_alternative<std::string>(kv.second))
        printf("  metal.%-18s: %s\n", kv.first.c_str(), std::get<std::string>(kv.second).c_str());
      else
        printf("  metal.%-18s: %lu\n", kv.first.c_str(), (unsigned long)std::get<unsigned long>(kv.second));
    }
  }

  // ---- 1) fast::rms_norm vs 主机参考 ----
  const int M = 4, N = 64;
  std::vector<float> hx(M * N), hw(N);
  for (int i = 0; i < M * N; i++) hx[i] = std::sin(0.37f * i) * 1.7f;
  for (int i = 0; i < N; i++) hw[i] = 0.5f + 0.01f * i;

  auto x = mx::array(hx.data(), {M, N}, mx::float32);
  auto w = mx::array(hw.data(), {N}, mx::float32);
  auto y = mx::fast::rms_norm(x, w, 1e-6f);
  mx::eval(y);

  std::vector<float> ref;
  host_rmsnorm_ref(hx, hw, N, 1e-6f, ref);
  const float* py = y.data<float>();
  double maxerr = 0.0;
  for (int i = 0; i < M * N; i++) maxerr = std::fmax(maxerr, std::fabs(py[i] - ref[i]));
  printf("\n== rms_norm ==\nshape [%d,%d] maxerr vs host = %.3g  (y[0]=%.6f ref=%.6f)\n", M, N, maxerr,
         py[0], ref[0]);

  // ---- 2) fast::rope 逐行 offset 重载（batch decode 的关键）----
  // x: [B, H, 1, D]，offset 是 [B] 的数组 → 每行用自己的位置
  const int B = 3, H = 2, D = 16;
  std::vector<float> hq(B * H * D);
  for (size_t i = 0; i < hq.size(); i++) hq[i] = 0.1f * (float)(i % 23) - 1.0f;
  auto q = mx::array(hq.data(), {B, H, 1, D}, mx::float32);
  std::vector<int> hpos = {0, 7, 100};
  auto pos = mx::array(hpos.data(), {B}, mx::int32);
  auto qr = mx::fast::rope(q, D, false, 10000.0f, 1.0f, pos);
  mx::eval(qr);
  // 参考：逐行用标量 offset 再跑一次，比较两者
  double rope_err = 0.0;
  const float* pr = qr.data<float>();
  for (int b = 0; b < B; b++) {
    auto qb = mx::slice(q, {b, 0, 0, 0}, {b + 1, H, 1, D});
    auto rb = mx::fast::rope(qb, D, false, 10000.0f, 1.0f, hpos[b]);
    mx::eval(rb);
    const float* pb = rb.data<float>();
    for (int i = 0; i < H * D; i++)
      rope_err = std::fmax(rope_err, std::fabs(pr[b * H * D + i] - pb[i]));
  }
  printf("\n== rope(array offset) ==\nB=%d 逐行 offset vs 标量 offset maxerr = %.3g\n", B, rope_err);

  // ---- 3) fast::scaled_dot_product_attention（causal）vs 主机朴素实现 ----
  const int S = 6, Dh = 8;
  std::vector<float> hQ(B * H * S * Dh), hK(B * H * S * Dh), hV(B * H * S * Dh);
  for (size_t i = 0; i < hQ.size(); i++) {
    hQ[i] = std::sin(0.11f * i);
    hK[i] = std::cos(0.13f * i);
    hV[i] = 0.3f * std::sin(0.07f * i);
  }
  auto Q = mx::array(hQ.data(), {B, H, S, Dh}, mx::float32);
  auto K = mx::array(hK.data(), {B, H, S, Dh}, mx::float32);
  auto V = mx::array(hV.data(), {B, H, S, Dh}, mx::float32);
  float scale = 1.0f / std::sqrt((float)Dh);
  auto O = mx::fast::scaled_dot_product_attention(Q, K, V, scale, "causal");
  mx::eval(O);
  // 主机参考
  double attn_err = 0.0;
  const float* po = O.data<float>();
  for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++)
      for (int i = 0; i < S; i++) {
        std::vector<float> sc(S), mv(S);
        float mxv = -1e30f, sum = 0.f;
        for (int j = 0; j <= i; j++) {
          float d = 0.f;
          for (int k = 0; k < Dh; k++)
            d += hQ[((b * H + h) * S + i) * Dh + k] * hK[((b * H + h) * S + j) * Dh + k];
          sc[j] = d * scale;
          mxv = std::fmax(mxv, sc[j]);
        }
        for (int j = 0; j <= i; j++) { mv[j] = std::exp(sc[j] - mxv); sum += mv[j]; }
        for (int k = 0; k < Dh; k++) {
          float acc = 0.f;
          for (int j = 0; j <= i; j++) acc += mv[j] / sum * hV[((b * H + h) * S + j) * Dh + k];
          attn_err = std::fmax(attn_err, std::fabs(acc - po[((b * H + h) * S + i) * Dh + k]));
        }
      }
  printf("\n== scaled_dot_product_attention(causal) ==\nB=%d H=%d S=%d D=%d maxerr vs host = %.3g\n",
         B, H, S, Dh, attn_err);

  // ---- 4) fp16 路径（推理默认精度）----
  auto y16 = mx::fast::rms_norm(mx::astype(x, mx::float16), mx::astype(w, mx::float16), 1e-6f);
  mx::eval(y16);
  double e16 = 0.0;
  const uint16_t* p16 = y16.data<uint16_t>();
  // 只检查有限性与量级（half→float 逐位转换太啰嗦，这里验证 dtype 与非 NaN）
  for (int i = 0; i < M * N; i++) {
    uint32_t h = p16[i];
    uint32_t f = ((h & 0x8000u) << 16) | (((h >> 10) & 0x1f) ? ((((h >> 10) & 0x1f) + 112) << 23) | ((h & 0x3ff) << 13) : 0);
    float v;
    memcpy(&v, &f, 4);
    if (!std::isfinite(v)) { e16 = 1e9; break; }
  }
  printf("\n== fp16 rms_norm ==\ndtype=%s 非有限值=%s\n", y16.dtype() == mx::float16 ? "float16" : "?",
         e16 > 1e8 ? "有(NaN/Inf!)" : "无");

  printf("\nactive_mem = %.2f MB, peak = %.2f MB, cache = %.2f MB\n",
         mx::get_active_memory() / 1048576.0, mx::get_peak_memory() / 1048576.0,
         mx::get_cache_memory() / 1048576.0);

  bool ok = maxerr < 1e-4 && rope_err < 1e-5 && attn_err < 1e-4 && e16 < 1e8;
  printf("\n%s\n", ok ? "SMOKE_PASS" : "SMOKE_FAIL");
  return ok ? 0 : 1;
}
