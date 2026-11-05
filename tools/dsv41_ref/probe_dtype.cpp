/* 探针：f32 × f16 matmul 与 quantized_matmul 的行为 */
#include <mlx/mlx.h>
#include <cstdio>
using namespace mlx::core;

int main() {
    auto pr = load_safetensors(
        "/Volumes/9100 PRO/models/dsv41/mlx-dense/model.safetensors");
    auto& W = pr.first.at("blk.0.attn_kv.weight");
    printf("W is_f32=%d shape=%d,%d\n", (int)(W.dtype() == float32), (int)W.shape(0), (int)W.shape(1));
    auto w32 = astype(W, float32);
    auto m1 = max(abs(reshape(w32, {-1})), -1);
    m1.eval();
    printf("W maxabs=%.6g\n", m1.data<float>()[0]);

    auto x = zeros({2, 5120}, float32) + 0.01f; /* 有限输入 */
    /* 1) f32 × f16 混合 */
    try {
        auto y = matmul(x, transpose(W));
        auto m = max(abs(reshape(astype(y, float32), {-1})), -1);
        m.eval();
        printf("mixed f32xf16: maxabs=%.6g\n", m.data<float>()[0]);
    } catch (std::exception& e) { printf("mixed EXC: %s\n", e.what()); }
    /* 2) f32 × f32 */
    auto y2 = matmul(x, transpose(w32));
    auto m2 = max(abs(reshape(y2, {-1})), -1);
    m2.eval();
    printf("f32xf32: maxabs=%.6g\n", m2.data<float>()[0]);
    /* 3) f32 × f16 但 x 先 astype f16 */
    auto y3 = matmul(astype(x, float16), transpose(W));
    auto m3 = max(abs(reshape(astype(y3, float32), {-1})), -1);
    m3.eval();
    printf("f16xf16: maxabs=%.6g\n", m3.data<float>()[0]);

    /* 4) quantized_matmul f32 x × q4 */
    auto& Wq = pr.first;
    try {
        auto y4 = quantized_matmul(x, Wq.at("blk.0.attn_q_b.weight"), Wq.at("blk.0.attn_q_b.weight.scales"),
                                   Wq.at("blk.0.attn_q_b.weight.biases"), true, 64, 4);
        auto m4 = max(abs(reshape(astype(y4, float32), {-1})), -1);
        m4.eval();
        printf("qmm f32xq4: maxabs=%.6g\n", m4.data<float>()[0]);
    } catch (std::exception& e) { printf("qmm EXC: %s\n", e.what()); }
    /* 5) f16 x × q4（qwen 路径的写法） */
    try {
        auto y5 = quantized_matmul(astype(x, float16), Wq.at("blk.0.attn_q_b.weight"),
                                   Wq.at("blk.0.attn_q_b.weight.scales"), Wq.at("blk.0.attn_q_b.weight.biases"),
                                   true, 64, 4);
        auto m5 = max(abs(reshape(astype(y5, float32), {-1})), -1);
        m5.eval();
        printf("qmm f16xq4: maxabs=%.6g\n", m5.data<float>()[0]);
    } catch (std::exception& e) { printf("qmm16 EXC: %s\n", e.what()); }
    return 0;
}