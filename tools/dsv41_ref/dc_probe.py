import sys

import mlx.core as mx
import numpy as np

d = sys.argv[1]
W = mx.load(d + "/model.safetensors")
y2 = np.fromfile("/tmp/y2_dump.bin", dtype=np.float32).reshape(6, 5120)

kg = "blk.1.ffn_gate_shexp.weight"
wg = np.array(mx.dequantize(W[kg], W[kg + ".scales"], W[kg + ".biases"], 64, 4).astype(mx.float32))
gi = np.array(W["blk.1.ffn_gate_inp.weight"].astype(mx.float32))

for nm, w in (("shexp_gate q4", wg), ("gate_inp f16", gi)):
    row_means = w.mean(axis=1)
    print(f"{nm}: global_mean={w.mean():+.5f} row_mean std={row_means.std():.5f} max={np.abs(row_means).max():.5f}")
    # 去掉每行均值后的点积 std
    wc = w - row_means[:, None]
    dots_c = y2 @ wc.T
    print(f"   dots after centering rows: std={dots_c.std():.4g} max={np.abs(dots_c).max():.4g}")
print("y2 token means:", [round(float(y2[t].mean()), 4) for t in range(6)])
yc = y2 - y2.mean(axis=1, keepdims=True)
dots_y = yc @ wg.T
print(f"dots with centered y2: std={dots_y.std():.4g} max={np.abs(dots_y).max():.4g}")
# 双中心
dots_both = yc @ (wg - wg.mean(axis=1, keepdims=True)).T
print(f"dots both centered:    std={dots_both.std():.4g} max={np.abs(dots_both).max():.4g}")
# 组内 bias 的分布：bias 应≈组 min，均值应≈对称中心
s = np.array(W[kg + ".scales"].astype(mx.float32))
b = np.array(W[kg + ".biases"].astype(mx.float32))
print(f"scales: mean={s.mean():.5g} min={s.min():.5g} max={s.max():.5g}")
print(f"biases: mean={b.mean():+.5g} min={b.min():+.5g} max={b.max():+.5g}")
# bias 在每行内的均值（如果 bias 有正的直流，反量化整体抬高）
rb = b.reshape(2304, -1)
print(f"per-row bias mean: mean={rb.mean(axis=1).mean():+.5g} std={rb.mean(axis=1).std():.5g}")
