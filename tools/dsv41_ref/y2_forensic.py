import sys

import mlx.core as mx
import numpy as np

d = sys.argv[1]
W = mx.load(d + "/model.safetensors")
y2 = np.fromfile("/tmp/y2_dump.bin", dtype=np.float32).reshape(6, 5120)

kg = "blk.1.ffn_gate_shexp.weight"
wg = np.array(mx.dequantize(W[kg], W[kg + ".scales"], W[kg + ".biases"], 64, 4).astype(mx.float32))
gi = np.array(W["blk.1.ffn_gate_inp.weight"].astype(mx.float32))

print("token norms ||y2_t||:", [round(float(np.linalg.norm(y2[t])), 3) for t in range(6)])
for nm, w in (("shexp_gate", wg), ("gate_inp", gi)):
    dots = y2 @ w.T  # [6, N]
    t, r = np.unravel_index(np.abs(dots).argmax(), dots.shape)
    cosv = dots[t, r] / (np.linalg.norm(y2[t]) * np.linalg.norm(w[r]))
    print(
        f"{nm}: max|dot|={np.abs(dots).max():.4g} @token{t} row{r} cos={cosv:.4f} "
        f"||y2_t||={np.linalg.norm(y2[t]):.3f} ||w_r||={np.linalg.norm(w[r]):.3f} "
        f"row_sigma={dots.std():.4g}"
    )
    # 各 token 的 dot 分布
    for t in range(6):
        print(f"   t{t}: max={np.abs(dots[t]).max():.4g} std={dots[t].std():.4g}")
# y2 各 token 与 shexp 全部行的最大 cos（找结构）
best_cos = 0
for t in range(6):
    c = np.abs(y2[t] @ wg.T) / (np.linalg.norm(y2[t]) * np.linalg.norm(wg, axis=1))
    best_cos = max(best_cos, c.max())
print("best |cos| over all (t,row):", round(float(best_cos), 4))
