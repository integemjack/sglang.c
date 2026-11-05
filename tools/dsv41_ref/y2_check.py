import sys

import mlx.core as mx
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "./models/dsv41/mlx-dense"
W = mx.load(d + "/model.safetensors")
y2 = mx.array(np.fromfile("/tmp/y2_dump.bin", dtype=np.float32)).reshape(6, 5120)
lg_ref = np.fromfile("/tmp/lg_dump.bin", dtype=np.float32)
print("y2: max|.|", float(mx.max(mx.abs(y2))), " rms", float(mx.sqrt(mx.mean(y2 * y2))))

gi = W["blk.1.ffn_gate_inp.weight"].astype(mx.float32)
lg = y2 @ gi.T
print(f"lg  py={float(mx.max(mx.abs(lg))):.6g}  cpp={lg_ref.max():.6g}  match={np.allclose(np.array(lg).ravel(), lg_ref, atol=1e-3)}")

kg = "blk.1.ffn_gate_shexp.weight"
wg = mx.dequantize(W[kg], W[kg + ".scales"], W[kg + ".biases"], 64, 4).astype(mx.float32)
sh_g = y2 @ wg.T
print(f"shg py={float(mx.max(mx.abs(sh_g))):.6g}  (cpp fast/slow = 28.18)")

ku = "blk.1.ffn_up_shexp.weight"
wu = mx.dequantize(W[ku], W[ku + ".scales"], W[ku + ".biases"], 64, 4).astype(mx.float32)
sh_u = y2 @ wu.T
kd = "blk.1.ffn_down_shexp.weight"
wd = mx.dequantize(W[kd], W[kd + ".scales"], W[kd + ".biases"], 64, 4).astype(mx.float32)
act = (mx.minimum(sh_g, 10) * mx.sigmoid(mx.minimum(sh_g, 10))) * mx.clip(sh_u, -10, 10)
sh = act @ wd.T
print(f"sh_u={float(mx.max(mx.abs(sh_u))):.6g}  sh(py end-to-end)={float(mx.max(mx.abs(sh))):.6g}")
# 对照 llama.cpp 语义：silu 后 clamp
act2 = mx.clip(mx.sigmoid(sh_g) * sh_g, -float("inf"), 10) * mx.clip(sh_u, -10, 10)
sh2 = act2 @ wd.T
print(f"sh(llama.cpp clamp语义)={float(mx.max(mx.abs(sh2))):.6g}")
