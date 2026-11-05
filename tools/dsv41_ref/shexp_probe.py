import mlx.core as mx
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "./models/dsv41/mlx-dense"
W = mx.load(d + "/model.safetensors")
print("tensors:", len(W))
for l in range(4):
    for name in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"):
        k = f"blk.{l}.{name}"
        w = W[k]
        sk = k + ".scales"
        if sk in W:
            s, b = W[sk], W[k + ".biases"]
            dq = mx.dequantize(w, s, b, 64, 4)
            info = f"quant {w.dtype} w={float(mx.max(mx.abs(w))):.4g} s=[{float(mx.min(s)):.4g},{float(mx.max(s)):.4g}]"
        else:
            dq = w.astype(mx.float32)
            info = f"plain {w.dtype}"
        print(
            f"{k:44s} {info:52s} dq_max={float(mx.max(mx.abs(dq))):.6g} "
            f"dq_mean={float(mx.mean(mx.abs(dq))):.6g}"
        )
# 对照：同层 routed gate（mlx-dense 里没有 _exps，用 attn 权重做对照）与 ffn_gate_inp
for l in range(4):
    k = f"blk.{l}.ffn_gate_inp.weight"
    if k in W:
        w = W[k]
        sk = k + ".scales"
        dq = mx.dequantize(w, W[sk], W[k + ".biases"], 64, 4) if sk in W else w.astype(mx.float32)
        print(f"{k:44s} dq_max={float(mx.max(mx.abs(dq))):.6g}")
    k2 = f"blk.{l}.attn_output_b.weight"
    if k2 in W:
        w = W[k2]
        sk = k2 + ".scales"
        dq = mx.dequantize(w, W[sk], W[k2 + ".biases"], 64, 4) if sk in W else w.astype(mx.float32)
        print(f"{k2:44s} dq_max={float(mx.max(mx.abs(dq))):.6g}")
