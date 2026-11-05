import mlx.core as mx
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "./models/dsv41/mlx-dense"
W = mx.load(d + "/model.safetensors")


def stats(k):
    if k + ".scales" in W:
        a = mx.dequantize(W[k], W[k + ".scales"], W[k + ".biases"], 64, 4).astype(mx.float32)
        kind = "q4"
    else:
        a = W[k].astype(mx.float32)
        kind = str(W[k].dtype)
    return kind, float(mx.mean(mx.abs(a))), float(mx.max(mx.abs(a))), float(mx.sqrt(mx.mean(mx.square(a))))


for k in [
    "blk.1.ffn_gate_shexp.weight",
    "blk.1.ffn_up_shexp.weight",
    "blk.1.ffn_down_shexp.weight",
    "blk.1.ffn_gate_inp.weight",
    "blk.1.attn_kv.weight",
    "blk.1.attn_q_a.weight",
    "blk.1.attn_q_b.weight",
    "blk.1.attn_output_b.weight",
    "blk.1.attn_output_a.weight",
    "token_embd.weight",
    "output.weight",
]:
    if k in W:
        kind, meanabs, maxabs, rms = stats(k)
        print(f"{k:40s} {kind:5s} mean|.|={meanabs:.5g} max={maxabs:.5g} rms={rms:.5g}")
    else:
        print(f"{k:40s} MISSING")
# 每行 L2 范数对比（同一输入下输出幅值 ∝ 行 L2）
import numpy as np

for k in ["blk.1.ffn_gate_shexp.weight", "blk.1.ffn_gate_inp.weight"]:
    if k + ".scales" in W:
        a = np.array(mx.dequantize(W[k], W[k + ".scales"], W[k + ".biases"], 64, 4).astype(mx.float32))
    else:
        a = np.array(W[k].astype(mx.float32))
    r2 = np.sqrt((a.astype(np.float64) ** 2).sum(axis=1))
    print(f"{k}: rowL2 mean={r2.mean():.4g} max={r2.max():.4g}")
