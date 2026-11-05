import mlx.core as mx
import numpy as np

base = "/Volumes/9100 PRO/models/dsv41/mlx-dense/"
import os, shutil
if os.path.exists("/tmp/dsv41_config.json") and not os.path.exists(base + "config.json"):
    shutil.move("/tmp/dsv41_config.json", base + "config.json")
    print("config.json 就位")

W = mx.load(base + "model.safetensors")
print(f"加载 {len(W)} 张量")
ks = sorted(W)
print("样例:", ks[:3], "...", ks[-3:])
qb = W["blk.0.attn_q_b.weight"]
print(f"q_b: {qb.shape} {qb.dtype}")
o = mx.quantized_matmul(mx.ones((1, 1280), mx.float32), W["blk.0.attn_q_b.weight"],
                        W["blk.0.attn_q_b.weight.scales"], W["blk.0.attn_q_b.weight.biases"],
                        transpose=True, group_size=64, bits=4)
o.eval()
print("qmm row-sum:", float(o.sum()), "max:", float(o.max()), "有限:", bool(mx.isfinite(o).all()))
emb = W["token_embd.weight"]
o2 = mx.quantized_matmul(mx.ones((1, 5120), mx.float32), emb, W["token_embd.weight.scales"],
                         W["token_embd.weight.biases"], transpose=True, group_size=64, bits=4)
print("emb qmm ok, max:", float(mx.abs(o2).max()))
tm = W["engram_token_map"]
print("token_map:", tm.shape, tm.dtype, "min/max:", int(tm.min()), int(tm.max()))
import subprocess
print(subprocess.run(["ls", "-la", base], capture_output=True, text=True).stdout)
